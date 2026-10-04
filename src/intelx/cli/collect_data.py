#!/usr/bin/env python3
import os
import sys
import csv
import json
import multiprocessing as mp
from math import pi
from time import time
import logging
from datetime import datetime

from intelx.data.ring_collector import initialize_global_workers, shutdown_global_workers, run_a_single_show
# Import functions from data_driver3
from intelx.data.driver import sample_all_show_spawn_points, visualize_spawn_points

from intelx.paths import REPO_ROOT, map_path, dataset_path

# Constants
SEASONS_PER_SHOW = 10
GPUS_PER_SHOW = 5
MAX_CONCURRENT_DRIVERS = 18
SHOWS_PER_MAP_CONCURRENT = 8

# Directories
MAP_INPUT_DIR = map_path("train_standardized")
DATA_OUTPUT_DIR = dataset_path("coverage_data_main_3")


def setup_logging(log_file):
    """Setup logging for a worker process"""
    logger = logging.getLogger(f"worker_{os.getpid()}")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_spawn_points(map_name, spawn_points, base_out):
    """
    Save spawn points with show_id mapping to JSON file for resume capability.
    Format: {show_id: [x, y], show_id: [x, y], ...}
    """
    spawn_file = os.path.join(base_out, "spawn_points.json")

    # Create mapping of show_id to spawn point
    spawn_mapping = {
        show_id: [float(x), float(y)]
        for show_id, (x, y) in enumerate(spawn_points)
    }

    # Add metadata
    spawn_data = {
        "map_name": map_name,
        "num_shows": len(spawn_points),
        "timestamp": datetime.now().isoformat(),
        "spawn_points": spawn_mapping
    }

    with open(spawn_file, 'w') as f:
        json.dump(spawn_data, f, indent=2)

    print(f"  💾 {map_name}: Saved spawn points to {spawn_file}")
    return spawn_file


def load_spawn_points(spawn_file):
    """
    Load spawn points from JSON file.
    Returns: dict with spawn_points mapping {show_id: [x, y]}
    """
    if not os.path.exists(spawn_file):
        return None

    with open(spawn_file, 'r') as f:
        spawn_data = json.load(f)

    # Convert string keys back to integers
    spawn_mapping = {
        int(show_id): tuple(coords)
        for show_id, coords in spawn_data["spawn_points"].items()
    }

    return spawn_mapping


def get_completed_shows(base_out, num_shows):
    """
    Check which shows have been completed by looking for HDF5 files.
    Returns: set of completed show IDs
    """
    completed = set()

    for show_id in range(num_shows):
        h5_file = os.path.join(base_out, f"show_data_{show_id:03d}.h5")
        if os.path.exists(h5_file):
            # Verify file is not corrupted
            try:
                import h5py
                with h5py.File(h5_file, 'r') as f:
                    # Basic validation
                    if 'origin_images' in f and 'target_heatmaps' in f:
                        completed.add(show_id)
            except:
                # Corrupted file, will be reprocessed
                pass

    return completed


def persistent_show_worker(worker_id, show_queue, result_queue, config, spawn_mapping, log_file):
    """
    Persistent show worker with comprehensive logging.
    """


    # Setup logging
    logger = setup_logging(log_file)
    logger.info(f"Persistent Worker {worker_id} starting (PID: {os.getpid()})")

    shows_completed = 0

    try:
        # Get first spawn point for GPU worker initialization
        first_spawn = list(spawn_mapping.values())[0] if spawn_mapping else (100.0, 100.0)

        logger.info(f"Initializing {GPUS_PER_SHOW} GPU workers...")
        print(f"    🚀 Persistent Worker {worker_id}: Initializing {GPUS_PER_SHOW} GPU workers...")

        initialize_global_workers(config, first_spawn, num_workers=GPUS_PER_SHOW)

        logger.info(f"GPU workers initialized successfully")
        print(f"    ✅ Persistent Worker {worker_id}: GPU workers ready")

        # Process shows from queue
        while True:
            try:
                show_task = show_queue.get(timeout=5)

                if show_task is None:  # Poison pill
                    logger.info(f"Received shutdown signal")
                    break

                show_id = show_task
                spawn_point = spawn_mapping[show_id]

                logger.info(f"Starting show {show_id} with spawn point {spawn_point}")
                print(f"    📍 Persistent Worker {worker_id}: Starting show {show_id}")

                show_start = time()

                # Run the show
                run_a_single_show(config, show_id, spawn_point, skip_worker_init=True)

                show_time = time() - show_start
                shows_completed += 1

                logger.info(f"Show {show_id} completed in {show_time:.1f}s")
                print(
                    f"    ✅ Persistent Worker {worker_id}: Show {show_id} completed in {show_time:.1f}s ({shows_completed} total)")

                # Report completion back to main process
                result_queue.put(('completed', show_id))

            except Exception as e:
                if "Empty" not in str(e):
                    logger.error(f"Error processing show: {e}", exc_info=True)
                    print(f"    ❌ Persistent Worker {worker_id}: Error: {e}")

    except Exception as e:
        logger.error(f"Fatal error in worker: {e}", exc_info=True)
        print(f"    ❌ Persistent Worker {worker_id}: Setup error: {e}")

    finally:
        logger.info(f"Shutting down after {shows_completed} shows")
        print(f"    🛑 Persistent Worker {worker_id}: Shutting down GPU workers ({shows_completed} shows completed)")
        try:
            shutdown_global_workers()
        except:
            pass


def map_worker(map_name, num_shows, config_template):
    """
    Worker for one map with resume capability and comprehensive logging.
    """
    # Setup logging
    log_dir = os.path.join(DATA_OUTPUT_DIR, "_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{map_name}_map.log")
    logger = setup_logging(log_file)

    logger.info(f"Map worker starting for {map_name} with {num_shows} shows")

    try:
        config = config_template.copy()
        config['MAP_PGM'] = os.path.join(MAP_INPUT_DIR, f"{map_name}.pgm")
        config['SEASONS_PER_SHOW'] = SEASONS_PER_SHOW
        config['NUM_WORKERS'] = GPUS_PER_SHOW

        base_out = os.path.join(DATA_OUTPUT_DIR, map_name)
        os.makedirs(base_out, exist_ok=True)
        config['BASE_OUTPUT_DIR'] = base_out

        spawn_file = os.path.join(base_out, "spawn_points.json")

        # Check for existing spawn points (resume capability)
        spawn_mapping = load_spawn_points(spawn_file)

        if spawn_mapping is not None:
            logger.info(f"Found existing spawn points file, loading {len(spawn_mapping)} points")
            print(f"  🔄 {map_name}: Resuming - loaded {len(spawn_mapping)} spawn points")

            # Verify spawn point count matches
            if len(spawn_mapping) != num_shows:
                logger.warning(f"Spawn point count mismatch: {len(spawn_mapping)} vs {num_shows}")
                print(f"  ⚠️  {map_name}: Spawn point count mismatch, resampling...")
                spawn_mapping = None

        # Sample new spawn points if needed
        if spawn_mapping is None:
            logger.info(f"Sampling {num_shows} new spawn points")
            spawn_points = sample_all_show_spawn_points(config, num_shows)

            # Create mapping
            spawn_mapping = {show_id: spawn_points[show_id] for show_id in range(len(spawn_points))}

            # Save for resume capability
            save_spawn_points(map_name, spawn_points, base_out)

            # Visualize
            viz_path = os.path.join(base_out, "spawn_points.png")
            visualize_spawn_points(config, spawn_points, filename=viz_path)
            logger.info(f"Spawn points sampled and saved")

        print(f"  📍 {map_name}: Using {len(spawn_mapping)} spawn points")

        # Check which shows are already completed
        completed_shows = get_completed_shows(base_out, num_shows)

        if completed_shows:
            logger.info(f"Found {len(completed_shows)} completed shows: {sorted(completed_shows)}")
            print(f"  ♻️  {map_name}: Found {len(completed_shows)} completed shows, skipping them")

        # Get remaining shows to process
        remaining_shows = [show_id for show_id in range(num_shows) if show_id not in completed_shows]

        if not remaining_shows:
            logger.info(f"All shows already completed")
            print(f"  ✅ {map_name}: All {num_shows} shows already completed, skipping")
            return

        logger.info(f"Processing {len(remaining_shows)} remaining shows: {remaining_shows}")
        print(f"  🚀 {map_name}: Processing {len(remaining_shows)}/{num_shows} remaining shows...")
        print(f"  🔧 {map_name}: Starting {SHOWS_PER_MAP_CONCURRENT} persistent show workers...")

        # Create queues
        show_queue = mp.Queue()
        result_queue = mp.Queue()

        # Add remaining shows to queue
        for show_id in remaining_shows:
            show_queue.put(show_id)

        # Add poison pills
        for _ in range(SHOWS_PER_MAP_CONCURRENT):
            show_queue.put(None)

        # Start persistent show workers
        map_start = time()
        persistent_workers = []

        for worker_id in range(SHOWS_PER_MAP_CONCURRENT):
            worker_log = os.path.join(log_dir, f"{map_name}_worker_{worker_id}.log")
            p = mp.Process(
                target=persistent_show_worker,
                args=(worker_id, show_queue, result_queue, config, spawn_mapping, worker_log)
            )
            p.start()
            persistent_workers.append((p, worker_id))
            logger.info(f"Started persistent worker {worker_id} (PID: {p.pid})")

        # Monitor progress
        shows_done = 0
        while shows_done < len(remaining_shows):
            try:
                status, show_id = result_queue.get(timeout=30)
                if status == 'completed':
                    shows_done += 1
                    logger.info(f"Progress: {shows_done}/{len(remaining_shows)} shows completed")
                    print(f"  📊 {map_name}: Progress {shows_done}/{len(remaining_shows)}")
            except:
                # Timeout, check if workers are still alive
                alive = sum(1 for p, _ in persistent_workers if p.is_alive())
                logger.warning(f"Progress check timeout, {alive} workers still alive")

        # Wait for all persistent workers to complete
        for p, worker_id in persistent_workers:
            p.join(timeout=10)
            if p.exitcode != 0:
                logger.warning(f"Persistent Worker {worker_id} exited with code {p.exitcode}")
                print(f"    ⚠️  Persistent Worker {worker_id} exited with code {p.exitcode}")

        map_time = time() - map_start

        logger.info(f"Map completed: {num_shows} shows in {map_time:.1f}s")
        print(f"✅ {map_name}: Completed {num_shows} shows in {map_time:.1f}s ({SEASONS_PER_SHOW} seasons each)")

    except Exception as e:
        logger.error(f"Map worker failed: {e}", exc_info=True)
        print(f"❌ {map_name}: Failed with error: {e}")


def main():
    print("🚀 Starting automated data collection with PERSISTENT WORKERS and RESUME capability...")

    # Load map list: the shows per map are Table 1 of the paper, allocated in
    # proportion to the complexity score kappa of Eq. (1).
    csv_path = os.path.join(REPO_ROOT, "data", "train_metadata.csv")
    maps = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            maps.append((row['alias'], int(row['num_shows'])))

    print(f"📋 Found {len(maps)} maps to process")
    print(f"⚙️  Max concurrent map drivers: {MAX_CONCURRENT_DRIVERS}")
    print(f"🎯 Persistent show workers per map: {SHOWS_PER_MAP_CONCURRENT}")
    print(f"💻 GPU workers per show: {GPUS_PER_SHOW}")
    print(f"📁 Logs directory: {os.path.join(DATA_OUTPUT_DIR, '_logs')}")
    print(f"💾 Spawn points saved for resume capability")

    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    # Config template
    RESOLUTION = 0.05
    WIN_M = 20
    LASER_SCAN_R = 10.0

    config_template = {
        'RESOLUTION': RESOLUTION,
        'ROBOT_R': 0.35,
        'FOV': 220 * pi / 180,
        'N_BEAMS': 662,
        'MAX_R': int(LASER_SCAN_R / RESOLUTION),
        'WIN_M': WIN_M,
        'ANG_VEL': pi,
        'DT': 0.1,
        'SPAWN_CLEARANCE': 0.5,
        'SHOW_HEAD': True,
        'VISUALIZE_FOV': False,
        'MAX_GOAL_DIST_M': 15.0,
        'RESET_AFTER_EPISODE': True,
        'INNER_RADIUS_R_M': 2.5,
        'OUTER_RADIUS_D_M': 10.0,
        'VISUALIZE_SEASON_GOALS': False,
        'VISUALIZE_EPISODE_GOAL': False,
        'STEP_SIZE_M': 0.1,
        'GOAL_DENSITY_PER_METER': 10,
    }
    config_template['RAD_CELLS'] = int(config_template['ROBOT_R'] / config_template['RESOLUTION'])
    config_template['WIN_PX'] = int(config_template['WIN_M'] / config_template['RESOLUTION'])

    # Process maps in batches
    total_maps = len(maps)
    completed = 0

    for i in range(0, total_maps, MAX_CONCURRENT_DRIVERS):
        batch = maps[i:i + MAX_CONCURRENT_DRIVERS]
        batch_size = len(batch)

        print(f"\n📦 Processing map batch {i // MAX_CONCURRENT_DRIVERS + 1}: {batch_size} maps")
        for map_name, num_shows in batch:
            print(f"   • {map_name}: {num_shows} shows")

        processes = []
        for map_name, num_shows in batch:
            p = mp.Process(target=map_worker, args=(map_name, num_shows, config_template))
            processes.append((p, map_name))

        batch_start = time()
        for p, _ in processes:
            p.start()

        for p, map_name in processes:
            p.join()
            if p.exitcode != 0:
                print(f"⚠️  Process for {map_name} exited with code {p.exitcode}")

        batch_time = time() - batch_start
        completed += batch_size
        print(f"🏁 Map batch completed in {batch_time:.1f}s. Progress: {completed}/{total_maps}")

    print(f"\n🎉 All {total_maps} maps completed!")
    print(f"📁 Data saved to: {DATA_OUTPUT_DIR}")
    print(f"📋 Logs saved to: {os.path.join(DATA_OUTPUT_DIR, '_logs')}")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()
