#!/usr/bin/env python3
import multiprocessing as mp
import numpy as np
import cv2
import os
from math import pi
import time

# Import the necessary functions and classes from your other files
from intelx.sim.simulator import ExplorationSimulator, find_spawn, load_pgm_map, seal_by_adjacent
from intelx.data.ring_collector import run_a_single_show

from intelx.paths import map_path, dataset_path


def sample_all_show_spawn_points(config, num_points):
    """Uniformly samples all spawn points needed for all shows."""
    print(f"Sampling {num_points} unique show start points...")
    occ = load_pgm_map(config['MAP_PGM'])
    occ = seal_by_adjacent(occ)

    spawn_points = []
    for _ in range(num_points * 10):
        if len(spawn_points) >= num_points: break
        try:
            point = find_spawn(occ, config['RESOLUTION'], config['SPAWN_CLEARANCE'] - config['ROBOT_R'] + 0.1)
            if point not in spawn_points:
                spawn_points.append(point)
        except RuntimeError:
            continue

    if len(spawn_points) < num_points:
        raise RuntimeError(f"Could only find {len(spawn_points)}/{num_points} valid spawn points.")
    print(f"Successfully sampled {len(spawn_points)} start points.")
    return spawn_points


def visualize_spawn_points(config, points, filename="show_spawn_points.png"):
    """
    Saves a clean image of the map with spawn points marked as hollow circles.
    Uses the original PGM for proper background.
    """
    print(f"Visualizing sampled spawn points and saving to {filename}...")

    # Load original PGM map
    occ = load_pgm_map(config['MAP_PGM'])
    H, W = occ.shape

    # FIXED: Ensure proper data type for OpenCV color conversion
    # Convert to unsigned 8-bit if needed
    if occ.dtype != np.uint8:
        # Normalize to 0-255 range and convert to uint8
        occ = ((occ - occ.min()) / (occ.max() - occ.min()) * 255).astype(np.uint8)

    # Convert grayscale PGM to RGB for colored overlays
    img = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

    # Robot radius in pixels (0.35m / 0.05m/pixel = 7 pixels)
    robot_radius_px = int(config.get('ROBOT_R', 0.35) / config.get('RESOLUTION', 0.05))

    # Draw spawn points
    for i, (x, y) in enumerate(points):
        # Filled red dot at center (2-pixel radius for visibility)
        cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)  # Red center dot

        # Hollow circle showing robot radius (7 pixels for 0.35m robot)
        cv2.circle(img, (int(x), int(y)), robot_radius_px, (0, 0, 255), 1)  # Red hollow circle

    # Save as PNG
    cv2.imwrite(filename, img)
    print(f"  Saved {len(points)} spawn points to {filename}")


def worker(args):
    """Unpacks arguments and calls the main simulation logic for a single show."""
    show_id, config, spawn_point, hdf5_path = args
    try:
        run_a_single_show(config, show_id, spawn_point)
    except Exception as e:
        process_name = mp.current_process().name
        print(f"!!!!!! {process_name} (Show {show_id}) encountered a fatal error: {e} !!!!!!")


if __name__ == '__main__':
    # --- Main Configuration ---
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Can only be set once
    NUM_SHOWS = 5
    SEASONS_PER_SHOW = 5
    NUM_PROCESSES = 1

    RESOLUTION = 0.05
    WIN_M = 20  # 20
    LASER_SCAN_R = 10.0

    # --- NEW: Specify the main output directory ---
    BASE_OUTPUT_DIR = dataset_path("dense_data/small_house_aws")

    config = {
        'MAP_PGM': map_path('train_standardized/small_house_aws.pgm'),
        'RESOLUTION': RESOLUTION, 'ROBOT_R': 0.35, 'FOV': 220 * pi / 180, 'N_BEAMS': 662,
        'MAX_R': int(LASER_SCAN_R / RESOLUTION), 'WIN_M': WIN_M,
        'ANG_VEL': pi, 'DT': 0.1,
        'SPAWN_CLEARANCE': 0.5, 'SHOW_HEAD': True, 'VISUALIZE_FOV': False,
        'MAX_GOAL_DIST_M': 15.0, 'RESET_AFTER_EPISODE': True,
        'SEASONS_PER_SHOW': SEASONS_PER_SHOW,
        'INNER_RADIUS_R_M': 2.5,
        'OUTER_RADIUS_D_M': 10.0,
        'VISUALIZE_SEASON_GOALS': False,
        'VISUALIZE_EPISODE_GOAL': False,
        'STEP_SIZE_M': 0.1,
        'GOAL_DENSITY_PER_METER': 10,

        # --- NEW: Add the base directory to the config dictionary ---
        'BASE_OUTPUT_DIR': BASE_OUTPUT_DIR,
        'NUM_WORKERS': 5,
    }
    config['RAD_CELLS'] = int(config['ROBOT_R'] / config['RESOLUTION'])
    config['WIN_PX'] = int(config['WIN_M'] / config['RESOLUTION'])

    # 1. Sample all spawn points
    all_show_spawn_points = sample_all_show_spawn_points(config, NUM_SHOWS)

    # Create the base output directory if it doesn't exist
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # --- UPDATED: Save the visualization inside the new directory ---
    viz_filename = os.path.join(BASE_OUTPUT_DIR, "all_show_spawn_points.png")
    visualize_spawn_points(config, all_show_spawn_points, filename=viz_filename)

    # 2. Prepare job arguments for the multiprocessing pool
    job_args = []
    for show_id, spawn_point in enumerate(all_show_spawn_points):
        # Create a unique HDF5 path for each worker
        hdf5_path = os.path.join(BASE_OUTPUT_DIR, f"show_data_{show_id:02d}.h5")
        job_args.append((show_id, config, spawn_point, hdf5_path))

    # 3. Use a multiprocessing Pool to run the shows in parallel
    print(f"\nDistributing {len(job_args)} shows across {NUM_PROCESSES} processes...")
    start_time = time.time()

    # with mp.Pool(processes=NUM_PROCESSES) as pool:
    #     pool.map(worker, job_args)
    for i in range(len(job_args)):
        worker(job_args[i])

    # worker(job_args)
    duration = time.time() - start_time
    print(f"\nAll shows have completed data collection. Total time: {duration:.2f} seconds.")