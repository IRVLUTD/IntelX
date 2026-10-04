#!/usr/bin/env python3
import numpy as np
import random
from math import pi
import time
import multiprocessing as mp
import os
import cv2
import h5py
from tqdm import tqdm
import atexit
import platform
import hashlib

# Import the simulator class and utilities from the other file
from intelx.sim.simulator import ExplorationSimulator, sample_goal, bresenham

# Global persistent worker infrastructure
GLOBAL_WORKER_POOL = None
GLOBAL_TASK_QUEUE = None
GLOBAL_RESULT_QUEUE = None


def initialize_global_workers(config, initial_spawn_point, num_workers=4):
    """Initialize global persistent workers that survive across all shows"""
    global GLOBAL_WORKER_POOL, GLOBAL_TASK_QUEUE, GLOBAL_RESULT_QUEUE

    if GLOBAL_WORKER_POOL is not None:
        return  # Already initialized

    print(f"🚀 Initializing {num_workers} global persistent workers...")

    GLOBAL_TASK_QUEUE = mp.Queue()
    GLOBAL_RESULT_QUEUE = mp.Queue()
    GLOBAL_WORKER_POOL = []

    for i in range(num_workers):
        worker = mp.Process(target=global_persistent_worker,
                            args=(GLOBAL_TASK_QUEUE, GLOBAL_RESULT_QUEUE, config, initial_spawn_point, i))
        worker.start()
        GLOBAL_WORKER_POOL.append(worker)

    print(f"✅ {num_workers} global workers ready")


def global_persistent_worker(task_queue, result_queue, config, initial_spawn_point, worker_id):
    """Global persistent worker that handles episodes across all shows"""
    pid = os.getpid()
    episode_count = 0  # FIXED: Initialize at the top

    try:
        # FIXED: Create a copy of config and add a dummy SAVE_DIR for workers
        worker_config = config.copy()
        worker_config['SAVE_DIR'] = '/tmp'  # Dummy directory since workers don't save files

        # Create simulator ONCE for all shows
        setup_start = time.time()
        temp_sim = ExplorationSimulator(worker_config, initial_spawn_point=initial_spawn_point,
                                        log_path=None, skip_initial_scan=True)
        setup_time = time.time() - setup_start
        print(f"    Global Worker {worker_id} (PID {pid}): Ready in {setup_time:.3f}s")

        while True:
            try:
                task = task_queue.get(timeout=10)  # 10 second timeout

                if task == "SHUTDOWN":
                    break

                show_id, season_origin_pose, explored_at_season_start, goal = task

                # Reuse simulator - just update state
                temp_sim.pose = season_origin_pose
                temp_sim.explored = explored_at_season_start.copy()

                # Run episode (should be ~0.01s after first GPU compilation)
                episode_data = temp_sim.run_episode_to_goal(goal)
                episode_count += 1

                # Send result back
                result_queue.put((show_id, episode_data))

            except Exception as e:
                print(f"    Global Worker {worker_id} episode error: {e}")
                result_queue.put((None, None))
                # Don't break here - continue processing other episodes

    except Exception as e:
        print(f"    Global Worker {worker_id} setup error: {e}")
        result_queue.put((None, None))

    print(f"    Global Worker {worker_id}: Completed {episode_count} episodes across all shows")


def run_episodes_with_global_workers(show_id, season_origin_pose, explored_at_season_start, local_goals):
    """Use global persistent workers (no creation overhead!)"""
    global GLOBAL_TASK_QUEUE, GLOBAL_RESULT_QUEUE

    # Add all episode tasks to global queue
    for goal in local_goals:
        GLOBAL_TASK_QUEUE.put((show_id, season_origin_pose, explored_at_season_start, goal))

    # Collect results
    results = []
    with tqdm(total=len(local_goals), desc="    Episodes") as pbar:
        for _ in range(len(local_goals)):
            result_show_id, episode_data = GLOBAL_RESULT_QUEUE.get()
            if episode_data is not None and result_show_id == show_id:
                results.append(episode_data)
            pbar.update(1)

    return results


def shutdown_global_workers():
    """Shutdown global workers - called at program end"""
    global GLOBAL_WORKER_POOL, GLOBAL_TASK_QUEUE, GLOBAL_RESULT_QUEUE

    if GLOBAL_WORKER_POOL is None:
        return

    print("\n🛑 Shutting down global workers...")

    # Send shutdown signals
    for _ in GLOBAL_WORKER_POOL:
        GLOBAL_TASK_QUEUE.put("SHUTDOWN")

    # Wait for workers to finish
    for i, worker in enumerate(GLOBAL_WORKER_POOL):
        worker.join(timeout=5)
        if worker.is_alive():
            print(f"    Force terminating worker {i}")
            worker.terminate()

    GLOBAL_WORKER_POOL = None
    GLOBAL_TASK_QUEUE = None
    GLOBAL_RESULT_QUEUE = None
    print("✅ Global workers shut down")


# Register cleanup function
atexit.register(shutdown_global_workers)


def sample_goals_on_circumference(simulator, radius_m, density_per_m):
    """
    FIXED: Only sample goals that are reachable AND have clear line-of-sight from robot
    """
    goal_start = time.time()

    # Get simulator state and configuration
    robot_x, robot_y = simulator.pose[:2]
    resolution = simulator.config['RESOLUTION']
    radius_px = radius_m / resolution
    clearance_px = int(simulator.config['SPAWN_CLEARANCE'] / resolution)
    H, W = simulator.occ.shape

    # --- Create a clearance kernel ---
    kernel_start = time.time()
    kernel_size = 2 * clearance_px + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # --- 1. Create Forbidden Zone for Obstacles ---
    occ_mask = (simulator.occ == 1).astype(np.uint8)
    obstacle_forbidden_zone = cv2.dilate(occ_mask, kernel)

    # --- 2. Create Forbidden Zone for the Frontier ---
    unexplored_mask = (~simulator.explored).astype(np.uint8)
    dilated_unexplored = cv2.dilate(unexplored_mask, np.ones((3, 3), np.uint8))
    frontier_mask = (dilated_unexplored == 1) & (simulator.explored)
    frontier_forbidden_zone = cv2.dilate(frontier_mask.astype(np.uint8), kernel)

    # --- 3. Create reachability mask from current position ---
    # Only consider goals that are in the same connected component as the robot
    from scipy.ndimage import label

    # Create a connectivity map (free space that's explored)
    free_and_explored = (simulator.occ == 0) & simulator.explored

    # Find connected components
    labeled_areas, num_areas = label(free_and_explored)

    # Find which area the robot is in
    robot_px, robot_py = int(robot_x), int(robot_y)
    if 0 <= robot_px < W and 0 <= robot_py < H:
        robot_area_label = labeled_areas[robot_py, robot_px]

        if robot_area_label == 0:
            print("Warning: Robot is not in an explored free area!")
            return []

        # Only consider areas reachable from robot's current position
        reachable_mask = (labeled_areas == robot_area_label)
    else:
        print("Warning: Robot is out of bounds!")
        return []

    # --- 4. Combine all forbidden zones + reachability ---
    combined_forbidden_zone = (obstacle_forbidden_zone == 1) | (frontier_forbidden_zone == 1) | (~reachable_mask)

    kernel_time = time.time() - kernel_start

    # Calculate the number of rays to cast based on density
    circumference_m = 2 * np.pi * radius_m
    num_rays = int(circumference_m * density_per_m)

    if num_rays == 0:
        return []

    final_goals = set()

    print(f"  Casting {num_rays} rays to find reachable frontier goals...")
    ray_start = time.time()
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays

        # Iteratively shrink the radius for this angle
        for r in range(int(radius_px), 0, -1):
            px = robot_x + r * np.cos(angle)
            py = robot_y + r * np.sin(angle)
            ix, iy = int(px), int(py)

            # Check if the point at the current radius is valid
            # It must be within bounds, explored, reachable, and NOT in the combined forbidden zone
            if (0 <= ix < W and 0 <= iy < H and
                    simulator.explored[iy, ix] and
                    reachable_mask[iy, ix] and  # Only reachable goals
                    not combined_forbidden_zone[iy, ix]):

                # NEW: Additional check for line-of-sight (no obstacles in straight path)
                has_clear_path = True
                for cx, cy in bresenham(robot_x, robot_y, px, py):
                    cell_x, cell_y = int(cx), int(cy)
                    if not (0 <= cell_x < W and 0 <= cell_y < H):
                        has_clear_path = False
                        break
                    if simulator.occ[cell_y, cell_x] == 1:  # Hit obstacle
                        has_clear_path = False
                        break

                if has_clear_path:
                    final_goals.add((px, py))
                    break  # Found the furthest valid point, move to the next angle

    ray_time = time.time() - ray_start
    goal_time = time.time() - goal_start

    print(f"  📊 GOAL_SAMPLING: Total: {goal_time:.3f}s (kernel: {kernel_time:.3f}s, rays: {ray_time:.3f}s)")
    print(
        f"  Generated {len(final_goals)} reachable goals with clear line-of-sight from robot area {robot_area_label}.")
    return list(final_goals)

def sample_new_season_origin(simulator, last_origin_pos, min_dist_m, max_dist_m):
    origin_start = time.time()

    print(f"  Searching for new season origin >{min_dist_m:.1f}m and <{max_dist_m:.1f}m away...")
    cfg = simulator.config
    resolution = cfg['RESOLUTION']
    min_dist_px, max_dist_px = min_dist_m / resolution, max_dist_m / resolution
    try:
        sample_start = time.time()
        xs, ys = sample_goal(simulator.explored, simulator.occ, resolution, cfg['SPAWN_CLEARANCE'],
                             current_pos=last_origin_pos, max_dist=max_dist_px, __return_all=True)
        sample_time = time.time() - sample_start

        if xs.size == 0:
            raise RuntimeError("No valid points found in the max radius.")

        filter_start = time.time()
        dist_sq = (xs - last_origin_pos[0]) ** 2 + (ys - last_origin_pos[1]) ** 2
        annulus_mask = (dist_sq > min_dist_px ** 2)
        final_xs, final_ys = xs[annulus_mask], ys[annulus_mask]
        filter_time = time.time() - filter_start

        if final_xs.size == 0:
            raise RuntimeError("Could not find a suitable new season origin in the annulus.")

        idx = random.randrange(len(final_xs))
        result = float(final_xs[idx]), float(final_ys[idx])

        origin_time = time.time() - origin_start
        print(f"  📊 SEASON_ORIGIN: Total: {origin_time:.3f}s (sample: {sample_time:.3f}s, filter: {filter_time:.3f}s)")
        return result

    except RuntimeError as e:
        origin_time = time.time() - origin_start
        print(f"  Error sampling new season origin: {e}")
        print(f"  📊 SEASON_ORIGIN: Failed after {origin_time:.3f}s")
        return None


def persistent_episode_worker(worker_queue, result_queue, config, initial_spawn_point):
    """
    NEW: Persistent worker that reuses the same simulator for multiple episodes
    """
    worker_pid = os.getpid()

    try:
        # Create simulator ONCE per worker process
        setup_start = time.time()
        temp_sim = ExplorationSimulator(config, initial_spawn_point=initial_spawn_point,
                                        log_path=None, skip_initial_scan=True)
        setup_time = time.time() - setup_start
        print(f"    Worker {worker_pid}: Simulator initialized in {setup_time:.3f}s")

        episode_count = 0

        while True:
            try:
                # Get episode task with timeout
                task = worker_queue.get(timeout=2)
                if task is None:  # Poison pill to shutdown
                    break

                season_origin_pose, explored_at_season_start, goal = task

                # Reuse the same simulator - just update state (very fast)
                temp_sim.pose = season_origin_pose
                temp_sim.explored = explored_at_season_start.copy()

                # Run episode (this should be fast now ~0.01s)
                episode_start = time.time()
                episode_data = temp_sim.run_episode_to_goal(goal)
                episode_time = time.time() - episode_start

                episode_count += 1

                # Log first few episodes per worker
                if episode_count <= 3:
                    print(
                        f"        Worker {worker_pid} Episode {episode_count}: {episode_time:.3f}s ({len(episode_data)} steps)")

                # Send result back
                result_queue.put(episode_data)

            except Exception as e:
                print(f"    Worker {worker_pid} error: {e}")
                result_queue.put(None)  # Signal error
                break

    except Exception as e:
        print(f"    Worker {worker_pid} setup error: {e}")
        result_queue.put(None)

    print(f"    Worker {worker_pid}: Completed {episode_count} episodes, shutting down")


def run_episodes_with_persistent_workers(config, initial_spawn_point, season_origin_pose,
                                         explored_at_season_start, local_goals, num_workers=4):
    """
    NEW: Run episodes using persistent worker processes
    """
    print(f"    Starting {num_workers} persistent workers for {len(local_goals)} episodes...")

    # Create queues
    worker_queue = mp.Queue()
    result_queue = mp.Queue()

    # Add all episode tasks to queue
    for goal in local_goals:
        worker_queue.put((season_origin_pose, explored_at_season_start, goal))

    # Start persistent workers
    workers = []
    for i in range(num_workers):
        worker = mp.Process(target=persistent_episode_worker,
                            args=(worker_queue, result_queue, config, initial_spawn_point))
        worker.start()
        workers.append(worker)

    # Collect results with progress bar
    results = []
    with tqdm(total=len(local_goals), desc="    Episodes") as pbar:
        for _ in range(len(local_goals)):
            result = result_queue.get()
            if result is not None:
                results.append(result)
            pbar.update(1)

    # Shutdown workers
    print(f"    Shutting down {len(workers)} workers...")
    for _ in workers:
        worker_queue.put(None)  # Poison pill

    for worker in workers:
        worker.join(timeout=5)
        if worker.is_alive():
            print(f"    Warning: Worker {worker.pid} did not shut down gracefully")
            worker.terminate()

    return results


# OLD episode_worker function (keep for reference but not used)
def episode_worker(args):
    episode_start = time.time()
    config, initial_spawn_point, season_origin_pose, explored_at_season_start, goal = args

    # Time simulator creation
    sim_start = time.time()
    temp_sim = ExplorationSimulator(config, initial_spawn_point=initial_spawn_point,
                                    log_path=None, skip_initial_scan=True)
    sim_time = time.time() - sim_start

    # Time state setup
    setup_start = time.time()
    temp_sim.pose = season_origin_pose
    temp_sim.explored = explored_at_season_start.copy()  # Inherit the exploration
    setup_time = time.time() - setup_start

    # Time episode execution
    exec_start = time.time()
    episode_data = temp_sim.run_episode_to_goal(goal)
    exec_time = time.time() - exec_start

    episode_time = time.time() - episode_start

    # Only print for first few episodes to avoid spam
    if not hasattr(episode_worker, 'count'):
        episode_worker.count = 0
    episode_worker.count += 1

    if episode_worker.count <= 5:
        print(
            f"        📊 EPISODE {episode_worker.count}: Total: {episode_time:.3f}s (sim: {sim_time:.3f}s, setup: {setup_time:.3f}s, exec: {exec_time:.3f}s, steps: {len(episode_data)})")

    return episode_data


def run_a_single_show(config, show_id, initial_spawn_point, skip_worker_init=False):
    """
    PARALLEL SHOWS VERSION with optional worker initialization skip.

    Args:
        skip_worker_init: If True, assumes GPU workers already initialized (for persistent workers)
    """
    show_start = time.time()
    process_name = mp.current_process().name
    print(f"🕒 PROFILING: {process_name} STARTING SHOW {show_id}...")

    # Initialize global workers ONLY if not skipped
    if not skip_worker_init:
        initialize_global_workers(config, initial_spawn_point, num_workers=config.get('NUM_WORKERS', 4))


    # Time initialization
    init_start = time.time()
    base_output_dir = config.get('BASE_OUTPUT_DIR', '.')
    show_save_dir = os.path.join(base_output_dir, f"show_{show_id:02d}_data")
    config['SAVE_DIR'] = show_save_dir
    hdf5_path = os.path.join(base_output_dir, f"show_data_{show_id:03d}.h5")
    init_time = time.time() - init_start

    # Time initialization
    # init_start = time.time()
    # base_output_dir = config.get('BASE_OUTPUT_DIR', '.')
    # show_save_dir = os.path.join(base_output_dir, f"show_{show_id:02d}_data")
    # config['SAVE_DIR'] = show_save_dir
    # hdf5_path = os.path.join(base_output_dir, f"show_data_{show_id:03d}.h5")
    # init_time = time.time() - init_start

    # Time simulator creation
    sim_start = time.time()
    simulator = ExplorationSimulator(config, initial_spawn_point=initial_spawn_point, log_path=None)
    sim_time = time.time() - sim_start

    # Time HDF5 setup
    hdf5_setup_start = time.time()
    img_h, img_w = config['WIN_PX'], config['WIN_PX']
    hdf5_setup_time = time.time() - hdf5_setup_start

    print(f"  📊 SHOW: Init: {init_time:.3f}s, Simulator: {sim_time:.3f}s, HDF5 setup: {hdf5_setup_time:.3f}s")

    # Open the HDF5 file once for this entire show
    with h5py.File(hdf5_path, 'w') as hf:
        # === SHOW-LEVEL METADATA ===
        # Map and environment
        hf.attrs['map_pgm'] = str(config.get('MAP_PGM', ''))
        hf.attrs['map_alias'] = os.path.basename(config.get('MAP_PGM', '')).replace('.pgm', '')
        hf.attrs['initial_spawn_point'] = np.array(initial_spawn_point, dtype=np.float32)  # [x_px, y_px]

        # Show structure
        hf.attrs['show_id'] = int(show_id)
        hf.attrs['seasons_per_show'] = int(config.get('SEASONS_PER_SHOW', 0))
        hf.attrs['num_workers'] = int(config.get('NUM_WORKERS', 1))

        # Robot configuration
        hf.attrs['robot_radius'] = float(config.get('ROBOT_R', 0.0))  # meters
        hf.attrs['spawn_clearance'] = float(config.get('SPAWN_CLEARANCE', 0.0))  # meters
        hf.attrs['step_size_m'] = float(config.get('STEP_SIZE_M', 0.0))  # meters
        hf.attrs['angular_velocity'] = float(config.get('ANG_VEL', 0.0))  # rad/s
        hf.attrs['dt'] = float(config.get('DT', 0.0))  # seconds

        # Sensor configuration
        hf.attrs['fov_rad'] = float(config.get('FOV', 0.0))  # radians
        hf.attrs['n_beams'] = int(config.get('N_BEAMS', 0))
        hf.attrs['max_range_px'] = int(config.get('MAX_R', 0))  # pixels
        hf.attrs['max_range_m'] = float(config.get('MAX_R', 0) * config.get('RESOLUTION', 0.0))  # meters

        # Window and visualization
        hf.attrs['win_px'] = int(config.get('WIN_PX', 0))  # pixels
        hf.attrs['win_m'] = float(config.get('WIN_M', 0.0))  # meters

        # Exploration parameters
        hf.attrs['inner_radius_m'] = float(config.get('INNER_RADIUS_R_M', 0.0))  # meters
        hf.attrs['outer_radius_m'] = float(config.get('OUTER_RADIUS_D_M', 0.0))  # meters
        hf.attrs['goal_density_per_meter'] = float(config.get('GOAL_DENSITY_PER_METER', 0.0))
        hf.attrs['max_goal_dist_m'] = float(config.get('MAX_GOAL_DIST_M', 0.0))  # meters

        # Data collection timestamp and version
        hf.attrs['creation_timestamp'] = time.time()
        hf.attrs['creation_date'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        hf.attrs['python_version'] = platform.python_version()
        hf.attrs['host_id'] = (
            platform.node() if os.environ.get('INTELX_RECORD_HOSTNAME')
            else hashlib.sha1(platform.node().encode()).hexdigest()[:8])

        # Time dataset creation
        dataset_start = time.time()
        origin_dset = hf.create_dataset('origin_images', (0, img_h, img_w), maxshape=(None, img_h, img_w),
                                        dtype='uint8', compression='gzip')
        heatmap_dset = hf.create_dataset('target_heatmaps', (0, img_h, img_w), maxshape=(None, img_h, img_w),
                                         dtype='float32', compression='gzip')
        dataset_time = time.time() - dataset_start
        print(f"  📊 SHOW: Dataset creation: {dataset_time:.3f}s")

        season_count_in_file = 0
        total_episodes = 0

        for season_index in range(config['SEASONS_PER_SHOW']):
            season_start = time.time()
            print(f"\n🕒 {process_name}: Show {show_id}, Season {season_index + 1}/{config['SEASONS_PER_SHOW']}")

            # --- Season Transition Logic (if applicable) ---
            if season_index > 0:
                transition_start = time.time()
                previous_season_origin = simulator.pose[:2]
                new_origin = sample_new_season_origin(simulator, last_origin_pos=previous_season_origin,
                                                      min_dist_m=config['INNER_RADIUS_R_M'],
                                                      max_dist_m=config['OUTER_RADIUS_D_M'])
                if not new_origin:
                    print(f"{process_name}: Could not find new origin. Ending show early.")
                    break

                print(f"  Transitioning to new season origin...")
                nav_start = time.time()
                simulator.run_episode_to_goal(new_origin)
                nav_time = time.time() - nav_start

                transition_time = time.time() - transition_start
                print(f"  📊 SEASON: Transition: {transition_time:.3f}s (navigation: {nav_time:.3f}s)")

            # --- In-Memory Data Collection for the Current Season ---
            data_prep_start = time.time()
            season_origin_pose = simulator.pose
            explored_at_season_start = simulator.explored.copy()

            origin_img_start = time.time()
            season_origin_image = simulator._build_origin_image(season_origin_pose,
                                                                simulator._simulate_scan(season_origin_pose))
            origin_img_time = time.time() - origin_img_start

            local_goals = sample_goals_on_circumference(
                simulator,
                radius_m=config['INNER_RADIUS_R_M'],
                density_per_m=config['GOAL_DENSITY_PER_METER']
            )
            if not local_goals:
                print(f"  No valid local goals found for Season {season_index + 1}. Skipping.")
                continue

            data_prep_time = time.time() - data_prep_start
            print(f"  📊 SEASON: Data prep: {data_prep_time:.3f}s (origin_img: {origin_img_time:.3f}s)")

            # --- USE GLOBAL PERSISTENT WORKERS (NO CREATION/DESTRUCTION!) ---
            print(f"  Using global workers for {len(local_goals)} episodes in Season {season_index + 1}...")

            mp_start = time.time()

            # Use global workers - no worker creation overhead!
            results = run_episodes_with_global_workers(
                show_id, season_origin_pose, explored_at_season_start, local_goals
            )

            mp_time = time.time() - mp_start
            print(f"  📊 SEASON: Global workers: {mp_time:.3f}s ({len(results)} episodes)")

            # --- FLATTEN THE RESULTS ---
            flatten_start = time.time()
            all_season_steps_data = [point for episode_list in results if episode_list for point in episode_list]
            flatten_time = time.time() - flatten_start

            # --- Process and Write this Season's Data to HDF5 ---
            if all_season_steps_data:
                process_start = time.time()
                sparse_heatmap = np.zeros((img_h, img_w), dtype=np.float32)

                # To handle multiple visits to the same pixel, we'll store all gains and average them
                pixel_gains = {}
                for (px, py), gain in all_season_steps_data:
                    if 0 <= py < img_h and 0 <= px < img_w:
                        if (px, py) not in pixel_gains:
                            pixel_gains[(px, py)] = []
                        pixel_gains[(px, py)].append(gain)

                for (px, py), gains in pixel_gains.items():
                    sparse_heatmap[py, px] = np.mean(gains)
                process_time = time.time() - process_start

                # Time HDF5 writing
                write_start = time.time()
                origin_dset.resize(season_count_in_file + 1, axis=0)
                heatmap_dset.resize(season_count_in_file + 1, axis=0)
                origin_dset[season_count_in_file] = season_origin_image
                heatmap_dset[season_count_in_file] = sparse_heatmap
                season_count_in_file += 1
                write_time = time.time() - write_start

                print(f"  📊 SEASON: Data processing: {process_time:.3f}s, HDF5 write: {write_time:.3f}s")
                print(f"  📊 SEASON: Flatten: {flatten_time:.3f}s, Total data points: {len(all_season_steps_data)}")

            season_time = time.time() - season_start
            total_episodes += len(results)
            print(
                f"\n🏁 {process_name}: Show {show_id}, Season {season_index + 1} COMPLETE: {season_time:.3f}s ({len(results)} episodes)")

        # === FINAL SHOW-LEVEL STATISTICS ===
        hf.attrs['total_seasons_completed'] = season_count_in_file
        hf.attrs['total_episodes'] = total_episodes
        hf.attrs['show_duration_seconds'] = time.time() - show_start

    show_time = time.time() - show_start


    print(f"🏁 {process_name}: SHOW {show_id} TOTAL: {show_time:.3f}s ({total_episodes} total episodes)")
    return show_time
