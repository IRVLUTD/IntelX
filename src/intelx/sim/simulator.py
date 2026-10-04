#!/usr/bin/env python3
import cv2
import numpy as np
import time
import os
import random
import heapq
from math import cos, sin, pi, atan2, hypot
from numba import njit, cuda # <--- ADD THIS IMPORT
import math
from intelx.planners.rrt_star import RRTStar
from intelx.data.collector import DataCollector


def astar_path_planner(occ_map, start_px, goal_px):
    """
    A robust A* grid-based path planner.
    """
    H, W = occ_map.shape
    start_node = (int(start_px[0]), int(start_px[1]))
    goal_node = (int(goal_px[0]), int(goal_px[1]))

    # Check if start or goal are inside an obstacle
    if not (0 <= start_node[0] < W and 0 <= start_node[1] < H) or occ_map[start_node[1], start_node[0]] == 1:
        print("A* Warning: Start node is in an obstacle or out of bounds.")
        return None
    if not (0 <= goal_node[0] < W and 0 <= goal_node[1] < H) or occ_map[goal_node[1], goal_node[0]] == 1:
        print("A* Warning: Goal node is in an obstacle or out of bounds.")
        return None

    def heuristic(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    # Neighbors include diagonals for smoother paths
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    open_set = [(0, start_node)]  # (priority, node)
    came_from = {}
    cost_so_far = {start_node: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return [(float(x), float(y)) for x, y in reversed(path)]  # Return as list of float tuples

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < W and 0 <= neighbor[1] < H):
                continue
            if occ_map[neighbor[1], neighbor[0]] == 1:
                continue

            new_cost = cost_so_far[current] + hypot(dx, dy)  # Cost is Euclidean distance
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    print("A* Warning: No path found.")
    return None  # No path found


class ExplorationSimulator:
    """
    An object-oriented class to manage and run the robot exploration simulation.
    """

    def __init__(self, config, initial_spawn_point=None, log_path=None, skip_initial_scan=False):
        """
        MODIFIED: `log_path` is ignored. This simulator is now fully "headless"
        and does not use a DataCollector.
        skip_initial_scan: True for episode workers (inherit exploration state)
                      False for main simulator (do the initial scan)
        """
        self.config = config
        self._setup_directories()
        self.occ = self._load_pgm_map(config['MAP_PGM'])
        self.occ = self._seal_by_adjacent(self.occ)
        self.H, self.W = self.occ.shape
        boundary_mask = self._find_boundary_obstacles()
        boundary_ys, boundary_xs = np.nonzero(boundary_mask)
        self.obs_list_meters = [(x * config['RESOLUTION'], y * config['RESOLUTION'], config['ROBOT_R'])
                                for x, y in zip(boundary_xs, boundary_ys)]
        self.explored = np.zeros_like(self.occ, dtype=bool)

        if initial_spawn_point is not None:
            x, y = initial_spawn_point
        else:
            x, y = self._find_spawn()
        self.pose = (x, y, 0.0)

        if not skip_initial_scan:
            # MAIN SIMULATOR: Do the initial scan once
            self._perform_initial_scan()
        else:
            # EPISODE WORKERS: Start with empty exploration (will be overwritten)
            self.explored = np.zeros_like(self.occ, dtype=bool)

        self.collector = None

    def run_simulation(self, num_episodes):
        # This method can be adapted for more complex scenarios,
        # like the seasonal collector script.
        for ep in range(num_episodes):
            print(f"\n--- Starting Episode {ep + 1}/{num_episodes} ---")
            try:
                goal_x, goal_y = self._sample_goal()
                print(f"Sampled Goal: ({goal_x:.1f}, {goal_y:.1f})")
                origin_map_fname = self.create_and_save_season_origin_map(self.pose)
                self.run_episode_to_goal((goal_x, goal_y), origin_map_fname)
            except RuntimeError as e:
                print(f"Could not sample new goal: {e}")
                break
        print("\nAll episodes completed. Closing collector.")
        self.collector.close()

    def create_and_save_season_origin_map(self, origin_pose, goals_to_visualize=None):
        cfg = self.config
        ranges = self._simulate_scan(origin_pose)
        goals_for_viz = goals_to_visualize if cfg.get('VISUALIZE_SEASON_GOALS') else None
        img = self._build_origin_image(origin_pose, ranges, goals_to_visualize=goals_for_viz)

        # CORRECTED: Changed self.collector.season to self.collector.season_count
        fname = os.path.join(cfg['SAVE_DIR'], f"origin_scan_s{self.collector.season:04d}.png")

        cv2.imwrite(fname, img)
        return fname

    def run_episode_to_goal(self, goal):
        """
        UPDATED: Use full A* path for long distances, direct for short
        """
        episode_start = time.time()

        # Time path planning
        plan_start = time.time()
        path = self._plan_path(goal)
        plan_time = time.time() - plan_start

        if path is None or len(path) == 0:
            print("Path planning failed for this episode.")
            return []

        # Time path conversion
        convert_start = time.time()

        # Check distance to decide how to use the path
        start_x, start_y = self.pose[:2]
        goal_x, goal_y = goal
        distance = hypot(goal_x - start_x, goal_y - start_y)

        if distance > 100:
            # Long distance: Use FULL A* path for proper navigation
            path_cells = [(xm / self.config['RESOLUTION'], ym / self.config['RESOLUTION']) for xm, ym in path]
            print(f"    Following full A* path with {len(path_cells)} waypoints")
        else:
            # Short distance: Direct to goal (existing behavior)
            path_cells = [(xm / self.config['RESOLUTION'], ym / self.config['RESOLUTION']) for xm, ym in path[0:1]]
            # print(f"    Going direct to goal")

        convert_time = time.time() - convert_start

        # Time coverage calculation
        coverage_start = time.time()
        origin_pose = self.pose
        origin_coverage = self._calculate_coverage(origin_pose)
        coverage_time = time.time() - coverage_start

        # Time path following (the big one!)
        follow_start = time.time()
        episode_data = self._follow_path(path_cells, origin_pose, origin_coverage, episode_goal=goal)
        follow_time = time.time() - follow_start

        episode_time = time.time() - episode_start

        # Log timing for first few episodes
        if not hasattr(self, 'episode_debug_count'):
            self.episode_debug_count = 0
        self.episode_debug_count += 1

        if self.episode_debug_count <= 3:
            print(f"          📊 EXEC_BREAKDOWN: Total: {episode_time:.3f}s")
            print(f"            - Plan: {plan_time:.3f}s")
            print(f"            - Convert: {convert_time:.3f}s")
            print(f"            - Coverage: {coverage_time:.3f}s")
            print(f"            - Follow: {follow_time:.3f}s ({len(episode_data)} steps)")

        return episode_data

    def _perform_initial_scan(self):
        # print("Performing initial 360-degree scan...")
        x, y, theta = self.pose
        accumulated = 0.0
        while accumulated < 2 * pi:
            dθ = self.config['ANG_VEL'] * self.config['DT']
            theta += dθ
            accumulated += abs(dθ)
            self._update_exploration((x, y, theta))
        self.pose = (x, y, theta)
        print("Initial scan complete.")

    def _sample_goal(self, origin_pos=None, max_dist_m=None):
        cfg = self.config
        pos = origin_pos if origin_pos is not None else self.pose[:2]
        dist_m = max_dist_m if max_dist_m is not None else cfg['MAX_GOAL_DIST_M']
        max_dist_pixels = dist_m / cfg['RESOLUTION']
        return self._sample_goal_util(self.explored, self.occ, cfg['RESOLUTION'], cfg['SPAWN_CLEARANCE'], current_pos=pos, max_dist=max_dist_pixels)

    def _plan_path(self, goal):
        """Auto-detect planning strategy with correct output format - A* replaces RRT*"""
        start_x, start_y = self.pose[:2]
        goal_x, goal_y = goal

        distance = hypot(goal_x - start_x, goal_y - start_y)

        # Use A* for long distances (likely season transitions)
        if distance > 60:  # 10m at 0.05 resolution - likely season transition
            print(f"    Using A* for long distance: {distance:.1f} pixels")

            # A* works in pixel coordinates
            path_px = astar_path_planner(self.occ, (start_x, start_y), (goal_x, goal_y))

            if path_px is None:
                print("    A* failed, trying direct path fallback...")
                # Simple fallback: direct path
                cfg = self.config
                start_m = [start_x * cfg['RESOLUTION'], start_y * cfg['RESOLUTION']]
                goal_m = [goal_x * cfg['RESOLUTION'], goal_y * cfg['RESOLUTION']]
                return [goal_m, start_m]

            # Convert A* path (pixels) to meters for robot following
            cfg = self.config
            path_meters = [[x * cfg['RESOLUTION'], y * cfg['RESOLUTION']] for x, y in path_px]

            # A* returns path from start to goal, but your system expects goal to start
            return list(reversed(path_meters))
        else:
            # FIXED: Simple path that matches format
            cfg = self.config

            if distance <= 60:
                # Direct path: convert pixels to meters, return as list of lists
                start_m = [start_x * cfg['RESOLUTION'], start_y * cfg['RESOLUTION']]
                goal_m = [goal_x * cfg['RESOLUTION'], goal_y * cfg['RESOLUTION']]
                return [goal_m, start_m]  # Goal first, then start
            else:
                # Waypoint path: convert pixels to meters
                num_waypoints = max(2, int(distance / (cfg["INNER_RADIUS_R_M"] / cfg["STEP_SIZE_M"])))
                path_meters = []

                for i in range(num_waypoints + 1):
                    t = i / num_waypoints
                    wx = start_x + t * (goal_x - start_x)
                    wy = start_y + t * (goal_y - start_y)
                    # Convert to meters and add as [x, y] list
                    path_meters.append([wx * cfg['RESOLUTION'], wy * cfg['RESOLUTION']])

                # Return path from goal to start
                return list(reversed(path_meters))

    def _follow_path_with_batching(self, path_cells, origin_pose, origin_coverage, episode_goal):
        """
        Process path with controlled batch sizes to manage GPU memory
        """
        cfg = self.config
        x, y, theta = self.pose
        collected_episode_data = []

        # Calculate all poses
        all_poses = []

        for wx, wy in path_cells:
            dx, dy = wx - x, wy - y
            dist_m = hypot(dx, dy) * cfg['RESOLUTION']
            if dist_m < 1e-6:
                continue

            # Calculate desired heading
            desired_heading = atan2(dy, dx)

            # Add turning poses (if needed for smooth rotation)
            angle_diff = self._normalize_angle(desired_heading - theta)
            while abs(angle_diff) > 0.05:
                sign = 1 if angle_diff > 0 else -1
                dθ = sign * min(cfg['ANG_VEL'] * 0.05, abs(angle_diff))
                theta = self._normalize_angle(theta + dθ)
                all_poses.append((x, y, theta))
                angle_diff = self._normalize_angle(desired_heading - theta)

            # Add movement poses
            steps = int(dist_m / cfg['STEP_SIZE_M'])
            if steps > 0:
                for step in range(steps):
                    frac = (step + 1) / steps
                    nx, ny = x + frac * dx, y + frac * dy
                    all_poses.append((nx, ny, desired_heading))

            x, y, theta = wx, wy, desired_heading

        if not all_poses:
            return collected_episode_data

        # Process in batches to manage GPU memory
        batch_size = 500  # Adjust based on GPU memory

        for batch_start in range(0, len(all_poses), batch_size):
            batch_end = min(batch_start + batch_size, len(all_poses))
            batch_poses = all_poses[batch_start:batch_end]

            # Batch GPU raytracing
            batch_scans = gpu_simulate_scan_batch(
                self.occ, batch_poses, cfg['MAX_R'], cfg['N_BEAMS'], cfg['FOV']
            )

            # Process batch results
            for pose, ranges in zip(batch_poses, batch_scans):
                self._update_exploration(pose, ranges)
                data_point = self._process_step_in_memory(pose, origin_pose, origin_coverage, ranges)
                if data_point:
                    collected_episode_data.append(data_point)

        if all_poses:
            self.pose = all_poses[-1]

        return collected_episode_data

    def _follow_path(self, path_cells, origin_pose, origin_coverage, episode_goal):
        """
        SINGLE OPTIMIZED VERSION: Batch GPU raytracing for entire episode path
        """
        follow_start = time.time()

        cfg = self.config
        x, y, theta = self.pose
        collected_episode_data = []

        # Time pose calculation
        pose_calc_start = time.time()
        all_poses = []

        for wx, wy in path_cells:
            dx, dy = wx - x, wy - y
            dist_m = hypot(dx, dy) * cfg['RESOLUTION']
            if dist_m < 1e-6:
                continue

            # Calculate desired heading
            desired_heading = atan2(dy, dx)

            # Add turning poses (if needed for smooth rotation)
            angle_diff = self._normalize_angle(desired_heading - theta)
            while abs(angle_diff) > 0.05:
                sign = 1 if angle_diff > 0 else -1
                dθ = sign * min(cfg['ANG_VEL'] * 0.05, abs(angle_diff))
                theta = self._normalize_angle(theta + dθ)
                all_poses.append((x, y, theta))
                angle_diff = self._normalize_angle(desired_heading - theta)

            # Add movement poses
            steps = int(dist_m / cfg['STEP_SIZE_M'])
            if steps > 0:
                for step in range(steps):
                    frac = (step + 1) / steps
                    nx, ny = x + frac * dx, y + frac * dy
                    all_poses.append((nx, ny, desired_heading))

            x, y, theta = wx, wy, desired_heading

        pose_calc_time = time.time() - pose_calc_start

        if not all_poses:
            return collected_episode_data

        # Time batch GPU scanning
        gpu_scan_start = time.time()
        batch_scans = gpu_simulate_scan_batch(
            self.occ, all_poses, cfg['MAX_R'], cfg['N_BEAMS'], cfg['FOV']
        )
        gpu_scan_time = time.time() - gpu_scan_start

        # Time processing loop
        process_start = time.time()
        exploration_time = 0
        data_processing_time = 0

        for pose, ranges in zip(all_poses, batch_scans):
            # Time exploration update
            exp_start = time.time()
            self._update_exploration(pose, ranges)
            exploration_time += time.time() - exp_start

            # Time data processing
            data_start = time.time()
            data_point = self._process_step_in_memory(pose, origin_pose, origin_coverage, ranges)
            if data_point:
                collected_episode_data.append(data_point)
            data_processing_time += time.time() - data_start

        process_time = time.time() - process_start

        # Update final pose
        if all_poses:
            self.pose = all_poses[-1]

        follow_time = time.time() - follow_start

        # Log detailed breakdown for first few episodes
        if not hasattr(self, 'follow_debug_count'):
            self.follow_debug_count = 0
        self.follow_debug_count += 1

        if self.follow_debug_count <= 3:
            print(f"            📊 FOLLOW_BREAKDOWN: {follow_time:.3f}s ({len(all_poses)} steps)")
            print(f"              - Pose calc: {pose_calc_time:.3f}s")
            print(f"              - GPU batch scan: {gpu_scan_time:.3f}s")
            print(f"              - Processing loop: {process_time:.3f}s")
            print(
                f"                • Exploration: {exploration_time:.3f}s ({exploration_time / len(all_poses) * 1000:.1f}ms/step)")
            print(
                f"                • Data processing: {data_processing_time:.3f}s ({data_processing_time / len(all_poses) * 1000:.1f}ms/step)")

        return collected_episode_data

    def _calculate_robot_coordinates_fast(self, current_pose, origin_pose):
        """
        SIMPLIFIED: Use OpenCV's rotation matrix directly
        """
        center = self.config['WIN_PX'] // 2
        origin_x, origin_y, origin_theta = origin_pose
        current_x, current_y, current_theta = current_pose

        # Step 1: Unrotated coordinates
        robot_dx = current_x - origin_x
        robot_dy = current_y - origin_y
        robot_img_x_unrotated = center + robot_dx  # Don't convert to int yet
        robot_img_y_unrotated = center + robot_dy

        # Step 2: Use OpenCV's rotation matrix (same as image building)
        angle_deg = 90.0 - (origin_theta * 180.0 / pi)

        # Create the same rotation matrix as cv2.getRotationMatrix2D
        import cv2
        M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)

        # Apply transformation to the point
        point = np.array([[robot_img_x_unrotated], [robot_img_y_unrotated], [1.0]])
        rotated_point = M @ point

        final_px = int(round(rotated_point[0, 0]))
        final_py = int(round(rotated_point[1, 0]))

        # Check bounds
        window_px = self.config['WIN_PX']
        if 0 <= final_px < window_px and 0 <= final_py < window_px:
            return final_px, final_py
        else:
            return None

    # def _calculate_robot_coordinates_fast(self, current_pose, origin_pose):
    #     """
    #     FAST: Extract robot coordinates without building the full image
    #     This replicates the EXACT same logic as build_local_image_fixed_origin_same_frame
    #     """
    #     center = self.config['WIN_PX'] // 2
    #     origin_x, origin_y, origin_theta = origin_pose
    #     current_x, current_y, current_theta = current_pose
    #
    #     # STEP 1: Calculate robot position in unrotated image space
    #     # (This is identical to your image building logic)
    #     # FIXED: Flip both axes to match heatmap coordinate system
    #     robot_dx = current_x - origin_x
    #     robot_dy = current_y - origin_y
    #     robot_img_x_unrotated = int(center - robot_dx)  # X flipped
    #     robot_img_y_unrotated = int(center - robot_dy)  # Y flipped
    #     # robot_dx = current_x - origin_x
    #     # robot_dy = current_y - origin_y
    #     # robot_img_x_unrotated = int(center + robot_dx)
    #     # robot_img_y_unrotated = int(center + robot_dy)
    #     # DEBUG: Print coordinate transformation
    #     print(f"DEBUG COORDS:")
    #     print(f"  Origin: ({origin_x:.1f}, {origin_y:.1f}, {origin_theta:.3f})")
    #     print(f"  Current: ({current_x:.1f}, {current_y:.1f}, {current_theta:.3f})")
    #     print(f"  Delta: dx={robot_dx:.1f}, dy={robot_dy:.1f}")
    #     print(f"  Image coords (unrotated): ({robot_img_x_unrotated}, {robot_img_y_unrotated})")
    #
    #     # STEP 2: Apply rotation transformation
    #     # (This matches your final rotation step)
    #     angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    #     angle_rad = angle_deg * pi / 180.0
    #
    #     # Rotate around center using same transformation as cv2.getRotationMatrix2D
    #     cos_angle = cos(angle_rad)
    #     sin_angle = sin(angle_rad)
    #
    #     # Translate to origin, rotate, translate back
    #     dx = robot_img_x_unrotated - center
    #     dy = robot_img_y_unrotated - center
    #
    #     rotated_x = cos_angle * dx - sin_angle * dy + center
    #     rotated_y = sin_angle * dx + cos_angle * dy + center
    #
    #     # Convert to final pixel coordinates
    #     final_px = int(rotated_x)
    #     final_py = int(rotated_y)
    #
    #     # Check bounds (same as your original bounds check)
    #     window_px = self.config['WIN_PX']
    #     if 0 <= final_px < window_px and 0 <= final_py < window_px:
    #         return final_px, final_py
    #     else:
    #         return None

    # def _process_step_in_memory(self, current_pose, origin_pose, origin_coverage, ranges):
    #     """Add validation to check coordinate consistency"""
    #
    #     # Fast method
    #     fast_coords = self._calculate_robot_coordinates_fast(current_pose, origin_pose)
    #
    #     # Slow method (for validation only - remove this after debugging)
    #     query_image_np = self._build_local_image(current_pose, origin_pose, ranges)
    #     robot_coords_slow = np.where(query_image_np == 200)
    #
    #     if robot_coords_slow[0].size > 0:
    #         y_min, y_max = robot_coords_slow[0].min(), robot_coords_slow[0].max()
    #         x_min, x_max = robot_coords_slow[1].min(), robot_coords_slow[1].max()
    #         px_slow = int((x_min + x_max) / 2)
    #         py_slow = int((y_min + y_max) / 2)
    #
    #         if fast_coords is not None:
    #             px_fast, py_fast = fast_coords
    #             coord_diff = abs(px_fast - px_slow) + abs(py_fast - py_slow)
    #
    #             if coord_diff > 2:  # Allow small rounding differences
    #                 print(f"⚠️ COORDINATE MISMATCH!")
    #                 print(f"  Fast: ({px_fast}, {py_fast})")
    #                 print(f"  Slow: ({px_slow}, {py_slow})")
    #                 print(f"  Difference: {coord_diff} pixels")
    #
    #     # Use fast result
    #     if fast_coords is not None:
    #         current_coverage = self._calculate_coverage_with_map(self.explored, origin_pose)
    #         coverage_gain = current_coverage - origin_coverage
    #         return (fast_coords, coverage_gain)
    #
    #     return None

    def _process_step_in_memory(self, current_pose, origin_pose, origin_coverage, ranges):
        """
        ULTRA-FAST: Skip image building, calculate robot coords directly
        """
        # Calculate coverage gain directly
        current_coverage = self._calculate_coverage_with_map(self.explored, origin_pose)
        coverage_gain = current_coverage - origin_coverage

        # Get robot coordinates using the exact same logic as image building
        robot_coords = self._calculate_robot_coordinates_fast(current_pose, origin_pose)

        if robot_coords is not None:
            px, py = robot_coords
            return ((px, py), coverage_gain)

        return None

    def _calculate_coverage_with_map(self, explored_map, pose):
        """Helper to calculate coverage on a specific explored map."""
        return calculate_coverage_fixed_origin(explored_map, pose, self.config['WIN_PX'])

    def _log_simulation_step(self, current_pose, origin_pose, ranges, episode_goal):
        cfg = self.config
        img_fname = f"{cfg['SAVE_DIR']}/frame_{self.frame_count:09d}.png"
        img = self._build_local_image(current_pose, origin_pose, ranges, episode_goal=episode_goal)
        cv2.imwrite(img_fname, img)
        coverage = self._calculate_coverage(origin_pose)
        self.collector.log_step(query=current_pose, query_map_fname=img_fname, coverage=coverage)
        self.frame_count += 1

    def _update_exploration(self, pose, ranges=None):
        """OPTIMIZED: Ultra-fast exploration update using Numba"""
        cfg = self.config
        if ranges is None:
            ranges = self._simulate_scan(pose)

        x, y, theta = pose
        scan_angles = np.linspace(-cfg['FOV'] / 2, cfg['FOV'] / 2, cfg['N_BEAMS']) + theta

        # Call the ultra-fast Numba-compiled function
        update_exploration_numba(self.explored, self.occ, x, y, ranges, scan_angles, self.W, self.H)

    def _build_origin_image(self, origin_pose, ranges, goals_to_visualize=None):
        cfg = self.config
        return build_origin_image_north_oriented(self.occ, self.explored, ranges, origin_pose, cfg['RESOLUTION'], cfg['WIN_PX'], cfg['RAD_CELLS'], cfg['FOV'], cfg['SHOW_HEAD'], goals_to_visualize=goals_to_visualize)

    def _build_local_image(self, current_pose, origin_pose, ranges, episode_goal=None):
        cfg = self.config
        goal_for_viz = episode_goal if cfg.get('VISUALIZE_EPISODE_GOAL') else None
        return build_local_image_fixed_origin_same_frame(self.occ, self.explored, ranges, current_pose, origin_pose, cfg['RESOLUTION'], cfg['WIN_PX'], cfg['RAD_CELLS'], cfg['FOV'], cfg['SHOW_HEAD'], cfg['VISUALIZE_FOV'], episode_goal=goal_for_viz)

    def _save_global_path_viz(self, boundary_mask, path, spawn, goal):
        cfg = self.config
        out_path = f'global_path_ep{self.collector.episode:02d}.png'
        save_global_path_viz(self.occ, boundary_mask, path, spawn, goal, cfg['RESOLUTION'], out_path)

    # ========================================
    # === Private Utility Methods ===
    # ========================================

    def _setup_directories(self): os.makedirs(self.config['SAVE_DIR'], exist_ok=True)
    def _load_pgm_map(self, pgm_path): return load_pgm_map(pgm_path)
    def _save_pgm_map(self, occ_data, out_path): save_pgm_map(occ_data, out_path)
    def _seal_by_adjacent(self, occ_data): return seal_by_adjacent(occ_data)
    def _find_spawn(self): return find_spawn(self.occ, self.config['RESOLUTION'], self.config['SPAWN_CLEARANCE'])
    def _bresenham(self, x0, y0, x1, y1): return bresenham(x0, y0, x1, y1)

    def _simulate_scan(self, pose):
        """
        Wrapper that calls the fast, GPU-accelerated scan simulation function.
        """
        cfg = self.config
        angles = np.linspace(-cfg['FOV'] / 2, cfg['FOV'] / 2, cfg['N_BEAMS']) + pose[2]

        # Call the new GPU host function
        return gpu_simulate_scan(self.occ, pose, angles, cfg['MAX_R'])

    def _find_boundary_obstacles(self): return find_boundary_obstacles(self.occ)
    def _sample_goal_util(self, *args, **kwargs): return sample_goal(*args, **kwargs)
    def _calculate_coverage(self, origin_pose): return calculate_coverage_fixed_origin(self.explored, origin_pose, self.config['WIN_PX'])
    def _normalize_angle(self, angle): return (angle + pi) % (2 * pi) - pi

# ===================================================================
# === Standalone Utility Functions (unchanged from original script) ===
# ===================================================================
# Note: These could also be private methods, but keeping them separate is also fine.

# Standalone Utility Functions
@njit(fastmath=True, cache=True)
def update_exploration_numba(explored, occ, x, y, ranges, scan_angles, W, H):
    """Ultra-fast Numba version"""
    for i in range(len(ranges)):
        r = ranges[i]
        ang = scan_angles[i]
        max_steps = int(r) + 1

        for step in range(max_steps):
            if step > r:
                break
            cx = x + step * cos(ang)
            cy = y + step * sin(ang)

            px = int(cx + 0.5)
            py = int(cy + 0.5)

            if 0 <= px < W and 0 <= py < H:
                explored[py, px] = True
                if occ[py, px] == 1:
                    break
            else:
                break

def save_pgm_map(occ, out_path):
    pgm = np.zeros_like(occ, dtype=np.uint8)
    pgm[occ == 0] = 254; pgm[occ == 1] = 0; pgm[occ == -1] = 205
    cv2.imwrite(out_path, pgm)

def load_pgm_map(pgm_path):
    raw = cv2.imread(pgm_path, cv2.IMREAD_UNCHANGED)
    if raw is None: raise FileNotFoundError(f"Cannot load map at {pgm_path}")
    grid = np.full(raw.shape, -1, dtype=np.int8)
    grid[raw == 254] = 0; grid[raw == 0] = 1
    return grid

def build_origin_image_north_oriented(occ, explored, ranges, origin_pose, resolution, window_px, robot_rad, fov,
                                      show_head=False, goals_to_visualize=None):
    """
    Builds the origin map with debugging visualizations added.
    """
    # base = np.full((window_px, window_px), 127, dtype=np.uint8)
    # center = window_px // 2
    # H, W = occ.shape
    # origin_x, origin_y, origin_theta = origin_pose
    # half = window_px // 2
    # Calculate buffer size for rotation
    buffer_size = int(window_px * np.sqrt(2) * 1.1)
    buffer_center = buffer_size // 2
    buffer_half = buffer_size // 2

    # Create buffer instead of final image
    base = np.full((buffer_size, buffer_size), 127, dtype=np.uint8)
    center = buffer_center  # Use buffer center
    H, W = occ.shape
    origin_x, origin_y, origin_theta = origin_pose
    half = buffer_half  # Use buffer half

    # Draw explored area
    for dy in range(-half, half):
        for dx in range(-half, half):
            gx, gy = int(origin_x + dx), int(origin_y + dy)
            ux, uy = center + dx, center + dy
            if 0 <= gx < W and 0 <= gy < H and explored[gy, gx]:
                base[uy, ux] = 255 if occ[gy, gx] == 0 else 0

    # Draw laser scan
    angles = np.linspace(-fov / 2, fov / 2, len(ranges)) + origin_theta
    for r, ang in zip(ranges, angles):
        ex, ey = origin_x + r * cos(ang), origin_y + r * sin(ang)
        for cx, cy in bresenham(origin_x, origin_y, ex, ey):
            dx_cell, dy_cell = cx - int(origin_x), cy - int(origin_y)
            ux, uy = center + dx_cell, center + dy_cell
            # if not (0 <= ux < window_px and 0 <= uy < window_px): break
            if not (0 <= ux < buffer_size and 0 <= uy < buffer_size): break
            if not (0 <= cy < H and 0 <= cx < W): break
            if occ[cy, cx] == 1:
                base[uy, ux] = 0; break
            else:
                base[uy, ux] = 255

    # Visualize all season goals (if provided)
    if goals_to_visualize:
        # print(f"--- DEBUG: Adding {len(goals_to_visualize)} goal markers... ---")
        for gx, gy in goals_to_visualize:
            dx, dy = gx - origin_x, gy - origin_y
            ux, uy = int(center + dx), int(center + dy)
            # if 0 <= ux < window_px and 0 <= uy < window_px:
            if 0 <= ux < buffer_size and 0 <= uy < buffer_size:
                # Draw a small, filled  for each goal
                # === FIX: Make dots larger and black for visibility on a grayscale map ===
                cv2.circle(base, (ux, uy), 2, 0, -1)  # Radius=5, Color=0 (black), Filled

    # Rotate image and draw robot
    # angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    # M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    # rotated = cv2.warpAffine(base, M, (window_px, window_px), flags=cv2.INTER_NEAREST, borderValue=127)
    # cv2.circle(rotated, (center, center), robot_rad, 200, thickness=-1)
    # if show_head:
    #     arrow_end_x, arrow_end_y = center, center - robot_rad - 3
    #     cv2.arrowedLine(rotated, (center, center), (arrow_end_x, arrow_end_y), color=1, thickness=2, tipLength=0.3)

    # return rotated

    # Rotate buffer (no cutoffs now!)
    angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    M = cv2.getRotationMatrix2D((buffer_center, buffer_center), angle_deg, 1.0)
    rotated_buffer = cv2.warpAffine(base, M, (buffer_size, buffer_size), flags=cv2.INTER_NEAREST, borderValue=127)

    # Crop back to original window size
    final_half = window_px // 2
    crop_start = buffer_center - final_half
    crop_end = buffer_center + final_half
    final_image = rotated_buffer[crop_start:crop_end, crop_start:crop_end]

    # Draw robot and arrow on final image
    final_center = window_px // 2
    cv2.circle(final_image, (final_center, final_center), robot_rad, 200, thickness=-1)
    if show_head:
        arrow_end_x, arrow_end_y = final_center, final_center - robot_rad - 3
        cv2.arrowedLine(final_image, (final_center, final_center), (arrow_end_x, arrow_end_y), color=1, thickness=2,
                        tipLength=0.3)

    return final_image

def build_virtual_pose_image_north_oriented(occ, explored, origin_pose, virtual_pose,
                                             resolution, window_px, robot_rad,
                                             show_head=True):
    """
    Renders a north-oriented local map image as if the robot were standing at
    `virtual_pose`, but using ONLY the already-explored map (no new scan rays).

    This is used for N-Polygon junction verification: we virtually move the
    camera to each polygon vertex without expanding the explored map or
    revealing any new information beyond what is already known.

    Args:
        occ:          occupancy grid (H x W), values: 0=free, 1=wall, -1=unknown
        explored:     boolean explored mask (H x W) — pass a FROZEN SNAPSHOT,
                      not self.simulator.explored directly
        origin_pose:  the robot's REAL current pose (x, y, theta) in pixels —
                      used ONLY for the north-orientation rotation angle
        virtual_pose: (x, y, theta) the hypothetical camera position in pixels
        resolution:   metres per pixel (unused here but kept for API consistency)
        window_px:    output image size in pixels (square)
        robot_rad:    robot marker radius in pixels
        show_head:    whether to draw the heading arrow

    Returns:
        uint8 numpy array of shape (window_px, window_px)
    """
    buffer_size = int(window_px * np.sqrt(2) * 1.1)
    buffer_center = buffer_size // 2
    buffer_half = buffer_size // 2

    base = np.full((buffer_size, buffer_size), 127, dtype=np.uint8)
    H, W = occ.shape

    # Camera is centred on virtual_pose, NOT origin_pose
    virt_x, virt_y, virt_theta = virtual_pose
    origin_x, origin_y, origin_theta = origin_pose   # only theta used for rotation

    # --- Draw explored area centred on virtual_pose ---
    for dy in range(-buffer_half, buffer_half):
        for dx in range(-buffer_half, buffer_half):
            gx = int(virt_x + dx)
            gy = int(virt_y + dy)
            ux = buffer_center + dx
            uy = buffer_center + dy
            if 0 <= gx < W and 0 <= gy < H and explored[gy, gx]:
                base[uy, ux] = 255 if occ[gy, gx] == 0 else 0
    # NOTE: No laser scan drawing — only the frozen explored map is used.

    # --- Rotate to north-orient using the REAL robot's heading ---
    # (same rotation as the main loop so the model sees the same orientation)
    angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    M = cv2.getRotationMatrix2D((buffer_center, buffer_center), angle_deg, 1.0)
    rotated_buffer = cv2.warpAffine(base, M, (buffer_size, buffer_size),
                                    flags=cv2.INTER_NEAREST, borderValue=127)

    # --- Crop to final window ---
    final_half = window_px // 2
    crop_start = buffer_center - final_half
    crop_end   = buffer_center + final_half
    final_image = rotated_buffer[crop_start:crop_end, crop_start:crop_end]

    # --- Draw robot marker at image centre ---
    final_center = window_px // 2
    cv2.circle(final_image, (final_center, final_center), robot_rad, 200, thickness=-1)
    if show_head:
        arrow_end_x = final_center
        arrow_end_y = final_center - robot_rad - 3
        cv2.arrowedLine(final_image, (final_center, final_center),
                        (arrow_end_x, arrow_end_y), color=1, thickness=2, tipLength=0.3)

    return final_image


def build_local_image_fixed_origin_same_frame(occ, explored, ranges, current_pose, origin_pose, resolution, window_px,
                                              robot_rad, fov, show_head, visualize_fov, episode_goal=None):
    """
    Builds the query map for each step.
    - CHANGED: Visualizes the current episode's goal as a small red dot.
    """
    """
        FIXED: Builds the query map without hard boundary cutoffs
        """
    base = np.full((window_px, window_px), 127, dtype=np.uint8)  # Gray = unknown
    center = window_px // 2
    H, W = occ.shape
    origin_x, origin_y, origin_theta = origin_pose
    current_x, current_y, current_theta = current_pose
    half = window_px // 2

    # FIXED: Extract window using proper array slicing to avoid cutoffs
    # Calculate global window bounds
    global_x_start = int(origin_x - half)
    global_y_start = int(origin_y - half)
    global_x_end = global_x_start + window_px
    global_y_end = global_y_start + window_px

    # Calculate overlap between window and map
    map_x_start = max(0, global_x_start)
    map_y_start = max(0, global_y_start)
    map_x_end = min(W, global_x_end)
    map_y_end = min(H, global_y_end)

    # Calculate corresponding local window coordinates
    local_x_start = map_x_start - global_x_start
    local_y_start = map_y_start - global_y_start
    local_x_end = local_x_start + (map_x_end - map_x_start)
    local_y_end = local_y_start + (map_y_end - map_y_start)

    # FIXED: Copy explored areas using array slicing (no hard cutoffs)
    if map_x_end > map_x_start and map_y_end > map_y_start:
        # Extract map regions
        map_explored = explored[map_y_start:map_y_end, map_x_start:map_x_end]
        map_occ = occ[map_y_start:map_y_end, map_x_start:map_x_end]

        # Set explored areas in local window
        local_region = base[local_y_start:local_y_end, local_x_start:local_x_end]
        local_region[map_explored] = np.where(map_occ[map_explored] == 0, 255, 0)

    # FIXED: Draw laser scan without hard cutoffs
    angles = np.linspace(-fov / 2, fov / 2, len(ranges)) + current_theta
    for r, ang in zip(ranges, angles):
        ex, ey = current_x + r * np.cos(ang), current_y + r * np.sin(ang)

        # Draw laser ray
        for cx, cy in bresenham(current_x, current_y, ex, ey):
            # Convert to local window coordinates
            local_x = int(cx - origin_x + center)
            local_y = int(cy - origin_y + center)

            # FIXED: Only break on map boundaries, not window boundaries
            if not (0 <= cy < H and 0 <= cx < W):
                break

            # Continue drawing if within local window
            if 0 <= local_x < window_px and 0 <= local_y < window_px:
                if occ[cy, cx] == 1:
                    base[local_y, local_x] = 0  # Obstacle
                    break
                else:
                    base[local_y, local_x] = 255  # Free space

    # Draw robot, FOV, and arrow (logic unchanged)
    robot_dx = current_x - origin_x
    robot_dy = current_y - origin_y
    robot_img_x_unrotated = int(center + robot_dx)
    robot_img_y_unrotated = int(center + robot_dy)
    if 0 <= robot_img_x_unrotated < window_px and 0 <= robot_img_y_unrotated < window_px:
        cv2.circle(base, (robot_img_x_unrotated, robot_img_y_unrotated), robot_rad, 200, thickness=-1)
        if visualize_fov:
            fov_line_len = robot_rad + 20
            start_angle, end_angle = current_theta - fov / 2, current_theta + fov / 2
            start_x = int(robot_img_x_unrotated + fov_line_len * cos(start_angle))
            start_y = int(robot_img_y_unrotated + fov_line_len * sin(start_angle))
            end_x = int(robot_img_x_unrotated + fov_line_len * cos(end_angle))
            end_y = int(robot_img_y_unrotated + fov_line_len * sin(end_angle))
            cv2.line(base, (robot_img_x_unrotated, robot_img_y_unrotated), (start_x, start_y), (0, 180, 0), 1)
            cv2.line(base, (robot_img_x_unrotated, robot_img_y_unrotated), (end_x, end_y), (0, 180, 0), 1)
        if show_head:
            arrow_len = robot_rad + 5
            end_x = int(robot_img_x_unrotated + arrow_len * cos(current_theta))
            end_y = int(robot_img_y_unrotated + arrow_len * sin(current_theta))
            cv2.arrowedLine(base, (robot_img_x_unrotated, robot_img_y_unrotated), (end_x, end_y), color=1, thickness=2,
                            tipLength=0.4)

    # Visualize the current episode's goal
    if episode_goal:
        gx, gy = episode_goal
        dx, dy = gx - origin_x, gy - origin_y
        ux, uy = int(center + dx), int(center + dy)
        if 0 <= ux < window_px and 0 <= uy < window_px:
            # Draw a small, filled red dot
            cv2.circle(base, (ux, uy), 2, (0, 0, 255), -1)

    # Rotate the final image
    angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    rotated = cv2.warpAffine(base, M, (window_px, window_px), flags=cv2.INTER_NEAREST, borderValue=127)

    return rotated


# (Other utility functions: seal_by_adjacent, bresenham, simulate_scan, calculate_coverage_fixed_origin, etc. remain here)
# ... (for brevity, I am omitting the other standalone functions that don't need changes, but they should be in this file)
def seal_by_adjacent(occ):
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    dilated_free = cv2.dilate((occ == 0).astype(np.uint8), kernel).astype(bool)
    unknown = (occ == -1)
    keep_unknown = unknown & dilated_free
    occ_fixed = occ.copy()
    occ_fixed[unknown & ~keep_unknown] = 1
    return occ_fixed

def bresenham(x0, y0, x1, y1):
    x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x += sx
        if e2 < dx: err += dx; y += sy


@cuda.jit
def raytrace_batch_kernel(batch_ranges, occ, poses_x, poses_y, poses_theta, all_angles, max_range, angles_per_pose):
    """
    Batch GPU kernel for multiple poses at once
    Each thread processes one ray across all poses
    """
    # Get unique thread ID
    ray_idx = cuda.grid(1)

    if ray_idx >= all_angles.shape[0]:
        return

    # Determine which pose this ray belongs to
    pose_idx = ray_idx // angles_per_pose
    local_ray_idx = ray_idx % angles_per_pose

    if pose_idx >= poses_x.shape[0]:
        return

    # Get pose data
    x = poses_x[pose_idx]
    y = poses_y[pose_idx]
    ang = all_angles[ray_idx]

    ex = x + max_range * math.cos(ang)
    ey = y + max_range * math.sin(ang)

    x0 = int(x); y0 = int(y)
    x1 = int(ex); y1 = int(ey)
    H = occ.shape[0]; W = occ.shape[1]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    px = x0; py = y0

    out = float(max_range)  # default

    while True:
        if not (0 <= px < W and 0 <= py < H):
            dxp = float(px) - x
            dyp = float(py) - y
            d = math.sqrt(dxp * dxp + dyp * dyp)
            if d < out:
                out = d
            break

        if occ[py, px] == 1:
            dxp = float(px) - x
            dyp = float(py) - y
            out = math.sqrt(dxp * dxp + dyp * dyp)
            break

        if px == x1 and py == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            px += sx
        if e2 < dx:
            err += dx
            py += sy

    batch_ranges[ray_idx] = out


def gpu_simulate_scan_batch(occ, poses_list, max_range, n_beams, fov):
    """
    Batch GPU raytracing for multiple poses

    Args:
        occ: Occupancy grid
        poses_list: List of (x, y, theta) poses
        max_range: Maximum laser range
        n_beams: Number of laser beams
        fov: Field of view

    Returns:
        List of range arrays, one per pose
    """
    n_poses = len(poses_list)
    angles_per_pose = n_beams
    total_rays = n_poses * angles_per_pose

    poses_x = np.array([p[0] for p in poses_list], dtype=np.float64)
    poses_y = np.array([p[1] for p in poses_list], dtype=np.float64)
    poses_theta = np.array([p[2] for p in poses_list], dtype=np.float64)

    # Prepare angle data for all poses
    all_angles = np.zeros(total_rays, dtype=np.float64)
    for i, (_, _, theta) in enumerate(poses_list):
        start = i * angles_per_pose
        end = start + angles_per_pose
        # avoid duplicate endpoint beam to keep spacing uniform
        beam_angles = np.linspace(-fov/2, fov/2, angles_per_pose, endpoint=False) + theta
        all_angles[start:end] = beam_angles

    # Move data to GPU - FIXED: Added missing d_poses_theta
    d_occ = cuda.to_device(occ)
    d_poses_x = cuda.to_device(poses_x)
    d_poses_y = cuda.to_device(poses_y)
    d_poses_theta = cuda.to_device(poses_theta)  # (not used in kernel but fine to keep)
    d_all_angles = cuda.to_device(all_angles)
    d_batch_ranges = cuda.device_array(total_rays, dtype=np.float64)

    # Configure GPU execution
    threads_per_block = 256
    blocks_per_grid = (total_rays + threads_per_block - 1) // threads_per_block

    # Launch batch kernel
    raytrace_batch_kernel[blocks_per_grid, threads_per_block](
        d_batch_ranges, d_occ, d_poses_x, d_poses_y, d_poses_theta,
        d_all_angles, max_range, angles_per_pose
    )

    # Copy results back to CPU
    batch_ranges = d_batch_ranges.copy_to_host()

    # Safety: sanitize
    bad = ~np.isfinite(batch_ranges) | (batch_ranges < 0)
    if np.any(bad):
        batch_ranges[bad] = float(max_range)

    # Split
    result_scans = []
    for i in range(n_poses):
        start = i * angles_per_pose
        end = start + angles_per_pose
        result_scans.append(batch_ranges[start:end])

    return result_scans


@cuda.jit
def raytrace_kernel(ranges, occ, x, y, angles, max_range):
    """
    Each thread traces ONE beam and MUST write a distance.
    Policy:
      - Hit obstacle in-bounds -> Euclidean distance to hit
      - Reached nominal endpoint with no hit -> max_range
      - Left map before hit -> distance traveled up to boundary (clamped to max_range)
    """
    # Get the unique ID for this thread, which corresponds to the beam index
    i = cuda.grid(1)

    # Ensure the thread index is not out of bounds
    if i >= ranges.shape[0]:
        return

    # Each thread computes one ray
    ang = angles[i]
    # Compute nominal endpoint
    ex = x + max_range * math.cos(ang)
    ey = y + max_range * math.sin(ang)

    # Inlined Bresenham's Algorithm
    x0, y0 = int(x), int(y)
    x1, y1 = int(ex), int(ey)
    H = occ.shape[0]
    W = occ.shape[1]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    px = x0
    py = y0

    # Default to max_range. We'll overwrite if we hit earlier or exit early.
    out = float(max_range)

    while True:
        if not (0 <= px < W and 0 <= py < H):
            # out-of-bounds: measure distance to last in-bounds cell center
            # (approx with current integer grid point)
            dxp = float(px) - x
            dyp = float(py) - y
            d = math.sqrt(dxp * dxp + dyp * dyp)
            if d < out:
                out = d
            break
        # NOTE: hypot is not available in cuda.jit, so we compute it manually
        if occ[py, px] == 1:
            dxp = float(px) - x
            dyp = float(py) - y
            out = math.sqrt(dxp * dxp + dyp * dyp)
            break

        if px == x1 and py == y1:
            # reached nominal endpoint w/o hit -> max_range
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            px += sx
        if e2 < dx:
            err += dx
            py += sy

    ranges[i] = out


def gpu_simulate_scan(occ, pose, angles, max_range):
    """
    This is the Host function. It runs on the CPU and manages the GPU kernel.
    """
    n_beams = len(angles)
    x, y, _ = pose

    # --- 1. Prepare data and move it to the GPU device ---
    # The map and angles only need to be sent once if they don't change
    d_occ = cuda.to_device(occ)
    d_angles = cuda.to_device(angles)
    # This array will store the results from the GPU
    d_ranges = cuda.device_array(n_beams, dtype=np.float64)

    # --- 2. Configure GPU execution parameters ---
    threads_per_block = 128
    blocks_per_grid = (n_beams + (threads_per_block - 1)) // threads_per_block

    # --- 3. Launch the Kernel on the GPU ---
    raytrace_kernel[blocks_per_grid, threads_per_block](d_ranges, d_occ, x, y, d_angles, max_range)

    # --- 4. Copy the results from the GPU back to the CPU ---
    ranges = d_ranges.copy_to_host()

    # --- Safety: sanitize anything weird (shouldn't trigger after kernel fix) ---
    # NaN/inf/neg -> max_range
    bad = ~np.isfinite(ranges) | (ranges < 0)
    if np.any(bad):
        ranges[bad] = float(max_range)

    return ranges

def calculate_coverage_fixed_origin(explored, origin_pose, window_px):
    origin_x, origin_y, _ = origin_pose
    ox = int(round(origin_x))
    oy = int(round(origin_y))
    half = window_px // 2

    # Check robot pixel is on the map
    H, W = explored.shape
    if not (0 <= ox < W and 0 <= oy < H):
        print(f"⚠️ Robot pixel ({ox},{oy}) out of bounds")
        return 0.0

    cx0, cy0 = ox - half, oy - half
    # Clip window to map edges but still count center
    x0, x1 = max(0, cx0), min(W, cx0 + window_px)
    y0, y1 = max(0, cy0), min(H, cy0 + window_px)

    origin_window = explored[y0:y1, x0:x1]
    # Check that window actually contains the robot cell
    win_cx = ox - x0
    win_cy = oy - y0
    if not (0 <= win_cx < origin_window.shape[1] and 0 <= win_cy < origin_window.shape[0]):
        print("⚠️ Robot cell outside clipped window")
        return 0.0
    if not origin_window[win_cy, win_cx]:
        print("⚠️ Robot cell not marked explored in window")

    # Uniform area for normalization: still use full window_px^2
    return np.count_nonzero(origin_window) / float(window_px * window_px)


# def calculate_coverage_fixed_origin(explored, origin_pose, window_px):
#     origin_x, origin_y, _ = origin_pose
#     half_win = window_px // 2
#     cx0, cy0 = int(origin_x - half_win), int(origin_y - half_win)
#     if cx0 < 0 or cy0 < 0 or cx0 + window_px > explored.shape[1] or cy0 + window_px > explored.shape[0]: return 0.0
#     origin_window = explored[cy0:cy0 + window_px, cx0:cx0 + window_px]
#     return np.count_nonzero(origin_window) / float(window_px * window_px)

def find_spawn(occ, resolution, spawn_clearance, margin=2):
    H, W = occ.shape
    clear_cells = int(spawn_clearance / resolution)
    kernel_size = 2 * (clear_cells + margin) + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    forbidden = cv2.dilate((occ == 1).astype(np.uint8), kernel)
    valid = np.logical_and(occ == 0, forbidden == 0)
    ys, xs = np.nonzero(valid)
    if len(xs) == 0: raise RuntimeError("No spawn location with required clearance!")
    idx = random.randrange(len(xs))
    return float(xs[idx]), float(ys[idx])

def sample_goal(explored, occ, resolution, spawn_clearance, current_pos, max_dist, margin=2, __return_all=False):
    clear_cells = int(spawn_clearance / resolution)
    k = 2 * (clear_cells + margin) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    forbidden = cv2.dilate((occ == 1).astype(np.uint8), kernel)
    valid_goal_mask = explored & (occ == 0) & (forbidden == 0)
    candidate_ys, candidate_xs = np.nonzero(valid_goal_mask)
    if candidate_xs.size == 0: raise RuntimeError("No valid goal locations found.")
    robot_x, robot_y = current_pos
    dist_sq = (candidate_xs - robot_x) ** 2 + (candidate_ys - robot_y) ** 2
    nearby_mask = dist_sq <= max_dist ** 2
    nearby_xs, nearby_ys = candidate_xs[nearby_mask], candidate_ys[nearby_mask]
    if __return_all:
        return nearby_xs, nearby_ys
    if nearby_xs.size == 0: raise RuntimeError(f"No valid goals found within {max_dist * resolution:.1f} meters.")
    idx = random.randrange(len(nearby_xs))
    return float(nearby_xs[idx]), float(nearby_ys[idx])

def find_boundary_obstacles(occ):
    free_mask = (occ == 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated_free_mask = cv2.dilate(free_mask, kernel, iterations=1)
    boundary_mask = (occ == 1) & (dilated_free_mask == 1)
    return boundary_mask

def save_global_path_viz(occ, boundary_mask, path, spawn, goal, resolution, out_path):
    H, W = occ.shape
    img = np.full((H, W, 3), (127, 127, 127), dtype=np.uint8)
    img[occ == 0] = (255, 255, 255); img[boundary_mask] = (0, 0, 0)
    if path:
        pts = np.array([[int(x), int(y)] for x, y in path], dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=(255, 0, 0), thickness=1)
    sx, sy = int(spawn[0]), int(spawn[1]); gx, gy = int(goal[0]), int(goal[1])
    cv2.circle(img, (sx, sy), 3, (0, 255, 0), thickness=-1)
    cv2.circle(img, (gx, gy), 3, (0, 0, 255), thickness=-1)
    scale = max(1, int(800 / max(H, W)))
    if scale > 1: img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out_path, img)
    print(f"Global path visualization saved to {out_path}")


if __name__ == '__main__':
    # Configuration is now a dictionary for cleaner passing
    config = {
        'MAP_PGM': 'house.pgm',
        'SAVE_DIR': 'local_scans',
        'RESOLUTION': 0.05,
        'ROBOT_R': 0.5,
        'FOV': 3 * pi / 2,
        'N_BEAMS': 360,
        'MAX_R': int(10.0 / 0.05),  # Max laser range in pixels
        'WIN_M': 20.0,
        'INTERVAL': 0.1,
        'LIN_VEL': 0.8,
        'ANG_VEL': pi,
        'DT': 0.05,
        'SPAWN_CLEARANCE': 0.75,
        'SHOW_HEAD': True,
        'VISUALIZE_FOV': True,
        'MAX_GOAL_DIST_M': 3.0,
        'RESET_AFTER_EPISODE': True  # <-- ADD THIS FLAG
    }
    # Derived configs
    config['RAD_CELLS'] = int(config['ROBOT_R'] / config['RESOLUTION'])
    config['WIN_PX'] = int(config['WIN_M'] / config['RESOLUTION'])

    # Create and run the simulator
    simulator = ExplorationSimulator(config)
    simulator.run_simulation(num_episodes=5)