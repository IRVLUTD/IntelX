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


def optimized_astar_path_planner(occ_map, start_px, goal_px, smooth_iterations=3,
                                 skip_postprocess=False):
    """
    Enhanced A* path planner with path smoothing for optimal, smooth paths.

    Args:
        occ_map (np.ndarray): The occupancy grid (0=free, 1=occupied).
        start_px (tuple): The (x, y) starting coordinates in pixels.
        goal_px (tuple): The (x, y) goal coordinates in pixels.
        smooth_iterations (int): Number of smoothing passes (default: 3)
        skip_postprocess (bool): return the RAW grid path straight from the
            search, skipping shortcutting, smoothing and subsampling. Those
            three exist to make a path pretty and drivable; a caller that only
            wants its LENGTH pays for all of them for nothing. shortcut_path is
            O(n^2) in waypoints with an is_line_free call per step, and
            is_line_free does ~18 array lookups per pixel of line -- which is
            where a 2000 px query spends tens of seconds.

    Returns:
        list or None: A list of (x, y) waypoints in pixels, or None if no path is found.
    """
    H, W = occ_map.shape
    start_node = (int(start_px[0]), int(start_px[1]))
    goal_node = (int(goal_px[0]), int(goal_px[1]))

    # Check if start or goal are inside an obstacle. A start that has slipped
    # onto a wall pixel plans from the nearest free cell instead, see
    # intelx.common.START_SNAP_PX.
    if not (0 <= start_node[0] < W and 0 <= start_node[1] < H) or occ_map[start_node[1], start_node[0]] == 1:
        from intelx.common import nearest_free_cell
        snapped = nearest_free_cell(occ_map, start_node)
        if snapped is None:
            print("A* Warning: Start node is in an obstacle or out of bounds.")
            return None
        start_node = snapped
    if not (0 <= goal_node[0] < W and 0 <= goal_node[1] < H) or occ_map[goal_node[1], goal_node[0]] == 1:
        print("A* Warning: Goal node is in an obstacle or out of bounds.")
        return None

    def heuristic(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])
    #
    # def is_line_free(p1, p2):
    #     """Check if straight line between two points is collision-free"""
    #     x0, y0 = int(p1[0]), int(p1[1])
    #     x1, y1 = int(p2[0]), int(p2[1])
    #
    #     # Bresenham's line algorithm
    #     dx = abs(x1 - x0)
    #     dy = abs(y1 - y0)
    #     sx = 1 if x0 < x1 else -1
    #     sy = 1 if y0 < y1 else -1
    #     err = dx - dy
    #
    #     x, y = x0, y0
    #     while True:
    #         if not (0 <= x < W and 0 <= y < H):
    #             return False
    #         if occ_map[y, x] == 1:
    #             return False
    #         if x == x1 and y == y1:
    #             break
    #         e2 = 2 * err
    #         if e2 > -dy:
    #             err -= dy
    #             x += sx
    #         if e2 < dx:
    #             err += dx
    #             y += sy
    #
    #     return True

    def is_line_free(p1, p2):
        """Check if a thick line (robot-width) between two points is collision-free."""
        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(p2[0]), float(p2[1])
        dist = hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            return True
        # Sample along the line at sub-pixel resolution
        n_samples = max(int(dist * 2), 2)
        for i in range(n_samples + 1):
            t = i / n_samples
            cx = x0 + t * (x1 - x0)
            cy = y0 + t * (y1 - y0)
            # Check a small neighbourhood (thick line = 1px radius)
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    nx, ny = int(cx + ox), int(cy + oy)
                    if not (0 <= nx < W and 0 <= ny < H):
                        return False
                    if occ_map[ny, nx] == 1:
                        return False
        return True

    # === STEP 1: A* Search with Better Cost Function ===
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # Cost weights: diagonal moves cost more to prefer straight lines
    move_costs = {
        (0, 1): 1.0, (0, -1): 1.0, (1, 0): 1.0, (-1, 0): 1.0,  # Cardinal
        (1, 1): 1.4, (1, -1): 1.4, (-1, 1): 1.4, (-1, -1): 1.4  # Diagonal
    }

    open_set = [(0, start_node)]
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
            path = list(reversed(path))
            break

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < W and 0 <= neighbor[1] < H):
                continue
            if occ_map[neighbor[1], neighbor[0]] == 1:
                continue

            # Use weighted cost
            new_cost = cost_so_far[current] + move_costs[(dx, dy)]

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current
    else:
        print("A* Warning: No path found.")
        return None

    # [FAST-ASTAR] Length-only callers stop here with the raw grid path.
    if skip_postprocess:
        return [(float(p[0]), float(p[1])) for p in path]

    # === STEP 2: Path Shortcutting (removes unnecessary waypoints) ===
    def shortcut_path(path):
        """Remove intermediate waypoints when straight line is possible"""
        if len(path) <= 2:
            return path

        shortcut = [path[0]]
        i = 0

        while i < len(path) - 1:
            # Try to skip ahead as far as possible
            j = len(path) - 1
            while j > i + 1:
                if is_line_free(path[i], path[j]):
                    shortcut.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # Can't skip, move to next point
                i += 1
                if i < len(path):
                    shortcut.append(path[i])

        return shortcut

    path = shortcut_path(path)

    # === STEP 3: Path Smoothing (gradient descent) ===
    def smooth_path(path, iterations=3):
        """Smooth path using weighted average"""
        if len(path) <= 2:
            return path

        smoothed = [list(p) for p in path]

        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                prev = smoothed[i - 1]
                curr = smoothed[i]
                next_pt = smoothed[i + 1]

                # Weighted average
                new_x = 0.25 * prev[0] + 0.5 * curr[0] + 0.25 * next_pt[0]
                new_y = 0.25 * prev[1] + 0.5 * curr[1] + 0.25 * next_pt[1]

                # Check if smoothed point is collision-free
                if is_line_free(prev, (new_x, new_y)) and is_line_free((new_x, new_y), next_pt):
                    if (0 <= int(new_x) < W and 0 <= int(new_y) < H and
                            occ_map[int(new_y), int(new_x)] == 0):
                        smoothed[i] = [new_x, new_y]

        return smoothed

    path = smooth_path(path, iterations=smooth_iterations)

    # === STEP 4: Subsample for cleaner visualization ===
    def subsample_path(path, max_points=50):
        """Keep only important waypoints"""
        if len(path) <= max_points:
            return path

        # Always keep start and end
        indices = [0]
        step = len(path) / (max_points - 1)
        for i in range(1, max_points - 1):
            indices.append(int(i * step))
        indices.append(len(path) - 1)

        return [path[i] for i in indices]

    path = subsample_path(path)

    # Convert to float tuples
    return [(float(p[0]), float(p[1])) for p in path]


def astar_path_planner(occ_map, start_px, goal_px):
    """
    A robust A* grid-based path planner.

    Args:
        occ_map (np.ndarray): The occupancy grid (0=free, 1=occupied).
        start_px (tuple): The (x, y) starting coordinates in pixels.
        goal_px (tuple): The (x, y) goal coordinates in pixels.

    Returns:
        list or None: A list of (x, y) waypoints in pixels, or None if no path is found.
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

    def __init__(self, config, initial_spawn_point=None, log_path='run_data.csv'):
        """
        Initializes the simulator.
        - FIXED to correctly handle NumPy arrays as spawn points.
        """
        self.config = config
        self._setup_directories()
        self.occ = self._load_pgm_map(config['MAP_PGM'])
        self.occ = self._seal_by_adjacent(self.occ)
        self._save_pgm_map(self.occ, 'modified_house.pgm')
        self.H, self.W = self.occ.shape
        boundary_mask = self._find_boundary_obstacles()
        boundary_ys, boundary_xs = np.nonzero(boundary_mask)
        self.obs_list_meters = [(x * config['RESOLUTION'], y * config['RESOLUTION'], config['ROBOT_R'])
                                for x, y in zip(boundary_xs, boundary_ys)]
        self.explored = np.zeros_like(self.occ, dtype=bool)

        # --- CORRECTED: Change the check to be 'is not None' ---
        if initial_spawn_point is not None:
            x, y = initial_spawn_point
            print(f"Using provided spawn point: ({x:.1f}, {y:.1f})")
        else:
            x, y = self._find_spawn()
        self.pose = (x, y, 0.0)

        self.frame_count = 0
        self.last_save_time = time.time()
        self._perform_initial_scan()

        self.collector = DataCollector(
            log_path=log_path,
            env_name=config['MAP_PGM'],
            origin=self.pose,
            origin_map_fname=os.path.join(config['SAVE_DIR'], "origin_scan.png"),
            pixel_to_meter=config['RESOLUTION']
        )

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
        MODIFIED: This method now returns the calculated A* path (in pixels) on success,
        or None on failure.
        """
        origin_pose = self.pose

        # This method no longer needs the collector, it's for pure simulation

        # Get both path formats from the planner
        path_meters, path_px = self._plan_path(goal)

        if path_meters is None:
            print("Planner failed to find a path for this episode.")
            return None  # Return None to signal failure

        path_cells = [(xm / self.config['RESOLUTION'], ym / self.config['RESOLUTION']) for xm, ym in path_meters]

        self._follow_path(path_cells, origin_pose, episode_goal=goal)

        # On success, return the pixel path for visualization
        return path_px

    # def run_episode_to_goal(self, goal):
    #     origin_pose = self.pose
    #     explored_at_origin = self.explored.copy()
    #     print(f"Episode origin: ({origin_pose[0]:.1f}, {origin_pose[1]:.1f}, {origin_pose[2]:.2f} rad)")
    #     self.collector.origin = origin_pose
    #     # self.collector.origin_map = origin_map_fname
    #     self.collector.origin_cov = self._calculate_coverage(origin_pose)
    #     self.collector.start_episode(goal=goal)
    #     path = self._plan_path(goal)
    #     if not path:
    #         print("RRT* failed to find a path for this episode.")
    #         return
    #     path_cells = [(xm / self.config['RESOLUTION'], ym / self.config['RESOLUTION']) for xm, ym in path]
    #     boundary_mask = self._find_boundary_obstacles()
    #     # self._save_global_path_viz(boundary_mask, path_cells, self.pose[:2], goal)
    #     self._follow_path(path_cells, origin_pose, episode_goal=goal)
    #     print(f"Reached goal for this episode.")
    #     if self.config.get('RESET_AFTER_EPISODE', False):
    #         print("Resetting simulation state to the episode's origin...")
    #         self.pose = origin_pose
    #         self.explored = explored_at_origin

    def _perform_initial_scan(self):
        print("Performing initial 360-degree scan...")
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
        """
        MODIFIED: This method now returns both the path in meters (for the robot to follow)
        and the path in pixels (for visualization).
        """
        start_coords = self.pose[:2]
        goal_coords = goal[:2]

        # A* works directly with pixel coordinates
        path_px = astar_path_planner(self.occ, start_coords, goal_coords)

        if path_px is None:
            return None, None  # Return failure for both

        # Convert to meters for the _follow_path method
        cfg = self.config
        path_meters = [(x * cfg['RESOLUTION'], y * cfg['RESOLUTION']) for x, y in path_px]

        return path_meters, path_px
    # --- 2. REPLACE THE _plan_path METHOD WITH THIS ---
    # def _plan_path(self, goal):
    #     """
    #     MODIFIED: This method now uses the robust A* planner instead of RRT*.
    #     """
    #     # A* works directly with pixel coordinates, so no need to convert to meters
    #     start_coords = self.pose[:2]
    #     goal_coords = goal[:2]  # Ensure we only take the (x, y) part of the goal
    #
    #     # Call the new A* planner
    #     path_px = astar_path_planner(self.occ, start_coords, goal_coords)
    #
    #     if path_px is None:
    #         return None  # Propagate the failure
    #
    #     # A* returns a path in pixel coordinates. The planner in the original RRTStar
    #     # expected a path in meters, so we convert it here for consistency.
    #     # This allows the rest of the simulator to work without changes.
    #     cfg = self.config
    #     path_meters = [(x * cfg['RESOLUTION'], y * cfg['RESOLUTION']) for x, y in path_px]
    #
    #     return path_meters


    def _follow_path(self, path_cells, origin_pose, episode_goal):
        """
        Executes robot motion based on a fixed step distance, not time.
        This is much faster for data generation.
        """
        cfg = self.config
        x, y, theta = self.pose

        for wx, wy in path_cells:
            dx, dy = wx - x, wy - y
            dist_m = hypot(dx, dy) * cfg['RESOLUTION']
            if dist_m < 1e-6:
                continue

            # --- Turn Phase (Unchanged) ---
            # The robot still turns in place to face the next waypoint.
            desired_heading = atan2(dy, dx)
            angle_diff = self._normalize_angle(desired_heading - theta)
            while abs(angle_diff) > 0.05:
                sign = 1 if angle_diff > 0 else -1
                # Use a fixed rotation step, DT is no longer needed here
                dθ = sign * min(cfg['ANG_VEL'] * 0.05, abs(angle_diff))
                theta = self._normalize_angle(theta + dθ)
                self._update_exploration((x, y, theta))
                angle_diff = self._normalize_angle(desired_heading - theta)
                # No time.sleep() is needed

            # --- Drive Phase (Simplified) ---
            # Calculate steps based on a fixed distance, not velocity and time
            steps = int(dist_m / cfg['STEP_SIZE_M'])

            if steps > 0:
                for step in range(steps):
                    # Linearly interpolate position for this step
                    frac = (step + 1) / steps
                    nx, ny = x + frac * dx, y + frac * dy
                    current_pose = (nx, ny, theta)

                    # At EVERY step, simulate a scan and log the data
                    ranges = self._simulate_scan(current_pose)
                    self._update_exploration(current_pose, ranges)
                    self._log_simulation_step(current_pose, origin_pose, ranges, episode_goal=episode_goal)
                    # No time.sleep() is needed

            # Snap to waypoint to correct for any drift
            x, y, theta = wx, wy, desired_heading

        self.pose = (x, y, theta)

    def _log_simulation_step(self, current_pose, origin_pose, ranges, episode_goal):
        cfg = self.config
        img_fname = f"{cfg['SAVE_DIR']}/frame_{self.frame_count:09d}.png"
        img = self._build_local_image(current_pose, origin_pose, ranges, episode_goal=episode_goal)
        cv2.imwrite(img_fname, img)
        coverage = self._calculate_coverage(origin_pose)
        self.collector.log_step(query=current_pose, query_map_fname=img_fname, coverage=coverage)
        self.frame_count += 1

    def _update_exploration(self, pose, ranges=None):
        cfg = self.config
        if ranges is None: ranges = self._simulate_scan(pose)
        x, y, theta = pose
        scan_angles = np.linspace(-cfg['FOV'] / 2, cfg['FOV'] / 2, cfg['N_BEAMS']) + theta
        for r, ang in zip(ranges, scan_angles):
            ex, ey = x + r * cos(ang), y + r * sin(ang)
            for cx, cy in self._bresenham(x, y, ex, ey):
                if 0 <= cx < self.W and 0 <= cy < self.H: self.explored[cy, cx] = True
                else: break

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
    # base = np.full((window_px, window_px), 127, dtype=np.uint8)
    # center = window_px // 2
    # H, W = occ.shape
    # origin_x, origin_y, origin_theta = origin_pose
    # current_x, current_y, current_theta = current_pose
    # half = window_px // 2
    #
    # # Draw explored area and laser scan (logic unchanged)
    # for dy in range(-half, half):
    #     for dx in range(-half, half):
    #         gx, gy = int(origin_x + dx), int(origin_y + dy)
    #         ux, uy = center + dx, center + dy
    #         if 0 <= gx < W and 0 <= gy < H:
    #             if explored[gy, gx]:
    #                 base[uy, ux] = 255 if occ[gy, gx] == 0 else 0
    #         # if 0 <= gx < W and 0 <= gy < H and explored[gy, gx]:
    #         #     base[uy, ux] = 255 if occ[gy, gx] == 0 else 0
    # angles = np.linspace(-fov / 2, fov / 2, len(ranges)) + current_theta
    # for r, ang in zip(ranges, angles):
    #     ex, ey = current_x + r * cos(ang), current_y + r * sin(ang)
    #     for cx, cy in bresenham(current_x, current_y, ex, ey):
    #         dx_cell, dy_cell = cx - int(origin_x), cy - int(origin_y)
    #         ux, uy = center + dx_cell, center + dy_cell
    #         if not (0 <= ux < window_px and 0 <= uy < window_px): break
    #         if not (0 <= cy < H and 0 <= cx < W): break
    #         if occ[cy, cx] == 1:
    #             base[uy, ux] = 0; break
    #         else:
    #             base[uy, ux] = 255

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
def raytrace_kernel(ranges, occ, x, y, angles, max_range):
    """
    This is the GPU Kernel. Each thread launched will execute this code
    for a single laser beam.
    """
    # Get the unique ID for this thread, which corresponds to the beam index
    i = cuda.grid(1)

    # Ensure the thread index is not out of bounds
    if i >= ranges.shape[0]:
        return

    # Each thread computes one ray
    ang = angles[i]
    ex = x + max_range * cos(ang)
    ey = y + max_range * sin(ang)

    # Inlined Bresenham's Algorithm
    x0, y0 = int(x), int(y)
    x1, y1 = int(ex), int(ey)
    H, W = occ.shape

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    px, py = x0, y0

    while True:
        if not (0 <= px < W and 0 <= py < H):
            break
        # NOTE: hypot is not available in cuda.jit, so we compute it manually
        if occ[py, px] == 1:
            ranges[i] = math.sqrt((px - x) ** 2 + (py - y) ** 2)
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

    return ranges

def calculate_coverage_fixed_origin(explored, origin_pose, window_px):
    origin_x, origin_y, _ = origin_pose
    half_win = window_px // 2
    cx0, cy0 = int(origin_x - half_win), int(origin_y - half_win)
    if cx0 < 0 or cy0 < 0 or cx0 + window_px > explored.shape[1] or cy0 + window_px > explored.shape[0]: return 0.0
    origin_window = explored[cy0:cy0 + window_px, cx0:cx0 + window_px]
    return np.count_nonzero(origin_window) / float(window_px * window_px)

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