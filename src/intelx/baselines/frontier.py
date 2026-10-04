import torch
import torch.nn as nn
# Remove or comment out: import segmentation_models_pytorch as smp
from intelx.models.unet import create_unet_resnet34, create_unet_resnet18
from scipy.ndimage import maximum_filter, gaussian_filter  # add alongside maximum_filter at top of file
from scipy.signal import find_peaks
import numpy as np
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
from math import pi, hypot
from collections import deque
from sklearn.cluster import KMeans
from tqdm import tqdm
import time
import threading
import queue
import copy
import enum
import heapq
from typing import Tuple

# Import the simulator and its helper functions
from intelx.sim.simulator import (
    ExplorationSimulator,
    build_origin_image_north_oriented,
    build_local_image_fixed_origin_same_frame
)

# Import A* planner from intelx.sim.simulator_astar
from intelx.sim.simulator_astar import optimized_astar_path_planner
from intelx.planners.global_priority import GlobalPriorityPlanner
from intelx.common import astar_with_clearance
from intelx.baselines.simple import free_cell_candidates

from intelx.paths import map_path, weights_path

# --- 1. Robot State Management ---
class RobotState(enum.Enum):
    IDLE = "idle"
    PLANNING = "planning"
    MOVING_TO_GOAL = "moving_to_goal"
    GOAL_REACHED = "goal_reached"
    PREDICTION_NEEDED = "prediction_needed"


def create_model_for_inference(checkpoint_path, device, model_type='resnet34'):
    """Load YOUR trained U-Net model"""
    print(f"Loading model: {model_type}")

    # Create model architecture
    if model_type == 'resnet18':
        model = create_unet_resnet18(in_channels=1, out_channels=1)
    elif model_type == 'resnet34':
        model = create_unet_resnet34(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle state dict (remove _orig_mod. prefix if present)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Clean state dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        clean_state_dict[new_key] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\u2713 Model loaded: {total_params:,} parameters")

    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_r2' in checkpoint:
        print(f"  Validation R\u00b2: {checkpoint['val_r2']:.4f}")

    return model


def _transform_image_goal_to_world(goal_px, origin_pose, image_size, resolution):
    """Transforms a goal from the image frame back to the world frame."""
    center_x, center_y = image_size // 2, image_size // 2
    origin_x, origin_y, origin_theta = origin_pose
    rot_angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    rot_angle_rad_inv = rot_angle_deg * np.pi / 180.0
    dx_img, dy_img = goal_px[0] - center_x, goal_px[1] - center_y
    cos_rot, sin_rot = np.cos(rot_angle_rad_inv), np.sin(rot_angle_rad_inv)
    dx_world = dx_img * cos_rot - dy_img * sin_rot
    dy_world = dx_img * sin_rot + dy_img * cos_rot
    return (origin_x + dx_world, origin_y + dy_world)


def safe_copy_pose(pose):
    """Safely copy pose regardless of type."""
    if hasattr(pose, 'copy'):
        return pose.copy()
    else:
        return np.array(pose)


def safe_copy_array(arr):
    """Safely copy array regardless of type."""
    if hasattr(arr, 'copy'):
        return arr.copy()
    else:
        return np.array(arr)


def check_line_of_sight(start_px, end_px, occ_map):
    """
    Checks if a straight line between two pixel coordinates is free of obstacles.
    Returns True if clear, False if blocked.
    """
    x0, y0 = start_px
    x1, y1 = end_px

    # Calculate distance
    dist = np.hypot(x1 - x0, y1 - y0)
    if dist < 1.0: return True

    # Sample points along the line
    num_samples = int(dist * 1.5)  # 1.5 samples per pixel
    if num_samples == 0: return True

    x_values = np.linspace(x0, x1, num_samples)
    y_values = np.linspace(y0, y1, num_samples)

    H, W = occ_map.shape

    for x, y in zip(x_values, y_values):
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            if occ_map[iy, ix] == 1:  # HIT WALL
                return False

    return True

# def astar_with_clearance(occ, start_px, goal_px, clearance_m, resolution):
#     """
#     A* with wall-distance cost penalty \u2014 no map inflation.
#     Instead of dilating walls (which seals narrow corridors), we add a
#     penalty to cells that are close to walls using a distance transform.
#     This naturally pushes the path to the centre of corridors.
#     Falls back gracefully if start/goal are in an obstacle.
#     """
#     from scipy.ndimage import distance_transform_edt
#
#     H, W = occ.shape
#     clearance_px = clearance_m / resolution  # e.g. 1.0/0.05 = 20px
#
#     # Pre-compute distance-to-nearest-wall for every free cell
#     # dist_map[y,x] = distance in pixels to the nearest wall pixel
#     dist_map = distance_transform_edt(occ == 0).astype(np.float32)
#
#     # Penalty weight: higher = stronger wall avoidance vs path length tradeoff
#     # At 0.5 this roughly doubles the cost of hugging a wall vs. centre path
#     WALL_PENALTY_WEIGHT = 3.0  # was 0.5
#
#     def wall_penalty(x, y):
#         d = dist_map[y, x]
#         if d >= clearance_px:
#             return 0.0
#         # Exponential: penalty explodes as d \u2192 0, smooth near clearance_px
#         return WALL_PENALTY_WEIGHT * np.exp(-d / (clearance_px * 0.3))
#
#     # Use the existing A* but with the penalty baked into the cost function
#     # We replicate the relevant parts here so we can inject the penalty
#     start_node = (int(start_px[0]), int(start_px[1]))
#     goal_node  = (int(goal_px[0]),  int(goal_px[1]))
#
#     # Validate start/goal on the original (unmodified) map
#     if not (0 <= start_node[0] < W and 0 <= start_node[1] < H) or occ[start_node[1], start_node[0]] == 1:
#         print("A* Clearance Warning: Start node is in an obstacle or out of bounds.")
#         return None
#     if not (0 <= goal_node[0] < W and 0 <= goal_node[1] < H) or occ[goal_node[1], goal_node[0]] == 1:
#         print("A* Clearance Warning: Goal node is in an obstacle or out of bounds.")
#         return None
#
#     def heuristic(a, b):
#         return hypot(a[0] - b[0], a[1] - b[1])
#
#     neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
#     move_costs = {
#         (0,1):1.0,(0,-1):1.0,(1,0):1.0,(-1,0):1.0,
#         (1,1):1.4,(1,-1):1.4,(-1,1):1.4,(-1,-1):1.4
#     }
#
#     open_set   = [(0, start_node)]
#     came_from  = {}
#     cost_so_far = {start_node: 0}
#
#     while open_set:
#         _, current = heapq.heappop(open_set)
#
#         if current == goal_node:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start_node)
#             path = list(reversed(path))
#             break
#
#         for dx, dy in neighbors:
#             nb = (current[0] + dx, current[1] + dy)
#             if not (0 <= nb[0] < W and 0 <= nb[1] < H):
#                 continue
#             if occ[nb[1], nb[0]] == 1:
#                 continue
#
#             # Base move cost + wall proximity penalty
#             new_cost = cost_so_far[current] + move_costs[(dx, dy)] + wall_penalty(nb[0], nb[1])
#
#             if nb not in cost_so_far or new_cost < cost_so_far[nb]:
#                 cost_so_far[nb] = new_cost
#                 heapq.heappush(open_set, (new_cost + heuristic(nb, goal_node), nb))
#                 came_from[nb] = current
#     else:
#         print("A* Warning: No path found.")
#         return None
#
#     # Shortcut path using original occ (no inflation \u2014 avoidance is in cost, not hard constraint)
#
#     def is_line_free(p1, p2):
#         x0, y0 = float(p1[0]), float(p1[1])
#         x1, y1 = float(p2[0]), float(p2[1])
#         dist = hypot(x1 - x0, y1 - y0)
#         if dist < 1e-6: return True
#         n = max(int(dist * 2), 2)
#         for i in range(n + 1):
#             t = i / n
#             cx, cy = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
#             if not (0 <= cx < W and 0 <= cy < H): return False
#             if occ[cy, cx] == 1: return False
#             if dist_map[cy, cx] < clearance_px:  # was clearance_px * 0.5
#                 return False
#         return True
#
#     # Path shortcutting (uses is_line_free with clearance check)
#     def shortcut_path(path):
#         if len(path) <= 2: return path
#         shortcut = [path[0]]
#         i = 0
#         while i < len(path) - 1:
#             j = len(path) - 1
#             while j > i + 1:
#                 if is_line_free(path[i], path[j]):
#                     shortcut.append(path[j])
#                     i = j
#                     break
#                 j -= 1
#             else:
#                 i += 1
#                 if i < len(path):
#                     shortcut.append(path[i])
#         return shortcut
#
#     path = shortcut_path(path)
#     return [(float(p[0]), float(p[1])) for p in path]

def _path_length_px(path):
    """Length of a waypoint path in pixels, the sum of its segments."""
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, np.float64)
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def extract_frontier_cells(occ_map, explored_map):
    """
    Finds all 'frontier cells' -> free cells that are adjacent to unexplored cells.
    """
    # Free cells we have seen
    free_cells = (occ_map == 0) & explored_map

    # Unexplored cells
    unexplored = ~explored_map

    # Dilate the unexplored mask by 1 pixel to find adjacencies
    kernel = np.ones((3, 3), dtype=bool)
    from scipy.ndimage import binary_dilation
    adjacent_to_unexplored = binary_dilation(unexplored, structure=kernel)

    # A frontier cell is free AND adjacent to an unexplored cell
    frontier_mask = free_cells & adjacent_to_unexplored

    return np.argwhere(frontier_mask)


def cluster_frontiers(frontier_cells, min_cluster_size=5, merge_dist=15.0):
    """
    Groups individual frontier cells into distinct frontiers (centroids).
    """
    if len(frontier_cells) == 0:
        return []

    # Simple Euclidean distance clustering
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=merge_dist, min_samples=min_cluster_size).fit(frontier_cells)

    labels = clustering.labels_
    unique_labels = set(labels)

    centroids = []
    for label in unique_labels:
        if label == -1: continue  # Noise

        cluster_points = frontier_cells[labels == label]
        cy, cx = cluster_points.mean(axis=0)
        size = len(cluster_points)

        centroids.append({
            'x': float(cx),
            'y': float(cy),
            'size': size,
            'cells': cluster_points
        })

    return centroids


# --- 3. Updated Goal-Based Autonomous Explorer with Continuous Map Updates ---
class UpdatedGoalBasedAutonomousExplorer:
    def __init__(self, simulator, model, config):
        self.simulator = simulator
        self.model = model
        self.config = config

        self.headless = getattr(config, 'HEADLESS', False)
        self.turbo_mode = getattr(config, 'TURBO_MODE', False)
        self.map_update_interval = getattr(config, 'MAP_UPDATE_INTERVAL', 5)
        self.save_interval = getattr(config, 'SAVE_INTERVAL', 1000)

        self.robot_radius_meters = self.config.sim.get('ROBOT_R')
        self.robot_radius_pixels = max(3, self.config.sim.get("RAD_CELLS"))

        self.local_interest_threshold = 0.35  # Threshold to switch to Global Hallucination
        self.nudge_distance_m = 4

        print(f"Robot radius: {self.robot_radius_meters}m ({self.robot_radius_pixels} pixels)")

        # Ensure pose is numpy array
        if not isinstance(self.simulator.pose, np.ndarray):
            self.simulator.pose = np.array(self.simulator.pose)

        # Transform for model inference
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Goal history
        self.goal_history = deque(maxlen=self.config.policy['goal_history_length'])

        # State machine for goal-based navigation
        self.robot_state = RobotState.PREDICTION_NEEDED
        self.current_goal = None
        self.current_path = None
        self.path_index = 0
        self.goal_tolerance = 3.0  # pixels

        # Add after the existing initialization code
        self.complete_trajectory = []  # Store entire robot trajectory
        self.trajectory_update_interval = 5  # Add point every 5 movements for efficiency

        # Thread communication
        self.prediction_request_queue = queue.Queue(maxsize=1)
        self.prediction_response_queue = queue.Queue(maxsize=1)
        self.state_queue = queue.Queue(maxsize=5)

        self.pose_history = []
        self.last_history_save_pos = None
        self.history_spacing_m = 0.5

        # --- NEW: Store junction nodes (>1 component areas) ---
        self.exploration_nodes = []

        # ADD THIS LINE: Persistent storage for potential junctions
        self.potential_nodes = []

        self.nodes_lock = threading.Lock()

        # Shared state variables
        try:
            initial_pose = safe_copy_pose(self.simulator.pose)
            initial_occ = safe_copy_array(self.simulator.occ)
            initial_explored = safe_copy_array(self.simulator.explored)
        except Exception as e:
            print(f"Warning: Issue copying initial state: {e}")
            initial_pose = np.array([0.0, 0.0, 0.0])
            initial_occ = np.zeros((100, 100), dtype=np.uint8)
            initial_explored = np.zeros((100, 100), dtype=bool)

        self.shared_state = {
            'pose': initial_pose,
            'exploration_pct': 0.0,
            'total_path_length': 0.0,
            'occ': initial_occ,
            'explored': initial_explored,
            'all_planned_paths': [],
            'current_goal': None,
            'robot_state': self.robot_state.value,
            'goal_count': 0
        }
        self.state_lock = threading.Lock()

        # Control flags
        self.running = threading.Event()
        self.running.set()
        self.exploration_complete = threading.Event()

        # Video recording setup
        try:
            # In __init__: don\u2019t create VideoWriter yet
            self.video_path = os.path.join(self.config.output_dir, "updated_goal_based_exploration.mp4")
            self.video_writer = None
            self.video_fps = 30.0

            print(f"Recording to {self.video_path}")
        except Exception as e:
            print(f"Warning: Video writer setup failed: {e}")
            self.video_writer = None

        # Threads
        self.prediction_thread = None
        self.simulation_thread = None
        self.visualization_thread = None

        # Latest data for visualization
        self.latest_prediction_data = None
        self.latest_state_data = None
        self.data_lock = threading.Lock()

        # Latest data
        self.last_frame = None  # ADD THIS LINE for video fallback
        self._last_saved_local_pred_id = -1

        # Global priority planner
        self.global_planner = GlobalPriorityPlanner(
            coverage_threshold=0.001,
            top_k_maxima=2,
            min_maxima_distance_m=2.0,
            local_maxima_radius_px=25,
            resolution_m_per_pixel=self.config.sim.get('RESOLUTION', 0.05)
        )

        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.goals_reached = 0

        print(f"Updated goal-based explorer initialized")

    def run_exploration_episode(self):
        """Main goal-based exploration loop."""
        print("Starting updated goal-based autonomous exploration...")

        try:
            # Start all threads
            self._start_threads()

            # Wait for threads to initialize
            time.sleep(2.0)
            print("All threads initialized, starting main loop...")

            # Main coordination loop with progress bar
            total_free_space = max(1, np.count_nonzero(self.simulator.occ == 0))
            pbar = tqdm(total=self.config.policy['exploration_target_pct'], desc="Updated Goal-Based Exploration")

            max_iterations = 100_000
            iteration_count = 0
            last_log_time = time.time()

            while (self.running.is_set() and
                   not self.exploration_complete.is_set() and
                   iteration_count < max_iterations):

                try:
                    iteration_count += 1
                    current_time = time.time()

                    # Debug logging every 10 seconds
                    if current_time - last_log_time > 10.0:
                        with self.state_lock:
                            current_state = self.shared_state.get('robot_state', 'unknown')
                            goal_count = self.shared_state.get('goal_count', 0)
                            exploration_pct = self.shared_state.get('exploration_pct', 0.0)

                        print(f"\n=== MAIN LOOP STATUS (Iteration {iteration_count}) ===")
                        print(f"Robot State: {current_state}")
                        print(f"Goals Reached: {goal_count}")
                        print(f"Exploration: {exploration_pct:.1f}%")
                        print("=" * 50)
                        last_log_time = current_time

                    # Update progress bar
                    try:
                        with self.state_lock:
                            current_exploration_pct = self.shared_state['exploration_pct']
                        pbar.update(max(0, min(100, current_exploration_pct - pbar.n)))

                        # Check exploration completion
                        if current_exploration_pct >= self.config.policy['exploration_target_pct']:
                            print(f"\nExploration target reached: {current_exploration_pct:.1f}%")
                            self.exploration_complete.set()
                            break
                    except Exception as e:
                        print(f"Warning: Progress update failed: {e}")

                    # Small delay
                    time.sleep(0.5)

                except KeyboardInterrupt:
                    print("\nExploration interrupted by user")
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    time.sleep(1.0)
                    continue

            if iteration_count >= max_iterations:
                print(f"\nSafety limit reached ({max_iterations} iterations)")

        except Exception as e:
            print(f"Critical error in exploration: {e}")

        finally:
            pbar.close()
            self._cleanup()
            self._print_exploration_summary()

    def _transform_world_to_image_goal(self, world_pos: Tuple[float, float],
                                       robot_pose: Tuple[float, float, float],
                                       image_size: int) -> Tuple[int, int]:
        """
        Transform world coordinates to image coordinates.
        Inverse of the transformation in global_priority_planner.
        """
        center_x, center_y = image_size // 2, image_size // 2
        robot_x, robot_y, robot_theta = robot_pose

        # Vector from robot to goal in world frame
        dx_world = world_pos[0] - robot_x
        dy_world = world_pos[1] - robot_y

        # Same rotation as planner (inverse)
        rot_angle_deg = 90.0 - (robot_theta * 180.0 / pi)
        rot_angle_rad_inv = rot_angle_deg * pi / 180.0

        cos_rot = np.cos(rot_angle_rad_inv)
        sin_rot = np.sin(rot_angle_rad_inv)

        # Inverse rotation transformation
        dx_img = dx_world * cos_rot + dy_world * sin_rot
        dy_img = -dx_world * sin_rot + dy_world * cos_rot

        goal_x = center_x + dx_img
        goal_y = center_y + dy_img

        return (int(goal_x), int(goal_y))

    def _start_threads(self):
        """Start all threads."""
        try:
            # Simple goal history priming
            self._simple_prime_goal_history()

            # Start threads
            try:
                self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
                self.prediction_thread.start()
                print("Prediction thread started")
            except Exception as e:
                print(f"Failed to start prediction thread: {e}")

            try:
                self.simulation_thread = threading.Thread(target=self._updated_simulation_worker, daemon=True)
                self.simulation_thread.start()
                print("Updated simulation thread started")
            except Exception as e:
                print(f"Failed to start simulation thread: {e}")

            try:
                self.visualization_thread = threading.Thread(target=self._visualization_worker, daemon=True)
                self.visualization_thread.start()
                print("Visualization thread started")
            except Exception as e:
                print(f"Failed to start visualization thread: {e}")

        except Exception as e:
            print(f"Error starting threads: {e}")

    def _simple_prime_goal_history(self):
        """Simple goal history initialization."""
        try:
            print("Simple goal history priming...")

            # Add some default goals to start with
            for i in range(self.goal_history.maxlen):
                default_goals = [(200, 200), (180, 220), (220, 180), (200, 180), (180, 200)]
                self.goal_history.append(default_goals[i % len(default_goals)])

            print(f"Simple goal history primed with {len(self.goal_history)} default goals.")

        except Exception as e:
            print(f"Error priming goal history: {e}")

    def _resolve_goal_px(self, world_xy, pose):
        """A world point -> an image goal THE SIMULATOR WILL ACCEPT, or None.

        The simulator tests int(_transform_image_goal_to_world(goal_px, ...)),
        not the point we hand it, and _transform_world_to_image_goal truncates.
        So reproduce the simulator's own check here and only return a goal that
        survives it. Candidates come from free_cell_candidates, nearest first,
        which also repairs a frontier centroid that landed inside a wall.
        """
        occ = self.simulator.occ
        H, W = occ.shape
        win = self.config.sim.get('WIN_PX', 400)
        res = self.config.sim.get('RESOLUTION', 0.05)
        snap_px = int(getattr(self.config, 'BASELINE_SNAP_PX', 40))

        for cx, cy in free_cell_candidates(occ, world_xy, snap_px):
            gpx = self._transform_world_to_image_goal((cx, cy), tuple(pose), win)
            bx, by = _transform_image_goal_to_world(gpx, tuple(pose), win, res)
            xi, yi = int(bx), int(by)
            if 0 <= xi < W and 0 <= yi < H and occ[yi, xi] == 0:
                return gpx
        return None

    #Temp
    def _prediction_worker(self):
        """
        BASELINE: Pure Geometry-Based Frontier Exploration.
        """
        print("BASELINE PLANNER: Frontier-based worker starting...")
        import cv2
        from scipy.signal import find_peaks

        prediction_count = 0

        # Blacklist to store unreachable/failed frontiers (x, y, timeout)
        blacklist = []
        blacklist_timeout = 45.0  # seconds

        # Anti-stuck tracking
        recent_poses = deque(maxlen=20)
        last_world_goal = None
        last_sample_t = 0.0                  # [STUCK] timer for pose sampling
        n_no_goal = 0                        # [NO-GOAL] consecutive empty passes
        frontier_centroids_for_viz = []      # [IDLE-COST] reused on idle passes

        while self.running.is_set():
            try:
                nav_request_pending = False
                try:
                    request_data = self.prediction_request_queue.get_nowait()
                    nav_request_pending = True
                    current_pose = request_data.get('pose')
                except queue.Empty:
                    current_pose = safe_copy_pose(self.simulator.pose)

                current_time = time.time()
                curr_px = (int(current_pose[0]), int(current_pose[1]))

                # --- STUCK DETECTION ---
                # [STUCK] Sampled on a 0.5 s timer. Appending once per loop meant
                # 20 samples spanned about a second, so the ordinary pause while
                # the sim planned the next goal read as "stuck" and blacklisted a
                # perfectly good frontier.
                if current_time - last_sample_t >= 0.5:
                    last_sample_t = current_time
                    recent_poses.append((current_pose[0], current_pose[1]))

                    if last_world_goal is not None and len(recent_poses) == recent_poses.maxlen:
                        start_p = recent_poses[0]
                        end_p = recent_poses[-1]
                        dist_moved = np.hypot(end_p[0] - start_p[0], end_p[1] - start_p[1])

                        if dist_moved < 3.0:
                            print("BASELINE: Robot appears stuck! Blacklisting current frontier.")
                            blacklist.append((last_world_goal[0], last_world_goal[1], current_time))
                            recent_poses.clear()
                            last_world_goal = None

                        # Clean up expired blacklist items
                blacklist = [b for b in blacklist if current_time - b[2] < blacklist_timeout]

                mode = "BASELINE_FRONTIER"
                chosen_goal_px = None
                # SUBMITTED_BEHAVIOUR restores what the submitted results used:
                # travel measured as the number of waypoints in the A* path, a
                # pass with no frontier inside the window ending the run when
                # it found no goal, and no end at all once frontiers ran out.
                submitted = bool(getattr(self.config, 'SUBMITTED_BEHAVIOUR', False))

                # --- 1. Extract and Cluster ALL Frontiers ---
                # [IDLE-COST] Only on a nav pass. This is DBSCAN over every
                # frontier cell of a multi-million-cell map, it ran ~20 times a
                # second, and its result is consumed only when a goal was
                # actually requested. Idle passes reuse the last centroids so the
                # visualiser still has something to draw.
                all_frontiers = []
                if nav_request_pending:
                    f_cells = extract_frontier_cells(self.simulator.occ, self.simulator.explored)
                    all_frontiers = cluster_frontiers(f_cells, min_cluster_size=self.config.MIN_FRONTIER_SIZE)
                    frontier_centroids_for_viz = [(f['x'], f['y']) for f in all_frontiers]

                if all_frontiers and nav_request_pending:
                    for f in all_frontiers:
                        f['dist_euclidean'] = np.hypot(f['x'] - curr_px[0], f['y'] - curr_px[1])

                    blacklist_dist_px = 2.0 / self.config.sim.get('RESOLUTION', 0.05)
                    valid_frontiers = []
                    for f in all_frontiers:
                        is_blacklisted = any(
                            np.hypot(f['x'] - b[0], f['y'] - b[1]) < blacklist_dist_px for b in blacklist)
                        if not is_blacklisted:
                            valid_frontiers.append(f)

                    # [BLACKLIST] If the blacklist matched every frontier,
                    # valid_frontiers is empty -> cands is empty -> the selection
                    # loop below never runs -> best_goal stays None, while mode
                    # has already been set to BASELINE_GLOBAL, which the
                    # simulator reads as "exploration over". The run is then
                    # declared exhausted with frontiers still on the map. Fall
                    # back to the full set instead.
                    if not valid_frontiers:
                        valid_frontiers = all_frontiers

                    # --- 2. Dynamic Window Filtering ---
                    local_radius_px = self.config.LOCAL_FRONTIER_WINDOW_M / self.config.sim.get('RESOLUTION', 0.05)
                    cands = [f for f in valid_frontiers if f['dist_euclidean'] <= local_radius_px]

                    if not cands:
                        # BASELINE_GLOBAL is also what tells the simulator the
                        # run is over, so it is set here only under the
                        # submitted behaviour; otherwise only on exhaustion.
                        if submitted:
                            mode = "BASELINE_GLOBAL"
                        cands = valid_frontiers

                    # --- 3. Greedy A* Selection ---
                    best_goal = None
                    best_score = float('inf')

                    cands.sort(key=lambda x: x['dist_euclidean'])

                    OBSTACLE_BUFFER_M = 1.0  # tunable \u2014 robot clearance from walls in metres

                    for cand in cands[:10]:
                        target_px = (int(cand['x']), int(cand['y']))

                        dx = curr_px[0] - target_px[0]
                        dy = curr_px[1] - target_px[1]
                        dist = max(1.0, np.hypot(dx, dy))

                        safe_target_px = (
                            int(target_px[0] + (dx / dist) * 3),
                            int(target_px[1] + (dy / dist) * 3)
                        )

                        H, W = self.simulator.occ.shape
                        safe_target_px = (
                            np.clip(safe_target_px[0], 0, W - 1),
                            np.clip(safe_target_px[1], 0, H - 1)
                        )

                        if self.simulator.occ[safe_target_px[1], safe_target_px[0]] == 0:
                            path = astar_with_clearance(
                                self.simulator.occ, curr_px, safe_target_px,
                                clearance_m=2.0,
                                resolution=self.config.sim.get('RESOLUTION', 0.05)
                            )
                        else:
                            path = astar_with_clearance(
                                self.simulator.occ, curr_px, safe_target_px,
                                clearance_m=2.0,
                                resolution=self.config.sim.get('RESOLUTION', 0.05)
                            )


                        if path is not None:
                            travel_distance = (len(path) if submitted
                                               else _path_length_px(path)
                                               ) * self.config.sim.get('RESOLUTION', 0.05)
                            frontier_size = cand['size']

                            score = (self.config.POTENTIAL_SCALE * travel_distance) - (
                                        self.config.GAIN_SCALE * frontier_size)

                            if score < best_score:
                                best_score = score
                                best_goal = safe_target_px if self.simulator.occ[safe_target_px[1], safe_target_px[
                                    0]] == 0 else target_px
                        else:
                            blacklist.append((cand['x'], cand['y'], current_time))

                    if best_goal is not None:
                        # [GOAL-REPAIR] see _resolve_goal_px: a centroid can sit
                        # in a wall, and the world -> image -> world round trip
                        # truncates, so the cell the SIMULATOR tests may differ
                        # from ours. An unrepaired goal is rejected with "Goal is
                        # inside obstacle!", the sim re-requests, this worker is
                        # deterministic, and the run livelocks with 0 m of path.
                        chosen_goal_px = self._resolve_goal_px(best_goal, current_pose)
                        if chosen_goal_px is not None:
                            last_world_goal = best_goal
                            n_no_goal = 0
                        else:
                            best_goal = None

                    if best_goal is None or chosen_goal_px is None:
                        # [NO-GOAL] Nothing usable. Answering None while mode is
                        # still BASELINE_FRONTIER makes the sim retry forever, so
                        # retry a bounded number of times (frontiers move as the
                        # map fills) and then say the run is exhausted.
                        last_world_goal = None
                        chosen_goal_px = None
                        n_no_goal += 1
                        if n_no_goal >= 5:
                            mode = "BASELINE_GLOBAL"
                            print("BASELINE: no reachable frontier in %d passes "
                                  "(%d frontier(s) found) -> exploration exhausted"
                                  % (n_no_goal, len(all_frontiers)))
                        else:
                            print("BASELINE: no usable goal (%d frontier(s), "
                                  "%d candidate(s)), retry %d/5"
                                  % (len(all_frontiers), len(cands), n_no_goal))

                elif nav_request_pending and not submitted:
                    # AutoX stops once the global frontier count falls below a
                    # threshold. With none left there is nothing to go to, and
                    # answering None as BASELINE_FRONTIER would re-request
                    # forever until the stall guard ended the run.
                    mode = "BASELINE_GLOBAL"
                    print("BASELINE: no frontier left -> exploration exhausted")

                # Inject into response (no active directions needed)
                local_components = {
                    'frontier_centroids': frontier_centroids_for_viz,
                }

                # 4. Response
                response = {
                    'heatmap': np.zeros((self.config.sim['WIN_PX'], self.config.sim['WIN_PX'])),
                    'goal_px': chosen_goal_px,
                    'selected_score': 1.0,
                    'final_heading': None,
                    'origin_img': np.zeros((self.config.sim['WIN_PX'], self.config.sim['WIN_PX'])),
                    'timestamp': time.time(),
                    'prediction_id': prediction_count,
                    'request_pose': current_pose,
                    'mode': mode,
                    'hallucinations': [],
                    'local_components': local_components,
                }

                with self.data_lock:
                    self.latest_prediction_data = response

                if nav_request_pending:
                    while not self.prediction_response_queue.empty():
                        try:
                            self.prediction_response_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.prediction_response_queue.put(response)

                prediction_count += 1
                time.sleep(0.05)

            except Exception as e:
                print(f"BASELINE PLANNER: Worker error: {e}")
                time.sleep(1)

    def _updated_simulation_worker(self):
        """UPDATED simulation worker with continuous map updates."""
        print("SIMULATION: Updated worker starting...")

        try:
            last_update_time = time.time()
            total_path_length = 0.0
            all_planned_paths = []
            movement_count = 0
            goal_count = 0
            map_update_count = 0

            # State machine variables
            current_state = RobotState.PREDICTION_NEEDED
            waiting_for_prediction_start = None

            while self.running.is_set() and not self.exploration_complete.is_set():
                try:
                    current_time = time.time()
                    # [FIXED-DT] dt used to come from the WALL CLOCK, so the
                    # distance moved per iteration depended on machine load:
                    # 0.75 px idle, up to 1.5 px busy. Scans are taken every 5
                    # movements, so under load the scan spacing doubled and the
                    # measured coverage curve moved with it. `ours` runs on a
                    # fixed dt, so the two were never comparable and every
                    # baseline number depended on N_WORKERS.
                    # This does NOT enable turbo: the 20 Hz sleep at the bottom of
                    # the loop is untouched, so speed is unchanged.
                    _fdt = getattr(self.config, 'FIXED_DT', None)
                    dt = (float(_fdt) if _fdt is not None
                          else min(0.1, current_time - last_update_time))
                    last_update_time = current_time

                    # State machine logic
                    if current_state == RobotState.PREDICTION_NEEDED:
                        print("SIMULATION: State = PREDICTION_NEEDED - requesting prediction")

                        # Request new prediction
                        try:
                            current_pose = safe_copy_pose(self.simulator.pose)
                            request_data = {
                                'pose': current_pose,
                                'timestamp': time.time()
                            }

                            # Clear old requests
                            while not self.prediction_request_queue.empty():
                                try:
                                    self.prediction_request_queue.get_nowait()
                                except queue.Empty:
                                    break

                            self.prediction_request_queue.put(request_data, timeout=1.0)
                            current_state = RobotState.PLANNING
                            waiting_for_prediction_start = time.time()
                            print("SIMULATION: Prediction requested, moving to PLANNING")
                        except Exception as e:
                            print(f"SIMULATION: Failed to request prediction: {e}")
                            time.sleep(1.0)

                    elif current_state == RobotState.PLANNING:
                        # Check for timeout
                        # if waiting_for_prediction_start and (time.time() - waiting_for_prediction_start) > 15.0:
                        #     print("SIMULATION: Prediction timeout (15s), restarting...")
                        _pred_timeout = float(getattr(self.config, 'PREDICTION_TIMEOUT_S', 15.0))
                        if waiting_for_prediction_start and (time.time() - waiting_for_prediction_start) > _pred_timeout:
                            print("SIMULATION: Prediction timeout (%.0fs), restarting..." % _pred_timeout)
                            current_state = RobotState.PREDICTION_NEEDED
                            continue

                        # Wait for prediction response
                        try:
                            prediction_response = self.prediction_response_queue.get(timeout=0.5)
                            print(f"SIMULATION: Received prediction response!")

                            # Check if a valid goal was actually found
                            goal_px = prediction_response.get('goal_px')
                            mode = prediction_response.get('mode', '')

                            if goal_px is None:
                                if mode == "BASELINE_GLOBAL":
                                    print("\nSIMULATION: All global frontiers unreachable! Exploration exhausted.")
                                    self.exploration_complete.set()
                                    break
                                else:
                                    print("SIMULATION: No valid goal found. Requesting new prediction.")
                                    current_state = RobotState.PREDICTION_NEEDED
                                    time.sleep(0.5)
                                    continue

                            request_pose = prediction_response['request_pose']

                            try:
                                world_goal = _transform_image_goal_to_world(
                                    goal_px,
                                    tuple(request_pose),
                                    self.config.sim.get('WIN_PX', 400),
                                    self.config.sim.get('RESOLUTION', 0.05)
                                )

                                print(f"SIMULATION: World goal: {world_goal}")

                                # Plan path to goal using A* planner
                                current_pos = safe_copy_pose(self.simulator.pose)[:2]
                                robot_px = (int(current_pos[0]), int(current_pos[1]))
                                goal_px_coords = (int(world_goal[0]), int(world_goal[1]))

                                print(f"SIMULATION: Planning path...")
                                print(f"  Robot position (px): {robot_px}")
                                print(f"  Goal position (px): {goal_px_coords}")
                                print(f"  Map size: {self.simulator.occ.shape}")

                                # Check if goal is valid BEFORE planning
                                H, W = self.simulator.occ.shape
                                if not (0 <= goal_px_coords[0] < W and 0 <= goal_px_coords[1] < H):
                                    print(f"SIMULATION: Goal out of bounds! Goal: {goal_px_coords}, Map: ({W}, {H})")
                                    current_state = RobotState.PREDICTION_NEEDED
                                    continue

                                if self.simulator.occ[goal_px_coords[1], goal_px_coords[0]] == 1:
                                    print(f"SIMULATION: Goal is inside obstacle!")
                                    current_state = RobotState.PREDICTION_NEEDED
                                    continue

                                print(f"SIMULATION: Goal is valid, calling A* planner...")

                                path_result = astar_with_clearance(
                                    self.simulator.occ, robot_px, goal_px_coords,
                                    clearance_m=2.0,
                                    resolution=self.config.sim.get('RESOLUTION', 0.05)
                                )

                                print(f"SIMULATION: Path result type: {type(path_result)}")

                                if path_result is None:
                                    print("SIMULATION: Inflated A* failed, retrying with original map...")
                                    path_result = optimized_astar_path_planner(self.simulator.occ, robot_px,
                                                                               goal_px_coords)
                                if path_result is not None:
                                    print(f"SIMULATION: Path result length: {len(path_result)}")
                                    print(
                                        f"SIMULATION: First few waypoints: {path_result[:3] if len(path_result) > 3 else path_result}")

                                if path_result is not None and len(path_result) > 0:
                                    # A* returns list of pixel coordinates directly
                                    path_px = path_result

                                    print(f"SIMULATION: A* returned path with {len(path_px)} waypoints")

                                    # Store PIXEL path for visualization
                                    self.current_path = [[float(p[0]), float(p[1])] for p in path_px]
                                    self.current_goal = world_goal
                                    self.path_index = 0
                                    all_planned_paths.append(copy.deepcopy(self.current_path))

                                    # Store prediction data for visualization
                                    with self.data_lock:
                                        self.latest_prediction_data = prediction_response

                                    current_state = RobotState.MOVING_TO_GOAL
                                    goal_count += 1

                                    print(f"SIMULATION: SUCCESS! Path planned with {len(self.current_path)} waypoints")
                                    print(f"SIMULATION: Goal #{goal_count} - Moving to MOVING_TO_GOAL state")
                                else:
                                    print("SIMULATION: Path planning failed - A* returned None or empty path")
                                    current_state = RobotState.PREDICTION_NEEDED

                            except Exception as e:
                                print(f"SIMULATION: Goal transformation/planning error: {e}")
                                import traceback
                                traceback.print_exc()
                                current_state = RobotState.PREDICTION_NEEDED

                        except queue.Empty:
                            pass  # Still waiting
                        except Exception as e:
                            print(f"SIMULATION: Prediction response error: {e}")

                    elif current_state == RobotState.MOVING_TO_GOAL:
                        # Execute movement along A* path WITH CONTINUOUS MAP UPDATES
                        if self.current_path and self.path_index < len(self.current_path):
                            try:
                                old_pose = safe_copy_pose(self.simulator.pose)

                                # Move smoothly toward next waypoint
                                target = np.array(self.current_path[self.path_index])
                                current_pos = old_pose[:2]

                                # Calculate movement
                                diff = target - current_pos
                                distance = np.linalg.norm(diff)

                                if distance < self.goal_tolerance:
                                    self.path_index += 1
                                    if self.path_index >= len(self.current_path):
                                        print(f"SIMULATION: GOAL #{goal_count} REACHED!")
                                        current_state = RobotState.GOAL_REACHED
                                    else:
                                        if self.path_index % 10 == 0:
                                            remaining = len(self.current_path) - self.path_index
                                            print(f"SIMULATION: Waypoint {self.path_index}, {remaining} remaining")
                                else:
                                    # Move toward waypoint
                                    max_speed = self.config.sim.get('STEP_SIZE_M', 0.1) / self.config.sim.get('DT', 0.1)
                                    step_size = min(max_speed * dt * 15, distance)

                                    if step_size > 0:
                                        direction = diff / distance
                                        new_pos = current_pos + direction * step_size

                                        # Update simulator pose
                                        new_theta = np.arctan2(direction[1], direction[0])
                                        self.simulator.pose = np.array([new_pos[0], new_pos[1], new_theta])

                                        # --- NEW: Record Pose History for Hallucination Worker ---
                                        # This creates the "Memory" required for the global planner to work
                                        curr_x, curr_y = new_pos[0], new_pos[1]
                                        save_memory = False

                                        if self.last_history_save_pos is None:
                                            save_memory = True
                                        else:
                                            lx, ly = self.last_history_save_pos
                                            # Calculate distance moved since last memory save
                                            dist_moved_mem = hypot(curr_x - lx, curr_y - ly) * self.config.sim.get(
                                                'RESOLUTION', 0.05)
                                            # Save every 0.5m (same as history_spacing_m)
                                            if dist_moved_mem > 0.5:
                                                save_memory = True

                                        if save_memory:
                                            self.pose_history.append(self.simulator.pose.copy())
                                            self.last_history_save_pos = (curr_x, curr_y)
                                            if len(self.pose_history) > 1000:
                                                self.pose_history.pop(0)
                                        # ---------------------------------------------------------

                                        if movement_count % self.trajectory_update_interval == 0:
                                            self.complete_trajectory.append([new_pos[0], new_pos[1]])

                                        # CRITICAL: Update exploration map at EVERY step (Optimized to every 5th count)
                                        if movement_count % 5 == 0:
                                            try:
                                                scan_result = self.simulator._simulate_scan(self.simulator.pose)
                                                self.simulator._update_exploration(self.simulator.pose, scan_result)
                                                map_update_count += 1
                                            except Exception as e:
                                                pass

                                        # Track path length
                                        distance_traveled = hypot(
                                            self.simulator.pose[0] - old_pose[0],
                                            self.simulator.pose[1] - old_pose[1]
                                        ) * self.config.sim.get('RESOLUTION', 0.05)
                                        total_path_length += distance_traveled
                                        movement_count += 1

                                        # Log movement progress
                                        if movement_count % 50 == 0:
                                            print(f"SIMULATION: Movement #{movement_count}, "
                                                  f"Map updates: {map_update_count}, "
                                                  f"Distance to waypoint: {distance:.1f}px")

                            except Exception as e:
                                print(f"SIMULATION: Movement execution failed: {e}")
                        else:
                            print("SIMULATION: No valid path, requesting new prediction")
                            current_state = RobotState.PREDICTION_NEEDED

                    elif current_state == RobotState.GOAL_REACHED:
                        print("SIMULATION: State = GOAL_REACHED - cleaning up")
                        self.current_goal = None
                        self.current_path = None
                        self.path_index = 0
                        current_state = RobotState.PREDICTION_NEEDED
                        print("SIMULATION: Ready for next goal!")

                    # Calculate exploration percentage USING UPDATED MAP
                    try:
                        total_free_space = max(1, np.count_nonzero(self.simulator.occ == 0))
                        explored_free_space = np.count_nonzero(
                            self.simulator.explored & (self.simulator.occ == 0)
                        )
                        exploration_pct = min(100.0, (explored_free_space / total_free_space) * 100)

                        # Log exploration progress occasionally
                        if movement_count > 0 and movement_count % 100 == 0:
                            print(f"SIMULATION: Exploration progress: {exploration_pct:.1f}% "
                                  f"({explored_free_space}/{total_free_space} cells)")

                    except Exception as e:
                        exploration_pct = 0.0

                    # Update shared state with CURRENT map data
                    try:
                        with self.state_lock:
                            self.shared_state['pose'] = safe_copy_pose(self.simulator.pose)
                            self.shared_state['exploration_pct'] = exploration_pct
                            self.shared_state['total_path_length'] = total_path_length
                            # CRITICAL: Copy the CURRENT map state
                            self.shared_state['occ'] = safe_copy_array(self.simulator.occ)
                            self.shared_state['explored'] = safe_copy_array(self.simulator.explored)
                            # Store only last 10 paths to avoid memory issues
                            self.shared_state['all_planned_paths'] = copy.deepcopy(
                                all_planned_paths[-10:]) if all_planned_paths else []

                            self.shared_state['current_goal'] = self.current_goal
                            self.shared_state['robot_state'] = current_state.value
                            self.shared_state['goal_count'] = goal_count
                            # In the shared state update:
                            self.shared_state['complete_trajectory'] = copy.deepcopy(
                                self.complete_trajectory[-500:])  # Last 500 points
                            self.shared_state['current_path'] = copy.deepcopy(
                                self.current_path) if self.current_path else None

                    except Exception as e:
                        if movement_count % 100 == 0:
                            print(f"SIMULATION: State update failed: {e}")

                    # Check completion
                    if exploration_pct >= self.config.policy['exploration_target_pct']:
                        print(f"SIMULATION: Exploration complete! {exploration_pct:.1f}%")
                        self.exploration_complete.set()

                    # Send state update for visualization with UPDATED map
                    try:
                        # Clear old state updates
                        while not self.state_queue.empty():
                            try:
                                self.state_queue.get_nowait()
                            except queue.Empty:
                                break

                        with self.nodes_lock:
                            safe_nodes = copy.deepcopy(self.exploration_nodes)

                        state_data = {
                            'pose': safe_copy_pose(self.simulator.pose),
                            'exploration_pct': exploration_pct,
                            'occ': safe_copy_array(self.simulator.occ),
                            'explored': safe_copy_array(self.simulator.explored),
                            'complete_trajectory': copy.deepcopy(self.complete_trajectory[-500:]),
                            'current_path': copy.deepcopy(self.current_path) if self.current_path else None,
                            'memory_points': copy.deepcopy(self.pose_history),
                            'exploration_nodes': safe_nodes,  # FIXED
                            'total_path_length': total_path_length,
                            'current_goal': self.current_goal,
                            'robot_state': current_state.value,
                            'goal_count': goal_count,
                            'movement_count': movement_count,
                            'map_update_count': map_update_count
                        }

                        self.state_queue.put(state_data, timeout=0.01)
                    except:
                        pass

                    # Maintain simulation rate
                    time.sleep(max(0.01, 0.05 - (time.time() - current_time)))

                except Exception as e:
                    print(f"SIMULATION: Worker error: {e}")
                    time.sleep(0.001)

        except Exception as e:
            print(f"SIMULATION: Critical error: {e}")

        print(f"SIMULATION: Worker stopping")
        print(f"SIMULATION: Goals: {goal_count}, Movements: {movement_count}, Map updates: {map_update_count}")

    def _visualization_worker(self):
        """
        Headless-Capable Worker.
        If HEADLESS=True: Saves PNG snapshots instead of showing a window.
        """
        print(f"VISUALIZATION: Worker starting (Headless: {self.headless})...")

        # Create output directory for snapshots
        snapshot_dir = os.path.join(self.config.output_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        prediction_data = None

        try:
            if not self.headless:
                cv2.namedWindow("Updated Goal-Based Exploration", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Updated Goal-Based Exploration", 1800, 1000)

            while self.running.is_set() and not self.exploration_complete.is_set():
                try:
                    try:
                        state_data = self.state_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    prediction_data = None
                    with self.data_lock:
                        prediction_data = self.latest_prediction_data

                    if state_data:
                        movement_count = state_data.get('movement_count', 0)
                        hallucinations = prediction_data.get('hallucinations', []) if prediction_data else []

                        # --- HEADLESS MODE ---
                        if self.headless:
                            # Save snapshot if interval met OR finished
                            if self.save_interval > 0 and movement_count > 0 and movement_count % self.save_interval == 0:
                                _, full_map = self._create_updated_global_view(state_data, prediction_data, hallucinations)
                                if full_map is not None:
                                    filename = f"{snapshot_dir}/step_{movement_count:06d}.png"
                                    cv2.imwrite(filename, full_map)
                                    print(f"VISUALIZATION: Saved snapshot {filename}")

                        # --- GUI MODE ---
                        else:
                            # In _visualization_worker, inside the GUI MODE where `frame` is computed:
                            frame = self._create_updated_visualization_frame(state_data, prediction_data)
                            if frame is not None:

                                cv2.imshow("Updated Goal-Based Exploration", frame)
                                if self.video_writer is None:
                                    h, w = frame.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                    self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w, h))
                                    if not self.video_writer.isOpened():
                                        print(f"ERROR: VideoWriter failed to open: {self.video_path}")
                                        self.video_writer = None

                                if self.video_writer is not None:
                                    self.video_writer.write(frame)

                                self.last_frame = frame.copy()

                            key = cv2.waitKey(1) & 0xFF
                            if key == 27:
                                self.running.clear()
                                break

                except Exception as e:
                    print(f"VISUALIZATION: Worker error: {e}")

        except Exception as e:
            print(f"VISUALIZATION: Critical error: {e}")

        finally:
            # SAVE FINAL FRAME
            print("VISUALIZATION: Saving final results...")
            try:
                with self.state_lock:
                    final_state = self.shared_state.copy()
                _, full_map = self._create_updated_global_view(final_state, prediction_data,[])
                if full_map is not None:
                    cv2.imwrite(os.path.join(self.config.output_dir, "final_trajectory.png"), full_map)
            except Exception as e:
                print(f"Could not save final image: {e}")

            if not self.headless:
                cv2.destroyAllWindows()

    # REPLACE _create_updated_visualization_frame
    def _create_updated_visualization_frame(self, state_data, prediction_data):
        """
        Creates Dashboard: [Zoom] [Top 3 Frontiers] [Global Map]
        """
        try:
            mode = "Initializing"
            hallucinations = []
            selected_local_score = 0.0

            if prediction_data:
                mode = prediction_data.get('mode', 'LOCAL')
                hallucinations = prediction_data.get('hallucinations', [])
                selected_local_score = prediction_data.get('selected_score', 0.0)

            views = self._create_updated_global_view(state_data, prediction_data, hallucinations)
            if views is None: return None
            zoomed_view, full_map_view = views

            h_target = zoomed_view.shape[0]  # ~650
            inspector_w = 320
            inspector_panel = np.full((h_target, inspector_w, 3), 40, dtype=np.uint8)

            y_offset = 15
            cv2.putText(inspector_panel, f"MODE: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 40

            if mode == "GLOBAL_HALLUCINATION" and hallucinations:
                # SHOW TOP 3
                for hal in hallucinations[:3]:
                    if y_offset + 160 > h_target: break

                    img_bgr = cv2.cvtColor(hal['img'], cv2.COLOR_GRAY2BGR) if len(hal['img'].shape) == 2 else hal[
                        'img'].copy()
                    hm = hal['heatmap']
                    hm_norm = ((hm - hm.min()) / (hm.max() - hm.min() + 1e-6) * 255).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_VIRIDIS)
                    blended = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

                    thumb_size = 150
                    thumb = cv2.resize(blended, (thumb_size, thumb_size))
                    color = hal['color']
                    cv2.rectangle(thumb, (0, 0), (thumb_size - 1, thumb_size - 1), color, 4)

                    # Red Crosshair on Max Point
                    lx, ly = hal['local_max_pt']
                    scale = thumb_size / hal['img'].shape[1]
                    cv2.drawMarker(thumb, (int(lx * scale), int(ly * scale)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

                    inspector_panel[y_offset:y_offset + thumb_size, 10:10 + thumb_size] = thumb
                    cv2.putText(inspector_panel, f"ID:{hal['id']}", (170, y_offset + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(inspector_panel, f"Sc:{hal['score']:.3f}", (170, y_offset + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += thumb_size + 15

            elif mode == "LOCAL" and prediction_data:
                origin_img = prediction_data.get('origin_img')
                heatmap = prediction_data.get('heatmap')
                goal_px = prediction_data.get('goal_px')
                if origin_img is not None:
                    img_bgr = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR) if len(
                        origin_img.shape) == 2 else origin_img.copy()
                    hm_norm = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) * 255).astype(
                        np.uint8)
                    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_VIRIDIS)
                    blended = cv2.addWeighted(img_bgr, 0.6, hm_color, 0.4, 0)

                    # --- NEW: Draw Detected Local Components ---
                    comp = prediction_data.get("local_components", None)
                    if comp is not None:
                        label_map = comp.get("label_map", None)
                        keep_ids = comp.get("keep_ids", [])
                        centroids = comp.get("centroids", {})

                        palette = [
                            (0, 255, 255), (255, 255, 0), (255, 0, 255),
                            (0, 165, 255), (0, 255, 0), (255, 0, 0),
                        ]
                        alpha = 0.4
                        robot_cx, robot_cy = blended.shape[1] // 2, blended.shape[0] // 2

                        for k, cc_id in enumerate(keep_ids[:6]):
                            color = palette[k % len(palette)]

                            # Only draw if label_map matches image dimensions
                            if label_map is not None and label_map.shape[:2] == blended.shape[:2]:
                                m = (label_map == cc_id)
                                # Semi-transparent filled mask overlay
                                blended[m] = (blended[m].astype(np.float32) * (1 - alpha) +
                                              np.array(color, dtype=np.float32) * alpha).astype(np.uint8)

                                # Contour outline
                                cmask = (m.astype(np.uint8) * 255)
                                contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(blended, contours, -1, color, 2)

                            # Weighted centroid: filled dot + ring + arrow
                            if cc_id in centroids:
                                cx, cy = int(round(centroids[cc_id][0])), int(round(centroids[cc_id][1]))
                                cv2.circle(blended, (cx, cy), 6, (255, 255, 255), -1)  # white fill
                                cv2.circle(blended, (cx, cy), 6, color, 2)  # colored ring
                                cv2.putText(blended, str(k + 1), (cx + 8, cy + 5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                                # Arrow from robot to centroid
                                cv2.arrowedLine(blended, (robot_cx, robot_cy), (cx, cy), color, 2, tipLength=0.25)

                    if goal_px:
                        cv2.drawMarker(blended, (int(goal_px[0]), int(goal_px[1])), (0, 0, 255), cv2.MARKER_CROSS, 25,
                                       2)

                    thumb_size = 280
                    thumb = cv2.resize(blended, (thumb_size, thumb_size))
                    inspector_panel[y_offset:y_offset + thumb_size, 20:20 + thumb_size] = thumb

                    cv2.putText(inspector_panel, "LOCAL PLANNER", (20, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    if goal_px:
                        cv2.putText(inspector_panel, f"Score: {selected_local_score:.6f}",
                                    (20, y_offset + thumb_size + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # --- NEW: Component Info Text ---
                    if comp is not None:
                        n_comp = len(comp.get('keep_ids', []))
                        angles = comp.get('peak_angles_deg', [])
                        cv2.putText(inspector_panel, f"Components: {n_comp}", (20, y_offset + thumb_size + 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        if angles:
                            cv2.putText(inspector_panel, f"Angles: {angles}", (20, y_offset + thumb_size + 85),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            scale = h_target / full_map_view.shape[0]
            w_new = int(full_map_view.shape[1] * scale)
            full_map_resized = cv2.resize(full_map_view, (w_new, h_target))
            return np.hstack((zoomed_view, inspector_panel, full_map_resized))

        except Exception as e:
            print(f"VISUALIZATION: Error {e}")
            return None

    # REPLACE _create_updated_global_view
    def _create_updated_global_view(self, state_data, prediction_data=None, hallucinations=[]):
        """
        Draws Global Map with:
        1. Robot Memory Trail (Gray Dots).
        2. Hallucination "Ghosts" (Cyan) connecting to Frontiers (Colored).
        """
        try:
            occ = state_data.get('occ')
            explored = state_data.get('explored')
            pose = state_data.get('pose')
            complete_trajectory = state_data.get('complete_trajectory', [])

            # --- NEW: Retrieve Memory Points ---
            memory_points = state_data.get('memory_points', [])
            # -----------------------------------

            if occ is None or explored is None or pose is None: return None

            H, W = occ.shape

            # 1. Base Map
            viz_map = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)
            viz_map[occ == 1] = (0, 0, 0)
            viz_map[explored] = (200, 200, 255)

            # 2. Draw Dense Path (Orange Line)
            if len(complete_trajectory) > 1:
                traj_pts = np.array([[int(x), int(y)] for x, y in complete_trajectory], dtype=np.int32)
                cv2.polylines(viz_map, [traj_pts], False, (0, 165, 255), 1)

            # --- NEW: Draw Memory Points (Gray Dots) ---
            if memory_points:
                for mp in memory_points:
                    mx, my = int(mp[0]), int(mp[1])
                    # Draw small gray dot indicating valid vantage point
                    cv2.circle(viz_map, (mx, my), 2, (100, 100, 100), -1)
            # -------------------------------------------

            # --- Draw Nodes, Branches, Frontier Centroids ---
            comp_data = {}
            if prediction_data:
                comp_data = prediction_data.get('local_components', {})
            active_dirs = {
                (d['node_id'], d['rank'])
                for d in comp_data.get('active_directions', [])
            }
            frontier_centroids = comp_data.get('frontier_centroids', [])

            dim_palette = [
                (0, 140, 140),  # Rank 1 \u2014 dim cyan
                (140, 140, 0),  # Rank 2 \u2014 dim yellow
                (140, 0, 140),  # Rank 3 \u2014 dim magenta
                (0, 80, 160),  # Rank 4 \u2014 dim orange-blue
            ]
            bright_palette = [
                (0, 255, 255),  # Rank 1 \u2014 bright cyan
                (255, 255, 0),  # Rank 2 \u2014 bright yellow
                (255, 0, 255),  # Rank 3 \u2014 bright magenta
                (0, 165, 255),  # Rank 4 \u2014 bright orange
            ]

            nodes = state_data.get('exploration_nodes', [])
            for node in nodes:
                nx, ny = int(node['x']), int(node['y'])

                # Junction center \u2014 purple diamond
                cv2.drawMarker(viz_map, (nx, ny), (255, 0, 255),
                               cv2.MARKER_DIAMOND, 16, 4)

                for comp in node.get('components', []):
                    cx, cy = int(comp['x']), int(comp['y'])
                    rank = comp['rank']
                    is_active = (node['id'], rank) in active_dirs

                    color = bright_palette[(rank - 1) % len(bright_palette)] if is_active \
                        else dim_palette[(rank - 1) % len(dim_palette)]

                    dot_r = 9 if is_active else 5
                    ring_r = 13 if is_active else 8
                    ring_t = 3 if is_active else 1

                    # Connecting line from node center to branch
                    cv2.line(viz_map, (nx, ny), (cx, cy), color, 1, cv2.LINE_AA)
                    # Filled branch dot
                    cv2.circle(viz_map, (cx, cy), dot_r, color, -1)
                    # White ring (thicker = active)
                    cv2.circle(viz_map, (cx, cy), ring_r, (255, 255, 255), ring_t)
                    # Second outer ring for active branches (clearly distinguishable glow)
                    if is_active:
                        cv2.circle(viz_map, (cx, cy), ring_r + 5, (255, 255, 255), 1)
                    # Rank label
                    cv2.putText(viz_map, str(rank), (cx - 3, cy + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw frontier centroids used for active branch assignment
            for fx, fy in frontier_centroids:
                fx, fy = int(fx), int(fy)
                # Small filled orange circle + white cross to distinguish from branches
                cv2.circle(viz_map, (fx, fy), 6, (0, 140, 255), -1)
                cv2.circle(viz_map, (fx, fy), 9, (255, 255, 255), 1)
                cv2.drawMarker(viz_map, (fx, fy), (255, 255, 255),
                               cv2.MARKER_CROSS, 10, 1)

            # --- NEW: Draw Persistent Potential Nodes (Lime Green) ---
            with self.nodes_lock:
                pot_nodes = copy.deepcopy(self.potential_nodes)

            for p_node in pot_nodes:
                nx, ny = int(p_node['x']), int(p_node['y'])

                # Junction center \u2014 Lime diamond only
                cv2.drawMarker(viz_map, (nx, ny), (0, 255, 0), cv2.MARKER_DIAMOND, 16, 4)

            # 3. Draw Current Path (Red Line)
            path = state_data.get('current_path')
            if path and len(path) > 1:
                pts = np.array([[int(x), int(y)] for x, y in path], dtype=np.int32)
                cv2.polylines(viz_map, [pts], False, (0, 0, 255), 2)

            # 5. Draw REAL Robot (Blue Circle)
            rx, ry, rtheta = pose[:3]
            cv2.circle(viz_map, (int(rx), int(ry)), self.robot_radius_pixels, (255, 0, 0), -1)
            arrow_end = (int(rx + 25 * np.cos(rtheta)), int(ry + 25 * np.sin(rtheta)))
            cv2.arrowedLine(viz_map, (int(rx), int(ry)), arrow_end, (255, 255, 0), 3)

            # 6. Zoom Crop (650x650)
            zoom_half_px = 300
            x1 = max(0, int(rx) - zoom_half_px)
            x2 = min(W, int(rx) + zoom_half_px)
            y1 = max(0, int(ry) - zoom_half_px)
            y2 = min(H, int(ry) + zoom_half_px)

            final_zoom = viz_map[y1:y2, x1:x2].copy()
            if final_zoom.shape[0] != 650 or final_zoom.shape[1] != 650:
                final_zoom = cv2.resize(final_zoom, (650, 650))

            return final_zoom, viz_map

        except Exception as e:
            print(f"VISUALIZATION: Global view error: {e}")
            return None

    def _create_simple_local_view(self, prediction_data):
        """Create local view showing the prediction heatmap overlay"""
        try:
            if prediction_data is None:
                win_px = self.config.sim.get('WIN_PX', 400)
                blank = np.full((win_px, win_px, 3), 64, dtype=np.uint8)
                cv2.putText(blank, "Waiting for prediction...", (50, win_px // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                return blank

            origin_img_np = prediction_data.get('origin_img')
            heatmap_np = prediction_data.get('heatmap')
            goal_px = prediction_data.get('goal_px')

            if origin_img_np is None or heatmap_np is None:
                win_px = self.config.sim.get('WIN_PX', 400)
                return np.full((win_px, win_px, 3), 64, dtype=np.uint8)

            # Convert origin image to BGR
            if len(origin_img_np.shape) == 2:
                origin_bgr = cv2.cvtColor(origin_img_np, cv2.COLOR_GRAY2BGR)
            else:
                origin_bgr = origin_img_np.copy()

            # Normalize heatmap to [0, 255] for visualization
            heatmap_norm = heatmap_np.copy()
            heatmap_min = heatmap_norm.min()
            heatmap_max = heatmap_norm.max()

            if heatmap_max > heatmap_min:
                # Normalize to 0-255
                heatmap_norm = ((heatmap_norm - heatmap_min) / (heatmap_max - heatmap_min) * 255).astype(np.uint8)
            else:
                heatmap_norm = np.zeros_like(heatmap_norm, dtype=np.uint8)

            # Apply colormap (VIRIDIS: purple=low, yellow=high - same as original)
            heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_VIRIDIS)

            # Blend with origin image (same alpha as original)
            alpha = 0.5
            overlay = cv2.addWeighted(origin_bgr, alpha, heatmap_colored, alpha, 0)

            # Draw goal marker (same as original)
            if goal_px and len(goal_px) >= 2:
                cv2.circle(overlay, (int(goal_px[0]), int(goal_px[1])), 10, (0, 255, 0), 3)
                cv2.circle(overlay, (int(goal_px[0]), int(goal_px[1])), 5, (255, 255, 255), -1)

            # Add text overlay with heatmap stats (NEW - helpful for debugging)
            max_val = heatmap_np.max()
            text = f"Pred Max: {max_val:.6f}"
            cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 1, cv2.LINE_AA)

            comp = prediction_data.get("local_components", None)
            if comp is not None:
                label_map = comp.get("label_map", None)
                keep_ids = comp.get("keep_ids", [])
                centroids = comp.get("centroids", {})

                palette = [
                    (0, 255, 255), (255, 255, 0), (255, 0, 255),
                    (0, 165, 255), (0, 255, 0), (255, 0, 0),
                ]
                alpha = 0.4

                for k, cc_id in enumerate(keep_ids[:6]):
                    color = palette[k % len(palette)]
                    m = (label_map == cc_id)

                    # Semi-transparent filled mask
                    overlay[m] = (
                            overlay[m].astype(np.float32) * (1 - alpha) +
                            np.array(color, dtype=np.float32) * alpha
                    ).astype(np.uint8)

                    # Contour outline
                    cmask = (m.astype(np.uint8) * 255)
                    contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay, contours, -1, color, 2)

                    # Weighted centroid: filled dot + ring
                    if cc_id in centroids:
                        cx, cy = int(round(centroids[cc_id][0])), int(round(centroids[cc_id][1]))
                        cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1)  # white fill
                        cv2.circle(overlay, (cx, cy), 6, color, 2)  # colored ring
                        cv2.putText(overlay, str(k + 1), (cx + 8, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            return overlay

        except Exception as e:
            print(f"VISUALIZATION: Local view error: {e}")
            import traceback
            traceback.print_exc()
            win_px = self.config.sim.get('WIN_PX', 400)
            return np.full((win_px, win_px, 3), 64, dtype=np.uint8)

    def _cleanup(self):
        """Cleanup resources."""
        print("Starting cleanup...")

        try:
            self.running.clear()

            # Wait for threads
            threads = [
                (self.prediction_thread, "prediction"),
                (self.simulation_thread, "simulation"),
                (self.visualization_thread, "visualization")
            ]

            for thread, name in threads:
                if thread and thread.is_alive():
                    try:
                        thread.join(timeout=3.0)
                        print(f"{name} thread finished")
                    except Exception as e:
                        print(f"Error joining {name} thread: {e}")

            # Cleanup resources
            try:
                if self.video_writer is not None:
                    self.video_writer.release()
                cv2.destroyAllWindows()
                print("Resources cleaned up")
            except Exception as e:
                print(f"Error during cleanup: {e}")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _print_exploration_summary(self):
        """Print exploration summary."""
        try:
            with self.state_lock:
                exploration_pct = self.shared_state.get('exploration_pct', 0.0)
                total_path_length = self.shared_state.get('total_path_length', 0.0)
                goal_count = self.shared_state.get('goal_count', 0)

            elapsed_time = time.time() - self.start_time

            print("\n" + "=" * 60)
            print("UPDATED GOAL-BASED EXPLORATION SUMMARY")
            print("=" * 60)
            print(f"Final Exploration: {exploration_pct:.2f}%")
            print(f"Goals Reached: {goal_count}")
            print(f"Total Path Length: {total_path_length:.2f} meters")
            print(f"Total Time: {elapsed_time:.1f} seconds")
            print(f"Average Goals per Minute: {goal_count / (elapsed_time / 60):.1f}")
            print("=" * 60)

        except Exception as e:
            print(f"Error printing summary: {e}")


# --- 4. Main Execution Block ---
if __name__ == '__main__':
    class ExplorationConfig:
        # UPDATED: Point to YOUR trained model
        MODEL_CHECKPOINT = weights_path("intelx_unet_resnet34.pt")
        MODEL_TYPE = 'resnet34'  # Must match your training
        output_dir = "exploration_results"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        HEADLESS = False  # True = No Window (Fast), False = Show Window
        TURBO_MODE = True  # True = No artificial sleep (Max Speed)
        MAP_UPDATE_INTERVAL = 10  # Update map every 10 steps (Higher = Faster)
        SAVE_INTERVAL = 1000  # Save snapshot every N steps (0 to disable)

        ROBOT_RADIUS_METERS = 0.35  # 0.5
        RESOLUTION = 0.05
        WIN_M = 20  # 20
        LASER_SCAN_R = 10

        # Baseline Frontier Parameters
        LOCAL_FRONTIER_WINDOW_M = 10.0
        GLOBAL_FRONTIER_WINDOW_M = 100.0
        MIN_FRONTIER_SIZE = 5

        # ROS explore_lite Cost Function Weights
        POTENTIAL_SCALE = 3.0  # Weight for distance (cost)
        GAIN_SCALE = 1.0  # Weight for frontier size (reward)

        sim = {  # data_remapped/train/50010535_PLAN1.pgm
            'MAP_PGM': map_path('train_standardized/small_house_aws.pgm'),
            'RESOLUTION': RESOLUTION, 'ROBOT_R': ROBOT_RADIUS_METERS, 'WIN_M': WIN_M,
            'SPAWN_CLEARANCE': 0.45, 'FOV': 220 * pi / 180, 'N_BEAMS': 662, 'MAX_R': int(LASER_SCAN_R / RESOLUTION),
            'STEP_SIZE_M': 0.1, 'ANG_VEL': pi, 'DT': 0.1, 'WIN_PX': int(WIN_M / RESOLUTION),
            'SAVE_DIR': 'temp_autonomous_exploration', 'SHOW_HEAD': True, 'VISUALIZE_FOV': False,
            'RAD_CELLS': int(ROBOT_RADIUS_METERS / RESOLUTION)
        }
        sim['ZOOM_WINDOW_M'] = 40.0
        sim['ZOOM_WINDOW_PX'] = int(sim['ZOOM_WINDOW_M'] / sim['RESOLUTION'])

        policy = {
            'exploration_target_pct': 98,
            'goal_history_length': 5
        }


    try:
        print("=" * 70)
        print("UPDATED GOAL-BASED AUTONOMOUS EXPLORATION SYSTEM")
        print("=" * 70)

        config = ExplorationConfig()
        os.makedirs(config.output_dir, exist_ok=True)

        print("Loading model...")
        model = create_model_for_inference(
            config.MODEL_CHECKPOINT,
            config.device,
            model_type=config.MODEL_TYPE
        )
        print(f"Model loaded on {config.device}")

        print("Initializing simulator...")
        simulator = ExplorationSimulator(
            config=config.sim,
            initial_spawn_point=None,
            log_path=None,
            skip_initial_scan=False
        )

        print("Simulator initialized successfully")

        print("Creating updated goal-based autonomous explorer...")
        explorer = UpdatedGoalBasedAutonomousExplorer(simulator, model, config)
        print("Explorer created successfully")

        print("Starting updated goal-based exploration...")
        explorer.run_exploration_episode()

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("Program finished - press Enter to exit")
        input()