"""
intelx.explorer
===============
Assembles the explorer from its mixins and owns construction + lifecycle.

    UpdatedGoalBasedAutonomousExplorer
        = PlanningMixin        (intelx.policy_ablation) -> WHICH goal  [edit for policy]
        + SimulationMixin      (intelx.simulation)    -> HOW to reach it
        + VisualizationMixin   (intelx.visualization) -> how to draw it

Only __init__, run_exploration_episode, _start_threads, _cleanup and
_print_exploration_summary live here. The class name and constructor signature
are unchanged from the monolith, so existing entry points keep working.
"""
import os
import time
import threading
import queue
import copy
from collections import deque

import numpy as np
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm

from intelx.common import RobotState, safe_copy_pose, safe_copy_array
from intelx.policy_ablation import PlanningMixin
from intelx.simulation import SimulationMixin
from intelx.visualization import VisualizationMixin
from intelx.planners.global_priority import GlobalPriorityPlanner


class UpdatedGoalBasedAutonomousExplorer(PlanningMixin, SimulationMixin, VisualizationMixin):
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

        # --- Apply custom spawn point ---
        spawn = getattr(self.config, 'SPAWN_POINT', None)
        if spawn is not None:
            sx, sy, stheta = float(spawn[0]), float(spawn[1]), float(spawn[2])
            H_map, W_map = self.simulator.occ.shape
            if 0 <= int(sx) < W_map and 0 <= int(sy) < H_map:
                if self.simulator.occ[int(sy), int(sx)] == 0:
                    self.simulator.pose = np.array([sx, sy, stheta])
                    # Reset explored map so robot starts fresh from new position
                    self.simulator.explored = np.zeros_like(self.simulator.explored)
                    print(f"SPAWN: Robot placed at ({sx:.0f}, {sy:.0f}), theta={stheta:.2f} rad")
                else:
                    print(f"SPAWN WARNING: ({sx:.0f},{sy:.0f}) is a wall - using default spawn.")
            else:
                print(f"SPAWN WARNING: ({sx:.0f},{sy:.0f}) out of map bounds ({W_map}x{H_map}) - using default.")

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

        self.spawn_pose = tuple(float(v) for v in np.asarray(initial_pose).ravel()[:3])
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
        self.detected_doorways_world = []  # persists doorway locations across frames
        self.doorways_lock = threading.Lock()

        self.state_lock = threading.Lock()

        # Control flags
        self.running = threading.Event()
        self.running.set()
        self.exploration_complete = threading.Event()

        # Video recording setup
        try:
            # Video recording setup - lazy init on first frame (size unknown until then)
            self.video_path = os.path.join(self.config.output_dir, "intelx_video.mp4")
            self.video_writer = None
            self.video_fps = 30.0
            print(f"Video will be saved to: {self.video_path}")
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

        # --- FIDELITY DIAGNOSTICS ---------------------------------------
        # These answer "did the speedups make the robot hop / thin out the
        # map?" empirically at the end of every run, instead of by argument.
        #   max_step_px  : largest single motion increment (should stay 0.75)
        #   scan_gap_*   : distance between consecutive laser scans fused
        #                  into the explored map
        #   pred_passes  : U-Net prediction passes actually executed
        self.diag = {
            'max_step_px': 0.0,
            'scan_count': 0,
            'scan_gap_sum': 0.0,
            'scan_gap_max': 0.0,
            'pred_passes': 0,
        }
        self.diag_lock = threading.Lock()
        # ----------------------------------------------------------------

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
                # Release video writer before destroying windows
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    print(f"Video saved: {self.video_path}")

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

            # --- FIDELITY CHECK -----------------------------------------
            # Confirms the speed settings did not make the robot hop or
            # thin out the laser scans that build the explored map.
            res = self.config.sim.get('RESOLUTION', 0.05)
            d = self.diag
            n = max(1, d['scan_count'])
            mean_gap = d['scan_gap_sum'] / n
            print("-" * 60)
            print("FIDELITY CHECK (speed settings vs map generation)")
            print(f"  Max motion step:      {d['max_step_px']:.3f} px "
                  f"({d['max_step_px'] * res * 100:.1f} cm)   [expect 0.750 px with FIXED_DT=0.05]")
            print(f"  Laser scans fused:    {d['scan_count']}")
            print(f"  Scan spacing mean:    {mean_gap:.2f} px ({mean_gap * res * 100:.1f} cm)")
            print(f"  Scan spacing max:     {d['scan_gap_max']:.2f} px "
                  f"({d['scan_gap_max'] * res * 100:.1f} cm)")
            if total_path_length > 0:
                print(f"  Prediction passes:    {d['pred_passes']} "
                      f"({d['pred_passes'] / total_path_length:.1f} per metre travelled)")
            print("  NOTE: a large max scan spacing means the sim outran the")
            print("        prediction/scan loop -- lower TURBO or raise fidelity.")
            print("=" * 60)

        except Exception as e:
            print(f"Error printing summary: {e}")