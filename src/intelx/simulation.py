"""
intelx.simulation
=================
`SimulationMixin` - the actuation half. Consumes goals produced by the policy
and drives the robot: state machine (PREDICTION_NEEDED -> PLANNING ->
MOVING_TO_GOAL -> GOAL_REACHED), path planning to the chosen goal, stepwise
motion, continuous map updates, and publishing shared/visualization state.

Edit this file to change HOW a goal is reached (path planner used, step size,
sensing cadence, completion criteria) - not WHICH goal is chosen (that is
intelx.policy_ablation).
"""
import time
import queue
import copy
from math import hypot

import numpy as np

from intelx.common import (
    RobotState,
    safe_copy_pose,
    safe_copy_array,
    _transform_image_goal_to_world,
    astar_with_clearance,
)
from intelx.sim.simulator_astar import optimized_astar_path_planner


# SPEED: copy.deepcopy on lists of [float, float] is very slow (it walks the
# object graph and memoises). These produce the same independent copies for the
# flat point-list structures used here, roughly an order of magnitude faster.
def _copy_pts(pts):
    """Copy a list of 2-element point sequences."""
    if not pts:
        return [] if pts is not None else None
    return [[p[0], p[1]] for p in pts]


def _copy_paths(paths):
    """Copy a list of point-lists."""
    return [[[p[0], p[1]] for p in path] for path in paths]


class SimulationMixin:
    """Actuation / state-machine behaviour. Mixed into the Explorer."""

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
            loop_count = 0
            exploration_pct = 0.0
            last_scan_pos = None

            # SPEED: `occ` is the static ground-truth map (only `explored` changes),
            # so snapshot it and its free-space mask once instead of per iteration.
            occ_snapshot = safe_copy_array(self.simulator.occ)
            free_mask = (occ_snapshot == 0)
            total_free_space = max(1, int(np.count_nonzero(free_mask)))

            # State machine variables
            current_state = RobotState.PREDICTION_NEEDED
            waiting_for_prediction_start = None

            # SPEED + REPRODUCIBILITY: decouple physics from the wall clock.
            # The original used dt = min(0.1, wall_clock_delta), so how far the
            # robot moved per iteration depended on how busy the machine was --
            # which made path length partly a measure of Python speed and made
            # runs non-reproducible. With a fixed dt the trajectory is identical
            # regardless of how fast the loop actually runs, so TURBO_MODE can
            # remove the sleep and finish the same episode in less wall time.
            # Set FIXED_DT = None in the config to restore the old behaviour.
            fixed_dt = getattr(self.config, 'FIXED_DT', 0.05)

            # [TURBO-SYNC v2] wall-clock deadline for the predictor-sync wait, so a
            # stalled/starved predictor can never freeze the robot (see below).
            self._sync_wait_start = None

            while self.running.is_set() and not self.exploration_complete.is_set():
                try:
                    current_time = time.time()
                    if fixed_dt is not None:
                        dt = float(fixed_dt)
                    else:
                        dt = min(0.1, current_time - last_update_time)
                    last_update_time = current_time

                    # State machine logic
                    if current_state == RobotState.PREDICTION_NEEDED:
                        print("SIMULATION: State = PREDICTION_NEEDED - requesting prediction")

                        # Request new prediction
                        try:
                            current_pose = safe_copy_pose(self.simulator.pose)
                            # [REQ-ID] Stamp every request. Without this the sim
                            # cannot tell a fresh decision from one computed for
                            # a pose it left a minute ago, and after a timeout
                            # there are two requests in flight -- so the worker
                            # answers the first, the sim drives it, the worker
                            # then answers the second (still for the OLD pose)
                            # and parks it in the queue, where the next get()
                            # returns it immediately.
                            self._req_seq = getattr(self, '_req_seq', 0) + 1
                            _expect_req_id = self._req_seq
                            request_data = {
                                'pose': current_pose,
                                'timestamp': time.time(),
                                'req_id': _expect_req_id,
                            }

                            # Clear old requests
                            while not self.prediction_request_queue.empty():
                                try:
                                    self.prediction_request_queue.get_nowait()
                                except queue.Empty:
                                    break
                            # [REQ-ID] and any response left over from a previous
                            # request, which would otherwise be consumed as if it
                            # answered this one.
                            while not self.prediction_response_queue.empty():
                                try:
                                    self.prediction_response_queue.get_nowait()
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
                        # Check for timeout.
                        # [TIMEOUT] This was a hardcoded 15.0 while
                        # intelx.baselines.frontier read PREDICTION_TIMEOUT_S, so
                        # the pinned 900 s applied to every baseline and not to
                        # `ours`. Fallback decisions measured 26-56 s, so this
                        # fired on essentially all of them, leaving two requests
                        # in flight every time.
                        _pred_timeout = float(getattr(self.config,
                                                      'PREDICTION_TIMEOUT_S', 15.0))
                        if waiting_for_prediction_start and (
                                time.time() - waiting_for_prediction_start) > _pred_timeout:
                            print("SIMULATION: Prediction timeout (%.0fs), restarting..."
                                  % _pred_timeout)
                            current_state = RobotState.PREDICTION_NEEDED
                            continue

                        # Wait for prediction response
                        try:
                            prediction_response = self.prediction_response_queue.get(timeout=0.5)

                            # [REQ-ID] Discard anything that does not answer the
                            # request we are actually waiting on. goal_px is an
                            # IMAGE-FRAME point relative to request_pose, so a
                            # stale response is not merely late -- it steers the
                            # robot from a pose it has already left.
                            _got_id = prediction_response.get('req_id')
                            if _got_id is not None and _got_id != _expect_req_id:
                                print("SIMULATION: discarding stale prediction "
                                      "(req_id %s, waiting for %s)"
                                      % (_got_id, _expect_req_id))
                                continue

                            print(f"SIMULATION: Received prediction response!")

                            # Transform goal to world coordinates
                            goal_px = prediction_response['goal_px']
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

                                # Use A* planner from intelx.sim.simulator_astar
                                # path_result = optimized_astar_path_planner(
                                #     self.simulator.occ,
                                #     robot_px,
                                #     goal_px_coords
                                # )
                                path_result = astar_with_clearance(
                                    self.simulator.occ,
                                    robot_px,
                                    goal_px_coords,
                                    clearance_m=1.0,
                                    resolution=self.config.sim.get('RESOLUTION', 0.05)
                                )

                                print(f"SIMULATION: Path result type: {type(path_result)}")
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
                        # [PREEMPT] Issue 2: while driving to a fallback frontier
                        # (which tends to be deeper than needed), the policy keeps
                        # running the local planner on its idle passes. The instant
                        # it sees real coverage on the way (or the enroute frontier
                        # enters the local window with no coverage -- the enclosed-
                        # room early-stop), it sets self.preempt_current_goal so we
                        # abandon this path and re-plan -- on the next
                        # PREDICTION_NEEDED->PLANNING the policy returns the (now
                        # available) local goal via normal arbitration. The flag is
                        # only ever set while the policy is en route to a fallback
                        # target, so a normal local goal is never interrupted.
                        if getattr(self, 'preempt_current_goal', False):
                            self.preempt_current_goal = False
                            print("SIMULATION: preempting current goal -> local coverage / early-stop detected en route")
                            self.current_path = None
                            self.path_index = 0
                            current_state = RobotState.PREDICTION_NEEDED
                            continue

                        # [TURBO-SYNC] Don't let the robot outrun the prediction
                        # thread in turbo. FIXED_DT makes the MOTION deterministic,
                        # but the preempt / early-stop signals are produced by the
                        # (GPU-bound) prediction thread on its idle passes. In turbo
                        # the unthrottled sim loop can drive the robot most of the
                        # way to the frontier before an idle pass even runs, so the
                        # preempt fires too late (the lag you saw). Bounding the
                        # robot to at most PRED_MAX_LAG_PX ahead of the last pose the
                        # predictor actually processed keeps the two threads in step,
                        # so preempt/early-stop can fire in time.
                        #
                        # No deadlock: the robot has already advanced to this pose, so
                        # when it pauses the predictor's next idle pass satisfies its
                        # own moved-enough gate (>= IDLE_PREDICT_EVERY_PX), processes
                        # this pose, and the lag clears (max_lag is clamped >= that
                        # gate). Skipped when idle prediction is off (nothing to sync)
                        # or turbo is off. Set TURBO_SYNC_PRED=False to disable.
                        # [TURBO-SYNC v2 -- DEADLOCK-PROOF]
                        # BUG FIXED HERE: v1 assumed that whenever the sim pauses, the
                        # predictor's next idle pass must fire. That is FALSE. The sim
                        # measures lag against latest_prediction_data['request_pose'],
                        # but the predictor gates its idle passes on its OWN anchor
                        # (last_idle_pos), and the two diverge after a NAV pass. If the
                        # robot sits >max_lag from request_pose but <IDLE_PREDICT_EVERY_PX
                        # from last_idle_pos, the predictor runs no idle pass, so
                        # request_pose never updates, so the lag never clears -- and the
                        # sim never asks for a new prediction because it is parked in
                        # MOVING_TO_GOAL. Hard deadlock (this is the "gets stuck" bug,
                        # and raising IDLE_PREDICT_EVERY_PX made it much more likely).
                        #
                        # Fix: never rely on that invariant. The wait is now BOUNDED by
                        # wall time -- after PRED_SYNC_MAX_WAIT_S we move anyway. Worst
                        # case the throttle degrades to "no throttle" for one step, which
                        # is exactly the old (fast) behaviour, so forward progress is
                        # guaranteed no matter what the predictor is doing.
                        #   PRED_SYNC_MAX_WAIT_S = 0.5   (0 disables the throttle entirely)
                        if (getattr(self, 'turbo_mode', False)
                                and getattr(self.config, 'TURBO_SYNC_PRED', True)
                                and float(getattr(self.config, 'IDLE_PREDICT_EVERY_PX', 1.0)) > 0):
                            max_lag = max(
                                float(getattr(self.config, 'PRED_MAX_LAG_PX', 3.0)),
                                float(getattr(self.config, 'IDLE_PREDICT_EVERY_PX', 1.0)),
                            )
                            max_wait_s = float(getattr(self.config, 'PRED_SYNC_MAX_WAIT_S', 0.5))
                            with self.data_lock:
                                _lpd = self.latest_prediction_data
                            _rp = _lpd.get('request_pose') if _lpd else None
                            _waiting = False
                            if _rp is not None and max_wait_s > 0:
                                _lag = hypot(self.simulator.pose[0] - _rp[0],
                                             self.simulator.pose[1] - _rp[1])
                                if _lag > max_lag:
                                    now_s = time.time()
                                    if self._sync_wait_start is None:
                                        self._sync_wait_start = now_s
                                    if (now_s - self._sync_wait_start) < max_wait_s:
                                        _waiting = True
                                    else:
                                        # Bail out: predictor is not catching up (or is
                                        # deadlocked). Move anyway and re-arm the timer.
                                        self._sync_wait_start = None
                                        try:
                                            self.diag['turbo_sync_timeouts'] = \
                                                self.diag.get('turbo_sync_timeouts', 0) + 1
                                        except Exception:
                                            pass
                            if _waiting:
                                try:
                                    self.diag['turbo_sync_waits'] = self.diag.get('turbo_sync_waits', 0) + 1
                                    if _lag > self.diag.get('turbo_sync_lag_max', 0.0):
                                        self.diag['turbo_sync_lag_max'] = float(_lag)
                                except Exception:
                                    pass
                                time.sleep(0.001)  # yield; predictor processes this pose next
                                continue
                            else:
                                self._sync_wait_start = None   # not waiting -> reset timer

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
                                        # DIAGNOSTIC: largest motion increment seen
                                        if step_size > self.diag['max_step_px']:
                                            self.diag['max_step_px'] = float(step_size)
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
                                        # NOTE: this used to be a hardcoded 5, ignoring the
                                        # MAP_UPDATE_INTERVAL config knob entirely. It now honours
                                        # the config; set MAP_UPDATE_INTERVAL = 5 to reproduce the
                                        # old behaviour exactly.
                                        if movement_count % self.map_update_interval == 0:
                                            try:
                                                scan_result = self.simulator._simulate_scan(self.simulator.pose)
                                                self.simulator._update_exploration(self.simulator.pose, scan_result)
                                                map_update_count += 1
                                                # DIAGNOSTIC: spacing between fused scans
                                                if last_scan_pos is not None:
                                                    gap = hypot(self.simulator.pose[0] - last_scan_pos[0],
                                                                self.simulator.pose[1] - last_scan_pos[1])
                                                    self.diag['scan_count'] += 1
                                                    self.diag['scan_gap_sum'] += gap
                                                    if gap > self.diag['scan_gap_max']:
                                                        self.diag['scan_gap_max'] = float(gap)
                                                last_scan_pos = (float(self.simulator.pose[0]),
                                                                 float(self.simulator.pose[1]))
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

                    # SPEED: publish state every N iterations instead of every one.
                    # The block below cost ~8 ms/iteration (mostly copy.deepcopy of
                    # plain float lists) plus two full-map scans, at 0.75 px of robot
                    # motion per iteration. Nothing downstream needs it at that rate:
                    # the viz thread throttles itself and the main loop polls at 2 Hz.
                    publish_every = max(1, int(getattr(self.config, 'STATE_PUBLISH_EVERY', 10)))
                    do_publish = (loop_count % publish_every == 0)
                    loop_count += 1

                    if do_publish:
                        # Calculate exploration percentage USING UPDATED MAP
                        try:
                            # occ is the static ground-truth map, so the free-space
                            # mask and its count are computed once, not every loop.
                            explored_free_space = np.count_nonzero(
                                self.simulator.explored & free_mask
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
                                # occ never changes -> share the one snapshot
                                self.shared_state['occ'] = occ_snapshot
                                self.shared_state['explored'] = safe_copy_array(self.simulator.explored)
                                # Store only last 10 paths to avoid memory issues
                                self.shared_state['all_planned_paths'] = _copy_paths(
                                    all_planned_paths[-10:]) if all_planned_paths else []

                                self.shared_state['current_goal'] = self.current_goal
                                self.shared_state['robot_state'] = current_state.value
                                self.shared_state['goal_count'] = goal_count
                                # In the shared state update:
                                self.shared_state['complete_trajectory'] = _copy_pts(
                                    self.complete_trajectory[-500:])  # Last 500 points
                                self.shared_state['current_path'] = _copy_pts(
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
                                'occ': occ_snapshot,
                                'explored': safe_copy_array(self.simulator.explored),
                                'complete_trajectory': _copy_pts(self.complete_trajectory[-500:]),
                                'current_path': _copy_pts(self.current_path) if self.current_path else None,
                                # pose entries are never mutated in place, so a shallow
                                # list copy is enough (deepcopy of 1000 arrays was not)
                                'memory_points': list(self.pose_history),
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

                    # Maintain simulation rate.
                    # TURBO_MODE was read into self.turbo_mode in __init__ but the
                    # original never used it: this sleep hard-capped the loop at
                    # ~20 Hz, i.e. the episode ran in real time (~0.75 m/s of robot
                    # motion). With fixed dt the trajectory does not depend on the
                    # loop rate, so in turbo we just yield instead of sleeping.
                    if self.turbo_mode:
                        time.sleep(0)          # yield to the other threads only
                    else:
                        time.sleep(max(0.01, 0.05 - (time.time() - current_time)))

                except Exception as e:
                    print(f"SIMULATION: Worker error: {e}")
                    time.sleep(0.001)

        except Exception as e:
            print(f"SIMULATION: Critical error: {e}")

        print(f"SIMULATION: Worker stopping")
        print(f"SIMULATION: Goals: {goal_count}, Movements: {movement_count}, Map updates: {map_update_count}")