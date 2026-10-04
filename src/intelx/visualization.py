"""
intelx.visualization
====================
`VisualizationMixin` - all rendering, and the video/snapshot worker loop. Pure
presentation: nothing here changes the robot's decisions, so you can ignore this
file entirely while iterating on the policy.

VIEW_STYLE in the config picks the frame. 'paper' (the default) is the live
view of intelx.viz.live, the look of the paper's videos. 'debug' is the older
three-panel dashboard: zoom, local planner inspector and global map.
"""
import os
import queue
import copy
import time

import numpy as np
import cv2


class VisualizationMixin:
    """Rendering behaviour. Mixed into the Explorer."""

    # ------------------------------------------------------------------
    # The paper-style live view (intelx.viz.live)
    # ------------------------------------------------------------------
    def _paper_style(self):
        return str(getattr(self.config, 'VIEW_STYLE', 'paper')) == 'paper'

    def _paper_view(self):
        if getattr(self, '_live_view', None) is None:
            from intelx.viz.live import LiveView
            from intelx.viz.style import pretty_map_name
            sim = self.config.sim
            stem = os.path.splitext(os.path.basename(str(sim.get('MAP_PGM', ''))))[0]
            local = bool(getattr(self.config, 'LOCAL_ONLY_MODE', False))
            self._live_view = LiveView(
                self.simulator.occ, method='local_only' if local else 'ours',
                title=("IntelX, local only: %s" if local else "IntelX: %s")
                % pretty_map_name(stem),
                spawn=getattr(self, 'spawn_pose', None), fov=sim.get('FOV'),
                max_r_px=sim.get('MAX_R'), win_px=sim.get('WIN_PX', 400),
                target_pct=self.config.policy.get('exploration_target_pct', 99))
        return self._live_view

    def _write_video(self, frame):
        if self.video_writer is None:
            h, w = frame.shape[:2]
            self.video_writer = cv2.VideoWriter(
                self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, (w, h))
            if self.video_writer.isOpened():
                print(f"VISUALIZATION: VideoWriter opened ({w}x{h}) -> {self.video_path}")
            else:
                print(f"VISUALIZATION: ERROR - VideoWriter failed to open: {self.video_path}")
                self.video_writer = None
        if self.video_writer is not None:
            self.video_writer.write(frame)

    def _paper_loop(self, view_fps):
        """Draw the live view on a timer, from the explorer's live state.

        The simulation thread publishes its state every STATE_PUBLISH_EVERY
        iterations, which in a turbo run can be once or twice a second; the
        view reads the pose and the explored map straight from the simulator
        instead, so it moves at VIEW_FPS."""
        from intelx.viz.live import snapshot
        window = "Updated Goal-Based Exploration"
        period = 1.0 / max(view_fps, 0.1)
        while self.running.is_set() and not self.exploration_complete.is_set():
            t0 = time.time()
            try:                                   # nothing else reads this queue
                while True:
                    self.state_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                s = snapshot(self)
                s['elapsed_s'] = time.time() - self.start_time
                frame = self._paper_view().render(s)
                self._write_video(frame)
                self.last_frame = frame
                if not self.headless:
                    cv2.imshow(window, frame)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        self.running.clear()
                        break
            except Exception as e:
                print(f"VISUALIZATION: Worker error: {e}")
            time.sleep(max(0.0, period - (time.time() - t0)))

    def _visualization_worker(self):
        """
        Headless-Capable Worker.
        If HEADLESS=True: Saves PNG snapshots instead of showing a window.
        """
        print(f"VISUALIZATION: Worker starting (Headless: {self.headless})...")
        # The paper view is drawn at a steady rate, and the video is written at
        # that same rate, so it plays back at the speed of the run.
        view_fps = float(getattr(self.config, 'VIEW_FPS', 10.0))
        if self._paper_style():
            self.video_fps = view_fps

        # Create output directory for snapshots
        snapshot_dir = os.path.join(self.config.output_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        prediction_data = None

        try:
            if not self.headless:
                cv2.namedWindow("Updated Goal-Based Exploration", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Updated Goal-Based Exploration",
                                 1600 if self._paper_style() else 1800,
                                 900 if self._paper_style() else 1000)
            if self._paper_style():
                self._paper_loop(view_fps)

            while (not self._paper_style() and self.running.is_set()
                   and not self.exploration_complete.is_set()):
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
                            # Build the FULL dashboard frame (zoom + LOCAL PLANNER
                            # inspector panel + global map) so the local-planner
                            # prediction window is visible in headless output too --
                            # not just the global map. This is the same frame the
                            # GUI shows, so LOCAL / LOCAL_ONLY / junction all render
                            # their prediction window identically.
                            frame = self._create_updated_visualization_frame(state_data, prediction_data)

                            # Write every frame to the video (lazy-init the writer
                            # with the actual frame size), matching GUI behaviour.
                            if frame is not None:
                                if self.video_writer is None:
                                    h, w = frame.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    self.video_writer = cv2.VideoWriter(
                                        self.video_path, fourcc, self.video_fps, (w, h))
                                    if self.video_writer.isOpened():
                                        print(f"VISUALIZATION: VideoWriter opened ({w}x{h}) -> {self.video_path}")
                                    else:
                                        print(f"VISUALIZATION: ERROR - VideoWriter failed to open: {self.video_path}")
                                        self.video_writer = None
                                if self.video_writer is not None:
                                    self.video_writer.write(frame)
                                self.last_frame = frame.copy()

                            # Save a PNG snapshot of the SAME dashboard frame at the
                            # configured interval.
                            if self.save_interval > 0 and movement_count > 0 and movement_count % self.save_interval == 0:
                                if frame is not None:
                                    filename = f"{snapshot_dir}/step_{movement_count:06d}.png"
                                    cv2.imwrite(filename, frame)
                                    print(f"VISUALIZATION: Saved snapshot {filename}")

                        # --- GUI MODE ---
                        else:
                            frame = self._create_updated_visualization_frame(state_data, prediction_data)
                            if frame is not None:
                                cv2.imshow("Updated Goal-Based Exploration", frame)
                                # Lazy VideoWriter init - use actual frame size
                                if self.video_writer is None:
                                    h, w = frame.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    self.video_writer = cv2.VideoWriter(self.video_path, fourcc, self.video_fps, (w, h))
                                    if self.video_writer.isOpened():
                                        print(f"VISUALIZATION: VideoWriter opened ({w}x{h}) -> {self.video_path}")
                                    else:
                                        print(f"VISUALIZATION: ERROR - VideoWriter failed to open: {self.video_path}")
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
                if self._paper_style():
                    from intelx.viz.live import snapshot
                    s = snapshot(self)
                    s['elapsed_s'] = time.time() - self.start_time
                    pct = final_state.get('exploration_pct', 0.0)
                    done = pct >= self.config.policy.get('exploration_target_pct', 99)
                    full_map = self._paper_view().render(
                        s, status='complete' if done else 'stopped')
                else:
                    _, full_map = self._create_updated_global_view(final_state, prediction_data, [])
                if full_map is not None:
                    cv2.imwrite(os.path.join(self.config.output_dir, "final_trajectory.png"), full_map)
            except Exception as e:
                print(f"Could not save final image: {e}")

            if not self.headless:
                cv2.destroyAllWindows()

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

            elif (mode == "LOCAL" or mode.startswith("LOCAL_ONLY")) and prediction_data:
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
                (0, 140, 140),  # Rank 1 - dim cyan
                (140, 140, 0),  # Rank 2 - dim yellow
                (140, 0, 140),  # Rank 3 - dim magenta
                (0, 80, 160),  # Rank 4 - dim orange-blue
            ]
            bright_palette = [
                (0, 255, 255),  # Rank 1 - bright cyan
                (255, 255, 0),  # Rank 2 - bright yellow
                (255, 0, 255),  # Rank 3 - bright magenta
                (0, 165, 255),  # Rank 4 - bright orange
            ]

            nodes = state_data.get('exploration_nodes', [])
            for node in nodes:
                nx, ny = int(node['x']), int(node['y'])

                # Junction center - purple diamond
                cv2.drawMarker(viz_map, (nx, ny), (255, 0, 255),
                               cv2.MARKER_DIAMOND, 16, 4)

                for comp in node.get('components', []):
                    cx, cy = int(comp['x']), int(comp['y'])
                    rank = comp['rank']
                    is_active = (node['id'], rank) in active_dirs
                    is_explored = comp.get('explored', False)

                    color = bright_palette[(rank - 1) % len(bright_palette)] if is_active \
                        else dim_palette[(rank - 1) % len(dim_palette)]

                    dot_r = 9 if is_active else 5
                    ring_r = 13 if is_active else 8
                    ring_t = 3 if is_active else 1

                    # Connecting line from node center to branch
                    cv2.line(viz_map, (nx, ny), (cx, cy), color, 1, cv2.LINE_AA)
                    # Filled branch dot
                    cv2.circle(viz_map, (cx, cy), dot_r, color, -1)
                    # Strike-through dot for explored branches
                    if is_explored:
                        cv2.line(viz_map, (cx - dot_r, cy), (cx + dot_r, cy), (255, 255, 255), 1)
                    # White ring (thicker = active)
                    cv2.circle(viz_map, (cx, cy), ring_r, (255, 255, 255), ring_t)
                    # Second outer ring for active branches
                    if is_active:
                        cv2.circle(viz_map, (cx, cy), ring_r + 5, (255, 255, 255), 1)
                    # Rank label inside dot
                    cv2.putText(viz_map, str(rank), (cx - 3, cy + 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

                    # --- Coverage value label below the branch dot ---
                    cov_val = comp.get('cov', None)
                    if cov_val is not None:
                        cov_str = f"c:{cov_val:.2f}"
                        (tw, th), _ = cv2.getTextSize(cov_str, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                        label_x = cx - tw // 2
                        label_y = cy + ring_r + th + 6
                        # Dark background pill for readability
                        cv2.rectangle(viz_map,
                                      (label_x - 2, label_y - th - 2),
                                      (label_x + tw + 2, label_y + 2),
                                      (20, 20, 20), -1)
                        # Text in bright green (easy to distinguish from map colours)
                        cv2.putText(viz_map, cov_str, (label_x, label_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                                    (0, 255, 128), 1, cv2.LINE_AA)

            # --- Draw detected doorways on global map ---
            with self.doorways_lock:
                doorways_to_draw = list(self.detected_doorways_world)

            for dw in doorways_to_draw:
                gx = int(round(dw['world_x']))
                gy = int(round(dw['world_y']))
                half_w = max(4, int(round(dw['width_px'] / 2)))

                # Skip if out of bounds
                if not (0 <= gx < viz_map.shape[1] and 0 <= gy < viz_map.shape[0]):
                    continue

                # Amber horizontal bar = the width of the doorway gap
                cv2.line(viz_map,
                         (gx - half_w, gy), (gx + half_w, gy),
                         (0, 180, 255), 2)

                # Diamond centre marker
                cv2.drawMarker(viz_map, (gx, gy),
                               (0, 180, 255), cv2.MARKER_DIAMOND, 14, 2)

                # Width label
                cv2.putText(viz_map,
                            f"D:{dw['width_m']:.1f}m",
                            (gx + 9, gy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1, cv2.LINE_AA)

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

                # Junction center - Lime diamond only
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