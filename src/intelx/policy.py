"""
intelx.policy
=============
The same policy as `intelx.policy_ablation`, without the ablation switches.
No entry point imports this file: `intelx.explorer` composes
`intelx.policy_ablation`, so edit that one to change exploration behaviour.

The prediction thread is split into two policy functions plus a thin
orchestrator, so the local and global planners can be tweaked in isolation:

    plan_local(heatmap, current_pose)
        [B] heatmap -> angular-Voronoi components (keep_ids / centroids / label_map)

    plan_global(heatmap, current_pose, local_components)
        [D] frontier pre-pass: build active_branch_map + active_directions
        [E] junction Case A (inside a known junction): follow the locked angle
        [F] junction Case B (potential new junction): N-polygon verification

    _prediction_worker()  (orchestrator)
        setup -> LOCAL_ONLY early-out -> plan_local -> plan_global
        -> [G] goal arbitration -> [H] 0-component fallback -> response.

CHANGES IN THIS REVISION
------------------------
1. _frontier_consumed ASKED THE WRONG QUESTION  (this was causing the
   oscillation, and it was introduced by the previous patch)

   It tested whether the box around the point was explored. The point it is
   handed is a frontier CENTROID -- the MEAN of a cluster of frontier cells --
   and a frontier cell is KNOWN-FREE by definition, so their mean sits in
   known-free space with known-free space all around it. The box was therefore
   explored almost always, and the function returned True for essentially every
   LIVE frontier. Two things broke:

     * EARLY-STOP fired the instant any fallback trip began, since after the
       previous patch this predicate WAS its entire condition. From the log:

           FALLBACK -> frontier (4108,2773) astar=272px
           CHOSE goal dist=172.0px
           [EARLY-STOP] (182px away) -> replanning
           FALLBACK -> frontier (4108,2773)     <- still live, correctly offered
           CHOSE goal dist=183.1px
           [EARLY-STOP] (190px away) -> replanning

       find_global_frontiers keeps returning that frontier because it IS there.
       No trip to a frontier ever completed.

     * The frontier COMMITMENT is gated on `not _frontier_consumed(...)`, so it
       never kept a target and has never once run.

2. THE ROOT CAUSE WAS UPSTREAM: the target is a cluster MEAN

   Fixing the predicate alone is not enough. On a frontier that wraps around a
   room, the mean sits ~2 m from the nearest frontier cell, so ANY radius test
   on the centroid is unreliable. _frontier_boundary_point exists to fix
   exactly this but returned early for any point that was not a wall, letting
   an interior centroid through untouched.

   Frontier targets are now ALWAYS snapped onto a real frontier cell. Verified
   on the wrap-around case: raw centroid (120,120) is not a frontier cell, the
   snapped target (119,80) is, and the consumption test then reads live ->
   False, explored away -> True, where the raw centroid gave True in both.

   That also means the robot drives to an actual frontier boundary instead of a
   mean that can sit deep inside explored space or on the far side of a wall.

3. STRUCTURED PER-DECISION LOG  (DECISION_LOG)

   The prose log has to be reverse-engineered with regexes, and one of mine
   gave a wrong answer during this investigation. DECISION_LOG writes one CSV
   row per nav decision with the inputs to every branch, so "why did it choose
   this goal" is answerable with pandas and no parsing. See _dec_row for the
   column list and what each is for.

State this mixin reads/writes on `self` (provided by the Explorer):
  read : simulator, model, config, transform, running, nodes_lock,
         exploration_nodes, global_planner, prediction_request_queue,
         prediction_response_queue, data_lock
  write: latest_prediction_data, exploration_nodes,
         current_junction_id (lazy), _junction_debug_count (lazy)
"""
import csv
import os
import time
import queue
import copy
import logging
from math import pi
from typing import Tuple

import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from intelx.sim.simulator import (
    build_origin_image_north_oriented,
    build_virtual_pose_image_north_oriented,
)
from intelx.sim.simulator_astar import optimized_astar_path_planner
from intelx.common import (
    _transform_image_goal_to_world,
    safe_copy_pose,
    nudge_goal_along_heading,
)


# ================================================================= [PROFILE]
# Wall + thread-CPU profiler. `with self._prof('name'):` accumulates, per named
# section: [wall_s, calls, cpu_s]. wall is elapsed time; cpu is this THREAD's CPU
# time (time.thread_time). The difference wall-cpu is WAIT -- time the prediction
# thread was blocked on the GIL (another thread running) or on a GPU sync.
try:
    _THREAD_TIME = time.thread_time
    _THREAD_TIME()  # probe availability
except Exception:  # pragma: no cover
    def _THREAD_TIME():
        return 0.0


class _Timer:
    __slots__ = ('acc', 'name', 't', 'c')

    def __init__(self, acc, name):
        self.acc = acc
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        self.c = _THREAD_TIME()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t
        dc = _THREAD_TIME() - self.c
        e = self.acc.get(self.name)
        if e is None:
            self.acc[self.name] = [dt, 1, dc]
        else:
            e[0] += dt
            e[1] += 1
            e[2] += dc
        return False


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULLCTX = _NullCtx()
# ============================================================= end [PROFILE]


# --- LOCAL PLANNER KNOBS -----------------------------------------------------
top_percentile = 96.0  # initial mask: top N% pixels survive
smooth_sigma = 2.0  # Gaussian blur for stable mask edges
n_angle_bins = 72  # 5 deg per bin; increase for finer angular resolution
angular_smooth_k = 3  # bins of circular smoothing on angular histogram
min_arc_gap_bins = 2  # legacy default; MIN_ARC_GAP_BINS (config) is used instead
peak_height_thr = 0.1  # normalized angular histogram peak must exceed this
min_area_px = 10  # discard directional sectors with too few pixels
rel_peak_thr = 0.25  # component's max (normalized) >= 25% of global max
coverage_thr = 0.0005  # absolute raw heatmap threshold
coverage_majority = 0.1  # fraction of component pixels that must exceed coverage_thr
max_components = 4  # cap output


# Column order for the DECISION_LOG csv. Kept module-level so an analysis
# script can import it rather than hardcoding the order.
DECISION_COLUMNS = [
    # who / when
    "pred", "t_s", "path_m", "coverage_pct",
    # robot state
    "pose_x", "pose_y", "heading_deg",
    # what the local planner saw
    "max_score", "n_comps", "keep_ids", "cov_sums",
    # what the global planner offered
    "junction_id", "lock_live", "n_active_dirs",
    # the decision
    "src", "goal_img_x", "goal_img_y", "goal_x", "goal_y",
    "dist_px", "from_heading_deg", "turn_from_prev_deg", "mode",
    # frontier bookkeeping
    "enroute_x", "enroute_y", "enroute_consumed", "committed_x", "committed_y",
    "snap_delta_px",
    # reachability of the goal actually issued (needs DECISION_LOG_ASTAR)
    "astar_px", "detour_ratio",
    # which guards fired since the previous decision
    "f_preempt", "f_early_stop", "f_no_uturn", "f_force_fallback",
    "f_stick", "f_commit", "f_nudge",
]


class PlanningMixin:
    """Prediction-thread behaviour. Mixed into UpdatedGoalBasedAutonomousExplorer."""

    def _transform_world_to_image_goal(self, world_pos: Tuple[float, float],
                                       robot_pose: Tuple[float, float, float],
                                       image_size: int) -> Tuple[int, int]:
        """World coordinates -> image coordinates. Inverse of the transform in
        global_priority_planner."""
        center_x, center_y = image_size // 2, image_size // 2
        robot_x, robot_y, robot_theta = robot_pose

        dx_world = world_pos[0] - robot_x
        dy_world = world_pos[1] - robot_y

        rot_angle_deg = 90.0 - (robot_theta * 180.0 / pi)
        rot_angle_rad_inv = rot_angle_deg * pi / 180.0

        cos_rot = np.cos(rot_angle_rad_inv)
        sin_rot = np.sin(rot_angle_rad_inv)

        dx_img = dx_world * cos_rot + dy_world * sin_rot
        dy_img = -dx_world * sin_rot + dy_world * cos_rot

        return (int(center_x + dx_img), int(center_y + dy_img))

    # ================================================================= [DBGLOG]
    #   DEBUG_LOG        (default True)  prose log, on/off
    #   DEBUG_LOG_IDLE   (default False) also log routine idle passes
    #   DEBUG_LOG_PATH   (default <output_dir>/prediction_debug.log)
    #   DECISION_LOG     (default False) structured per-decision csv
    #   DECISION_LOG_ASTAR (default False) add astar_px / detour_ratio
    # ========================================================================
    def _dbg_log(self):
        """Lazily create a file logger, one per resolved path."""
        lg = getattr(self, '_dbg_logger', None)
        if lg is not None:
            return lg
        path = getattr(self.config, 'DEBUG_LOG_PATH', None)
        if not path:
            outdir = getattr(self.config, 'output_dir', '.') or '.'
            try:
                os.makedirs(outdir, exist_ok=True)
            except Exception:
                outdir = '.'
            path = os.path.join(outdir, 'prediction_debug.log')
        # Name the logger after the resolved PATH, not id(self). CPython reuses
        # id() once an object is collected, and a sweep worker runs several modes
        # in sequence -- so the second explorer could match the first's id, find
        # lg.handlers already populated, skip adding its own handler, and
        # silently write into the PREVIOUS run's file.
        lg = logging.getLogger('intelx.pred.%s' % os.path.abspath(path))
        lg.setLevel(logging.DEBUG)
        lg.propagate = False
        if not lg.handlers:
            try:
                fh = logging.FileHandler(path, mode='a')
            except Exception:
                fh = logging.StreamHandler()
            fh.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', '%H:%M:%S'))
            lg.addHandler(fh)
        self._dbg_logger = lg
        self._dbg_log_path = path
        return lg

    def _dbg(self, msg):
        """One line to the prose debug log (no-op if DEBUG_LOG is off)."""
        if not getattr(self.config, 'DEBUG_LOG', True):
            return
        try:
            self._dbg_log().debug(msg)
        except Exception:
            pass

    # ---------------------------------------------------------------- [DECLOG]
    def _dec_flag(self, name):
        """Record that a guard fired during THIS pass.

        Flags are reset at the top of every pass and written out with the
        decision row, so the csv answers 'did preempt fire before this goal was
        chosen' without anyone having to correlate timestamps across log lines.
        """
        try:
            self._dec_flags.add(name)
        except Exception:
            self._dec_flags = set([name])

    def _dec_writer(self):
        """Lazily open the per-decision csv next to the prose log.

        One file per run: <same stem as the prose log>.decisions.csv, or
        <output_dir>/decisions.csv when DEBUG_LOG_PATH is not set. Header is
        written once, and the handle is flushed per row so a killed run keeps
        everything decided so far.
        """
        w = getattr(self, '_dec_w', None)
        if w is not None:
            return w
        path = getattr(self.config, 'DECISION_LOG_PATH', None)
        if not path:
            base = getattr(self.config, 'DEBUG_LOG_PATH', None)
            if base:
                path = os.path.splitext(base)[0] + '.decisions.csv'
            else:
                outdir = getattr(self.config, 'output_dir', '.') or '.'
                path = os.path.join(outdir, 'decisions.csv')
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            new = (not os.path.exists(path)) or os.path.getsize(path) == 0
            fh = open(path, 'a', newline='')
            w = csv.DictWriter(fh, fieldnames=DECISION_COLUMNS,
                               extrasaction='ignore')
            if new:
                w.writeheader()
            self._dec_fh = fh
            self._dec_w = w
            self._dec_path = path
            print("PREDICTION: decision log -> %s" % path)
        except Exception as e:
            print("PREDICTION: could not open decision log: %s" % e)
            self._dec_w = False          # False = tried and failed, do not retry
            return False
        return w

    def _dec_row(self, **kw):
        """Write one row. Never raises: logging must not break the run."""
        if not getattr(self.config, 'DECISION_LOG', False):
            return
        try:
            w = self._dec_writer()
            if not w:
                return
            row = dict((c, "") for c in DECISION_COLUMNS)
            row.update(kw)
            w.writerow(row)
            self._dec_fh.flush()
        except Exception:
            pass

    def _prof(self, name):
        """[PROFILE] Context manager timing a named section."""
        acc = getattr(self, '_prof_acc', None)
        if acc is None or not getattr(self.config, 'PROFILE', True):
            return _NULLCTX
        return _Timer(acc, name)

    def _prof_dump(self):
        """[PROFILE] Per-section breakdown to the debug log; resets the window."""
        if not getattr(self.config, 'PROFILE', True):
            return
        acc = getattr(self, '_prof_acc', None)
        if not acc:
            return
        wall = max(1e-9, time.perf_counter() - getattr(self, '_prof_t0', time.perf_counter()))
        passes = max(1, getattr(self, '_prof_passes', 0))
        items = sorted(acc.items(), key=lambda kv: kv[1][0], reverse=True)
        self._dbg("-" * 92)
        self._dbg(f"[PROFILE] {passes} passes / {wall:.1f}s wall = {wall * 1000.0 / passes:.0f} ms/pass "
                  f"(wait% = blocked on GIL/GPU, not computing; nested timers overlap):")
        for name, e in items:
            tot = e[0]
            calls = e[1]
            cpu = e[2] if len(e) > 2 else 0.0
            waitp = 100.0 * max(0.0, tot - cpu) / tot if tot > 0 else 0.0
            self._dbg(f"    {name:16s} wall={tot:7.2f}s cpu={cpu:7.2f}s wait={waitp:4.0f}%  "
                      f"calls={calls:6d} wall/call={tot * 1000.0 / max(1, calls):7.1f}ms "
                      f"cpu/call={cpu * 1000.0 / max(1, calls):7.1f}ms  wall_share={100.0 * tot / wall:4.0f}%")
        self._dbg("-" * 92)
        self._prof_acc = {}
        self._prof_t0 = time.perf_counter()
        self._prof_passes = 0

    def _ang_from_heading_deg(self, wx, wy, pose):
        """Signed angle (deg) of a world point relative to the robot's heading.
        ~0 = straight ahead, +-180 = directly behind."""
        a = np.degrees(np.arctan2(wy - pose[1], wx - pose[0]) - pose[2])
        return (a + 180.0) % 360.0 - 180.0

    def _img_to_world_dbg(self, px, pose):
        """image goal -> world, tolerant wrapper for logging only."""
        try:
            return _transform_image_goal_to_world(
                (int(px[0]), int(px[1])), tuple(pose),
                self.config.sim.get('WIN_PX', 400), self.config.sim.get('RESOLUTION', 0.05))
        except Exception:
            return (float('nan'), float('nan'))
    # ============================================================= end [DBGLOG]

    # ================================================================= [CHURN]
    # A goal that lands on the robot is reached instantly; a goal in a wall is
    # rejected instantly. Either way the simulator immediately re-requests, so
    # the goal count spikes while the robot never moves.
    #   Config: MIN_GOAL_ADVANCE_PX = 15.0
    # ========================================================================
    def _goal_is_degenerate(self, goal_px, current_pose):
        """True if goal_px sits on the robot or in a wall / OOB. Free (0) and
        UNKNOWN (-1) cells are NOT flagged."""
        if goal_px is None:
            return True
        WIN = self.config.sim.get('WIN_PX', 400)
        RES = self.config.sim.get('RESOLUTION', 0.05)
        c0 = WIN // 2
        min_px = float(getattr(self.config, 'MIN_GOAL_ADVANCE_PX', 15.0))
        if np.hypot(goal_px[0] - c0, goal_px[1] - c0) < min_px:
            return True                                   # essentially on the robot
        wx, wy = _transform_image_goal_to_world(goal_px, tuple(current_pose), WIN, RES)
        occ = self.simulator.occ
        H, W = occ.shape
        xi, yi = int(wx), int(wy)
        if not (0 <= xi < W and 0 <= yi < H):
            return True                                   # out of bounds
        if occ[yi, xi] == 1:
            return True                                   # inside a wall
        return False

    def _nudge_to_free(self, goal_px, current_pose, max_px=120.0):
        """March OUTWARD from the robot along the degenerate goal's heading and
        return the first non-wall image goal at >= MIN_GOAL_ADVANCE_PX."""
        WIN = self.config.sim.get('WIN_PX', 400)
        RES = self.config.sim.get('RESOLUTION', 0.05)
        c0 = WIN // 2
        min_px = float(getattr(self.config, 'MIN_GOAL_ADVANCE_PX', 15.0))
        occ = self.simulator.occ
        H, W = occ.shape
        dx, dy = (goal_px[0] - c0, goal_px[1] - c0) if goal_px is not None else (0.0, 0.0)
        d = np.hypot(dx, dy)
        ux, uy = (dx / d, dy / d) if d > 1e-6 else (0.0, -1.0)  # on-robot -> image "up"
        dist = min_px
        while dist <= max_px:
            gx, gy = c0 + ux * dist, c0 + uy * dist
            wx, wy = _transform_image_goal_to_world((int(gx), int(gy)), tuple(current_pose), WIN, RES)
            xi, yi = int(wx), int(wy)
            if 0 <= xi < W and 0 <= yi < H and occ[yi, xi] != 1:  # free or unknown, not wall
                return (int(gx), int(gy))
            dist += 4.0
        return None
    # =============================================================== end [CHURN]

    def _to_input(self, origin_img):
        """[SPEED-TRANSFORM] uint8 HxW image -> normalised (1,1,H,W) tensor on device.

        Replaces the PIL+torchvision path, which measured ~125 ms of CPU per pass
        (the GPU forward itself is ~7 ms). Self-validates against self.transform
        on first use and falls back permanently if they disagree, so this can
        never silently change what the model sees.
        Config: FAST_TRANSFORM (default True).
        """
        dev = self.config.device
        if not getattr(self.config, 'FAST_TRANSFORM', True) or getattr(self, '_fast_tf_bad', False):
            return self.transform(Image.fromarray(origin_img)).unsqueeze(0).to(dev)

        mean = float(getattr(self.config, 'INPUT_MEAN', 0.5))
        std = float(getattr(self.config, 'INPUT_STD', 0.5))
        t = torch.from_numpy(np.ascontiguousarray(origin_img))
        fast = t.to(dev, non_blocking=True).float().div_(255.0).sub_(mean).div_(std)
        fast = fast.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)

        if not getattr(self, '_fast_tf_checked', False):
            self._fast_tf_checked = True
            try:
                ref = self.transform(Image.fromarray(origin_img)).unsqueeze(0).to(dev)
                if ref.shape != fast.shape:
                    raise ValueError(f"shape {tuple(fast.shape)} != reference {tuple(ref.shape)}")
                diff = float((ref - fast).abs().max())
                if diff > 1e-4:
                    raise ValueError(f"max abs diff {diff:.2e} > 1e-4")
                print(f"PREDICTION: [SPEED-TRANSFORM] fast on-device transform validated "
                      f"(max diff {diff:.2e}).")
                self._dbg(f"[SPEED-TRANSFORM] validated, max diff {diff:.2e}")
            except Exception as e:
                self._fast_tf_bad = True
                print(f"PREDICTION: [SPEED-TRANSFORM] DISABLED -- fast path differs from "
                      f"self.transform ({e}). Using the original transform.")
                self._dbg(f"[SPEED-TRANSFORM] disabled: {e}")
                return self.transform(Image.fromarray(origin_img)).unsqueeze(0).to(dev)
        return fast

    def _astar_windowed(self, a_px, b_px):
        """[SPEED-ASTAR] A* restricted to a bounding box around a and b.

        A* cost scales with the SEARCHED AREA, and a full-grid search on this map
        costs ~1-2 s per call. Almost every query here is between two points a
        few hundred px apart. If the optimal route leaves the window the crop
        search fails and we fall back to the FULL map, so no path is ever lost.
        ASTAR_WINDOW_MARGIN_PX = 0 disables the window entirely.
        """
        occ = self.simulator.occ
        H, W = occ.shape
        margin = float(getattr(self.config, 'ASTAR_WINDOW_MARGIN_PX', 150.0))
        ax, ay = int(a_px[0]), int(a_px[1])
        bx, by = int(b_px[0]), int(b_px[1])

        if margin > 0:
            x0 = max(0, min(ax, bx) - int(margin))
            x1 = min(W, max(ax, bx) + int(margin) + 1)
            y0 = max(0, min(ay, by) - int(margin))
            y1 = min(H, max(ay, by) + int(margin) + 1)
            if (x1 - x0) * (y1 - y0) < 0.6 * (H * W):
                sub = occ[y0:y1, x0:x1]
                try:
                    # [FAST-ASTAR] Both callers of _astar_windowed want a NUMBER
                    # (a summed length, or a waypoint count), never a drivable
                    # path -- so shortcutting, smoothing and subsampling are pure
                    # waste, and they dominate the 26-56 s a fallback decision was
                    # taking. Subsampling also capped the result at 50 waypoints,
                    # which silently broke the prepass's `len(path)` ranking for
                    # exactly the long paths it mattered for.
                    p = optimized_astar_path_planner(
                        sub, (ax - x0, ay - y0), (bx - x0, by - y0),
                        skip_postprocess=True)
                    if p and len(p) >= 2:
                        try:
                            self.diag['astar_windowed'] = self.diag.get('astar_windowed', 0) + 1
                        except Exception:
                            pass
                        return [(pt[0] + x0, pt[1] + y0) for pt in p]
                except Exception:
                    pass
                try:
                    self.diag['astar_window_miss'] = self.diag.get('astar_window_miss', 0) + 1
                except Exception:
                    pass
        return optimized_astar_path_planner(occ, (ax, ay), (bx, by),
                                            skip_postprocess=True)   # [FAST-ASTAR]

    def _astar_len(self, a_px, b_px):
        """TRUE A* path length in pixels, or None if the planner did not reach
        the goal.

        The old version returned len(path) -- the WAYPOINT COUNT -- and
        optimized_astar_path_planner returns a short truncated path when the goal
        is far or unreachable, which made a cross-map frontier look like the
        'nearest' one. Now: sum of segment lengths; reject if the path does not
        END at the goal; reject if impossibly shorter than the straight line.
        Config: ASTAR_END_TOL_PX (default 20).
        """
        try:
            a = (int(a_px[0]), int(a_px[1]))
            b = (int(b_px[0]), int(b_px[1]))
            p = self._astar_windowed(a, b)
            if not p or len(p) < 2:
                return None
            end = p[-1]
            tol = float(getattr(self.config, 'ASTAR_END_TOL_PX', 20.0))
            if np.hypot(end[0] - b[0], end[1] - b[1]) > tol:
                return None                                   # planner didn't reach goal
            seg = 0.0
            for i in range(len(p) - 1):
                seg += np.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1])
            euclid = np.hypot(b[0] - a[0], b[1] - a[1])
            if seg < euclid - 1.0:
                return None                                   # impossibly short -> bogus
            return float(seg)
        except Exception:
            return None

    def _frontier_boundary_point(self, f_px, search_px=None):
        """Snap a frontier target onto a REAL frontier cell.

        Frontier cell = KNOWN-FREE (explored & occ==0) with at least one
        not-yet-explored 8-neighbour. The returned point is the MIDPOINT of the
        nearest such segment: the cell closest to that segment's own mean.

        THIS NOW ALWAYS SNAPS. It used to return early for any point that was
        not a wall, which let a cluster-MEAN centroid through untouched -- and
        that centroid is what the fallback drove to and what _frontier_consumed
        was then asked about. On a frontier that wraps around a room the mean
        sits ~2 m from the nearest frontier cell, so BOTH the goal and the
        consumption test were operating on a point that is not on the frontier
        at all. Verified on that case: raw centroid (120,120) is not a frontier
        cell; the snapped target (119,80) is, and the consumption test then reads
        live -> False, explored away -> True, where the raw centroid gave True in
        both.

        Config: FRONTIER_SNAP_SEARCH_PX (default 100). Raised from 60 because
        the search must now reach the cells of a wrap-around cluster from its
        mean, not merely escape a wall.
        """
        occ = self.simulator.occ
        explored = self.simulator.explored
        H, W = occ.shape
        fx, fy = int(round(f_px[0])), int(round(f_px[1]))

        if search_px is None:
            search_px = int(getattr(self.config, 'FRONTIER_SNAP_SEARCH_PX', 100))

        x0, x1 = max(0, fx - search_px), min(W, fx + search_px + 1)
        y0, y1 = max(0, fy - search_px), min(H, fy + search_px + 1)
        if x1 <= x0 or y1 <= y0:
            return (float(f_px[0]), float(f_px[1]))

        occ_w = occ[y0:y1, x0:x1]
        exp_w = explored[y0:y1, x0:x1].astype(bool)
        known_free = exp_w & (occ_w == 0)
        # 8-neighbour presence of an unexplored cell. BORDER_CONSTANT 0 so the
        # crop edge does not fabricate a frontier.
        unexplored = (~exp_w).astype(np.uint8)
        neigh = cv2.dilate(unexplored, np.ones((3, 3), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        frontier = known_free & neigh

        cx_w, cy_w = fx - x0, fy - y0
        ys, xs = np.where(frontier)
        if len(xs) == 0:
            # No frontier boundary nearby -> nearest known-free cell (still a
            # real, reachable cell), else give up and return the point unchanged.
            yk, xk = np.where(known_free)
            if len(xk) == 0:
                return (float(f_px[0]), float(f_px[1]))
            i = int(np.argmin((xk - cx_w) ** 2 + (yk - cy_w) ** 2))
            return (float(xk[i] + x0), float(yk[i] + y0))

        # Cluster the frontier cells; take the segment nearest the input point,
        # then that segment's midpoint (the cell nearest its own mean).
        mask = np.zeros(frontier.shape, np.uint8)
        mask[ys, xs] = 1
        n_lab, labels = cv2.connectedComponents(mask, connectivity=8)
        best_seg = None
        best_d = None
        for lab in range(1, n_lab):
            sy, sx = np.where(labels == lab)
            d2 = float(np.min((sx - cx_w) ** 2 + (sy - cy_w) ** 2))
            if best_d is None or d2 < best_d:
                best_d = d2
                best_seg = (sx, sy)
        sx, sy = best_seg
        mx, my = float(sx.mean()), float(sy.mean())
        j = int(np.argmin((sx - mx) ** 2 + (sy - my) ** 2))
        return (float(sx[j] + x0), float(sy[j] + y0))

    def _frontier_consumed(self, f_px):
        """Is there still an actual FRONTIER at this point?

        A frontier CELL is known-free (explored AND occ == 0) with at least one
        unexplored 8-neighbour -- the same definition _frontier_boundary_point
        and the Tier-0 baselines use.

        THE PREVIOUS VERSION ASKED THE WRONG QUESTION: whether the box around the
        point was explored. The point handed to it is a frontier CENTROID, the
        MEAN of a cluster of frontier cells, and every one of those cells is
        known-free -- so the mean sits in known-free space and the box was
        explored almost always. It returned True for essentially every LIVE
        frontier, which meant early-stop fired the instant any fallback trip
        began (after the previous patch this predicate WAS its whole condition)
        and the frontier commitment, gated on `not _frontier_consumed(...)`,
        never once kept a target.

        This is only reliable because the caller now snaps its target onto a real
        frontier cell first. On a frontier wrapping around a room the centroid
        can sit 2 m from the nearest frontier cell, so no radius test on the raw
        centroid works at any radius.
        """
        try:
            explored = self.simulator.explored
            occ = self.simulator.occ
            H, W = explored.shape
            res = self.config.sim.get('RESOLUTION', 0.05)
            r = max(3, int(round(1.0 / res)))
            x, y = int(f_px[0]), int(f_px[1])
            if not (0 <= x < W and 0 <= y < H):
                return True                      # off-map: nothing to go to
            x0, x1 = max(0, x - r), min(W, x + r + 1)
            y0, y1 = max(0, y - r), min(H, y + r + 1)
            exp = explored[y0:y1, x0:x1].astype(bool)
            if exp.size == 0:
                return True
            known_free = exp & (occ[y0:y1, x0:x1] == 0)
            if not known_free.any():
                return True                      # nothing to stand on here
            neigh = cv2.dilate((~exp).astype(np.uint8), np.ones((3, 3), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
            return not bool((known_free & neigh).any())
        except Exception:
            return False                         # unsure -> keep going

    # =============================================================== [NO-UTURN]
    # If the local goal points backward and no forward coverage exists, route to
    # a real FRONTIER instead of U-turning on a weak lobe.
    # Config: ANTI_UTURN (default True), UTURN_DEG (default 110).
    # ========================================================================
    def _is_backward(self, goal_px, pose):
        """True if goal_px points more than UTURN_DEG from the robot's heading."""
        WIN = self.config.sim.get('WIN_PX', 400)
        RES = self.config.sim.get('RESOLUTION', 0.05)
        wx, wy = _transform_image_goal_to_world(
            (int(goal_px[0]), int(goal_px[1])), tuple(pose), WIN, RES)
        d = abs((np.arctan2(wy - pose[1], wx - pose[0]) - pose[2] + np.pi) % (2 * np.pi) - np.pi)
        return np.degrees(d) > float(getattr(self.config, 'UTURN_DEG', 110.0))

    def _strongest_forward(self, keep_ids, centroids, pose):
        """Strongest local component within UTURN_DEG of the heading, or None."""
        WIN = self.config.sim.get('WIN_PX', 400)
        RES = self.config.sim.get('RESOLUTION', 0.05)
        ut = np.radians(float(getattr(self.config, 'UTURN_DEG', 110.0)))
        for cc in keep_ids:
            if cc not in centroids:
                continue
            cx, cy = centroids[cc]
            wx, wy = _transform_image_goal_to_world((int(cx), int(cy)), tuple(pose), WIN, RES)
            d = abs((np.arctan2(wy - pose[1], wx - pose[0]) - pose[2] + np.pi) % (2 * np.pi) - np.pi)
            if d <= ut:
                return (int(cx), int(cy))
        return None
    # =========================================================== end [NO-UTURN]

    def _sticky_component(self, keep_ids, local_components, current_pose):
        """Which component to follow, preferring the one chosen last pass.

        keep_ids is re-sorted by coverage sum every pass and [0] is taken, so two
        components with near-equal sums swap rank between passes and the goal
        alternates between two directions -- 30-67% of all goal-direction
        reversals in the logs, firing with no guard involved.

        cc_id is a peak index and is not stable across passes, so the previous
        choice is remembered by WORLD POSITION. If a component still sits within
        STICK_TOL_M of it, keep that one unless a rival's coverage exceeds it by
        STICK_MARGIN.
        """
        if not keep_ids:
            return None
        best = keep_ids[0]
        prev = getattr(self, '_last_comp_world', None)
        if prev is None or len(keep_ids) < 2:
            return best
        try:
            sums = local_components.get('cov_sums') or {}
            if not sums:
                return best
            WIN = self.config.sim.get('WIN_PX', 400)
            RES = self.config.sim.get('RESOLUTION', 0.05)
            cents = local_components.get('centroids', {})
            tol = float(getattr(self.config, 'STICK_TOL_M', 2.0)) / RES
            margin = float(getattr(self.config, 'STICK_MARGIN', 0.25))
            best_sum = float(sums.get(best, 0.0))
            for cc in keep_ids:
                if cc not in cents:
                    continue
                cx, cy = cents[cc]
                wx, wy = _transform_image_goal_to_world(
                    (int(cx), int(cy)), tuple(current_pose), WIN, RES)
                if np.hypot(wx - prev[0], wy - prev[1]) > tol:
                    continue
                if float(sums.get(cc, 0.0)) >= best_sum * (1.0 - margin):
                    if cc != best:
                        self._dec_flag('stick')
                        self._dbg("    [STICK] keeping cc%d (cov %.4f) over cc%d "
                                  "(cov %.4f) -- within %.1fm of last choice"
                                  % (cc, sums.get(cc, 0.0), best, best_sum,
                                     float(getattr(self.config, 'STICK_TOL_M', 2.0))))
                    return cc
                break
        except Exception:
            pass
        return best

    def _simple_prime_goal_history(self):
        """Simple goal history initialization."""
        try:
            print("Simple goal history priming...")
            for i in range(self.goal_history.maxlen):
                default_goals = [(200, 200), (180, 220), (220, 180), (200, 180), (180, 200)]
                self.goal_history.append(default_goals[i % len(default_goals)])
            print(f"Simple goal history primed with {len(self.goal_history)} default goals.")
        except Exception as e:
            print(f"Error priming goal history: {e}")

    def plan_local(self, heatmap, current_pose):
        """[B] Local planner: current heatmap -> components."""
        H, W = heatmap.shape
        img_cy, img_cx = H // 2, W // 2  # robot center (image IS robot-centric)

        # 1. Normalize raw heatmap
        h = heatmap.astype(np.float32)
        h_min, h_max = float(h.min()), float(h.max())
        h_norm = (h - h_min) / (h_max - h_min + 1e-6)

        # 2. Smooth + threshold to get a stable binary mask
        h_smooth = gaussian_filter(h_norm, sigma=smooth_sigma)
        hs_min, hs_max = float(h_smooth.min()), float(h_smooth.max())
        h_smooth_norm = (h_smooth - hs_min) / (hs_max - hs_min + 1e-6)
        thr = np.percentile(h_smooth_norm, top_percentile)
        binary = (h_smooth_norm >= thr)

        # 3. Per-pixel polar coordinates from the robot centre
        ys_all, xs_all = np.mgrid[0:H, 0:W]
        angle_map = np.arctan2(
            (ys_all - img_cy).astype(np.float32),
            (xs_all - img_cx).astype(np.float32)
        )  # -pi to pi

        # 4. Angular histogram: weighted by h_norm, only within the mask
        h_in_mask = np.where(binary, h_norm, 0.0)
        angle_bin_edges = np.linspace(-np.pi, np.pi, n_angle_bins + 1)
        angular_hist, _ = np.histogram(angle_map, bins=angle_bin_edges, weights=h_in_mask)

        # 5. Circular smooth (tile 3x to handle wrap-around)
        angular_hist_norm = angular_hist / (angular_hist.max() + 1e-9)
        tiled = np.tile(angular_hist_norm, 3)
        tiled_smooth = gaussian_filter(tiled, sigma=float(angular_smooth_k))
        angular_hist_smooth = tiled_smooth[n_angle_bins: 2 * n_angle_bins]

        # 6. Angular peaks.
        # [STICK] 72 bins over 360 deg is 5 deg per bin, so the legacy default of
        # 2 is TEN degrees, not the ~25 its comment claimed. At 10 deg the
        # histogram splits into near-duplicate components with near-equal
        # coverage sums, which is what makes the ranking flip between passes and
        # the goal alternate. MIN_ARC_GAP_BINS = 5 is the documented intent.
        peaks, _ = find_peaks(
            angular_hist_smooth,
            height=peak_height_thr,
            distance=int(getattr(self.config, 'MIN_ARC_GAP_BINS', 5))
        )
        if len(peaks) == 0 and angular_hist_smooth.max() > 1e-9:
            peaks = np.array([int(np.argmax(angular_hist_smooth))])
        if len(peaks) > max_components:
            peak_vals = angular_hist_smooth[peaks]
            peaks = peaks[np.argsort(peak_vals)[::-1][:max_components]]

        bin_centers = 0.5 * (angle_bin_edges[:-1] + angle_bin_edges[1:])
        peak_angles = bin_centers[peaks]  # radians

        # 7. Angular Voronoi: assign each masked pixel to the nearest peak angle
        label_map = np.zeros((H, W), dtype=np.uint16)
        keep_ids = []
        centroids = {}
        _cov_sums = {}                     # [STICK] cc_id -> coverage sum

        if len(peak_angles) > 0:
            ys_m, xs_m = np.where(binary)
            pix_angles = angle_map[ys_m, xs_m]

            diff = pix_angles[:, None] - peak_angles[None, :]
            diff = (diff + np.pi) % (2 * np.pi) - np.pi
            nearest_peak_idx = np.argmin(np.abs(diff), axis=1)

            for k in range(len(peak_angles)):
                cc_id = k + 1
                member = (nearest_peak_idx == k)
                if not member.any():
                    continue

                comp_ys = ys_m[member]
                comp_xs = xs_m[member]

                if len(comp_ys) < min_area_px:
                    continue

                norm_vals = h_norm[comp_ys, comp_xs]
                raw_vals = heatmap[comp_ys, comp_xs]

                # Gate 1: relative peak (scale-invariant noise filter)
                if float(norm_vals.max()) < rel_peak_thr:
                    continue

                # Gate 2: majority absolute coverage threshold
                if float((raw_vals > coverage_thr).sum()) / len(raw_vals) < coverage_majority:
                    continue

                weights = raw_vals.clip(0)
                total_w = weights.sum()
                if total_w > 1e-9:
                    cx_c = float((comp_xs * weights).sum() / total_w)
                    cy_c = float((comp_ys * weights).sum() / total_w)
                else:
                    cx_c, cy_c = float(comp_xs.mean()), float(comp_ys.mean())

                label_map[comp_ys, comp_xs] = cc_id
                keep_ids.append(cc_id)
                centroids[cc_id] = (cx_c, cy_c)

            def get_coverage_sum(cc_id):
                ys, xs = np.where(label_map == cc_id)
                return float(heatmap[ys, xs].sum())

            # [STICK] keep the sums: the arbitration needs them to decide whether
            # a rival component is enough better to justify turning.
            _cov_sums = dict((cc, get_coverage_sum(cc)) for cc in keep_ids)
            keep_ids.sort(key=lambda c: _cov_sums[c], reverse=True)

        local_components = {
            "label_map": label_map,
            "keep_ids": keep_ids,
            "centroids": centroids,
            "cov_sums": _cov_sums,
            "peak_angles_deg": [round(float(a * 180 / np.pi), 1) for a in peak_angles],
            "angular_hist": angular_hist_smooth.tolist(),
            "thr": float(thr),
            "detected_doorways": [],
        }
        return local_components

    def plan_local_centroid(self, heatmap):
        """PURE LOCAL POLICY: goal = intensity-weighted centroid of the coverage
        field. The ablation baseline -- no junctions, no global planner.

        Knobs: LOCAL_CENTROID_PERCENTILE (default 0 = use everything),
               LOCAL_CENTROID_WEIGHT_POWER (default 1.0).
        """
        H, W = heatmap.shape
        img_cx, img_cy = W // 2, H // 2

        w = np.clip(heatmap.astype(np.float64), 0.0, None)

        pct = float(getattr(self.config, 'LOCAL_CENTROID_PERCENTILE', 0.0))
        if pct > 0.0:
            thr = np.percentile(w, pct)
            w = np.where(w >= thr, w, 0.0)

        power = float(getattr(self.config, 'LOCAL_CENTROID_WEIGHT_POWER', 1.0))
        if power != 1.0:
            w = w ** power

        total = float(w.sum())
        if total <= 1e-12:
            gx, gy = img_cx, img_cy          # no signal at all -> stay put
        else:
            ys, xs = np.mgrid[0:H, 0:W]
            gx = float((xs * w).sum() / total)
            gy = float((ys * w).sum() / total)

        goal_px = (int(round(gx)), int(round(gy)))
        info = {'centroid_img': (gx, gy), 'total_weight': total,
                'percentile': pct, 'power': power}
        return goal_px, info

    def plan_global(self, heatmap, current_pose, local_components, allow_prepass_refresh=True):
        """[D]+[E]+[F] Global/junction planner.

        Returns (chosen_goal_px, active_directions, frontier_centroids_for_viz).
        chosen_goal_px is None when no junction lock applies.
        """
        label_map = local_components.get("label_map")

        def get_coverage_sum(cc_id):
            ys, xs = np.where(label_map == cc_id)
            return float(heatmap[ys, xs].sum())

        # --- 0. PRE-COMPUTE ACTIVE BRANCH MAP & DIRECTIONS (cached) ---
        # [PREPASS-CACHE] find_global_frontiers + up to ~12 A* calls, and each A*
        # costs ~1 s on a large partially-explored map, so this dominates the
        # prediction thread. The result changes slowly, so recompute only every
        # PREPASS_EVERY_PX of motion or when a node is added, and serve a cache
        # otherwise. The only per-pass-varying field, `dist`, is refreshed cheaply.
        active_branch_map = {}
        frontier_centroids_for_viz = []
        active_directions = []

        prepass_every = float(getattr(self.config, 'PREPASS_EVERY_PX', 20.0))
        _cache = getattr(self, '_prepass_cache', None)
        with self.nodes_lock:
            _n_nodes = len(self.exploration_nodes)
        _need_refresh = (
            _cache is None
            or prepass_every <= 0
            or _cache.get('n_nodes') != _n_nodes
            or np.hypot(current_pose[0] - _cache['pose'][0],
                        current_pose[1] - _cache['pose'][1]) >= prepass_every
        )
        # [IDLE-CHEAP] Idle passes never return a goal, so they serve the cache
        # rather than recomputing it; the refresh happens on the next nav pass.
        if not allow_prepass_refresh and _cache is not None:
            _need_refresh = False

        if _need_refresh:
            try:
                self.diag['prepass_refresh'] = self.diag.get('prepass_refresh', 0) + 1
            except Exception:
                pass
            _prep_t0 = time.perf_counter()
            _prep_c0 = _THREAD_TIME()
            _snap_delta_max = 0.0
            try:
                with self.nodes_lock:
                    pre_valid_nodes = copy.deepcopy(self.exploration_nodes)

                if pre_valid_nodes:
                    curr_px_pre = (current_pose[0], current_pose[1])
                    pre_frontier_data = self.global_planner.find_global_frontiers(
                        self.simulator.occ, self.simulator.explored, curr_px_pre
                    )
                    pre_frontier_pts = []
                    if pre_frontier_data:
                        _snap = getattr(self.config, 'SNAP_FRONTIER_OFF_OBSTACLE', True)
                        for f_item in pre_frontier_data:
                            fx, fy, _ = f_item['pose']
                            # ALWAYS snap onto a real frontier cell. The value
                            # returned by find_global_frontiers is a cluster MEAN
                            # and need not lie on the frontier at all; the goal we
                            # drive to and the point _frontier_consumed is asked
                            # about must both be an actual frontier cell.
                            if _snap:
                                _fx0, _fy0 = fx, fy
                                fx, fy = self._frontier_boundary_point((fx, fy))
                                _snap_delta_max = max(
                                    _snap_delta_max,
                                    float(np.hypot(fx - _fx0, fy - _fy0)))
                            euc_dist = np.hypot(fx - curr_px_pre[0], fy - curr_px_pre[1])
                            pre_frontier_pts.append((float(fx), float(fy), euc_dist))
                        pre_frontier_pts.sort(key=lambda f: f[2])
                        pre_frontier_pts = pre_frontier_pts[:4]

                    frontier_centroids_for_viz = [(fp[0], fp[1]) for fp in pre_frontier_pts]

                    all_branches_pre = []
                    for node in pre_valid_nodes:
                        for comp in node.get('components', []):
                            all_branches_pre.append({
                                'node_id': node['id'], 'rank': comp['rank'],
                                'x': comp['x'], 'y': comp['y'],
                            })

                    for fx, fy, _ in pre_frontier_pts:
                        f_px = (int(fx), int(fy))
                        branch_candidates = sorted(
                            all_branches_pre,
                            key=lambda b: np.hypot(b['x'] - fx, b['y'] - fy)
                        )[:3]
                        best_branch = None
                        min_cost = float('inf')
                        for branch in branch_candidates:
                            b_px = (int(branch['x']), int(branch['y']))
                            path = self._astar_windowed(b_px, f_px)
                            if path is not None and len(path) < min_cost:
                                min_cost = len(path)
                                best_branch = branch
                        if best_branch is not None:
                            key = (best_branch['node_id'], best_branch['rank'])
                            active_branch_map[key] = f_px

                    for node in pre_valid_nodes:
                        for comp in node.get('components', []):
                            key = (node['id'], comp['rank'])
                            if key in active_branch_map:
                                dist = np.hypot(comp['x'] - curr_px_pre[0],
                                                comp['y'] - curr_px_pre[1])
                                active_directions.append({
                                    'node_id': node['id'], 'rank': comp['rank'],
                                    'x': comp['x'], 'y': comp['y'], 'dist': dist,
                                    'frontier_px': active_branch_map[key]
                                })
            except Exception as e:
                print(f"PREDICTION: Pre-pass active branch map failed: {e}")

            self._last_snap_delta_px = _snap_delta_max

            _acc = getattr(self, '_prof_acc', None)
            if _acc is not None and getattr(self.config, 'PROFILE', True):
                _e = _acc.get('prepass_refresh')
                _dt = time.perf_counter() - _prep_t0
                _dc = _THREAD_TIME() - _prep_c0
                if _e is None:
                    _acc['prepass_refresh'] = [_dt, 1, _dc]
                else:
                    _e[0] += _dt
                    _e[1] += 1
                    _e[2] += _dc

            self._prepass_cache = {
                'pose': (float(current_pose[0]), float(current_pose[1])),
                'n_nodes': _n_nodes,
                'active_branch_map': active_branch_map,
                'active_directions': active_directions,
                'frontier_centroids': frontier_centroids_for_viz,
            }
        else:
            try:
                self.diag['prepass_cache_hit'] = self.diag.get('prepass_cache_hit', 0) + 1
            except Exception:
                pass
            active_branch_map = _cache['active_branch_map']
            frontier_centroids_for_viz = _cache['frontier_centroids']
            active_directions = []
            for d in _cache['active_directions']:
                d2 = dict(d)
                d2['dist'] = float(np.hypot(d['x'] - current_pose[0], d['y'] - current_pose[1]))
                active_directions.append(d2)

        # --- 1. Identify Junctions & Lock Direction ---
        if not hasattr(self, 'current_junction_id'):
            self.current_junction_id = None

        keep_ids = local_components.get('keep_ids', [])
        centroids = local_components.get('centroids', {})

        chosen_goal_px = None
        junction_lock_dist_m = float(getattr(self.config, 'JUNCTION_LOCK_DIST_M', 3.0))
        merge_dist_px = junction_lock_dist_m / self.config.sim.get('RESOLUTION', 0.05)

        curr_x, curr_y = current_pose[0], current_pose[1]
        active_node = None

        with self.nodes_lock:
            for node in self.exploration_nodes:
                if np.hypot(node['x'] - curr_x, node['y'] - curr_y) < merge_dist_px:
                    active_node = node
                    break

        self._last_lock_live = False

        # Case A: Inside known junction
        if active_node is not None:
            if self.current_junction_id != active_node['id']:
                self.current_junction_id = active_node['id']
                active_node['count'] += 1

                unexplored = [c for c in active_node['components'] if not c.get('explored', False)]
                if unexplored:
                    unexplored.sort(key=lambda c: c['rank'], reverse=True)

                    # Only pick this branch if a frontier centroid claims it.
                    chosen_comp = None
                    for candidate in unexplored:
                        key = (active_node['id'], candidate['rank'])
                        if key in active_branch_map:
                            chosen_comp = candidate
                            break

                    if chosen_comp is None:
                        print(f"PREDICTION: Junction {active_node['id']}: "
                              f"all unexplored branches are frontier-free. Skipping.")
                        best_comp = None
                        min_diff = float('inf')
                        for comp in active_node['components']:
                            comp_angle = np.arctan2(comp['y'] - active_node['y'],
                                                    comp['x'] - active_node['x'])
                            diff = abs((comp_angle - current_pose[2] + np.pi) % (2 * np.pi) - np.pi)
                            if diff < min_diff:
                                min_diff = diff
                                best_comp = comp
                        if best_comp:
                            active_node['locked_angle'] = np.arctan2(
                                best_comp['y'] - active_node['y'],
                                best_comp['x'] - active_node['x'])
                        # [LOCK] nothing was committed to -> let the local planner
                        # run unconstrained here
                        active_node['locked_rank'] = None
                    else:
                        for c in active_node['components']:
                            if c['rank'] == chosen_comp['rank']:
                                c['explored'] = True

                        active_node['locked_angle'] = np.arctan2(
                            chosen_comp['y'] - active_node['y'],
                            chosen_comp['x'] - active_node['x'])
                        # [LOCK] remember WHICH branch this commitment is to, so
                        # the lock can be released when that branch's frontier is
                        # gone instead of persisting on mere proximity
                        active_node['locked_rank'] = chosen_comp['rank']
                        print(f"\nPREDICTION: Entered Junction {active_node['id']}. "
                              f"Selected unexplored Rank {chosen_comp['rank']}!")
                        self._dbg(f"    JUNCTION Node {active_node['id']} ENTER: selected "
                                  f"unexplored Rank {chosen_comp['rank']} "
                                  f"lock={np.degrees(active_node['locked_angle']):+.1f}deg")
                        for c in active_node['components']:
                            wdir = np.degrees(np.arctan2(c['y'] - active_node['y'],
                                                         c['x'] - active_node['x']))
                            self._dbg(f"      branch Rank{c['rank']} world=({c['x']:.1f},{c['y']:.1f}) "
                                      f"world_dir={wdir:+.1f}deg "
                                      f"from_heading={self._ang_from_heading_deg(c['x'], c['y'], current_pose):+.1f}deg "
                                      f"explored={c.get('explored', False)} "
                                      f"frontier={'Y' if (active_node['id'], c['rank']) in active_branch_map else 'n'}")
                else:
                    best_comp = None
                    min_diff = float('inf')
                    for comp in active_node['components']:
                        comp_angle = np.arctan2(comp['y'] - active_node['y'],
                                                comp['x'] - active_node['x'])
                        diff = abs((comp_angle - current_pose[2] + np.pi) % (2 * np.pi) - np.pi)
                        if diff < min_diff:
                            min_diff = diff
                            best_comp = comp
                    if best_comp:
                        active_node['locked_angle'] = np.arctan2(
                            best_comp['y'] - active_node['y'],
                            best_comp['x'] - active_node['x'])
                    # [LOCK] every branch explored -> nothing to commit to
                    active_node['locked_rank'] = None
                    print(f"\nPREDICTION: Passing through fully-explored Junction "
                          f"{active_node['id']}.")
                    self._dbg(f"    JUNCTION Node {active_node['id']} PASS-THROUGH "
                              f"(all explored): least-turning "
                              f"lock={np.degrees(active_node['locked_angle']):+.1f}deg "
                              f"robot_heading={np.degrees(current_pose[2]):+.1f}deg")

            # [LOCK] The lock is an ANGULAR CONSTRAINT on the local planner, not a
            # goal source, and it is held only while there is a live branch to
            # commit to. Released when locked_rank is None, or when that branch is
            # no longer in active_branch_map (its frontier is gone).
            locked_angle = active_node.get('locked_angle')
            locked_rank = active_node.get('locked_rank')
            _lock_live = (locked_rank is not None
                          and (active_node['id'], locked_rank) in active_branch_map)
            self._last_lock_live = bool(_lock_live)
            _tol_deg = float(getattr(self.config, 'LOCK_TOLERANCE_DEG', 45.0))

            frame_components = []
            for cc_id in keep_ids:
                cx_img, cy_img = centroids[cc_id]
                world_px = _transform_image_goal_to_world(
                    (int(cx_img), int(cy_img)), tuple(current_pose),
                    self.config.sim.get('WIN_PX', 400),
                    self.config.sim.get('RESOLUTION', 0.05)
                )
                frame_components.append({
                    'cc_id': cc_id, 'cx_img': cx_img, 'cy_img': cy_img,
                    'x': float(world_px[0]), 'y': float(world_px[1])
                })

            # keep_ids is strongest-first, so the FIRST component inside the
            # tolerance is the local planner's own top choice among those the
            # lock allows. This used to be argmin over ANGLE across all
            # components, which overrode the local ranking entirely.
            best_comp = None
            if _lock_live and locked_angle is not None and frame_components:
                _tol = np.radians(_tol_deg)
                for comp in frame_components:
                    comp_angle = np.arctan2(comp['y'] - curr_y, comp['x'] - curr_x)
                    if abs((comp_angle - locked_angle + np.pi) % (2 * np.pi) - np.pi) <= _tol:
                        best_comp = comp
                        break

            if best_comp is not None:
                chosen_goal_px = (int(best_comp['cx_img']), int(best_comp['cy_img']))

            if getattr(self.config, 'DEBUG_LOG', True) and frame_components:
                if not _lock_live:
                    self._dbg(f"    JUNCTION Node {active_node['id']} LOCK-RELEASED "
                              f"(locked_rank={locked_rank}; "
                              f"{'that branch has no frontier left' if locked_rank is not None else 'nothing was committed to'}) "
                              f"-> local planner unconstrained")
                else:
                    self._dbg(f"    JUNCTION Node {active_node['id']} FOLLOW-LOCK "
                              f"lock={np.degrees(locked_angle):+.1f}deg rank={locked_rank} "
                              f"tol={_tol_deg:.0f}deg ({len(frame_components)} local comps; "
                              f"lock FILTERS, local ranking picks):")
                    for comp in frame_components:
                        comp_angle = np.arctan2(comp['y'] - curr_y, comp['x'] - curr_x)
                        adiff = np.degrees(abs((comp_angle - locked_angle + np.pi) % (2 * np.pi) - np.pi))
                        ddist = np.hypot(comp['x'] - curr_x, comp['y'] - curr_y)
                        star = ' <== CHOSEN' if (best_comp is not None and comp is best_comp) else ''
                        tag = 'ok ' if adiff <= _tol_deg else 'REJ'
                        self._dbg(f"      [{tag}] cc{comp['cc_id']} world=({comp['x']:.1f},{comp['y']:.1f}) "
                                  f"angle_to_lock={adiff:.1f}deg dist={ddist:.1f}px "
                                  f"from_heading={self._ang_from_heading_deg(comp['x'], comp['y'], current_pose):+.1f}deg{star}")

        # Case B: Potential new junction -> N-Polygon Verification
        elif len(keep_ids) > 1:
            _jv_t0 = time.perf_counter()
            _jv_c0 = _THREAD_TIME()
            self.current_junction_id = None
            print("\n" + "=" * 60)
            print(f"JUNCTION_DBG: Potential junction at ({curr_x:.1f}, {curr_y:.1f})")
            print(f"JUNCTION_DBG: keep_ids from single frame = {keep_ids}")
            for kid in keep_ids:
                cov = get_coverage_sum(kid)
                cx_c, cy_c = centroids[kid]
                world_px = _transform_image_goal_to_world(
                    (int(cx_c), int(cy_c)), tuple(current_pose),
                    self.config.sim.get('WIN_PX', 400),
                    self.config.sim.get('RESOLUTION', 0.05))
                print(f"  Component {kid}: img_centroid=({cx_c:.1f},{cy_c:.1f})  "
                      f"world=({world_px[0]:.1f},{world_px[1]:.1f})  cov={cov:.3f}")
            print("=" * 60)
            print("JUNCTION_DBG: Running N-Polygon Verification...")

            N_points = 10
            radius_m = 0.5
            radius_px = radius_m / self.config.sim.get('RESOLUTION', 0.05)
            required_consensus = 5

            test_poses = [current_pose]
            base_angle = current_pose[2]
            for i in range(N_points):
                theta = base_angle + (2 * np.pi * i / N_points)
                px = current_pose[0] + radius_px * np.cos(theta)
                py = current_pose[1] + radius_px * np.sin(theta)
                test_poses.append((px, py, current_pose[2]))

            print(f"JUNCTION_DBG: Generated {len(test_poses)} test poses "
                  f"(radius={radius_m}m = {radius_px:.1f}px)")

            batch_tensors = []
            valid_poses = []
            skipped_wall = 0
            skipped_oob = 0

            explored_snapshot = self.simulator.explored.copy()

            for tp in test_poses:
                tp_x, tp_y = int(tp[0]), int(tp[1])
                H_occ, W_occ = self.simulator.occ.shape
                if not (0 <= tp_y < H_occ and 0 <= tp_x < W_occ):
                    skipped_oob += 1
                    continue
                if self.simulator.occ[tp_y, tp_x] == 1:
                    skipped_wall += 1
                    continue

                v_img = build_virtual_pose_image_north_oriented(
                    self.simulator.occ, explored_snapshot, current_pose, tp,
                    self.config.sim['RESOLUTION'], self.config.sim['WIN_PX'],
                    self.config.sim['RAD_CELLS'], show_head=True)
                batch_tensors.append(self.transform(Image.fromarray(v_img)))
                valid_poses.append(tp)

            print(f"JUNCTION_DBG: valid_poses={len(valid_poses)}, "
                  f"skipped_wall={skipped_wall}, skipped_oob={skipped_oob}")
            print(f"JUNCTION_DBG: required_consensus={required_consensus}")

            if len(batch_tensors) < required_consensus:
                print(f"JUNCTION_DBG: ABORT - only {len(batch_tensors)} valid poses, "
                      f"need {required_consensus}. Too many wall hits.")
            else:
                batch_stack = torch.stack(batch_tensors).to(self.config.device)
                with torch.no_grad():
                    batch_output = self.model(batch_stack).cpu().numpy().squeeze(1)

                if getattr(self.config, 'JUNCTION_DEBUG', False):
                    self._dump_junction_debug(batch_tensors, batch_output,
                                              valid_poses, current_pose)

                verified_count = 0
                valid_node_x, valid_node_y = [], []
                valid_components_world = []

                print(f"JUNCTION_DBG: Per-pose U-Net results:")
                for idx, v_heatmap in enumerate(batch_output):
                    h_norm_v = (v_heatmap - v_heatmap.min()) / (v_heatmap.max() - v_heatmap.min() + 1e-6)
                    h_smooth_v = gaussian_filter(h_norm_v, sigma=2.0)
                    hs_norm_v = (h_smooth_v - h_smooth_v.min()) / (h_smooth_v.max() - h_smooth_v.min() + 1e-6)
                    binary_v = (hs_norm_v >= np.percentile(hs_norm_v, 96.0))

                    ys_v, xs_v = np.mgrid[0:v_heatmap.shape[0], 0:v_heatmap.shape[1]]
                    angle_map_v = np.arctan2(
                        (ys_v - v_heatmap.shape[0] // 2).astype(np.float32),
                        (xs_v - v_heatmap.shape[1] // 2).astype(np.float32))
                    angular_hist_v, _ = np.histogram(
                        angle_map_v, bins=72, weights=np.where(binary_v, h_norm_v, 0))
                    tiled_v = np.tile(angular_hist_v / (angular_hist_v.max() + 1e-9), 3)
                    smooth_hist_v = gaussian_filter(tiled_v, sigma=3.0)[72:144]
                    peaks_v, _ = find_peaks(
                        smooth_hist_v, height=0.1,
                        distance=int(getattr(self.config, 'MIN_ARC_GAP_BINS', 5)))

                    heatmap_max_v = float(v_heatmap.max())
                    n_peaks_v = len(peaks_v)
                    peak_angles_v_deg = [round(float((-np.pi + (p / 72.0) * 2 * np.pi) * 180 / np.pi), 1)
                                         for p in peaks_v]
                    print(f"  Pose[{idx:02d}] heatmap_max={heatmap_max_v:.5f}  "
                          f"n_peaks={n_peaks_v}  peak_angles_deg={peak_angles_v_deg}")

                    if n_peaks_v > 1:
                        verified_count += 1
                        valid_node_x.append(valid_poses[idx][0])
                        valid_node_y.append(valid_poses[idx][1])

                        bin_centers_v = np.linspace(-np.pi, np.pi, 73)
                        bin_centers_v = 0.5 * (bin_centers_v[:-1] + bin_centers_v[1:])
                        peak_angles_rad_v = bin_centers_v[peaks_v]

                        ys_m_v, xs_m_v = np.where(binary_v)
                        pix_angles_v = angle_map_v[ys_m_v, xs_m_v]
                        diff_v = pix_angles_v[:, None] - peak_angles_rad_v[None, :]
                        diff_v = (diff_v + np.pi) % (2 * np.pi) - np.pi
                        nearest_peak_v = np.argmin(np.abs(diff_v), axis=1)

                        comp_count_this_pose = 0
                        for k_v in range(len(peak_angles_rad_v)):
                            member_v = (nearest_peak_v == k_v)
                            if not member_v.any():
                                continue
                            comp_ys_v = ys_m_v[member_v]
                            comp_xs_v = xs_m_v[member_v]
                            if len(comp_ys_v) < min_area_px:
                                continue
                            norm_vals_v = h_norm_v[comp_ys_v, comp_xs_v]
                            raw_vals_v = v_heatmap[comp_ys_v, comp_xs_v]
                            if float(norm_vals_v.max()) < rel_peak_thr:
                                continue
                            if float((raw_vals_v > coverage_thr).sum()) / len(raw_vals_v) < coverage_majority:
                                continue
                            weights_v = raw_vals_v.clip(0)
                            total_w_v = weights_v.sum()
                            if total_w_v > 1e-9:
                                cx_v = float((comp_xs_v * weights_v).sum() / total_w_v)
                                cy_v = float((comp_ys_v * weights_v).sum() / total_w_v)
                            else:
                                cx_v, cy_v = float(comp_xs_v.mean()), float(comp_ys_v.mean())
                            cov_sum_v = float(raw_vals_v.sum())
                            world_v = _transform_image_goal_to_world(
                                (int(cx_v), int(cy_v)), tuple(valid_poses[idx]),
                                self.config.sim.get('WIN_PX', 400),
                                self.config.sim.get('RESOLUTION', 0.05))
                            valid_components_world.append(
                                {'x': float(world_v[0]), 'y': float(world_v[1]),
                                 'cov': cov_sum_v})
                            comp_count_this_pose += 1

                        print(f"    -> Verified! {comp_count_this_pose} components "
                              f"extracted from this pose.")
                    else:
                        print(f"    -> REJECTED (need >1 peak, got {n_peaks_v})")

                print(f"\nJUNCTION_DBG: verified_count={verified_count}/{len(valid_poses)}, "
                      f"required={required_consensus}")
                print(f"JUNCTION_DBG: total valid_components_world collected = "
                      f"{len(valid_components_world)}")

                if verified_count >= required_consensus:
                    print(f"JUNCTION_DBG: Consensus reached! Proceeding to clustering...")
                    avg_node_x = sum(valid_node_x) / len(valid_node_x)
                    avg_node_y = sum(valid_node_y) / len(valid_node_y)

                    clusters = []
                    angle_tolerance_rad = np.radians(30.0)
                    for comp in valid_components_world:
                        comp_angle = np.arctan2(comp['y'] - avg_node_y, comp['x'] - avg_node_x)
                        placed = False
                        for cluster in clusters:
                            base_ang = np.arctan2(cluster[0]['y'] - avg_node_y,
                                                  cluster[0]['x'] - avg_node_x)
                            diff_ang = (comp_angle - base_ang + np.pi) % (2 * np.pi) - np.pi
                            if abs(diff_ang) < angle_tolerance_rad:
                                cluster.append(comp)
                                placed = True
                                break
                        if not placed:
                            clusters.append([comp])

                    print(f"JUNCTION_DBG: Angle-based clustering -> {len(clusters)} clusters:")
                    for ci, cl in enumerate(clusters):
                        avg_angle_deg = float(np.degrees(np.arctan2(
                            sum(c['y'] for c in cl) / len(cl) - avg_node_y,
                            sum(c['x'] for c in cl) / len(cl) - avg_node_x)))
                        print(f"  Cluster[{ci}]: size={len(cl)}, angle={avg_angle_deg:.1f} deg, "
                              f"avg_cov={sum(c['cov'] for c in cl) / len(cl):.3f}")

                    min_cluster_size = 2
                    averaged_components = []
                    for cluster in clusters:
                        if len(cluster) >= min_cluster_size:
                            averaged_components.append({
                                'x': sum(c['x'] for c in cluster) / len(cluster),
                                'y': sum(c['y'] for c in cluster) / len(cluster),
                                'cov': sum(c['cov'] for c in cluster) / len(cluster)})
                        else:
                            print(f"JUNCTION_DBG:   Cluster DROPPED (size={len(cluster)} "
                                  f"< min={min_cluster_size})")

                    print(f"JUNCTION_DBG: averaged_components after filter = "
                          f"{len(averaged_components)}")
                    averaged_components.sort(key=lambda c: c['cov'], reverse=True)

                    if not averaged_components:
                        print("JUNCTION_DBG: Consensus reached but all clusters below "
                              "min_cluster_size=2. No junction registered.")
                    else:
                        frame_components = []
                        for rank_idx, comp in enumerate(averaged_components):
                            img_px = self._transform_world_to_image_goal(
                                (comp['x'], comp['y']), tuple(current_pose),
                                self.config.sim.get('WIN_PX', 400))
                            frame_components.append({
                                'rank': rank_idx + 1, 'x': comp['x'], 'y': comp['y'],
                                'cx_img': float(img_px[0]), 'cy_img': float(img_px[1]),
                                'explored': False, 'cov': comp['cov']})
                        print(f"JUNCTION_DBG: Final frame_components ({len(frame_components)}):")
                        for fc in frame_components:
                            print(f"  Rank {fc['rank']}: world=({fc['x']:.1f},{fc['y']:.1f})  "
                                  f"cov={fc['cov']:.3f}")

                        node_id = len(self.exploration_nodes)
                        frame_components.sort(key=lambda c: c['rank'], reverse=True)
                        chosen_comp = frame_components[0]
                        chosen_comp['explored'] = True
                        locked_angle = np.arctan2(chosen_comp['y'] - avg_node_y,
                                                  chosen_comp['x'] - avg_node_x)
                        frame_components.sort(key=lambda c: c['rank'])

                        with self.nodes_lock:
                            self.exploration_nodes.append({
                                'id': node_id, 'x': avg_node_x, 'y': avg_node_y,
                                'count': 1, 'components': frame_components,
                                'locked_angle': locked_angle,
                                'locked_rank': chosen_comp['rank']})
                        self.current_junction_id = node_id
                        chosen_goal_px = (int(chosen_comp['cx_img']), int(chosen_comp['cy_img']))
                        print(f"JUNCTION_DBG: Node {node_id} created! Chose Rank "
                              f"{chosen_comp['rank']} (cov={chosen_comp['cov']:.3f})")
                else:
                    print(f"JUNCTION_DBG: Junction REJECTED - {verified_count}/"
                          f"{len(valid_poses)} < required={required_consensus}")

            _acc = getattr(self, '_prof_acc', None)
            if _acc is not None and getattr(self.config, 'PROFILE', True):
                _e = _acc.get('junction_verify')
                _dt = time.perf_counter() - _jv_t0
                _dc = _THREAD_TIME() - _jv_c0
                if _e is None:
                    _acc['junction_verify'] = [_dt, 1, _dc]
                else:
                    _e[0] += _dt
                    _e[1] += 1
                    _e[2] += _dc
        else:
            self.current_junction_id = None  # Clear lock outside radius

        return chosen_goal_px, active_directions, frontier_centroids_for_viz

    def _dump_junction_debug(self, batch_tensors, batch_output, valid_poses, current_pose):
        """Case-B debug image dump. Side-effect only: writes PNGs under
        output_dir/junction_debug/."""
        dbg_dir = os.path.join(self.config.output_dir, "junction_debug")
        os.makedirs(dbg_dir, exist_ok=True)
        _dbg_id = getattr(self, '_junction_debug_count', 0)
        self._junction_debug_count = _dbg_id + 1
        jdir = os.path.join(dbg_dir, f"junction_{_dbg_id:04d}")
        os.makedirs(jdir, exist_ok=True)

        WIN_PX = self.config.sim.get('WIN_PX', 400)

        occ_vis = self.simulator.occ.copy().astype(np.uint8)
        H_o, W_o = occ_vis.shape
        overview_bgr = np.zeros((H_o, W_o, 3), dtype=np.uint8)
        overview_bgr[occ_vis == 0] = (220, 220, 220)
        overview_bgr[occ_vis == 1] = (0, 0, 0)
        overview_bgr[occ_vis == -1] = (127, 127, 127)

        exp = self.simulator.explored
        overview_bgr[exp & (occ_vis == 0)] = (255, 255, 255)
        overview_bgr[exp & (occ_vis == 1)] = (40, 40, 40)

        for i, tp in enumerate(valid_poses):
            tx, ty = int(tp[0]), int(tp[1])
            cv2.circle(overview_bgr, (tx, ty), 4, (255, 200, 0), -1)
            cv2.putText(overview_bgr, str(i), (tx + 5, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 0), 1)

        rx, ry = int(current_pose[0]), int(current_pose[1])
        cv2.drawMarker(overview_bgr, (rx, ry), (0, 255, 0), cv2.MARKER_STAR, 18, 2)

        pad = WIN_PX // 2 + 30
        x1_c = max(0, rx - pad)
        x2_c = min(W_o, rx + pad)
        y1_c = max(0, ry - pad)
        y2_c = min(H_o, ry + pad)
        cv2.imwrite(os.path.join(jdir, "00_overview.png"),
                    overview_bgr[y1_c:y2_c, x1_c:x2_c])

        CMAP = cv2.COLORMAP_JET
        for idx in range(len(batch_tensors)):
            t = batch_tensors[idx]
            img_np = (t.numpy()[0] * 0.5 + 0.5)
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            local_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

            v_heatmap = batch_output[idx]
            h_min_v, h_max_v = v_heatmap.min(), v_heatmap.max()
            h_norm_disp = ((v_heatmap - h_min_v) / (h_max_v - h_min_v + 1e-9) * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(h_norm_disp, CMAP)
            if heatmap_color.shape[:2] != local_bgr.shape[:2]:
                heatmap_color = cv2.resize(heatmap_color,
                                           (local_bgr.shape[1], local_bgr.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            ctr = WIN_PX // 2
            cv2.circle(local_bgr, (ctr, ctr), 5, (0, 255, 0), -1)

            pose_lbl = (f"Pose[{idx:02d}] ({valid_poses[idx][0]:.0f},"
                        f"{valid_poses[idx][1]:.0f}) | max={v_heatmap.max():.6f}")
            bar = np.zeros((28, local_bgr.shape[1] * 2, 3), dtype=np.uint8)
            cv2.putText(bar, pose_lbl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 255, 200), 1)
            cv2.imwrite(os.path.join(jdir, f"pose_{idx:02d}.png"),
                        np.vstack([np.hstack([local_bgr, heatmap_color]), bar]))

        print(f"JUNCTION_DBG: Saved debug images -> {jdir}/")

    def _prediction_worker(self):
        """Orchestrator: setup -> LOCAL_ONLY -> plan_local -> plan_global
        -> [G] goal arbitration -> [H] fallback -> response."""
        print("PREDICTION: Memory-Based Worker starting...")

        prediction_count = 0

        # Idle passes are NOT idle -- they run plan_global, which creates junction
        # nodes, marks branches explored and sets locked angles. So the throttle
        # is by DISTANCE TRAVELLED, not wall-clock: predictions-per-metre stays
        # constant however fast turbo makes the loop run.
        #   IDLE_PREDICT_EVERY_PX = 1.0  -> ~1 prediction per pixel of motion
        #   IDLE_PREDICT_EVERY_PX = 0    -> only predict when a goal is requested
        idle_every_px = float(getattr(self.config, 'IDLE_PREDICT_EVERY_PX', 1.0))
        last_idle_pos = None

        # [PREEMPT] state
        self._enroute_frontier = None
        self.preempt_current_goal = False
        self._enroute_goal_dist_px = 0.0
        self._committed_frontier = None
        self._last_comp_world = None
        self._last_snap_delta_px = 0.0
        self._last_lock_live = False
        self._no_frontier_count = 0

        # [DECLOG] state
        self._dec_flags = set()
        self._dec_prev_angle = None
        self._dec_t0 = time.time()

        self._prof_acc = {}
        self._prof_t0 = time.perf_counter()
        self._prof_passes = 0

        if getattr(self.config, 'DEBUG_LOG', True):
            try:
                self._dbg_log()
                print(f"PREDICTION: debug log -> {getattr(self, '_dbg_log_path', '?')}")
                self._dbg("#" * 78)
                self._dbg(f"# PREDICTION WORKER START  idle_every_px={idle_every_px} "
                          f"DEBUG_LOG_IDLE={getattr(self.config, 'DEBUG_LOG_IDLE', False)}")
            except Exception as e:
                print(f"PREDICTION: could not open debug log: {e}")

        while self.running.is_set():
            try:
                nav_request_pending = False
                _req_id = None                     # [REQ-ID] echoed in the response
                try:
                    request_data = self.prediction_request_queue.get_nowait()
                    nav_request_pending = True
                    current_pose = request_data.get('pose')
                    _req_id = request_data.get('req_id')
                except queue.Empty:
                    pose_now = safe_copy_pose(self.simulator.pose)
                    if idle_every_px > 0:
                        moved_enough = (
                            last_idle_pos is None or
                            np.hypot(pose_now[0] - last_idle_pos[0],
                                     pose_now[1] - last_idle_pos[1]) >= idle_every_px
                        )
                    else:
                        moved_enough = False

                    if moved_enough:
                        last_idle_pos = (float(pose_now[0]), float(pose_now[1]))
                        current_pose = pose_now
                    else:
                        # Block briefly on the request queue instead of spinning:
                        # this frees the GIL and GPU for the sim thread.
                        try:
                            request_data = self.prediction_request_queue.get(timeout=0.02)
                            nav_request_pending = True
                            current_pose = request_data.get('pose')
                            _req_id = request_data.get('req_id')
                        except queue.Empty:
                            continue

                # [DECLOG] guard flags accumulate across the idle passes that
                # follow a nav pass, and are written out with the NEXT nav
                # decision, so a row says which guards fired before that choice.
                if nav_request_pending:
                    pass
                else:
                    pass

                # ==========================================
                # 1. LOCAL PLANNER (runs continuously)
                # ==========================================
                with self._prof('scan_build'):
                    scan_result = self.simulator._simulate_scan(current_pose)
                    origin_img = build_origin_image_north_oriented(
                        self.simulator.occ, self.simulator.explored, scan_result,
                        current_pose,
                        self.config.sim['RESOLUTION'], self.config.sim['WIN_PX'],
                        self.config.sim['RAD_CELLS'], self.config.sim['FOV'],
                        show_head=self.config.sim.get('SHOW_HEAD', False),
                        goals_to_visualize=None)

                with self._prof('model_infer'):
                    input_tensor = self._to_input(origin_img)
                    with torch.no_grad():
                        heatmap = self.model(input_tensor).cpu().squeeze().numpy()

                max_score = float(heatmap.max())
                self.diag['pred_passes'] += 1
                if getattr(self.config, 'PROFILE', True):
                    self._prof_passes = getattr(self, '_prof_passes', 0) + 1
                    if self._prof_passes >= int(getattr(self.config, 'PROFILE_EVERY', 50)):
                        self._prof_dump()

                # ============================================================
                # LOCAL-ONLY MODE: pure coverage-prediction policy (ablation).
                # ============================================================
                if getattr(self.config, 'LOCAL_ONLY_MODE', False):
                    goal_mode = getattr(self.config, 'LOCAL_ONLY_GOAL', 'centroid')
                    if goal_mode == 'argmax':
                        peak_yx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        goal_px = (int(peak_yx[1]), int(peak_yx[0]))
                        sel_score = float(heatmap[peak_yx])
                        cent_info = {'centroid_img': goal_px}
                    else:
                        goal_px, cent_info = self.plan_local_centroid(heatmap)
                        sel_score = float(max_score)

                    cx_i, cy_i = cent_info['centroid_img']
                    response = {
                        'heatmap': heatmap, 'goal_px': goal_px,
                        'req_id': _req_id,                    # [REQ-ID]
                        'selected_score': sel_score, 'final_heading': None,
                        'origin_img': origin_img, 'timestamp': time.time(),
                        'prediction_id': prediction_count,
                        'request_pose': current_pose,
                        'mode': 'LOCAL_ONLY_' + goal_mode.upper(),
                        'hallucinations': [],
                        'local_components': {
                            'label_map': np.zeros_like(heatmap, dtype=np.uint16),
                            'keep_ids': [1],
                            'centroids': {1: (float(cx_i), float(cy_i))},
                            'detected_doorways': [], 'active_directions': [],
                            'frontier_centroids': []},
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
                    continue
                # ============================================================

                # [B] LOCAL PLANNER
                with self._prof('plan_local'):
                    local_components = self.plan_local(heatmap, current_pose)
                keep_ids = local_components['keep_ids']

                _log_this = getattr(self.config, 'DEBUG_LOG', True) and (
                    nav_request_pending or getattr(self.config, 'DEBUG_LOG_IDLE', False))
                if _log_this:
                    self._dbg("=" * 78)
                    self._dbg(f"[{'NAV ' if nav_request_pending else 'IDLE'}] pred#{prediction_count} "
                              f"pose=({current_pose[0]:.1f},{current_pose[1]:.1f}) "
                              f"heading={np.degrees(current_pose[2]):+.1f}deg "
                              f"max_score={max_score:.5f} keep_ids={keep_ids}")
                    RES = self.config.sim.get('RESOLUTION', 0.05)
                    for cc in keep_ids:
                        cx_i, cy_i = local_components['centroids'][cc]
                        wx, wy = self._img_to_world_dbg((cx_i, cy_i), current_pose)
                        dist_px = np.hypot(wx - current_pose[0], wy - current_pose[1])
                        self._dbg(f"    keep_id {cc}: img=({int(cx_i)},{int(cy_i)}) "
                                  f"world=({wx:.1f},{wy:.1f}) "
                                  f"dist={dist_px:.1f}px/{dist_px * RES:.2f}m "
                                  f"from_heading={self._ang_from_heading_deg(wx, wy, current_pose):+.1f}deg")

                # [D]+[E]+[F] GLOBAL / JUNCTION PLANNER
                # [IDLE-CHEAP] If the robot is not within the junction radius of
                # any node AND there is at most one component, neither Case A nor
                # Case B can fire and the whole call reduces to clearing
                # current_junction_id -- so do that directly.
                _skip_global = False
                if not nav_request_pending and getattr(self.config, 'IDLE_SKIP_GLOBAL', True):
                    if len(keep_ids) <= 1:
                        _merge_px = (float(getattr(self.config, 'JUNCTION_LOCK_DIST_M', 3.0))
                                     / self.config.sim.get('RESOLUTION', 0.05))
                        with self.nodes_lock:
                            _near_node = any(
                                np.hypot(nd['x'] - current_pose[0],
                                         nd['y'] - current_pose[1]) < _merge_px
                                for nd in self.exploration_nodes)
                        _skip_global = not _near_node

                if _skip_global:
                    self.current_junction_id = None
                    chosen_goal_px = None
                    _pc = getattr(self, '_prepass_cache', None)
                    active_directions = _pc['active_directions'] if _pc else []
                    frontier_centroids_for_viz = _pc['frontier_centroids'] if _pc else []
                    try:
                        self.diag['idle_global_skipped'] = self.diag.get('idle_global_skipped', 0) + 1
                    except Exception:
                        pass
                else:
                    with self._prof('plan_global'):
                        chosen_goal_px, active_directions, frontier_centroids_for_viz = \
                            self.plan_global(heatmap, current_pose, local_components,
                                             allow_prepass_refresh=nav_request_pending)

                if _log_this:
                    self._dbg(f"    plan_global: chosen_goal_px(junction)={chosen_goal_px} "
                              f"current_junction_id={getattr(self, 'current_junction_id', None)} "
                              f"active_directions={len(active_directions)}")

                best_proactive_goal = None

                # --- 2. Select Local Goal ---
                # [NO-GOAL] Reset every pass. The old final `else` left goal_px
                # UNBOUND on the first pass, so the degenerate check raised
                # NameError and the worker looped without ever answering the sim.
                goal_px = None
                if chosen_goal_px is not None:
                    goal_px = chosen_goal_px
                    _goal_src = "JUNCTION-LOCK"
                elif best_proactive_goal is not None:
                    goal_px = self._transform_world_to_image_goal(
                        best_proactive_goal, tuple(current_pose),
                        self.config.sim.get('WIN_PX', 400))
                    _goal_src = "PROACTIVE"
                elif len(keep_ids) >= 1:
                    self._no_frontier_count = 0        # [FRONTIER-FALLBACK]
                    self._committed_frontier = None    # [COMMIT] local planner has it
                    # [STICK] prefer the component chosen last pass
                    target_cc_id = self._sticky_component(
                        keep_ids, local_components, current_pose)
                    cx, cy = local_components['centroids'][target_cc_id]
                    goal_px = (int(cx), int(cy))
                    _goal_src = f"KEEP_IDS[0]=cc{target_cc_id}"
                    try:
                        _wx, _wy = _transform_image_goal_to_world(
                            (int(cx), int(cy)), tuple(current_pose),
                            self.config.sim.get('WIN_PX', 400),
                            self.config.sim.get('RESOLUTION', 0.05))
                        self._last_comp_world = (float(_wx), float(_wy))
                    except Exception:
                        self._last_comp_world = None
                elif max_score > coverage_thr:
                    r, c = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                    goal_px = (int(c), int(r))
                    _goal_src = "ARGMAX"
                else:
                    _goal_src = "NONE(loop-carried)"

                if _log_this:
                    self._dbg(f"    ARBITRATION source={_goal_src} -> goal_px={goal_px}")

                selected_local_score = float(max_score)
                final_heading = None
                mode = "LOCAL"
                hallucinations = []
                local_components['active_directions'] = active_directions
                local_components['frontier_centroids'] = frontier_centroids_for_viz

                # [CHURN] a goal on the robot or in a wall is reached/rejected
                # instantly -> goal count spikes, no motion.
                _churn_degenerate = self._goal_is_degenerate(goal_px, current_pose)

                # [PREEMPT] A fresh nav request means any prior 'en route' state
                # is done or aborted: clear it and consume the pending signal.
                if nav_request_pending:
                    self._enroute_frontier = None
                    self.preempt_current_goal = False
                # Otherwise (idle pass while still driving to a fallback frontier):
                # if the local planner now sees REAL coverage on the way, signal
                # the sim to abort and re-plan, so we switch to the local planner
                # BEFORE reaching the frontier.
                elif (getattr(self.config, 'PREEMPT_ENROUTE', True)
                        and getattr(self, '_enroute_goal_dist_px', 0.0)
                            >= float(getattr(self.config, 'PREEMPT_MIN_GOAL_PX', 50.0))
                        and len(keep_ids) >= 1
                        and max_score >= float(getattr(self.config, 'PREEMPT_MIN_SCORE', coverage_thr))
                        and not _churn_degenerate):
                    # Gated on goal DISTANCE, not on whether the goal came from
                    # the frontier fallback -- otherwise a local, junction or
                    # nudge goal could never be interrupted. Short
                    # component-centroid goals stay uninterruptible: abandoning
                    # those on every new prediction is the churn this avoids.
                    _d = getattr(self, '_enroute_goal_dist_px', 0.0)
                    self.preempt_current_goal = True
                    self._enroute_frontier = None
                    self._enroute_goal_dist_px = 0.0
                    self._dec_flag('preempt')
                    self._dbg(f"    [PREEMPT] coverage en route (score={max_score:.5f}, "
                              f"comps={len(keep_ids)}, goal was {_d:.0f}px away) "
                              f"-> switch to local planner")
                    print("PREDICTION: [PREEMPT] local coverage detected en route "
                          f"(score={max_score:.5f}, comps={len(keep_ids)}, "
                          f"goal {_d:.0f}px away) -> switching to local planner.")

                # [EARLY-STOP] Driving to a frontier with NO above-threshold
                # coverage the whole way, so the preempt above never fires. Stop
                # only when the frontier has actually been CONSUMED -- i.e. no
                # frontier cell remains near it. Testing where the target sits in
                # the image instead fired on every trip while the frontier was
                # still live, so no trip ever completed.
                elif (getattr(self.config, 'EARLY_STOP_ENROUTE', True)
                        and self._enroute_frontier is not None
                        and max_score < float(getattr(self.config, 'PREEMPT_MIN_SCORE', coverage_thr))):
                    if self._frontier_consumed(self._enroute_frontier):
                        dist_px = np.hypot(self._enroute_frontier[0] - current_pose[0],
                                           self._enroute_frontier[1] - current_pose[1])
                        self.preempt_current_goal = True
                        self._enroute_frontier = None
                        self._committed_frontier = None
                        self._enroute_goal_dist_px = 0.0
                        self._dec_flag('early_stop')
                        self._dbg(f"    [EARLY-STOP] enroute frontier CONSUMED "
                                  f"({dist_px:.0f}px away, max_score={max_score:.5f}) "
                                  f"-> replanning")
                        print("PREDICTION: [EARLY-STOP] enroute frontier has been "
                              f"explored away ({dist_px:.0f}px away) -> re-planning.")

                # [NO-UTURN] If the chosen local goal points backward, prefer the
                # strongest FORWARD component; if there is none, force the
                # frontier fallback rather than U-turning on a weak lobe.
                force_fallback = False
                if (nav_request_pending and goal_px is not None
                        and getattr(self.config, 'ANTI_UTURN', True)
                        and len(keep_ids) >= 1 and not _churn_degenerate
                        and self._is_backward(goal_px, current_pose)):
                    _fwd = self._strongest_forward(
                        keep_ids, local_components.get('centroids', {}), current_pose)
                    if _fwd is not None:
                        self._dec_flag('no_uturn')
                        self._dbg(f"    [NO-UTURN] backward goal {goal_px} -> forward comp {_fwd}")
                        goal_px = _fwd
                    else:
                        self._dec_flag('force_fallback')
                        self._dbg(f"    [NO-UTURN] backward goal {goal_px}, no forward "
                                  f"coverage -> route to frontier fallback")
                        force_fallback = True

                # --- 3. FALLBACK: nearest reachable frontier -------------------
                if (nav_request_pending
                        and (len(keep_ids) == 0 or _churn_degenerate or force_fallback)
                        and best_proactive_goal is None):
                    # [STICK] the local planner is not driving any more, so the
                    # remembered component is stale.
                    self._last_comp_world = None
                    if active_directions:
                        self._no_frontier_count = 0
                        robot_px = (int(current_pose[0]), int(current_pose[1]))
                        margin = float(getattr(self.config, 'FRONTIER_DIRECT_MARGIN_PX', 0.0))

                        scored = []
                        _dbg_rows = []
                        with self._prof('fallback_astar'):
                            for d in active_directions:
                                cd = self._astar_len(robot_px, d['frontier_px'])
                                if cd is not None:
                                    scored.append((cd, d))
                                _dbg_rows.append((d, cd))

                        if getattr(self.config, 'DEBUG_LOG', True):
                            self._dbg(f"    FALLBACK ({'forced-uturn' if force_fallback else 'keep_ids empty/degenerate'}): "
                                      f"{len(active_directions)} active directions:")
                            for d, cd in _dbg_rows:
                                fpx = d['frontier_px']
                                euc = np.hypot(fpx[0] - current_pose[0], fpx[1] - current_pose[1])
                                self._dbg(f"      Node{d['node_id']} Rank{d['rank']} "
                                          f"branch=({d['x']:.0f},{d['y']:.0f}) "
                                          f"frontier=({fpx[0]},{fpx[1]}) "
                                          f"astar={('%.0f' % cd) if cd is not None else 'UNREACH'}px "
                                          f"euclid={euc:.0f}px "
                                          f"frontier_from_heading={self._ang_from_heading_deg(fpx[0], fpx[1], current_pose):+.1f}deg")

                        if scored:
                            scored.sort(key=lambda t: t[0])
                            cost_direct, best = scored[0]

                            # [COMMIT] Hysteresis. This was a pure argmin
                            # recomputed every nav pass with _enroute_frontier
                            # cleared at the top of each one, so two comparably
                            # distant frontiers swapped nearest-rank as the robot
                            # advanced on one and neither was ever reached.
                            _com = getattr(self, '_committed_frontier', None)
                            if _com is not None and not self._frontier_consumed(_com):
                                _keep = None
                                for _cd, _d in scored:
                                    if np.hypot(_d['frontier_px'][0] - _com[0],
                                                _d['frontier_px'][1] - _com[1]) <= 30.0:
                                        _keep = (_cd, _d)
                                        break
                                if _keep is not None:
                                    _mar = float(getattr(self.config,
                                                         'FRONTIER_SWITCH_MARGIN', 0.35))
                                    if _keep[0] <= cost_direct * (1.0 + _mar):
                                        if _keep[1] is not best:
                                            self._dec_flag('commit')
                                            self._dbg("      [COMMIT] keeping %s (%.0fpx) "
                                                      "over %s (%.0fpx)"
                                                      % (_com, _keep[0],
                                                         best['frontier_px'], cost_direct))
                                        cost_direct, best = _keep
                            self._committed_frontier = (int(best['frontier_px'][0]),
                                                        int(best['frontier_px'][1]))
                            f_px = best['frontier_px']
                            b_px = (int(best['x']), int(best['y']))
                            cost_rb = self._astar_len(robot_px, b_px)
                            cost_bf = self._astar_len(b_px, f_px)
                            via = (cost_rb + cost_bf) if (cost_rb is not None and cost_bf is not None) else None
                            go_direct = (via is None) or (cost_direct <= via + margin)

                            if go_direct:
                                target_world_px = f_px
                                print(f"PREDICTION: 0 components -> DIRECT to frontier of "
                                      f"Node {best['node_id']} Rank-{best['rank']} "
                                      f"(direct={cost_direct:.0f}"
                                      f"{'' if via is None else ' <= via=%.0f' % via})")
                            else:
                                target_world_px = b_px
                                print(f"PREDICTION: 0 components -> via active branch Node "
                                      f"{best['node_id']} Rank-{best['rank']} "
                                      f"(via={via:.0f} < direct={cost_direct:.0f})")

                            self._enroute_frontier = f_px
                        else:
                            active_directions.sort(key=lambda d: d['dist'])
                            best = active_directions[0]
                            if best['dist'] < 8.0:
                                target_world_px = best['frontier_px']
                                print(f"PREDICTION: 0 components at Branch Node "
                                      f"{best['node_id']} Rank-{best['rank']}. "
                                      f"Handing off directly to Frontier Centroid!")
                            else:
                                target_world_px = (int(best['x']), int(best['y']))
                                print(f"PREDICTION: 0 components -> returning to active "
                                      f"branch: Node {best['node_id']} "
                                      f"Rank-{best['rank']} at {best['dist']:.1f}px")
                            self._enroute_frontier = best['frontier_px']

                        goal_px = self._transform_world_to_image_goal(
                            target_world_px, tuple(current_pose),
                            self.config.sim.get('WIN_PX', 400))
                        local_components['fallback_target_node'] = target_world_px
                        local_components['fallback_target_frontier'] = target_world_px
                        mode = "LOCAL"

                    else:
                        # [FRONTIER-FALLBACK] Nothing local and no junction branch
                        # holds a frontier -- but that does NOT mean there are no
                        # frontiers. The prepass keeps only the 4 nearest and only
                        # those a branch can claim through an A* call, so frontiers
                        # far from any node are dropped on a technicality.
                        #
                        # This used to fall back to the nearest junction NODE
                        # CENTRE, which is somewhere the robot has already been:
                        # reaching it explored nothing, the next pass faced the
                        # same situation, and "nearest node with dist > 5.0"
                        # flipped between the two nodes either side of the robot.
                        # Go to the nearest REACHABLE FRONTIER instead -- a target
                        # that cannot survive being reached, so it cannot
                        # oscillate.
                        _ff_goal = None
                        try:
                            _rpx = (int(current_pose[0]), int(current_pose[1]))
                            _fdata = self.global_planner.find_global_frontiers(
                                self.simulator.occ, self.simulator.explored, _rpx)
                            _snap = getattr(self.config, 'SNAP_FRONTIER_OFF_OBSTACLE', True)
                            _fc = []
                            for _fi in (_fdata or []):
                                _fx, _fy, _ = _fi['pose']
                                if _snap:
                                    _fx, _fy = self._frontier_boundary_point((_fx, _fy))
                                _fc.append((np.hypot(_fx - _rpx[0], _fy - _rpx[1]),
                                            (int(_fx), int(_fy))))
                            _fc.sort(key=lambda t: t[0])
                            _lim = int(getattr(self.config, 'FALLBACK_FRONTIER_CANDS', 10))
                            if getattr(self.config, 'DEBUG_LOG', True):
                                self._dbg("    FRONTIER-FALLBACK: %d frontier(s), trying "
                                          "the %d nearest" % (len(_fc), min(_lim, len(_fc))))
                            for _euc, _fpx in _fc[:_lim]:
                                _al = self._astar_len(_rpx, _fpx)
                                if _al is not None:
                                    _ff_goal = _fpx
                                    if getattr(self.config, 'DEBUG_LOG', True):
                                        self._dbg("      chose frontier (%d,%d) "
                                                  "euclid=%.0fpx astar=%.0fpx"
                                                  % (_fpx[0], _fpx[1], _euc, _al))
                                    break
                        except Exception as e:
                            print("PREDICTION: frontier fallback failed: %s" % e)

                        if _ff_goal is not None:
                            self._no_frontier_count = 0
                            target_world_px = _ff_goal
                            goal_px = self._transform_world_to_image_goal(
                                target_world_px, tuple(current_pose),
                                self.config.sim.get('WIN_PX', 400))
                            local_components['fallback_target_node'] = target_world_px
                            local_components['fallback_target_frontier'] = target_world_px
                            self._enroute_frontier = target_world_px
                            self._committed_frontier = (int(target_world_px[0]),
                                                        int(target_world_px[1]))
                            mode = "LOCAL"
                            print("PREDICTION: no active branches -> nearest reachable "
                                  "FRONTIER at %s" % (target_world_px,))
                        else:
                            self._no_frontier_count = getattr(self, '_no_frontier_count', 0) + 1
                            _nf_lim = int(getattr(self.config, 'NO_FRONTIER_EXHAUST', 5))
                            print("PREDICTION: no active branches and no reachable "
                                  "frontier (%d/%d)" % (self._no_frontier_count, _nf_lim))
                            if self._no_frontier_count >= _nf_lim:
                                print("PREDICTION: [EXHAUSTED] nothing reachable left to "
                                      "explore. Ending the episode.")
                                self._dbg("    [EXHAUSTED] no reachable frontier in %d "
                                          "passes" % self._no_frontier_count)
                                self.exploration_complete.set()

                # [CHURN] last resort: nudge a still-degenerate goal outward.
                if nav_request_pending and self._goal_is_degenerate(goal_px, current_pose):
                    _nud = self._nudge_to_free(goal_px, current_pose)
                    if _nud is None:
                        _w = nudge_goal_along_heading(
                            current_pose, self.simulator.occ,
                            float(getattr(self.config, 'NO_GOAL_NUDGE_PX', 60.0)))
                        if _w is not None:
                            _nud = self._transform_world_to_image_goal(
                                _w, tuple(current_pose),
                                self.config.sim.get('WIN_PX', 400))
                            mode = "NUDGE"
                            print("PREDICTION: [NO-GOAL] no usable goal from any source "
                                  "-> nudging along heading to %s" % (_nud,))
                    if _nud is not None:
                        goal_px = _nud
                        self._dec_flag('nudge')
                        if getattr(self.config, 'DEBUG_LOG', True):
                            self._dbg(f"    CHURN nudge applied -> goal_px={goal_px}")

                # [NO-GOAL] The simulator cannot consume goal_px=None: it calls
                # _transform_image_goal_to_world on it, raises, and re-requests
                # forever. A nav pass must never answer with None.
                if nav_request_pending and goal_px is None:
                    print("PREDICTION: [NO-GOAL] boxed in at "
                          f"({current_pose[0]:.0f},{current_pose[1]:.0f}) -- no free cell "
                          "reachable in any direction. Ending the episode.")
                    self._dbg("    [NO-GOAL] boxed in, setting exploration_complete")
                    self.exploration_complete.set()
                    break

                _wg = (self._img_to_world_dbg(goal_px, current_pose)
                       if goal_px is not None else (float('nan'), float('nan')))
                _dm = (np.hypot(_wg[0] - current_pose[0], _wg[1] - current_pose[1])
                       if goal_px is not None else float('nan'))

                if _log_this:
                    self._dbg(f"    CHOSE goal_px={goal_px} world=({_wg[0]:.1f},{_wg[1]:.1f}) "
                              f"dist={_dm:.1f}px "
                              f"from_heading={self._ang_from_heading_deg(_wg[0], _wg[1], current_pose):+.1f}deg "
                              f"mode={mode}")

                # [PREEMPT] Record how far this goal is, so an idle pass can
                # decide whether the trip is long enough to be worth abandoning.
                if nav_request_pending:
                    self._enroute_goal_dist_px = float(_dm) if goal_px is not None else 0.0

                # ---------------------------------------------------- [DECLOG]
                # One structured row per NAV decision. Everything a "why this
                # goal" question needs, so nobody has to regex the prose log --
                # which is how a wrong conclusion got drawn during the
                # oscillation investigation.
                if nav_request_pending and getattr(self.config, 'DECISION_LOG', False):
                    try:
                        _ang = (np.arctan2(_wg[1] - current_pose[1],
                                           _wg[0] - current_pose[0])
                                if (goal_px is not None and _dm > 1.0) else None)
                        _turn = ""
                        if _ang is not None and self._dec_prev_angle is not None:
                            _turn = round(float(np.degrees(abs(
                                (_ang - self._dec_prev_angle + np.pi) % (2 * np.pi) - np.pi))), 1)
                        if _ang is not None:
                            self._dec_prev_angle = _ang

                        _ef = getattr(self, '_enroute_frontier', None)
                        _cf = getattr(self, '_committed_frontier', None)

                        # astar_px is the one expensive field: a full A* per nav
                        # decision. Its own flag so a sweep never pays for it.
                        _ap, _dr = "", ""
                        if (getattr(self.config, 'DECISION_LOG_ASTAR', False)
                                and goal_px is not None and np.isfinite(_dm)):
                            _v = self._astar_len(
                                (int(current_pose[0]), int(current_pose[1])),
                                (int(_wg[0]), int(_wg[1])))
                            if _v is not None:
                                _ap = round(float(_v), 1)
                                _dr = round(float(_v) / max(1.0, float(_dm)), 2)

                        with self.state_lock:
                            _pm = float(self.shared_state.get('total_path_length', 0.0))
                            _cv = float(self.shared_state.get('exploration_pct', 0.0))

                        self._dec_row(
                            pred=prediction_count,
                            t_s=round(time.time() - self._dec_t0, 2),
                            path_m=round(_pm, 2), coverage_pct=round(_cv, 2),
                            pose_x=round(float(current_pose[0]), 1),
                            pose_y=round(float(current_pose[1]), 1),
                            heading_deg=round(float(np.degrees(current_pose[2])), 1),
                            max_score=round(max_score, 6),
                            n_comps=len(keep_ids),
                            keep_ids="|".join(str(c) for c in keep_ids),
                            cov_sums="|".join(
                                "%.3f" % local_components.get('cov_sums', {}).get(c, 0.0)
                                for c in keep_ids),
                            junction_id=getattr(self, 'current_junction_id', None),
                            lock_live=int(bool(getattr(self, '_last_lock_live', False))),
                            n_active_dirs=len(active_directions),
                            src=_goal_src,
                            goal_img_x=(goal_px[0] if goal_px else ""),
                            goal_img_y=(goal_px[1] if goal_px else ""),
                            goal_x=round(float(_wg[0]), 1),
                            goal_y=round(float(_wg[1]), 1),
                            dist_px=round(float(_dm), 1),
                            from_heading_deg=round(self._ang_from_heading_deg(
                                _wg[0], _wg[1], current_pose), 1),
                            turn_from_prev_deg=_turn,
                            mode=mode,
                            enroute_x=(int(_ef[0]) if _ef is not None else ""),
                            enroute_y=(int(_ef[1]) if _ef is not None else ""),
                            enroute_consumed=(int(self._frontier_consumed(_ef))
                                              if _ef is not None else ""),
                            committed_x=(int(_cf[0]) if _cf is not None else ""),
                            committed_y=(int(_cf[1]) if _cf is not None else ""),
                            snap_delta_px=round(float(getattr(
                                self, '_last_snap_delta_px', 0.0)), 1),
                            astar_px=_ap, detour_ratio=_dr,
                            f_preempt=int('preempt' in self._dec_flags),
                            f_early_stop=int('early_stop' in self._dec_flags),
                            f_no_uturn=int('no_uturn' in self._dec_flags),
                            f_force_fallback=int('force_fallback' in self._dec_flags),
                            f_stick=int('stick' in self._dec_flags),
                            f_commit=int('commit' in self._dec_flags),
                            f_nudge=int('nudge' in self._dec_flags),
                        )
                    except Exception:
                        pass
                    self._dec_flags = set()
                # ------------------------------------------------ end [DECLOG]

                response = {
                    'heatmap': heatmap, 'goal_px': goal_px,
                    'req_id': _req_id,                        # [REQ-ID]
                    'selected_score': selected_local_score,
                    'final_heading': final_heading, 'origin_img': origin_img,
                    'timestamp': time.time(), 'prediction_id': prediction_count,
                    'request_pose': current_pose, 'mode': mode,
                    'hallucinations': hallucinations,
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

            except Exception as e:
                print(f"PREDICTION: Worker error: {e}")
                time.sleep(0.2)