"""
intelx.baselines.simple
=======================
Tier-0 (geometry-only, NO learned model) exploration baselines, each exposed as
a drop-in `_prediction_worker` mixin that plugs into your monolith explorer
(UpdatedGoalBasedAutonomousExplorer) exactly like the frontier baseline does.

    NearestFrontierMixin      -- nearest frontier by path length [Yamauchi 1997]. 'nearest'
    CostUtilityFrontierMixin  -- max g * exp(-lambda * L)  [Gonzalez-Banos and
                                 Latombe 2002].                                 'cost_utility'
    RandomFrontierMixin       -- random among the top-N nearest frontiers.       'random'
    GeometricIGFrontierMixin  -- max E[I] - alpha * L  [Stachniss et al. 2005].  'geom_ig'

L is the A* path length to the frontier. g and E[I] are the same ray-cast gain:
the unobserved cells a 360-degree scan from the frontier would cover, cast on
the robot's own map (see raycast_gain).

All four choose among the accessible, unvisited frontiers, as Yamauchi's
frontier-based exploration does: a frontier the robot could not make progress
toward is set aside for a while, and one it has already reached is not chosen
again. A frontier that survives the robot standing on it borders cells no scan
can reach, and returning to it forever is what made a nearest-first robot
circle without gaining coverage.

SUBMITTED_BEHAVIOUR = True in the config restores the rules the submitted
results were produced with: L counted as the number of waypoints in the
post-processed A* path, Cost-Utility as explore_lite's POTENTIAL*L - GAIN*size,
Geometric IG as (unobserved cells in a disk) / L, and a pass with every
frontier beyond the local window ending the run when it found no goal.

USE (in the sweep, per mode):
    from intelx.baselines.simple import NearestFrontierMixin
    from intelx.baselines.frontier import UpdatedGoalBasedAutonomousExplorer as Base
    class E(NearestFrontierMixin, Base): pass
    explorer = E(simulator, model, config)     # model is unused by Tier-0 modes

All four share `_FrontierSweepBase._prediction_worker` (frontier extraction,
blacklist, stuck detection, goal emission, same response contract as the
baseline) and only differ in `_select_frontier(...)`. The selection functions at
the top are pure and unit-tested; the worker is thin plumbing around them.

The worker uses A* path cost when `optimized_astar_path_planner` is importable
(your env), else falls back to Euclidean -- so ranking still works in any env.

THREE FIXES over the original, all in the worker; the selection policies are
untouched:

  1. GOAL REPAIR. A frontier centroid is the MEAN of a DBSCAN cluster, so it can
     land in a wall (or in unknown space, which the cost function also rejected
     because it tested `!= 0` and unknown is -1). Every candidate then scored
     np.inf, _select_frontier returned None, and the worker answered with
     goal_px=None while mode was still 'BASELINE_FRONTIER'. The simulator reads
     that as "no goal yet", sleeps 0.5 s and re-requests; the answer never
     changes, so the robot stands still until STALL_IDLE_S. That is the 1800 s
     stall on all four Tier-0 modes. Candidates are now snapped to a nearby free
     cell and validated through the same world -> image -> world round trip the
     simulator performs, so an accepted goal is one the simulator will accept.

  2. IDLE-PASS COST. extract_frontier_cells + cluster_frontiers ran on EVERY
     loop iteration, nav or not, i.e. DBSCAN over the frontier cells of a
     19M-cell map at ~50 Hz. They now run only when a goal was actually
     requested; idle passes reuse the last result for visualisation.

  3. A* FAN-OUT. Every candidate got a full-map A* call (2-20 s each on a large
     map), and select_random called cost inside sorted() and again for the
     winner. Costs are now memoised per pass, and only the
     BASELINE_MAX_ASTAR_CANDS nearest candidates are scored -- the same
     top-K-by-Euclidean-then-A* approximation the reference frontier baseline
     uses. Set BASELINE_MAX_ASTAR_CANDS = 0 to score every candidate.
"""
import time
import queue
from math import hypot

import numpy as np
from scipy.ndimage import binary_dilation
from sklearn.cluster import DBSCAN

# A* is optional at import time so this module loads for unit tests without the
# heavy sim deps. The worker resolves it lazily.
try:
    from intelx.sim.simulator_astar import optimized_astar_path_planner as _ASTAR
except Exception:
    _ASTAR = None

try:
    from intelx.common import safe_copy_pose as _safe_copy_pose
except Exception:
    def _safe_copy_pose(p):
        return p.copy() if hasattr(p, 'copy') else np.array(p)

try:
    from intelx.common import _transform_image_goal_to_world
except Exception:
    _transform_image_goal_to_world = None


# How many candidates get a full A* call. The rest are dropped by Euclidean
# distance first. 0 disables the cap.
DEFAULT_MAX_ASTAR_CANDS = 12
# How far to look for a free cell when a frontier centroid lands in a wall.
DEFAULT_SNAP_PX = 40
# lambda of Gonzalez-Banos and Latombe, per metre of path: a frontier 5 m
# further away needs e ~ 2.7 times the gain. COST_UTILITY_LAMBDA overrides it.
DEFAULT_LAMBDA_PER_M = 0.2
# alpha of Stachniss et al., square metres of gain one metre of path is worth.
# They set it experimentally; GEOM_IG_ALPHA overrides it.
DEFAULT_ALPHA_M2_PER_M = 1.0


# ---------------------------------------------------------------------------
# Frontier geometry (pure)
# ---------------------------------------------------------------------------
def extract_frontier_cells(occ_map, explored_map):
    """Free, seen cells adjacent to an unexplored cell -> (N,2) array of (y,x)."""
    free_seen = (occ_map == 0) & explored_map
    adj = binary_dilation(~explored_map, structure=np.ones((3, 3), dtype=bool))
    return np.argwhere(free_seen & adj)


def cluster_frontiers(frontier_cells, min_cluster_size=5, merge_dist=15.0):
    """DBSCAN cluster -> list of {x, y, size} (x,y are the cluster centroid)."""
    if len(frontier_cells) == 0:
        return []
    labels = DBSCAN(eps=merge_dist, min_samples=min_cluster_size).fit(frontier_cells).labels_
    out = []
    for lab in set(labels):
        if lab == -1:
            continue
        pts = frontier_cells[labels == lab]
        cy, cx = pts.mean(axis=0)
        out.append({'x': float(cx), 'y': float(cy), 'size': int(len(pts))})
    return out


def path_length_px(path):
    """Length of a waypoint path in pixels, the sum of its segments."""
    if path is None or len(path) < 2:
        return 0.0
    p = np.asarray(path, np.float64)
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum())


def raycast_gain(occ, explored_map, xy, max_r_px, n_beams=720):
    """Unobserved cells a 360-degree scan from xy would cover.

    The information gain of Stachniss et al., where the entropy change of a
    beam is taken as proportional to the unknown cells it covers, and the
    visible new area of Gonzalez-Banos and Latombe. Cast on the robot's own map:
    a beam stops at a wall it has OBSERVED, and runs on through unobserved
    cells, since the robot cannot know what is there. No ground truth beyond
    its own observations is consulted."""
    H, W = explored_map.shape
    ang = np.linspace(0.0, 2.0 * np.pi, int(n_beams), endpoint=False)
    rr = np.arange(1, int(max_r_px) + 1, dtype=np.float64)
    xs = (float(xy[0]) + np.cos(ang)[:, None] * rr[None, :]).astype(np.int64)
    ys = (float(xy[1]) + np.sin(ang)[:, None] * rr[None, :]).astype(np.int64)
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xc, yc = np.clip(xs, 0, W - 1), np.clip(ys, 0, H - 1)
    seen = explored_map[yc, xc]
    stop = (seen & (occ[yc, xc] == 1)) | (~inb)
    first = np.where(stop.any(1), stop.argmax(1), stop.shape[1])
    keep = (np.arange(stop.shape[1])[None, :] < first[:, None]) & inb & (~seen)
    return int(np.unique(yc[keep] * W + xc[keep]).size)


def geometric_ig(frontier, explored_map, sensor_px):
    """Geometric information gain proxy (NO prediction): number of currently
    UNEXPLORED cells within one sensor radius of the frontier. This is the fair
    'information gain without a learned predictor' signal."""
    H, W = explored_map.shape
    fx, fy = int(frontier['x']), int(frontier['y'])
    r = int(sensor_px)
    y0, y1 = max(0, fy - r), min(H, fy + r + 1)
    x0, x1 = max(0, fx - r), min(W, fx + r + 1)
    patch = explored_map[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disk = (yy - fy) ** 2 + (xx - fx) ** 2 <= r * r
    return int(np.count_nonzero(disk & (~patch)))


def free_cell_candidates(occ, xy, max_r_px=DEFAULT_SNAP_PX, max_n=32, min_clear_px=3):
    """Free cells near xy, nearest first, for the caller to validate.

    A DBSCAN centroid is a MEAN, so on a curved or wrap-around frontier it falls
    behind the wall the frontier curves around. The caller cannot simply take the
    nearest free cell either: the goal then makes a world -> image -> world round
    trip through _transform_world_to_image_goal, which ends in int() TRUNCATION,
    so the point the simulator finally tests can sit up to ~1.4 px from the cell
    we picked -- and the nearest free cell to a walled point is by construction
    right against the wall. So we return a ranked list and the caller keeps the
    first entry that survives its own round trip.

    Cells with at least `min_clear_px` of wall clearance come first, since a goal
    hugging a wall also makes the clearance A* work harder.

    (Twin of the same helper in intelx.baselines.tier1. Kept local so Tier-0 has no
    dependency on the Tier-1 client.)
    """
    H, W = occ.shape
    gx, gy = int(round(float(xy[0]))), int(round(float(xy[1])))
    x0, x1 = max(0, gx - max_r_px), min(W, gx + max_r_px + 1)
    y0, y1 = max(0, gy - max_r_px), min(H, gy + max_r_px + 1)
    if x1 <= x0 or y1 <= y0:
        return []
    sub = occ[y0:y1, x0:x1]
    free = (sub == 0)
    if not free.any():
        return []
    roomy = free
    if min_clear_px and min_clear_px > 0:
        try:
            from scipy.ndimage import distance_transform_edt
            cand = free & (distance_transform_edt(free) >= float(min_clear_px))
            if cand.any():
                roomy = cand
        except Exception:
            pass
    ys, xs = np.nonzero(roomy)
    d2 = (xs + x0 - gx) ** 2 + (ys + y0 - gy) ** 2
    order = np.argsort(d2)[:max_n]
    out = [(float(xs[i] + x0), float(ys[i] + y0)) for i in order]
    if 0 <= gx < W and 0 <= gy < H and occ[gy, gx] == 0:
        out.insert(0, (float(xy[0]), float(xy[1])))
    return out


# ---------------------------------------------------------------------------
# Selection policies (pure -- these are what unit tests target)
# `cost` is a callable robot_px, target_xy -> path cost (float, np.inf if none).
# ---------------------------------------------------------------------------
def select_nearest(frontiers, robot_px, cost):
    best, best_c = None, np.inf
    for f in frontiers:
        c = cost(robot_px, (f['x'], f['y']))
        if c < best_c:
            best_c, best = c, f
    return best, best_c


def select_cost_utility(frontiers, robot_px, cost, potential_scale, gain_scale, res):
    """explore_lite score = POTENTIAL*travel_m - GAIN*frontier_size ; minimise.
    The submitted Cost-utility rule (SUBMITTED_BEHAVIOUR)."""
    best, best_s, best_c = None, np.inf, np.inf
    for f in frontiers:
        c = cost(robot_px, (f['x'], f['y']))
        if not np.isfinite(c):
            continue
        s = potential_scale * (c * res) - gain_scale * f['size']
        if s < best_s:
            best_s, best, best_c = s, f, c
    return best, best_c


def select_random(frontiers, robot_px, cost, rng, top_n):
    ranked = sorted(frontiers, key=lambda f: cost(robot_px, (f['x'], f['y'])))
    ranked = [f for f in ranked if np.isfinite(cost(robot_px, (f['x'], f['y'])))]
    if not ranked:
        return None, np.inf
    pool = ranked[:max(1, min(top_n, len(ranked)))]
    f = pool[int(rng.integers(len(pool)))]
    return f, cost(robot_px, (f['x'], f['y']))


def select_gain_discounted(frontiers, robot_px, cost, gain, lam, res):
    """Gonzalez-Banos and Latombe: maximise g * exp(-lambda * L), L in metres."""
    best, best_u, best_c = None, -np.inf, np.inf
    for f in frontiers:
        c = cost(robot_px, (f['x'], f['y']))
        if not np.isfinite(c):
            continue
        u = gain(f) * np.exp(-lam * c * res)
        if u > best_u:
            best_u, best, best_c = u, f, c
    return best, best_c


def select_gain_minus_cost(frontiers, robot_px, cost, gain, alpha, res):
    """Stachniss et al.: maximise E[I] - alpha * L, with E[I] in square metres
    and L in metres."""
    best, best_u, best_c = None, -np.inf, np.inf
    for f in frontiers:
        c = cost(robot_px, (f['x'], f['y']))
        if not np.isfinite(c):
            continue
        u = gain(f) * res * res - alpha * c * res
        if u > best_u:
            best_u, best, best_c = u, f, c
    return best, best_c


def select_geometric_ig(frontiers, robot_px, cost, ig):
    """Maximise geometric_ig / travel_cost (distance-normalised info gain).
    The submitted Geometric IG rule (SUBMITTED_BEHAVIOUR)."""
    best, best_u, best_c = None, -np.inf, np.inf
    for f in frontiers:
        c = cost(robot_px, (f['x'], f['y']))
        if not np.isfinite(c) or c <= 0:
            continue
        u = ig(f) / c
        if u > best_u:
            best_u, best, best_c = u, f, c
    return best, best_c


# ---------------------------------------------------------------------------
# Shared worker
# ---------------------------------------------------------------------------
class _FrontierSweepBase:
    """Frontier-exploration worker shared by all Tier-0 baselines. Provides
    `_prediction_worker`; subclasses implement `_select_frontier`.

    Reuses the explorer's `_transform_world_to_image_goal`, `simulator`, queues,
    locks, and `config`. Emits the same response dict the baseline emits, so the
    simulation/visualisation threads need no changes. Sets goal_px=None +
    mode='BASELINE_GLOBAL' when frontiers are exhausted so the sim terminates."""

    #: subclasses set this for logging/metrics
    BASELINE_MODE = "frontier"

    #: consecutive passes with no usable goal before the run is called exhausted
    MAX_NO_GOAL_PASSES = 5

    def _submitted(self):
        return bool(getattr(getattr(self, 'config', None), 'SUBMITTED_BEHAVIOUR', False))

    def _cost_fn(self):
        """robot_px, target_xy -> path cost in PIXELS (A* length if available,
        else Euclidean), memoised for this pass. Under SUBMITTED_BEHAVIOUR the
        cost is the number of waypoints in the post-processed path, which is
        what the submitted results used.

        The original recomputed a full-map A* for every call, and select_random
        called it three times per candidate. On a 4399 px map an A* run costs
        seconds, so a single decision could take minutes.
        """
        occ = self.simulator.occ
        cache = {}
        submitted = self._submitted()

        def cost(robot_px, target_xy):
            key = (int(target_xy[0]), int(target_xy[1]))
            if key in cache:
                return cache[key]
            if _ASTAR is not None:
                # free cells only: unknown (-1) is not a place we can aim at
                if occ[key[1], key[0]] != 0:
                    v = np.inf
                else:
                    start = (int(robot_px[0]), int(robot_px[1]))
                    if submitted:
                        p = _ASTAR(occ, start, key)
                        v = len(p) if p else np.inf
                    else:
                        p = _ASTAR(occ, start, key, skip_postprocess=True)
                        v = path_length_px(p) if p else np.inf
            else:
                v = hypot(key[0] - robot_px[0], key[1] - robot_px[1])
            cache[key] = v
            return v

        return cost

    def _resolve_goal_px(self, world_xy, pose):
        """A frontier point -> an image goal the SIMULATOR will accept.

        The simulator does not test the point we send. It tests
            int(_transform_image_goal_to_world(goal_px, request_pose, ...))
        and rejects it with "Goal is inside obstacle!" if that cell is a wall.
        Because _transform_world_to_image_goal truncates to int, that cell can
        differ from the one we chose. So reproduce the simulator's own test here
        and only return a goal that passes it.

        Returns (goal_px, (wx, wy)) or (None, None).
        """
        occ = self.simulator.occ
        H, W = occ.shape
        win = self.config.sim.get('WIN_PX', 400)
        res = self.config.sim.get('RESOLUTION', 0.05)
        snap_px = int(getattr(self.config, 'BASELINE_SNAP_PX', DEFAULT_SNAP_PX))

        for cx, cy in free_cell_candidates(occ, world_xy, snap_px):
            gpx = self._transform_world_to_image_goal((cx, cy), tuple(pose), win)
            if _transform_image_goal_to_world is None:
                return gpx, (cx, cy)
            bx, by = _transform_image_goal_to_world(gpx, tuple(pose), win, res)
            xi, yi = int(bx), int(by)
            if 0 <= xi < W and 0 <= yi < H and occ[yi, xi] == 0:
                return gpx, (cx, cy)
        return None, None

    def _select_frontier(self, frontiers, robot_px, cost):
        raise NotImplementedError

    def _prediction_worker(self):
        print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: worker starting...")
        res = self.config.sim.get('RESOLUTION', 0.05)
        min_size = int(getattr(self.config, 'MIN_FRONTIER_SIZE', 5))
        local_win_px = float(getattr(self.config, 'LOCAL_FRONTIER_WINDOW_M', 10.0)) / res
        max_astar = int(getattr(self.config, 'BASELINE_MAX_ASTAR_CANDS',
                                DEFAULT_MAX_ASTAR_CANDS))
        bl, bl_timeout = [], 45.0
        recent, last_sample_t = [], 0.0
        last_goal_world = None
        rng = np.random.default_rng(int(getattr(self.config, 'BASELINE_SEED', 0)))
        self._rng = rng
        visited = []                      # frontiers already reached
        visit_px = float(getattr(self.config, 'VISITED_FRONTIER_M', 1.0)) / res
        prediction_count = 0
        n_no_goal = 0
        frontier_centroids_for_viz = []

        while self.running.is_set():
            try:
                nav = False
                try:
                    req = self.prediction_request_queue.get_nowait()
                    nav = True
                    current_pose = req.get('pose')
                except queue.Empty:
                    current_pose = _safe_copy_pose(self.simulator.pose)

                now = time.time()
                curr_px = (int(current_pose[0]), int(current_pose[1]))

                # stuck detection -> blacklist current goal. Sampled on a TIMER,
                # not once per loop: the loop spins at ~50 Hz, so 20 samples used
                # to cover 0.4 s, which is far too short a window to mean "stuck"
                # and fired every time the robot paused for a new goal.
                if now - last_sample_t >= 0.5:
                    last_sample_t = now
                    recent.append((current_pose[0], current_pose[1]))
                    if len(recent) > 20:
                        recent.pop(0)
                    if last_goal_world is not None and len(recent) >= 20:
                        moved = hypot(recent[-1][0] - recent[0][0],
                                      recent[-1][1] - recent[0][1])
                        if moved < 3.0:
                            bl.append((last_goal_world[0], last_goal_world[1], now))
                            recent.clear()
                            last_goal_world = None
                bl = [b for b in bl if now - b[2] < bl_timeout]

                mode = "BASELINE_FRONTIER"
                chosen_goal_px = None

                # Frontier extraction is EXPENSIVE (DBSCAN over every frontier
                # cell of a multi-million-cell map) and its result is only ever
                # consumed on a nav pass. It used to run on every loop iteration,
                # i.e. ~50 times a second whether or not anyone asked for a goal.
                if nav:
                    if (not self._submitted() and last_goal_world is not None
                            and hypot(curr_px[0] - last_goal_world[0],
                                      curr_px[1] - last_goal_world[1]) <= visit_px):
                        visited.append(last_goal_world)
                    cells = extract_frontier_cells(self.simulator.occ,
                                                   self.simulator.explored)
                    frontiers = cluster_frontiers(cells, min_size)
                    if visited and not self._submitted():
                        frontiers = [f for f in frontiers
                                     if not any(hypot(f['x'] - v[0], f['y'] - v[1]) < visit_px
                                                for v in visited)]
                    frontier_centroids_for_viz = [(f['x'], f['y']) for f in frontiers]

                    bl_px = 2.0 / res
                    valid = [f for f in frontiers
                             if not any(hypot(f['x'] - b[0], f['y'] - b[1]) < bl_px
                                        for b in bl)]
                    if not valid:
                        valid = frontiers
                    for f in valid:
                        f['dist_euclidean'] = hypot(f['x'] - curr_px[0],
                                                    f['y'] - curr_px[1])
                    cands = [f for f in valid if f['dist_euclidean'] <= local_win_px]
                    if not cands:
                        # BASELINE_GLOBAL is also what tells the simulator the
                        # run is over. Under the submitted behaviour a pass with
                        # every frontier beyond the local window therefore ended
                        # the run whenever it found no goal; otherwise the mode
                        # says so only once exploration really is exhausted.
                        if self._submitted():
                            mode = "BASELINE_GLOBAL"
                        cands = valid

                    # Repair each candidate BEFORE scoring. A centroid in a wall
                    # scored np.inf, and when every candidate did, _select_frontier
                    # returned None and the worker answered goal_px=None while
                    # still claiming BASELINE_FRONTIER -- which the simulator
                    # retries forever. Snapping also means the cost function is
                    # ranking points the robot can actually be sent to.
                    cands.sort(key=lambda f: f['dist_euclidean'])
                    usable = []
                    for f in cands:
                        gpx, w = self._resolve_goal_px((f['x'], f['y']), current_pose)
                        if gpx is None:
                            continue
                        f = dict(f)
                        f['x'], f['y'] = w[0], w[1]      # snapped, simulator-valid
                        f['goal_px'] = gpx
                        usable.append(f)
                        if max_astar > 0 and len(usable) >= max_astar:
                            break

                    best = None
                    if usable:
                        cost = self._cost_fn()
                        best, _c = self._select_frontier(usable, curr_px, cost)

                        # Every selection policy returns None when every candidate
                        # scores np.inf, which happens when A* fails on all of
                        # them -- and _cost_fn falls back to np.inf for any target
                        # optimized_astar_path_planner cannot solve, which on a
                        # 4399x4399 map is not rare for long paths. Left alone,
                        # five such passes in a row emit BASELINE_GLOBAL and the
                        # run is declared exhausted while frontiers are still on
                        # the map: nearest/random/frontier/geom_ig all quit at
                        # 15-45% coverage on 50010536_PLAN4 while cost_utility
                        # finished the same map from the same spawn.
                        #
                        # A frontier we could not COST is not a frontier that does
                        # not exist. Fall back to the nearest usable one by
                        # straight-line distance and let the simulator's own
                        # planner (astar_with_clearance, which retries with
                        # optimized_astar) work out the route.
                        if best is None:
                            best = min(usable, key=lambda f: f['dist_euclidean'])
                            self._n_astar_fallback = getattr(
                                self, '_n_astar_fallback', 0) + 1
                            print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: A* could "
                                  f"not cost any of {len(usable)} candidate(s); "
                                  f"taking the nearest by straight line "
                                  f"({best['dist_euclidean']:.0f}px) rather than "
                                  f"declaring exhaustion")

                    if best is not None:
                        chosen_goal_px = best['goal_px']
                        last_goal_world = (best['x'], best['y'])
                        n_no_goal = 0
                    elif not frontiers and not self._submitted():
                        mode = "BASELINE_GLOBAL"
                        print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: no frontier "
                              f"left -> exploration exhausted")
                    else:
                        last_goal_world = None
                        n_no_goal += 1
                        # Nothing reachable. Retry a few times (frontiers move as
                        # the map fills), then say the run is exhausted rather
                        # than answer None forever and hang at the spawn.
                        if n_no_goal >= self.MAX_NO_GOAL_PASSES:
                            mode = "BASELINE_GLOBAL"
                            # Reaching here now means NO frontier could be
                            # resolved to a simulator-acceptable cell at all, for
                            # MAX_NO_GOAL_PASSES passes running -- not merely that
                            # A* failed to cost them.
                            print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: no "
                                  f"frontier resolved to a reachable cell in "
                                  f"{n_no_goal} passes ({len(frontiers)} "
                                  f"frontier(s) found) -> exploration exhausted")
                        else:
                            print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: no "
                                  f"usable goal ({len(frontiers)} frontier(s), "
                                  f"{len(cands)} candidate(s)), retry "
                                  f"{n_no_goal}/{self.MAX_NO_GOAL_PASSES}")

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
                    'local_components': {'frontier_centroids': frontier_centroids_for_viz},
                }
                with self.data_lock:
                    self.latest_prediction_data = response
                if nav:
                    while not self.prediction_response_queue.empty():
                        try:
                            self.prediction_response_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.prediction_response_queue.put(response)

                prediction_count += 1
                time.sleep(0.02)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"SWEEP BASELINE [{self.BASELINE_MODE}]: worker error: {e}")
                time.sleep(0.2)


class NearestFrontierMixin(_FrontierSweepBase):
    BASELINE_MODE = "nearest"
    def _select_frontier(self, frontiers, robot_px, cost):
        return select_nearest(frontiers, robot_px, cost)


class _GainMixin:
    """The ray-cast gain at a frontier, from the robot's own map."""
    def _gain_fn(self):
        res = self.config.sim.get('RESOLUTION', 0.05)
        r_px = float(getattr(self.config, 'LASER_SCAN_R', 10)) / res
        occ, explored = self.simulator.occ, self.simulator.explored
        return lambda f: raycast_gain(occ, explored, (f['x'], f['y']), r_px)


class CostUtilityFrontierMixin(_GainMixin, _FrontierSweepBase):
    """Gonzalez-Banos and Latombe [13]: max g * exp(-lambda * L)."""
    BASELINE_MODE = "cost_utility"
    def _select_frontier(self, frontiers, robot_px, cost):
        res = self.config.sim.get('RESOLUTION', 0.05)
        if self._submitted():
            return select_cost_utility(
                frontiers, robot_px, cost,
                float(getattr(self.config, 'POTENTIAL_SCALE', 3.0)),
                float(getattr(self.config, 'GAIN_SCALE', 1.0)), res)
        return select_gain_discounted(
            frontiers, robot_px, cost, self._gain_fn(),
            float(getattr(self.config, 'COST_UTILITY_LAMBDA', DEFAULT_LAMBDA_PER_M)), res)


class RandomFrontierMixin(_FrontierSweepBase):
    BASELINE_MODE = "random"
    def _select_frontier(self, frontiers, robot_px, cost):
        return select_random(frontiers, robot_px, cost, self._rng,
                             int(getattr(self.config, 'RANDOM_TOP_N', 5)))


class GeometricIGFrontierMixin(_GainMixin, _FrontierSweepBase):
    """Stachniss et al. [12]: max E[I] - alpha * L, E[I] by ray casting."""
    BASELINE_MODE = "geom_ig"
    def _select_frontier(self, frontiers, robot_px, cost):
        res = self.config.sim.get('RESOLUTION', 0.05)
        if self._submitted():
            sensor_px = float(getattr(self.config, 'LASER_SCAN_R', 10)) / res
            ig = lambda f: geometric_ig(f, self.simulator.explored, sensor_px)
            return select_geometric_ig(frontiers, robot_px, cost, ig)
        return select_gain_minus_cost(
            frontiers, robot_px, cost, self._gain_fn(),
            float(getattr(self.config, 'GEOM_IG_ALPHA', DEFAULT_ALPHA_M2_PER_M)), res)


# Registry the sweep imports.
BASELINE_MIXINS = {
    "nearest": NearestFrontierMixin,
    "cost_utility": CostUtilityFrontierMixin,
    "random": RandomFrontierMixin,
    "geom_ig": GeometricIGFrontierMixin,
}


# ---------------------------------------------------------------------------
# self-test: pure functions only, no sim
# ---------------------------------------------------------------------------
def _selftest():
    print("== frontier geometry ==")
    occ = np.zeros((60, 60), np.int8)
    occ[30, :] = 1
    explored = np.zeros((60, 60), bool)
    explored[:30, :] = True
    cells = extract_frontier_cells(occ, explored)
    assert len(cells) > 0
    fr = cluster_frontiers(cells, 5)
    print("   %d frontier cell(s) -> %d cluster(s)" % (len(cells), len(fr)))

    print("== goal repair ==")
    occ2 = np.zeros((100, 100), np.int8)
    occ2[40:60, 40:60] = 1
    assert free_cell_candidates(occ2, (10.5, 10.5))[0] == (10.5, 10.5), "free stays"
    c = free_cell_candidates(occ2, (50.0, 50.0), 40)
    assert c and all(occ2[int(y), int(x)] == 0 for x, y in c), "walled -> free cells"
    print("   walled centroid (50,50) -> %d candidate(s), nearest %s" % (len(c), c[0]))
    assert free_cell_candidates(np.ones((40, 40), np.int8), (20.0, 20.0), 10) == []
    print("   fully walled -> no candidates (caller must retry, not hang)")

    print("== selection policies ==")
    fs = [{'x': 10.0, 'y': 0.0, 'size': 5},
          {'x': 40.0, 'y': 0.0, 'size': 90},
          {'x': 90.0, 'y': 0.0, 'size': 10}]
    cost = lambda r, t: abs(t[0] - r[0])
    b, _ = select_nearest(fs, (0, 0), cost)
    assert b['x'] == 10.0
    b, _ = select_cost_utility(fs, (0, 0), cost, 3.0, 1.0, 0.05)
    assert b['x'] == 40.0, "big frontier must beat a near tiny one"
    b, _ = select_geometric_ig(fs, (0, 0), cost, lambda f: f['size'])
    assert b['x'] == 40.0
    print("   nearest -> x=10, cost_utility -> x=40, geom_ig -> x=40  OK")

    # all-unreachable must return None, not raise: that is the case the worker
    # now turns into a retry-then-exhaust instead of an infinite hang
    inf_cost = lambda r, t: np.inf
    for fn in (select_nearest,
               lambda f, r, c: select_cost_utility(f, r, c, 3.0, 1.0, 0.05),
               lambda f, r, c: select_geometric_ig(f, r, c, lambda x: 1.0),
               lambda f, r, c: select_random(f, r, c, np.random.default_rng(0), 5)):
        b, _ = fn(fs, (0, 0), inf_cost)
        assert b is None, fn
    print("   every policy returns None when nothing is reachable  OK")

    print("== cost memoisation ==")
    calls = {'n': 0}
    class _Fake(_FrontierSweepBase):
        def __init__(s):
            class _S: occ = np.zeros((50, 50), np.int8)
            s.simulator = _S()
    f = _Fake()
    global _ASTAR
    saved = _ASTAR
    def fake_astar(occ, a, b, **kw):
        calls['n'] += 1
        return [a, b]
    _ASTAR = fake_astar
    c = f._cost_fn()
    for _ in range(5):
        c((0, 0), (10, 10))
    _ASTAR = saved
    assert calls['n'] == 1, calls
    print("   5 identical queries -> %d A* call(s)  OK" % calls['n'])

    print("== travel is path length, not waypoints ==")
    assert abs(path_length_px([(0, 0), (3, 4), (3, 10)]) - 11.0) < 1e-9
    assert path_length_px([(1, 1)]) == 0.0 and path_length_px(None) == 0.0
    c = f._cost_fn()
    _ASTAR = lambda occ, a, b, **kw: [a, (a[0] + 30, a[1]), b]
    v = c((0, 0), (30, 40))
    _ASTAR = saved
    assert abs(v - 70.0) < 1e-9, v       # 30 + 40, where waypoints gave 3
    print("   (0,0)->(30,0)->(30,40) costs %.0f px  OK" % v)

    print("== ray-cast gain ==")
    occ_r = np.zeros((101, 101), np.int8)
    occ_r[:, 60] = 1                                 # a wall at x = 60
    seen = np.zeros((101, 101), bool)
    g_unknown = raycast_gain(occ_r, seen, (50, 50), 20)
    seen[:, 60] = True                                # the wall is now observed
    g_walled = raycast_gain(occ_r, seen, (50, 50), 20)
    seen[45:56, 30:60] = True                         # a strip is explored
    g_strip = raycast_gain(occ_r, seen, (50, 50), 20)
    seen[30:71, 30:60] = True                         # the near side is explored
    g_none = raycast_gain(occ_r, seen, (50, 50), 20)
    assert g_unknown > g_walled > g_strip > g_none == 0, (
        g_unknown, g_walled, g_strip, g_none)
    print("   unknown %d > wall observed %d > strip explored %d > all seen %d  OK"
          % (g_unknown, g_walled, g_strip, g_none))

    print("== cited utilities ==")
    fr = [dict(x=10.0, y=0.0, g=100.0), dict(x=200.0, y=0.0, g=400.0)]
    cst = lambda r, t: abs(t[0] - r[0])               # pixels
    gain = lambda f_: f_['g']
    near, _ = select_gain_discounted(fr, (0, 0), cst, gain, lam=0.2, res=0.05)
    far, _ = select_gain_discounted(fr, (0, 0), cst, gain, lam=0.01, res=0.05)
    assert near['x'] == 10.0 and far['x'] == 200.0
    near2, _ = select_gain_minus_cost(fr, (0, 0), cst, gain, alpha=1.0, res=0.05)
    far2, _ = select_gain_minus_cost(fr, (0, 0), cst, gain, alpha=0.01, res=0.05)
    assert near2['x'] == 10.0 and far2['x'] == 200.0
    print("   g*exp(-lambda*L) and E[I] - alpha*L trade gain for distance  OK")

    print("\n*** baselines_simple self-test PASSED ***")


if __name__ == '__main__':
    _selftest()