"""
intelx.baselines.tier1  --  Tier-1 baselines (MapEx / UPEN / IG-Hector)
===================================================================
These run in YOUR simulator, on YOUR maps and spawns, with YOUR sensor and
metrics -- only the PLANNING RULE comes from MapEx. The rule itself is not
reimplemented here: every decision is made by the authors' own code running in
their `lama` env behind a socket (see mapex_service.py). This module is purely
the client: format conversion, coordinate conversion, and a `_prediction_worker`
that fits your explorer's interface.

    your sim (py3.9)                     mapex_service.py (py3.6 lama env)
      occ + explored @0.05 m/px  ---->   pad 500, LaMa ensemble x3,
      as 0/0.5/1 @0.1 m/px               their score_frontiers / RRT / hector
      goal in your pixel frame   <----   best frontier (unpadded, 0.1 m/px)

WHY THIS SHAPE
    Reimplementing their probabilistic raycast + visibility mask + ensemble
    variance is exactly where a baseline quietly becomes weaker than published.
    Calling their code removes that risk; all we own is conversion, which is
    unit-testable (and tested at the bottom of this file).

HONEST DEVIATION -- read before quoting hectoraug numbers
    `visvarprob` (MapEx) and `upen` are GOAL-BASED: they emit a target, which is
    exactly what your state machine consumes, so they transfer faithfully.
    `hectoraug` in the original is NOT goal-based -- it follows the gradient of a
    Hector exploration transform every step. Your simulator drives to goals, so
    here it is adapted: the service returns its best frontier and we navigate to
    it. That is a real deviation and must be stated in the paper. If IG-Hector
    matters as a headline baseline, run it natively in their repo instead.

SECOND DEVIATION -- goal snapping (see snap_goal_to_free_cell)
    Their frontier centre can land on an occupied cell in your finer grid. We
    move it to the nearest free cell. This changes WHERE within the chosen
    frontier the robot drives, never WHICH frontier was chosen, and it is the
    same repair your own policy applies to its own frontier centroids
    (intelx.policy_ablation._frontier_boundary_point). State it in the paper.

TERMINATION SEMANTICS -- do not weaken this
    `BASELINE_GLOBAL` is the signal the monolith reads as "exploration is over";
    it sets exploration_complete and ends the episode. It must therefore be
    emitted ONLY when the service reports a genuine `no_frontiers`, or after
    MAX_CONSECUTIVE_FAILURES real failures in a row. A transient service error
    emits `TIER1_RETRY` instead, which the monolith handles by re-requesting.
    Collapsing the two (the previous behaviour) meant one socket hiccup ended a
    run and it was scored as a completed episode.

USAGE
    1. In the lama env:   python mapex_service.py --port 5555
    2. In run_sweep2.py SWEEP:
           MAPEX_SERVICE_HOST = "127.0.0.1"
           MAPEX_SERVICE_PORT = 5555
           MODES = [..., "mapex", "upen", "hectoraug"]
           MODE_SENSORS = {"mapex": ["intelx", "mapex"]}
           MAPEX_UPEN_CONFIG = {..., "reach_horizon": 50,
                                "upen_mode": "exploration"}
           TIER1_SNAP_PX = 40          # optional, default 40 (2 m at 0.05 m/px)
"""
import queue
import socket
import struct
import pickle
import time

import numpy as np

PROTO = 2                       # py3.6-compatible pickle
THEIR_RES = 0.1                 # m/px the LaMa models were trained at

# How many consecutive service failures before we accept the run is dead and
# terminate cleanly rather than spinning until MAX_WALL_S.
MAX_CONSECUTIVE_FAILURES = 5

# How far to look for a free cell when their goal lands in a wall. 40 px at
# 0.05 m/px is 2 m, comfortably more than the slack the 0.1 -> 0.05 resolution
# change introduces, and still local to the chosen frontier.
DEFAULT_SNAP_PX = 40


# ---------------------------------------------------------------- transport

# ---------------------------------------------------------------------------
# NumPy-version-independent array codec.
# NumPy 2.x pickles ndarrays with a reference to `numpy._core`, which does not
# exist in NumPy 1.x -- so a py3.9/np2 client and a py3.6/np1 service cannot
# exchange pickled arrays (ModuleNotFoundError: No module named 'numpy._core').
# We therefore never pickle an ndarray: arrays are sent as raw bytes plus dtype
# string and shape, which every NumPy version reconstructs identically.
# ---------------------------------------------------------------------------
_ARR_TAG = '__nd__'


def _enc(obj):
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        return {_ARR_TAG: True, 'b': a.tobytes(),
                'd': a.dtype.str, 's': tuple(int(x) for x in a.shape)}
    if isinstance(obj, dict):
        return dict((k, _enc(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        t = [_enc(v) for v in obj]
        return t if isinstance(obj, list) else tuple(t)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _dec(obj):
    if isinstance(obj, dict):
        if obj.get(_ARR_TAG):
            return np.frombuffer(obj['b'], dtype=np.dtype(obj['d'])).reshape(obj['s'])
        return dict((k, _dec(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        t = [_dec(v) for v in obj]
        return t if isinstance(obj, list) else tuple(t)
    return obj


def _send(sock, obj):
    data = pickle.dumps(_enc(obj), protocol=PROTO)
    sock.sendall(struct.pack('>I', len(data)) + data)


def _recv_all(sock, n):
    buf = b''
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise EOFError('service closed the connection')
        buf += c
    return buf


def _recv(sock):
    (n,) = struct.unpack('>I', _recv_all(sock, 4))
    return _dec(pickle.loads(_recv_all(sock, n)))


class MapExClient(object):
    """Reconnecting client. One per explorer instance (one per worker process)."""

    def __init__(self, host='127.0.0.1', port=5555, timeout=600.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self.sock = s

    def request(self, msg, retries=2):
        for attempt in range(retries + 1):
            try:
                if self.sock is None:
                    self._connect()
                _send(self.sock, msg)
                return _recv(self.sock)
            except Exception as e:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
                if attempt == retries:
                    return {'ok': False, 'goal': None, 'error': 'service: %s' % e,
                            'n_frontiers': 0, 'fallback': None}
                time.sleep(0.5)

    def ping(self):
        return self.request({'cmd': 'ping'})


# ---------------------------------------------------------------- conversion
def occ_explored_to_obs_map(occ, explored, src_res=0.05, dst_res=THEIR_RES):
    """YOUR (occ, explored) -> THEIR obs_map convention, at THEIR resolution.

    Their convention (sim_utils / lama input):
        0.0 = known free, 0.5 = unknown, 1.0 = known occupied
    Yours:
        occ: 0 free, 1 wall, -1 unknown-in-groundtruth;  explored: bool

    Anything not yet explored is 0.5 regardless of ground truth -- the planner
    must only ever see what the robot has actually observed.

    Downsampling uses a MAX over each block for occupancy so thin walls survive
    (a mean would dissolve a 1-px wall into free space and let raycasts leak
    through it, which would silently flatter the baseline's information gain).
    """
    obs = np.full(occ.shape, 0.5, dtype=np.float32)
    ex = explored.astype(bool)
    obs[ex & (occ == 0)] = 0.0
    obs[ex & (occ == 1)] = 1.0

    step = int(round(dst_res / src_res))
    if step <= 1:
        return obs
    H, W = obs.shape
    H2, W2 = (H // step) * step, (W // step) * step
    blocks = obs[:H2, :W2].reshape(H2 // step, step, W2 // step, step)
    occ_any = (blocks == 1.0).any(axis=(1, 3))          # wall wins
    free_any = (blocks == 0.0).any(axis=(1, 3))         # then free
    out = np.full(occ_any.shape, 0.5, dtype=np.float32)
    out[free_any] = 0.0
    out[occ_any] = 1.0
    return out


def their_rc_to_your_xy(rc, src_res=THEIR_RES, dst_res=0.05):
    """(row, col) in their grid -> (x, y) in yours. Note the axis swap: their
    arrays are (row, col) = (y, x); your world coords are (x, y)."""
    scale = src_res / dst_res
    return (float(rc[1]) * scale, float(rc[0]) * scale)


def your_xy_to_their_rc(xy, src_res=0.05, dst_res=THEIR_RES):
    scale = src_res / dst_res
    return [float(xy[1]) * scale, float(xy[0]) * scale]


def snap_goal_to_free_cell(occ, xy, max_r_px=DEFAULT_SNAP_PX):
    """Move a goal that landed in a wall to the nearest free cell.

    WHY THIS IS NEEDED
        Their frontier centre is the MEAN of a cluster of frontier cells,
        computed on their 0.1 m/px grid. Two things can put it inside an
        obstacle in your 0.05 m/px grid: the mean of a curved or wrap-around
        frontier falls behind the wall it curves around, and the 2x upsample
        adds up to a their-pixel of slack. The simulator then prints "Goal is
        inside obstacle!", drops the goal and re-requests. The service is
        deterministic and the robot has not moved, so it returns the IDENTICAL
        goal, and the run livelocks: one full planning call burned per
        rejection, no motion, no goal_count increment, until the harness
        declares a stall. (It escapes eventually only because each rejected
        request used to append another duplicate pose to pose_list, perturbing
        their scoring until the winner flipped. That was an accident, not a
        recovery, and the duplicate is no longer appended.)

    WHAT IT DOES AND DOES NOT CHANGE
        It does NOT change which frontier their planner selected: the nearest
        free cell to a frontier centre is on the same frontier. It changes only
        where within that frontier the robot aims, and it repairs an artefact of
        the resolution change WE imposed on them. Your own policy applies the
        same repair to its own frontier centroids
        (intelx.policy_ablation._frontier_boundary_point, the [FRONTIER-SNAP] block), so
        both sides of the comparison are treated identically.

    Returns (x, y) floats, or None if there is no free cell within max_r_px
    (in which case the caller should treat the reply as a failure, not as
    exhaustion).
    """
    H, W = occ.shape
    gx, gy = int(round(float(xy[0]))), int(round(float(xy[1])))

    if 0 <= gx < W and 0 <= gy < H and occ[gy, gx] == 0:
        return (float(xy[0]), float(xy[1]))          # already fine, leave it

    x0, x1 = max(0, gx - max_r_px), min(W, gx + max_r_px + 1)
    y0, y1 = max(0, gy - max_r_px), min(H, gy + max_r_px + 1)
    if x1 <= x0 or y1 <= y0:
        return None                                   # goal far outside the map

    sub = occ[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub == 0)
    if len(xs) == 0:
        return None
    d2 = (xs + x0 - gx) ** 2 + (ys + y0 - gy) ** 2
    i = int(np.argmin(d2))
    return (float(xs[i] + x0), float(ys[i] + y0))


def free_cell_candidates(occ, xy, max_r_px=DEFAULT_SNAP_PX, max_n=64,
                         min_clear_px=3):
    """Free cells near xy, nearest first, for the caller to validate.

    snap_goal_to_free_cell picks the single nearest free cell, which is not
    enough: the goal then makes a world -> image -> world round trip through
    _transform_world_to_image_goal, which ends in int() TRUNCATION, so the point
    the simulator finally tests can sit up to ~1.4 px away from the cell we
    chose. The nearest free cell to a walled point is by construction right
    against the wall, so that drift puts it straight back inside the obstacle
    and the rejection loop resumes. The caller therefore walks this list and
    keeps the first candidate that survives its own round trip.

    Cells with at least `min_clear_px` of wall clearance are offered first (a
    goal hugging a wall also makes the clearance A* work harder), falling back
    to every free cell when the neighbourhood is too tight for that.
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
            clear = distance_transform_edt(free)
            cand = free & (clear >= float(min_clear_px))
            if cand.any():
                roomy = cand
        except Exception:
            pass

    ys, xs = np.nonzero(roomy)
    d2 = (xs + x0 - gx) ** 2 + (ys + y0 - gy) ** 2
    order = np.argsort(d2)[:max_n]
    out = [(float(xs[i] + x0), float(ys[i] + y0)) for i in order]

    # the untouched goal first when it is already free, so a good goal is never
    # perturbed just because a roomier cell sits nearby
    if 0 <= gx < W and 0 <= gy < H and occ[gy, gx] == 0:
        out.insert(0, (float(xy[0]), float(xy[1])))
    return out


# ---------------------------------------------------------------- mixins
class _Tier1Base(object):
    """Shared `_prediction_worker`: ask the service for a goal, hand it to the
    simulator in the image frame it expects. Mirrors the Tier-0 mixins so the
    surrounding machinery (queues, response dict, termination) is identical."""

    TIER1_MODE = None                     # 'visvarprob' | 'upen' | 'hectoraug'

    def _resolve_goal_px(self, world_xy, pose, win_px, res, snap_px):
        """Their world goal -> an image goal the simulator will actually accept.

        The simulator does NOT test the point we send. It tests
            int(_transform_image_goal_to_world(goal_px, request_pose, ...))
        and rejects it with "Goal is inside obstacle!" if that cell is a wall.
        Because _transform_world_to_image_goal truncates to int, that cell can
        differ from the one we picked. So we reproduce the simulator's own
        computation here and only return a goal that passes it. Anything else
        restarts the reject-and-re-request loop, which costs a full planning
        call per iteration while the robot stands still.

        Returns (goal_px, snapped) with goal_px None if nothing nearby works.
        """
        from intelx.common import _transform_image_goal_to_world

        occ = self.simulator.occ
        H, W = occ.shape
        wx, wy = float(world_xy[0]), float(world_xy[1])

        for cx, cy in free_cell_candidates(occ, (wx, wy), snap_px):
            gpx = self._transform_world_to_image_goal((cx, cy), tuple(pose), win_px)
            bx, by = _transform_image_goal_to_world(gpx, tuple(pose), win_px, res)
            xi, yi = int(bx), int(by)
            if 0 <= xi < W and 0 <= yi < H and occ[yi, xi] == 0:
                moved = abs(cx - wx) > 1e-6 or abs(cy - wy) > 1e-6
                return gpx, moved
        return None, False

    def _tier1_client(self):
        c = getattr(self, '_mapex_client', None)
        if c is None:
            c = MapExClient(getattr(self.config, 'MAPEX_SERVICE_HOST', '127.0.0.1'),
                            int(getattr(self.config, 'MAPEX_SERVICE_PORT', 5555)),
                            float(getattr(self.config, 'MAPEX_SERVICE_TIMEOUT', 600.0)))
            self._mapex_client = c
        return c

    def _prediction_worker(self):
        from intelx.common import safe_copy_pose
        mode = self.TIER1_MODE
        print('PREDICTION: Tier-1 worker starting (%s)' % mode)
        client = self._tier1_client()
        pong = client.ping()
        if not pong.get('ok'):
            print('PREDICTION: *** MapEx service unreachable: %s ***' % pong.get('error'))
        WIN = self.config.sim.get('WIN_PX', 400)
        RES = self.config.sim.get('RESOLUTION', 0.05)
        snap_px = int(getattr(self.config, 'TIER1_SNAP_PX', DEFAULT_SNAP_PX))
        pose_hist = []
        n = 0
        n_fail = 0                        # CONSECUTIVE service failures
        last_raw_goal = None              # to detect the re-request livelock
        n_repeat = 0

        # Per-run call accounting, read by run_sweep2's run_task and written to
        # results.csv. `fallbacks` is the one that matters for honesty: a
        # non-zero count means some decisions were NOT made by the named
        # planner, and that run must not be reported as that method.
        self._tier1_stats = {'calls': 0, 'service_s_sum': 0.0,
                             'fallbacks': 0, 'failures': 0,
                             'n_frontiers_sum': 0, 'snapped': 0, 'repeats': 0,
                             'path_truncated': 0}

        while self.running.is_set():
            try:
                try:
                    req = self.prediction_request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                pose = req.get('pose')
                if pose is None:
                    pose = safe_copy_pose(self.simulator.pose)

                # Only record a pose that is actually new. A rejected goal makes
                # the sim re-request from the SAME pose, and appending the
                # duplicate silently changes their revisit scoring.
                rc_now = your_xy_to_their_rc((pose[0], pose[1]), RES, THEIR_RES)
                if (not pose_hist
                        or abs(rc_now[0] - pose_hist[-1][0]) > 1e-6
                        or abs(rc_now[1] - pose_hist[-1][1]) > 1e-6):
                    pose_hist.append(rc_now)

                obs_map = occ_explored_to_obs_map(self.simulator.occ,
                                                  self.simulator.explored,
                                                  RES, THEIR_RES)
                rep = client.request({
                    'mode': mode,
                    'obs_map': obs_map,
                    'cur_pose': rc_now,
                    'pose_list': pose_hist[-200:],
                    'cfg': {'pixel_per_meter': int(round(1.0 / THEIR_RES)),
                            'pred_laser_range_m': getattr(self.config, 'MAPEX_PRED_RANGE_M', 20),
                            'pred_num_laser': getattr(self.config, 'MAPEX_PRED_BEAMS', 250),
                            'upen_config': getattr(self.config, 'MAPEX_UPEN_CONFIG', {})},
                })

                # ---- accounting -------------------------------------------
                st = self._tier1_stats
                st['calls'] += 1
                if rep.get('service_s'):
                    st['service_s_sum'] += float(rep['service_s'])
                if rep.get('n_frontiers'):
                    st['n_frontiers_sum'] += int(rep['n_frontiers'])
                if rep.get('fallback'):
                    st['fallbacks'] += 1
                if not rep.get('ok'):
                    st['failures'] += 1

                raw_goal = rep.get('goal')
                if raw_goal is not None and raw_goal == last_raw_goal:
                    n_repeat += 1
                    st['repeats'] += 1
                    if n_repeat in (3, 10, 30):
                        print('PREDICTION: %s returned the SAME goal %d times in a '
                              'row (%s). The sim is rejecting it and re-requesting; '
                              'each repeat costs a full planning call.'
                              % (mode, n_repeat, raw_goal))
                else:
                    n_repeat = 0
                last_raw_goal = raw_goal

                # ---- goal + termination semantics --------------------------
                goal_px = None
                err = rep.get('error') or ''
                if rep.get('ok') and raw_goal is not None:
                    wx, wy = their_rc_to_your_xy(raw_goal, THEIR_RES, RES)
                    goal_px, was_snapped = self._resolve_goal_px(
                        (wx, wy), pose, WIN, RES, snap_px)

                    # UPEN's goal is an RRT NODE, and the tree expands through
                    # anything that is not a SEEN wall -- including the whole
                    # unobserved area outside the building. So its goal can land
                    # where no reachable cell exists within snap_px and nothing
                    # can be snapped to. When that happens, walk UPEN's own
                    # chosen path back from the far end and take the furthest
                    # node the simulator will accept. That keeps the direction
                    # and the route UPEN picked instead of substituting a goal it
                    # did not choose, and it mirrors what their own reach_horizon
                    # truncation does.
                    if goal_px is None and rep.get('path'):
                        for node in reversed(rep['path']):
                            px, py = their_rc_to_your_xy(node, THEIR_RES, RES)
                            goal_px, _sn = self._resolve_goal_px(
                                (px, py), pose, WIN, RES, snap_px)
                            if goal_px is not None:
                                was_snapped = True
                                st['path_truncated'] = st.get('path_truncated', 0) + 1
                                break

                    if goal_px is None:
                        # No nearby cell survives the sim's own goal test. Do NOT
                        # call this exhaustion; let the sim re-request from a new
                        # pose.
                        n_fail += 1
                        resp_mode = ('BASELINE_GLOBAL'
                                     if n_fail >= MAX_CONSECUTIVE_FAILURES
                                     else 'TIER1_RETRY')
                        print('PREDICTION: %s -> goal (%.1f,%.1f) has no simulator-'
                              'acceptable cell within %d px, retry %d/%d'
                              % (mode, wx, wy, snap_px, n_fail,
                                 MAX_CONSECUTIVE_FAILURES))
                    else:
                        if was_snapped:
                            st['snapped'] += 1
                        resp_mode = 'TIER1_%s' % mode.upper()
                        n_fail = 0
                elif rep.get('ok') and err == 'no_frontiers':
                    # The ONLY genuine exhaustion signal.
                    resp_mode = 'BASELINE_GLOBAL'
                    print('PREDICTION: %s -> no frontiers left, exploration exhausted'
                          % mode)
                else:
                    n_fail += 1
                    if n_fail >= MAX_CONSECUTIVE_FAILURES:
                        resp_mode = 'BASELINE_GLOBAL'      # give up cleanly
                        print('PREDICTION: %s -> %d consecutive service failures '
                              '(%s), ending run' % (mode, n_fail, err))
                    else:
                        resp_mode = 'TIER1_RETRY'          # transient: sim re-requests
                        print('PREDICTION: %s -> service failure (%s), retry %d/%d'
                              % (mode, err, n_fail, MAX_CONSECUTIVE_FAILURES))

                if rep.get('fallback'):
                    print('PREDICTION: *** %s FALLBACK -> %s *** (this decision was '
                          'NOT made by %s)' % (mode, rep['fallback'], mode))

                response = {
                    'heatmap': np.zeros((WIN, WIN), np.float32),
                    'goal_px': goal_px,
                    'selected_score': 0.0,
                    'final_heading': None,
                    'origin_img': np.zeros((WIN, WIN), np.uint8),
                    'timestamp': time.time(),
                    'prediction_id': n,
                    'request_pose': pose,
                    'mode': resp_mode,
                    'hallucinations': [],
                    'local_components': {
                        'label_map': np.zeros((WIN, WIN), np.uint16),
                        'keep_ids': [], 'centroids': {}, 'detected_doorways': [],
                        'active_directions': [], 'frontier_centroids': [],
                        'tier1_service_s': rep.get('service_s'),
                        'tier1_n_frontiers': rep.get('n_frontiers'),
                        'tier1_fallback': rep.get('fallback'),
                    },
                }
                with self.data_lock:
                    self.latest_prediction_data = response
                while not self.prediction_response_queue.empty():
                    try:
                        self.prediction_response_queue.get_nowait()
                    except queue.Empty:
                        break
                self.prediction_response_queue.put(response)
                n += 1
            except Exception as e:
                print('PREDICTION: Tier-1 worker error: %s' % e)
                time.sleep(0.2)


class MapExMixin(_Tier1Base):
    """MapEx (Ho & Kim et al., ICRA 2025) -- 'visvarprob': visibility mask +
    ensemble variance + probabilistic raycast, scored / distance."""
    TIER1_MODE = 'visvarprob'


class UpenMixin(_Tier1Base):
    """UPEN (Georgakis et al., ICRA 2022) -- RRT* goal sampling driven by
    ensemble prediction variance.

    Requires MAPEX_UPEN_CONFIG to carry `reach_horizon` and `upen_mode`; without
    them their get_rrt_goal raises TypeError and the service falls back to the
    nearest frontier, which would make this baseline a nearest-frontier planner
    wearing a UPEN label. The service reports that as `fallback`, and it is
    counted in _tier1_stats['fallbacks']."""
    TIER1_MODE = 'upen'


class HectorAugMixin(_Tier1Base):
    """IG-Hector (Shrestha et al., ICRA 2019), as implemented by MapEx.
    NOTE the goal-based adaptation documented at the top of this file."""
    TIER1_MODE = 'hectoraug'


TIER1_MIXINS = {
    'mapex': MapExMixin,
    'upen': UpenMixin,
    'hectoraug': HectorAugMixin,
}


# ---------------------------------------------------------------- self-test
def _selftest():
    print('== obs_map conversion ==')
    occ = np.zeros((40, 40), np.int8)
    occ[10, :] = 1                              # a 1-px horizontal wall
    explored = np.zeros((40, 40), bool)
    explored[5:35, 5:35] = True
    obs = occ_explored_to_obs_map(occ, explored, 0.05, 0.1)
    print('   40x40 @0.05 ->', obs.shape, '@0.1')
    assert obs.shape == (20, 20)
    vals = set(np.unique(obs).tolist())
    assert vals <= {0.0, 0.5, 1.0}, vals
    assert (obs[5] == 1.0).any(), 'thin wall must survive downsampling'
    # blocks 0-1 cover source rows 0-3, all outside explored[5:35] -> unknown.
    # (block 2 covers rows 4-5 and row 5 IS explored, so it is legitimately free)
    assert (obs[:2, :2] == 0.5).all(), 'unexplored must stay unknown'
    print('   values', sorted(vals), '| thin wall preserved | unexplored=0.5  OK')

    # unexplored ground-truth walls must NOT leak into the planner's view
    occ2 = np.ones((20, 20), np.int8)
    ex2 = np.zeros((20, 20), bool)
    assert (occ_explored_to_obs_map(occ2, ex2, 0.05, 0.1) == 0.5).all()
    print('   unobserved ground truth never leaks  OK')

    print('== coordinate round trip ==')
    for xy in [(100.0, 200.0), (37.5, 12.25), (0.0, 0.0)]:
        rc = your_xy_to_their_rc(xy, 0.05, 0.1)
        back = their_rc_to_your_xy(rc, 0.1, 0.05)
        print('   xy=%-16s -> rc=%-16s -> xy=%s' % (xy, [round(v, 2) for v in rc],
                                                    tuple(round(v, 2) for v in back)))
        assert abs(back[0] - xy[0]) < 1e-6 and abs(back[1] - xy[1]) < 1e-6
    # axis swap is real, not accidental
    assert your_xy_to_their_rc((10.0, 0.0), 0.05, 0.1) == [0.0, 5.0]
    print('   (x,y)<->(row,col) swap verified  OK')

    print('== goal snapping ==')
    occ3 = np.zeros((100, 100), np.int8)
    occ3[40:60, 40:60] = 1                      # a 20x20 block of wall
    # already free -> returned untouched, fractional part preserved
    g = snap_goal_to_free_cell(occ3, (10.25, 10.75))
    assert g == (10.25, 10.75), g
    print('   free goal untouched:', g)
    # inside the wall -> nearest free cell, and it really is free
    g = snap_goal_to_free_cell(occ3, (50.0, 50.0), 40)
    assert g is not None and occ3[int(g[1]), int(g[0])] == 0, g
    d = ((g[0] - 50.0) ** 2 + (g[1] - 50.0) ** 2) ** 0.5
    assert d <= 15.0, d                         # block half-width is 10
    print('   walled goal (50,50) -> %s, %.1f px away, free' % (g, d))
    # just inside the edge -> snaps a short distance out
    g = snap_goal_to_free_cell(occ3, (41.0, 50.0), 40)
    assert g is not None and occ3[int(g[1]), int(g[0])] == 0
    print('   edge goal (41,50) -> %s' % (g,))
    # no free cell in range -> None, so the caller retries instead of quitting
    occ4 = np.ones((60, 60), np.int8)
    assert snap_goal_to_free_cell(occ4, (30.0, 30.0), 10) is None
    print('   fully walled neighbourhood -> None  OK')
    # out of bounds is handled, not crashed
    assert snap_goal_to_free_cell(occ3, (-500.0, -500.0), 40) is None
    print('   far out-of-bounds -> None  OK')

    print('== candidates + round-trip resolution ==')
    cands = free_cell_candidates(occ3, (50.0, 50.0), 40)
    assert cands, 'walled goal must still offer candidates'
    assert all(occ3[int(c[1]), int(c[0])] == 0 for c in cands)
    print('   %d candidates for the walled goal, nearest %s' % (len(cands), cands[0]))
    # an already-free goal is offered untouched, first
    cands = free_cell_candidates(occ3, (10.25, 10.75), 40)
    assert cands[0] == (10.25, 10.75), cands[0]
    print('   free goal offered untouched first:', cands[0])

    # the resolver must reproduce the SIMULATOR's test, not its own: the world
    # -> image -> world round trip truncates, so a cell we like can come back as
    # a wall. Drive the real code path with a stub explorer.
    from intelx.common import _transform_image_goal_to_world

    class _StubSim(object):
        def __init__(s, occ):
            s.occ = occ

    class _Stub(_Tier1Base):
        def __init__(s, occ):
            s.simulator = _StubSim(occ)

        def _transform_world_to_image_goal(s, world_pos, robot_pose, image_size):
            import numpy as _np
            from math import pi as _pi
            cx = cy = image_size // 2
            rx, ry, rth = robot_pose
            dxw, dyw = world_pos[0] - rx, world_pos[1] - ry
            a = (90.0 - (rth * 180.0 / _pi)) * _pi / 180.0
            c, sn = _np.cos(a), _np.sin(a)
            return (int(cx + dxw * c + dyw * sn), int(cy - dxw * sn + dyw * c))

    stub = _Stub(occ3)
    pose = (30.0, 30.0, 0.37)
    gpx, moved = stub._resolve_goal_px((50.0, 50.0), pose, 400, 0.05, 40)
    assert gpx is not None, 'a walled goal beside open space must resolve'
    bx, by = _transform_image_goal_to_world(gpx, pose, 400, 0.05)
    assert occ3[int(by), int(bx)] == 0, (gpx, bx, by)
    assert moved is True
    print('   walled goal -> image %s -> world (%.2f,%.2f), free after round trip'
          % (gpx, bx, by))
    # nothing free anywhere -> None, and the caller retries rather than quitting
    gpx, moved = _Stub(occ4)._resolve_goal_px((30.0, 30.0), pose, 400, 0.05, 10)
    assert gpx is None and moved is False
    print('   fully walled -> (None, False)  OK')

    print('== transport round trip (mock service) ==')
    import threading
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0)); srv.listen(1)
    port = srv.getsockname()[1]

    def mock():
        conn, _ = srv.accept()
        while True:
            try:
                req = _recv(conn)
            except EOFError:
                break
            if req.get('cmd') == 'ping':
                _send(conn, {'ok': True, 'pong': True})
            else:
                assert req['obs_map'].dtype == np.float32
                _send(conn, {'ok': True, 'goal': [12.0, 34.0],
                             'n_frontiers': 3, 'service_s': 0.01, 'error': '',
                             'fallback': None})
    threading.Thread(target=mock, daemon=True).start()

    c = MapExClient('127.0.0.1', port, timeout=5)
    print('   ping ->', c.ping())
    rep = c.request({'mode': 'visvarprob',
                     'obs_map': np.full((20, 20), 0.5, np.float32),
                     'cur_pose': [10, 10]})
    print('   plan ->', rep)
    assert rep['goal'] == [12.0, 34.0]
    print('   goal in your frame:', their_rc_to_your_xy(rep['goal']))

    print('== failure handling ==')
    dead = MapExClient('127.0.0.1', 1, timeout=0.2)
    r = dead.request({'mode': 'visvarprob', 'obs_map': np.zeros((4, 4), np.float32),
                      'cur_pose': [0, 0]}, retries=0)
    print('   unreachable service ->', r['ok'], '|', r['error'][:40])
    assert r['ok'] is False and r['goal'] is None
    print('\n*** intelx.baselines.tier1 self-test PASSED ***')


if __name__ == '__main__':
    _selftest()