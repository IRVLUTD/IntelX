"""
mapex_service.py  --  run this INSIDE the MapEx `lama` conda env (python 3.6)
=============================================================================
Serves MapEx / UPEN / IG-Hector planning decisions over a local TCP socket so
they can be used as baselines from a different python environment.

WHY A SERVICE
    Their stack is python 3.6 + old torch + LaMa + range_libc; your sweep is
    python 3.11 + torch 2.7. Those cannot share a process. Rather than
    REIMPLEMENTING their information-gain maths (the classic way to publish an
    accidentally-crippled baseline), we call THEIR code unmodified:
        - their FrontierPlanner.get_frontier_centers_given_obs_map
        - their FrontierPlanner.score_frontiers  (visibility mask, probabilistic
          raycast, ensemble variance)
        - their upen_baseline.get_rrt_goal
        - their get_hector_exploration_transform_map
    The only thing we write is format/coordinate conversion, which is testable.

CONTRACT (client works in THEIR resolution, 0.1 m/px, UNPADDED)
    request  = {'mode': 'visvarprob'|'upen'|'hectoraug',
                'obs_map': float32 HxW in {0.0 free, 0.5 unknown, 1.0 occupied},
                'cur_pose': [row, col],          # unpadded, same grid as obs_map
                'pose_list': [[row, col], ...]}  # optional history
    response = {'ok': bool, 'goal': [row, col] (unpadded) or None,
                'n_frontiers': int, 'mode': str, 'error': str,
                'fallback': None or str}

    The pd_size=500 padding their raycaster needs is applied and removed INSIDE
    this service, so the client never has to know about it.

    'fallback' is None when the goal really came from the named planner. For
    UPEN it names the reason we had to substitute the nearest frontier, so a run
    can never be silently scored as UPEN when it was actually nearest-frontier.
    Treat any run with a non-null fallback as unreportable for that method.

INSTALLING IT
    This file is not standalone. It calls MapEx's own planner, so it has to run
    inside MapEx's environment, with MapEx's modules importable and its
    working directory set to their scripts/. Given the MapEx root, it sets
    up the last two itself.

    MapEx is linked at third_party/MapEx as a git submodule.

    1. Fetch it: git submodule update --init third_party/MapEx
    2. Install its `lama` conda environment by their instructions.
    3. Run this file from the repository root, pointed at it:

           conda activate lama
           python services/mapex_service.py --mapex-root third_party/MapEx --port 5555

       Without --mapex-root or MAPEX_ROOT, the MapEx root is taken to be the
       parent of this file's directory, which is right when the file is copied
       into MapEx's own scripts/ directory, beside explore.py.

    Check it is alive, from any environment:

           python services/mapex_service.py --selftest-client --port 5555
"""
from __future__ import print_function

import os
import sys
import time
import socket
import struct
import pickle
import threading
import traceback

import numpy as np

PD_SIZE = 500          # their raycast border padding (sim_utils pads by laser range)
# Linear size, in their 0.1 m/px pixels, that UPEN's published RRT defaults were
# tuned for. Used to scale those defaults up on larger maps; see _upen_goal.
UPEN_REFERENCE_PX = 1300
UPEN_MAX_ITERS_CAP = 60000     # keep a pathological map from running forever
PROTO = 2              # pickle protocol 2 => readable by py2/3.6/3.9 alike


# --------------------------------------------------------------------------
# wire protocol: 4-byte big-endian length + pickled dict
# --------------------------------------------------------------------------

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


def send_msg(sock, obj):
    data = pickle.dumps(_enc(obj), protocol=PROTO)
    sock.sendall(struct.pack('>I', len(data)) + data)


def recv_all(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError('socket closed')
        buf += chunk
    return buf


def recv_msg(sock):
    raw = recv_all(sock, 4)
    (n,) = struct.unpack('>I', raw)
    return _dec(pickle.loads(recv_all(sock, n)))


# --------------------------------------------------------------------------
# the planner backend (imports THEIR modules)
# --------------------------------------------------------------------------
class MapExBackend(object):
    def __init__(self, mapex_root, device='cuda:0'):
        self.root = mapex_root
        scripts = os.path.join(mapex_root, 'scripts')

        # IMPORT ORDER MATTERS -- their modules are circular:
        #   sim_utils (line 15)         -> from scripts import simple_mask_utils
        #   simple_mask_utils (line 23) -> from sim_utils import makePyOMap
        # and makePyOMap is defined AFTER that import. explore.py survives this
        # only because it runs with the MapEx ROOT on sys.path, so
        # `scripts.simple_mask_utils` and `simple_mask_utils` are two DISTINCT
        # module objects and the cycle resolves. We must reproduce that exactly:
        # root first, scripts second, and import through the `scripts` package.
        # Their code also does sys.path.append('../lama') -- a RELATIVE path --
        # so the working directory has to be scripts/.
        for p in (scripts, mapex_root):
            if p in sys.path:
                sys.path.remove(p)
        sys.path.insert(0, scripts)          # for the inner `from sim_utils ...`
        sys.path.insert(0, mapex_root)       # for `from scripts import ...`
        try:
            os.chdir(scripts)                # for '../lama'
        except Exception as e:
            print('[service] could not chdir to %s: %s' % (scripts, e))

        import torch
        self.torch = torch
        self.device = device

        import albumentations as A
        import cv2 as _cv2
        # EXACTLY their get_lama_padding_transform() (explore.py:63): LaMa needs
        # H,W divisible by 16, so the map handed to score_frontiers is padded the
        # same way the model input is -- otherwise their assert
        #   obs_map.shape == pred_maputils.shape
        # fires (e.g. 1300 vs 1312).
        self._pad16 = A.PadIfNeeded(min_height=None, min_width=None,
                                    pad_height_divisor=16, pad_width_divisor=16,
                                    border_mode=_cv2.BORDER_CONSTANT, value=0)

        from scripts import sim_utils as su
        from scripts import simple_mask_utils as smu
        from scripts.lama_pred_utils import (load_lama_model, get_lama_transform,
                                             convert_obsimg_to_model_input)
        self.su, self.smu = su, smu
        self.convert_obsimg_to_model_input = convert_obsimg_to_model_input

        # --- load the ensemble (3 LaMa models) + the all-train model ---------
        pm = os.path.join(mapex_root, 'pretrained_models')
        ens_dir = os.path.join(pm, 'weights', 'lama_ensemble')
        self.models = []
        for d in sorted(os.listdir(ens_dir)):
            mp = os.path.join(ens_dir, d)
            if os.path.isdir(mp):
                print('[service] loading ensemble model:', mp)
                self.models.append(load_lama_model(mp, device=device))
        if not self.models:
            raise RuntimeError('no ensemble models found under %s' % ens_dir)

        big = os.path.join(pm, 'weights', 'big_lama')
        print('[service] loading big_lama:', big)
        self.alltrain = load_lama_model(big, device=device)

        self.map_transform = get_lama_transform('default_map_eval', (512, 512))
        print('[service] backend ready: %d ensemble models' % len(self.models))

        # per-mode planner objects (their class, unmodified)
        self._planners = {}

    def planner(self, score_mode):
        if score_mode not in self._planners:
            self._planners[score_mode] = self.su.FrontierPlanner(score_mode=score_mode)
        return self._planners[score_mode]

    # ---------------- prediction ----------------
    def _obs_to_3ch(self, obs_map_padded):
        """Build the 3-channel image convert_obsimg_to_model_input expects.

        THE BUG THIS FIXES. That function derives LaMa's inpainting mask AFTER
        applying map_transform:

            mask = ((obs_img[0] > 0.49) & (obs_img[0] < 0.51))

        i.e. it looks for the UNKNOWN value 0.5 on a 0-1 image. But
        convert_01_single_channel_to_0_255_3_channel does `img *= 255`, so
        unknown arrives as 127.5 and that test matches NOTHING unless the
        transform divides by 255 first. With an empty mask LaMa returns
            mask*predicted + (1-mask)*image  ==  the input,
        identically for every ensemble member. Measured: output range 0-255 and
        per-pixel ensemble variance of exactly 0.000e+00. UPEN maximises that
        variance, so it had nothing to choose between; and _predict thresholds
        the same output at > 0.5, which on a 0-255 map marks everything occupied,
        so mapex and hectoraug were raycasting against a solid-wall prediction.

        So probe the transform once and feed it whichever scale makes the mask
        fire. If map_transform does normalise, their original 0-255 path is used
        unchanged.
        """
        if getattr(self, '_tf_normalises', None) is None:
            probe = np.full((16, 16, 3), 0.5 * 255.0, dtype=np.float32)
            try:
                t = self.map_transform(image=probe, obs_img=probe)
                self._tf_normalises = bool(np.max(t['obs_img']) <= 1.5)
            except Exception as e:
                print('[service] transform probe failed (%s); assuming it does '
                      'not normalise' % e)
                self._tf_normalises = False
            print('[service] map_transform %s a 0-255 input -> feeding it %s'
                  % ('normalises' if self._tf_normalises else 'does NOT normalise',
                     '0-255 (their path)' if self._tf_normalises else '0-1'))

        if self._tf_normalises:
            return self.smu.convert_01_single_channel_to_0_255_3_channel(obs_map_padded)
        # 0/0.5/1 straight through, so the 0.49-0.51 mask test sees unknown
        return np.stack([obs_map_padded] * 3, axis=2).astype(np.float32)

    def _ensemble_stack(self, obs_map_padded):
        """Run the 3 ensemble models once and return the stacked predictions.
        Shared by _predict and _upen_goal so UPEN does not pay for the ensemble
        twice in one call."""
        torch = self.torch
        obs_img = self._obs_to_3ch(obs_map_padded)
        preds = []
        for m in self.models:
            batch, mask = self.convert_obsimg_to_model_input(
                obs_img, self.map_transform, self.device)
            if not getattr(self, '_mask_logged', False):
                self._mask_logged = True
                cov = float(np.mean(np.asarray(mask) > 0))
                print('[service] LaMa inpainting mask covers %.2f%% of the input'
                      % (100.0 * cov))
                if cov <= 0.0:
                    print('[service] *** MASK IS EMPTY: LaMa will return the '
                          'input unchanged, every ensemble member will be '
                          'identical, and the variance UPEN maximises will be '
                          'exactly zero. ***')
            with torch.no_grad():
                out = m(batch)
            preds.append(out['inpainted'][0][0])
        stack = torch.stack(preds, dim=0)

        # SCALE. Measured on a real call, out['inpainted'] comes back on a
        # 0-255 scale, not 0-1. Everything downstream assumes 0-1:
        #   _predict does (inpainted > 0.5) to build pred_maputils for their
        #   probabilistic raycast, and on a 0-255 map that is True almost
        #   everywhere -- so score_frontiers has been raycasting against a
        #   "solid wall everywhere" prediction for mapex AND hectoraug.
        # Normalise once, here, so every consumer sees the 0-1 convention their
        # code expects. Guarded on the observed range so a model that already
        # returns 0-1 is left alone.
        mx = float(stack.max())
        if mx > 1.5:
            stack = stack / 255.0
        return stack, obs_img

    def _predict(self, obs_map_padded):
        """obs_map (0/0.5/1) -> (mean_map np.float32, var_map torch, pred_maputils).
        Mirrors explore.py: ensemble -> var/mean; big_lama -> map for raycast."""
        torch = self.torch
        stack, obs_img = self._ensemble_stack(obs_map_padded)
        var_map = torch.var(stack, dim=0)
        mean_map = np.mean(stack.cpu().numpy(), axis=0)

        batch, mask = self.convert_obsimg_to_model_input(
            obs_img, self.map_transform, self.device)
        with torch.no_grad():
            at = self.alltrain(batch)
        at_np = at['inpainted'][0][0].cpu().numpy()
        if float(at_np.max()) > 1.5:            # same 0-255 guard as the ensemble
            at_np = at_np / 255.0
        pred_maputils = (at_np > 0.5).astype(np.float32)
        return mean_map, var_map, pred_maputils

    # ---------------- one planning decision ----------------
    def plan(self, mode, obs_map, cur_pose, pose_list=None, cfg=None):
        cfg = cfg or {}
        pad = PD_SIZE
        obs_p = np.pad(obs_map, pad, mode='constant', constant_values=0.5)
        pose_p = np.array([cur_pose[0] + pad, cur_pose[1] + pad], dtype=float)
        plist = (np.array(pose_list, dtype=float) + pad) if pose_list else np.array([pose_p])

        pred_vis_config = {'laser_range_m': cfg.get('pred_laser_range_m', 20),
                           'pixel_per_meter': cfg.get('pixel_per_meter', 10),
                           'num_laser': cfg.get('pred_num_laser', 250)}

        score_mode = {'visvarprob': 'visvarprob',
                      'hectoraug': 'hectoraug',
                      'upen': 'nearest'}[mode]
        fp = self.planner(score_mode)

        centers, fmap, n_large = fp.get_frontier_centers_given_obs_map(obs_p)
        if len(centers) == 0:
            return {'ok': True, 'goal': None, 'n_frontiers': 0, 'mode': mode,
                    'error': 'no_frontiers', 'fallback': None}

        fallback = None
        if mode == 'upen':
            # UPEN does NOT use score_frontiers, so mean_map / var_map /
            # pred_maputils from _predict were computed and thrown away here.
            # That was 4 LaMa forwards per call on top of the 3 _upen_goal runs.
            #
            goal_p, fallback = self._upen_goal(obs_p, pose_p, centers, cfg)
        else:
            mean_map, var_map, pred_maputils = self._predict(obs_p)

            # Their exact call convention (explore.py): frontier centers and
            # cur_pose stay in the pd_size-padded frame, while obs_map is
            # additionally padded to a multiple of 16 so it matches the LaMa
            # output shape.
            obs_p16 = self._pad16(image=obs_p)['image']

            fc, cost_list, _, _, best_i, _ = fp.score_frontiers(
                centers, pose_p, plist, pred_maputils, pred_vis_config,
                obs_map=obs_p16, mean_map=mean_map, var_map=var_map)
            # their convention: total_frontier_cost_list is a COST -> argmin
            best = int(np.argmin(cost_list))
            goal_p = np.array(fc[best], dtype=float)

        if goal_p is None:
            return {'ok': True, 'goal': None, 'n_frontiers': len(centers),
                    'mode': mode, 'error': 'no_goal', 'fallback': fallback}
        if fallback:
            print('[service] *** %s FELL BACK to nearest frontier: %s ***'
                  % (mode, fallback))
        out = {'ok': True, 'mode': mode, 'n_frontiers': len(centers),
               'goal': [float(goal_p[0] - pad), float(goal_p[1] - pad)],
               'error': '', 'fallback': fallback}
        if mode == 'upen' and getattr(self, '_upen_last_params', None):
            out['upen_params'] = self._upen_last_params
        if mode == 'upen' and getattr(self, '_upen_last_path', None):
            # unpadded [row, col], nearest the robot first, capped so the message
            # stays small
            out['path'] = [[n[0] - pad, n[1] - pad]
                           for n in self._upen_last_path[:200]]
        return out

    def _upen_diagnose(self, upen_baseline, grid, ens, stack, pose_p, gsel, uc,
                       rrt_iters, expand_dis, circle_dist, _obs_img_ref=None):
        """Say WHY get_rrt_goal returned None. In 'exploration' mode it returns
        None only when best_path is None, and best_path is picked by

            best_path_var = 0
            for i: if path_sum_var[i] > best_path_var: best_path = paths[i]

        so None means either planning() produced NO PATHS, or every path scored
        <= 0 on the ensemble-variance objective (the test is a strict > against
        an initial 0). Those need opposite fixes:

          no paths      -> the RRT cannot expand. Look at `grid`: binarymap is
                           (grid == 1), and we build grid from the ensemble MEAN
                           thresholded at 0.5, so if the prediction reads as
                           occupied around the start the tree has nowhere to go.
          zero variance -> the three ensemble members agree everywhere along
                           every path, so UPEN's objective is flat and it has
                           nothing to prefer. That is a property of the input,
                           not a bug, and it is a reportable finding.

        Re-runs their own planning() and eval_path_expl() unmodified. Only ever
        called on the failure path.
        """
        try:
            binarymap = (grid == 1)
            free_frac = float((~binarymap).float().mean())
            rrt = upen_baseline.RRTStar(
                start=[int(pose_p[0]), int(pose_p[1])],
                obstacle_list=None,
                goal=[int(gsel[0]), int(gsel[1])],
                rand_area=[0, binarymap.shape[0]],
                max_iter=rrt_iters,
                expand_dis=expand_dis,
                goal_sample_rate=uc.get('goal_sample_rate', -1),
                connect_circle_dist=circle_dist,
                occupancy_map=binarymap)
            paths = rrt.planning(animation=False,
                                 use_straight_line=uc.get('rrt_straight_line', False),
                                 exploration=True,
                                 horizon=uc['reach_horizon'])
            n = len(paths) if paths else 0
            if not n:
                return ('planning() returned 0 paths (grid %dx%d, %.1f%% free, '
                        'start [%d,%d], expand_dis %.1f) -> the tree could not '
                        'expand'
                        % (binarymap.shape[0], binarymap.shape[1],
                           100.0 * free_frac, int(pose_p[0]), int(pose_p[1]),
                           expand_dis))
            var = upen_baseline.eval_path_expl(ens, paths, uc['reach_horizon'])
            try:
                vmax = float(max(var))
                vpos = int(sum(1 for v in var if v > 0))
            except Exception:
                vmax, vpos = float('nan'), -1
            lens = [len(p) for p in paths]
            msg = ('%d path(s), length %d-%d, %d with variance > 0, max '
                   'variance %.3e' % (n, min(lens), max(lens), vpos, vmax))
            if vpos:
                return msg + ' -> unexpected: a path scored > 0'

            # A sum of EXACTLY 0.0 over 3 float32 models and 50 points per path
            # is not chance. Three things produce it, and they need different
            # fixes, so measure which:
            #   (a) the models genuinely agree      -> per-pixel variance is ~0
            #   (b) eval_path_expl reads channel 0  -> we only fill channel 1
            #       (occ_chan = 1, copied from explore.py), so channel 0 is zeros
            #   (c) index order                     -> RRT nodes come from
            #       start=[row, col]; sampling [n1, n0] instead of [n0, n1] reads
            #       transposed coordinates
            var_map = self.torch.var(stack, dim=0)
            H, W = int(var_map.shape[0]), int(var_map.shape[1])
            ch0 = float(self.torch.var(ens[:, 0, 0, :, :], dim=0).max())
            ch1 = float(self.torch.var(ens[:, 0, 1, :, :], dim=0).max())

            p0 = paths[0]
            horizon = int(uc['reach_horizon'])

            def _sum_along(swap):
                tot, hits = 0.0, 0
                for nd in p0[:min(len(p0), horizon)]:
                    try:
                        a, b = (int(nd[1]), int(nd[0])) if swap else (int(nd[0]), int(nd[1]))
                    except Exception:
                        continue
                    if 0 <= a < H and 0 <= b < W:
                        tot += float(var_map[a, b])
                        hits += 1
                return tot, hits

            s_rc, n_rc = _sum_along(False)
            s_cr, n_cr = _sum_along(True)

            # If LaMa returned the INPUT unchanged, every model agrees to the
            # bit and the variance is exactly 0. That happens when the mask
            # convert_obsimg_to_model_input builds is empty, i.e. nothing was
            # marked for inpainting: out['inpainted'] is
            # mask*predicted + (1-mask)*original.
            # Compare against the model INPUT, not `grid`: grid is on the
            # 0/0.5/1 obs convention while the raw model output was 0-255, so the
            # earlier version of this check was comparing two different scales
            # and its number meant nothing.
            try:
                ref = self.torch.as_tensor(
                    np.asarray(_obs_img_ref, dtype=np.float32)[..., 0] / 255.0,
                    device=stack.device)
                if ref.shape == stack[0].shape:
                    same = float((self.torch.abs(stack[0] - ref) < 1e-4).float().mean())
                    msg += (' | model output == input on %.1f%% of pixels'
                            % (100.0 * same))
            except Exception as _e:
                msg += ' | input-comparison failed (%s)' % _e

            return (msg + ' | models=%d out=[%.4f,%.4f] per-pixel var: max %.3e, '
                    '%.2f%% > 0 | ens ch0 var %.3e ch1 var %.3e | path[0] sum: '
                    '[n0,n1] %.3e over %d pt, [n1,n0] %.3e over %d pt'
                    % (int(stack.shape[0]), float(stack.min()), float(stack.max()),
                       float(var_map.max()),
                       100.0 * float((var_map > 0).float().mean()),
                       ch0, ch1, s_rc, n_rc, s_cr, n_cr))
        except Exception as e:
            return 'diagnosis failed: %s' % e

    def _upen_goal(self, obs_p, pose_p, centers, cfg):
        """UPEN: RRT* over the ensemble variance.

        Returns (goal_rowcol, fallback_reason). fallback_reason is None when the
        goal genuinely came from their RRT. Otherwise it names why we substituted
        the nearest frontier, and the caller reports it to the client so the run
        can be excluded rather than silently mislabelled as UPEN.

        NOTE on their RRTStar: it is constructed with
            rand_area=[0, binarymap.shape[0]]
        which bounds BOTH sampled coordinates by the ROW count. On their roughly
        square maps that is harmless. On a wide floorplan the sampler can never
        reach columns beyond the row count, so UPEN is limited to part of the
        map. That is their code, left unmodified on purpose; state it in the
        paper rather than patching it here.
        """
        uc = (cfg or {}).get('upen_config', {}) or {}

        self._upen_last_path = None

        def _nearest():
            d = np.linalg.norm(np.array(centers) - pose_p, axis=1)
            return np.array(centers[int(np.argmin(d))], dtype=float)

        # --- size-aware RRT parameters -------------------------------------
        # Their defaults (rrt_max_iters 2500, expand_dis 5, connect_circle_dist
        # 20) are tuned for KTH maps of roughly UPEN_REFERENCE_PX at 0.1 m/px,
        # about 130 m across. A 4399 px map at 0.05 m/px becomes 2200 px here and
        # PD_SIZE padding takes it to ~3200 px, about 320 m: six times the area,
        # with a step still fixed at 0.5 m. RRT* then exhausts max_iter without
        # ever connecting ('reached max iteration') and returns no goal, so every
        # decision silently fell back to the nearest frontier.
        #
        # We therefore hold the parameters constant RELATIVE TO MAP SIZE rather
        # than in absolute pixels: the step and the connect radius scale with the
        # linear dimension, and the iteration budget with the area. Set
        # 'autoscale': False in upen_config to force their raw values.
        dim = float(max(obs_p.shape))
        ref = float(uc.get('reference_px', UPEN_REFERENCE_PX))
        # OFF by default. This was written on the theory that RRT could not reach
        # the goal on a large map. In 'exploration' mode it never tries to: it
        # runs the full max_iter and returns a LIST of paths, which is why their
        # config sets goal_sample_rate to -1 ("ensure that goal sampling rate is
        # below 0 for exploration"). Scaling also raised expand_dis to ~1.2 m,
        # which is wider than a doorway. Turn it on only with evidence.
        f = max(1.0, dim / ref) if uc.get('autoscale', False) else 1.0
        rrt_iters = int(min(UPEN_MAX_ITERS_CAP,
                            round(float(uc.get('rrt_max_iters', 2500)) * f * f)))
        expand_dis = float(uc.get('expand_dis', 5)) * f
        circle_dist = float(uc.get('connect_circle_dist', 20)) * f
        if f > 1.0 and not getattr(self, '_upen_scale_logged', False):
            self._upen_scale_logged = True
            print('[service] UPEN autoscale: map %d px vs reference %d px -> '
                  'x%.2f  (max_iter %d, expand_dis %.1f, connect_circle %.1f)'
                  % (dim, ref, f, rrt_iters, expand_dis, circle_dist))
        self._upen_last_params = {'scale': round(f, 3), 'rrt_max_iters': rrt_iters,
                                  'expand_dis': round(expand_dis, 2),
                                  'connect_circle_dist': round(circle_dist, 2),
                                  'map_px': int(dim)}

        # get_rrt_goal requires reach_horizon and upen_mode. If they are absent
        # the call raises TypeError and every decision silently degrades to
        # nearest-frontier, so refuse up front and say so.
        missing = [k for k in ('reach_horizon', 'upen_mode') if k not in uc]
        if missing:
            return _nearest(), 'upen_config missing %s' % ','.join(missing)

        try:
            from scripts import upen_baseline
            torch = self.torch

            stack, _obs_img = self._ensemble_stack(obs_p)

            # grid is the OBSERVED occupancy map, exactly as explore.py:402 does
            #     planning_grid = torch.tensor(padded_obs_map).cuda()
            # get_rrt_goal then takes binarymap = (grid == 1), so only cells
            # actually SEEN to be occupied block the tree, and unknown (0.5)
            # counts as free. That is what lets UPEN's RRT expand into
            # unexplored space, which is the whole point of the method.
            #
            # This was previously (stack.mean(dim=0) > 0.5), the thresholded LaMa
            # ensemble. On a mostly-unknown input the ensemble saturates to
            # occupied, so the collision map came back 0.1% free, the tree had
            # nowhere to expand, and every call fell back to the nearest
            # frontier. The prediction belongs in `ensemble` below, which feeds
            # the variance objective; it is not the collision map.
            # LaMa pads to a multiple of 16, so its output is e.g. 3200 while
            # obs_p is 3199. Their padded_obs_map is already pad16'd, which is
            # why explore.py's shapes line up; do the same here so `grid` and
            # `ensemble` share a shape. PadIfNeeded adds to bottom/right when one
            # pixel is needed, so the top-left origin is unchanged and the
            # padded-frame coordinates (pose_p, centers) stay valid.
            gh, gw = int(stack.shape[-2]), int(stack.shape[-1])
            grid_np = self._pad16(image=obs_p)['image']
            if grid_np.shape[0] != gh or grid_np.shape[1] != gw:
                fixed = np.full((gh, gw), 0.5, dtype=np.float32)   # 0.5 = unknown
                h = min(gh, grid_np.shape[0])
                w = min(gw, grid_np.shape[1])
                fixed[:h, :w] = grid_np[:h, :w]
                grid_np = fixed
            grid = torch.as_tensor(grid_np, device=stack.device).float()
            ens = torch.zeros((stack.shape[0], 1, 2, grid.shape[0], grid.shape[1]))
            ens = ens.to(stack.device)
            ens[:, 0, 1, :, :] = stack

            pose_coords = torch.tensor([[[pose_p[1], pose_p[0]]]]).float().to(stack.device)
            d = np.linalg.norm(np.array(centers) - pose_p, axis=1)
            gsel = centers[int(np.argmin(d))]
            goal_coords = torch.tensor([[[gsel[1], gsel[0]]]]).float().to(stack.device)

            out = upen_baseline.get_rrt_goal(
                pose_coords=pose_coords.clone(), goal=goal_coords.clone(),
                grid=grid, ensemble=ens,
                rrt_max_iters=rrt_iters,
                expand_dis=expand_dis,
                goal_sample_rate=uc.get('goal_sample_rate', -1),
                connect_circle_dist=circle_dist,
                rrt_num_path=uc.get('rrt_num_path', 1000),
                rrt_straight_line=uc.get('rrt_straight_line', False),
                reach_horizon=uc['reach_horizon'],
                upen_mode=uc['upen_mode'])

            # explore.py unpacks (rrt_goal, rrt_best_path, path_dict)
            g = out[0] if isinstance(out, (tuple, list)) else out
            if g is None:
                why = self._upen_diagnose(upen_baseline, grid, ens, stack, pose_p,
                                          gsel, uc, rrt_iters, expand_dis,
                                          circle_dist, _obs_img)
                return _nearest(), 'rrt returned no goal: %s' % why
            # their get_rrt_goal returns
            #   torch.tensor([[[int(best_path[n][1]), int(best_path[n][0])]]]).cuda()
            # so it is a CUDA tensor: np.array() on it raises. Bring it to host
            # first. The [col, row] ordering is undone by the swap on return.
            # out = (goal, best_path, path_dict). Keep best_path: UPEN's goal is
            # an RRT NODE, and the tree expands through everything that is not a
            # SEEN wall -- which includes the whole unobserved region outside the
            # building. So the goal can land where no reachable cell exists at
            # all, and the client cannot snap it anywhere sensible. Handing back
            # the path lets the client walk UPEN's own route and stop at the
            # furthest point it can actually reach, rather than inventing a
            # different goal.
            try:
                bp = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
                if bp:
                    self._upen_last_path = [[float(n[0]), float(n[1])] for n in bp]
                else:
                    self._upen_last_path = None
            except Exception:
                self._upen_last_path = None

            if hasattr(g, 'detach'):
                g = g.detach().cpu().numpy()
            g = np.asarray(g).flatten()
            if g.size < 2:
                return _nearest(), 'rrt goal had %d components' % g.size
            return np.array([float(g[1]), float(g[0])], dtype=float), None   # (x,y)->(row,col)
        except Exception as e:
            traceback.print_exc()
            return _nearest(), 'rrt raised: %s' % e


# --------------------------------------------------------------------------
# server
# --------------------------------------------------------------------------
def serve(port, mapex_root, device):
    backend = MapExBackend(mapex_root, device)
    lock = threading.Lock()          # one GPU -> serialise planning calls

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(64)
    print('[service] listening on 127.0.0.1:%d' % port)

    def handle(conn):
        try:
            while True:
                try:
                    req = recv_msg(conn)
                except EOFError:
                    break
                if req.get('cmd') == 'ping':
                    send_msg(conn, {'ok': True, 'pong': True,
                                    'n_models': len(backend.models)})
                    continue
                t0 = time.time()
                try:
                    with lock:
                        rep = backend.plan(req['mode'],
                                           np.asarray(req['obs_map'], dtype=np.float32),
                                           req['cur_pose'],
                                           req.get('pose_list'),
                                           req.get('cfg'))
                except Exception as e:
                    traceback.print_exc()
                    rep = {'ok': False, 'goal': None, 'error': str(e),
                           'mode': req.get('mode', '?'), 'n_frontiers': 0,
                           'fallback': None}
                rep['service_s'] = time.time() - t0
                send_msg(conn, rep)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,)).start()


def _selftest_client(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', port))
    send_msg(s, {'cmd': 'ping'})
    print('[client] ping ->', recv_msg(s))
    obs = np.full((300, 300), 0.5, np.float32)
    obs[100:200, 100:200] = 0.0
    obs[100:200, 100] = 1.0
    send_msg(s, {'mode': 'visvarprob', 'obs_map': obs, 'cur_pose': [150, 150]})
    print('[client] plan ->', recv_msg(s))
    s.close()


if __name__ == '__main__':
    port = 5555
    # Copied into MapEx's own scripts/ directory, MapEx's root is this file's
    # parent. MAPEX_ROOT overrides that, and so does --mapex-root, e.g. the
    # submodule at third_party/MapEx.
    root = os.path.expanduser(os.environ.get(
        'MAPEX_ROOT',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    device = 'cuda:0'
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--port':
            port = int(args[i + 1])
        elif a == '--mapex-root':
            root = os.path.expanduser(args[i + 1])
        elif a == '--device':
            device = args[i + 1]
    # Absolute, because the backend changes into scripts/ before it imports:
    # a relative root such as third_party/MapEx would stop resolving there.
    root = os.path.abspath(root)
    if '--selftest-client' in args:
        _selftest_client(port)
    else:
        serve(port, root, device)