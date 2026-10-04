"""The live view of one exploration run, in the style of the paper's videos.

Drawn in the world frame: the occupancy map as it is explored, the laser scan,
the predicted coverage-gain field warped in from the pose it was predicted
from, its ranked components, the junction nodes and their branches, the
frontiers held for the fallback, the goal, the trail and the robot. Beside it,
coverage and path, the decision in force, and a legend. A baseline gets the
same frame without the parts it does not have.

Everything is drawn from a snapshot of what the run itself published, so the
field, the components and the goal on screen are the ones the planner used.
The same renderer serves run_intelx, the single-method runner and the
benchmark's per-run videos.

    view = LiveView(sim.occ, method="ours", title="IntelX: Small Office 3")
    player = LivePlayer(view, show=True, video_path="run.mp4")
    player.update(explorer)        # as often as you like
    player.finish(explorer, "complete")

It runs beside the simulation and prediction threads, so it is written to do
its work in a few large OpenCV and numpy calls: the static parts of the frame
are drawn once, and every blend touches only the region it changes.
"""
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from intelx.viz import style as S

CANVAS_W, CANVAS_H = 1280, 900   # the map
HUD_W = 420                      # the side panel
MARGIN = 20
HEADER_H = 70
CROP_MARGIN_PX = 30

FIELD_ALPHA = 0.78               # opacity at the field's peak
FLASH_FRAMES = 5                 # freshly observed cells glow for this many frames
FLASH_STRENGTH = 0.30
SCAN_BEAMS = 120                 # beams drawn; the sensor casts more
SCAN_FILL_ALPHA, SCAN_RAY_ALPHA, SCAN_HIT_ALPHA = 0.10, 0.22, 0.90
EXPLORED_BRANCH_ALPHA = 0.40     # branches already taken are faded, not hidden
NODE_SIZE_PX = 11
ROBOT_SIZE_PX = 13
WINDOW_LW = 3
TRAIL_BANDS = 10                 # the trail fades with age in this many steps
TRAIL_FADE_MIN = 0.45
TRAIL_WIDTH = 2

INTELX_METHODS = ("ours", "local_only")


def decision_label(pred, method):
    """What decided the current goal, in the words the paper's videos use."""
    if not pred:
        return "Planning"
    if method not in INTELX_METHODS:
        return S.method_label(method)
    if str(pred.get('mode', '') or '').startswith('LOCAL_ONLY'):
        return "Local Only"
    src = str(pred.get('src', '') or '')
    if src.startswith('NONE'):
        return "Frontier Fallback"
    if 'LOCK' in src:
        return "Junction Lock"
    return "Local Planner"


def _copy_nodes(nodes):
    out = []
    for n in nodes or []:
        d = dict(n)
        d['components'] = [dict(c) for c in n.get('components', []) or []]
        out.append(d)
    return out


def snapshot(explorer):
    """What a frame needs, read from a running explorer under its own locks.

    Works for IntelX and for every baseline, since they publish the same
    shared_state and latest_prediction_data. The pose and the explored map
    come straight from the simulator, which updates them every step, so the
    view is not held to the rate the simulation thread publishes its state.
    Reading them is safe: the pose is replaced, never mutated, and the map
    only ever gains cells."""
    s = {}
    with explorer.state_lock:
        ss = explorer.shared_state
        s['explored'] = ss.get('explored')
        s['pose'] = ss.get('pose')
        s['coverage'] = float(ss.get('exploration_pct', 0.0))
        s['path_m'] = float(ss.get('total_path_length', 0.0))
        s['goals'] = int(ss.get('goal_count', 0))
        s['goal'] = ss.get('current_goal')
    sim = getattr(explorer, 'simulator', None)
    if sim is not None and getattr(sim, 'explored', None) is not None:
        s['pose'] = np.array(sim.pose, dtype=float)
        s['explored'] = sim.explored
    lock = getattr(explorer, 'data_lock', None)
    if lock is not None:
        with lock:
            s['pred'] = getattr(explorer, 'latest_prediction_data', None)
    lock = getattr(explorer, 'nodes_lock', None)
    if lock is not None:
        with lock:
            s['nodes'] = _copy_nodes(getattr(explorer, 'exploration_nodes', []))
    return s


# ----------------------------------------------------------------------
# Region helpers: draw into a small copy, then put it back
# ----------------------------------------------------------------------
def _bounds(pts, pad, W, H):
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    x0 = max(0, int(np.floor(p[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(p[:, 1].min())) - pad)
    x1 = min(W, int(np.ceil(p[:, 0].max())) + pad + 1)
    y1 = min(H, int(np.ceil(p[:, 1].max())) + pad + 1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _blend_in(img, b, fn, alpha):
    """fn(overlay, dx, dy) draws into the region b with coordinates shifted by
    (-dx, -dy); the result is blended back at alpha."""
    if b is None:
        return
    x0, y0, x1, y1 = b
    roi = img[y0:y1, x0:x1]
    ov = roi.copy()
    fn(ov, x0, y0)
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, dst=roi)


class LiveView:
    """Renders snapshots of one run into frames.

    occ is the ground-truth grid (0 free, 1 wall, anything else unknown). The
    crop, the scale and the static parts of the frame are fixed once, and the
    trail and the reveal glow are kept between frames, so call render() on
    successive snapshots of one run."""

    def __init__(self, occ, *, method="ours", title="", subtitle=None, spawn=None,
                 fov=220 * np.pi / 180, max_r_px=200, win_px=400,
                 target_pct=99.0):
        self.occ = np.asarray(occ)
        self.method = method
        self.intelx = method in INTELX_METHODS
        self.title = title or S.method_label(method)
        if subtitle is None:
            subtitle = ("Predicted Coverage-Gain Field Overlaid in the World Frame"
                        if self.intelx else
                        "Frontier Candidates and the Goal Chosen, in the World Frame")
        self.subtitle = subtitle
        self.spawn = spawn
        self.fov, self.max_r, self.win = float(fov), int(max_r_px), int(win_px)
        self.target = float(target_pct or 99.0)
        self.color = S.method_color(method)
        self.accent = S.bgr(S.ROBOT_COLOR) if self.intelx else self.color
        self.trail_color = S.bgr(S.TRAIL_COLOR) if self.intelx else self.color

        H0, W0 = self.occ.shape
        x0, x1, y0, y1 = S.free_bbox(self.occ, CROP_MARGIN_PX)
        cw, ch = x1 - x0, y1 - y0
        self.x0, self.y0 = x0, y0
        self.scale = min(CANVAS_W / float(cw), CANVAS_H / float(ch))
        dw, dh = max(1, int(cw * self.scale)), max(1, int(ch * self.scale))
        self.dw, self.dh = dw, dh
        self.ox, self.oy = (CANVAS_W - dw) // 2, (CANVAS_H - dh) // 2
        # Nearest-neighbour downsampling as one flat take, fixed once, so the
        # cost of a frame tracks the canvas and not the map.
        yy = (np.arange(dh) * ch / dh).astype(np.int64) + y0
        xx = (np.arange(dw) * cw / dw).astype(np.int64) + x0
        self._take = (yy[:, None] * W0 + xx[None, :]).ravel()
        sub_occ = self.occ.ravel()[self._take].reshape(dh, dw)
        self._wall = (sub_occ != 0).astype(np.uint8)
        self._palette = np.array([S.THEME['unknown'], S.THEME['free'],
                                  S.THEME['wall']], np.uint8)
        self.age = np.zeros((dh, dw), np.int16)
        self.prev = np.zeros((dh, dw), bool)
        # Explored space only grows, so the panel is kept and only cells seen
        # for the first time are recoloured.
        self.panel = np.empty((dh, dw, 3), np.uint8)
        self.panel[:] = S.THEME['unknown']
        self._canvas_bg = np.empty((CANVAS_H, CANVAS_W, 3), np.uint8)
        self._canvas_bg[:] = S.THEME['bg']
        self.trail = []
        self.W = CANVAS_W + HUD_W + 3 * MARGIN
        self.H = CANVAS_H + HEADER_H + MARGIN
        self.hx = CANVAS_W + 2 * MARGIN
        self._glyphs = {}
        self._tpl = self._template()

    @property
    def size(self):
        return self.W, self.H

    def D(self, x, y):
        """World pixel -> canvas pixel."""
        return (int((x - self.x0) * self.scale + self.ox),
                int((y - self.y0) * self.scale + self.oy))

    # ------------------------------------------------------------------
    def _legend(self):
        L = [("Robot", 'chevron', self.accent, None)]
        if self.intelx:
            L += [("Local Window", 'square', S.bgr(S.WINDOW_COLOR), None),
                  ("Predicted Coverage Gain", 'field', None, None)]
        L += [("Laser Scan", 'scan', S.bgr(S.SCAN_COLOR), None),
              ("Spawn", 'star', S.bgr(S.SPAWN_COLOR), None),
              ("Goal Chosen", 'x', S.bgr(S.GOAL_COLOR), None)]
        if self.intelx:
            L += [("Junction Node", 'diamond', S.bgr(S.NODE_INK), S.bgr(S.NODE_FILL))]
            L += [("Branch / Component, Rank %d" % (i + 1), 'circle',
                   S.bgr(S.RANK_RAMPS[i][1]), S.bgr(S.RANK_RAMPS[i][0]))
                  for i in range(3)]
            L += [("Frontier (Fallback)", 'x', S.bgr(S.FRONTIER_INK), None)]
        else:
            L += [("Frontier Candidate", 'x', S.bgr(S.FRONTIER_INK), None)]
        L += [("Path Travelled", 'line', self.trail_color, None)]
        return L

    def _stat_rows(self):
        return 6 if self.intelx else 4

    def _template(self):
        """Everything that never changes: page, title, captions and legend."""
        T = S.THEME
        f = np.empty((self.H, self.W, 3), np.uint8)
        f[:] = T['bg']
        hx, y = self.hx, HEADER_H
        items = [(MARGIN, 18, self.title, 'display', 28, T['text'], 'la'),
                 (MARGIN, 50, self.subtitle, 'body', 14, T['muted'], 'la'),
                 (hx, y + 58, "Coverage", 'body', 13, T['muted'], 'la'),
                 (hx + 220, y + 58, "Path", 'body', 13, T['muted'], 'la')]
        self._legend_y = y + 178 + 26 * self._stat_rows() + 40
        for j, (lbl, kind, c, fc) in enumerate(self._legend()):
            S.draw_swatch(f, kind, (hx + 10, self._legend_y + 11 + j * 30), c, fc)
            items.append((hx + 30, self._legend_y + j * 30, lbl, 'body', 15,
                          T['text'], 'la'))
        return S.draw_text(f, items)

    def _glyph(self, text, color):
        """A small text tile with alpha, rendered once and reused."""
        key = (text, color)
        if key not in self._glyphs:
            font = S.fonts().get('display', 11)
            bb = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
            w, h = bb[2] - bb[0] + 2, bb[3] - bb[1] + 2
            tile = Image.new("L", (w, h), 0)
            ImageDraw.Draw(tile).text((1 - bb[0], 1 - bb[1]), text, font=font, fill=255)
            self._glyphs[key] = (np.asarray(tile, np.float32)[..., None] / 255.0,
                                 np.array(color, np.float32))
        return self._glyphs[key]

    def _paste_label(self, img, cx, cy, text, color):
        a, col = self._glyph(text, color)
        h, w = a.shape[:2]
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        if x0 < 0 or y0 < 0 or x0 + w > img.shape[1] or y0 + h > img.shape[0]:
            return
        roi = img[y0:y0 + h, x0:x0 + w].astype(np.float32)
        img[y0:y0 + h, x0:x0 + w] = (roi * (1 - a) + col * a).astype(np.uint8)

    # ------------------------------------------------------------------
    def _map(self, explored):
        img = self._canvas_bg.copy()
        panel = self.panel
        if explored is not None:
            sub = np.asarray(explored).ravel()[self._take].reshape(self.dh, self.dw)
            sub = sub.astype(bool, copy=False)
            new = sub & (~self.prev)
            if new.any():
                panel[new] = self._palette[1 + self._wall[new]]
                self.age[new] = FLASH_FRAMES
            self.prev = sub
        m = self.age > 0
        if m.any():
            # the reveal glow is drawn on a copy, so the kept panel stays clean
            panel = panel.copy()
            a = (self.age[m].astype(np.float32) / FLASH_FRAMES * FLASH_STRENGTH)[:, None]
            panel[m] = (panel[m] * (1 - a) + np.array(self.color, np.float32) * a
                        ).astype(np.uint8)
            self.age[m] -= 1
        img[self.oy:self.oy + self.dh, self.ox:self.ox + self.dw] = panel
        return img

    def _scan(self, canvas, pose, rd):
        ex, ey, hit = S.scan_fan(self.occ, float(pose[0]), float(pose[1]),
                                 float(pose[2]), self.fov, SCAN_BEAMS, self.max_r)
        ends = np.stack([(ex - self.x0) * self.scale + self.ox,
                         (ey - self.y0) * self.scale + self.oy], 1).astype(np.int32)
        col = S.bgr(S.SCAN_COLOR)
        b = _bounds(np.vstack([ends, [rd]]), 3, CANVAS_W, CANVAS_H)
        fan = np.vstack([[rd], ends]).astype(np.int32)

        def cone(o, dx, dy):
            cv2.fillPoly(o, [fan - (dx, dy)], col, cv2.LINE_AA)
        _blend_in(canvas, b, cone, SCAN_FILL_ALPHA)

        def rays(o, dx, dy):
            c = (rd[0] - dx, rd[1] - dy)
            segs = [np.array([c, (e[0] - dx, e[1] - dy)], np.int32) for e in ends]
            cv2.polylines(o, segs, False, col, 1, cv2.LINE_AA)
        _blend_in(canvas, b, rays, SCAN_RAY_ALPHA)

        hits = ends[hit]
        if len(hits):
            def dots(o, dx, dy):
                for e in hits:
                    cv2.circle(o, (int(e[0] - dx), int(e[1] - dy)), 2, col, -1, cv2.LINE_AA)
            _blend_in(canvas, b, dots, SCAN_HIT_ALPHA)

    def _field(self, canvas, pred, Md):
        """The predicted field, at the pose it was predicted FROM, blended only
        inside the warped window's footprint."""
        col, fmask = S.colorize_field(pred.get('heatmap'))
        if col is None:
            return
        w = self.win
        corners = [S.apply_pt(Md, u, v) for u, v in ((0, 0), (w, 0), (w, w), (0, w))]
        b = _bounds(corners, 2, CANVAS_W, CANVAS_H)
        if b is None:
            return
        x0, y0, x1, y1 = b
        M = Md.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        size = (x1 - x0, y1 - y0)
        wc = cv2.warpAffine(col, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        wm = cv2.warpAffine(fmask, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        # the field is mostly empty, so only its visible pixels are blended
        m = wm > 0
        if not m.any():
            return
        a = (wm[m].astype(np.float32) * (FIELD_ALPHA / 255.0))[:, None]
        roi = canvas[y0:y1, x0:x1]
        roi[m] = (roi[m] * (1 - a) + wc[m] * a).astype(np.uint8)

    def _trail(self, canvas):
        n = len(self.trail)
        if n < 2:
            return
        pts = np.asarray(self.trail, np.int32)
        b = _bounds(pts, TRAIL_WIDTH + 2, CANVAS_W, CANVAS_H)
        if b is None:
            return
        x0, y0, x1, y1 = b
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        shifted = pts - (x0, y0)
        # Oldest band first, so where segments overlap the newer one wins.
        edges = np.linspace(0, n - 1, min(TRAIL_BANDS, n - 1) + 1).astype(int)
        for k in range(len(edges) - 1):
            seg = shifted[edges[k]:edges[k + 1] + 1]
            a = TRAIL_FADE_MIN + (1.0 - TRAIL_FADE_MIN) * (edges[k + 1] / float(n - 1)) ** 0.7
            cv2.polylines(mask, [seg.reshape(-1, 1, 2)], False, int(a * 255),
                          TRAIL_WIDTH, cv2.LINE_AA)
        m = (mask.astype(np.float32) / 255.0)[..., None]
        roi = canvas[y0:y1, x0:x1]
        roi[:] = (roi * (1 - m) + np.array(self.trail_color, np.float32) * m
                  ).astype(np.uint8)

    def _robot(self, canvas, rd, heading):
        r = int(ROBOT_SIZE_PX * 2.5)
        x0, y0 = max(0, rd[0] - r), max(0, rd[1] - r)
        x1, y1 = min(CANVAS_W, rd[0] + r), min(CANVAS_H, rd[1] + r)
        if x1 <= x0 or y1 <= y0:
            return
        roi = np.ascontiguousarray(canvas[y0:y1, x0:x1])
        S.draw_chevron(roi, (rd[0] - x0, rd[1] - y0), heading, ROBOT_SIZE_PX, self.accent)
        canvas[y0:y1, x0:x1] = roi

    # ------------------------------------------------------------------
    def render(self, s, status=None):
        """One frame from one snapshot. status, when given, is how the run
        ended, and is shown in place of the live decision."""
        pose = s.get('pose')
        pred = s.get('pred') or {}
        lc = pred.get('local_components') or {}
        labels = []                          # (x, y, text, bgr)

        canvas = self._map(s.get('explored'))
        rd = None
        if pose is not None and len(pose) >= 3:
            rd = self.D(float(pose[0]), float(pose[1]))
            if not self.trail or abs(rd[0] - self.trail[-1][0]) + abs(rd[1] - self.trail[-1][1]) >= 1:
                self.trail.append(rd)
            if status is None:
                self._scan(canvas, pose, rd)

        hm = pred.get('heatmap')
        field_max = float(np.max(hm)) if hm is not None and np.size(hm) else 0.0
        Md = None
        if self.intelx and pred.get('request_pose') is not None:
            Md = S.compose(S.img_to_world_matrix(pred['request_pose'], self.win),
                           self.scale, self.x0, self.y0, self.ox, self.oy)
            if field_max > 0:
                self._field(canvas, pred, Md)

        self._trail(canvas)

        # components, outlined in their rank colour; keep_ids is strongest first
        cents = []
        keep = list(lc.get('keep_ids', []) or [])
        lm = lc.get('label_map')
        if self.intelx and Md is not None and keep and lm is not None and np.any(lm):
            lmw = cv2.warpAffine(np.asarray(lm).astype(np.uint8), Md,
                                 (CANVAS_W, CANVAS_H), flags=cv2.INTER_NEAREST,
                                 borderValue=0)
            centroids = lc.get('centroids', {}) or {}
            for ci, cid in enumerate(keep):
                fill_c, ink_c = S.RANK_RAMPS[ci % len(S.RANK_RAMPS)]
                cs, _ = cv2.findContours((lmw == int(cid)).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, cs, -1, S.bgr(ink_c), 2, cv2.LINE_AA)
                c_ = centroids.get(cid)
                if c_ is not None and np.all(np.isfinite(c_)):
                    cx, cy = S.apply_pt(Md, c_[0], c_[1])
                    cents.append((int(cx), int(cy), ci, fill_c, ink_c))

        # frontiers held for the fallback, or a baseline's candidates
        frontiers = lc.get('frontier_centroids', []) or []
        for f in frontiers:
            fd = self.D(float(f[0]), float(f[1]))
            cv2.drawMarker(canvas, fd, S.bgr(S.FRONTIER_INK), cv2.MARKER_TILTED_CROSS,
                           13, 4, cv2.LINE_AA)
            cv2.drawMarker(canvas, fd, S.bgr(S.FRONTIER_FILL),
                           cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)

        # junction nodes and their ranked branches; taken branches are faded,
        # all in one blend
        nodes = s.get('nodes') or []
        faded, pts = [], []
        for nd in nodes:
            nd_ = self.D(float(nd['x']), float(nd['y']))
            for c in nd.get('components', []):
                rk = int(c.get('rank', 1))
                fill_c, ink_c = S.RANK_RAMPS[(rk - 1) % len(S.RANK_RAMPS)]
                bd = self.D(float(c['x']), float(c['y']))
                item = (nd_, bd, S.bgr(fill_c), S.bgr(ink_c))
                if c.get('explored'):
                    faded.append(item)
                    pts += [nd_, bd]
                else:
                    cv2.line(canvas, nd_, bd, item[3], 1, cv2.LINE_AA)
                    cv2.circle(canvas, bd, 6, item[2], -1, cv2.LINE_AA)
                    cv2.circle(canvas, bd, 6, item[3], 1, cv2.LINE_AA)
                labels.append((bd[0], bd[1] - 12, str(rk), S.bgr(ink_c)))
        if faded:
            def draw_faded(o, dx, dy):
                for nd_, bd, fc, ic in faded:
                    a, b_ = (nd_[0] - dx, nd_[1] - dy), (bd[0] - dx, bd[1] - dy)
                    cv2.line(o, a, b_, ic, 1, cv2.LINE_AA)
                    cv2.circle(o, b_, 6, fc, -1, cv2.LINE_AA)
                    cv2.circle(o, b_, 6, ic, 1, cv2.LINE_AA)
            _blend_in(canvas, _bounds(pts, 8, CANVAS_W, CANVAS_H), draw_faded,
                      EXPLORED_BRANCH_ALPHA)
        for nd in nodes:
            nd_ = self.D(float(nd['x']), float(nd['y']))
            q = NODE_SIZE_PX
            dia = np.array([[nd_[0], nd_[1] - q], [nd_[0] + q, nd_[1]],
                            [nd_[0], nd_[1] + q], [nd_[0] - q, nd_[1]]],
                           np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(canvas, [dia], S.bgr(S.NODE_FILL), cv2.LINE_AA)
            cv2.polylines(canvas, [dia], True, S.bgr(S.NODE_INK), 2, cv2.LINE_AA)

        for cx, cy, ci, fill_c, ink_c in cents:
            cv2.circle(canvas, (cx, cy), 8, S.bgr(fill_c), -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 8, S.bgr(ink_c), 2, cv2.LINE_AA)
            labels.append((cx, cy, str(ci + 1), S.bgr(ink_c)))

        # the local window, aligned to the robot's real heading
        if self.intelx and rd is not None:
            h = (self.win / 2.0) * self.scale
            th = float(pose[2])
            f = np.array([np.cos(th), np.sin(th)])
            n = np.array([-np.sin(th), np.cos(th)])
            c0 = np.asarray(rd, float)
            box = np.array([c0 + h * f + h * n, c0 + h * f - h * n,
                            c0 - h * f - h * n, c0 - h * f + h * n], np.int32)
            cv2.polylines(canvas, [box.reshape(-1, 1, 2)], True,
                          S.bgr(S.WINDOW_COLOR), WINDOW_LW, cv2.LINE_AA)

        goal = s.get('goal')
        if goal is not None and len(goal) >= 2 and np.all(np.isfinite(goal[:2])):
            gd = self.D(float(goal[0]), float(goal[1]))
            cv2.drawMarker(canvas, gd, (255, 255, 255), cv2.MARKER_TILTED_CROSS,
                           16, 5, cv2.LINE_AA)
            cv2.drawMarker(canvas, gd, S.bgr(S.GOAL_COLOR), cv2.MARKER_TILTED_CROSS,
                           14, 3, cv2.LINE_AA)
        if self.spawn is not None:
            cv2.drawMarker(canvas, self.D(self.spawn[0], self.spawn[1]),
                           S.bgr(S.SPAWN_COLOR), cv2.MARKER_STAR, 16, 2, cv2.LINE_AA)
        if rd is not None:
            self._robot(canvas, rd, float(pose[2]))
        for x, y, t, c in labels:
            self._paste_label(canvas, x, y, t, c)

        return self._assemble(canvas, s, pred, len(keep) if self.intelx else None,
                              field_max, len(nodes), len(frontiers), status)

    # ------------------------------------------------------------------
    def _assemble(self, canvas, s, pred, n_comp, field_max, n_nodes, n_frontiers,
                  status):
        T = S.THEME
        frame = self._tpl.copy()
        frame[HEADER_H:HEADER_H + CANVAS_H, MARGIN:MARGIN + CANVAS_W] = canvas
        hx, y = self.hx, HEADER_H
        items = [(hx, y + 8, "%.1f%%" % s.get('coverage', 0.0), 'mono', 40,
                  self.accent, 'la'),
                 (hx + 220, y + 8, "%.0f m" % s.get('path_m', 0.0), 'mono', 40,
                  T['text'], 'la')]
        if status is None:
            items += [(hx, y + 110, "Decision", 'body', 13, T['muted'], 'la'),
                      (hx, y + 132, decision_label(pred, self.method), 'display', 22,
                       T['text'], 'la')]
        else:
            done = status == 'complete'
            items += [(hx, y + 110, "Run Ended", 'body', 13, T['muted'], 'la'),
                      (hx, y + 132, "Complete" if done else
                       status.replace('_', ' ').title(), 'display', 22,
                       T['ok'] if done else T['bad'], 'la')]
        rows = []
        if self.intelx:
            rows += [("Components", "%d" % (n_comp or 0)),
                     ("Field Max", "%.4f" % field_max),
                     ("Junctions", "%d" % n_nodes)]
        else:
            rows += [("Frontiers", "%d" % n_frontiers)]
        rows += [("Goals", "%d" % s.get('goals', 0))]
        if s.get('elapsed_s') is not None:
            m_, s_ = divmod(int(s['elapsed_s']), 60)
            rows += [("Wall Time", "%d:%02d" % (m_, s_))]
        for j, (k, v) in enumerate(rows[:self._stat_rows()]):
            items.append((hx, y + 178 + 26 * j, "%-12s %s" % (k, v), 'mono', 17,
                          T['text'], 'la'))
        S.draw_text_roi(frame, items, hx, y, self.W, self._legend_y - 4)

        # progress toward the coverage target, along the top edge
        u = min(1.0, max(0.0, s.get('coverage', 0.0) / self.target))
        cv2.line(frame, (0, 2), (int(self.W * u), 2), self.accent, 3, cv2.LINE_AA)
        return frame


class LivePlayer:
    """Shows LiveView frames in a window, writes them to an mp4, or both.

    update() renders the explorer's current state; finish() renders the last
    frame with the outcome, holds it at the end of the video and saves it as a
    PNG beside the video. Pressing Esc or q in the window sets stop_requested."""

    def __init__(self, view, *, show=True, video_path=None, fps=10.0,
                 window="IntelX", hold_s=2.0, max_window_w=1600):
        self.view = view
        self.show = bool(show)
        self.video_path = video_path
        self.fps = float(fps)
        self.window = window
        self.hold_s = float(hold_s)
        self.stop_requested = False
        self.last_frame = None
        self._t0 = time.time()
        self._vw = S.open_writer(video_path, fps, view.size) if video_path else None
        if video_path and self._vw is None:
            print("LIVE VIEW: could not open a video writer for %s" % video_path)
        if self.show:
            try:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                w, h = view.size
                f = min(1.0, max_window_w / float(w))
                cv2.resizeWindow(window, int(w * f), int(h * f))
            except cv2.error as e:
                print("LIVE VIEW: no display (%s), continuing without a window"
                      % str(e).splitlines()[0])
                self.show = False

    def push(self, frame):
        self.last_frame = frame
        if self._vw is not None:
            self._vw.write(frame)
        if self.show:
            cv2.imshow(self.window, frame)
            if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                self.stop_requested = True

    def update(self, explorer, now=None):
        s = snapshot(explorer)
        s['elapsed_s'] = (time.time() - self._t0) if now is None else now
        self.push(self.view.render(s))

    def finish(self, explorer, status):
        """The final frame, with how the run ended, held at the end."""
        try:
            s = snapshot(explorer)
            s['elapsed_s'] = time.time() - self._t0
            frame = self.view.render(s, status=status)
        except Exception:
            frame = self.last_frame
        if frame is None:
            return
        for _ in range(max(1, int(self.hold_s * self.fps))):
            self.push(frame)
        if self.video_path:
            cv2.imwrite(os.path.splitext(self.video_path)[0] + "_final.png", frame)

    def close(self):
        if self._vw is not None:
            self._vw.release()
            self._vw = None
        if self.show:
            try:
                cv2.destroyWindow(self.window)
            except cv2.error:
                pass
