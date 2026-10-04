"""Every method on one map, side by side: the video a sweep ends with.

One panel per method. Its explored area is rebuilt by ray-casting the
trajectory the run stored, under that method's own sensor, with the laser fan,
a trail fading with age and a pulsing robot drawn over it. When a method stops,
its panel gets an outcome border, green for reaching the target and red for
stopping short, and the shortest completed path gets a FASTEST chip. Beside the
panels, coverage against path length for every method.

The methods advance together by DISTANCE TRAVELLED, not wall time, since path
length is what the results compare. Wall time depends on how many runs shared
the machine.

    python -m intelx.viz.showcase results/<run_name>
    python -m intelx.viz.showcase results/run_method --out results/videos

It reads the curves npz files in each given directory and its curves/, groups
them by map and seed, and writes showcase_<map>_s<seed>.mp4 for each.
intelx.cli.run_sweep_all calls it at the end of a sweep. Ported from the
scripts the paper's videos were made with.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from intelx.viz import style as S

FPS = 30
DURATION_S = 26.0
HOLD_END_S = 2.5
GRID_COLS = 3
PANEL_W, PANEL_H = 380, 248
PLOT_W = 560
MARGIN = 18
HEADER_H = 74
CROP_MARGIN_PX = 18

RAYCAST_EVERY_PX = 8.0       # 0.4 m between fused scans
RAYCAST_BEAMS = 180
POSE_CHUNK = 48              # poses cast at once
DEFAULT_FOV = 220.0 * np.pi / 180.0
DEFAULT_MAX_R = 200

FLASH_FRAMES = 5
FLASH_STRENGTH = 0.45
SCAN_BEAMS = 140
SCAN_FILL, SCAN_EDGE = 0.22, 0.55
TRAIL_SEGS = 10
TRAIL_MIN = 0.18
PULSE_HZ = 1.1
COMPLETE_PCT = 98.5          # counts as having reached the 99% target
TARGET_PCT = 99.0
DIM_OTHERS = 0.42            # opacity of every curve but IntelX's
SENSOR_PREFERENCE = {"mapex": ["mapex", "intelx"], "upen": ["mapex", "intelx"],
                     "hectoraug": ["mapex", "intelx"]}


def _xp():
    """cupy when a working GPU is there, numpy otherwise. Same result."""
    try:
        import cupy as cp
        a = cp.asarray(np.zeros((4, 4), np.uint8))
        _ = int((a[cp.asarray([0, 1])][:, cp.asarray([0, 1])] + 1).sum())
        return cp
    except Exception:
        return np


def _np(a):
    return a.get() if hasattr(a, 'get') else a


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def index_runs(dirs):
    """{(map, seed): {mode: (curve, traj, meta)}} from every curves npz.

    Read from each file's own metadata, so the benchmark's curves/ and the
    single-method runner's output both work. Where a method ran under several
    sensors, its own is preferred."""
    found = {}
    for d in dirs:
        # the folder itself and its curves/, never records/, which holds one
        # npz per decision
        files = (glob.glob(os.path.join(d, "*.npz"))
                 + glob.glob(os.path.join(d, "curves", "*.npz")))
        for f in sorted(files):
            try:
                z = np.load(f, allow_pickle=True)
                if not {"curve", "traj", "meta"} <= set(z.files):
                    continue
                meta = json.loads(str(z["meta"]))
                curve, traj = z["curve"], z["traj"]
            except Exception:
                continue
            if len(curve) < 2 or len(traj) < 2:
                continue
            key, mode = (meta["map"], int(meta["seed"])), meta["mode"]
            pref = SENSOR_PREFERENCE.get(mode, ["intelx", "mapex"])
            rank = pref.index(meta.get("sensor")) if meta.get("sensor") in pref else 9
            old = found.get(key, {}).get(mode)
            if old is None or rank < old[3] or (rank == old[3] and os.path.getmtime(f) > old[4]):
                found.setdefault(key, {})[mode] = (curve, traj, meta, rank, os.path.getmtime(f))
    return {k: {m: v[:3] for m, v in g.items()} for k, g in found.items()}


def load_occ(path):
    """The map the simulator ran on: 0 free, 1 wall."""
    from intelx.sim.simulator import load_pgm_map, seal_by_adjacent
    occ = seal_by_adjacent(load_pgm_map(path))
    return (occ != 0).astype(np.int8)


def to_landscape(occ, runs, spawn):
    """Turn a portrait map, and everything in its pixels, a quarter turn."""
    H, W = occ.shape
    ys, xs = np.nonzero(occ == 0)
    if (xs.max() - xs.min()) >= (ys.max() - ys.min()):
        return occ, runs, spawn
    out = {}
    for m, (curve, traj, meta) in runs.items():
        t = traj.copy()
        t[:, 1], t[:, 2] = traj[:, 2], (W - 1) - traj[:, 1]
        t[:, 3] = traj[:, 3] - np.pi / 2.0
        out[m] = (curve, t, meta)
    sp = [spawn[1], (W - 1) - spawn[0]] + list(spawn[2:]) if spawn is not None else None
    return np.ascontiguousarray(np.rot90(occ, 1)), out, sp


def resample(curve, traj, grid):
    """Each run at the path lengths on `grid`. curve rows are (t, path_m, cov),
    traj rows are (t, x, y, theta)."""
    t_c, p_c, c_c = curve[:, 0], curve[:, 1], curve[:, 2]
    t_at = np.interp(grid, p_c, t_c)
    return dict(th=np.interp(t_at, traj[:, 0], np.unwrap(traj[:, 3])),
                x=np.interp(t_at, traj[:, 0], traj[:, 1]),
                y=np.interp(t_at, traj[:, 0], traj[:, 2]),
                cov=np.interp(t_at, t_c, c_c), path=np.interp(t_at, t_c, p_c),
                live=grid <= p_c[-1] + 1e-9, final_cov=float(c_c[-1]),
                final_path=float(p_c[-1]), traj=traj)


def cast_batch(xp, occ, explored, poses, fov, n_beams, max_r):
    """Fuse many scans at once; the simulator's per-beam walk, vectorised."""
    if len(poses) == 0:
        return
    H, W = occ.shape
    r = xp.arange(1, int(max_r) + 1, dtype=xp.float32)
    for c0 in range(0, len(poses), POSE_CHUNK):
        ch = poses[c0:c0 + POSE_CHUNK]
        px = xp.asarray(ch[:, 0], dtype=xp.float32)[:, None, None]
        py = xp.asarray(ch[:, 1], dtype=xp.float32)[:, None, None]
        th = xp.asarray(ch[:, 2], dtype=xp.float32)[:, None]
        ang = xp.linspace(-fov / 2.0, fov / 2.0, n_beams, dtype=xp.float32)[None, :] + th
        xs = (px + xp.cos(ang)[:, :, None] * r[None, None, :]).astype(xp.int32)
        ys = (py + xp.sin(ang)[:, :, None] * r[None, None, :]).astype(xp.int32)
        inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        xs, ys = xp.clip(xs, 0, W - 1), xp.clip(ys, 0, H - 1)
        blocked = (occ[ys, xs] == 1) | (~inb)
        first = xp.where(blocked.any(2), blocked.argmax(2), blocked.shape[2] - 1)
        keep = (xp.arange(blocked.shape[2])[None, None, :] <= first[:, :, None]) & inb
        explored[ys[keep], xs[keep]] = True


def ease(u):
    return u * u * (3.0 - 2.0 * u)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def build_panel(xp, sub_occ, explored, yy, xx, age, prev, color):
    sub = explored[yy][:, xx]
    new = sub & (~prev)
    age[new] = FLASH_FRAMES
    img = xp.empty(sub.shape + (3,), dtype=xp.uint8)
    img[:] = xp.asarray(S.THEME['unknown'], dtype=xp.uint8)
    img[sub & (sub_occ == 0)] = xp.asarray(S.THEME['free'], dtype=xp.uint8)
    img[sub & (sub_occ == 1)] = xp.asarray(S.THEME['wall'], dtype=xp.uint8)
    if bool((age > 0).any()):
        a = (age.astype(xp.float32) / FLASH_FRAMES)[..., None] * FLASH_STRENGTH
        img = (img.astype(xp.float32) * (1 - a)
               + xp.asarray(color, dtype=xp.float32) * a).astype(xp.uint8)
    return np.ascontiguousarray(_np(img)), sub


def panel(img, r, kk, color, geo, spawn, t, occ, sensor, live):
    """The scan, trail, spawn and robot over one method's panel."""
    x0, y0, scale, ox, oy = geo
    ph, pw = img.shape[:2]

    def P(i):
        return (int((r["x"][i] - x0) * scale) + ox, int((r["y"][i] - y0) * scale) + oy)

    if live:
        fov, mr = sensor
        ex, ey, _ = S.scan_fan(occ, r["x"][kk], r["y"][kk], r["th"][kk], fov, SCAN_BEAMS, mr)
        pts = np.stack([(ex - x0) * scale + ox, (ey - y0) * scale + oy], 1)
        poly = np.vstack([[P(kk)], pts]).astype(np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], color, cv2.LINE_AA)
        cv2.addWeighted(ov, SCAN_FILL, img, 1.0 - SCAN_FILL, 0, dst=img)
        ov = img.copy()
        cv2.polylines(ov, [pts.astype(np.int32)], False, color, 1, cv2.LINE_AA)
        for q in pts[::6].astype(np.int32):
            cv2.circle(ov, (int(q[0]), int(q[1])), 1, color, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, SCAN_EDGE, img, 1.0 - SCAN_EDGE, 0, dst=img)

    if kk > 2:
        # Fade by blending with the panel, segment by segment, inside each
        # segment's own box: correct over any background and cheap.
        step = max(1, kk // TRAIL_SEGS)
        for s0 in range(0, kk, step):
            s1 = min(kk, s0 + step)
            f = TRAIL_MIN + (1.0 - TRAIL_MIN) * (s1 / float(kk))
            pts = np.array([P(i) for i in range(s0, s1 + 1)], np.int32).reshape(-1, 1, 2)
            xl, xh = max(0, int(pts[:, 0, 0].min()) - 6), min(pw, int(pts[:, 0, 0].max()) + 7)
            yl, yh = max(0, int(pts[:, 0, 1].min()) - 6), min(ph, int(pts[:, 0, 1].max()) + 7)
            if xh <= xl or yh <= yl:
                continue
            roi = img[yl:yh, xl:xh]
            sh = pts - np.array([[[xl, yl]]], np.int32)
            ov = roi.copy()
            cv2.polylines(ov, [sh], False, color, 6, cv2.LINE_AA)        # glow
            cv2.addWeighted(ov, f * 0.35, roi, 1.0 - f * 0.35, 0, dst=roi)
            ov = roi.copy()
            cv2.polylines(ov, [sh], False, color, 2, cv2.LINE_AA)        # core
            cv2.addWeighted(ov, f, roi, 1.0 - f, 0, dst=roi)

    if spawn is not None:
        sp = (int((spawn[0] - x0) * scale) + ox, int((spawn[1] - y0) * scale) + oy)
        cv2.drawMarker(img, sp, S.bgr(S.SPAWN_COLOR), cv2.MARKER_STAR, 12, 2, cv2.LINE_AA)

    p = P(kk)
    if live:
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * PULSE_HZ * t)
        for rad, fade in ((int(11 + 5 * pulse), 0.20), (int(7 + 2 * pulse), 0.40)):
            ov = img.copy()
            cv2.circle(ov, p, rad, color, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, fade, img, 1.0 - fade, 0, dst=img)
        cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, 4, (250, 250, 250), 1, cv2.LINE_AA)
    elif r["final_cov"] >= COMPLETE_PCT:
        cv2.circle(img, p, 8, color, 2, cv2.LINE_AA)
        cv2.circle(img, p, 3, color, -1, cv2.LINE_AA)
    else:
        cv2.drawMarker(img, p, color, cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
    return img


def plot(w, h, runs, k, xmax, modes):
    T = S.THEME
    img = np.empty((h, w, 3), np.uint8)
    img[:] = T['bg']
    L, R, Tp, B = 62, 18, 34, 46
    pw, ph = w - L - R, h - Tp - B
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = int(Tp + ph * (1.0 - frac))
        cv2.line(img, (L, yy), (L + pw, yy), T['grid'], 1, cv2.LINE_AA)
    yt = int(Tp + ph * (1.0 - TARGET_PCT / 100.0))
    for xx in range(L, L + pw, 14):
        cv2.line(img, (xx, yt), (min(xx + 7, L + pw), yt), T['muted'], 1)
    cv2.line(img, (L, Tp), (L, Tp + ph), T['rule'], 1)
    cv2.line(img, (L, Tp + ph), (L + pw, Tp + ph), T['rule'], 1)
    for mode in modes:
        r = runs[mode]
        n = min(k + 1, len(r["path"]))
        if n < 2:
            continue
        px = L + r["path"][:n] / max(xmax, 1e-9) * pw
        py = Tp + ph * (1.0 - r["cov"][:n] / 100.0)
        pts = np.stack([px, py], 1).astype(np.int32).reshape(-1, 1, 2)
        c = S.method_color(mode)
        if mode == "ours":
            ov = img.copy()
            cv2.polylines(ov, [pts], False, c, 8, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.30, img, 0.70, 0, dst=img)
            cv2.polylines(img, [pts], False, c, 3, cv2.LINE_AA)
            cv2.circle(img, (int(px[-1]), int(py[-1])), 5, c, -1, cv2.LINE_AA)
            cv2.circle(img, (int(px[-1]), int(py[-1])), 5, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # every other method is quieter, so IntelX reads as the subject
            ov = img.copy()
            cv2.polylines(ov, [pts], False, c, 2, cv2.LINE_AA)
            cv2.circle(ov, (int(px[-1]), int(py[-1])), 3, c, -1, cv2.LINE_AA)
            cv2.addWeighted(ov, DIM_OTHERS, img, 1.0 - DIM_OTHERS, 0, dst=img)
    return img, (L, Tp, pw, ph)


def pick_winner(runs):
    """Shortest path among the methods that finished the map."""
    done = [(r["final_path"], m) for m, r in runs.items() if r["final_cov"] >= COMPLETE_PCT]
    return min(done)[1] if done else None


def render(name, subtitle, runs, occ, sensors, spawn, out_path):
    """One map, every method in `runs`, written to out_path."""
    T = S.THEME
    xp = _xp()
    x0, x1, y0, y1 = S.free_bbox(occ, CROP_MARGIN_PX)
    cw, ch = x1 - x0, y1 - y0
    scale = min(PANEL_W / float(cw), PANEL_H / float(ch))
    dw, dh = max(1, int(cw * scale)), max(1, int(ch * scale))
    ox, oy = (PANEL_W - dw) // 2, (PANEL_H - dh) // 2
    yy = xp.asarray((np.arange(dh) * ch / dh).astype(np.int32) + y0)
    xx = xp.asarray((np.arange(dw) * cw / dw).astype(np.int32) + x0)
    occ_g = xp.asarray(occ)
    sub_occ = occ_g[yy][:, xx]

    modes = [m for m in S.ORDER if m in runs] + [m for m in runs if m not in S.ORDER]
    winner = pick_winner(runs)
    rows = int(np.ceil(len(modes) / GRID_COLS))
    cell_h = PANEL_H + 34
    grid_w = GRID_COLS * PANEL_W + (GRID_COLS + 1) * MARGIN
    W = grid_w + PLOT_W + MARGIN
    H = max(HEADER_H + rows * (cell_h + MARGIN) + MARGIN, 600)

    masks = {m: xp.zeros(occ.shape, dtype=bool) for m in modes}
    age = {m: xp.zeros((dh, dw), dtype=xp.int16) for m in modes}
    prev = {m: xp.zeros((dh, dw), dtype=bool) for m in modes}
    done_to = {m: 0.0 for m in modes}
    cum = {m: np.concatenate([[0.0], np.cumsum(np.hypot(
        np.diff(runs[m]["traj"][:, 1]), np.diff(runs[m]["traj"][:, 2])))]) for m in modes}
    res = 0.05
    n = len(runs[modes[0]]["x"])
    xmax = max(r["final_path"] for r in runs.values())
    vw = S.open_writer(out_path, FPS, (W, H))
    if vw is None:
        print("  could not open a writer for %s" % out_path)
        return False

    frame = None
    for k in range(n):
        t = k / float(FPS)
        frame = np.empty((H, W, 3), np.uint8)
        frame[:] = T['bg']
        items = []
        for i, mode in enumerate(modes):
            r = runs[mode]
            color = S.method_color(mode)
            want = r["path"][k] / res
            if want > done_to[mode]:
                tj, cm = r["traj"], cum[mode]
                lo = np.searchsorted(cm, done_to[mode])
                hi = min(np.searchsorted(cm, want) + 1, len(tj))
                sel, last = [], done_to[mode]
                for j in range(lo, hi):
                    if j != lo and cm[j] - last < RAYCAST_EVERY_PX:
                        continue
                    last = cm[j]
                    sel.append((tj[j, 1], tj[j, 2], tj[j, 3]))
                if sel:
                    fov, mr = sensors[mode]
                    cast_batch(xp, occ_g, masks[mode], np.asarray(sel, np.float32),
                               fov, RAYCAST_BEAMS, mr)
                done_to[mode] = want
            small, sub = build_panel(xp, sub_occ, masks[mode], yy, xx, age[mode],
                                     prev[mode], color)
            prev[mode] = sub
            age[mode] = xp.maximum(age[mode] - 1, 0)
            live = bool(r["live"][k]) and k < n - 1
            img = np.empty((PANEL_H, PANEL_W, 3), np.uint8)
            img[:] = T['bg']
            img[oy:oy + dh, ox:ox + dw] = small
            img = panel(img, r, k, color, (x0, y0, scale, ox, oy), spawn, t, occ,
                        sensors[mode], live)
            gr, gc = divmod(i, GRID_COLS)
            px, py = MARGIN + gc * (PANEL_W + MARGIN), HEADER_H + gr * (cell_h + MARGIN)
            frame[py:py + PANEL_H, px:px + PANEL_W] = img
            # the border gives the OUTCOME, and only once that run has stopped
            if live:
                bcol, blw = T['rule'], 1
            elif r["final_cov"] >= COMPLETE_PCT:
                bcol, blw = T['ok'], 2
            else:
                bcol, blw = T['bad'], 2
            if mode == winner and not live:
                blw = 3
            cv2.rectangle(frame, (px - 1, py - 1), (px + PANEL_W, py + PANEL_H), bcol,
                          blw, cv2.LINE_AA)
            cv2.line(frame, (px, py + PANEL_H + 17), (px + 18, py + PANEL_H + 17),
                     color, 3, cv2.LINE_AA)
            if mode == winner and not live:
                cw_, chp = 66, 19
                cx0, cy0 = px + PANEL_W - cw_ - 6, py + 6
                cv2.rectangle(frame, (cx0, cy0), (cx0 + cw_, cy0 + chp), (255, 255, 255),
                              -1, cv2.LINE_AA)
                cv2.rectangle(frame, (cx0, cy0), (cx0 + cw_, cy0 + chp), T['ok'], 1,
                              cv2.LINE_AA)
                items.append((cx0 + cw_ // 2, cy0 + chp // 2, "FASTEST", 'display', 12,
                              T['ok'], 'mm'))
            items.append((px + 26, py + PANEL_H + 10, S.method_label(mode),
                          'display' if mode == winner else 'body', 15,
                          T['text'] if mode == winner else T['muted'], 'la'))
            items.append((px + PANEL_W, py + PANEL_H + 10,
                          "%5.1f%%   %6.0f m" % (r["cov"][k], r["path"][k]), 'mono', 14,
                          color if mode == winner else T['muted'], 'ra'))

        pl, (L, Tp, pw, ph) = plot(PLOT_W, H - HEADER_H - MARGIN, runs, k, xmax, modes)
        frame[HEADER_H:HEADER_H + pl.shape[0], grid_w:grid_w + PLOT_W] = pl
        items += [
            (grid_w + L - 44, HEADER_H + Tp + ph // 2, "MAP COVERAGE  (%)", 'body', 14,
             T['muted'], 'mm', 90),
            (grid_w + L + pw // 2, HEADER_H + Tp + ph + 26, "PATH LENGTH  (m)", 'body', 14,
             T['muted'], 'ma'),
            (grid_w + L - 10, HEADER_H + Tp, "100", 'mono', 13, T['muted'], 'ra'),
            (grid_w + L - 10, HEADER_H + Tp + ph, "0", 'mono', 13, T['muted'], 'ra'),
            (grid_w + L + pw, HEADER_H + Tp + ph + 6, "%d" % int(xmax), 'mono', 13,
             T['muted'], 'ra'),
            (MARGIN, 20, name, 'display', 30, T['text'], 'la'),
            (MARGIN, 52, subtitle.upper(), 'body', 14, T['muted'], 'la'),
        ]
        # the shortest completed path so far, which ticks down as faster
        # methods finish
        now = runs[modes[0]]["path"][k]
        fin = [(r["final_path"], m) for m, r in runs.items()
               if r["final_cov"] >= COMPLETE_PCT and r["final_path"] <= now + 1e-9]
        if fin:
            best_p, best_m = min(fin)
            items += [(W - MARGIN, 14, "%.0f m" % best_p, 'mono', 30,
                       S.method_color(best_m), 'ra'),
                      (W - MARGIN, 52, "MINIMUM DISTANCE:  %s" % S.method_label(best_m).upper(),
                       'body', 13, T['muted'], 'ra')]
        else:
            items += [(W - MARGIN, 14, "--", 'mono', 30, T['muted'], 'ra'),
                      (W - MARGIN, 52, "MINIMUM DISTANCE:", 'body', 13, T['muted'], 'ra')]
        cv2.line(frame, (0, 2), (int(W * (k + 1) / float(n)), 2), S.method_color('ours'),
                 3, cv2.LINE_AA)
        frame = S.draw_text(frame, items)
        vw.write(frame)
    for _ in range(int(HOLD_END_S * FPS)):
        vw.write(frame)
    vw.release()
    return True


def render_dirs(dirs, out_dir=None, quiet=False):
    """A showcase per (map, seed) found under dirs. Returns the paths written."""
    made = []
    groups = index_runs(dirs)
    n_frames = int(DURATION_S * FPS)
    for (map_name, seed), raw in sorted(groups.items()):
        meta0 = next(iter(raw.values()))[2]
        occ = load_occ(meta0["map_path"])
        spawn = meta0.get("spawn")
        occ, raw, spawn = to_landscape(occ, raw, spawn)
        hi = max(float(c[-1, 1]) for c, _, _ in raw.values())
        if hi <= 0:
            continue
        grid = ease(np.linspace(0.0, 1.0, n_frames)) * hi
        runs, sensors = {}, {}
        for mode, (curve, traj, meta) in raw.items():
            runs[mode] = resample(curve, traj, grid)
            sp = meta.get("sensor_profile") or {}
            sensors[mode] = (float(sp.get("FOV", DEFAULT_FOV)), int(sp.get("MAX_R", DEFAULT_MAX_R)))
        d = out_dir or os.path.join(dirs[0], "videos")
        out = os.path.join(d, "showcase_%s_s%d.mp4" % (map_name, seed))
        if not quiet:
            print("showcase  %s, seed %d: %s" % (map_name, seed, ", ".join(
                S.method_label(m) for m in S.ORDER if m in runs)))
        if render(S.pretty_map_name(map_name),
                  "%d methods, synchronised by distance travelled" % len(runs),
                  runs, occ, sensors, spawn, out):
            made.append(out)
            if not quiet:
                print("          wrote %s" % out)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m intelx.viz.showcase",
                                 description="Every method on one map, side by side.")
    ap.add_argument("dirs", nargs="+", help="run directories holding curves npz files")
    ap.add_argument("--out", default=None, help="default <first dir>/videos")
    a = ap.parse_args(argv)
    made = render_dirs(a.dirs, a.out)
    print("%d video(s)" % len(made))
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())
