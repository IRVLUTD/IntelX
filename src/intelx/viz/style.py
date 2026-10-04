"""The shared look of every IntelX video: palette, fonts, glyphs and text.

Ported from the figure scripts the paper's videos were made with, so a live run,
a single-method run and a sweep's showcase all read as one system. Text is drawn
with TrueType through PIL rather than OpenCV's Hershey strokes, and every number
uses a monospace face so digits do not jitter as coverage and path tick over.

The fonts are the DejaVu faces that ship with matplotlib, which this package
already depends on, so every machine renders the same frame.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ----------------------------------------------------------------------
# Palette
# ----------------------------------------------------------------------
THEME = dict(bg=(248, 248, 248), panel=(255, 255, 255), unknown=(150, 150, 150),
             free=(252, 252, 252), wall=(25, 25, 25), text=(28, 30, 34),
             muted=(120, 128, 138), rule=(210, 214, 220), grid=(232, 234, 238),
             ok=(70, 160, 90), bad=(60, 60, 200))

# Method order, labels and trail colours, as in the paper's figures.
ORDER = ["random", "nearest", "cost_utility", "geom_ig", "frontier",
         "hectoraug", "upen", "mapex", "local_only", "ours"]
LABEL = {"random": "Random frontier", "nearest": "Nearest frontier",
         "cost_utility": "Cost-utility", "geom_ig": "Geometric IG",
         "frontier": "Frontier (AutoX)", "hectoraug": "IG-Hector",
         "upen": "UPEN", "mapex": "MapEx", "local_only": "IntelX, local only",
         "ours": "IntelX (Ours)"}
COLOR_HEX = {"random": "#9AA3AE", "nearest": "#B49A5E", "cost_utility": "#9B7BD4",
             "geom_ig": "#56B4E9", "frontier": "#F0A830", "hectoraug": "#E48BC0",
             "upen": "#12C48B", "mapex": "#3A9BE0", "local_only": "#C0504D",
             "ours": "#FF6B2C"}

# The overlays of the method view.
RANK_RAMPS = [("#FFF2CC", "#D8A200"),    # rank 1, yellow   (fill, ink)
              ("#B1DDF0", "#1E86B0"),    # rank 2, cyan
              ("#F9CFE4", "#D6197E"),    # rank 3, pink
              ("#DCF0D6", "#3E8E5A")]    # rank 4, green
NODE_FILL, NODE_INK = "#E7D4F7", "#7E3FB8"           # junction diamond
FRONTIER_FILL, FRONTIER_INK = "#FFD8B0", "#E06A00"   # frontier cross
SPAWN_COLOR = "#2E9E4F"
GOAL_COLOR = "#D62728"
ROBOT_COLOR = "#FF6B2C"
TRAIL_COLOR = "#FF6E3C"
WINDOW_COLOR = "#BDBDBD"
SCAN_COLOR = "#FF8A1F"


def bgr(h):
    """'#RRGGBB' -> an OpenCV (B, G, R) tuple."""
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


COLOR = {m: bgr(c) for m, c in COLOR_HEX.items()}


def method_color(mode):
    return COLOR.get(mode, bgr(ROBOT_COLOR))


def method_label(mode):
    return LABEL.get(mode, mode)


# ----------------------------------------------------------------------
# Typography
# ----------------------------------------------------------------------
FONT_DISPLAY = ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"]
FONT_BODY = ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"]
FONT_MONO = ["DejaVuSansMono-Bold.ttf", "DejaVuSansMono.ttf",
             "LiberationMono-Bold.ttf", "Courier New.ttf"]


def _font_dirs():
    dirs = []
    try:
        import matplotlib
        dirs.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except Exception:
        pass
    dirs += ["/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.fonts"), os.path.expanduser("~/Library/Fonts"),
             "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]
    return dirs


class Fonts:
    """Resolve each face once, from matplotlib's bundled fonts first."""

    def __init__(self):
        self._files = {}
        for d in _font_dirs():
            if not os.path.isdir(d):
                continue
            for root, _dirs, files in os.walk(d):
                for f in files:
                    if f.lower().endswith((".ttf", ".otf")):
                        self._files.setdefault(f.lower(), os.path.join(root, f))
        self.display = self._pick(FONT_DISPLAY)
        self.body = self._pick(FONT_BODY)
        self.mono = self._pick(FONT_MONO)
        self._cache = {}

    def _pick(self, prefs):
        for p in prefs:
            hit = self._files.get(p.lower())
            if hit:
                return hit
        return None

    def get(self, which, size):
        key = (which, size)
        if key not in self._cache:
            path = getattr(self, which)
            try:
                self._cache[key] = (ImageFont.truetype(path, size) if path
                                    else ImageFont.load_default())
            except Exception:
                self._cache[key] = ImageFont.load_default()
        return self._cache[key]


_FONTS = None


def fonts():
    global _FONTS
    if _FONTS is None:
        _FONTS = Fonts()
    return _FONTS


def draw_text_roi(frame, items, x0, y0, x1, y1):
    """draw_text, converting only frame[y0:y1, x0:x1]. Item coordinates stay in
    frame pixels. Much cheaper than a full-frame pass when the text is local."""
    if not items:
        return frame
    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    shifted = [(it[0] - x0, it[1] - y0) + tuple(it[2:]) for it in items]
    frame[y0:y1, x0:x1] = draw_text(roi, shifted)
    return frame


ACRONYMS = {"slam": "SLAM", "msd": "MSD", "aws": "AWS", "kth": "KTH", "ood": "OOD"}


def pretty_map_name(stem):
    """'small_office3' -> 'Small Office 3', for titles."""
    import re
    words = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", stem.replace("_", " ")).split()
    return " ".join(ACRONYMS.get(w.lower(), w if w.isupper() else w.capitalize())
                    for w in words)


def draw_text(frame, items):
    """Draw every queued string in one PIL pass.

    Converting the frame per string would dominate the render, so callers
    queue (x, y, text, face, size, bgr_colour, anchor[, angle]) and this draws
    them together. The anchor follows PIL: 'la' left-top, 'ra' right-top,
    'mm' centred.
    """
    if not items:
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    F = fonts()
    for it in items:
        x, y, txt, which, size, col, anchor = it[:7]
        angle = it[7] if len(it) > 7 else 0
        f = F.get(which, size)
        rgb = (col[2], col[1], col[0])
        if not angle:
            d.text((x, y), txt, font=f, fill=rgb, anchor=anchor)
            continue
        # PIL cannot draw at an angle, so the string goes on its own tile,
        # which is turned and pasted.
        bb = d.textbbox((0, 0), txt, font=f)
        tile = Image.new("RGBA", (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text((2 - bb[0], 2 - bb[1]), txt, font=f,
                                  fill=rgb + (255,))
        tile = tile.rotate(angle, expand=True, resample=Image.BICUBIC)
        img.paste(tile, (int(x - tile.width / 2), int(y - tile.height / 2)), tile)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# Glyphs
# ----------------------------------------------------------------------
def blend_poly(img, fn, alpha):
    """Draw with fn onto a copy and blend it back at alpha."""
    if alpha >= 0.999:
        fn(img)
        return
    ov = img.copy()
    fn(ov)
    cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0, dst=img)


def draw_chevron(img, c, heading, size, color, border=2.6):
    """A navigation chevron pointing along `heading`, in the world frame.

    The white rim is a slightly larger chevron underneath and the colour goes
    fully on top; stroking an outline over the fill instead swallowed most of a
    small chevron."""
    loc = np.array([[1.00, 0.00], [-0.62, 0.66], [-0.26, 0.00], [-0.62, -0.66]])
    ctr = loc.mean(axis=0)
    ch, sh = np.cos(heading), np.sin(heading)
    R = np.array([[ch, -sh], [sh, ch]])

    def shape(sz, grow=0.0):
        g = loc + (loc - ctr) * (grow / max(sz, 1e-6))
        return ((g @ R.T) * sz + np.asarray(c, float)).astype(np.int32).reshape(-1, 1, 2)

    rim = shape(size, border)
    shadow = rim + np.array([[[1, 2]]], np.int32)
    blend_poly(img, lambda o: cv2.fillPoly(o, [shadow], (0, 0, 0), cv2.LINE_AA), 0.25)
    cv2.fillPoly(img, [rim], (255, 255, 255), cv2.LINE_AA)
    cv2.fillPoly(img, [shape(size)], color, cv2.LINE_AA)


def draw_trail(img, pts, color, fade_min=0.45, gamma=0.7, width=2):
    """The travelled path, fading with age, blended through one alpha mask."""
    if len(pts) < 2:
        return
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    p = np.asarray(pts, np.int32)
    n = len(p)
    for i in range(n - 1):
        a = fade_min + (1.0 - fade_min) * ((i + 1) / float(n - 1)) ** gamma
        cv2.line(mask, (int(p[i][0]), int(p[i][1])),
                 (int(p[i + 1][0]), int(p[i + 1][1])), int(a * 255), width,
                 cv2.LINE_AA)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    img[:] = (img.astype(np.float32) * (1 - m)
              + np.array(color, np.float32) * m).astype(np.uint8)


def draw_swatch(img, kind, c, color, fill=None):
    """A legend swatch drawn in the same shape as the thing on the map."""
    x, y = c
    if kind == 'chevron':
        draw_chevron(img, (x, y), 0.0, 8, color, 2.0)
    elif kind == 'square':
        cv2.rectangle(img, (x - 7, y - 7), (x + 7, y + 7), color, 2, cv2.LINE_AA)
    elif kind == 'x':
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, 13, 3, cv2.LINE_AA)
    elif kind == 'diamond':
        d = np.array([[x, y - 8], [x + 8, y], [x, y + 8], [x - 8, y]],
                     np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(img, [d], fill or color, cv2.LINE_AA)
        cv2.polylines(img, [d], True, color, 2, cv2.LINE_AA)
    elif kind == 'circle':
        cv2.circle(img, (x, y), 6, fill or color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 6, color, 2, cv2.LINE_AA)
    elif kind == 'line':
        cv2.line(img, (x - 9, y), (x + 9, y), color, 3, cv2.LINE_AA)
    elif kind == 'scan':
        for a in np.linspace(-0.7, 0.7, 5):
            cv2.line(img, (x - 8, y), (int(x - 8 + 18 * np.cos(a)),
                                       int(y + 18 * np.sin(a))), color, 1, cv2.LINE_AA)
        cv2.circle(img, (x - 8, y), 2, color, -1, cv2.LINE_AA)
    elif kind == 'star':
        cv2.drawMarker(img, (x, y), color, cv2.MARKER_STAR, 15, 2, cv2.LINE_AA)
    elif kind == 'field':
        bar = cv2.applyColorMap(np.linspace(0, 255, 18).astype(np.uint8)[None, :],
                                cv2.COLORMAP_VIRIDIS)[0]
        for i, col in enumerate(bar):
            cv2.line(img, (x - 9 + i, y - 6), (x - 9 + i, y + 6),
                     tuple(int(v) for v in col), 1)
    else:
        cv2.circle(img, (x, y), 7, color, -1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def scan_fan(occ, x, y, th, fov, n_beams, max_r):
    """Where each beam of one scan stopped, and whether it met a wall.

    The simulator's geometry, beams over the FOV around the real heading, each
    stopping at the first occupied cell, vectorised over beams. Drawing only."""
    H, W = occ.shape
    ang = np.linspace(-fov / 2.0, fov / 2.0, n_beams, dtype=np.float32) + th
    rr = np.arange(1, int(max_r) + 1, dtype=np.float32)
    xs = (x + np.cos(ang)[:, None] * rr[None, :]).astype(np.int32)
    ys = (y + np.sin(ang)[:, None] * rr[None, :]).astype(np.int32)
    inb = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    wall = np.zeros_like(inb)
    wall[inb] = occ[ys[inb], xs[inb]] == 1
    stop = wall | (~inb)
    first = np.where(stop.any(1), stop.argmax(1), stop.shape[1] - 1)
    rng = (first + 1).astype(np.float32)
    hit = wall[np.arange(len(ang)), first]
    return x + np.cos(ang) * rng, y + np.sin(ang) * rng, hit


def img_to_world_matrix(pose, win):
    """2x3 affine, window pixel -> world pixel.

    The policy's _transform_image_goal_to_world written as a matrix, so a whole
    image can be moved at once."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    rot = np.pi / 2.0 - th
    c, s = np.cos(rot), np.sin(rot)
    cc = win // 2
    return np.array([[c, -s, x - c * cc + s * cc],
                     [s, c, y - s * cc - c * cc]], np.float64)


def compose(Mw, scale, x0, y0, ox, oy):
    """window -> world -> display, as one matrix."""
    S = np.array([[scale, 0.0, ox - scale * x0],
                  [0.0, scale, oy - scale * y0],
                  [0.0, 0.0, 1.0]])
    return (S @ np.vstack([Mw, [0.0, 0.0, 1.0]]))[:2]


def apply_pt(M, u, v):
    return (M[0, 0] * u + M[0, 1] * v + M[0, 2],
            M[1, 0] * u + M[1, 1] * v + M[1, 2])


def colorize_field(hm, floor=0.04, gamma=0.8):
    """Viridis colour, and an opacity that rises with intensity, so weak gain
    fades into the map and the eye goes where the network expects coverage."""
    h = np.asarray(hm, np.float32)
    mx = float(h.max()) if h.size else 0.0
    if mx <= 0:
        return None, None
    n = np.clip(h / mx, 0, 1)
    n[n < floor] = 0.0
    col = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    alpha = (np.power(n, gamma) * 255).astype(np.uint8)
    return col, alpha


def free_bbox(occ, margin):
    """The crop that holds every free cell, with a margin, as (x0, x1, y0, y1)."""
    ys, xs = np.nonzero(occ == 0)
    if len(xs) == 0:
        return 0, occ.shape[1], 0, occ.shape[0]
    return (max(0, int(xs.min()) - margin), min(occ.shape[1], int(xs.max()) + margin),
            max(0, int(ys.min()) - margin), min(occ.shape[0], int(ys.max()) + margin))


def open_writer(path, fps, size):
    """An mp4 writer, or None. size is (width, height)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
    return vw if vw.isOpened() else None
