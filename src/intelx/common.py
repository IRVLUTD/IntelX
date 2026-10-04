"""
intelx.common
=============
Leaf-level utilities shared across the exploration stack. Nothing here imports
any other intelx_* module, so it can be imported from anywhere without cycles.

Contents (all moved verbatim from the original monolith):
  - RobotState                       : the navigation state-machine enum
  - create_model_for_inference       : checkpoint loader
  - _transform_image_goal_to_world   : image-frame goal  -> world-pixel goal
  - safe_copy_pose / safe_copy_array : type-agnostic copies
  - check_line_of_sight              : straight-line occupancy check
  - detect_doorways_from_components  : row-width chokepoint doorway detector
  - astar_with_clearance            : A* with wall-distance cost penalty

You will rarely edit this file. Policy lives in intelx.policy_ablation.
"""
import enum
import heapq
import threading
from collections import OrderedDict
import numpy as np
from math import pi, hypot

import torch

from intelx.models.unet import (
    create_unet_resnet34,
    create_unet_resnet18,
)


# --- Robot State Management ---
class RobotState(enum.Enum):
    IDLE = "idle"
    PLANNING = "planning"
    MOVING_TO_GOAL = "moving_to_goal"
    GOAL_REACHED = "goal_reached"
    PREDICTION_NEEDED = "prediction_needed"


def create_model_for_inference(checkpoint_path, device, model_type='resnet34'):
    """Load YOUR trained U-Net model"""
    print(f"Loading model: {model_type}")

    # Create model architecture
    if model_type == 'resnet18':
        model = create_unet_resnet18(in_channels=1, out_channels=1)
    elif model_type == 'resnet34':
        model = create_unet_resnet34(in_channels=1, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle state dict (remove _orig_mod. prefix if present)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Clean state dict
    clean_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_orig_mod.', '')
        clean_state_dict[new_key] = v

    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\u2713 Model loaded: {total_params:,} parameters")

    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    if 'val_r2' in checkpoint:
        print(f"  Validation R\u00b2: {checkpoint['val_r2']:.4f}")

    return model


def _transform_image_goal_to_world(goal_px, origin_pose, image_size, resolution):
    """Transforms a goal from the image frame back to the world frame."""
    center_x, center_y = image_size // 2, image_size // 2
    origin_x, origin_y, origin_theta = origin_pose
    rot_angle_deg = 90.0 - (origin_theta * 180.0 / pi)
    rot_angle_rad_inv = rot_angle_deg * np.pi / 180.0
    dx_img, dy_img = goal_px[0] - center_x, goal_px[1] - center_y
    cos_rot, sin_rot = np.cos(rot_angle_rad_inv), np.sin(rot_angle_rad_inv)
    dx_world = dx_img * cos_rot - dy_img * sin_rot
    dy_world = dx_img * sin_rot + dy_img * cos_rot
    return (origin_x + dx_world, origin_y + dy_world)


def safe_copy_pose(pose):
    """Safely copy pose regardless of type."""
    if hasattr(pose, 'copy'):
        return pose.copy()
    else:
        return np.array(pose)


def safe_copy_array(arr):
    """Safely copy array regardless of type."""
    if hasattr(arr, 'copy'):
        return arr.copy()
    else:
        return np.array(arr)


def snap_goal_to_free(world_goal, robot_pose, occ, min_advance_px=3.0, wall_margin_px=4.0):
    """Make a world-frame goal reachable by pulling it back along the ray from
    the robot until it sits on a free cell, with a margin so it never hugs a wall.

    A coverage-centroid goal can legitimately land inside a wall: the centre of
    mass of a multi-modal coverage field falls *between* two rooms, i.e. on the
    wall separating them. If that goal is handed to the simulator it rejects it
    ("goal inside obstacle") and immediately re-requests -- and a deterministic
    policy returns the identical goal, so the two threads spin forever. This
    snaps the goal to the last free cell the robot can see toward it, backed off
    `wall_margin_px` from the obstacle so the ~1 px world<->image round-trip the
    caller does cannot flip it back into the wall.

    Returns a (x, y) float world point that is free and at least
    `min_advance_px` from the robot, or None if there is no usable free cell
    along the ray (robot facing a wall) -- the caller should then fall back.
    """
    H, W = occ.shape
    rx, ry = float(robot_pose[0]), float(robot_pose[1])
    gx, gy = float(world_goal[0]), float(world_goal[1])

    def is_free(x, y):
        # NOTE: use int() truncation, matching how the simulator indexes the
        # goal cell (occ[int(wy), int(wx)]). Using round() here disagrees with
        # the sim at sub-pixel x/y near a wall (e.g. x=0.5 -> round 1 [free] but
        # int 0 [wall]) and lets a goal through that the sim then rejects.
        xi, yi = int(x), int(y)
        return 0 <= xi < W and 0 <= yi < H and occ[yi, xi] == 0

    dist = np.hypot(gx - rx, gy - ry)
    if dist < 1.0:
        return None  # goal is essentially the robot -> nothing to drive toward

    n = max(2, int(dist))
    furthest_free = 0.0
    hit_wall = False
    for i in range(1, n + 1):
        t = i / n
        x = rx + t * (gx - rx)
        y = ry + t * (gy - ry)
        if is_free(x, y):
            furthest_free = t * dist
        else:
            hit_wall = True
            break

    if furthest_free <= 0.0:
        return None

    reach = furthest_free
    if hit_wall:
        reach = furthest_free - wall_margin_px   # keep clear of the wall
    if reach < min_advance_px:
        return None

    t = reach / dist
    sx = rx + t * (gx - rx)
    sy = ry + t * (gy - ry)
    if not is_free(sx, sy):
        return None
    return (sx, sy)


def nudge_goal_along_heading(robot_pose, occ, distance_px, n_dirs=8, min_advance_px=3.0):
    """Fallback when no coverage goal is usable (blank prediction, or the goal
    snapped away to nothing): pick a free point ~distance_px away, preferring
    straight ahead and fanning outwards. Keeps a pure-local robot moving instead
    of freezing on a zero-distance goal. Returns (x, y) world point or None if
    the robot is fully boxed in."""
    rx, ry, rth = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    # try headings: straight ahead first, then +/- fan
    offsets = [0.0]
    step = 2.0 * np.pi / n_dirs
    for k in range(1, n_dirs // 2 + 1):
        offsets += [k * step, -k * step]
    for off in offsets:
        th = rth + off
        target = (rx + distance_px * np.cos(th), ry + distance_px * np.sin(th))
        snapped = snap_goal_to_free(target, robot_pose, occ, min_advance_px)
        if snapped is not None:
            return snapped
    return None


def check_line_of_sight(start_px, end_px, occ_map):
    """
    Checks if a straight line between two pixel coordinates is free of obstacles.
    Returns True if clear, False if blocked.
    """
    x0, y0 = start_px
    x1, y1 = end_px

    # Calculate distance
    dist = np.hypot(x1 - x0, y1 - y0)
    if dist < 1.0: return True

    # Sample points along the line
    num_samples = int(dist * 1.5)  # 1.5 samples per pixel
    if num_samples == 0: return True

    x_values = np.linspace(x0, x1, num_samples)
    y_values = np.linspace(y0, y1, num_samples)

    H, W = occ_map.shape

    for x, y in zip(x_values, y_values):
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            if occ_map[iy, ix] == 1:  # HIT WALL
                return False

    return True


def detect_doorways_from_components(label_map, heatmap, robot_cx, robot_cy, resolution):

    MAX_DOOR_WIDTH_M  = 3.0   # max physical width to count as doorway
    MIN_DIP_FRACTION  = 0.15  # chokepoint must be at least 15% narrower than neighbourhood avg
    NEIGHBOUR_ROWS    = 6     # how many rows above/below define the "neighbourhood"

    doorways = []
    if label_map is None or not np.any(label_map > 0):
        return doorways

    master = (label_map > 0).astype(np.uint8)
    H, W = master.shape

    row_widths = master.sum(axis=1).astype(np.float32)

    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    row_widths_smooth = uniform_filter1d(row_widths, size=3)

    inv_widths = -row_widths_smooth
    min_peak_prominence = max(1.5, row_widths_smooth.max() * 0.08)  # 8% of max width
    minima, props = find_peaks(inv_widths, prominence=min_peak_prominence, distance=4)

    max_door_px = int(MAX_DOOR_WIDTH_M / resolution)

    print(f"  DOORWAY_DBG [profile]: max_width={row_widths_smooth.max():.1f}px, "
          f"prominence_threshold={min_peak_prominence:.2f}, "
          f"max_door_px={max_door_px}, minima_found={len(minima)}")

    for row_idx in minima:
        choke_w = float(row_widths_smooth[row_idx])

        if choke_w < 1.0:
            print(f"  DOORWAY_DBG [row {row_idx}]: REJECTED choke_w<1px")
            continue
        if choke_w > max_door_px:
            print(f"  DOORWAY_DBG [row {row_idx}]: REJECTED too wide ({choke_w:.1f}px > {max_door_px}px)")
            continue

        # Neighbourhood = rows within NEIGHBOUR_ROWS above AND below, excluding the chokepoint itself
        above_rows = row_widths_smooth[max(0, row_idx - NEIGHBOUR_ROWS) : row_idx]
        below_rows = row_widths_smooth[row_idx + 1 : min(H, row_idx + NEIGHBOUR_ROWS + 1)]

        if len(above_rows) < 1 or len(below_rows) < 1:
            print(f"  DOORWAY_DBG [row {row_idx}]: REJECTED not enough neighbour rows")
            continue

        neighbour_avg = float(np.concatenate([above_rows, below_rows]).mean())

        # The dip: how much narrower is the chokepoint vs its neighbourhood?
        dip = (neighbour_avg - choke_w) / (neighbour_avg + 1e-6)

        print(f"  DOORWAY_DBG [row {row_idx}]: choke={choke_w:.1f}px ({choke_w*resolution:.2f}m), "
              f"neighbour_avg={neighbour_avg:.1f}px, dip={dip:.3f} (need>={MIN_DIP_FRACTION:.2f})")

        if dip < MIN_DIP_FRACTION:
            print(f"    -> REJECTED: dip {dip:.3f} < {MIN_DIP_FRACTION}")
            continue

        choke_xs = np.where(master[row_idx] > 0)[0]
        if len(choke_xs) == 0:
            continue

        door_img_x = float(choke_xs.mean())
        door_img_y = float(row_idx)

        print(f"    -> ACCEPTED: doorway at img=({door_img_x:.1f},{door_img_y:.1f})")

        doorways.append({
            'img_x': door_img_x,
            'img_y': door_img_y,
            'dx_img': door_img_x - robot_cx,
            'dy_img': door_img_y - robot_cy,
            'width_px': int(round(choke_w)),
            'width_m': choke_w * resolution,
        })

    return doorways


# ---------------------------------------------------------------------------
# SPEED: clearance-map cache.
# `occ` is the STATIC ground-truth map (only `explored` changes during an
# episode), so the distance transform and the wall-penalty field are constant
# for the whole run. The original recomputed both on every A* call.
# The penalty field is also precomputed as an array so the A* inner loop does
# an array lookup instead of a per-node np.exp() call. Values are bitwise
# identical to the original formula.
# Call clear_clearance_cache() if you ever mutate occ in place.
# ---------------------------------------------------------------------------
_CLEARANCE_CACHE = OrderedDict()
_CLEARANCE_LOCK = threading.Lock()
_CLEARANCE_MAXSIZE = 6   # >1 so several explorers (e.g. a baseline-vs-intelx
                         # comparison run) can each keep their map cached at once,
                         # instead of thrashing a single-entry cache to zero benefit.


def clear_clearance_cache():
    """Drop cached distance/penalty maps (call if `occ` is ever mutated)."""
    with _CLEARANCE_LOCK:
        _CLEARANCE_CACHE.clear()


def _occ_signature(occ):
    """Cheap fingerprint of a static occupancy grid (strided sample, not O(n))."""
    return (id(occ), occ.shape, occ.dtype.str,
            int(occ[::64, ::64].sum()), int(occ[::37, ::53].sum()))


def _get_clearance_maps(occ, clearance_px, wall_penalty_weight):
    key = (_occ_signature(occ), float(clearance_px), float(wall_penalty_weight))
    with _CLEARANCE_LOCK:
        hit = _CLEARANCE_CACHE.get(key)
        if hit is not None:
            _CLEARANCE_CACHE.move_to_end(key)     # LRU touch
            return hit
    # Compute OUTSIDE the lock -- the EDT is the expensive part and we don't want
    # to serialise two explorers' cold-start computations behind one lock.
    from scipy.ndimage import distance_transform_edt
    dist_map = distance_transform_edt(occ == 0).astype(np.float32)
    penalty_map = np.where(
        dist_map < clearance_px,
        wall_penalty_weight * np.exp(-dist_map / (clearance_px * 0.3)),
        0.0,
    ).astype(np.float32)
    with _CLEARANCE_LOCK:
        _CLEARANCE_CACHE[key] = (dist_map, penalty_map)
        _CLEARANCE_CACHE.move_to_end(key)
        while len(_CLEARANCE_CACHE) > _CLEARANCE_MAXSIZE:
            _CLEARANCE_CACHE.popitem(last=False)   # evict least-recently-used
    return dist_map, penalty_map


# Cap the backward scan in shortcut_path. The original scanned from the end of
# the path down to i on every step (O(n^2) line-of-sight checks). With a 1 m
# clearance requirement almost every long shortcut fails anyway, so a bounded
# lookahead loses essentially nothing. Set to 0 to restore the original
# unbounded behaviour.
SHORTCUT_MAX_LOOKAHEAD = 60


# The robot follows waypoints with a few pixels of tolerance and no collision
# check, so cutting a corner can leave its cell on a wall pixel. Every plan from
# that cell used to be refused, and the robot never moved again: a run ended
# stalled with the robot parked one pixel inside a wall. A start that falls on
# a wall is moved to the nearest passable cell within this many pixels, which
# matches the waypoint tolerance that put it there. A robot genuinely inside a
# wall is still refused. 0 restores the behaviour the submitted results were
# produced with, which SUBMITTED_BEHAVIOUR in the config does.
DEFAULT_START_SNAP_PX = 3
START_SNAP_PX = DEFAULT_START_SNAP_PX


def apply_submitted_behaviour(config):
    """Make the planners match the config's SUBMITTED_BEHAVIOUR setting."""
    global START_SNAP_PX
    START_SNAP_PX = (0 if getattr(config, 'SUBMITTED_BEHAVIOUR', False)
                     else DEFAULT_START_SNAP_PX)


def nearest_free_cell(occ, xy, max_r=None):
    """The passable cell (occ != 1) nearest to xy within max_r pixels
    (START_SNAP_PX when None), as an (x, y) int tuple, or None. xy itself is
    returned when it is passable."""
    if max_r is None:
        max_r = START_SNAP_PX
    H, W = occ.shape
    x, y = int(xy[0]), int(xy[1])
    if 0 <= x < W and 0 <= y < H and occ[y, x] != 1:
        return (x, y)
    best = None
    for r in range(1, int(max_r) + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and occ[ny, nx] != 1:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, (nx, ny))
        if best is not None:
            return best[1]
    return None


def astar_with_clearance(occ, start_px, goal_px, clearance_m, resolution):
    """
    A* with wall-distance cost penalty - no map inflation.
    Instead of dilating walls (which seals narrow corridors), we add a
    penalty to cells that are close to walls using a distance transform.
    This naturally pushes the path to the centre of corridors.
    A start that has slipped onto a wall pixel plans from the nearest free
    cell (START_SNAP_PX); a goal in an obstacle is refused.
    """
    H, W = occ.shape
    clearance_px = clearance_m / resolution  # e.g. 1.0/0.05 = 20px

    # Penalty weight: higher = stronger wall avoidance vs path length tradeoff
    # At 0.5 this roughly doubles the cost of hugging a wall vs. centre path
    WALL_PENALTY_WEIGHT = 3.0  # was 0.5

    # SPEED: both maps are cached across calls (occ is static)
    dist_map, penalty_map = _get_clearance_maps(occ, clearance_px, WALL_PENALTY_WEIGHT)

    # Use the existing A* but with the penalty baked into the cost function
    # We replicate the relevant parts here so we can inject the penalty
    start_node = (int(start_px[0]), int(start_px[1]))
    goal_node  = (int(goal_px[0]),  int(goal_px[1]))

    # Validate start/goal on the original (unmodified) map
    if not (0 <= start_node[0] < W and 0 <= start_node[1] < H) or occ[start_node[1], start_node[0]] == 1:
        snapped = nearest_free_cell(occ, start_node)
        if snapped is None:
            print("A* Clearance Warning: Start node is in an obstacle or out of bounds.")
            return None
        start_node = snapped
    if not (0 <= goal_node[0] < W and 0 <= goal_node[1] < H) or occ[goal_node[1], goal_node[0]] == 1:
        print("A* Clearance Warning: Goal node is in an obstacle or out of bounds.")
        return None

    def heuristic(a, b):
        return hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    move_costs = {
        (0,1):1.0,(0,-1):1.0,(1,0):1.0,(-1,0):1.0,
        (1,1):1.4,(1,-1):1.4,(-1,1):1.4,(-1,-1):1.4
    }

    open_set   = [(0, start_node)]
    came_from  = {}
    cost_so_far = {start_node: 0}
    closed = set()          # SPEED: the original re-expanded already-settled nodes

    # SPEED: hoist attribute/dict lookups out of the inner loop
    _push, _pop = heapq.heappush, heapq.heappop
    _pen = penalty_map
    _occ = occ
    gx, gy = goal_node

    while open_set:
        _, current = _pop(open_set)

        if current == goal_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            path = list(reversed(path))
            break

        if current in closed:
            continue
        closed.add(current)

        cx_cur, cy_cur = current
        g_cur = cost_so_far[current]

        for dx, dy in neighbors:
            nx, ny = cx_cur + dx, cy_cur + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if _occ[ny, nx] == 1:
                continue
            nb = (nx, ny)
            if nb in closed:
                continue

            # Base move cost + wall proximity penalty (precomputed lookup)
            new_cost = g_cur + move_costs[(dx, dy)] + _pen[ny, nx]

            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                _push(open_set, (new_cost + hypot(nx - gx, ny - gy), nb))
                came_from[nb] = current
    else:
        print("A* Warning: No path found.")
        return None

    # Shortcut path using original occ (no inflation - avoidance is in cost, not hard constraint)

    def is_line_free(p1, p2):
        x0, y0 = float(p1[0]), float(p1[1])
        x1, y1 = float(p2[0]), float(p2[1])
        dist = hypot(x1 - x0, y1 - y0)
        if dist < 1e-6: return True
        n = max(int(dist * 2), 2)
        for i in range(n + 1):
            t = i / n
            cx, cy = int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))
            if not (0 <= cx < W and 0 <= cy < H): return False
            if occ[cy, cx] == 1: return False
            if dist_map[cy, cx] < clearance_px:  # was clearance_px * 0.5
                return False
        return True

    # Path shortcutting (uses is_line_free with clearance check)
    def shortcut_path(path):
        if len(path) <= 2: return path
        shortcut = [path[0]]
        i = 0
        n_path = len(path)
        while i < len(path) - 1:
            # SPEED: bounded lookahead (see SHORTCUT_MAX_LOOKAHEAD above)
            if SHORTCUT_MAX_LOOKAHEAD > 0:
                j = min(n_path - 1, i + SHORTCUT_MAX_LOOKAHEAD)
            else:
                j = n_path - 1
            while j > i + 1:
                if is_line_free(path[i], path[j]):
                    shortcut.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                i += 1
                if i < len(path):
                    shortcut.append(path[i])
        return shortcut

    path = shortcut_path(path)
    return [(float(p[0]), float(p[1])) for p in path]