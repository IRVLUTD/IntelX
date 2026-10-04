"""
intelx.metrics
==============
Every metric for one exploration run, as PURE functions of the raw recording.
No sim/model imports, so this is unit-testable and -- importantly -- re-runnable
offline: the sweep stores the raw arrays per run, so a metric added later can be
computed from the stored data WITHOUT repeating the experiments.

RAW RECORDING (what the sweep stores per run, in one .npz):
    curve      (N,3) float  [wall_s, path_m, coverage_pct]   sampled at POLL_S
    traj       (N,4) float  [wall_s, x_px, y_px, theta_rad]  sampled at POLL_S
    goals      (M,3) float  [wall_s, path_m, goal_index]     one row per new goal
    meta       dict         map/resolution/free-area/config

METRIC GROUPS
    efficiency : path_at_{90,95,99}, total_path_m, auc_coverage, area_per_metre,
                 path_norm  (path / sqrt(free_area) -- comparable across maps)
    completion : success, final_coverage, reached_*, stop_reason, censored
    compute    : wall_s, planning_calls, planning_per_m, ms_per_decision, s_per_m
    quality    : reversals_per_100m, turn_rad_per_m, revisit_ratio

NOTE ON AUC (the "rate of exploration" scalar): coverage is integrated over a
FIXED path budget shared by every method on that map, with coverage held at its
final value once a run completes. A method that finishes early therefore holds
high coverage for the remainder and scores high. Normalising by each run's own
path length instead would perversely reward runs that stop early at low coverage.
"""
import numpy as np

# heading change (deg) beyond which a step counts as a direction reversal
REVERSAL_DEG = 110.0
# ignore sub-pixel jitter when measuring headings
MIN_STEP_PX = 1.0


# ----------------------------------------------------------------- efficiency
def path_at_coverage(curve, pct):
    """Path length (m) at which coverage first reaches `pct`. None if never."""
    for row in curve:
        if row[2] >= pct:
            return float(row[1])
    return None


def auc_coverage(curve, path_budget_m, final_hold=True):
    """Normalised area under coverage(path) over [0, path_budget_m] -> [0,1].

    This is the single-scalar 'rate of exploration'. Higher = covered more, sooner.
    If the run ends before the budget and final_hold, coverage is held at its last
    value for the remainder (the robot is done; the map stays explored)."""
    if len(curve) == 0 or path_budget_m <= 0:
        return 0.0
    c = np.asarray(curve, dtype=float)
    p = np.clip(c[:, 1], 0, path_budget_m)
    cov = np.clip(c[:, 2], 0, 100.0)
    # dedupe/monotonise path for a well-defined integral
    order = np.argsort(p, kind='stable')
    p, cov = p[order], cov[order]
    area = np.trapezoid(cov, p) if hasattr(np, 'trapezoid') else np.trapz(cov, p)
    if final_hold and p[-1] < path_budget_m:
        area += cov[-1] * (path_budget_m - p[-1])
    return float(area / (100.0 * path_budget_m))


def area_per_metre(curve, free_area_m2):
    """Square metres of free space covered per metre travelled (final)."""
    if len(curve) == 0:
        return 0.0
    total_path = float(curve[-1][1])
    if total_path <= 1e-9:
        return 0.0
    return float(curve[-1][2] / 100.0 * free_area_m2 / total_path)


def path_normalised(total_path_m, free_area_m2):
    """Path length in units of sqrt(free area) -- dimensionless, so runs on maps
    of very different size can be pooled or averaged meaningfully."""
    if free_area_m2 <= 0:
        return None
    return float(total_path_m / np.sqrt(free_area_m2))


# ----------------------------------------------------------------- quality
def _headings(traj):
    """Headings (rad) of successive moves, ignoring sub-pixel jitter."""
    if len(traj) < 2:
        return np.zeros(0), np.zeros(0)
    t = np.asarray(traj, dtype=float)
    dx = np.diff(t[:, 1])
    dy = np.diff(t[:, 2])
    step = np.hypot(dx, dy)
    keep = step >= MIN_STEP_PX
    return np.arctan2(dy[keep], dx[keep]), step[keep]


def turning_and_reversals(traj, resolution_m=0.05):
    """(total_turn_rad, reversals) over the trajectory.

    A reversal is a heading change beyond REVERSAL_DEG between consecutive moves
    -- i.e. the robot doubling back. This is the quantitative version of the
    oscillation behaviour, and it separates a committed planner from one that
    ping-pongs even when both cover the map."""
    h, _ = _headings(traj)
    if len(h) < 2:
        return 0.0, 0
    d = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi)
    return float(d.sum()), int((np.degrees(d) > REVERSAL_DEG).sum())


def revisit_ratio(traj, cell_px=1.0):
    """Fraction of trajectory samples whose grid cell had already been visited.
    High = the robot spends its path re-crossing ground it has already covered."""
    if len(traj) == 0:
        return 0.0
    t = np.asarray(traj, dtype=float)
    cells = np.stack([np.floor(t[:, 1] / cell_px), np.floor(t[:, 2] / cell_px)], 1)
    seen = set()
    revisits = 0
    for cx, cy in cells:
        k = (int(cx), int(cy))
        if k in seen:
            revisits += 1
        else:
            seen.add(k)
    return float(revisits / len(cells))


# ----------------------------------------------------------------- assembly
def compute_all(curve, traj, goals, meta):
    """All metrics for one run. `meta` supplies:
    path_budget_m, free_area_m2, resolution, target_pct, wall_s, stop_reason."""
    curve = np.asarray(curve, dtype=float).reshape(-1, 3) if len(curve) else np.zeros((0, 3))
    traj = np.asarray(traj, dtype=float).reshape(-1, 4) if len(traj) else np.zeros((0, 4))
    goals = np.asarray(goals, dtype=float).reshape(-1, 3) if len(goals) else np.zeros((0, 3))

    budget = float(meta.get('path_budget_m', 400.0))
    free_area = float(meta.get('free_area_m2', 0.0))
    target = float(meta.get('target_pct', 98.0))
    wall_s = float(meta.get('wall_s', 0.0))
    checkpoints = meta.get('checkpoints', [90.0, 95.0, 99.0])

    total_path = float(curve[-1][1]) if len(curve) else 0.0
    final_cov = float(curve[-1][2]) if len(curve) else 0.0
    out = {'total_path_m': total_path, 'final_coverage': final_cov}

    # efficiency ---------------------------------------------------------
    for c in checkpoints:
        v = path_at_coverage(curve, c)
        out[f'path_at_{int(c)}'] = v if v is not None else budget   # censored at budget
        out[f'reached_{int(c)}'] = int(v is not None)
    out['auc_coverage'] = auc_coverage(curve, budget)
    out['area_per_metre'] = area_per_metre(curve, free_area)
    pn = path_normalised(total_path, free_area)
    out['path_norm'] = pn if pn is not None else ''

    # completion ---------------------------------------------------------
    out['success'] = int(final_cov >= target)
    out['stop_reason'] = meta.get('stop_reason', '')
    out['censored'] = int(meta.get('stop_reason', '') in
                          ('budget', 'timeout', 'error', 'stalled', 'no_progress', 'exhausted'))

    # compute ------------------------------------------------------------
    n_goals = int(len(goals))
    out['wall_s'] = wall_s
    out['planning_calls'] = n_goals
    out['planning_per_m'] = float(n_goals / total_path) if total_path > 1e-9 else 0.0
    out['s_per_m'] = float(wall_s / total_path) if total_path > 1e-9 else 0.0
    out['ms_per_decision'] = float(wall_s * 1000.0 / n_goals) if n_goals else 0.0

    # quality ------------------------------------------------------------
    turn, rev = turning_and_reversals(traj, meta.get('resolution', 0.05))
    out['total_turn_rad'] = turn
    out['turn_rad_per_m'] = float(turn / total_path) if total_path > 1e-9 else 0.0
    out['reversals'] = rev
    out['reversals_per_100m'] = float(rev * 100.0 / total_path) if total_path > 1e-9 else 0.0
    out['revisit_ratio'] = revisit_ratio(traj)
    return out


METRIC_COLUMNS = [
    'total_path_m', 'final_coverage',
    'path_at_90', 'reached_90', 'path_at_95', 'reached_95', 'path_at_99', 'reached_99',
    'auc_coverage', 'area_per_metre', 'path_norm',
    'success', 'stop_reason', 'censored',
    'wall_s', 'planning_calls', 'planning_per_m', 's_per_m', 'ms_per_decision',
    'total_turn_rad', 'turn_rad_per_m', 'reversals', 'reversals_per_100m', 'revisit_ratio',
]

# For the results table: which way is better, and how to format.
METRIC_INFO = {
    'path_at_90':        ('Path@90 (m)',      'lower',  '{:.1f}'),
    'path_at_95':        ('Path@95 (m)',      'lower',  '{:.1f}'),
    'path_at_99':        ('Path@99 (m)',      'lower',  '{:.1f}'),
    'total_path_m':      ('Total path (m)',   'lower',  '{:.1f}'),
    'path_norm':         ('Norm. path',       'lower',  '{:.2f}'),
    'auc_coverage':      ('AUC cov.',         'higher', '{:.3f}'),
    'area_per_metre':    ('m2 / m',           'higher', '{:.2f}'),
    'success':           ('Success (%)',      'higher', '{:.0f}'),
    'final_coverage':    ('Coverage (%)',     'higher', '{:.1f}'),
    'wall_s':            ('Time (s)',         'lower',  '{:.0f}'),
    's_per_m':           ('s / m',            'lower',  '{:.2f}'),
    'ms_per_decision':   ('ms / decision',    'lower',  '{:.0f}'),
    'planning_per_m':    ('Plans / m',        'lower',  '{:.2f}'),
    'reversals_per_100m': ('Reversals/100m',  'lower',  '{:.1f}'),
    'turn_rad_per_m':    ('Turn (rad/m)',     'lower',  '{:.2f}'),
    'revisit_ratio':     ('Revisit',          'lower',  '{:.2f}'),
}