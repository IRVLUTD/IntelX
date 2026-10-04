"""Time the monolith's A* against the cached one, on the real map."""
import time

import numpy as np

from intelx.sim.simulator import ExplorationSimulator
from intelx.baselines.frontier import astar_with_clearance as astar_monolith
from intelx.common import astar_with_clearance as astar_cached
from intelx.probes.tier1 import SIM_CFG, SPAWN_XY, RESOLUTION

sim = ExplorationSimulator(config=SIM_CFG, initial_spawn_point=None,
                           log_path=None, skip_initial_scan=True)
occ = sim.occ
sx, sy = int(SPAWN_XY[0]), int(SPAWN_XY[1])
free = np.argwhere(occ == 0)
dist = np.hypot(free[:, 1] - sx, free[:, 0] - sy)

print("%8s %10s %12s %12s" % ("goal_px", "waypoints", "monolith_s", "cached_s"))
for target in [100, 300, 600, 1200, 2000]:
    i = int(np.argmin(np.abs(dist - target)))
    gy, gx = int(free[i, 0]), int(free[i, 1])

    t0 = time.time()
    p1 = astar_monolith(occ, (sx, sy), (gx, gy), clearance_m=2.0,
                        resolution=RESOLUTION)
    t_mono = time.time() - t0

    t0 = time.time()
    p2 = astar_cached(occ, (sx, sy), (gx, gy), clearance_m=2.0,
                      resolution=RESOLUTION)
    t_cache = time.time() - t0

    print("%8d %10s %12.1f %12.1f"
          % (target, len(p1) if p1 else None, t_mono, t_cache))