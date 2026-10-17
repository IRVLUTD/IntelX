"""
intelx.probes.tier1
===================
Measures what ONE Tier-1 planning call actually costs on a real map, and how
that cost scales with the number of frontiers.

Run this in the same env / cwd as run_sweep.py, with mapex_service.py already
running in the lama env. It never touches the simulator's motion loop, so it
isolates the service from everything else.

    python intelx.probes.tier1

Output per row:
    mode  explored_radius_px  n_frontiers  service_s  roundtrip_s  obs_MB

Read it like this:
  - service_s is the service's own timing of backend.plan()
  - roundtrip_s includes conversion + transport; the gap is your client cost
  - if service_s grows with n_frontiers, score_frontiers is the bottleneck
  - if the observed freeze in a real run is ~2x service_s, the 15 s PLANNING
    timeout in the monolith is issuing a duplicate call per goal
"""
import math
import time

import numpy as np

from intelx.baselines.tier1 import (
    MapExClient,
    occ_explored_to_obs_map,
    your_xy_to_their_rc,
    THEIR_RES,
)
from intelx.sim.simulator import ExplorationSimulator

from intelx.paths import map_path


# --- edit these ------------------------------------------------------------
MAP_PGM = (map_path("test/medium_office6.pgm"))
SPAWN_XY = (3726.0, 1838.0)          # seed 1000 spawn from spawns.json
HOST = "127.0.0.1"
PORT = 5555
TIMEOUT_S = 1800.0                   # generous: we are measuring, not running
MODES = ["visvarprob", "upen", "hectoraug"]
# EXPLORED_RADII_PX = [200, 400, 800, 1600]
EXPLORED_RADII_PX = [400, 900, 1600, 2600]
RESOLUTION = 0.05
PRED_RANGE_M = 20
PRED_BEAMS = 250
UPEN_CONFIG = {"rrt_max_iters": 2500, "expand_dis": 5,
               "goal_sample_rate": -1, "connect_circle_dist": 20,
               "rrt_num_path": 1000, "rrt_straight_line": False,
               "reach_horizon": 50, "upen_mode": "exploration"}
# ---------------------------------------------------------------------------

SIM_CFG = {
    "MAP_PGM": MAP_PGM,
    "RESOLUTION": RESOLUTION,
    "ROBOT_R": 0.35,
    "WIN_M": 20,
    "SPAWN_CLEARANCE": 0.45,
    "FOV": 2 * math.pi,
    "N_BEAMS": 2500,
    "MAX_R": int(20.0 / RESOLUTION),
    "STEP_SIZE_M": 0.1,
    "ANG_VEL": math.pi,
    "DT": 0.1,
    "WIN_PX": int(20 / RESOLUTION),
    "SAVE_DIR": "temp_probe",
    "SHOW_HEAD": True,
    "VISUALIZE_FOV": False,
    "RAD_CELLS": int(0.35 / RESOLUTION),
    "ZOOM_WINDOW_M": 40.0,
    "ZOOM_WINDOW_PX": int(40.0 / RESOLUTION),
}


def main():
    sim = ExplorationSimulator(config=SIM_CFG, initial_spawn_point=None,
                               log_path=None, skip_initial_scan=True)
    occ = sim.occ
    H, W = occ.shape
    print("map %s  occ %dx%d  free cells %d"
          % (MAP_PGM.split("/")[-1], W, H, int(np.count_nonzero(occ == 0))))

    sx, sy = SPAWN_XY
    yy, xx = np.ogrid[:H, :W]
    d2 = (xx - sx) ** 2 + (yy - sy) ** 2

    client = MapExClient(HOST, PORT, timeout=TIMEOUT_S)
    pong = client.ping()
    if not pong.get("ok"):
        print("service not reachable: %s" % pong.get("error"))
        return
    print("service ok, %s models\n" % pong.get("n_models", "?"))

    print("%-12s %8s %8s %10s %11s %8s"
          % ("mode", "radius", "n_front", "service_s", "roundtrip_s", "obs_MB"))

    for r in EXPLORED_RADII_PX:
        # explored = d2 <= float(r) * r
        explored = (np.abs(yy - sy) <= 250) & (np.abs(xx - sx) <= r)
        obs_map = occ_explored_to_obs_map(occ, explored, RESOLUTION, THEIR_RES)
        obs_mb = obs_map.nbytes / 1e6
        cur = your_xy_to_their_rc((sx, sy), RESOLUTION, THEIR_RES)

        for mode in MODES:
            t0 = time.time()
            rep = client.request({
                "mode": mode,
                "obs_map": obs_map,
                "cur_pose": cur,
                "pose_list": [cur],
                "cfg": {"pixel_per_meter": int(round(1.0 / THEIR_RES)),
                        "pred_laser_range_m": PRED_RANGE_M,
                        "pred_num_laser": PRED_BEAMS,
                        "upen_config": UPEN_CONFIG},
            }, retries=0)
            rt = time.time() - t0
            print("%-12s %8d %8s %10s %11.1f %8.1f"
                  % (mode, r, rep.get("n_frontiers", "?"),
                     ("%.1f" % rep["service_s"]) if rep.get("service_s") else "-",
                     rt, obs_mb))
            if not rep.get("ok"):
                print("    error: %s" % rep.get("error"))


if __name__ == "__main__":
    main()