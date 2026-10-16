"""
intelx.cli.run_sweep_all  --  ONE paired benchmark for every method
===========================================================
Replaces run_sweep.py (Tier-0 + ours) and run_sweep2.py (Tier-1). All nine
methods, one seed base, one results.csv, one spawns.json.

    Tier-0 (geometry only)  nearest, cost_utility, random, geom_ig
    Reference frontier      frontier
    Ours                    ours          (+ local_only ablation, off by default)
    Tier-1 (prediction)     mapex, upen, hectoraug   via mapex_service.py

WHY ONE FILE
    The two-file split is what let SEED_BASE drift apart: Tier-0 ran seeds
    1005..1014 and Tier-1 ran 1000..1004, so no (map, seed) ever had both tiers
    and the paired design the whole benchmark exists for did not hold across
    them. With one SEED_BASE that cannot happen again. Modes that cannot afford
    ten seeds take a PREFIX of the same sequence via MODE_RUNS, so their rows
    still pair with everyone else's.

WHAT PAIRING BUYS YOU
    For each (map, seed) we draw ONE spawn and run EVERY method from it. Pairing
    cancels per-spawn difficulty, so "does ours beat X" is far more powerful than
    independent runs, and a per-(map, seed) trajectory figure with all methods
    becomes possible.

BEFORE A TIER-1 RUN
    conda activate lama && python mapex_service.py --port 5555

RUN
    python -m intelx.cli.run_sweep_all             # uses SWEEP below
    python -m intelx.cli.run_sweep_all --selftest  # no sim/model/service needed
"""
import os
import csv
import math
import sys
import time
import threading
import numpy as np

from intelx import metrics

from intelx.paths import map_path, weights_path
from intelx.paths import RESULTS_ROOT as _RESULTS_ROOT


def _default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# Extra per-run columns describing the Tier-1 service calls; blank for every
# other mode. `tier1_fallbacks` is the one that matters for honesty: non-zero
# means some decisions were NOT made by the named planner, and that row must not
# be reported as that method.
TIER1_COLUMNS = ["tier1_calls", "tier1_service_s_mean", "tier1_frontiers_mean",
                 "tier1_fallbacks", "tier1_failures", "tier1_snapped",
                 "tier1_repeats"]


# How long a mode may show NO POSE CHANGE before the run is called stalled.
#
# A single window cannot serve both kinds of mode. Tier-1 needs a long one, or a
# slow planning call gets mistaken for a parked robot and the run is killed and
# censored at MAX_PATH_M. Tier-0 and ours decide in well under 30 s, so the same
# long window means a genuinely BROKEN mode burns half an hour per seed before
# anyone sees it: the runs with total_path_m = 0.0 each sat here for 1801 s.
#
# Override per run by setting STALL_IDLE_S_BY_MODE on SWEEP; anything not listed
# falls back to SWEEP.STALL_IDLE_S.
DEFAULT_STALL_IDLE_S_BY_MODE = {
    "nearest": 300.0, "cost_utility": 300.0, "random": 300.0, "geom_ig": 300.0,
    "frontier": 300.0, "ours": 300.0, "local_only": 300.0,
    "mapex": 1800.0, "upen": 1800.0, "hectoraug": 1800.0,
}


def stall_window_for(config, mode):
    """Stall window for one mode: the per-mode table wins, then SWEEP's default."""
    table = dict(DEFAULT_STALL_IDLE_S_BY_MODE)
    table.update(getattr(config, 'STALL_IDLE_S_BY_MODE', {}) or {})
    return float(table.get(mode, getattr(config, 'STALL_IDLE_S', 1800.0)))


# ===========================================================================
# CONFIG -- edit this block.
# ===========================================================================
class SWEEP:
    # =======================================================================
    # MODEL + SIMULATOR  (the explorers read these exact attribute names)
    # =======================================================================
    MODEL_CHECKPOINT = weights_path("checkpoint_best.pth")
    MODEL_TYPE = "resnet34"
    device = _default_device()
    output_dir = "sweep_out"

    ROBOT_RADIUS_METERS = 0.35
    RESOLUTION = 0.05
    WIN_M = 20
    LASER_SCAN_R = 10

    # Baseline sim config. The SENSOR fields here are overwritten per run from
    # SENSOR_PROFILES, so treat the values below as placeholders: what a run
    # actually used is recorded in the `sensor` column and the npz meta.
    sim = {
        "MAP_PGM": "",                      # overwritten per map by the sweep
        "RESOLUTION": RESOLUTION, "ROBOT_R": ROBOT_RADIUS_METERS, "WIN_M": WIN_M,
        "SPAWN_CLEARANCE": 0.45, "FOV": 220 * math.pi / 180, "N_BEAMS": 662,
        "MAX_R": int(LASER_SCAN_R / RESOLUTION),
        "STEP_SIZE_M": 0.1, "ANG_VEL": math.pi, "DT": 0.1,
        "WIN_PX": int(WIN_M / RESOLUTION),
        "SAVE_DIR": "temp_sweep", "SHOW_HEAD": True, "VISUALIZE_FOV": False,
        "RAD_CELLS": int(ROBOT_RADIUS_METERS / RESOLUTION),
        "ZOOM_WINDOW_M": 40.0,
        "ZOOM_WINDOW_PX": int(40.0 / RESOLUTION),
    }

    policy = {"exploration_target_pct": 99, "goal_history_length": 5}

    # --- baseline (frontier / Tier-0) knobs --------------------------------
    LOCAL_FRONTIER_WINDOW_M = 10.0
    GLOBAL_FRONTIER_WINDOW_M = 100.0
    MIN_FRONTIER_SIZE = 5
    POTENTIAL_SCALE = 3.0
    GAIN_SCALE = 1.0

    # --- intelx knobs ------------------------------------------------------
    LOCAL_ONLY_MODE = False
    LOCAL_ONLY_GOAL = "argmax"
    LOCAL_CENTROID_PERCENTILE = 0.0
    LOCAL_CENTROID_WEIGHT_POWER = 1.0
    SAVE_INTERVAL = 0

    # --- maps -------------------------------------------------------------
    MAPS = [
        # map_path("test_standardized/medium_office6.pgm"),
        # DUPLICATES OF TRAINING MAPS -- byte-identical, do not re-enable:
        #   0510034869_A-40 == medium_office3, 0510032217_A-40 == large_office1
        # map_path("test_standardized/small_office3.pgm"),
        # map_path("test_standardized/large_office7.pgm"),
        # map_path("ood_test_standardized/house_expo/large/houseexpo_large_3.pgm"),
        # map_path("ood_test_standardized/house_expo/large/new/houseexpo_large_3.pgm"),
        # map_path("ood_test_standardized/msd/large/msd_large_1.pgm"),
        # map_path("ood_test_standardized/aws_maps/hospital.pgm"),
        # map_path("train_standardized/atrium1.pgm"),
        # map_path("train_standardized/corridor1.pgm"),
        # map_path("train_standardized/small_lounge1.pgm"),
        # map_path("train_standardized/small_lounge2.pgm"),
        # map_path("train_standardized/small_office1.pgm"),
        # map_path("train_standardized/small_office2.pgm"),
        # map_path("train_standardized/small_house_aws.pgm"),
        # map_path("train_standardized/medium_office1.pgm"),
        # map_path("train_standardized/medium_office2.pgm"),
        # map_path("train_standardized/medium_office3.pgm"),
        # map_path("train_standardized/medium_office4.pgm"),
        # map_path("train_standardized/medium_office5.pgm"),
        # map_path("train_standardized/large_office1.pgm"),
        # map_path("train_standardized/large_office2.pgm"),
        # map_path("train_standardized/large_office3.pgm"),
        # map_path("train_standardized/large_office4.pgm"),
        # map_path("train_standardized/large_office5.pgm"),
        # map_path("train_standardized/large_office6.pgm"),
        # map_path("msd_final/small/msd_small_1.pgm"),
        # map_path("msd_final/small/msd_small_2.pgm"),
        # map_path("msd_final/small/msd_small_3.pgm"),
        # map_path("msd_final/medium/msd_medium_1.pgm"),
        # map_path("msd_final/medium/msd_medium_2.pgm"),
        # map_path("msd_final/medium/msd_medium_3.pgm"),
        # map_path("msd_final/large/msd_large_2.pgm"),
        # map_path("msd_final/large/msd_large_3.pgm"),
        # map_path("ood_test_standardized/house_expo/small/houseexpo_small_1.pgm"),
        # map_path("ood_test_standardized/house_expo/small/houseexpo_small_2.pgm"),
        # map_path("ood_test_standardized/house_expo/small/houseexpo_small_3.pgm"),
        # map_path("ood_test_standardized/house_expo/medium/houseexpo_medium_1.pgm"),
        # map_path("ood_test_standardized/house_expo/medium/houseexpo_medium_2.pgm"),
        # map_path("ood_test_standardized/house_expo/medium/houseexpo_medium_3.pgm"),
        # map_path("ood_test_standardized/house_expo/large/houseexpo_large_1.pgm"),
        # map_path("ood_test_standardized/house_expo/large/houseexpo_large_2.pgm"),
        map_path("real/real_slam_corridor.pgm"),
        # map_path("ood_test_standardized/house_expo/large/houseexpo_large_1.pgm"),
        # map_path("ood_test_standardized/house_expo/medium/houseexpo_medium_1.pgm"),
        # map_path("ood_test_standardized/house_expo/small/houseexpo_small_2.pgm"),
        # map_path("ood_test_standardized/msd/medium/msd_medium_2.pgm"),
        # map_path("ood_test_standardized/msd/medium/msd_medium_3.pgm"),
        # map_path("ood_test_standardized/msd/large/msd_large_2.pgm")

    ]

    # --- how many spawns (runs) per map -----------------------------------
    # SEED_BASE + k is the seed for run k, and spawns.json makes seed k the same
    # spawn for every mode and every script, forever. ONE base for all methods:
    # this is what makes the design paired.
    #
    # Pick the base to match whichever spawns you already have in spawns.json:
    # the old Tier-0 file used 1005..1014, the old Tier-1 file used 1000..1004.
    # Whatever you choose, it now applies to EVERY method, which is the point.
    SEED_BASE = 1001
    RUNS_PER_MAP = 1
    MAP_RUNS = {}                          # {"corridor1": 6}  per-map override

    # A mode that cannot afford the full RUNS_PER_MAP takes a PREFIX of the same
    # seed sequence, never a different one. With 5 here, mapex runs 1005..1009,
    # which is a strict subset of everyone else's 1005..1014, so every mapex row
    # pairs and the extra five seeds are simply unpaired for that comparison.
    # Tier-1 is ~10 s of planning per decision, so budget accordingly. Empty
    # means every method runs all RUNS_PER_MAP seeds, which is the fully paired
    # case and what you want if you can afford it.
    MODE_RUNS = {
        # "mapex": 5,
        # "upen": 5,
        # "hectoraug": 5,
    }

    # Map categories -> the results table groups rows by these. Recorded per row
    # so any grouping can be done later without re-running.
    MAP_CATEGORY = {
        # "small_house_aws": "small",
        # "corridor1":       "corridor",
    }

    # --- which policies to compare ---------------------------------------
    MODES = [
        "nearest",
        "cost_utility", "random", "geom_ig",
             "frontier",
             "ours",
        #      "mapex",
        #      "upen",
        #      "hectoraug"
        # "local_only"
             ]


    # add "local_only" for the pure-coverage ablation; it is model-driven and
    # deliberately stalls once the field flattens, which is used as a figure.

    # --- SENSOR PROFILES --------------------------------------------------
    # A method's performance depends heavily on what the robot can SEE, so the
    # sensor is a controlled variable, not a fixed one. Every profile runs in
    # THIS simulator on THESE maps with THESE spawns and metrics; only the
    # sensing changes. That isolates "is the planner better" from "does it have
    # a better lidar", which running the other authors' simulator could not.
    #
    #   intelx : our sensor (the one our coverage U-Net was trained for)
    #   mapex  : MapEx's sensor, so their method is also evaluated in the regime
    #            its predictor was trained for
    #
    # BOTH must stay defined here. apply_sensor_profile silently returns
    # config.sim unchanged for an unknown profile name, so a mode listed against
    # a missing profile would be LABELLED with it while using whatever sensor
    # happened to be loaded. preflight() makes that fatal.
    SENSOR_PROFILES = {
        "intelx": {"FOV": 220 * math.pi / 180, "N_BEAMS": 662,
                   "MAX_R": int(10.0 / 0.05)},
        "mapex":  {"FOV": 2 * math.pi, "N_BEAMS": 2500,
                   "MAX_R": int(20.0 / 0.05)},
    }
    DEFAULT_SENSOR = "intelx"

    # Which sensor profile(s) each mode runs under. A mode listed with two
    # profiles runs twice and appears as two rows; the `sensor` column
    # distinguishes them.
    #
    # As configured below the Tier-1 methods run ONLY under their own 20 m /
    # 360 deg sensing, which is more sensing than ours gets. That favours them
    # and must be stated, not hidden. Add "intelx" to their lists for the
    # controlled head-to-head; the gap between a method's two rows then
    # quantifies how much of its result is the sensor rather than the planner.
    MODE_SENSORS = {
        "nearest":      ["intelx"],
        "cost_utility": ["intelx"],
        "random":       ["intelx"],
        "geom_ig":      ["intelx"],
        "frontier":     ["intelx"],
        "ours":         ["intelx"],
        "local_only":   ["intelx"],
        "mapex":        [           # "intelx",
                         "mapex"],
        "upen":         [           # "intelx",
                         "mapex"],
        "hectoraug":    [           # "intelx",
                         "mapex"],
    }

    # Restrict a mode to specific maps (by map NAME, no extension). Any mode not
    # listed runs on every map in MAPS.
    MODE_MAPS = {
        # "local_only": ["small_office1"],
    }

    # --- Tier-1 service ----------------------------------------------------
    MAPEX_SERVICE_HOST = "127.0.0.1"
    # One or more service instances. Each sweep worker is pinned to one port
    # (round robin by worker index) and keeps a persistent socket to it.
    #
    # WHY MORE THAN ONE. mapex_service serialises plan() behind a single lock, so
    # ONE service can only ever have ONE decision in flight. A decision costs
    # ~10 s of service, then 2-20 s of A* and several seconds of driving that the
    # service sits out. So one service both caps you at one concurrent request
    # AND idles ~60% of the time. More services is the only way to raise that
    # ceiling.
    #
    # SIZING. In single-phase mode (PHASE_SPLIT_TIER1 = False) up to N_WORKERS
    # tasks are live at once and most of them will be inside a Tier-1 mode most
    # of the time, because Tier-1 dominates the runtime. So aim for
    #     len(PORTS) >= N_WORKERS / TIER1_WORKERS_PER_SERVICE
    # With N_WORKERS = 10 and 2 per service that is 5 services. preflight()
    # checks this and warns if you are oversubscribed.
    #
    # MEMORY. Each instance loads 4 LaMa models. On a 48 GB card 5 instances is
    # comfortable; watch nvidia-smi on the first run and add or remove ports.
    # Note the sweep workers also put the coverage U-Net on the same GPU, one
    # copy per worker, for the `ours` mode.
    #
    # Start them in the lama env, one terminal each or backgrounded:
    #     for p in 5555 5556 5557 5558 5559; do
    #         python mapex_service.py --port $p --device cuda:0 &
    #     done
    # preflight() prints the exact command for any port it cannot reach.
    MAPEX_SERVICE_PORTS = []#5557,]# 5558, 5559]
    MAPEX_SERVICE_PORT = 5555              # back-compat; ignored if PORTS is set
    MAPEX_SERVICE_TIMEOUT = 600.0
    MAPEX_PRED_RANGE_M = 20                # their pred_vis_configs
    MAPEX_PRED_BEAMS = 250
    # reach_horizon and upen_mode are REQUIRED by their get_rrt_goal. Without
    # them the call raises TypeError, the service falls back to the nearest
    # frontier, and the UPEN rows become a nearest-frontier planner wearing a
    # UPEN label. Values are theirs: reach_horizon from MapEx configs/base.yaml,
    # upen_mode hardcoded at their explore.py call site.
    MAPEX_UPEN_CONFIG = {"rrt_max_iters": 2500, "expand_dis": 5,
                         "goal_sample_rate": -1, "connect_circle_dist": 20,
                         "rrt_num_path": 1000, "rrt_straight_line": False,
                         "reach_horizon": 50,
                         "upen_mode": "exploration",
                         # "autoscale": True
                         }
    # How far to look for a simulator-acceptable cell when their frontier centre
    # lands in a wall (40 px = 2 m). See intelx.baselines.tier1.free_cell_candidates.
    TIER1_SNAP_PX = 40

    # --- metrics / termination -------------------------------------------
    # Checkpoints must all be <= TARGET_PCT, otherwise the run terminates before
    # the checkpoint can ever be reached and that column is censored for everyone.
    COVERAGE_CHECKPOINTS = [90.0, 95.0, 99.0]
    TARGET_PCT = 99.0                      # run terminates at/above this coverage
    # NOT the stopping criterion. Runs terminate on COVERAGE (target reached),
    # TIME (MAX_WALL_S) or STALL. MAX_PATH_M is a last-resort safety cap, and the
    # reference length used to normalise AUC and to censor path_at_* when a
    # checkpoint is never reached. Keep it generous so it effectively never
    # fires; a 'budget' stop_reason means it was too low.
    MAX_PATH_M = 5000.0
    # Wall-clock cap per run. A SAFETY NET, not the experimental budget. If a
    # healthy run trips it, its path metrics are censored at MAX_PATH_M even
    # though the robot never spent that path, which understates the method.
    # MEASURED on 50010536_PLAN4 (4399x4399): a Tier-1 decision costs ~10 s of
    # service (visvarprob/hectoraug; ~3.6 s upen) plus 2-20 s of A*, and the
    # robot drives at 0.75 m/s. A full 99% Tier-1 run is hours.
    MAX_WALL_S = 40000.0

    # --- stall detection ---------------------------------------------------
    # STALL_IDLE_S is measured against POSE CHANGE ONLY, so a robot legitimately
    # waiting on a slow planning call looks identical to a parked one. It must
    # stay well above the worst per-decision cost; at 200 s a single Tier-1
    # decision killed runs and censored their path metrics at MAX_PATH_M.
    # Fallback for any mode not in STALL_IDLE_S_BY_MODE (module level, above).
    STALL_IDLE_S = 1800.0                  # no position change at all for this long
    STALL_NO_PROGRESS_GOALS = 50           # goals planned with no coverage gain
    COV_PROGRESS_EPS = 0.001               # % coverage that counts as "progress"
    NO_PROGRESS_MAX_COV = 97.0             # above this, never call it "no progress"
                                           # (tail scrap-chasing is slow by nature)
    POLL_S = 0.25                          # how often to sample (coverage, path)
    SPAWN_CLEARANCE_M = 0.45               # spawn must be this clear of walls

    # --- rich per-decision recording --------------------------------------
    # Heatmaps, network inputs, local components and the junction graph, for
    # runs you will actually visualise. EMPTY BY DEFAULT: a 400x400 float32
    # field is 640 kB and a long run makes ~40,000 passes, so recording
    # everything would be tens of GB per run. Name the handful you need.
    #
    #   RECORD_RUNS = [("large_office7", 1004, "ours")]
    #   RECORD_RUNS = [("large_office7", 1004, None)]   # every mode, that seed
    # RECORD_RUNS = [("msd_large_2", 1000, "ours"), ("msd_medium_2", 1000, "ours"), ("msd_medium_3", 1000, "ours")]
    RECORD_RUNS = [
        # ("medium_office6", 1004, "ours"),
                   # ("small_office1", 1003, "local_only")
                   ]
    RECORD_EVERY_PX = 0.0                  # extra spatial thinning; 0 = every decision
    RECORD_IDLE_EVERY_PX = 10.0             # also record idle passes every N px;
                                           # 10 gives a smooth field for a demo video
    RECORD_HEATMAP = True                  # float16; the big one
    RECORD_INPUT = True                    # the image the network saw
    RECORD_LABELMAP = True                 # component masks

    PIN_CONFIG = {                         # forced onto the run config for ALL modes
        "TURBO_MODE": True,
        "FIXED_DT": 0.05,
        "MAP_UPDATE_INTERVAL": 5,
        "STATE_PUBLISH_EVERY": 10,
        # Preempt, early-stop and junction detection all run ONLY on idle
        # passes, so this sets how many chances per metre they get. It was 8.0
        # here against a code default of 1.0, which meant the sweep ran a
        # measurably different policy from every intelx.cli.run_intelx debug session.
        "IDLE_PREDICT_EVERY_PX": 1.0,
        "TURBO_SYNC_PRED": True,
        "PRED_MAX_LAG_PX": 12.0,
        "PRED_SYNC_MAX_WAIT_S": 0.5,
        "DEBUG_LOG": True,                 # False for the real sweep
        "DECISION_LOG": True,              # one csv row per nav decision
        "DECISION_LOG_ASTAR": True,        # adds astar_px/detour_ratio: one A*
                                           # per decision, so False for sweeps
        "PROFILE": False,
        "JUNCTION_DEBUG": False,
        "HEADLESS": True,
        # How long the sim's PLANNING state waits for a goal before re-requesting.
        # The hardcoded 15 s was far below the real Tier-1 decision cost, so the
        # sim timed out repeatedly per goal and each timeout queued a REDUNDANT
        # planning call behind the live one.
        "PREDICTION_TIMEOUT_S": 900,
        # --- intelx policy ---------------------------------------------------
        # ONLY what differs from the code default in intelx_policy11.py. The
        # other fifteen knobs that used to sit here were byte-identical to their
        # defaults and bought nothing but noise; EARLY_STOP_MARGIN_PX,
        # EARLY_STOP_MIN_DIST_PX and FRONTIER_CONSUMED_R_M are gone entirely
        # because early-stop now tests whether the frontier has been CONSUMED
        # rather than where it sits in the image.
        #
        # The original point of pinning was never the values -- it was that the
        # sweep and intelx.cli.run_intelx disagreed. Keep them agreeing.
        "PREEMPT_MIN_SCORE": 0.005,        # default 0.0005 is the noise floor
        "BRANCH_ORDER": "weakest",
        "LOCAL_FIELD": "unet",
        # --- baselines -------------------------------------------------------
        # False runs every baseline as its cited paper defines it. True restores
        # the behaviour the submitted results were produced with: the travel
        # cost counted in waypoints, the original Cost-Utility and Geometric IG
        # scores, the one-pass exhaustion (see intelx.baselines.simple), and A*
        # refusing to plan from a robot one pixel inside a wall
        # (intelx.common.START_SNAP_PX).
        "SUBMITTED_BEHAVIOUR": False,
    }

    # --- parallelisation --------------------------------------------------
    # One TASK = one (map, seed): that worker runs every mode for that spawn,
    # sequentially, so all methods in a pair see identical contention.
    # With PHASE_SPLIT_TIER1 the sweep runs non-Tier-1 modes in a pool first,
    # then Tier-1. The spawns are identical either way so pairing is unaffected;
    # only the "identical contention within a pair" property is, and that matters
    # solely for the timing columns, which are valid only at TIMING_RERUN anyway.
    #
    # EXECUTION controls how the two kinds of work are scheduled:
    #
    #   "concurrent" (default) TWO pools running AT THE SAME TIME: N_WORKERS on
    #                 Tier-0 + ours, and N_WORKERS_TIER1 on Tier-1. This is the
    #                 only setting that keeps the services busy from the first
    #                 second, and it bounds Tier-1 concurrency to what the
    #                 services can actually absorb.
    #   "split"       the two pools one after the other. Use it for a timing run
    #                 where Tier-1 should have the machine to itself.
    #   "single"      one pool over tasks that each run every mode. NOTE: a task
    #                 walks its modes in MODES order, so every worker finishes
    #                 all the Tier-0 modes before any of them reaches mapex, and
    #                 the services idle until then. Almost never what you want.
    #
    # TIER1_WORKERS_PER_SERVICE is the concurrency each service is assumed to
    # absorb. It sizes N_WORKERS_TIER1 in split mode and the preflight
    # oversubscription check in single-phase mode. 2 is a sensible start given
    # the ~40% service duty cycle; raise it until tier1_service_s_mean in
    # results.csv starts climbing, which is where calls queue rather than run.
    #
    # BEFORE YOU RAISE ANY OF THESE: the monolith must honour FIXED_DT. It
    # originally derived dt from the wall clock, so under load its step size
    # grows toward 1.5 px, scans go from every 3.75 px to every 7.5 px, and the
    # measured coverage curve changes. Without that patch more workers do not
    # just cost wall time, they change the numbers. preflight() checks and warns.
    EXECUTION = "concurrent"               # "concurrent" | "split" | "single"
    N_WORKERS = 40                         # workers on Tier-0 + ours
    TIER1_WORKERS_PER_SERVICE = 1
    N_WORKERS_TIER1 = None                 # None -> len(PORTS) * PER_SERVICE
    TIMING_RERUN = False                   # True -> forces 1 worker everywhere

    # --- per-run video ----------------------------------------------------
    # 'live' draws each run as intelx.viz.live does, the look of the paper's
    # videos; 'basic' is a plain preview, cheaper to render.
    SAVE_VIDEO = True
    VIDEO_STYLE = "live"
    # One video per map at the end of the sweep, every method side by side and
    # synchronised by distance travelled (intelx.viz.showcase).
    SHOWCASE_VIDEO = True
    VIDEO_EVERY_S = 0.5                    # seconds of wall time between frames
    VIDEO_FPS = 10.0
    VIDEO_MAX_PX = 700                     # longest side of the rendered frame

    # --- output layout ----------------------------------------------------
    # RESULTS_ROOT is intelx.paths.RESULTS_ROOT: <repo>/results by default,
    # redirected with INTELX_RESULTS_ROOT.
    #   <RESULTS_ROOT>/
    #       spawns.json              <- SHARED across runs, keep this file
    #       <RUN_NAME>/
    #           results.csv
    #           results_meta.json
    #           curves/              <- raw .npz per run (offline re-metrics)
    #           videos/              <- one mp4 per run
    #           tables/              <- written by make_tables.py
    RESULTS_ROOT = str(_RESULTS_ROOT)
    RUN_NAME = "real_corridor_intelx_tier0"                          # "" -> auto timestamp run_YYYYmmdd_HHMM
    SAVE_CURVES = True


def make_run_video(config, mode, sim, sim_cfg, map_name, seed, spawn, path,
                   show=False, fps=None):
    """The recorder for one run: the live view (VIDEO_STYLE 'live') or the
    plain preview ('basic')."""
    fps = float(fps or config.VIDEO_FPS)
    if str(getattr(config, 'VIDEO_STYLE', 'live')) != 'live' and not show:
        return RunVideo(path, sim.occ, sim_cfg.get('RESOLUTION', 0.05), fps,
                        config.VIDEO_MAX_PX, title=f"{map_name} s{seed} {mode}")
    from intelx.viz.live import LiveView, LivePlayer
    from intelx.viz.style import method_label, pretty_map_name
    view = LiveView(sim.occ, method=mode,
                    title="%s: %s" % (method_label(mode), pretty_map_name(map_name)),
                    spawn=spawn, fov=sim_cfg.get('FOV'), max_r_px=sim_cfg.get('MAX_R'),
                    win_px=sim_cfg.get('WIN_PX', 400),
                    target_pct=getattr(config, 'TARGET_PCT', 99.0))
    return LivePlayer(view, show=show, video_path=path, fps=fps)


# ===========================================================================
# PURE HELPERS (unit-tested; no sim/model imports)
# ===========================================================================
def pick_spawn(occ, seed, clearance_px):
    """Deterministic spawn: a free cell at least `clearance_px` from any wall,
    chosen by a seeded RNG. Returns (x, y, theta). Same seed -> same spawn."""
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(occ == 0)
    ys, xs = np.where(dist >= clearance_px)
    if len(xs) == 0:                       # fallback: any free cell
        ys, xs = np.where(occ == 0)
    rng = np.random.default_rng(seed)
    i = int(rng.integers(len(xs)))
    theta = float(rng.uniform(-np.pi, np.pi))
    return (float(xs[i]), float(ys[i]), theta)


def path_at_coverage(curve, checkpoint):
    """First path length (m) at which coverage (%) reaches `checkpoint`, linearly
    interpolated between samples. `curve` is a list of (path_m, cov_pct), assumed
    time-ordered (path non-decreasing). None if the checkpoint is never reached."""
    if not curve:
        return None
    prev_p, prev_c = curve[0]
    if prev_c >= checkpoint:
        return float(prev_p)
    for p, c in curve[1:]:
        if c >= checkpoint:
            if c == prev_c:
                return float(p)
            frac = (checkpoint - prev_c) / (c - prev_c)
            return float(prev_p + frac * (p - prev_p))
        prev_p, prev_c = p, c
    return None


def curve_to_metrics(curve, checkpoints, max_path_m):
    """Reduce a (path,cov) curve to a metrics dict: path@checkpoint (censored to
    max_path_m if never reached), final coverage, total path, reached flags."""
    m = {}
    final_cov = curve[-1][1] if curve else 0.0
    total_path = curve[-1][0] if curve else 0.0
    for cp in checkpoints:
        p = path_at_coverage(curve, cp)
        m[f"path_at_{int(cp)}"] = float(max_path_m) if p is None else p
        m[f"reached_{int(cp)}"] = int(p is not None)
    m["final_coverage"] = float(final_cov)
    m["total_path_m"] = float(total_path)
    return m


def write_row(csv_path, row, header_order):
    """Append one row; write header if the file is new. Flushed each call so a
    crash mid-sweep keeps everything already computed."""
    new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header_order)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def tier1_row_stats(explorer):
    """Summarise the Tier-1 service calls made during one run, for the CSV.
    Returns "" for every column when the explorer is not a Tier-1 one."""
    st = getattr(explorer, '_tier1_stats', None) if explorer is not None else None
    if not st or not st.get('calls'):
        return dict((c, "") for c in TIER1_COLUMNS)
    calls = int(st['calls'])
    return {
        "tier1_calls": calls,
        "tier1_service_s_mean": round(float(st.get('service_s_sum', 0.0)) / calls, 2),
        "tier1_frontiers_mean": round(float(st.get('n_frontiers_sum', 0)) / calls, 2),
        # decisions NOT made by the named planner -> the row is not that method
        "tier1_fallbacks": int(st.get('fallbacks', 0)),
        "tier1_failures": int(st.get('failures', 0)),
        # goals their planner put inside a wall, repaired client-side
        "tier1_snapped": int(st.get('snapped', 0)),
        # identical goal on consecutive calls: the sim rejected and re-requested
        "tier1_repeats": int(st.get('repeats', 0)),
    }


# ===========================================================================
# MONITORED RUN (works with any explorer exposing the standard interface)
# ===========================================================================
def monitor_run(explorer, checkpoints, max_path_m, max_wall_s, poll_s,
                video=None, video_every_s=0.5, stall_idle_s=None):
    """Start prediction + simulation threads (NO visualisation) and RECORD the
    raw run: coverage/path curve, pose trajectory, and goal (planning) events.

    Recording the raw arrays -- not just summary numbers -- means any metric we
    think of later can be computed offline from the stored .npz WITHOUT
    re-running. Returns (curve, traj, goals, stop_reason) where
        curve (N,3): wall_s, path_m, coverage_pct
        traj  (N,4): wall_s, x_px, y_px, theta_rad
        goals (M,3): wall_s, path_m, goal_index
    """
    explorer.running.set()
    pt = threading.Thread(target=explorer._prediction_worker, daemon=True)
    st = threading.Thread(target=explorer._updated_simulation_worker, daemon=True)
    pt.start(); st.start()

    curve, traj, goals = [], [], []
    # goal_count starts at 0, so seeding this at -1 made the FIRST poll record a
    # goal event that never happened: every run's goal count was one too high and
    # a run that planned nothing still reported 1.
    last_goal = 0
    last_frame = -1e9
    t0 = time.time()
    stop = "complete"
    idle_s = float(SWEEP.STALL_IDLE_S if stall_idle_s is None else stall_idle_s)
    no_prog_goals = int(getattr(SWEEP, 'STALL_NO_PROGRESS_GOALS', 0) or 0)
    cov_eps = float(getattr(SWEEP, 'COV_PROGRESS_EPS', 0.01))
    prev_xy = None
    last_move_t = 0.0
    best_cov = -1.0
    goals_at_best_cov = 0
    try:
        while True:
            time.sleep(poll_s)
            now = time.time() - t0
            # Read the flag BEFORE the state. The simulator publishes its
            # coverage and only then sets the flag, so a flag read first is
            # always matched by the coverage that set it. Read after, as it
            # was, a run could finish while this poll drew its video frame and
            # be labelled with the coverage from before: a run that reached
            # 99% recorded as exhausted at 98.97%, with P99 never reached.
            done = explorer.exploration_complete.is_set()
            with explorer.state_lock:
                cov = float(explorer.shared_state.get('exploration_pct', 0.0))
                path_m = float(explorer.shared_state.get('total_path_length', 0.0))
                pose = explorer.shared_state.get('pose', None)
                gcount = int(explorer.shared_state.get('goal_count', 0))
            curve.append((now, path_m, cov))
            if pose is not None and len(pose) >= 3:
                traj.append((now, float(pose[0]), float(pose[1]), float(pose[2])))
            if gcount != last_goal:                    # a new goal was planned
                goals.append((now, path_m, float(gcount)))
                last_goal = gcount
            if video is not None and (now - last_frame) >= video_every_s:
                last_frame = now
                try:
                    if hasattr(video, 'update'):        # intelx.viz.live.LivePlayer
                        video.update(explorer, now)
                    else:                               # RunVideo
                        with explorer.state_lock:
                            expl = explorer.shared_state.get('explored', None)
                            goal_w = explorer.shared_state.get('current_goal', None)
                        if expl is not None:
                            video.frame(expl, pose, path_m, cov, goal_w)
                except Exception:
                    pass
                if getattr(video, 'stop_requested', False):
                    stop = "user"; break               # Esc in the live window
            # --- stall checks -------------------------------------------
            if pose is not None and len(pose) >= 2:
                xy = (float(pose[0]), float(pose[1]))
                if prev_xy is None or (abs(xy[0] - prev_xy[0]) > 1e-9 or
                                       abs(xy[1] - prev_xy[1]) > 1e-9):
                    last_move_t = now          # the robot actually moved
                prev_xy = xy
            if cov > best_cov + cov_eps:
                best_cov = cov                 # coverage genuinely improved
                goals_at_best_cov = gcount

            if done:
                # The simulator sets this flag BOTH when the coverage target is
                # reached AND when the policy runs out of frontiers. Those are
                # very different outcomes, so label them differently: otherwise a
                # run that gave up at 68% is indistinguishable from one that
                # finished, and completion rates look better than they are.
                stop = ("complete" if cov >= float(getattr(SWEEP, 'TARGET_PCT', 99.0))
                        else "exhausted")
                break
            if path_m >= max_path_m:
                stop = "budget"; break
            if idle_s > 0 and (now - last_move_t) >= idle_s:
                stop = "stalled"; break        # parked for STALL_IDLE_S
            # Do NOT trip no-progress once essentially finished: in the last ~1%
            # the robot chases scraps and a single goal can legitimately add
            # < COV_PROGRESS_EPS, which killed otherwise-successful runs.
            if (no_prog_goals > 0 and (gcount - goals_at_best_cov) > no_prog_goals
                    and cov < float(getattr(SWEEP, 'NO_PROGRESS_MAX_COV', 97.0))):
                stop = "no_progress"; break    # planning but gaining nothing
            if time.time() - t0 >= max_wall_s:
                stop = "timeout"; break
    finally:
        explorer.running.clear()
        explorer.exploration_complete.set()
        pt.join(timeout=3.0); st.join(timeout=3.0)
        if video is not None and hasattr(video, 'finish'):
            try:
                video.finish(explorer, stop)
            except Exception:
                pass
    return curve, traj, goals, stop


def map_free_area_m2(occ, resolution):
    """Free (traversable) area of the map in square metres, used to normalise
    path length across maps of very different size."""
    return float(np.count_nonzero(occ == 0) * resolution * resolution)


# ===========================================================================
# SIM / MODEL / EXPLORER BUILDERS  (import the user's modules lazily)
# ===========================================================================
def build_simulator(map_path, config, sim_cfg=None):
    from intelx.sim.simulator import ExplorationSimulator
    cfg = dict(sim_cfg if sim_cfg is not None else config.sim)
    cfg['MAP_PGM'] = map_path
    return ExplorationSimulator(config=cfg, initial_spawn_point=None,
                                log_path=None, skip_initial_scan=True)


# Which module provides the frontier monolith that `frontier`, all Tier-0 mixins
# and all Tier-1 mixins sit on. intelx.baselines.frontier is THE published
# baseline; it is named in the meta json so the paper can state exactly what was
# run.
_RESOLVED = {}

FRONTIER_MODULE_ORDER = ("intelx.baselines.frontier",)


def _base_monolith():
    """The frontier monolith the Tier-0 and Tier-1 mixins sit on."""
    for mod in FRONTIER_MODULE_ORDER:
        try:
            m = __import__(mod, fromlist=["UpdatedGoalBasedAutonomousExplorer"])
            cls = getattr(m, "UpdatedGoalBasedAutonomousExplorer")
            _RESOLVED["frontier_module"] = mod
            return cls
        except Exception:
            continue
    raise ImportError("Could not import the frontier monolith "
                      "(tried %s)." % ", ".join(FRONTIER_MODULE_ORDER))


def _ours_class():
    """The real intelx explorer -- the same class intelx.cli.run_intelx uses, so the sweep
    measures exactly what you ship."""
    from intelx.explorer import UpdatedGoalBasedAutonomousExplorer as IntelX
    return IntelX


def _tier0(mode):
    """Geometry-only baselines: a _prediction_worker mixin over the monolith."""
    from intelx.baselines.simple import BASELINE_MIXINS
    mixin = BASELINE_MIXINS[mode]
    return type(f"Explorer_{mode}", (mixin, _base_monolith()), {})


def _tier1(mode):
    """Tier-1 (MapEx family). The planning decision is made by the AUTHORS' code
    running in their `lama` env behind mapex_service.py; we only convert formats
    and coordinates, so the baseline cannot be accidentally weakened."""
    from intelx.baselines.tier1 import TIER1_MIXINS
    return type("Explorer_%s" % mode, (TIER1_MIXINS[mode], _base_monolith()), {})


MODE_REGISTRY = {
    "nearest":      lambda: _tier0("nearest"),
    "cost_utility": lambda: _tier0("cost_utility"),
    "random":       lambda: _tier0("random"),
    "geom_ig":      lambda: _tier0("geom_ig"),
    "frontier":     lambda: _base_monolith(),
    "ours":         _ours_class,
    "local_only":   _ours_class,          # same class; LOCAL_ONLY_MODE per-run
    # --- Tier-1: require mapex_service.py running in the lama env ------------
    "mapex":        lambda: _tier1("mapex"),        # visvarprob = MapEx ICRA'25
    "upen":         lambda: _tier1("upen"),         # Georgakis et al. ICRA'22
    "hectoraug":    lambda: _tier1("hectoraug"),    # Shrestha et al. ICRA'19
}

TIER0_MODES = {"nearest", "cost_utility", "random", "geom_ig"}
TIER1_MODES = {"mapex", "upen", "hectoraug"}
NEEDS_MODEL = {"ours", "local_only"}       # need the coverage-prediction U-Net

# per-mode config overrides applied on top of PIN_CONFIG
MODE_CONFIG = {
    "local_only": {"LOCAL_ONLY_MODE": True},
}


def _explorer_class_for_mode(mode):
    """Return an explorer CLASS whose _prediction_worker is `mode`'s policy."""
    if mode not in MODE_REGISTRY:
        raise ValueError(f"unknown mode: {mode} (known: {sorted(MODE_REGISTRY)})")
    return MODE_REGISTRY[mode]()


def build_explorer(mode, sim, model, config, seed):
    config.BASELINE_SEED = seed            # the random baseline reads this
    # per-mode overrides (e.g. the local_only ablation); reset first so modes
    # cannot leak settings into one another within a sweep
    for k in set().union(*[set(v) for v in MODE_CONFIG.values()]) if MODE_CONFIG else set():
        if hasattr(config, k):
            try:
                delattr(config, k)
            except Exception:
                setattr(config, k, False)
    for k, v in MODE_CONFIG.get(mode, {}).items():
        setattr(config, k, v)
    cls = _explorer_class_for_mode(mode)
    exp = cls(sim, model, config)
    exp.headless = True                    # never open a window during a sweep
    exp.save_interval = 0
    return exp


def force_spawn(explorer, sim, spawn):
    """Put the robot at the canonical spawn and take the initial scan, so every
    mode starts from an identical state for this (map, seed)."""
    x, y, th = spawn
    sim.pose = np.array([x, y, th], dtype=float)
    try:
        scan = sim._simulate_scan(sim.pose)
        sim._update_exploration(sim.pose, scan)
    except Exception as e:
        print(f"  [spawn] initial scan failed: {e}")
    explorer.simulator = sim


# ===========================================================================
# SPAWN MANIFEST -- spawns are recorded once and replayed forever, so a baseline
# added months later starts from EXACTLY the same poses and its rows pair with
# the existing ones.
# ===========================================================================
def load_spawn_manifest(path):
    if path and os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    return {}


def save_spawn_manifest(path, manifest):
    if not path:
        return
    import json
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def get_spawn(manifest, map_name, seed, occ, clearance_px):
    """Authoritative spawn for (map, seed): the manifest wins if it has one,
    otherwise compute deterministically and record it."""
    key = f"{map_name}:{seed}"
    if key in manifest:
        v = manifest[key]
        return (float(v[0]), float(v[1]), float(v[2])), False
    sp = pick_spawn(occ, seed, clearance_px)
    manifest[key] = [float(sp[0]), float(sp[1]), float(sp[2])]
    return sp, True


# ===========================================================================
# LIGHTWEIGHT PER-RUN VIDEO
# The explorer's own 3-panel dashboard is far too expensive to run thousands of
# times (it was a major GIL contender). This renders a single downscaled global
# view every VIDEO_EVERY_S.
# NOTE: poses come from shared_state at that interval, so the drawn trajectory
# cuts corners and can appear to cross walls. It is a preview, not the
# trajectory; explorer.complete_trajectory is the dense one.
# ===========================================================================
class RunVideo:
    def __init__(self, path, occ, resolution, fps, max_px, title=""):
        import cv2
        self.cv2 = cv2
        H, W = occ.shape
        self.scale = min(1.0, float(max_px) / max(H, W))
        self.w, self.h = int(W * self.scale), int(H * self.scale)
        base = np.full((H, W, 3), 60, np.uint8)
        base[occ == 0] = (110, 110, 110)
        base[occ == 1] = (25, 25, 25)
        self.base = cv2.resize(base, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.title = title
        self.path = path
        self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(fps), (self.w, self.h))
        self.ok = self.writer.isOpened()
        self.pts = []

    def frame(self, explored, pose, path_m, cov, goal=None):
        if not self.ok:
            return
        cv2 = self.cv2
        img = self.base.copy()
        ex = cv2.resize(explored.astype(np.uint8) * 255, (self.w, self.h),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        img[ex] = (0.55 * img[ex] + np.array([115, 60, 0]) * 0.45).astype(np.uint8)
        if pose is not None:
            self.pts.append((int(pose[0] * self.scale), int(pose[1] * self.scale)))
            if len(self.pts) > 1:
                cv2.polylines(img, [np.array(self.pts, np.int32)], False, (0, 230, 255), 1)
            cv2.circle(img, self.pts[-1], 4, (0, 255, 0), -1)
        if goal is not None:
            cv2.drawMarker(img, (int(goal[0] * self.scale), int(goal[1] * self.scale)),
                           (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.rectangle(img, (0, 0), (self.w, 24), (20, 20, 20), -1)
        cv2.putText(img, f"{self.title}  {path_m:6.1f}m  {cov:5.1f}%", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        self.writer.write(img)

    def close(self):
        try:
            if self.ok:
                self.writer.release()
        except Exception:
            pass


# ===========================================================================
# WORKER: one (map, seed) -> its modes, sequentially. Module-level so it can be
# pickled by multiprocessing.
# ===========================================================================
_WORKER_MODEL = {"m": None}


def _worker_model(config):
    if _WORKER_MODEL["m"] is None:
        from intelx.common import create_model_for_inference
        _WORKER_MODEL["m"] = create_model_for_inference(
            config.MODEL_CHECKPOINT, config.device, model_type=config.MODEL_TYPE)
    return _WORKER_MODEL["m"]


def run_task(args):
    """(map_path, seed, spawn, modes, paths) -> result rows. Runs in a worker.

    NOTE: with multiprocessing's 'spawn' start method the child re-imports this
    module, so anything the PARENT computed on SWEEP (pinned config, resolved
    paths) is absent here. Both are re-applied / passed in explicitly."""
    map_path, seed, spawn, modes, paths = args
    config = SWEEP
    _pin_config(config)                      # must be re-applied in the child
    map_name = os.path.splitext(os.path.basename(map_path))[0]
    res = config.sim.get('RESOLUTION', 0.05)
    rows = []

    pairs = _mode_sensor_pairs(config, modes)
    model = _worker_model(config) if any(m in NEEDS_MODEL for m, _ in pairs) else None
    if any(m in TIER1_MODES for m, _ in pairs):
        _port = _pin_service_port(config)
        print(f"  [worker {_worker_index()}] Tier-1 service port {_port}")

    for mode, sensor in pairs:
        t0 = time.time()
        vid = None
        exp = None
        # [DBGLOG] One log file per run. The policy defaults to
        # <output_dir>/prediction_debug.log, which is the SAME path for every
        # run and every worker process -- with N_WORKERS > 1 they all append to
        # it and interleave into nonsense. Only computed when DEBUG_LOG is on,
        # so a normal sweep pays nothing.
        if getattr(config, 'DEBUG_LOG', False):
            try:
                _ld = os.path.join(paths["run_dir"], "logs")
                os.makedirs(_ld, exist_ok=True)
                config.DEBUG_LOG_PATH = os.path.join(
                    _ld, "%s_s%d_%s_%s.log" % (map_name, seed, mode, sensor))
            except Exception as _e:
                print("  [dbglog] could not set per-run log path: %s" % _e)
        # [RECORD] set only for runs named in RECORD_RUNS, so every other run
        # in the sweep writes nothing and pays nothing.
        config.RECORD_DIR = None
        for _m, _s, _md in getattr(config, 'RECORD_RUNS', []) or []:
            if _m == map_name and int(_s) == int(seed) and (_md in (None, mode)):
                config.RECORD_DIR = os.path.join(
                    paths["run_dir"], "records",
                    "%s_s%d_%s_%s" % (map_name, seed, mode, sensor))
                break

        # the explorer reads config.sim, so point it at this run's sensor profile
        sim_cfg = apply_sensor_profile(config, sensor)
        base_sim = config.sim
        config.sim = sim_cfg
        try:
            sim = build_simulator(map_path, config, sim_cfg)
            free_area_m2 = map_free_area_m2(sim.occ, res)
            exp = build_explorer(mode, sim, model, config, seed)
            force_spawn(exp, sim, spawn)
            if getattr(config, 'SAVE_VIDEO', False):
                os.makedirs(paths["videos"], exist_ok=True)
                vid = make_run_video(
                    config, mode, sim, sim_cfg, map_name, seed, spawn,
                    os.path.join(paths["videos"], f"{map_name}_s{seed}_{mode}_{sensor}.mp4"))
            _idle = stall_window_for(config, mode)
            curve, traj, gevents, stop = monitor_run(
                exp, config.COVERAGE_CHECKPOINTS, config.MAX_PATH_M,
                config.MAX_WALL_S, config.POLL_S, video=vid,
                video_every_s=getattr(config, 'VIDEO_EVERY_S', 0.5),
                stall_idle_s=_idle)
            wall_s = time.time() - t0
            met = metrics.compute_all(curve, traj, gevents, {
                'path_budget_m': config.MAX_PATH_M, 'free_area_m2': free_area_m2,
                'target_pct': config.TARGET_PCT, 'wall_s': wall_s,
                'stop_reason': stop, 'resolution': res,
                'checkpoints': config.COVERAGE_CHECKPOINTS})
            goals = len(gevents)
        except Exception as e:
            import traceback
            traceback.print_exc()
            curve = traj = gevents = []
            goals = 0
            wall_s = time.time() - t0
            free_area_m2 = 0.0
            met = metrics.compute_all([], [], [], {
                'path_budget_m': config.MAX_PATH_M, 'free_area_m2': 0.0,
                'target_pct': config.TARGET_PCT, 'wall_s': wall_s,
                'stop_reason': 'error', 'resolution': res,
                'checkpoints': config.COVERAGE_CHECKPOINTS})
        finally:
            if vid is not None:
                vid.close()
            config.sim = base_sim               # restore for the next pair

        # [RECORD] a manifest beside the pass files, so a visualiser does not
        # have to re-derive what the run was from the directory name
        if getattr(config, 'RECORD_DIR', None):
            try:
                import json as _json
                os.makedirs(config.RECORD_DIR, exist_ok=True)
                with open(os.path.join(config.RECORD_DIR, "meta.json"), "w") as _f:
                    _json.dump({"map": map_name, "map_path": map_path,
                                "seed": seed, "mode": mode, "sensor": sensor,
                                "spawn": list(spawn), "resolution": res,
                                "win_px": config.sim.get('WIN_PX', 400),
                                "sensor_profile": dict(getattr(
                                    config, 'SENSOR_PROFILES', {}).get(sensor, {})),
                                "stop_reason": met.get('stop_reason', ''),
                                "final_coverage": met.get('final_coverage'),
                                "total_path_m": met.get('total_path_m'),
                                "pinned": dict(getattr(config, 'PIN_CONFIG', {}))},
                               _f, indent=2)
            except Exception as _e:
                print("  [record] could not write meta.json: %s" % _e)

        t1stats = tier1_row_stats(exp)

        row = {"map": map_name,
               "category": getattr(config, 'MAP_CATEGORY', {}).get(map_name, ""),
               "seed": seed, "mode": mode, "sensor": sensor,
               "spawn_x": round(spawn[0], 2), "spawn_y": round(spawn[1], 2),
               "spawn_theta": round(spawn[2], 4), "goals": goals}
        row.update(t1stats)
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()})
        rows.append(row)

        # RAW recording + per-run metadata -> any future metric, no re-running
        if config.SAVE_CURVES and curve:
            import json
            os.makedirs(paths["curves"], exist_ok=True)
            np.savez_compressed(
                os.path.join(paths["curves"], f"{map_name}_s{seed}_{mode}_{sensor}.npz"),
                curve=np.array(curve, dtype=np.float32),
                traj=np.array(traj, dtype=np.float32),
                goals=np.array(gevents, dtype=np.float32),
                meta=np.array(json.dumps({
                    "map": map_name, "map_path": map_path, "seed": seed, "mode": mode,
                    "sensor": sensor, "sensor_profile": dict(
                        getattr(config, 'SENSOR_PROFILES', {}).get(sensor, {})),
                    "spawn": list(spawn), "resolution": res,
                    "free_area_m2": free_area_m2, "path_budget_m": config.MAX_PATH_M,
                    "target_pct": config.TARGET_PCT, "poll_s": config.POLL_S,
                    "wall_s": wall_s, "stop_reason": met.get('stop_reason', ''),
                    "tier1": t1stats,
                    "pinned": dict(getattr(config, 'PIN_CONFIG', {})),
                })))
        _fb = t1stats.get("tier1_fallbacks", "")
        print(f"  [{map_name} s{seed} {mode}@{sensor}] {met.get('stop_reason')} "
              f"cov={met['final_coverage']:.1f}% p95={met.get('path_at_95')} "
              f"({wall_s:.0f}s)"
              + (f"  *** {_fb} FALLBACK DECISIONS ***" if _fb not in ("", 0) else ""))
    return rows


# ===========================================================================
# MAIN SWEEP
# ===========================================================================
def _service_ports(config):
    """The Tier-1 service ports, as a list. Falls back to the single-port
    setting so an older config keeps working."""
    ports = list(getattr(config, 'MAPEX_SERVICE_PORTS', []) or [])
    if not ports:
        ports = [int(getattr(config, 'MAPEX_SERVICE_PORT', 5555))]
    return [int(p) for p in ports]


def _worker_index():
    """0-based index of this pool worker, or 0 when running sequentially.
    Pool identities are 1-based and keep incrementing as maxtasksperchild
    recycles processes, which is fine: we only use it modulo the port count."""
    try:
        from multiprocessing import current_process
        ident = getattr(current_process(), '_identity', ()) or ()
        if ident:
            return int(ident[0]) - 1
    except Exception:
        pass
    return 0


def _pin_service_port(config):
    """Give this worker its own service instance and return the port.

    intelx.baselines.tier1._tier1_client reads config.MAPEX_SERVICE_PORT and caches one
    persistent socket per explorer, so setting it here pins the worker to a
    single service for the whole task instead of reconnecting per call."""
    ports = _service_ports(config)
    port = ports[_worker_index() % len(ports)]
    config.MAPEX_SERVICE_PORT = port
    return port


def _pin_config(config):
    """Force the reproducibility/speed settings onto the run config and return a
    dict of what was applied (recorded alongside the results)."""
    applied = {}
    for k, v in getattr(config, 'PIN_CONFIG', {}).items():
        setattr(config, k, v)
        applied[k] = v
    from intelx.common import apply_submitted_behaviour
    apply_submitted_behaviour(config)
    return applied


def _modes_for_map(config, map_name):
    """Modes that should run on this map, honouring MODE_MAPS restrictions."""
    out = []
    for mode in config.MODES:
        allowed = getattr(config, 'MODE_MAPS', {}).get(mode)
        if allowed is None or map_name in allowed:
            out.append(mode)
    return out


def _runs_for_map(config, map_name):
    """Spawns drawn for this map: the most any single mode will use."""
    return int(getattr(config, 'MAP_RUNS', {}).get(map_name, config.RUNS_PER_MAP))


def _runs_for_mode(config, map_name, mode):
    """Spawns this mode uses on this map. Always a PREFIX of the map's seed
    sequence, so a cheap mode's rows are a strict subset of an expensive one's
    and every row it produces still pairs."""
    return min(_runs_for_map(config, map_name),
               int(getattr(config, 'MODE_RUNS', {}).get(mode, config.RUNS_PER_MAP)))


def _mode_sensor_pairs(config, modes):
    """Expand each mode into the (mode, sensor) variants it runs under."""
    out = []
    for m in modes:
        for sensor in getattr(config, 'MODE_SENSORS', {}).get(
                m, [getattr(config, 'DEFAULT_SENSOR', 'intelx')]):
            out.append((m, sensor))
    return out


def apply_sensor_profile(config, sensor):
    """Return a copy of config.sim with the sensor profile applied. Only the
    sensing changes -- map, resolution, robot radius and step size stay
    identical, so any difference is attributable to sensing plus policy."""
    cfg = dict(config.sim)
    prof = getattr(config, 'SENSOR_PROFILES', {}).get(sensor)
    if prof:
        cfg.update(prof)
    return cfg


def _split_tasks(tasks):
    """(non_tier1_tasks, tier1_tasks), each with its modes filtered. A task with
    no modes left in a phase is dropped from that phase."""
    a, b = [], []
    for map_path, seed, spawn, modes, paths in tasks:
        m0 = [m for m in modes if m not in TIER1_MODES]
        m1 = [m for m in modes if m in TIER1_MODES]
        if m0:
            a.append((map_path, seed, spawn, m0, paths))
        if m1:
            b.append((map_path, seed, spawn, m1, paths))
    return a, b


def resolve_paths(config, run_name=None):
    """Build (and create) the output layout. Returns a plain dict so it can be
    handed to worker processes -- with the 'spawn' start method a child
    re-imports this module and would otherwise not see what the parent
    computed."""
    root = getattr(config, 'RESULTS_ROOT', '') or str(_RESULTS_ROOT)
    name = run_name or getattr(config, 'RUN_NAME', '') or \
        time.strftime("run_%Y%m%d_%H%M")
    run_dir = os.path.join(root, name)
    p = {
        "root": root,
        "run_name": name,
        "run_dir": run_dir,
        "csv": os.path.join(run_dir, "results.csv"),
        "meta": os.path.join(run_dir, "results_meta.json"),
        "curves": os.path.join(run_dir, "curves"),
        "videos": os.path.join(run_dir, "videos"),
        "tables": os.path.join(run_dir, "tables"),
        "spawns": os.path.join(root, "spawns.json"),   # SHARED across runs
    }
    for k in ("run_dir", "curves", "videos", "tables"):
        os.makedirs(p[k], exist_ok=True)
    return p


def preflight(config):
    """Fail fast with a readable message instead of a traceback 20 lines deep."""
    errs, warns = [], []
    if not config.MAPS:
        errs.append("SWEEP.MAPS is empty -- add your .pgm map paths.")
    for m in config.MAPS:
        if not os.path.exists(m):
            errs.append(f"map not found: {m}")
    bad_cp = [c for c in config.COVERAGE_CHECKPOINTS if c > config.TARGET_PCT]
    if bad_cp:
        errs.append(f"COVERAGE_CHECKPOINTS {bad_cp} exceed TARGET_PCT="
                    f"{config.TARGET_PCT}: the run stops before those can be "
                    f"reached, so they would be censored for every method.")
    tgt = config.policy.get("exploration_target_pct")
    if tgt is not None and abs(float(tgt) - float(config.TARGET_PCT)) > 1e-6:
        warns.append(f"policy['exploration_target_pct']={tgt} != TARGET_PCT="
                     f"{config.TARGET_PCT}; the simulator stops at the former and "
                     f"metrics score against the latter.")
    unknown = [m for m in config.MODES if m not in MODE_REGISTRY]
    if unknown:
        errs.append(f"unknown mode(s) {unknown}; known: {sorted(MODE_REGISTRY)}")

    # every mode must have a sensor profile that EXISTS: apply_sensor_profile
    # silently returns config.sim unchanged for an unknown name, so the row would
    # be labelled with a profile it never used
    for mode in config.MODES:
        sensors = getattr(config, 'MODE_SENSORS', {}).get(
            mode, [getattr(config, 'DEFAULT_SENSOR', 'intelx')])
        for s in sensors:
            if s not in getattr(config, 'SENSOR_PROFILES', {}):
                errs.append(f"mode '{mode}' is listed against sensor '{s}' but "
                            f"SENSOR_PROFILES has no such profile; the run would "
                            f"be labelled '{s}' while using the default sensor.")

    if any(m in NEEDS_MODEL for m in config.MODES):
        if not os.path.exists(config.MODEL_CHECKPOINT):
            errs.append(f"model checkpoint not found: {config.MODEL_CHECKPOINT} "
                        f"(needed by {sorted(set(config.MODES) & NEEDS_MODEL)})")
    if any(m in TIER0_MODES for m in config.MODES):
        try:
            from intelx.baselines import simple as baselines_simple  # noqa: F401
        except Exception as e:
            errs.append(f"Tier-0 modes need intelx.baselines.simple ({e})")
    if "frontier" in config.MODES or any(m in TIER0_MODES for m in config.MODES) \
            or any(m in TIER1_MODES for m in config.MODES):
        try:
            cls = _base_monolith()
        except Exception as e:
            errs.append(f"frontier monolith not importable: {e}")
            cls = None
        # FAIRNESS: intelx.simulation honours FIXED_DT and TURBO_MODE; the
        # monolith originally derived dt from the wall clock and slept 50 ms
        # every iteration regardless of turbo. Left unpatched, every baseline
        # runs real-time with a step size that grows under machine load, while
        # ours runs uncapped and deterministic -- so baseline coverage curves
        # depend on how busy the box was.
        if cls is not None:
            try:
                import inspect
                src = inspect.getsource(cls._updated_simulation_worker)
                if 'FIXED_DT' not in src:
                    parallel = (int(getattr(config, 'N_WORKERS', 1)) > 1
                                or len(_service_ports(config)) *
                                int(getattr(config, 'TIER1_WORKERS_PER_SERVICE', 1)) > 1)
                    msg = ("the frontier monolith's _updated_simulation_worker does "
                           "not read FIXED_DT: baselines will use wall-clock dt and "
                           "the 20 Hz sleep while 'ours' runs on FIXED_DT + turbo. "
                           "Step size then varies with machine load, which changes "
                           "scan spacing and therefore measured coverage.")
                    if parallel:
                        msg += (" YOU ARE RUNNING MULTIPLE WORKERS, so contention "
                                "will change the baselines' RESULTS, not just their "
                                "wall time. Patch the monolith or drop to one "
                                "worker everywhere before generating final numbers.")
                    warns.append(msg)
            except Exception:
                pass

    tier1 = [m for m in config.MODES if m in TIER1_MODES]
    if tier1:
        # UPEN silently degrades to nearest-frontier without these two keys
        if "upen" in tier1:
            uc = getattr(config, 'MAPEX_UPEN_CONFIG', {}) or {}
            miss = [k for k in ("reach_horizon", "upen_mode") if k not in uc]
            if miss:
                errs.append(f"MAPEX_UPEN_CONFIG is missing {miss}; their "
                            f"get_rrt_goal requires them and the service would "
                            f"fall back to the nearest frontier on every call, "
                            f"making 'upen' a nearest-frontier baseline.")
        try:
            from intelx.baselines.tier1 import MapExClient
            ports = _service_ports(config)
            live = []
            for p in ports:
                c = MapExClient(config.MAPEX_SERVICE_HOST, p, timeout=5.0)
                pong = c.ping()
                if pong.get('ok'):
                    live.append(p)
                    print("[preflight] MapEx service OK on %s:%d (%s models)"
                          % (config.MAPEX_SERVICE_HOST, p, pong.get('n_models', '?')))
                else:
                    errs.append("Tier-1 modes %s: no service on %s:%d (%s). Start "
                                "it in the lama env: python mapex_service.py "
                                "--port %d"
                                % (tier1, config.MAPEX_SERVICE_HOST, p,
                                   pong.get('error'), p))
            per = int(getattr(config, 'TIER1_WORKERS_PER_SERVICE', 1))
            split = getattr(config, 'EXECUTION', 'concurrent') == 'split'
            if live:
                cap = len(live) * per
                if split:
                    print("[preflight] %d service instance(s); Tier-1 phase will "
                          "use %d worker(s)" % (len(live), cap))
                else:
                    nw = int(getattr(config, 'N_WORKERS_TIER1', None) or cap)
                    print("[preflight] %d service instance(s), capacity ~%d "
                          "concurrent decision(s); single phase with %d worker(s)"
                          % (len(live), cap, nw))
                    if nw > cap:
                        warns.append(
                            "Tier-1 workers=%d exceed the service capacity "
                            "(%d service(s) x %d = %d). mapex_service serialises "
                            "plan() behind one lock, so the surplus workers will "
                            "queue and tier1_service_s_mean will not show it "
                            "(the service times only its own work, not the wait). "
                            "Either start %d more service(s) on extra ports, or "
                            "set N_WORKERS_TIER1 = %d."
                            % (nw, len(live), per, cap,
                               -(-nw // per) - len(live), cap))
        except Exception as e:
            errs.append("Tier-1 modes %s: cannot reach mapex_service.py (%s). "
                        "Start it in the lama env: python mapex_service.py --port %d"
                        % (tier1, e, _service_ports(config)[0]))

    for mode, maps in getattr(config, 'MODE_MAPS', {}).items():
        names = {os.path.splitext(os.path.basename(m))[0] for m in config.MAPS}
        missing = [x for x in maps if x not in names]
        if missing:
            warns.append(f"MODE_MAPS['{mode}'] lists map(s) not in MAPS: {missing}")

    # unequal seed counts are legitimate but change what pairs, so say it
    for map_path in config.MAPS:
        mn = os.path.splitext(os.path.basename(map_path))[0]
        counts = dict((m, _runs_for_mode(config, mn, m))
                      for m in _modes_for_map(config, mn))
        if counts and len(set(counts.values())) > 1:
            k = min(counts.values())
            warns.append(f"{mn}: modes use different seed counts {counts}; only "
                         f"the first {k} seed(s) have every method, so paired "
                         f"tests must be restricted to those.")
            break

    if getattr(config, 'N_WORKERS', 1) > 1 and not getattr(config, 'TIMING_RERUN', False):
        warns.append("N_WORKERS > 1: wall_s / s_per_m / ms_per_decision will be "
                     "inflated by contention (path & quality metrics are fine).")
    for w in warns:
        print(f"[preflight] WARNING: {w}")
    if errs:
        print("\n[preflight] CANNOT START:")
        for e in errs:
            print(f"  - {e}")
        return False
    print(f"[preflight] ok: {len(config.MAPS)} map(s), modes={list(config.MODES)}, "
          f"device={config.device}")
    return True


def run_sweep(config=SWEEP):
    if not preflight(config):
        return
    header = (["map", "category", "seed", "mode", "sensor", "spawn_x", "spawn_y",
               "spawn_theta", "goals"] + TIER1_COLUMNS + metrics.METRIC_COLUMNS)
    paths = resolve_paths(config)
    print(f"[sweep] output layout:")
    print(f"    run dir : {paths['run_dir']}")
    print(f"    csv     : {paths['csv']}")
    print(f"    curves  : {paths['curves']}/")
    print(f"    videos  : {paths['videos']}/")
    print(f"    tables  : {paths['tables']}/")
    print(f"    spawns  : {paths['spawns']}   (shared across runs -- keep this)")

    pinned = _pin_config(config)
    timing_rerun = bool(getattr(config, 'TIMING_RERUN', False))
    n_workers = 1 if timing_rerun else max(1, int(getattr(config, 'N_WORKERS', 1)))
    ports = _service_ports(config)
    n_workers_t1 = getattr(config, 'N_WORKERS_TIER1', None)
    if n_workers_t1 is None:
        n_workers_t1 = len(ports) * int(getattr(config, 'TIER1_WORKERS_PER_SERVICE', 1))
    n_workers_t1 = 1 if timing_rerun else max(1, int(n_workers_t1))

    # ---- build the task list, resolving spawns from the manifest -----------
    manifest = load_spawn_manifest(paths["spawns"])
    n_new = 0
    tasks = []
    for map_path in config.MAPS:
        map_name = os.path.splitext(os.path.basename(map_path))[0]
        modes = _modes_for_map(config, map_name)
        if not modes:
            print(f"[sweep] {map_name}: no modes enabled, skipping.")
            continue
        n_runs = _runs_for_map(config, map_name)
        probe = build_simulator(map_path, config)
        occ0 = probe.occ.copy()
        del probe
        res = config.sim.get('RESOLUTION', 0.05)
        clearance_px = config.SPAWN_CLEARANCE_M / res
        counts = dict((m, _runs_for_mode(config, map_name, m)) for m in modes)
        print(f"[sweep] {map_name}: free area {map_free_area_m2(occ0, res):.0f} m2, "
              f"{n_runs} spawn(s), runs per mode {counts}")
        for k in range(n_runs):
            seed = config.SEED_BASE + k
            modes_k = [m for m in modes if k < counts[m]]
            if not modes_k:
                continue
            spawn, is_new = get_spawn(manifest, map_name, seed, occ0, clearance_px)
            n_new += int(is_new)
            tasks.append((map_path, seed, spawn, modes_k, paths))
    save_spawn_manifest(paths["spawns"], manifest)
    print(f"[sweep] spawns: {len(manifest)} in manifest ({n_new} new) -> {paths['spawns']}")

    total_runs = sum(len(_mode_sensor_pairs(config, t[3])) for t in tasks)
    print(f"[sweep] {len(tasks)} task(s) / {total_runs} run(s)")

    # ---- sweep-level metadata ---------------------------------------------
    try:
        import json
        with open(paths["meta"], "w") as f:
            json.dump({"pinned": pinned, "modes": list(config.MODES),
                       "maps": [os.path.basename(m) for m in config.MAPS],
                       "map_category": getattr(config, 'MAP_CATEGORY', {}),
                       "runs_per_map": config.RUNS_PER_MAP,
                       "map_runs": getattr(config, 'MAP_RUNS', {}),
                       "mode_runs": getattr(config, 'MODE_RUNS', {}),
                       "mode_maps": getattr(config, 'MODE_MAPS', {}),
                       "seed_base": config.SEED_BASE,
                       "n_workers": n_workers, "n_workers_tier1": n_workers_t1,
                       "mapex_service_ports": ports,
                       "tier1_workers_per_service": getattr(
                           config, 'TIER1_WORKERS_PER_SERVICE', 1),
                       "execution": getattr(config, 'EXECUTION', 'concurrent'),
                       "sensor_profiles": getattr(config, 'SENSOR_PROFILES', {}),
                       "mode_sensors": getattr(config, 'MODE_SENSORS', {}),
                       "mapex_upen_config": getattr(config, 'MAPEX_UPEN_CONFIG', {}),
                       "mapex_pred_range_m": getattr(config, 'MAPEX_PRED_RANGE_M', None),
                       "mapex_pred_beams": getattr(config, 'MAPEX_PRED_BEAMS', None),
                       "checkpoints": config.COVERAGE_CHECKPOINTS,
                       "max_path_m": config.MAX_PATH_M, "max_wall_s": config.MAX_WALL_S,
                       "stall_idle_s": config.STALL_IDLE_S,
                       "stall_idle_s_by_mode": dict(
                           DEFAULT_STALL_IDLE_S_BY_MODE,
                           **(getattr(config, 'STALL_IDLE_S_BY_MODE', {}) or {})),
                       "target_pct": config.TARGET_PCT,
                       "spawns_json": paths["spawns"], "run_dir": paths["run_dir"],
                       "timing_valid": timing_rerun,
                       "video": bool(getattr(config, 'SAVE_VIDEO', False)),
                       "frontier_module": _RESOLVED.get("frontier_module", FRONTIER_MODULE_ORDER[0])}, f, indent=2)
    except Exception as e:
        print(f"[sweep] could not write meta json: {e}")

    # ---- execute -----------------------------------------------------------
    t_start = time.time()
    all_rows = []

    # Results are appended AS EACH TASK FINISHES, not at the end: a multi-hour
    # sweep that crashes (or is Ctrl-C'd) must not lose everything.
    _rows_lock = threading.Lock()

    def _record(rows):
        """Append rows to memory and to the CSV. Both pools call this from their
        own thread, so it is serialised."""
        with _rows_lock:
            all_rows.extend(rows)
            new = not os.path.exists(paths["csv"])
            with open(paths["csv"], "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=header)
                if new:
                    w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in header})

    if os.path.exists(paths["csv"]):
        os.remove(paths["csv"])            # fresh file for this run dir

    def _execute(phase_tasks, workers, label):
        if not phase_tasks:
            return
        runs = sum(len(_mode_sensor_pairs(config, t[3])) for t in phase_tasks)
        print(f"\n[sweep] === {label}: {len(phase_tasks)} task(s) / {runs} run(s) "
              f"on {workers} worker(s) ===")
        t_phase = time.time()
        if workers == 1:
            for i, t in enumerate(phase_tasks, 1):
                print(f"[{label} {i}/{len(phase_tasks)}] "
                      f"{os.path.basename(t[0])} seed{t[1]} {t[3]}")
                rows_i = run_task(t)
                _record(rows_i)
        else:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")      # required for CUDA in subprocesses
            with ctx.Pool(processes=workers, maxtasksperchild=4) as pool:
                for i, rows in enumerate(pool.imap_unordered(run_task, phase_tasks), 1):
                    _record(rows)
                    el = time.time() - t_phase
                    eta = el / i * (len(phase_tasks) - i)
                    print(f"[{label} {i}/{len(phase_tasks)}] done | "
                          f"elapsed {el/60:.1f}m | eta {eta/60:.1f}m")

    execution = getattr(config, 'EXECUTION', 'concurrent')
    if getattr(config, 'PHASE_SPLIT_TIER1', None) is True:
        execution = 'split'                # back-compat with the old flag

    if execution == 'single':
        # One pool, tasks that each run every mode. A task walks its modes in
        # MODES order, so the services stay idle until every worker has finished
        # its Tier-0 modes. Kept for completeness; "concurrent" is better.
        _execute(tasks, n_workers, "all modes")
    else:
        other_tasks, tier1_tasks = _split_tasks(tasks)
        if execution == 'split':
            _execute(other_tasks, n_workers, "phase 1 (tier0 + ours)")
            _execute(tier1_tasks, n_workers_t1, "phase 2 (tier1)")
        else:
            # Concurrent: both pools live at once, so the services are busy from
            # the start instead of waiting out the Tier-0 work. Each pool has its
            # own worker count, which is what bounds Tier-1 to the number of
            # decisions the services can actually absorb.
            threads = []
            for _t, _w, _lab in ((other_tasks, n_workers, "tier0+ours"),
                                 (tier1_tasks, n_workers_t1, "tier1")):
                if not _t:
                    continue
                th = threading.Thread(target=_execute, args=(_t, _w, _lab),
                                      daemon=True)
                th.start()
                threads.append(th)
            for th in threads:
                th.join()

    print(f"\nSweep done: {len(all_rows)} runs in {(time.time()-t_start)/60:.1f} min "
          f"-> {paths['csv']}")

    # --- stop-reason summary -------------------------------------------------
    by_mode = {}
    for r in all_rows:
        key = f"{r['mode']}@{r.get('sensor', '')}"
        by_mode.setdefault(key, {}).setdefault(r.get("stop_reason", "?"), 0)
        by_mode[key][r.get("stop_reason", "?")] += 1
    print("\nstop_reason by mode:")
    n_to = n_stall = 0
    for mode in sorted(by_mode):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(by_mode[mode].items()))
        print(f"  {mode:<20} {parts}")
        n_to += by_mode[mode].get("timeout", 0)
        n_stall += by_mode[mode].get("stalled", 0)
    if n_to:
        print(f"\n  *** {n_to} run(s) hit MAX_WALL_S={config.MAX_WALL_S:.0f}s. Their path")
        print("      metrics are censored at MAX_PATH_M and UNDERSTATE those methods.")
        print("      Raise MAX_WALL_S (it is a safety net, not the budget) and re-run,")
        print("      or treat those rows as invalid rather than censored.")
    if n_stall:
        print(f"\n  *** {n_stall} run(s) stalled (no motion for STALL_IDLE_S).")
        print("      For Tier-1 that usually means a planning call ran longer than the")
        print("      stall window, NOT that the robot was stuck. Same censoring applies.")

    # --- fallback audit: a run containing decisions the named planner did not
    # make is not that method and must not be reported as such.
    bad = [r for r in all_rows if r.get("tier1_fallbacks") not in ("", 0, None)]
    if bad:
        print(f"\n  *** {len(bad)} run(s) contain FALLBACK decisions (the named "
              f"planner did not choose the goal):")
        for r in bad:
            print(f"      {r['map']} s{r['seed']} {r['mode']}@{r.get('sensor','')}: "
                  f"{r['tier1_fallbacks']}/{r['tier1_calls']} calls")
        print("      Exclude these rows, or fix the cause and re-run them.")

    # --- pairing audit: which (map, seed) actually have every method ----------
    want = set()
    for m in config.MODES:
        for s in getattr(config, 'MODE_SENSORS', {}).get(
                m, [getattr(config, 'DEFAULT_SENSOR', 'intelx')]):
            want.add((m, s))
    have = {}
    for r in all_rows:
        have.setdefault((r["map"], r["seed"]), set()).add((r["mode"], r.get("sensor", "")))
    full = sorted(k for k, v in have.items() if want <= v)
    print(f"\npairing: {len(full)}/{len(have)} (map, seed) cell(s) have all "
          f"{len(want)} method/sensor combinations")
    if len(full) < len(have):
        print("      Restrict paired tests to the complete cells. A cell is "
              "incomplete when a mode has a smaller MODE_RUNS, or a run errored.")

    if n_workers > 1 and not timing_rerun:
        print("\n[sweep] NOTE: wall_s / s_per_m / ms_per_decision are inflated by "
              "parallel contention. Re-run a subset with TIMING_RERUN=True for the "
              "timing column.")

    # --- the showcase: every method on each map, side by side ----------------
    # Rendered from the curves just saved, so it can never cost a result.
    if getattr(config, 'SHOWCASE_VIDEO', True) and getattr(config, 'SAVE_CURVES', True):
        try:
            from intelx.viz.showcase import render_dirs
            made = render_dirs([paths["run_dir"]], out_dir=paths["videos"])
            print("\n[sweep] showcase: %d video(s) in %s" % (len(made), paths["videos"]))
        except Exception as e:
            print("\n[sweep] showcase skipped: %s" % e)


# ===========================================================================
# SELF-TEST (no sim/model/service): validates spawn seeding, seed nesting,
# task splitting, the monitor loop and CSV writing
# ===========================================================================
def _selftest():
    import tempfile
    print("== pure helpers ==")
    occ = np.zeros((100, 100), np.int8); occ[:, 80:] = 1
    a = pick_spawn(occ, 42, 5.0); b = pick_spawn(occ, 42, 5.0); c = pick_spawn(occ, 43, 5.0)
    assert a == b and a != c, "spawn must be deterministic per seed and vary across seeds"
    assert occ[int(a[1]), int(a[0])] == 0, "spawn must be free"
    print("  pick_spawn deterministic + free:", tuple(round(v, 1) for v in a))

    curve = [(0, 0), (5, 40), (10, 92), (15, 96), (20, 98)]
    p90 = path_at_coverage(curve, 90); p95 = path_at_coverage(curve, 95); p99 = path_at_coverage(curve, 99)
    print(f"  path@90={p90:.2f} path@95={p95:.2f} path@99={p99}")
    assert 5 < p90 < 10 and 10 < p95 < 15 and p99 is None
    m = curve_to_metrics(curve, [90, 95, 99], max_path_m=400.0)
    assert m["reached_90"] == 1 and m["reached_99"] == 0 and m["path_at_99"] == 400.0
    print("  censoring OK: path_at_99 censored to budget, reached_99=0")

    print("== seed nesting (the pairing guarantee) ==")
    class _C:
        RUNS_PER_MAP = 10
        MAP_RUNS = {}
        MODE_RUNS = {"mapex": 5, "upen": 5}
        SEED_BASE = 1005
        MODES = ["ours", "mapex", "upen"]
        MODE_MAPS = {}
    seeds = {}
    for mode in _C.MODES:
        n = _runs_for_mode(_C, "m", mode)
        seeds[mode] = [_C.SEED_BASE + k for k in range(n)]
    print("  ", {k: (v[0], v[-1], len(v)) for k, v in seeds.items()})
    assert set(seeds["mapex"]) < set(seeds["ours"]), "cheap mode must be a subset"
    assert seeds["mapex"] == seeds["upen"]
    assert seeds["ours"][:5] == seeds["mapex"], "must be a PREFIX, not a sample"
    print("  every mapex/upen seed also has ours -> those rows pair  OK")

    print("== tier1 stat summarisation ==")
    class _E(object):
        pass
    e = _E()
    assert tier1_row_stats(None)["tier1_calls"] == ""
    assert tier1_row_stats(e)["tier1_calls"] == ""
    e._tier1_stats = {'calls': 4, 'service_s_sum': 40.0, 'fallbacks': 1,
                      'failures': 0, 'n_frontiers_sum': 12, 'snapped': 2,
                      'repeats': 3}
    s = tier1_row_stats(e)
    assert s["tier1_service_s_mean"] == 10.0 and s["tier1_fallbacks"] == 1
    assert s["tier1_snapped"] == 2 and s["tier1_repeats"] == 3
    assert set(s) == set(TIER1_COLUMNS), set(s) ^ set(TIER1_COLUMNS)
    print("  ", s)

    print("== per-mode stall window ==")
    class _S:
        STALL_IDLE_S = 1800.0
    assert stall_window_for(_S, "ours") == 300.0
    assert stall_window_for(_S, "nearest") == 300.0
    assert stall_window_for(_S, "mapex") == 1800.0
    assert stall_window_for(_S, "something_new") == 1800.0, "unknown -> SWEEP default"
    class _S2:
        STALL_IDLE_S = 1800.0
        STALL_IDLE_S_BY_MODE = {"ours": 60.0}
    assert stall_window_for(_S2, "ours") == 60.0, "config override must win"
    assert stall_window_for(_S2, "mapex") == 1800.0
    print("  ours/tier0 300 s, tier1 1800 s, unknown -> default, override honoured")

    print("== service port assignment ==")
    class _P:
        MAPEX_SERVICE_PORTS = [5555, 5556]
        MAPEX_SERVICE_PORT = 5555
    assert _service_ports(_P) == [5555, 5556]
    class _P1:
        MAPEX_SERVICE_PORT = 5599
    assert _service_ports(_P1) == [5599], "single-port configs must still work"
    seen = set()
    for idx in range(6):
        seen.add(_service_ports(_P)[idx % len(_service_ports(_P))])
    assert seen == {5555, 5556}, seen
    _pin_service_port(_P)
    assert _P.MAPEX_SERVICE_PORT in (5555, 5556)
    print("  ports", _service_ports(_P), "| round robin covers both | pinned",
          _P.MAPEX_SERVICE_PORT)

    print("== task splitting (tier1 gets its own pool) ==")
    t = [("m.pgm", 1005, (0, 0, 0), ["ours", "frontier", "mapex"], {}),
         ("m.pgm", 1006, (0, 0, 0), ["ours", "frontier"], {})]
    other, tier1 = _split_tasks(t)
    assert [x[3] for x in other] == [["ours", "frontier"], ["ours", "frontier"]]
    assert [x[3] for x in tier1] == [["mapex"]]
    assert tier1[0][2] == other[0][2], "both phases must use the SAME spawn"
    print("  phase 1 modes", [x[3] for x in other], "| phase 2", [x[3] for x in tier1])

    print("== monitored run with a FAKE explorer ==")
    class FakeSim:
        def __init__(s): s.pose = np.zeros(3); s.occ = occ; s.explored = np.zeros_like(occ, bool)
        def _simulate_scan(s, p): return None
        def _update_exploration(s, p, sc): pass
    class FakeExplorer:
        """Mimics the real explorer interface. `eff` = coverage-% gained per metre
        of path -> higher eff reaches a checkpoint in LESS path (the real metric).
        Both advance path at the same speed; only efficiency differs."""
        def __init__(s, eff):
            s.simulator = FakeSim(); s.state_lock = threading.Lock()
            s.running = threading.Event(); s.exploration_complete = threading.Event()
            s.shared_state = {'exploration_pct': 0.0, 'total_path_length': 0.0, 'goal_count': 0}
            s._eff = eff
        def _prediction_worker(s):
            while s.running.is_set(): time.sleep(0.01)
        def _updated_simulation_worker(s):
            p = 0.0
            while s.running.is_set() and not s.exploration_complete.is_set():
                time.sleep(0.02); p += 0.5           # same speed for both
                with s.state_lock:
                    s.shared_state['total_path_length'] = p
                    s.shared_state['exploration_pct'] = min(100.0, p * s._eff)
                    s.shared_state['goal_count'] += 1
                if s.shared_state['exploration_pct'] >= 98.0:
                    s.exploration_complete.set()
    fast = FakeExplorer(eff=10.0); slow = FakeExplorer(eff=5.0)   # paired: same "spawn"
    cf, tjf, gvf, sf = monitor_run(fast, [90, 95], 400, 10, 0.05)
    cs, tjs, gvs, ss = monitor_run(slow, [90, 95], 400, 10, 0.05)
    _meta = {'path_budget_m': 400, 'free_area_m2': 500.0, 'target_pct': 98.0,
             'wall_s': 1.0, 'stop_reason': sf, 'resolution': 0.05,
             'checkpoints': [90, 95]}
    mf = metrics.compute_all(cf, tjf, gvf, _meta)
    ms = metrics.compute_all(cs, tjs, gvs, dict(_meta, stop_reason=ss))
    print(f"  recorded raw: curve={len(cf)} traj={len(tjf)} goal_events={len(gvf)}")
    print(f"  fast AUC={mf['auc_coverage']:.3f} vs slow AUC={ms['auc_coverage']:.3f}")
    assert mf['auc_coverage'] > ms['auc_coverage'], "AUC must rank the faster explorer higher"
    print(f"  fast: stop={sf} path@95={mf['path_at_95']:.2f}")
    print(f"  slow: stop={ss} path@95={ms['path_at_95']:.2f}")
    assert mf['path_at_95'] < ms['path_at_95'], "faster explorer should reach 95% in less path"

    print("== CSV writing ==")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        hdr = ["map", "seed", "mode", "path_at_95"]
        write_row(p, {"map": "m", "seed": 0, "mode": "fast", "path_at_95": mf['path_at_95']}, hdr)
        write_row(p, {"map": "m", "seed": 0, "mode": "slow", "path_at_95": ms['path_at_95']}, hdr)
        import pandas as pd
        df = pd.read_csv(p)
        assert len(df) == 2 and set(df['mode']) == {"fast", "slow"}
        print("  wrote", len(df), "paired rows for one (map,seed)")
    print("\n*** run_sweep_all self-test PASSED ***")


def main(argv=None):
    """Run the paired benchmark, or the self-test with --selftest.

    The self-test needs no simulator, no checkpoint and no service, so it is
    the fast way to check an installation."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--selftest" in argv:
        _selftest()
    else:
        run_sweep(SWEEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())