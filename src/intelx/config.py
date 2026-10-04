"""The configuration for a single exploration episode.

Every knob a run reads lives here. Nothing in this file is policy. To change
how the robot decides where to go, edit `intelx.policy_ablation`.

`MAP_UPDATE_INTERVAL` controls how often a scan is fused into the explored
map, so it moves the coverage metric directly. Leave it at 5 for any run you
intend to report.
"""
from math import pi

import torch

from intelx.paths import RESULTS_ROOT, map_path, weights_path


class ExplorationConfig:
    # UPDATED: Point to YOUR trained model
    MODEL_CHECKPOINT = weights_path("checkpoint_best.pth")
    MODEL_TYPE = 'resnet34'  # Must match your training
    # The video, final_trajectory.png and prediction_debug.log land here.
    output_dir = str(RESULTS_ROOT / "run_intelx")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    HEADLESS = False  # True = No Window (Fast), False = Show Window
    # 'paper' draws the run the way the paper's videos do (intelx.viz.live);
    # 'debug' is the older three-panel dashboard. VIEW_FPS is how often the
    # view is drawn, and the frame rate of the saved video.
    VIEW_STYLE = 'paper'
    VIEW_FPS = 10.0
    TURBO_MODE = False #True  # True = No artificial sleep (Max Speed) -- now actually used
    # NOTE: this controls how often a laser scan is taken and fused into the
    # explored map, so it DIRECTLY affects the generated occupancy/explored map
    # and therefore your coverage metric. The original ignored this knob and
    # always used 5. Keep it at 5 for any run you intend to report; raise it
    # only for quick smoke tests where map fidelity does not matter.
    MAP_UPDATE_INTERVAL = 5  # scan every N movement steps (5 -> one scan per 18.8 cm)
    SAVE_INTERVAL = 1000  # Save snapshot every N steps (0 to disable)

    # --- SPEED KNOBS -------------------------------------------------------
    # Fixed physics timestep. The original derived dt from the wall clock, so
    # the distance moved per iteration depended on machine load (0.75 px when
    # idle, up to 1.50 px when busy) and TURBO_MODE could not work. Pinning it
    # to 0.05 reproduces the nominal 20 Hz step exactly and makes it constant.
    # Set to None to restore the old wall-clock behaviour.
    FIXED_DT = 0.05

    # How often the sim thread publishes state to the viz/main threads.
    # Does not affect the robot, the scans or the map -- only how often the
    # snapshots used for drawing and progress reporting are refreshed.
    STATE_PUBLISH_EVERY = 10

    # Spacing (in pixels of robot motion) between "free" prediction passes made
    # when no goal was requested. These passes still run plan_global, so they
    # affect junction detection -- the throttle is by distance, not time, so
    # this density is unchanged by TURBO_MODE. 0 = predict only on goal request.
    IDLE_PREDICT_EVERY_PX = 1.0

    # Write the 12-image polygon-verification dump on every junction check.
    # Costs ~400 ms and ~10 MB each time, inside the prediction thread.
    JUNCTION_DEBUG = False
    # -----------------------------------------------------------------------

    ROBOT_RADIUS_METERS = 0.35  # 0.5
    RESOLUTION = 0.05
    WIN_M = 20  # 20
    LASER_SCAN_R = 10

    # Custom spawn: None = use map default, or (x_px, y_px, theta_rad)
    SPAWN_POINT = None
    # e.g. (850, 750, 0.0)
    # --- LOCAL-ONLY POLICY (ablation baseline) -----------------------------
    # When True, skip ALL junction detection and the global planner. The robot
    # follows the raw coverage-prediction model alone. This isolates the
    # coverage policy from the scaffolding, which is what you want when the
    # junction machinery's behaviour depends heavily on where the robot spawns.
    LOCAL_ONLY_MODE = False
    # How the local-only goal is chosen:
    #   'centroid' -> intensity-weighted centroid of the whole coverage field
    #                 (the pure "follow the coverage prediction" policy)
    #   'argmax'   -> single hottest pixel (the previous local-only behaviour)
    LOCAL_ONLY_GOAL = 'argmax'
    # Optional: keep only the top (100 - PERCENTILE)% of coverage pixels before
    # taking the centroid. 0 = use all predictions. Raise toward ~85-95 if the
    # full-field centroid sits too close to the robot.
    LOCAL_CENTROID_PERCENTILE = 0.0
    # Optional: sharpen the weighting (w ** POWER) toward strong coverage.
    LOCAL_CENTROID_WEIGHT_POWER = 1.0
    # -----------------------------------------------------------------------

    # True restores the behaviour the submitted results were produced with,
    # here A* refusing to plan from a robot one pixel inside a wall
    # (intelx.common.START_SNAP_PX). The benchmark has the same switch.
    SUBMITTED_BEHAVIOUR = False

    # The three maps this release ships:
    #   test_standardized/small_office3.pgm
    #   ood_test_standardized/aws_maps/hospital.pgm
    #   real/real_slam_corridor.pgm
    sim = {
        'MAP_PGM': map_path("test_standardized/small_office3.pgm"),
        'RESOLUTION': RESOLUTION, 'ROBOT_R': ROBOT_RADIUS_METERS, 'WIN_M': WIN_M,
        'SPAWN_CLEARANCE': 0.45, 'FOV': 220 * pi / 180, 'N_BEAMS': 662, 'MAX_R': int(LASER_SCAN_R / RESOLUTION),
        'STEP_SIZE_M': 0.1, 'ANG_VEL': pi, 'DT': 0.1, 'WIN_PX': int(WIN_M / RESOLUTION),
        'SAVE_DIR': 'temp_autonomous_exploration', 'SHOW_HEAD': True, 'VISUALIZE_FOV': False,
        'RAD_CELLS': int(ROBOT_RADIUS_METERS / RESOLUTION)
    }
    sim['ZOOM_WINDOW_M'] = 40.0
    sim['ZOOM_WINDOW_PX'] = int(sim['ZOOM_WINDOW_M'] / sim['RESOLUTION'])

    policy = {
        'exploration_target_pct': 99,   # rho* of the paper
        'goal_history_length': 5
    }

    # --- ENROUTE GUARDS ----------------------------------------------------
    # EARLY_STOP_MARGIN_PX shrinks the "frontier is already sensed" box. At the
    # default 0 the box is the full 400 px window, whose corners reach 283 px
    # (14.1 m) against a 10 m sensor, so a frontier gets dropped as sensed while
    # still 12-14 m away. 59 puts the corner at 199 px, about 9.97 m.
    EARLY_STOP_MARGIN_PX = 59
    # PREEMPT_MIN_SCORE defaults to coverage_thr (5e-4). Measured over 257 nav
    # passes, max_score is ~0.00027 when no component survives the gates and
    # ~0.0143 when one does, so 5e-3 sits between the two populations.
    PREEMPT_MIN_SCORE = 0.005


__all__ = ["ExplorationConfig"]
