"""
intelx.cli.run_method
=====================
One run of any method on any map, drawn live the way the paper's videos are.

    python -m intelx.cli.run_method --method ours
    python -m intelx.cli.run_method --method nearest --map ood/aws_maps/hospital.pgm
    python -m intelx.cli.run_method --method ours --set BRANCH_ORDER=strongest
    python -m intelx.cli.run_method --list

The run is set up exactly as one cell of the benchmark (intelx.cli.run_sweep_all):
the same pinned configuration, the method's own sensor profile, and the spawn
the benchmark draws for that map and seed. What you watch is what a row of
results.csv measures.

It writes <results>/run_method/<map>_s<seed>_<method>.mp4, the final frame
beside it, and the run's curves npz in the benchmark's format. Esc or q in the
window ends the run early. MapEx, UPEN and IG-Hector need their service up first,
see services/README.md.
"""
import argparse
import ast
import json
import os
import sys
import time

import numpy as np

from intelx import metrics
from intelx.paths import map_path, results_path
from intelx.cli import run_sweep_all as sweep


def _parse_set(items):
    """KEY=VALUE pairs -> dict, VALUE read as a Python literal when it is one."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit("--set expects KEY=VALUE, got %r" % it)
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def _service_alive(config):
    from intelx.baselines.tier1 import MapExClient
    port = (sweep._service_ports(config) or [getattr(config, 'MAPEX_SERVICE_PORT', 5555)])[0]
    try:
        return bool(MapExClient(config.MAPEX_SERVICE_HOST, port, timeout=5.0).ping()), port
    except Exception:
        return False, port


def main(argv=None):
    from intelx.viz.style import method_label
    ap = argparse.ArgumentParser(
        prog="python -m intelx.cli.run_method",
        description="One run of any method on any map, drawn live.",
        epilog="Ablations of IntelX: --set BRANCH_ORDER=strongest, "
               "--set LOCAL_FIELD=geom_ig, or --method local_only.")
    ap.add_argument("--method", choices=sorted(sweep.MODE_REGISTRY))
    ap.add_argument("--map", default="test/small_office3.pgm",
                    help="under the maps root, or an absolute path")
    ap.add_argument("--seed", type=int, default=sweep.SWEEP.SEED_BASE,
                    help="picks the spawn, as in the benchmark")
    ap.add_argument("--minutes", type=float, default=60.0,
                    help="wall-clock limit (default 60)")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="how often the view is drawn, and the video's frame rate")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                    help="override a setting, e.g. BRANCH_ORDER=strongest")
    ap.add_argument("--out", default=None, help="default <results>/run_method")
    ap.add_argument("--list", action="store_true", help="list the methods and exit")
    args = ap.parse_args(argv)

    if args.list or not args.method:
        for m in sorted(sweep.MODE_REGISTRY):
            extra = "  (needs the MapEx service)" if m in sweep.TIER1_MODES else ""
            print("  %-13s %s%s" % (m, method_label(m), extra))
        return 0 if args.list else 2

    mode = args.method
    config = sweep.SWEEP
    sweep._pin_config(config)
    for k, v in _parse_set(args.set).items():
        setattr(config, k, v)
    from intelx.common import apply_submitted_behaviour
    apply_submitted_behaviour(config)

    mp = map_path(args.map)
    if not os.path.exists(mp):
        print("no such map: %s" % mp)
        return 1
    map_name = os.path.splitext(os.path.basename(mp))[0]
    if mode in sweep.TIER1_MODES:
        ok, port = _service_alive(config)
        if not ok:
            print("%s needs the MapEx service on %s:%s, and it is not answering. "
                  "See services/README.md." % (mode, config.MAPEX_SERVICE_HOST, port))
            return 1
        config.MAPEX_SERVICE_PORT = port

    sensor = sweep._mode_sensor_pairs(config, [mode])[0][1]
    sim_cfg = sweep.apply_sensor_profile(config, sensor)
    config.sim = sim_cfg
    res = sim_cfg.get('RESOLUTION', 0.05)

    sim = sweep.build_simulator(mp, config, sim_cfg)
    spawn = sweep.pick_spawn(sim.occ, args.seed, config.SPAWN_CLEARANCE_M / res)
    model = sweep._worker_model(config) if mode in sweep.NEEDS_MODEL else None
    exp = sweep.build_explorer(mode, sim, model, config, args.seed)
    sweep.force_spawn(exp, sim, spawn)

    out_dir = args.out or results_path("run_method")
    os.makedirs(out_dir, exist_ok=True)
    stem = "%s_s%d_%s" % (map_name, args.seed, mode)
    player = sweep.make_run_video(config, mode, sim, sim_cfg, map_name, args.seed,
                                  spawn, os.path.join(out_dir, stem + ".mp4"),
                                  show=not args.headless, fps=args.fps)
    print("%s on %s, seed %d, sensor %s. Esc or q in the window to stop."
          % (method_label(mode), map_name, args.seed, sensor))

    t0 = time.time()
    try:
        curve, traj, goals, stop = sweep.monitor_run(
            exp, config.COVERAGE_CHECKPOINTS, config.MAX_PATH_M, args.minutes * 60.0,
            min(config.POLL_S, 1.0 / args.fps), video=player,
            video_every_s=1.0 / args.fps, stall_idle_s=sweep.stall_window_for(config, mode))
    finally:
        player.close()
    wall_s = time.time() - t0

    free_area = sweep.map_free_area_m2(sim.occ, res)
    met = metrics.compute_all(curve, traj, goals, {
        'path_budget_m': config.MAX_PATH_M, 'free_area_m2': free_area,
        'target_pct': config.TARGET_PCT, 'wall_s': wall_s, 'stop_reason': stop,
        'resolution': res, 'checkpoints': config.COVERAGE_CHECKPOINTS})
    if curve:
        np.savez_compressed(
            os.path.join(out_dir, stem + ".npz"),
            curve=np.array(curve, np.float32), traj=np.array(traj, np.float32),
            goals=np.array(goals, np.float32),
            meta=np.array(json.dumps({
                "map": map_name, "map_path": mp, "seed": args.seed, "mode": mode,
                "sensor": sensor, "spawn": list(spawn), "resolution": res,
                "sensor_profile": dict(getattr(config, 'SENSOR_PROFILES', {}).get(sensor, {})),
                "free_area_m2": free_area, "wall_s": wall_s, "stop_reason": stop,
                "overrides": _parse_set(args.set)})))
    print("\n%s: %s at %.1f%% after %.0f m in %.0f s"
          % (method_label(mode), stop, met['final_coverage'], met['total_path_m'], wall_s))
    print("video  %s" % os.path.join(out_dir, stem + ".mp4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
