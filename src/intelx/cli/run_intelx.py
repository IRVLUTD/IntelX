"""
intelx.cli.run_intelx
=====================
One exploration episode, end to end.

    python -m intelx.cli.run_intelx

This file is the boot sequence and nothing else. The knobs live in
`intelx.config`; the decision of where to go next lives in
`intelx.policy_ablation`.

Module map
----------
    intelx.config          ExplorationConfig, every knob a run reads
    intelx.paths           map, weight and result roots, all relative
    intelx.common          RobotState, the model loader, coordinate
                           transforms, line of sight, doorway detection,
                           astar_with_clearance. Imports nothing local.
    intelx.policy_ablation PlanningMixin -> _prediction_worker, the local
                           and global policy, with the ablation switches.
                           This is the policy file the explorer composes.
    intelx.policy          the same policy without the switches, unused
    intelx.simulation      SimulationMixin -> _updated_simulation_worker,
                           the state machine, movement and map updates
    intelx.visualization   VisualizationMixin, drawing and video
    intelx.explorer        UpdatedGoalBasedAutonomousExplorer, which
                           composes the three mixins and runs an episode
    intelx.sim.simulator   the 2D simulator and its CUDA ray casting
    intelx.planners        junction and branch bookkeeping, RRT, RRT*
    intelx.baselines       every method the paper compares against

UpdatedGoalBasedAutonomousExplorer keeps its (simulator, model, config)
constructor and its run_exploration_episode(), so anything that imported it
before still works.
"""

import argparse
import os

import torch

from intelx.common import apply_submitted_behaviour, create_model_for_inference
from intelx.explorer import UpdatedGoalBasedAutonomousExplorer
from intelx.sim.simulator import ExplorationSimulator

from intelx.config import ExplorationConfig


# --- Main Execution Block --------------------------------------------------
def main(argv=None):
    """Run one episode. Returns a process exit status."""
    argparse.ArgumentParser(
        prog="python -m intelx.cli.run_intelx",
        description="One IntelX episode. Every knob lives in ExplorationConfig "
                    "in intelx/config.py; for other methods, maps or seeds use "
                    "intelx.cli.run_method.").parse_args(argv)
    print("=" * 70)
    print("UPDATED GOAL-BASED AUTONOMOUS EXPLORATION SYSTEM")
    print("=" * 70)

    config = ExplorationConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    apply_submitted_behaviour(config)

    print("Loading model...")
    model = create_model_for_inference(
        config.MODEL_CHECKPOINT,
        config.device,
        model_type=config.MODEL_TYPE
    )
    print(f"Model loaded on {config.device}")

    print("Initializing simulator...")
    simulator = ExplorationSimulator(
        config=config.sim,
        initial_spawn_point=None,
        log_path=None,
        skip_initial_scan=False
    )

    print("Simulator initialized successfully")

    print("Creating updated goal-based autonomous explorer...")
    explorer = UpdatedGoalBasedAutonomousExplorer(simulator, model, config)
    print("Explorer created successfully")

    print("Starting updated goal-based exploration...")
    explorer.run_exploration_episode()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())