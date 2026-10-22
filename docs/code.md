# Code Map

## Layout

```text
src/intelx/
  config.py            ExplorationConfig, every knob a run reads
  paths.py             map, weight and result roots, all relative
  common.py            RobotState, the model loader, transforms, line of
                       sight, doorway detection, A* with clearance
  policy_ablation.py   PlanningMixin, the local and global policy, with the
                       ablation switches. The one the explorer composes
  policy.py            the same policy without the switches, unused
  simulation.py        SimulationMixin, the state machine and map updates
  visualization.py     drawing and video
  explorer.py          composes the three mixins, runs an episode
  viz/                 style.py, live.py, showcase.py: the live view and the
                       sweep's showcase, in the look of the paper's videos
  metrics.py           P90/P95/P99, AUC, the two efficiencies
  cli/                 collect_data.py, train.py, evaluate.py,
                       run_intelx.py, run_method.py, run_sweep_all.py
  data/                driver.py, ring_collector.py, dataset.py,
                       collector.py
  sim/                 simulator.py, simulator_astar.py
  planners/            global_priority.py, rrt.py, rrt_star.py
  baselines/           simple.py, tier1.py, frontier.py
  models/              unet.py
  probes/              astar.py, tier1.py
tests/                 layout, paths and hygiene checks
scripts/               checkpoint tools
data/maps/             occupancy maps, see data/README.md
data/train_metadata.csv  the training maps and their shows, Table 1
data/coverage/         the generated training set, see data/README.md
weights/               checkpoints, see weights/README.md
results/               run output, ignored by git
services/              mapex_service.py, the bridge to MapEx
third_party/MapEx/     MapEx, a git submodule, for the three prediction baselines
docs/                  this documentation, built with MkDocs (mkdocs.yml)
docs/assets/           the wordmark, the figures and the README's images
.github/workflows/     the tests, run on every push, the documentation site, and
                       releases from version tags (notes in .github/release-notes/)
CITATION.cff           how to cite this repository
```

## Modules

| Module | What It Holds |
|---|---|
| `intelx.config` | `ExplorationConfig`. Nothing here is policy |
| `intelx.paths` | `map_path`, `dataset_path`, `weights_path`, `results_path`, and the environment overrides |
| `intelx.common` | `RobotState`, `create_model_for_inference`, coordinate transforms, line of sight, doorway detection, `astar_with_clearance`. Imports nothing local |
| `intelx.policy_ablation` | `PlanningMixin`, whose `_prediction_worker` is the local and global policy, with the ablation switches (`BRANCH_ORDER`, `LOCAL_FIELD`, `LOCAL_ONLY_MODE`) exposed. `intelx.explorer` composes it, so this is the file to edit |
| `intelx.policy` | the same policy without the ablation switches. No entry point imports it |
| `intelx.simulation` | `SimulationMixin`, the state machine, movement and map updates |
| `intelx.visualization` | `VisualizationMixin`, drawing and video |
| `intelx.explorer` | `UpdatedGoalBasedAutonomousExplorer`, the three mixins composed, and one episode |
| `intelx.models.unet` | `create_unet_resnet34`, `create_unet_resnet18`, the coverage-gain regressor |
| `intelx.metrics` | `METRIC_COLUMNS` and `compute_all`, which the benchmark scores with |
| `intelx.sim.simulator` | the 2D simulator and its CUDA ray casting |
| `intelx.sim.simulator_astar` | an older copy of the simulator. Only its `optimized_astar_path_planner` is used, by the policy and the baselines |
| `intelx.planners` | `global_priority` for junction and branch bookkeeping, `rrt` and `rrt_star` for UPEN |
| `intelx.baselines` | `simple` for the four geometry-only variants, `tier1` for the three prediction methods, `frontier` for our previous work, the dynamic-window frontier explorer |
| `intelx.probes` | timing checks, not used by the benchmark |
| `intelx.data.driver` | spawn sampling and the coverage shows for one map |
| `intelx.data.ring_collector` | the persistent GPU workers that run a show and record what it observes. The controller initialises them once and passes `skip_worker_init=True`; `driver` lets each pool process initialise its own, which the initialiser makes a no-op once they exist |
| `intelx.data.dataset` | `HybridChunkedDataset`, the chunked reader training and evaluation load through |
| `intelx.data.collector` | the CSV logger used during collection |
| `intelx.cli.collect_data` | the controller that runs collection across every map |
| `intelx.cli.train` | the training loop, Accelerate, checkpointing and curves |
| `intelx.cli.evaluate` | prediction overlays and inference timings for a checkpoint |
| `intelx.cli.run_method` | one run of any method on any map, set up as one cell of the benchmark, drawn live |
| `intelx.viz.live` | `LiveView` and `LivePlayer`, the live view of one run for every method |
| `intelx.viz.showcase` | every method on one map, side by side, from a sweep's curves |
| `intelx.viz.style` | the palette, fonts and glyphs the videos share |

`explorer.py` composes `PlanningMixin` from `policy_ablation.py`,
`SimulationMixin` and `VisualizationMixin`. Policy changes go in
`policy_ablation.py`. Nothing in `cli/` is policy.

## Tests

```bash
python -m pytest tests
```

The three files read the source rather than importing it, so none of them
needs torch, CUDA or a checkpoint.

| File | What It Checks |
|---|---|
| `test_layout.py` | every file parses, every `intelx.*` import names a module that is in the tree, no flat import survived, every package has an `__init__` |
| `test_paths.py` | the four roots resolve inside the repository, an environment variable redirects each one, an absolute path is still honoured |
| `test_hygiene.py` | no file carries a home directory, a tracker entity or a routable address, no map path is absolute, and nothing writes the machine's identity into a file it produces |

The hygiene file also checks itself. Each of its patterns is run against a
known-bad sample, so a regex that silently matches nothing fails rather than
passing everything, and the file count is checked against the tree, so a skip
rule that quietly excluded a package fails too.
