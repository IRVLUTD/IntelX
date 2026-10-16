<h1 align="center">IntelX: Autonomous Indoor Exploration via Local Coverage Gain Prediction and Structure-Aware Planning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-Required-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/ICRA%202027-Under%20Review-00629B?style=flat-square&logo=ieee&logoColor=white" alt="ICRA 2027">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

<p align="center">
  <img src="assets/intelx-wordmark.svg" alt="IntelX" width="470">
</p>

<p align="center">
  <a href="https://sauravdosi.com/"><strong>Saurav Dosi</strong></a>
  ·
  <a href="https://www.linkedin.com/in/itay-kadosh/"><strong>Itay Kadosh</strong></a>
  ·
  <a href="https://yuxng.github.io/"><strong>Yu Xiang</strong></a>
</p>

<p align="center">
  <a href="https://labs.utdallas.edu/irvl/">Intelligent Robotics and Vision Lab</a>, Department of Computer Science, The University of Texas at Dallas
</p>

<h3 align="center"><a href="<ARXIV_LINK>">arXiv</a> | <a href="<PROJECT_PAGE_LINK>">Project Page</a> | <a href="<VIDEO_LINK>">Video</a></h3>

One U-Net regresses, over a robot-frame occupancy image, how much new area the
sensor would gain from each nearby goal. The planner reads angular components
of that field as candidate directions, reads multi-peaked structure as a
junction, and commits to branches weakest first.

This repository holds the simulator, the coverage-gain regressor, the policy,
and every baseline the paper compares against.

---

## Install

Python **3.11** and an NVIDIA GPU. The simulator casts its rays in numba
`@cuda.jit` kernels, so the GPU is required. Inference falls back to CPU, the
simulator does not.

```bash
git clone https://github.com/IRVLUTD/IntelX.git
cd IntelX
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is the frozen environment this release was tested in. It
takes torch 2.11 and torchvision 0.26 in their CUDA 12.8 builds from the
PyTorch index, which run on every GPU from Turing (`sm_75`) to Blackwell
(`sm_120`), and `numba-cuda`, which brings the CUDA compiler the kernels are
built with, so no system CUDA toolkit is needed. On an older GPU, install a
torch build that supports it first, then `pip install -e .`, which resolves
the rest from `pyproject.toml`, the loose set of packages this tree imports.

Then put the trained checkpoint in `weights/`, see [Downloads](#downloads).

## Downloads

| Item | Link | Goes In |
|---|---|---|
| Trained Checkpoint | [`checkpoint_best.pth`](https://utdallas.box.com/s/2tuurbusuawkowwe9rjnmsbkhtgm6b5o), 293.4 MB | `weights/`, see [`weights/README.md`](weights/README.md) |
| Occupancy Maps | [Maps](<MAPS_LINK>) | `data/maps/`, see [`data/README.md`](data/README.md) |
| Coverage Dataset | [Dataset](<DATASET_LINK>) | `data/coverage/`, see [`data/README.md`](data/README.md) |

Three maps are included, one of each kind.

| Map | Kind |
|---|---|
| `data/maps/test_standardized/small_office3.pgm` | a held-out KTH floor plan |
| `data/maps/ood_test_standardized/aws_maps/hospital.pgm` | an out-of-distribution AWS RoboMaker world |
| `data/maps/real/real_slam_corridor.pgm` | the real corridor, from a SLAM run |

---

## Run

### Training the Regressor

Four steps, in order. The first is a long multi-GPU job, so read the
constants at the top of `collect_data.py` before starting it. With the coverage
dataset from [Downloads](#downloads), start at `train`.

```bash
python -m intelx.cli.collect_data    # coverage data, over your maps
<CHUNKING_SCRIPT>                    # the collected shows, as training chunks
python -m intelx.cli.train           # the U-Net
python -m intelx.cli.evaluate        # what it predicts, side by side
```

`collect_data` runs the 18 training maps listed in `data/train_metadata.csv`,
each with the number of shows Table 1 of the paper gives it, in proportion to
its complexity score. It reads the maps from `train_standardized/` under the
maps root, samples spawns, runs the exploration shows that record how much new
area each goal would gain, and writes the result under `INTELX_DATASET_ROOT`.
`intelx.data.driver` does one map at a time and is what the controller drives.
`<CHUNKING_SCRIPT>` packs the collected shows into `train_chunks/`. `train`
reads those chunks, at batch size 32 for up to 300 epochs, keeps the checkpoint with the best validation R², and writes each
run, with the settings it used, to `weights/runs/<name>`. `evaluate` loads a
checkpoint and writes prediction overlays and inference timings to
`results/evaluation_large`.

Everything each step reads and writes is a class attribute at the top of its
file. None of the three takes command-line arguments.

### One Episode

```bash
python -m intelx.cli.run_intelx          # or: intelx-run
```

It explores until 99% of the free space is covered, drawing the run live the
way the paper's videos do: the predicted coverage-gain field in the world
frame, its ranked components, the junctions and their branches, the frontier
fallback, the goal and the decision in force. It writes the video,
`final_trajectory.png` and `prediction_debug.log` to `results/run_intelx/`.
Every knob lives in `ExplorationConfig` in `src/intelx/config.py`. The ones
that matter.

| Setting | Default | What It Does |
|---|---|---|
| `MODEL_CHECKPOINT` | `weights_path("checkpoint_best.pth")` | the trained U-Net |
| `MODEL_TYPE` | `resnet34` | must match the checkpoint |
| `sim['MAP_PGM']` | `map_path("test_standardized/small_office3.pgm")` | the occupancy map to explore |
| `HEADLESS` | `False` | `True` runs with no window, much faster |
| `VIEW_STYLE` / `VIEW_FPS` | `'paper'` / `10` | the live view and how often it is drawn; `'debug'` gives the older three-panel dashboard |
| `TURBO_MODE` | `False` | `True` drops the artificial sleeps. `FIXED_DT` keeps the map identical, and the benchmark pins it on |
| `MAP_UPDATE_INTERVAL` | `5` | scan every N steps. Leave it at 5 for any reported run, it changes the coverage metric |
| `FIXED_DT` | `0.05` | pinned physics step, 20 Hz. `None` restores wall-clock timing |
| `LOCAL_ONLY_MODE` | `False` | `True` disables junctions and the global planner, the Section V ablation |
| `WIN_M` / `LASER_SCAN_R` | `20` / `10` | prediction window and sensor range, in metres |

### Any Single Method

```bash
python -m intelx.cli.run_method --list                     # or: intelx-method
python -m intelx.cli.run_method --method nearest --map ood_test_standardized/aws_maps/hospital.pgm
python -m intelx.cli.run_method --method ours --set BRANCH_ORDER=strongest
```

One run of any method on any map, drawn live in the same view. It is set up
exactly as one cell of the benchmark: the same pinned settings, the method's own
sensor, and the spawn the benchmark draws for that map and `--seed`, so what you
watch is what a row of `results.csv` measures. It writes
`results/run_method/<map>_s<seed>_<method>.mp4`, the final frame beside it and
the run's curves. `--headless` drops the window and `--minutes` caps the run;
Esc ends it early. `--set KEY=VALUE` overrides any setting, which is how the
ablations run: `BRANCH_ORDER=strongest` commits to the strongest branch first,
`LOCAL_FIELD=geom_ig` replaces the learned field with a geometric one, and
`--method local_only` drops the global planner.

### The Benchmark

```bash
python -m intelx.cli.run_sweep_all --selftest   # no simulator, model or GPU
python -m intelx.cli.run_sweep_all              # or: intelx-sweep
```

One spawn is drawn per `(map, seed)` and **every** method runs from it, so a
difference between methods is attributable to the policy rather than to the
start. Runs land in one `results.csv` beside a shared `spawns.json`.

Ten entries are registered, nine methods and one ablation.

| Key | In the Paper |
|---|---|
| `random` `nearest` `cost_utility` `geom_ig` | Random Frontier, Nearest Frontier, Cost-Utility, Geometric IG: the four geometry-only frontier variants |
| `frontier` | Frontier (AutoX), the deployed reference |
| `ours` | IntelX |
| `local_only` | IntelX, Local Only: the ablation with the local planner alone |
| `mapex` `upen` `hectoraug` | MapEx, UPEN, IG-Hector |

The last three need [MapEx](https://github.com/castacks/MapEx) installed and run behind a service, which
must be up first. Every other method, `ours` included, runs without it.

The geometry-only methods follow the rules of the papers they come from. L is
the length of the A* path to the frontier, and the gain is the number of
unobserved cells a 360-degree scan from the frontier would cover, cast on the
robot's own map, stopping at walls it has seen. All four choose among the
accessible, unvisited frontiers, as Yamauchi's frontier-based exploration does:
a frontier the robot made no progress toward is set aside for a while, and one
it has already reached is not chosen again (`VISITED_FRONTIER_M`).

| Key | Goal Chosen | Reference |
|---|---|---|
| `nearest` | the frontier with the shortest L | Yamauchi 1997 |
| `random` | uniformly among the five frontiers with the shortest L | |
| `cost_utility` | max gain · exp(−λL), λ = 0.2 per metre (`COST_UTILITY_LAMBDA`) | González-Baños and Latombe 2002 |
| `geom_ig` | max gain − αL, gain in m², α = 1 m² per metre (`GEOM_IG_ALPHA`) | Stachniss et al. 2005 |
| `frontier` | min 3L − size within 10 m, over every frontier when none is that close | AutoX |

`SUBMITTED_BEHAVIOUR` in `PIN_CONFIG` restores the behaviour the submitted
results were produced with. L there counts the waypoints of the post-processed
A* path rather than measuring its length, Cost-Utility is explore_lite's
3L − size and Geometric IG is the unobserved cells in a disk divided by L,
reached frontiers stay candidates, a pass with no frontier inside the window
ends the run when it finds no goal, and A* refuses to plan from a robot that
has slipped one pixel onto a wall. Every run
records the value it ran with in `results_meta.json`.

Each run is recorded as a video in the live view (`VIDEO_STYLE`), and every
sweep ends with a showcase per map, every method side by side on its own copy
of the map with the coverage curves beside them, synchronised by distance
travelled. Both land in `<run>/videos/`. To render a showcase again, or from
runs of `run_method`:

```bash
python -m intelx.viz.showcase results/<run_name>          # or: intelx-showcase
```

`services/mapex_service.py` calls MapEx's own planner rather than a
reimplementation of it, so it runs inside MapEx's environment. Install
[MapEx](https://github.com/castacks/MapEx) and its `lama` conda environment by their instructions, copy the
file into MapEx's own [`scripts/`](https://github.com/castacks/MapEx/tree/main/scripts) directory beside `explore.py`, and start
it from there.

```bash
conda activate lama
cd /path/to/MapEx/scripts
python mapex_service.py --port 5555
```

[`services/README.md`](services/README.md) has the placement steps and the
wire protocol.

Two sensing regimes are defined and the mapping between them is deliberate.
`intelx` is 220 degrees over 662 beams and is used by every geometry-only
method, the reference, and ours. `mapex` is 360 degrees over 2500 beams and is
used by the three prediction baselines, which are evaluated under the sensing
they were trained for.

### Paths

Nothing in this tree stores an absolute path. Maps, weights and results resolve
from the repository root, and each root takes an environment variable, which is
how a run on a shared machine reaches a copy of the data without editing a
single source file.

```bash
export INTELX_MAPS_ROOT=/path/to/maps          # default <repo>/data/maps
export INTELX_DATASET_ROOT=/path/to/coverage   # default <repo>/data/coverage
export INTELX_WEIGHTS_ROOT=/path/to/weights    # default <repo>/weights
export INTELX_RESULTS_ROOT=/path/to/results    # default <repo>/results
```

### Tests

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

---

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
services/              mapex_service.py, to be copied into MapEx
assets/                the wordmark
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
| `intelx.baselines` | `simple` for the four geometry-only variants, `tier1` for the three prediction methods, `frontier` for the deployed reference |
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

---

## Citation

```bibtex
<BIBTEX>
```

---

## License

MIT, see [LICENSE](LICENSE).

`src/intelx/planners/rrt.py` and `src/intelx/planners/rrt_star.py` are adapted
from PythonRobotics, MIT licensed. The original notice stays in each file.
