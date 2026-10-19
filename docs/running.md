# Running IntelX

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

Then put the trained checkpoint in `weights/`, see the [README](../README.md#getting-started).

## One Episode

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
| `sim['MAP_PGM']` | `map_path("test/small_office3.pgm")` | the occupancy map to explore |
| `HEADLESS` | `False` | `True` runs with no window, much faster |
| `VIEW_STYLE` / `VIEW_FPS` | `'paper'` / `10` | the live view and how often it is drawn; `'debug'` gives the older three-panel dashboard |
| `TURBO_MODE` | `False` | `True` drops the artificial sleeps. `FIXED_DT` keeps the map identical, and the benchmark pins it on |
| `MAP_UPDATE_INTERVAL` | `5` | scan every N steps. Leave it at 5 for any reported run, it changes the coverage metric |
| `FIXED_DT` | `0.05` | pinned physics step, 20 Hz. `None` restores wall-clock timing |
| `LOCAL_ONLY_MODE` | `False` | `True` disables junctions and the global planner, the Section V ablation |
| `WIN_M` / `LASER_SCAN_R` | `20` / `10` | prediction window and sensor range, in metres |

## Any Single Method

```bash
python -m intelx.cli.run_method --list                     # or: intelx-method
python -m intelx.cli.run_method --method nearest --map ood/aws_maps/hospital.pgm
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

## Paths

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
