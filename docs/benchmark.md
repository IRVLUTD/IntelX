# The Benchmark

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
| `frontier` | Frontier (AutoX), our previous work, a dynamic-window frontier explorer |
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
reimplementation of it, so it runs inside MapEx's environment.
[MapEx](https://github.com/castacks/MapEx) is linked at `third_party/MapEx` as a git submodule, pinned to
commit `53636bd`. Fetch it, install its `lama` conda environment by their
instructions, and start the service from the repository root.

```bash
git submodule update --init third_party/MapEx
conda activate lama
python services/mapex_service.py --mapex-root third_party/MapEx --port 5555
```

The [MapEx Service](mapex-service.md) page has the other ways to run it and the
wire protocol.

Two sensing regimes are defined and the mapping between them is deliberate.
`intelx` is 220 degrees over 662 beams and is used by every geometry-only
method, the reference, and ours. `mapex` is 360 degrees over 2500 beams and is
used by the three prediction baselines, which are evaluated under the sensing
they were trained for.
