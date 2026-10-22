---
hide:
  - toc
---

<div class="ix-home" markdown>

<div class="ix-hero" markdown>

<p class="ix-eyebrow">IntelX Docs</p>

# How much new space would I gain by standing there?

<p class="ix-lead">IntelX answers that question for every nearby goal with one lightweight U-Net, and explores unseen indoor spaces on the junctions it reads from the answer.</p>

These pages cover installing IntelX, running it and every baseline in one simulator, reproducing the benchmark, and training the coverage-gain regressor.

<div class="ix-actions"><a class="ix-primary" href="getting-started/">Get Started <span aria-hidden="true">↗</span></a><a class="ix-secondary" href="benchmark/">Run the Benchmark <span aria-hidden="true">→</span></a><a class="ix-secondary" href="https://irvlutd.github.io/IntelX/">Project Page <span aria-hidden="true">→</span></a></div>

</div>

<figure class="ix-figure"><img src="assets/intelx-overview.gif" width="1440" height="612" alt="IntelX overview: a robot in an unseen 2D environment builds its map from a 220-degree laser scan; a robot-centred 20 m window of that map goes through a U-Net that returns a coverage-gain heatmap; the heatmap's angular components are candidate directions, and multi-peaked structure marks a junction with branches; the local and global planner commits to those branches, falls back to frontier centroids when it is stuck, and explores the whole floor along a short path."><figcaption>From the robot's scan and local window to the predicted coverage gain, its candidate directions and junctions, and the explored floor.</figcaption></figure>

## Try It Now

```bash
git clone https://github.com/IRVLUTD/IntelX.git
cd IntelX
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m intelx.cli.run_sweep_all --selftest      # the benchmark's self-test, no GPU
```

The self-test runs the benchmark's own helpers, from spawn sampling and the coverage metrics to task splitting and CSV writing, with no simulator, checkpoint or GPU, so it is the fast way to check an installation. To watch IntelX explore, add the checkpoint and follow [Get Started](getting-started.md).

## What the Paper Found

<div class="ix-stats">
<div><span class="ix-stat">&gt;20<small>%</small></span><span>Shorter Path to 99% Coverage Than Every Baseline That Finishes the Larger Maps</span></div>
<div><span class="ix-stat">8.1<small> ms</small></span><span>Per 20 m Coverage-Gain Field, About 120 FPS</span></div>
<div><span class="ix-stat">10<sup>6</sup></span><span>Training Samples, Each Labelled by a Simulation Oracle</span></div>
<div><span class="ix-stat">8</span><span>Baselines in One Simulator, Run From the Same Spawns</span></div>
</div>

<p class="ix-note">From the paper: the larger held-out maps are Medium Office 6 and Large Office 7 (Table 3), and the speed is Table 2's.</p>

<figure class="ix-figure"><img src="assets/results-large-office7.gif" width="907" height="1062" loading="lazy" alt="All nine methods exploring KTH Large Office 7 from the same spawn, replayed side by side and synchronized by distance travelled, with their coverage curves below. IntelX reaches 99% coverage at 1,687 m; IG-Hector, UPEN, MapEx and Cost-Utility reach it later, and Random Frontier, Nearest Frontier, Geometric IG and Frontier (AutoX) stop short."><figcaption>All nine methods on KTH Large Office 7 from the same spawn, synchronized by distance travelled. The <a href="https://irvlutd.github.io/IntelX/#results">project page</a> replays the other maps too.</figcaption></figure>

## How It Fits Together

Every method runs in one 2D simulator that casts its laser rays in CUDA kernels. IntelX's regressor, a U-Net with a ResNet-34 encoder, predicts over a robot-centred 20 m window how much new area the sensor would gain from each nearby goal. The planner reads angular components of that field as candidate directions, reads multi-peaked structure as a junction, and commits to branches weakest first. The global planner keeps each junction's branches and, when the window offers nothing, heads for the nearest active branch or frontier. The benchmark draws one spawn per map and seed and runs every method from it.

| Key | Method | Notes |
|:--|:--|:--|
| `ours` | IntelX | the coverage-gain field, the local and the global planner |
| `local_only` | IntelX, Local Only | the ablation without the global planner |
| `random` `nearest` `cost_utility` `geom_ig` | Random Frontier, Nearest Frontier, Cost-Utility, Geometric IG | geometry only |
| `frontier` | Frontier (AutoX) | our previous work, a dynamic-window frontier explorer |
| `mapex` `upen` `hectoraug` | MapEx, UPEN, IG-Hector | need [MapEx](https://github.com/castacks/MapEx) and its [service](mapex-service.md) |

## Where to Go Next

<div class="ix-routes">
<a href="getting-started/"><span class="ix-route-number">01</span><span><strong>Get Started</strong><small>Install, add the checkpoint and the maps, and run IntelX.</small></span><span aria-hidden="true">↗</span></a>
<a href="running/"><span class="ix-route-number">02</span><span><strong>Running IntelX</strong><small>One episode, any single method, the settings and the paths.</small></span><span aria-hidden="true">↗</span></a>
<a href="benchmark/"><span class="ix-route-number">03</span><span><strong>The Benchmark</strong><small>The paired protocol, every baseline and the MapEx service.</small></span><span aria-hidden="true">↗</span></a>
<a href="training/"><span class="ix-route-number">04</span><span><strong>Training the Regressor</strong><small>Collect coverage data, train the U-Net and evaluate it.</small></span><span aria-hidden="true">↗</span></a>
<a href="code/"><span class="ix-route-number">05</span><span><strong>Code Map</strong><small>The layout, every module and the tests.</small></span><span aria-hidden="true">↗</span></a>
<a href="citation/"><span class="ix-route-number">06</span><span><strong>Citation and Licenses</strong><small>The citation, coming with the arXiv paper, and the licenses of what IntelX builds on.</small></span><span aria-hidden="true">↗</span></a>
</div>

</div>
