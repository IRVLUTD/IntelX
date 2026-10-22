# Get Started

IntelX needs Python **3.11** and an NVIDIA GPU. The simulator casts its rays in
numba `@cuda.jit` kernels, so the GPU is required. Inference falls back to CPU,
the simulator does not.

## 1. Install

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

## 2. Add the Checkpoint

Download [`checkpoint_best.pth`](https://utdallas.box.com/s/2tuurbusuawkowwe9rjnmsbkhtgm6b5o)
(293.4 MB) and put it in `weights/`. Its checksum and what it holds are on the
[Checkpoint](weights.md) page.

```bash
sha256sum weights/checkpoint_best.pth      # macOS: shasum -a 256
```

## 3. Get the Maps

Three maps ship with the repository, one of each kind. The
[Box folder](https://utdallas.box.com/s/stgu95mdcf3p7tgp5z5a53pcxiuwrplu) holds
every test, out-of-distribution and real map; the contents of its `maps/`
folder go in `data/maps/`. The 18 training maps and the coverage dataset (about
1 TB) are not public yet; `collect_data` regenerates the dataset from the
training maps. [Maps and Data](data.md) has the layout.

## 4. Run IntelX

```bash
python -m intelx.cli.run_intelx                    # IntelX on Small Office 3, drawn live
python -m intelx.cli.run_method --method nearest --map ood/aws_maps/hospital.pgm
python -m intelx.cli.run_sweep_all --selftest      # the benchmark's self-test, no GPU
```

The first explores Small Office 3 until 99% of the free space is covered and
writes the video to `results/run_intelx/`. [Running IntelX](running.md) covers
the settings, any single method and the paths.

## 5. Check the Tree

```bash
python -m pytest tests
```

The tests read the source rather than importing it, so none of them needs
torch, CUDA or a checkpoint. The [Code Map](code.md#tests) lists what each one
checks.

## Next

- [The Benchmark](benchmark.md): every method from the same spawns, and the MapEx service the prediction baselines need.
- [Training the Regressor](training.md): collect coverage data, train the U-Net and evaluate it.
