# Checkpoints

**Download:** [`checkpoint_best.pth`](<GOOGLE_DRIVE_LINK>), 293.4 MB. Put it in
this directory.

```text
weights/checkpoint_best.pth
sha256 4f82a3fa54a0349720b2b3c42ee812ea744d539060146fbd2be8ca16d6a69ed4
```

```bash
sha256sum weights/checkpoint_best.pth      # macOS: shasum -a 256
```

`ExplorationConfig.MODEL_CHECKPOINT`, the benchmark and `intelx.cli.evaluate`
all read `weights/checkpoint_best.pth` by default. To keep checkpoints
somewhere else, redirect the root rather than editing a source file.

```bash
export INTELX_WEIGHTS_ROOT=/path/to/your/checkpoints
```

## What It Is

| | |
|---|---|
| Architecture | U-Net with a ResNet-34 encoder, `intelx.models.unet` |
| Parameters | 24,423,009, plus 18,988 BatchNorm buffer values |
| Selection | epoch 31, validation R² 0.357 |
| Format | the training checkpoint as `intelx.cli.train` writes it |

It is the full training checkpoint, so it carries `model_state_dict` together
with `optimizer_state_dict`, `epoch`, `val_r2` and `val_loss`. The AdamW
moments are two thirds of its size. The model keys carry the `_orig_mod.`
prefix that `torch.compile` adds, which
`intelx.common.create_model_for_inference` strips on load. It contains tensors
and plain values only, so `torch.load(path, weights_only=True)` reads it.

## Tools

`scripts/inspect_checkpoint.py` lists everything a checkpoint carries besides
tensors, without executing its pickle.

```bash
python scripts/inspect_checkpoint.py weights/checkpoint_best.pth
```

`scripts/strip_checkpoint.py` writes a 49 MB weights-only FP16 copy, should you
want one. It keeps the thirteen BatchNorm `running_var` buffers that exceed the
FP16 range at FP32. Point `MODEL_CHECKPOINT` at the copy to use it.

```bash
python scripts/strip_checkpoint.py weights/checkpoint_best.pth weights/intelx_unet_resnet34_fp16.pt
```

Both need numpy and the standard library only.
