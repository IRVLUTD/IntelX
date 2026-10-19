# Training the Regressor

Four steps, in order. The first is a long multi-GPU job, so read the
constants at the top of `collect_data.py` before starting it. With a collected
coverage dataset, start at `train`.

```bash
python -m intelx.cli.collect_data    # coverage data, over your maps
<CHUNKING_SCRIPT>                    # the collected shows, as training chunks
python -m intelx.cli.train           # the U-Net
python -m intelx.cli.evaluate        # what it predicts, side by side
```

`collect_data` runs the 18 training maps listed in `data/train_metadata.csv`,
each with the number of shows Table 1 of the paper gives it, in proportion to
its complexity score. It reads the maps from `train/` under the
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
