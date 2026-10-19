# Prediction-Baseline Service

`mapex_service.py` serves MapEx, UPEN and IG-Hector planning decisions over a
local socket, so the sweep can use them as baselines from a different python
environment. Only the `mapex`, `upen` and `hectoraug` methods need it. Every
other method, `ours` included, runs without it.

## Why It Runs Inside MapEx

It calls MapEx's own planner rather than a reimplementation of it, so it has
to run inside MapEx's environment, with MapEx's modules importable and the
working directory set to their `scripts/`. Their stack is an older python with
an old torch, LaMa and range_libc, which cannot share a process with this one.
Calling their code unmodified is the point. Reimplementing their
information-gain maths is how a baseline ends up accidentally crippled, so the
only thing this file contains is format and coordinate conversion.

## Installation

[MapEx](https://github.com/castacks/MapEx) is linked at `third_party/MapEx` as a git submodule, pinned to
commit `53636bd`.

1. Fetch it: `git submodule update --init third_party/MapEx`.
2. Install its `lama` conda environment by their instructions.
3. Start the service from the repository root.

```bash
conda activate lama
python services/mapex_service.py --mapex-root third_party/MapEx --port 5555
```

`--mapex-root`, or `MAPEX_ROOT`, can point at any MapEx checkout. Without
either, the MapEx root is taken to be the parent of the directory this file is
in, which is right when the file is copied into MapEx's own
[`scripts/`](https://github.com/castacks/MapEx/tree/main/scripts) directory,
beside `explore.py`. `--device` selects the GPU.

Check it is alive, from any environment.

```bash
python services/mapex_service.py --selftest-client --port 5555
```

The sweep expects it on `127.0.0.1:5555`, or on each port in
`SWEEP.MAPEX_SERVICE_PORTS` when several workers run in parallel.

## The Wire Protocol

The client is `MapExClient` in `src/intelx/baselines/tier1.py`, and it is the
specification. A message is a 4-byte big-endian length followed by a pickle.

```python
data = pickle.dumps(obj, protocol=PROTO)
sock.sendall(struct.pack(">I", len(data)) + data)
```

Two requests arrive.

| Request | Reply |
|---|---|
| `{"cmd": "ping"}` | anything truthy |
| a plan request, below | a plan reply, below |

```python
# request
{
  "mode":      "visvarprob" | "upen" | "hectoraug",
  "obs_map":   np.float32 array, 0.0 known free, 0.5 unknown, 1.0 known occupied,
  "cur_pose":  [row, col] in their frame,
  "pose_list": the last 200 poses, same frame,
  "cfg": {"pixel_per_meter": int,
          "pred_laser_range_m": float,
          "pred_num_laser": int,
          "upen_config": {"reach_horizon": ..., "upen_mode": ...}},
}

# reply
{
  "ok":             bool,
  "goal":           [row, col] or None,
  "error":          str, when ok is False,
  "path":           list of [row, col], optional,
  "path_truncated": bool, optional,
  "service_s":      float, seconds spent planning, optional,
  "n_frontiers":    int, optional,
  "fallback":       None, or the reason a substitute goal was returned,
}
```

The client works in MapEx's resolution, 0.1 m per pixel, unpadded. The 500
pixel border their raycaster needs is applied and removed inside the service,
so the client never sees it. `occ_explored_to_obs_map` in the client converts
our occupancy and explored arrays into their convention. Downsampling takes a
max over each block so a one-pixel wall survives, which stops raycasts leaking
through it and flattering the baseline's information gain.

`fallback` is `None` when the goal really came from the named planner. For
UPEN it names the reason the nearest frontier was substituted instead, so a
run can never be silently scored as UPEN when it was not. Treat any run with a
non-null `fallback` as unreportable for that method.

A reply with `ok: False` is recorded as a failure rather than crashing the
sweep, and the client reconnects on its own.

## Checking the Client Without the Service

`python -m intelx.baselines.tier1` starts a mock service on a random port and
exercises the client against it, including the unreachable-service path.
