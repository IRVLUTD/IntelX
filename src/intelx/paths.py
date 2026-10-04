"""Where maps, weights and results live.

No file in this tree stores an absolute path. Everything is resolved from the
repository root, and each root can be redirected with an environment variable,
which is how a run on a shared machine reaches a copy of the data without
editing a single source file.

    INTELX_MAPS_ROOT      default <repo>/data/maps
    INTELX_DATASET_ROOT   default <repo>/data/coverage
    INTELX_WEIGHTS_ROOT   default <repo>/weights
    INTELX_RESULTS_ROOT   default <repo>/results

The three helpers return `str`, not `Path`, because the simulator and the
loaders pass these straight to `cv2.imread` and `open`.
"""
from __future__ import annotations

import os
from pathlib import Path

# <repo>/src/intelx/paths.py -> <repo>
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent


def _root(env: str, default: str) -> Path:
    raw = os.environ.get(env)
    return Path(raw).expanduser().resolve() if raw else REPO_ROOT / default


MAPS_ROOT = _root("INTELX_MAPS_ROOT", "data/maps")
DATASET_ROOT = _root("INTELX_DATASET_ROOT", "data/coverage")
WEIGHTS_ROOT = _root("INTELX_WEIGHTS_ROOT", "weights")
RESULTS_ROOT = _root("INTELX_RESULTS_ROOT", "results")


def _under(root: Path, rel: str | os.PathLike) -> str:
    """Resolve `rel` under `root`. An absolute `rel`, or one that already
    resolves from the working directory, is honoured as given, so an existing
    local layout keeps working without any environment variable."""
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    here = root / p
    if here.exists() or not p.exists():
        return str(here)
    return str(p.resolve())


def map_path(rel: str | os.PathLike) -> str:
    """An occupancy map, by its path under the maps root.

    >>> map_path("test_standardized/small_office3.pgm")  # doctest: +SKIP
    '/.../data/maps/test_standardized/small_office3.pgm'
    """
    return _under(MAPS_ROOT, rel)


def dataset_path(rel: str | os.PathLike = "") -> str:
    """Generated training data, by its path under the dataset root.

    This is what the collectors write and what training reads. It is separate
    from the maps root because a coverage chunk is not a floor plan, and the
    two are usually on different disks."""
    return _under(DATASET_ROOT, rel) if rel else str(DATASET_ROOT)


def weights_path(rel: str | os.PathLike) -> str:
    """A checkpoint, by its file name under the weights root."""
    return _under(WEIGHTS_ROOT, rel)


def results_path(rel: str | os.PathLike = "") -> str:
    """A run directory under the results root, created if it is missing."""
    p = Path(_under(RESULTS_ROOT, rel)) if rel else RESULTS_ROOT
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


__all__ = ["REPO_ROOT", "MAPS_ROOT", "DATASET_ROOT", "WEIGHTS_ROOT",
           "RESULTS_ROOT", "map_path", "dataset_path", "weights_path",
           "results_path"]
