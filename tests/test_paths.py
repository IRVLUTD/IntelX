"""The path resolver, which is what replaced the absolute paths."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from intelx import paths  # noqa: E402


def test_repo_root_is_this_repository():
    assert os.path.isdir(os.path.join(paths.REPO_ROOT, "src", "intelx"))
    assert os.path.isfile(os.path.join(paths.REPO_ROOT, "pyproject.toml"))


def test_defaults_sit_under_the_repository():
    for root in (paths.MAPS_ROOT, paths.WEIGHTS_ROOT, paths.RESULTS_ROOT):
        assert str(root).startswith(str(paths.REPO_ROOT))


def test_map_path_is_relative_to_the_maps_root():
    p = paths.map_path("test_standardized/small_office3.pgm")
    assert p == os.path.join(str(paths.MAPS_ROOT),
                             "test_standardized", "small_office3.pgm")
    assert isinstance(p, str)


def test_weights_path_is_relative_to_the_weights_root():
    assert paths.weights_path("intelx_unet_resnet34.pt") == os.path.join(
        str(paths.WEIGHTS_ROOT), "intelx_unet_resnet34.pt")


def test_an_absolute_path_is_honoured_as_given():
    assert paths.map_path("/tmp/elsewhere.pgm") == "/tmp/elsewhere.pgm"


def test_the_env_override_redirects_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELX_MAPS_ROOT", str(tmp_path))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.map_path("a.pgm") == str(tmp_path / "a.pgm")
    finally:
        monkeypatch.delenv("INTELX_MAPS_ROOT")
        importlib.reload(paths)


def test_results_root_is_created_on_demand(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELX_RESULTS_ROOT", str(tmp_path / "r"))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        d = reloaded.results_path("run1")
        assert os.path.isdir(d)
    finally:
        monkeypatch.delenv("INTELX_RESULTS_ROOT")
        importlib.reload(paths)
