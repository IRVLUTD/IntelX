"""The repository is wired to itself.

These checks read the source rather than importing it, so they pass without
torch, without CUDA, and without the modules that have not been added yet.
They are what catches a move that left an import behind.
"""
import ast
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")


def py_files():
    for root, _, files in os.walk(SRC):
        for f in sorted(files):
            if f.endswith(".py"):
                yield os.path.join(root, f)


def module_of(path):
    rel = os.path.relpath(path, SRC)[:-3]
    return rel.replace(os.sep, ".").removesuffix(".__init__")


ALL = sorted(py_files())
MODULES = {module_of(p) for p in ALL}


@pytest.mark.parametrize("path", ALL, ids=module_of)
def test_parses(path):
    ast.parse(open(path, encoding="utf-8").read(), filename=path)


@pytest.mark.parametrize("path", ALL, ids=module_of)
def test_intra_repo_imports_resolve(path):
    """Every `intelx.*` import names a module that exists in the tree."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            mod = node.module
            if not mod.startswith("intelx"):
                continue
            if mod in MODULES:
                continue
            # `from intelx.baselines import simple` names the package
            assert any("%s.%s" % (mod, a.name) in MODULES for a in node.names), (
                "%s imports %s, which is not in the tree" % (path, mod))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("intelx"):
                    assert a.name in MODULES, (
                        "%s imports %s, which is not in the tree"
                        % (path, a.name))


@pytest.mark.parametrize("path", ALL, ids=module_of)
def test_no_flat_imports_survive(path):
    """Nothing imports by the old flat module names or through `main`."""
    stale = {"main", "baseline_explorer",
             "baselines_simple", "baselines_tier1", "data_collector",
             "exploration_simulator4", "exploration_simulator_astar",
             "global_priority_planner_frontier", "sim4_speedup",
             "intelx_common", "intelx_policy", "intelx_explorer",
             "intelx_simulation", "intelx_visualization", "tier1_probe"}
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")[0]
            assert head not in stale, "%s: from %s import ..." % (
                path, node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] not in stale, (
                    "%s: import %s" % (path, a.name))


def test_every_package_has_an_init():
    for root, _, files in os.walk(os.path.join(SRC, "intelx")):
        if any(f.endswith(".py") and f != "__init__.py" for f in files):
            assert "__init__.py" in files, "%s is not a package" % root
