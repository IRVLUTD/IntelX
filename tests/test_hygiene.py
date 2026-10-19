"""No file in this tree carries a home directory, a machine or an address.

So the tree runs anywhere and leaks nothing about where it was developed. Add
a pattern here the moment you find one that got through.
"""
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Anywhere in the tree.
SKIP_ANYWHERE = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                 ".ruff_cache", ".mypy_cache", "node_modules", ".eggs",
                 "build", "dist"}
# Only at the top level. `src/intelx/data` is a package and must be scanned,
# so these are matched by path rather than by name.
# third_party holds other projects' repositories as git submodules (MapEx);
# their code is theirs, so it is not held to this repository's rules.
SKIP_AT_ROOT = {"data", "weights", "results", "third_party"}


def prune(root, dirs):
    rel = os.path.relpath(root, REPO)
    keep = []
    for d in dirs:
        if d in SKIP_ANYWHERE or d.endswith(".egg-info"):
            continue
        if rel == "." and d in SKIP_AT_ROOT:
            continue
        keep.append(d)
    dirs[:] = keep
TEXT_EXT = {".py", ".md", ".txt", ".toml", ".cfg", ".yaml", ".yml", ".json",
            ".sh", ".gitignore", ""}

# This file's own known-bad samples are the only thing excused.
ALLOWED = ("tests/test_hygiene.py",)

PATTERNS = {
    "a home directory": re.compile(r"/(?:home|Users)/[A-Za-z0-9._-]+"),
    "a Windows user profile": re.compile(r"[A-Za-z]:\\Users\\"),
    # an assignment, not prose that merely names the thing
    "a wandb entity": re.compile(
        r"wandb[^\n]{0,60}?entity\s*[=:]\s*[\"']"),
    # a host, not a version pin, so the quad has to sit in a host position
    "a routable IP address": re.compile(
        r"(?:[\"'@]|//)(?!127\.0\.0\.1\b|0\.0\.0\.0\b)"
        r"(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Each pattern must actually fire. A regex that silently matches nothing is
# worse than no check at all, so the guard is guarded.
SAMPLES = {
    "a home directory": "path = '/home/alice/work'",
    "a Windows user profile": r"path = 'C:\Users\alice'",
    "a wandb entity": "wandb.init(project='p', entity='somelab')",
    "a routable IP address": "host = '10.0.1.7'",
}


# The directories in SKIP_AT_ROOT hold generated data, but these files in them
# ship with the repository and are scanned like any other.
SHIPPED_IN_SKIPPED = ("data/README.md", "weights/README.md")


def text_files(skip=ALLOWED):
    for root, dirs, files in os.walk(REPO):
        prune(root, dirs)
        for f in sorted(files):
            p = os.path.join(root, f)
            if os.path.splitext(f)[1] in TEXT_EXT:
                rel = os.path.relpath(p, REPO)
                if not rel.startswith(skip):
                    yield rel
    for rel in SHIPPED_IN_SKIPPED:
        if os.path.exists(os.path.join(REPO, rel)):
            yield rel


FILES = sorted(text_files())


@pytest.mark.parametrize("what", sorted(PATTERNS))
def test_the_pattern_actually_fires(what):
    assert PATTERNS[what].search(SAMPLES[what]), (
        "%r never matches, so it protects nothing" % what)


def test_localhost_is_not_flagged():
    rx = PATTERNS["a routable IP address"]
    assert not rx.search("HOST = '127.0.0.1'")
    assert not rx.search("srv.bind(('127.0.0.1', 0))")
    assert rx.search("HOST = '192.168.1.40'")


def test_a_version_pin_is_not_an_address():
    rx = PATTERNS["a routable IP address"]
    for pin in ("nvidia-cublas-cu12==12.6.4.1", "nvidia-nvcomp-cu12==5.0.0.6",
                "python-dateutil==2.9.0.post0"):
        assert not rx.search(pin), pin


@pytest.mark.parametrize("rel", FILES)
def test_no_identifying_strings(rel):
    try:
        body = open(os.path.join(REPO, rel), encoding="utf-8").read()
    except UnicodeDecodeError:
        pytest.skip("not text")
    for what, rx in PATTERNS.items():
        m = rx.search(body)
        assert not m, "%s line %d carries %s: %r" % (
            rel, body[:m.start()].count("\n") + 1, what, m.group(0))


# Calls that return the identity of the machine or the person running it.
# These are fine to compute; what is not fine is writing one into a file the
# release ships. A call is accepted when the same statement hashes it or
# checks the opt-in environment variable.
IDENTITY_CALLS = {
    ("platform", "node"), ("socket", "gethostname"), ("socket", "getfqdn"),
    ("getpass", "getuser"), ("os", "getlogin"),
}


def test_no_machine_identity_reaches_a_file():
    import ast
    src = os.path.join(REPO, "src")
    bad = []
    for root, dirs, files in os.walk(src):
        prune(root, dirs)
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            body = open(p, encoding="utf-8").read()
            tree = ast.parse(body, filename=p)
            lines = body.split(chr(10))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if not (isinstance(fn, ast.Attribute)
                        and isinstance(fn.value, ast.Name)):
                    continue
                if (fn.value.id, fn.attr) not in IDENTITY_CALLS:
                    continue
                lo = max(0, node.lineno - 4)
                hi = min(len(lines), node.lineno + 3)
                near = chr(10).join(lines[lo:hi])
                if "hashlib" in near or "INTELX_RECORD_HOSTNAME" in near:
                    continue
                bad.append("%s line %d calls %s.%s unguarded"
                           % (os.path.relpath(p, REPO), node.lineno,
                              fn.value.id, fn.attr))
    assert not bad, "; ".join(bad)


def test_the_identity_guard_would_fire():
    """The check above is worth having only if it catches the plain form."""
    import ast
    tree = ast.parse("hf.attrs['hostname'] = platform.node()")
    call = [n for n in ast.walk(tree) if isinstance(n, ast.Call)][0]
    assert (call.func.value.id, call.func.attr) in IDENTITY_CALLS


def test_the_scan_reaches_every_source_file():
    """A skip rule that quietly excluded a package would make the rest of this
    file meaningless, so the count is checked against the tree."""
    on_disk = set()
    for root, _, files in os.walk(os.path.join(REPO, "src")):
        if "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                on_disk.add(os.path.relpath(os.path.join(root, f), REPO))
    exempt = {p for p in on_disk if p.startswith(ALLOWED)}
    missed = sorted(on_disk - set(FILES) - exempt)
    assert not missed, "not scanned: %s" % ", ".join(missed)
    assert len(on_disk) - len(exempt) > 25, "the scan found suspiciously few files"


def test_the_sweep_is_not_pinned_to_one_machine():
    """Maps reach the code through map_path, never as a literal path."""
    import ast
    src = os.path.join(REPO, "src")
    for root, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            tree = ast.parse(open(p, encoding="utf-8").read(), filename=p)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(
                        node.value, str) and node.value.endswith(".pgm"):
                    assert not os.path.isabs(node.value), (
                        "%s line %d hard-codes %s"
                        % (os.path.relpath(p, REPO), node.lineno, node.value))
