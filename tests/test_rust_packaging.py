"""Release-identity gates for the Rust artifacts (#119, #115).

Cargo and PyPI versions are immutable: one version string must never name
two different sources. crates.io winning 0.2.0 was packaged from e1135d8,
and the in-repo crate later gained public API (ordered_kernel) while still
saying 0.2.0, as did the fastrace 0.2.0 sdist that vendored it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip("tomllib needs python 3.11", allow_module_level=True)
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / "rust" / "winning"
FASTRACE = ROOT / "rust" / "fastrace"

# version -> commit the crates.io artifact was packaged from (its
# .cargo_vcs_info.json). Add a row when publishing.
PUBLISHED_CRATE = {
    "0.2.0": "e1135d8c8139d3f4a973b8154dd375c302ca219a",
}
PACKAGED = ["rust/winning/src", "rust/winning/Cargo.toml",
            "rust/winning/README.md"]


def _toml(p):
    return tomllib.loads(p.read_text())


def test_fastrace_declares_the_core_version_it_vendors():
    core = _toml(CRATE / "Cargo.toml")["package"]["version"]
    dep = _toml(FASTRACE / "Cargo.toml")["dependencies"]["winning"]
    assert isinstance(dep, dict) and dep.get("path") == "../winning"
    assert dep.get("version") == core, (
        "rust/fastrace must pin the path crate's version so packaged "
        "metadata records which winning it was built against")


def test_unchanged_source_or_new_crate_version():
    version = _toml(CRATE / "Cargo.toml")["package"]["version"]
    commit = PUBLISHED_CRATE.get(version)
    if commit is None:
        return                      # unpublished version: free to change
    try:
        r = subprocess.run(["git", "diff", "--quiet", commit, "--", *PACKAGED],
                           cwd=ROOT, capture_output=True)
    except FileNotFoundError:
        pytest.skip("git unavailable")
    if r.returncode not in (0, 1):
        pytest.skip(f"published commit {commit[:7]} not in this clone")
    assert r.returncode == 0, (
        f"rust/winning differs from the source published as {version}; "
        "bump the crate version")


def test_fastrace_wheel_declares_numpy():
    deps = _toml(FASTRACE / "pyproject.toml")["project"].get("dependencies",
                                                             [])
    assert any(d.split()[0].lower().startswith("numpy") for d in deps), deps
