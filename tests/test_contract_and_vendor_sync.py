"""The cross-repo contract holds, and the vendored fastmvn copy is in
sync with the package module (divergence must be a conscious act)."""
from pathlib import Path


def test_contract():
    from winning.contract import verify
    assert verify(verbose=False)


def test_fastmvn_vendored_copy_is_identical():
    root = Path(__file__).resolve().parents[1]
    a = (root / "winning" / "fastmvn.py").read_text()
    b = (root / "python" / "fastmvn" / "src" / "fastmvn" / "core.py").read_text()
    assert a == b, ("python/fastmvn vendors winning/fastmvn.py; they have "
                    "diverged -- sync deliberately and update both tests")


def test_rprobitfast_engine_is_identical_to_mlogitfast():
    root = Path(__file__).resolve().parents[1]
    a = (root / "r" / "mlogitfast" / "R" / "mlogit_fast.R").read_text()
    b = (root / "r" / "rprobitfast" / "R" / "engine.R").read_text()
    assert a == b, ("r/rprobitfast/R/engine.R is a sync-guarded copy of "
                    "r/mlogitfast/R/mlogit_fast.R; they have diverged")
