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


def test_rprobitfast_engine_is_prefix_of_mlogitfast():
    # Conscious divergence (2026-08-28, CRAN prep): engine.R used to be a
    # FULL copy of mlogit_fast.R, but that shipped a dead, unexported
    # copy of mlogitfast's interface inside rprobitfast (and drew a CRAN
    # NOTE for its stray imports). engine.R is now the shared internals
    # only, and the guard is that it remains an exact PREFIX of
    # mlogit_fast.R: the engine cannot drift between the two packages,
    # while the interface tail belongs to mlogitfast alone.
    root = Path(__file__).resolve().parents[1]
    a = (root / "r" / "mlogitfast" / "R" / "mlogit_fast.R").read_text()
    b = (root / "r" / "rprobitfast" / "R" / "engine.R").read_text()
    assert len(b) > 3000, "engine.R suspiciously small; sync check void"
    assert a.startswith(b.rstrip("\n")), (
        "r/rprobitfast/R/engine.R must be an exact prefix of "
        "r/mlogitfast/R/mlogit_fast.R (shared engine internals); "
        "they have diverged -- sync deliberately and update this test")
