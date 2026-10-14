"""Default results must not drift by accident.

tests/golden/default_drift.npz pins what every call in
tests/ordinary_battery.py returns at default settings. A change that
moves any of them beyond the tolerance below fails here. That is the
point: a default may change only on purpose. If a PR corrects a
default, regenerate the file and say in the PR which entries moved,
by how much, and why each move is a correction:

    PYTHONPATH=. python tests/test_default_drift.py --regenerate

The tolerance absorbs platform and library rounding (numpy/scipy/libm
versions, the compiled kernels against numpy, BLAS on Linux against
Accelerate on macOS), which moves these values by ~1e-15, and nothing
more: the smallest intended correction this suite has had to absorb
was 1.7e-16 relative on single-cluster blocks (#471), the largest
platform drift seen on ordinary inputs ~1e-13 in inverse abilities.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ordinary_battery import calls, flatten  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "default_drift.npz"
RTOL, ATOL = 1e-9, 1e-11
FIT_ATOL = 1e-8

CALLS = list(calls())


def _compute():
    return {name: flatten(thunk()) for name, thunk in CALLS}


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.fail(f"{GOLDEN} is missing; regenerate it (see module "
                    "docstring)")
    with np.load(GOLDEN) as g:
        return {k: g[k] for k in g.files}


def test_golden_covers_the_battery(golden):
    names = {name for name, _ in CALLS}
    assert names == set(golden), (
        "battery and golden disagree; regenerate on purpose: "
        f"missing {sorted(names - set(golden))[:5]}, "
        f"stale {sorted(set(golden) - names)[:5]}")


@pytest.mark.parametrize("name,thunk", CALLS, ids=[c[0] for c in CALLS])
def test_default_does_not_drift(golden, name, thunk):
    got = flatten(thunk())
    want = golden[name]
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    # fit_covariance is an iterative optimiser on top of LAPACK
    # eigendecompositions, so platform rounding reaches ~1e-11 in its
    # fitted covariance (seen on macOS CI); a real change moves it far more
    atol = FIT_ATOL if name.startswith("fit_covariance/") else ATOL
    np.testing.assert_allclose(got, want, rtol=RTOL, atol=atol,
                               err_msg=f"{name} moved: a default changed")


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        sys.exit("pass --regenerate to rewrite " + str(GOLDEN))
    GOLDEN.parent.mkdir(exist_ok=True)
    np.savez_compressed(GOLDEN, **_compute())
    print(f"wrote {GOLDEN} ({len(CALLS)} entries)")
