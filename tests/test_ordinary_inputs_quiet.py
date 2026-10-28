"""Ordinary inputs must be quiet: no warning, no exception.

A warning that fires on everyday inputs teaches callers to ignore
warnings, which defeats the ones that matter. Every call in
tests/ordinary_battery.py is an ordinary field (see its docstring), so
none of them may warn or raise. A PR that adds a warning, a refusal or
a convergence check must keep this file green; if a new warning fires
here, its threshold is wrong for ordinary data, not this test.

Context: a contrast-residual warning at 0.05 on fit_covariance fired on
15 of 25 ordinary sample covariances before review caught it (#505,
PR #736).
"""
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from ordinary_battery import calls  # noqa: E402

CALLS = list(calls())


@pytest.mark.parametrize("name,thunk", CALLS, ids=[c[0] for c in CALLS])
def test_ordinary_call_is_quiet(name, thunk):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thunk()
    noisy = [f"{w.category.__name__}: {w.message}" for w in caught
             # third-party deprecations are not ours to police here
             if not issubclass(w.category, (DeprecationWarning,
                                            PendingDeprecationWarning))]
    assert not noisy, f"{name} warned on an ordinary input: {noisy}"
