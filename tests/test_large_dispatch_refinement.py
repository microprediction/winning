"""#117: crossing the n*Q memory threshold of the Rust front door must be a
storage optimization only. The large branch used to skip forward_grid's
sharpness refinement and integrate a sharp field on the requested 257
points (TV 2.5e-2 against the refined lattice)."""
import warnings

import numpy as np
import pytest

pytest.importorskip("fastrace")

import winning.factor.races as R
from winning.factor.core import qmc_nodes


def _sharp_field(n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    mu = np.linspace(-.1, .1, n)
    V = np.c_[.1 * np.cos(th), .1 * np.sin(th)]
    D = np.full(n, 1e-6)                    # sharpness = 100
    F, W = qmc_nodes(2, m=7)
    return mu, V, D, F, W


def _price(threshold, monkeypatch, **kw):
    mu, V, D, F, W = _sharp_field()
    monkeypatch.setattr(R, "_LARGE_DISPATCH_ENTRIES", threshold)
    seen = {}
    real = R._fastrace.forward_and_slopes

    def spy(mu_, V_, D_, F_, W_, points, lo, hi):
        seen["dx"] = (hi - lo) / (points - 1)
        return real(mu_, V_, D_, F_, W_, points, lo, hi)

    monkeypatch.setattr(R._fastrace, "forward_and_slopes", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = R.race_probabilities(mu, V=V, D=D, F=F, W=W, points=257, **kw)
    return p, seen["dx"]


@pytest.mark.parametrize("window", ["bulk", "span"])
def test_threshold_does_not_change_the_discretization(monkeypatch, window):
    mu, V, D, F, W = _sharp_field()
    nq = len(mu) * len(F)
    smin = float(np.sqrt(D.min()))
    p_below, dx_below = _price(nq + 1, monkeypatch, window=window)  # ordinary
    p_above, dx_above = _price(nq - 1, monkeypatch, window=window)  # large
    assert dx_below <= 0.5 * smin * (1 + 1e-12)
    assert dx_above <= 0.5 * smin * (1 + 1e-12), (
        "large branch skipped the sharpness refinement")
    tv = 0.5 * np.abs(p_above - p_below).sum()
    assert tv < 1e-3, tv
