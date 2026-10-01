"""race_probabilities(cov=C) must reproduce an exactly-structured C.

Reported by the allocation session 2026-09-21 as "cov= collapses to one
factor" -- 5.8e-3 off a 3-factor correlation, ~21 standard errors of a
3M-path simulation, invariant to points and window. Reproduced here
WITHOUT simulation: an exact r-factor correlation has an exact V/D, so
cov=C must agree with V=V, D=D to quadrature precision.

Measured on main, n=8, points=1001:

    exact 1-factor   fit rank 8   D_min 1e-6   max|cov= - V/D| 5.3e-3
    exact 3-factor   fit rank 8   D_min 1e-6   max|cov= - V/D| 6.6e-3
    exact 5-factor   fit rank 8   D_min 1e-6   max|cov= - V/D| 8.4e-3

The mechanism is the opposite of a one-factor collapse. fit_covariance
stacks k=3 global factors + block loadings + m=5 eigendirections + D by
diagonal matching floored at 1e-6 diag(C). Validated at large n; at n=8
the stages sum to n and it degenerates into a FULL-rank fit with D on
the floor. The fit residual is then ~1e-6 -- so no warning fires, they
judge fit quality -- but with D ~ 0 the conditional race is a near-step
the factor-node quadrature cannot resolve. That is the points/window
invariance: the error lives in the factor integral, not the lattice.

Resolved by routing, not by fixing the fit: cov= sends the forward
normal race to scrambled-Sobol GHK when the fit degenerates, so the
first test now passes to GHK's accuracy. The second pins that the fit
itself is unchanged, so the routing stays necessary.
"""
import warnings

import numpy as np
import pytest

import winning.factor as wf
from winning.factor.core import fit_covariance

N = 8
MU = np.linspace(-0.6, 0.6, N)


def exact_correlation(r, seed):
    """Unit-diagonal C = V V' + diag(D) with genuine rank r."""
    rng = np.random.default_rng(seed)
    V = rng.normal(size=(N, r)) * 0.35
    V = V / np.sqrt((V ** 2).sum(1, keepdims=True))
    V = V * np.sqrt(0.6) * rng.random((N, 1)) ** 0.3
    D = 1.0 - (V ** 2).sum(1)
    assert D.min() > 0.05
    return V, D, V @ V.T + np.diag(D)


@pytest.mark.parametrize("r", [1, 3, 5])
def test_cov_reproduces_an_exactly_structured_correlation(r):
    """The fit still degenerates (pinned below); cov= now routes the
    forward normal race to scrambled-Sobol GHK instead of using it, so
    the answer is within GHK's 1024-node accuracy (~5e-4) of the exact
    V/D lattice rather than 5-8e-3 off it."""
    V, D, C = exact_correlation(r, seed=5 + r)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # routed: no warning
        p_vd = np.asarray(wf.race_probabilities(MU, V=V, D=D, points=1001))
        p_cov = np.asarray(wf.race_probabilities(MU, cov=C, points=1001))
    assert np.abs(p_cov - p_vd).max() < 2e-3


@pytest.mark.parametrize("r", [1, 3, 5])
def test_the_defect_is_the_fit_not_the_race(r):
    """Pins the mechanism so the fix goes to the right place: the fit
    chooses full rank and binds the D floor although an exact rank-r fit
    exists, while its own residual report says the fit is nearly exact."""
    V, D, C = exact_correlation(r, seed=5 + r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Vf, Df, F, W, rep = fit_covariance(C, return_report=True)
    assert rep["rank"] == N, "no longer full rank: the xfail above may lift"
    assert Df.min() < 1e-5, "D no longer on the floor: the xfail above may lift"
    assert rep["projected_residual_max"] < 1e-4, \
        "the residual report would now catch this; recheck the warnings"
