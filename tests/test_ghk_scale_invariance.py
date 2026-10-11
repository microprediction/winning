"""GHK and its sibling orthant methods are homogeneous in the utility
unit and blind to a common shock (#409, #101, #302).

Rescaling a race (mu -> c mu, V -> c V, D -> c^2 D) describes the same
argmax law. A binary race has a single truncation step, so every GHK
draw carries the same weight and these are exact checks, not Monte
Carlo comparisons: the absolute 1e-12 ridge gave Phi(c / sqrt(2c^2 +
1e-12)), which is 0.7181 at c = 1e-6 and 0.5040 at c = 1e-8 against the
scale-free Phi(1/sqrt 2) = 0.7602.
"""
import warnings

import numpy as np
import pytest
from scipy.special import ndtr

from winning.methods.native import ghk, qmc_ghk, tilting
from winning.methods.orthant_extra import genz_bretz, mendell_elston

EXACT = float(ndtr(1 / np.sqrt(2)))


@pytest.mark.parametrize("method", [ghk, qmc_ghk, tilting, mendell_elston,
                                    genz_bretz])
@pytest.mark.parametrize("c", [1.0, 1e-6, 1e-8, 1e-12])
def test_a_binary_race_in_tiny_units_is_the_same_race(method, c):
    mu = np.array([0.0, c])
    p, _ = method(mu, np.empty((2, 0)), np.array([c * c, c * c]))
    assert p[1] == pytest.approx(EXACT, abs=1e-10)


def test_qmc_ghk_cov_door_is_scale_free_too():
    c = 1e-8
    p, _ = qmc_ghk(np.array([0.0, c]), None, None, cov=c * c * np.eye(2))
    assert p[1] == pytest.approx(EXACT, abs=1e-10)


@pytest.mark.parametrize("method", [ghk, qmc_ghk, tilting])
@pytest.mark.parametrize("v", [0.0, 1e4, 1e8])
def test_a_common_loading_cancels(method, v):
    """V -> V + 1 c' adds the same shock to every utility. Materialising
    Sigma = V V' + diag(D) first rounded 1e16 + 1 to 1e16, the contrast
    came back zero, and the 76/24 race was priced as [0, 1] (#302)."""
    p, _ = method(np.array([0.0, 1.0]), np.full((2, 1), v), np.ones(2))
    assert p[1] == pytest.approx(EXACT, abs=1e-10)


def test_a_three_way_race_is_scale_free():
    rng = np.random.default_rng(3)
    V = rng.normal(size=(4, 2)) * 0.6
    D = 0.5 + rng.random(4)
    mu = rng.normal(size=4) * 0.5
    p1, _ = qmc_ghk(mu, V, D, budget=512, seed=4)
    for c in (1e-6, 1e-9):
        pc, _ = qmc_ghk(c * mu, c * V, c * c * D, budget=512, seed=4)
        assert np.max(np.abs(pc - p1)) < 1e-9


def test_fit_covariance_is_homogeneous():
    """The second #101 counterexample: fitted at 1e-10 times the
    covariance, the same standardized race came back at rank 6 and
    sharpness 1113 instead of rank 3 and 3.16, and priced 2.7e-3 away in
    total variation."""
    from winning.factor.core import fit_covariance
    from winning.factor.races import race_probabilities
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6, 2))
    C = A @ A.T + np.diag(np.exp(rng.normal(size=6)))
    mu = np.array([-1.1, -.4, 0., .2, .7, 1.4])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for s in (1.0, 1e-10, 1e8):
            V, D, F, W, rep = fit_covariance(s * C, return_report=True)
            out[s] = (rep["rank"], rep["sharpness"],
                      race_probabilities(np.sqrt(s) * mu, V=V, D=D, F=F,
                                         W=W))
    for s in (1e-10, 1e8):
        assert out[s][0] == out[1.0][0]
        assert out[s][1] == pytest.approx(out[1.0][1], rel=1e-9)
        assert 0.5 * np.abs(out[s][2] - out[1.0][2]).sum() < 1e-12
