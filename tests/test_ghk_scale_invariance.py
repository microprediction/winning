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


@pytest.mark.parametrize("method", [ghk, qmc_ghk])
@pytest.mark.parametrize("c", [1e-150, 1e-155])
def test_subnormal_scale_is_the_same_race_409(method, c):
    # the relative ridge's 1e-300 floor was an absolute 1e-312 below it:
    # D = 1e-310 priced 0.7597, D = 1e-320 0.5000
    mu = np.array([0.0, c])
    p, _ = method(mu, np.empty((2, 0)), np.array([c * c, c * c]),
                  budget=32, seed=9)
    assert p[1] == pytest.approx(EXACT, abs=1e-12)


@pytest.mark.parametrize("method,budget", [(ghk, 200_000),
                                           (qmc_ghk, 65_536)])
def test_a_valid_small_pivot_is_not_inflated_409(method, budget):
    # near-twins share a unit factor with variance 2^-43 each, their gap
    # scaled to that direction. The always-on relative ridge priced
    # [0.5, 0.167, 0.333] at any budget; exact via a standardized
    # bivariate CDF: [0.49999992, 0.07932763, 0.42067245]
    delta = 2.0 ** -43
    a = 0.5 * np.sqrt(2 * delta)
    V = np.array([[0.0], [1.0], [1.0]])
    D = np.array([1.0, delta, delta])
    p, _ = method(-np.array([0.0, a, -a]), V, D, budget=budget, seed=9)
    np.testing.assert_allclose(p, [0.49999992, 0.07932763, 0.42067245],
                               atol=2e-3)


@pytest.mark.parametrize("method", [ghk, qmc_ghk])
def test_overflowing_variance_sum_is_the_same_race_409(method):
    # finite variances whose trace overflows (5e307 each): the scale is
    # formed without overflow, so no RuntimeWarning, and the race equals
    # its reciprocal unit change mu /= 1e154, D /= 1e308
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p, _ = method(np.array([0.0, 1e154, -1e154]), np.empty((3, 0)),
                      np.full(3, 5e307), budget=4096, seed=9)
    q, _ = method(np.array([0.0, 1.0, -1.0]), np.empty((3, 0)),
                  np.full(3, 0.5), budget=4096, seed=9)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p, q, rtol=0, atol=1e-12)
