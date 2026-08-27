"""N = 2 inversion regression.

For N = 2 the photo-finish graph K_2 is bipartite: the normalized
Laplacian eigenvalue is exactly 2, so the undamped Jacobi update on the
mean-zero quotient has eigenvalue -1 and is a local two-cycle. Before
the fix, abilities_from_probabilities_factor returned silently at the
iteration cap with log-share errors up to ~1 for mid-range targets
(the flat residual never triggered the growth safeguard). The Gaussian
kernel now returns the closed form, and the general-base inversion uses
fixed damping 0.7 at N = 2.
"""

import numpy as np
import pytest

from winning.factor.core import (abilities_from_probabilities_factor,
                                 win_probabilities_factor, hermite_nodes)
from winning.factor.races import abilities_from_race, race_probabilities


@pytest.mark.parametrize("p1", [0.06, 0.27, 0.39, 0.5, 0.65, 0.94, 0.999])
def test_gaussian_n2_round_trip(p1):
    rng = np.random.default_rng(7)
    F, W = hermite_nodes(2)
    V = rng.normal(0.0, 0.6, (2, 2))
    D = np.array([0.8, 1.7])
    p = np.array([p1, 1.0 - p1])
    mu = abilities_from_probabilities_factor(p, V, D, F, W)
    assert abs(mu.sum()) < 1e-12
    p_back = win_probabilities_factor(mu, V, D, F, W)
    p_back = p_back / p_back.sum()
    assert np.abs(np.log(p_back) - np.log(p)).max() < 1e-5


def test_gaussian_n2_matches_closed_form():
    from scipy.special import ndtri
    F, W = hermite_nodes(1)
    V = np.array([[0.5], [-0.3]])
    D = np.array([1.0, 2.0])
    p = np.array([0.3, 0.7])
    mu = abilities_from_probabilities_factor(p, V, D, F, W)
    s = np.sqrt(D.sum() + np.sum((V[0] - V[1]) ** 2))
    assert np.allclose(mu[1] - mu[0], s * ndtri(p[0]), atol=1e-12)


def test_gaussian_n2_reports_converged():
    F, W = hermite_nodes(2)
    V = np.zeros((2, 1))
    D = np.ones(2)
    p = np.array([0.94, 0.06])
    mu, info = abilities_from_probabilities_factor(
        p, V, np.ones(2), F, W, return_info=True)
    assert info["converged"]


@pytest.mark.parametrize("base", ["normal", "gumbel"])
def test_general_base_n2_round_trip(base):
    rng = np.random.default_rng(11)
    F, W = hermite_nodes(1)
    V = rng.normal(0.0, 0.5, (2, 1))
    D = np.array([1.0, 1.3])
    p = np.array([0.61, 0.39])
    mu = abilities_from_race(p, V=V, D=D, F=F, W=W, base=base)
    p_back = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base)
    p_back = p_back / p_back.sum()
    assert np.abs(np.log(p_back) - np.log(p)).max() < 1e-6


def test_n3_unaffected():
    rng = np.random.default_rng(3)
    F, W = hermite_nodes(2)
    mu0 = rng.normal(0, 1, 3)
    mu0 -= mu0.mean()
    V = rng.normal(0.0, 0.6, (3, 2))
    D = rng.uniform(0.5, 2.0, 3)
    p = win_probabilities_factor(mu0, V, D, F, W)
    mu = abilities_from_probabilities_factor(p / p.sum(), V, D, F, W)
    assert np.abs(mu - mu0).max() < 1e-5
