"""Anchors for every arena contestant: N=2 closed form, N=5 vs MC."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from winning.methods import METHODS  # noqa: E402

RNG = np.random.default_rng(21)

mu2 = np.array([0.3, -0.2])
V2 = np.array([[0.6], [-0.1]])
D2 = np.array([0.8, 1.2])
EXACT2 = ndtr((mu2[0] - mu2[1])
              / np.sqrt((V2[0, 0] - V2[1, 0]) ** 2 + D2[0] + D2[1]))


def _problem5():
    mu = RNG.normal(0, 1, 5)
    V = RNG.normal(0, 0.35, (5, 2))
    D = RNG.uniform(0.5, 1.5, 5)
    L = np.linalg.cholesky(V @ V.T + np.diag(D) + 1e-12 * np.eye(5))
    U = mu[None, :] + RNG.standard_normal((2_000_000, 5)) @ L.T
    truth = np.bincount(np.argmax(U, 1), minlength=5) / 2e6
    return mu, V, D, truth


BUDGET2 = {"lattice": None, "direct_mc": 400_000, "sobol_direct": 2**17,
           "factor_rqmc": 2**14, "ghk": 100_000, "qmc_ghk": 2**14,
           "tilting": 50_000, "genz_bretz": 2**14, "mendell_elston": None,
           "ep_orthant": None, "smc_orthant": 50_000, "stern": 30_000}


# deterministic APPROXIMATIONS are anchored to their documented accuracy
# class, not to exactness; everything else is anchored tight
APPROX_TOL5 = {"mendell_elston": 5e-2, "ep_orthant": 2e-2}


@pytest.mark.parametrize("name", sorted(METHODS))
def test_binary_closed_form(name):
    p, _ = METHODS[name](mu2, V2, D2, budget=BUDGET2[name], seed=5)
    assert abs(p[0] - EXACT2) < 4e-3   # all methods exact-class at N=2


BUDGET5 = {"lattice": None, "direct_mc": 400_000, "sobol_direct": 2**16,
           "factor_rqmc": 2**13, "ghk": 20_000, "qmc_ghk": 2**13,
           "tilting": 20_000, "genz_bretz": 2**13, "mendell_elston": None,
           "ep_orthant": None, "smc_orthant": 20_000, "stern": 8_000}


@pytest.mark.parametrize("name", sorted(METHODS))
def test_five_way_vs_mc(name):
    mu, V, D, truth = _problem5()
    p, _ = METHODS[name](mu, V, D, budget=BUDGET5[name], seed=7)
    assert np.abs(p - truth).max() < APPROX_TOL5.get(name, 5e-3)
