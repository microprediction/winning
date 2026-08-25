"""Tests for exp13: correctness anchors for the GHK benchmark's three methods."""

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments" / "exp13_ghk_benchmark"))
import run_ghk_benchmark as bm  # noqa: E402


def test_binary_probit_closed_form_all_methods():
    mu = np.array([0.3, -0.2])
    V = np.array([[0.6], [-0.1]])
    D = np.array([0.8, 1.2])
    var_diff = (V[0, 0] - V[1, 0]) ** 2 + D[0] + D[1]
    exact = ndtr((mu[0] - mu[1]) / np.sqrt(var_diff))
    assert abs(bm.lattice_shares(mu, V, D)[0] - exact) < 5e-4
    Sigma = V @ V.T + np.diag(D)
    assert abs(bm.ghk_prob(mu, Sigma, 0, R=100_000) - exact) < 3e-3
    assert abs(bm.mc_shares(mu, V, D, 400_000)[0] - exact) < 3e-3


def test_small_field_methods_agree_with_mc():
    rng = np.random.default_rng(2)
    mu, V, D = bm.make_problem(6, 2, rng)
    truth = bm.mc_shares(mu, V, D, 2_000_000)
    assert np.abs(bm.lattice_shares(mu, V, D) - truth).max() < 1.5e-3
    assert np.abs(bm.ghk_all_shares(mu, V, D, R=20_000) - truth).max() < 4e-3


def test_ghk_common_random_numbers_are_deterministic():
    rng = np.random.default_rng(3)
    mu, V, D = bm.make_problem(8, 2, rng)
    Sigma = V @ V.T + np.diag(D)
    u = np.random.default_rng(5).random((500, 7))
    a = bm.ghk_prob(mu, Sigma, 2, u=u)
    b = bm.ghk_prob(mu, Sigma, 2, u=u)
    assert a == b


def test_shares_sum_to_one_and_positive():
    rng = np.random.default_rng(4)
    mu, V, D = bm.make_problem(40, 3, rng)
    p = bm.lattice_shares(mu, V, D)
    assert abs(p.sum() - 1.0) < 1e-12
    assert np.all(p > 0)


def test_inversion_roundtrip_small():
    rng = np.random.default_rng(6)
    mu, V, D = bm.make_problem(30, 2, rng)
    from raceutil import abilities_from_probabilities_factor, hermite_nodes, \
        win_probabilities_factor
    F, W = hermite_nodes(2)
    target = bm.lattice_shares(mu, V, D)
    a_hat = abilities_from_probabilities_factor(target, V, D, F, W)
    back = win_probabilities_factor(a_hat, V, D, F, W)
    assert np.abs(back - target).max() < 2e-3
