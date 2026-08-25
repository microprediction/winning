"""Tests for exp15: the triple-tie derivative identity and the error certificate."""

import sys
from pathlib import Path

import numpy as np

for sub in ("", "exp14_boundaries", "exp15_perturbation_certificate"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "experiments" / sub))
import run_certificate as ct  # noqa: E402
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

RNG = np.random.default_rng(11)


def small_problem(n=8):
    mu = RNG.normal(0, 0.6, n)
    V = RNG.normal(0, 0.4, (n, 2))
    D = RNG.uniform(0.6, 1.2, n)
    return mu, V, D


def test_price_identity_matches_finite_differences():
    n = 8
    mu, V, D = small_problem(n)
    F, W = hermite_nodes(3)
    Vp = np.hstack([V, np.zeros((n, 1))])
    p0 = win_probabilities_factor(mu, Vp, D, F, W)
    F2, W2 = hermite_nodes(2)
    eps = 2e-3
    for (j, k) in ((0, 5), (2, 7)):
        V2, D2 = ct.perturb_entry(Vp, D, j, k, eps, slot=2)
        fd = (win_probabilities_factor(mu, V2, D2, F, W) - p0) / eps
        t = ct.tie_vector(mu, V, D, F2, W2, j, k)
        others = [i for i in range(n) if i not in (j, k)]
        assert np.abs(fd[others] - t[others]).max() < 5e-4
        assert abs(fd.sum()) < 1e-10                    # sum rule
        # NOTE: winner-term negativity and |fd| <= T_jk were both REFUTED by
        # exp15 (max observed ratio 1.27); no sign/bound assertion here.


def test_pair_totals_match_tie_vectors_and_are_nonnegative():
    n = 7
    mu, V, D = small_problem(n)
    F, W = hermite_nodes(2)
    T = ct.tie_pair_totals(mu, V, D, F, W)
    assert np.abs(T - T.T).max() < 1e-12
    assert T.min() >= 0.0
    for (j, k) in ((1, 4), (0, 6)):
        t = ct.tie_vector(mu, V, D, F, W, j, k)
        assert abs(T[j, k] - t.sum()) < 1e-10


def test_certificate_bounds_actual_error_on_small_case():
    n = 12
    mu, V, D = small_problem(n)
    F6, W6 = hermite_nodes(4)
    Vp = np.hstack([V, np.zeros((n, 2))])
    p0 = win_probabilities_factor(mu, Vp, D, F6, W6)
    T = ct.tie_pair_totals(mu, V, D, *hermite_nodes(2))
    for scale in (0.03, 0.1):
        Vx = RNG.normal(0, scale, (n, 2))
        V1 = np.hstack([V, Vx])
        D1 = np.maximum(D - np.sum(Vx**2, axis=1), 1e-3)
        p1 = win_probabilities_factor(mu, V1, D1, F6, W6)
        dS = Vx @ Vx.T; np.fill_diagonal(dS, 0.0)
        cert = 0.5 * float(np.sum(np.abs(dS) * T))
        actual = np.abs(p1 - p0).max()
        # empirical conservative estimate (held in 100% of exp15 tests, but the
        # underlying per-entry bound is refuted at ratio <= 1.27: not a theorem)
        assert cert >= actual
        assert cert < 100 * actual
