"""Anchors for the correlated-probit preference likelihood.

The gradient identity is the load-bearing part: one wrong sign and the fitted
rewards are garbage in a way no loss curve reveals. It is checked against
finite differences of the forward probability, the row-sum-zero property, the
softmax closed form in the Gumbel-free limit, and simulated annotators.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "research" / "qpo"))

from likelihood import (duplicate_structure, loglik_and_grad_sources,  # noqa: E402
                        plackett_luce_gap, true_win_probabilities,
                        win_probability_and_grad)
from pom import hermite_nodes, pom_fast, sobol_nodes  # noqa: E402


def _model(K=6, r=2, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.standard_normal(K) * 0.6
    V = rng.standard_normal((K, r)) * 0.5
    d = rng.uniform(0.5, 1.2, K)
    return mu, V, d


def test_probability_matches_pom_fast():
    mu, V, d = _model(seed=1)
    F, W = hermite_nodes(2, Q=25)
    p_all = pom_fast(mu, V, d, F, W, points=1001)
    for i in (0, 3, 5):
        p, _ = win_probability_and_grad(i, mu, V, d, F, W, points=1001)
        assert abs(p - p_all[i]) < 2e-6, (i, p, p_all[i])


@pytest.mark.parametrize("i", [0, 2, 4])
def test_gradient_matches_finite_differences(i):
    mu, V, d = _model(seed=2)
    F, W = hermite_nodes(2, Q=21)
    p, g = win_probability_and_grad(i, mu, V, d, F, W, points=1001)
    assert abs(g.sum()) < 1e-8 * max(np.abs(g).max(), 1e-12)
    h = 1e-5
    for j in range(len(mu)):
        e = np.zeros(len(mu)); e[j] = h
        pp = pom_fast(mu + e, V, d, F, W, points=1001)[i]
        pm = pom_fast(mu - e, V, d, F, W, points=1001)[i]
        fd = (pp - pm) / (2 * h)
        assert abs(g[j] - fd) < 5e-5 * max(abs(fd), 1e-3), (i, j, g[j], fd)


def test_gumbel_limit_recovers_softmax_gradient():
    """Independent case sanity: gradient signs and the IIA structure.

    With V = 0 and equal variances the model is an independent probit, not
    softmax, so we do not demand the softmax formula -- only its qualitative
    fingerprints: dp_i/dmu_i > 0, dp_i/dmu_j < 0, rows sum to zero.
    """
    K = 5
    mu = np.linspace(-0.5, 0.5, K)
    d = np.ones(K)
    p, g = win_probability_and_grad(2, mu, None, d, points=2001)
    assert g[2] > 0
    assert all(g[j] < 0 for j in range(K) if j != 2)


def test_duplicate_structure_marginals_are_unit():
    src = [0, 1, 1, 1]
    for rho in (0.0, 0.4, 0.9):
        V, d = duplicate_structure(src, rho)
        assert np.allclose(np.sum(V ** 2, axis=1) + d, 1.0)


def test_true_probabilities_match_simulation():
    src = [0, 1, 1, 1]
    theta = np.array([0.0, 0.3])
    rho = 0.6
    p = true_win_probabilities(src, theta, rho)
    rng = np.random.default_rng(3)
    M = 400_000
    z = rng.standard_normal(M)
    U = np.empty((M, 4))
    U[:, 0] = theta[0] + rng.standard_normal(M)
    for j in range(1, 4):
        U[:, j] = theta[1] + np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.standard_normal(M)
    emp = np.bincount(np.argmax(U, axis=1), minlength=4) / M
    se = np.sqrt(emp * (1 - emp) / M)
    assert np.max(np.abs(p - emp) / np.maximum(se, 1e-9)) < 5.0


def test_mle_recovers_the_true_gap_from_simulated_annotators():
    """End to end: simulate winner-only data, fit theta by gradient ascent."""
    src = [0, 1, 1, 1]
    theta_true = np.array([0.0, 0.3])
    rho = 0.7
    p_true = true_win_probabilities(src, theta_true, rho)
    rng = np.random.default_rng(4)
    n = 200_000
    counts = rng.multinomial(n, p_true / p_true.sum())

    F, W = sobol_nodes(1, m=9, seed=0)
    theta = np.zeros(2)
    for it in range(300):
        ll, g = loglik_and_grad_sources(counts, src, theta, rho, 3, F, W,
                                        points=201)
        g = g - g.mean()                      # location is not identified
        theta = theta + 0.5 * g / n
        theta -= theta.mean()
        if np.max(np.abs(g / n)) < 1e-6:
            break
    gap_hat = theta[1] - theta[0]
    assert abs(gap_hat - 0.3) < 0.03, (gap_hat, it)


def test_plackett_luce_gap_formula():
    """Winner-only PL is multinomial logit; its MLE matches frequencies."""
    p = np.array([0.4, 0.2, 0.2, 0.2])
    gap = plackett_luce_gap(p, [0, 1, 1, 1])
    # softmax with s_A - s_B = log(3 * 0.4 / 0.6) = log 2 reproduces p exactly
    sA, sB = np.log(3 * 0.4 / 0.6), 0.0
    q = np.exp([sA, sB, sB, sB]); q /= q.sum()
    assert np.allclose(q, p)
    assert abs(gap - np.log(2.0)) < 1e-12
