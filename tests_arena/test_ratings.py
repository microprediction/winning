"""Moment identities for the exact N-way rating update, vs Monte Carlo."""

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from winning.ratings import update_winner  # noqa: E402


def test_exact_moments_match_monte_carlo():
    rng = np.random.default_rng(11)
    n = 8
    m = rng.normal(0, 1.0, n)
    v = rng.uniform(0.3, 0.8, n)
    beta2 = 0.5
    i = 3
    m_new, v_new, p_i = update_winner(m, v, i, beta2)
    R = 2_000_000
    s = m[None, :] + np.sqrt(v)[None, :] * rng.standard_normal((R, n))
    x = s + np.sqrt(beta2) * rng.standard_normal((R, n))
    win = np.argmax(x, axis=1) == i
    se = np.sqrt(v.max() / win.sum())
    assert abs(p_i - win.mean()) < 5e-4
    assert np.abs(m_new - s[win].mean(0)).max() < 5 * se
    assert np.abs(v_new - s[win].var(0)).max() < 10 * se


def test_order_pass_matches_closed_form_n2():
    from winning.ratings.nway import _order_pass
    m = np.array([0.3, -0.4]); sd = np.array([1.1, 0.9])
    logP, grad = _order_pass(m, sd, [0, 1])
    z = (m[0] - m[1]) / np.hypot(sd[0], sd[1])
    assert abs(logP - np.log(ndtr(z))) < 1e-4


def test_order_pass_gradient_matches_fd_n20():
    from winning.ratings.nway import _order_pass
    rng = np.random.default_rng(7)
    n = 20
    m = rng.normal(0, 1.0, n); sd = np.full(n, 1.0)
    order = list(rng.permutation(n))
    _, grad = _order_pass(m, sd, order)
    eps = 1e-6
    for j in (order[0], order[10], order[-1]):
        e = np.zeros(n); e[j] = eps
        fd = (_order_pass(m + e, sd, order)[0]
              - _order_pass(m - e, sd, order)[0]) / (2 * eps)
        assert abs(grad[j] - fd) < 1e-3
