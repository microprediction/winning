"""Moment identities for the exact N-way rating update, vs Monte Carlo."""

import sys
from pathlib import Path

import numpy as np

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
