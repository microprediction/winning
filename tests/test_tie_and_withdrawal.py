"""A dead heat for first as evidence, and a withdrawal that removes an
entrant from the outcome's field."""
import numpy as np
import pytest

from winning.ratings.nway import (update_winner_correlated, update_tie_correlated,
                                  _tie_logdens_grad)
from winning.ratings.simulate import correlated_draws
from winning.factor.races import race_probabilities
from winning.factor.polish import race_jacobian

M = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
VV = np.array([0.8, 1.0, 0.6, 1.2, 0.9])
L = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])


def test_one_tied_entrant_is_the_winner_and_two_is_the_tie_density():
    rng = np.random.default_rng(7)
    mm = rng.normal(size=5); D = rng.uniform(0.5, 2.0, size=5)
    lp, g = _tie_logdens_grad(mm, D, [2])
    assert abs(np.exp(lp) - race_probabilities(-mm, D=D, points=2001)[2]) < 1e-9
    # the pair tie density is the win Jacobian's off-diagonal (min-wins engine, sign flipped)
    lp2, g2 = _tie_logdens_grad(mm, D, [1, 3])
    J = race_jacobian(-mm, D=D, points=2001)
    assert abs(np.exp(lp2) - J[1, 3]) < 1e-7, (np.exp(lp2), J[1, 3])
    for g_ in (g, g2):
        assert abs(g_.sum()) < 1e-9
    h = 1e-5
    for j in range(5):
        e = np.zeros(5); e[j] = h
        fd = (_tie_logdens_grad(mm + e, D, [1, 3])[0] - _tie_logdens_grad(mm - e, D, [1, 3])[0]) / (2 * h)
        assert abs(fd - g2[j]) < 1e-6


def test_tie_update_reduces_to_winner():
    a = update_winner_correlated(M, VV, 3, L, 1.0)
    b = update_tie_correlated(M, VV, [3], V=L, beta2=1.0)
    np.testing.assert_allclose(a[0], b[0], atol=2e-6)
    np.testing.assert_allclose(a[1], b[1], rtol=5e-4)
    assert abs(a[2] - b[2]) < 1e-6


def test_tie_update_matches_mc_posterior_in_the_narrow_band_limit():
    # the event has probability zero; the referee conditions on the pair
    # finishing within eps of each other and both above the rest
    rng = np.random.default_rng(13)
    s, X = correlated_draws(rng, 6_000_000, M, VV, L, 1.0)
    i, j = 3, 0
    others = [k for k in range(5) if k not in (i, j)]
    lo = np.minimum(X[:, i], X[:, j])
    keep = (np.abs(X[:, i] - X[:, j]) < 0.03) & (X[:, others].max(axis=1) < lo)
    s_t = s[keep]
    m_mc, v_mc = s_t.mean(axis=0), s_t.var(axis=0)
    m_hat, v_hat, _ = update_tie_correlated(M, VV, [i, j], V=L, beta2=1.0)
    se = np.sqrt(v_mc / len(s_t))
    assert len(s_t) > 10_000
    assert np.abs(m_hat - m_mc).max() < 4 * se.max() + 5e-3, (m_hat, m_mc)
    assert np.abs(v_hat - v_mc).max() < 0.02
    # a tie between the two says more about them than one winning alone
    _, v_w, _ = update_winner_correlated(M, VV, i, L, 1.0)
    assert v_hat[j] < v_w[j]


@pytest.mark.parametrize("bad", [[0, 0], [5], [], [True]])
def test_tie_label_door(bad):
    with pytest.raises(ValueError):
        update_tie_correlated(M, VV, bad, V=L)
