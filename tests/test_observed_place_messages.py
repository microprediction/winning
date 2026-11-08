"""Two more kinds of evidence a win node accepts: "i finished among the
top k" and "the top-k set was S" (order unknown). Both ride the factor
mixture of update_winner_correlated; both are refereed by rejection-
sampled Monte Carlo posteriors of the generative model, as that update
is, and reduce to it at k = 1."""
import numpy as np
import pytest

from winning.ratings.nway import (update_winner_correlated, update_topk_correlated,
                                  update_topset_correlated, _topset_logp_grad)
from winning.ratings.simulate import correlated_draws
from winning.factor.races import race_probabilities
from winning.factor.topk import top_k_probabilities

M = np.array([0.3, 0.0, -0.2, 0.1, -0.4])
VV = np.array([0.8, 1.0, 0.6, 1.2, 0.9])
L = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])


def test_k_equals_one_is_the_observed_winner():
    for V in (None, L):
        a = update_winner_correlated(M, VV, 3, np.zeros((5, 1)) if V is None else V, 1.0)
        b = update_topk_correlated(M, VV, 3, 1, V=V, beta2=1.0)
        c = update_topset_correlated(M, VV, [3], V=V, beta2=1.0)
        for x, y in ((a, b), (a, c)):
            np.testing.assert_allclose(x[0], y[0], atol=2e-6)
            np.testing.assert_allclose(x[1], y[1], rtol=2e-4)
            assert abs(x[2] - y[2]) < 1e-6


def test_topset_probability_and_gradient_are_exact_on_the_independent_race():
    rng = np.random.default_rng(3)
    mm = rng.normal(size=6); D = rng.uniform(0.5, 2.0, size=6)
    top = [1, 4]
    lp, g = _topset_logp_grad(mm, D, top)
    # the gradient sums to zero: a common shift of every mean moves nothing
    assert abs(g.sum()) < 1e-9
    # against central finite differences of the same lattice integral
    h = 1e-5
    for j in range(6):
        e = np.zeros(6); e[j] = h
        fd = (_topset_logp_grad(mm + e, D, top)[0] - _topset_logp_grad(mm - e, D, top)[0]) / (2 * h)
        assert abs(fd - g[j]) < 1e-6, (j, fd, g[j])
    # |top| = 1 is the win probability of the race, min-wins engine
    lp1, _ = _topset_logp_grad(mm, D, [2])
    assert abs(np.exp(lp1) - race_probabilities(-mm, D=D, points=2001)[2]) < 1e-9
    # |top| = n-1 names the last finisher: the win probability of -X
    lpn, _ = _topset_logp_grad(mm, D, [0, 1, 2, 3, 5])
    assert abs(np.exp(lpn) - race_probabilities(mm, D=D, points=2001)[4]) < 1e-9
    # members' gradients are nonnegative, outsiders' nonpositive
    assert (g[top] >= 0).all() and (np.delete(g, top) <= 0).all()


def test_topk_logz_is_the_membership_under_the_predictive():
    m, v, lz = update_topk_correlated(M, VV, 2, 2, V=None, beta2=1.0)
    q = top_k_probabilities(-M, 2, D=VV + 1.0)[2]
    assert abs(np.exp(lz) - q) < 1e-8


def _mc(rng, n, keep_fn):
    s, X = correlated_draws(rng, n, M, VV, L, 1.0)
    keep = keep_fn(X)
    return s[keep], keep.mean(), len(s)


def test_topk_update_matches_mc_posterior():
    rng = np.random.default_rng(10)
    s_k, p_mc, N = _mc(rng, 2_000_000, lambda X: (np.argsort(-X, axis=1)[:, :2] == 3).any(axis=1))
    m_mc, v_mc = s_k.mean(axis=0), s_k.var(axis=0)
    m_hat, v_hat, lz = update_topk_correlated(M, VV, 3, 2, V=L, beta2=1.0)
    se = np.sqrt(v_mc / len(s_k))
    assert np.abs(m_hat - m_mc).max() < 4 * se.max() + 2e-3, (m_hat, m_mc)
    assert np.abs(v_hat - v_mc).max() < 0.01
    assert abs(np.exp(lz) - p_mc) < 3 * np.sqrt(0.25 / N)


def test_topset_update_matches_mc_posterior():
    rng = np.random.default_rng(11)
    top = {0, 3}
    s_t, p_mc, N = _mc(rng, 2_000_000, lambda X: np.array([set(r[:2]) == top for r in np.argsort(-X, axis=1)]))
    m_mc, v_mc = s_t.mean(axis=0), s_t.var(axis=0)
    m_hat, v_hat, lz = update_topset_correlated(M, VV, [0, 3], V=L, beta2=1.0)
    se = np.sqrt(v_mc / len(s_t))
    assert np.abs(m_hat - m_mc).max() < 4 * se.max() + 3e-3, (m_hat, m_mc)
    assert np.abs(v_hat - v_mc).max() < 0.015
    assert abs(np.exp(lz) - p_mc) < 3 * np.sqrt(0.25 / N)
    # the set is more evidence than membership of one, less than the order
    m_k, v_k, _ = update_topk_correlated(M, VV, 0, 2, V=L, beta2=1.0)
    assert v_hat.mean() < v_k.mean()


@pytest.mark.parametrize("bad", [[0, 0], [5], [-1], [True], []])
def test_topset_label_door(bad):
    with pytest.raises(ValueError):
        update_topset_correlated(M, VV, bad, V=L)


@pytest.mark.parametrize("bad", [5, -1, True, 1.5])
def test_topk_label_door(bad):
    with pytest.raises(ValueError):
        update_topk_correlated(M, VV, bad, 2, V=L)
    with pytest.raises(ValueError):
        update_topk_correlated(M, VV, 1, 0, V=L)


def test_non_normal_base_is_refused_not_mispriced():
    with pytest.raises(NotImplementedError):
        update_topk_correlated(M, VV, 1, 2, V=L, base="laplace")
