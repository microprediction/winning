"""Every Gaussian two-runner rating kernel is the one-dimensional
contrast it mathematically is (#92).

For a pair with belief N(m, S), noise beta2 and loadings V, the event
"i beats j" is a' x > 0 with a = e_i - e_j, s^2 = a'Sa + b_i + b_j +
||V_i - V_j||^2, z = a'm / s, and the exact posterior is

    E[s | win]   = m + S a lam / s,          lam = phi(z) / Phi(z)
    Cov[s | win] = S - S a a' S lam (lam + z) / s^2

The lattice kernels missed a narrow runner next to a diffuse one: at
variances [1, 1e6] complementary order probabilities summed to 1.628
and the winner update returned means 2.5x too small with the loser's
variance floored at 1e-6 (truth: 3.6e5)."""

import numpy as np
import pytest
from scipy.stats import norm

from winning.ratings.full import update_order_full, update_winner_full
from winning.ratings.nway import (order_loglik, predictive_win_probabilities,
                                  update_order_correlated, update_ranking_exact,
                                  update_winner, update_winner_correlated)


def exact(m, S, i, beta2, V=None):
    m = np.asarray(m, float); S = np.asarray(S, float)
    b = np.broadcast_to(np.asarray(beta2, float), (2,))
    a = np.zeros(2); a[i], a[1 - i] = 1.0, -1.0
    s2 = a @ S @ a + b.sum()
    if V is not None:
        d = np.asarray(V, float)[i] - np.asarray(V, float)[1 - i]
        s2 += float(d @ d)
    s = np.sqrt(s2)
    z = (a @ m) / s
    lam = norm.pdf(z) / norm.cdf(z)
    Sa = S @ a
    return (m + Sa * lam / s, S - np.outer(Sa, Sa) * lam * (lam + z) / s2,
            float(norm.logcdf(z)))


RATIOS = [1.0, 1e2, 1e4, 1e6]


@pytest.mark.parametrize("big", RATIOS)
@pytest.mark.parametrize("i", [0, 1])
def test_independent_winner_and_order(big, i):
    m = np.array([0.3, -0.2]); v = np.array([1.0, big]); b = np.array([0.5, 2.0])
    em, eS, elz = exact(m, np.diag(v), i, b)
    mw, vw, pw = update_winner(m, v, i, beta2=b)
    assert np.allclose(mw, em, rtol=1e-6, atol=1e-12), (mw, em)
    assert np.allclose(vw, np.diag(eS), rtol=1e-4), (vw, np.diag(eS))
    assert abs(np.log(pw) - elz) < 1e-12
    mo, vo = update_ranking_exact(m, v, [i, 1 - i], beta2=b)
    assert np.allclose(mo, em, rtol=1e-6, atol=1e-12)
    assert np.allclose(vo, np.diag(eS), rtol=1e-4)


@pytest.mark.parametrize("big", RATIOS)
def test_complementary_orders_sum_to_one(big):
    m = np.array([0.4, 0.0]); sd = np.sqrt(np.array([1.0, big]) + 1.0)
    p01 = np.exp(order_loglik(m, sd, [0, 1])[0])
    p10 = np.exp(order_loglik(m, sd, [1, 0])[0])
    assert abs(p01 + p10 - 1.0) < 1e-14
    pred = predictive_win_probabilities(m, np.array([1.0, big]), beta2=1.0)
    assert abs(pred[0] - p01) < 1e-9


@pytest.mark.parametrize("big", [1.0, 1e4])
def test_correlated_winner_and_order(big):
    m = np.array([0.3, -0.2]); v = np.array([1.0, big]); b = 1.0
    V = np.array([[0.8], [-0.4]])
    em, eS, elz = exact(m, np.diag(v), 0, b, V=V)
    for mm, vv, lz in (update_winner_correlated(m, v, 0, V, beta2=b),
                       update_order_correlated(m, v, [0, 1], V, beta2=b)):
        assert abs(lz - elz) < 1e-6, (lz, elz)
        assert np.allclose(mm, em, rtol=1e-5, atol=1e-9), (mm, em)
        assert np.allclose(vv, np.diag(eS), rtol=1e-3), (vv, np.diag(eS))


def test_full_covariance_winner_equals_order_and_the_exact_contrast():
    rng = np.random.default_rng(9)
    L = rng.normal(size=(2, 2))
    S = L @ L.T + np.diag([0.5, 2.0])
    V = rng.normal(size=(2, 2))
    b = np.array([0.5, 2.0])
    m = rng.normal(size=2)
    a = update_winner_full(m, S, 0, V=V, beta2=b)
    o = update_order_full(m, S, [0, 1], V=V, beta2=b)
    assert np.allclose(a[0], o[0], atol=1e-12)
    assert np.allclose(a[1], o[1], atol=1e-12)
    assert abs(a[2] - o[2]) < 1e-12
    em, eS, elz = exact(m, S, 0, b, V=V)
    assert abs(a[2] - elz) < 5e-3
    assert np.allclose(a[0], em, atol=5e-3)
    assert np.allclose(a[1], eS, atol=2e-2)


def test_permuting_the_pair_permutes_the_answer():
    m = np.array([0.3, -0.2]); v = np.array([1.0, 1e5]); b = np.array([0.5, 2.0])
    a = update_winner(m, v, 0, beta2=b)
    c = update_winner(m[::-1], v[::-1], 1, beta2=b[::-1])
    assert np.allclose(a[0], c[0][::-1], rtol=1e-12)
    assert np.allclose(a[1], c[1][::-1], rtol=1e-9)
