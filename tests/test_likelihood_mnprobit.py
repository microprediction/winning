"""Tests for winning.likelihood and winning.mnprobit."""
import numpy as np
import pytest

from winning.likelihood import choice_loglik_and_score
from winning.mnprobit import MNProbit, MNProbitClassifier


def test_score_matches_finite_differences_both_branches():
    rng = np.random.default_rng(2)
    T, J, r = 40, 4, 2
    mu = rng.normal(size=(T, J))
    choice = rng.integers(0, J, T)
    for scale in (0.5, 6.0):        # mild (GH) and sharp (Sobol) branches
        V = rng.normal(size=(J, r)) * scale
        ll, dmu, dV = choice_loglik_and_score(mu, V, choice)
        h = 1e-6
        for (t, j) in [(0, 0), (5, 2), (17, 3)]:
            mp = mu.copy(); mp[t, j] += h
            mm = mu.copy(); mm[t, j] -= h
            lp = choice_loglik_and_score(mp, V, choice)[0]
            lm = choice_loglik_and_score(mm, V, choice)[0]
            assert abs((lp - lm) / (2 * h) - dmu[t, j]) < 1e-6
        for (a, b) in [(1, 0), (3, 1)]:
            Vp = V.copy(); Vp[a, b] += h
            Vm = V.copy(); Vm[a, b] -= h
            lp = choice_loglik_and_score(mu, Vp, choice)[0]
            lm = choice_loglik_and_score(mu, Vm, choice)[0]
            assert abs((lp - lm) / (2 * h) - dV[a, b]) < 1e-6


def test_common_utility_shift_is_invisible():
    rng = np.random.default_rng(3)
    T, J = 30, 4
    mu = rng.normal(size=(T, J))
    V = rng.normal(size=(J, 2)) * 0.6
    choice = rng.integers(0, J, T)
    ll1 = choice_loglik_and_score(mu, V, choice)[0]
    ll2 = choice_loglik_and_score(mu + 3.7, V, choice)[0]
    assert abs(ll1 - ll2) < 1e-9


def test_zero_loadings_reduce_to_independent_probit():
    # with V = 0 and D = 1, P(k wins) has an exact 1-D integral referee
    from scipy.integrate import quad
    from scipy.stats import norm
    rng = np.random.default_rng(4)
    J = 3
    mu = rng.normal(size=(1, J))
    V = np.zeros((J, 1))

    def p_exact(k):
        rivals = [j for j in range(J) if j != k]

        def f(z):
            val = norm.pdf(z)
            for j in rivals:
                val *= norm.cdf(mu[0, k] - mu[0, j] + z)
            return val
        return quad(f, -10, 10, epsabs=1e-13)[0]

    for k in range(J):
        ll = choice_loglik_and_score(mu, V, np.array([k]), Qz=31)[0]
        assert abs(np.exp(ll) - p_exact(k)) < 1e-8


def test_synthetic_recovery_direction():
    rng = np.random.default_rng(5)
    T, J = 1500, 3
    X = rng.normal(size=(T, J, 1))
    beta_true = np.array([0.0, 0.0, 1.2])   # intercepts, then slope
    V_true = np.array([[0.0], [0.8], [0.4]])
    Z = np.zeros((T, J, 2))
    for j in range(1, J):
        Z[:, j, j - 1] = 1.0
    mu = np.concatenate([Z, X], axis=2) @ beta_true
    eps = (V_true @ rng.normal(size=(1, T))).T + rng.normal(size=(T, J))
    choice = (mu + eps).argmax(axis=1)
    m = MNProbit(X, choice, r=1).fit(polish=False)
    # slope recovered with the right sign and rough magnitude
    assert m.params_[-1] > 0.6
    ll_null = choice_loglik_and_score(
        np.zeros((T, J)), np.zeros((J, 1)), choice)[0]
    assert m.loglik_ > ll_null


def test_classifier_api():
    rng = np.random.default_rng(6)
    T, J = 300, 3
    X = rng.normal(size=(T, J, 1))
    y = rng.integers(0, J, T)
    clf = MNProbitClassifier(r=1).fit(X, y)
    P = clf.predict_proba(X)
    assert P.shape == (T, J)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
    assert clf.predict(X).shape == (T,)
