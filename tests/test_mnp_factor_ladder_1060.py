"""MNP choice likelihood: the factor tensor's order rises with the
loading sharpness (#1060).

The fixed 7^r Gauss-Hermite factor tensor under-resolved the conditional
choice probability below the sharpness-3 Sobol switch: the exchangeable
three-way race priced 0.3296 against an exact 1/3, and random fields at
sharpness 2.5-2.9 were up to 0.16 nats off. Past a C1 blend band at
smooth sharpness 0.8-1.0 the tensor has 31 nodes per factor at rank 1
and 11 at rank 2; below the band (and past the Sobol switch) every
observation is priced exactly as before.
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import winning.likelihood as L
from winning.likelihood import (_choice_logprob_terms, _choice_terms,
                                _smooth_sharpness, choice_loglik_and_score,
                                default_qz, nodes_for_likelihood)

S3 = np.sqrt(3.0)
V_EXCH = np.array([[2.0, 0.0], [-1.0, S3], [-1.0, -S3]])   # sharpness 2.927


def _probs(V):
    lp = _choice_logprob_terms(np.zeros((3, 3)), V, None, np.arange(3),
                               score=False)[0]
    return np.exp(lp)


def test_exchangeable_race_is_one_third():
    """0.3296 / 0.3353 / 0.3353 on the fixed 7x7 tensor; exact 1/3."""
    p = _probs(V_EXCH)
    assert np.abs(p - 1.0 / 3.0).max() < 1.5e-3


def test_exchangeable_race_rotated_by_15_degrees():
    """The same law rotated: 0.3287 / 0.3287 / 0.3429 before."""
    th = np.deg2rad(15.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    p = _probs(V_EXCH @ R)
    assert np.abs(p - 1.0 / 3.0).max() < 3e-3
    assert p.max() - p.min() < 5e-3


def _exact_logp(mu, V, k):
    J = len(mu)
    rv = [j for j in range(J) if j != k]
    d = V[k] - V[rv]
    cov = d @ d.T + 1.0 + np.eye(J - 1)
    # tolerances on .cdf, not the frozen constructor: older scipy (the
    # python 3.10 CI runner) refuses them there
    return np.log(multivariate_normal.cdf(
        np.zeros(J - 1), mean=mu[rv] - mu[k], cov=cov, abseps=1e-12,
        releps=1e-10))


# worst |log p - exact| over 6 fields x every choice; the fixed tensor
# gave r=1: 1.1e-3, 1.2e-2, 2.9e-2, 9.2e-2 and r=2: 9.3e-4, 4.6e-3,
# 2.4e-2, 3.5e-2 at these sharpnesses
@pytest.mark.parametrize("r,target,tol", [
    (1, 1.5, 1e-4), (1, 2.0, 1e-4), (1, 2.5, 1e-3), (1, 2.8, 1e-3),
    (2, 1.5, 2e-4), (2, 2.0, 2e-3), (2, 2.5, 1e-2), (2, 2.8, 1.2e-2)])
def test_accuracy_against_exact_mvn_below_the_sobol_switch(r, target, tol):
    rng = np.random.default_rng([1060, r, int(10 * target)])
    worst = 0.0
    for J in (3, 4):
        for _ in range(3):
            V = rng.normal(size=(J, r))
            V *= target / _smooth_sharpness(V, np.ones(J))[0]
            mu = rng.normal(scale=0.7, size=J)
            lp = _choice_logprob_terms(np.tile(mu, (J, 1)), V, None,
                                       np.arange(J), score=False)[0]
            ex = np.array([_exact_logp(mu, V, k) for k in range(J)])
            worst = max(worst, np.abs(lp - ex).max())
    assert worst < tol


def _field(r, J=5, T=12, seed=0):
    rng = np.random.default_rng([seed, r, J])
    mu = rng.normal(size=(T, J))
    ch = rng.integers(0, J, T)
    V = rng.normal(size=(J, r))
    return mu, ch, V / _smooth_sharpness(V, np.ones(J))[0]


@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("edge", [0.8, 0.9, 1.0])
def test_the_ladder_band_is_continuous_and_c1(r, edge):
    """Value and loading score move by O(eps) across either edge of the
    band and inside it: no jump in the likelihood or its slope."""
    mu, ch, V0 = _field(r)
    for eps in (1e-6, 1e-9):
        lo = choice_loglik_and_score(mu, V0 * (edge - eps), ch)
        hi = choice_loglik_and_score(mu, V0 * (edge + eps), ch)
        assert abs(hi[0] - lo[0]) < 2e4 * eps
        assert np.abs(hi[2] - lo[2]).max() < 2e4 * eps
        assert np.abs(hi[1] - lo[1]).max() < 2e4 * eps


@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("target", [0.85, 0.95, 0.999, 1.5, 2.6, 2.95])
def test_score_is_the_derivative_of_the_returned_value(r, target):
    mu, ch, V0 = _field(r, J=4, T=6, seed=1)
    V = V0 * target
    ll, dmu, dV = choice_loglik_and_score(mu, V, ch)
    h = 1e-6
    num = np.zeros_like(V)
    for i in range(V.shape[0]):
        for c in range(r):
            Vp, Vm = V.copy(), V.copy()
            Vp[i, c] += h
            Vm[i, c] -= h
            num[i, c] = (choice_loglik_and_score(mu, Vp, ch)[0]
                         - choice_loglik_and_score(mu, Vm, ch)[0]) / (2 * h)
    assert np.abs(num - dV).max() < 1e-6
    t, j = 2, 1
    mp, mm = mu.copy(), mu.copy()
    mp[t, j] += h
    mm[t, j] -= h
    g = (choice_loglik_and_score(mp, V, ch)[0]
         - choice_loglik_and_score(mm, V, ch)[0]) / (2 * h)
    assert abs(g - dmu[t, j]) < 1e-6


def _fixed_tensor(mu, V, ch, Qf):
    J = mu.shape[1]
    F, W = nodes_for_likelihood(V.shape[1], Qf, default_qz(J), 0.0)
    return _choice_terms(mu, V - V.mean(axis=0), ch, np.ones(J), F, W)


@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("target", [0.3, 0.79, 0.8])
def test_below_the_band_the_fixed_tensor_is_unchanged_to_the_bit(r, target):
    mu, ch, V0 = _field(r, seed=2)
    V = V0 * target
    new = _choice_logprob_terms(mu, V, None, ch)
    old = _fixed_tensor(mu, V, ch, 7)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("r", [1, 2])
def test_past_the_band_the_tensor_is_the_high_order_one(r):
    mu, ch, V0 = _field(r, seed=3)
    V = V0 * 2.0
    new = _choice_logprob_terms(mu, V, None, ch)
    hi = _fixed_tensor(mu, V, ch, L._FACTOR_QF_HIGH[r])
    for a, b in zip(new, hi):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("r", [1, 2])
def test_a_caller_qf_at_or_above_the_ladder_keeps_its_tensor(r):
    mu, ch, V0 = _field(r, seed=4)
    q = L._FACTOR_QF_HIGH[r] + 2
    for target in (0.9, 2.0):
        V = V0 * target
        new = _choice_logprob_terms(mu, V, None, ch, Qf=q)
        old = _fixed_tensor(mu, V, ch, q)
        for a, b in zip(new, old):
            assert np.array_equal(a, b)


def test_rank_three_has_no_ladder():
    mu, ch, V0 = _field(3, seed=5)
    V = V0 * 2.0
    new = _choice_logprob_terms(mu, V, None, ch)
    old = _fixed_tensor(mu, V, ch, 7)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("r", [1, 2])
def test_past_the_sobol_switch_nothing_changes(r):
    mu, ch, V0 = _field(r, seed=6)
    V = V0 * 3.5
    J = mu.shape[1]
    new = _choice_logprob_terms(mu, V, None, ch)
    F, W = nodes_for_likelihood(r, 7, default_qz(J), np.inf)
    old = _choice_terms(mu, V - V.mean(axis=0), ch, np.ones(J), F, W,
                        tilt=False)
    for a, b in zip(new, old):
        assert np.array_equal(a, b)
