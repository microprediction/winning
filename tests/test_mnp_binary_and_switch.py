"""MNP choice likelihood: the binary closed form (#128, #270, #345,
#352) and a continuous node-family switch (#420, #213)."""

import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import norm

from winning import mnprobit as M
from winning.likelihood import choice_loglik_and_score, sharpness_bound


def _pair_oracle(g, a=0.0, D=(1.0, 1.0)):
    s = np.sqrt(D[0] + D[1] + a * a)
    z = g / s
    lp = log_ndtr(z)
    lam = np.exp(norm.logpdf(z) - lp)
    return lp, lam / s, -lam * g * a / s ** 3      # value, d/dg, d/da


@pytest.mark.parametrize("g", [-60.0, -20.0, -8.0, -3.0, 0.0, 2.5])
@pytest.mark.parametrize("a", [0.0, 1.3, 4.3, 3 * np.sqrt(2.0)])
def test_binary_value_and_score_are_the_contrast(g, a):
    mu = np.array([[g, 0.0]])
    V = np.array([[0.0], [a]])
    ll, dmu, dV = choice_loglik_and_score(mu, V, [0])
    lp, dg, da = _pair_oracle(g, a)
    assert abs(ll - lp) <= 1e-12 * max(1.0, abs(lp))
    assert abs(dmu[0, 0] - dg) <= 1e-12 * max(1.0, dg)
    assert abs(dmu[0, 1] + dg) <= 1e-12 * max(1.0, dg)
    assert abs(dV[1, 0] - da) <= 1e-12 * max(1.0, abs(da))
    assert abs(dV[0, 0] + da) <= 1e-12 * max(1.0, abs(da))


def test_binary_ignores_an_unused_loading_column():
    mu = np.array([[-7.5, 0.0]])
    a = choice_loglik_and_score(mu, np.array([[0.0], [4.3]]), [0])
    b = choice_loglik_and_score(mu, np.array([[0.0, 0.0], [4.3, 0.0]]), [0])
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    assert np.array_equal(b[2][:, 1], np.zeros(2))


@pytest.mark.parametrize("eps", [1e-4, 1e-8, 0.0])
@pytest.mark.parametrize("k", [0, 1])
def test_binary_near_deterministic_rival(eps, k):
    ll, dmu, _ = choice_loglik_and_score(np.zeros((1, 2)), np.zeros((2, 1)),
                                         [k], D=[eps, 1.0])
    assert abs(np.exp(ll) - 0.5) < 1e-15
    assert abs(dmu[0, k] - np.sqrt(2 / np.pi) / np.sqrt(1 + eps)) < 1e-14


def test_a_deterministic_alternative_is_refused_at_J3():
    with pytest.raises(ValueError, match="D_j > 0"):
        choice_loglik_and_score(np.zeros((1, 3)), np.zeros((3, 1)), [1],
                                D=[0.0, 1.0, 1.0])


def _field(seed=0, T=30, J=4):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(T, J)), rng.integers(0, J, T),
            rng.normal(size=(J, 2)))


@pytest.mark.parametrize("target", [2.9, 3.0, 3.1])
def test_the_family_switch_is_continuous_in_the_loadings(target):
    """A 1e-10 relative change of V across the threshold moved the
    binary log-likelihood by 0.0253 nats (#420); at J = 4 the switch is
    now a blend, so value and score move by O(1e-10)."""
    mu, ch, V0 = _field()
    V0 = V0 * (target / sharpness_bound(V0))
    lo = choice_loglik_and_score(mu, V0 * (1 - 1e-10), ch)
    hi = choice_loglik_and_score(mu, V0 * (1 + 1e-10), ch)
    assert abs(lo[0] - hi[0]) < 1e-7
    assert np.abs(lo[2] - hi[2]).max() < 1e-6


@pytest.mark.parametrize("target", [2.95, 3.0, 3.05])
def test_the_blend_score_is_the_gradient_of_the_blend(target):
    mu, ch, V0 = _field(seed=1)
    V = V0 * (target / sharpness_bound(V0))
    _, dmu, dV = choice_loglik_and_score(mu, V, ch)
    h = 1e-6
    for (a, c) in [(1, 0), (3, 1), (0, 1)]:
        Vp = V.copy(); Vp[a, c] += h
        Vm = V.copy(); Vm[a, c] -= h
        fd = (choice_loglik_and_score(mu, Vp, ch)[0]
              - choice_loglik_and_score(mu, Vm, ch)[0]) / (2 * h)
        assert abs(fd - dV[a, c]) < 1e-6, (a, c, fd, dV[a, c])
    mp = mu.copy(); mp[4, 2] += h
    mm = mu.copy(); mm[4, 2] -= h
    fd = (choice_loglik_and_score(mp, V, ch)[0]
          - choice_loglik_and_score(mm, V, ch)[0]) / (2 * h)
    assert abs(fd - dmu[4, 2]) < 1e-6


def test_a_common_loading_offset_cannot_switch_prediction():
    """#213 reopening: an offset of 100 moved the bound from 3.0 to
    3.0000000000000044 and swapped 49 Hermite nodes for 1024 Sobol."""
    V = np.array([[0.0], [3.0 / np.sqrt(2.0) / (2.0 / 3.0)], [0.0]])
    V = V * (3.0 / sharpness_bound(V))
    mu = np.array([[0.1, -0.2, 0.3]])
    p0 = np.array([M._prob_of(mu, V, k)[0] for k in range(3)])
    p1 = np.array([M._prob_of(mu, V + 100.0, k)[0] for k in range(3)])
    assert np.abs(p0 - p1).max() < 1e-9, (p0, p1)


def test_prediction_is_the_likelihood():
    mu, ch, V0 = _field(seed=2, T=5)
    V = V0 * (3.0 / sharpness_bound(V0))
    for k in range(4):
        want = np.array([np.exp(choice_loglik_and_score(mu[[t]], V, [k])[0])
                         for t in range(len(mu))])
        assert np.allclose(M._prob_of(mu, V, k), want, rtol=1e-13, atol=0)
