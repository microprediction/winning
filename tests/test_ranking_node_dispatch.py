"""The mixed Plackett-Luce rule escalates with the loadings.

#272. `ranking_loglik_and_score` always called `_factor_nodes`, which
for rank one or two uses a FIXED Gauss-Hermite tensor of order `Qf`,
default 7 -- no dispatch on the loadings, no convergence check, no
warning. Unlike the choice likelihood, which escalates past a sharpness
of 3.

A length-one ranking is a mixed-logit choice probability, so a rank-one
factor has a direct one-dimensional reference integral and the failure
is exact, not a disagreement between two approximations:

    probability      0.6927739380  against  0.6136638494   (12.9% high)
    utility score    0.04810496    against  0.07341285
    loading score    0.00023193    against  0.02040027     (88x small)

Those are derivatives of the TRUE mixed likelihood, so no
finite-difference test against the same 7-node rule could ever see it,
and the Monte Carlo test used loadings near one where Qf=7 and Qf=21
agree.

Raising the order does not rescue it: the integrand is a softmax, and
Gauss-Hermite is non-monotone on it -- 0.6928, 0.6374, 0.6128, 0.6121,
0.6137 at 7, 15, 31, 63, 127, reaching 5.7e-8 only at 255. The
low-discrepancy rule the helper already uses past rank two gets 9.3e-8
from 1024 nodes.
"""
import numpy as np
import pytest
from scipy.integrate import quad

from winning.likelihood import (_FACTOR_SOBOL_SHARP, _factor_nodes,
                                ranking_loglik_and_score, sharpness_bound)

MU = np.array([-3.77297390, 3.21965260, -2.49073669, 0.70744695])
V1 = np.array([-0.27673232, -3.64790301, -0.85196895, 4.77660428])
WINNER = 1


def _softmax(z, mu=MU, v=V1):
    a = mu + v * z
    e = np.exp(a - a.max())
    return e / e.sum()


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


@pytest.fixture(scope="module")
def oracle():
    p = quad(lambda z: _phi(z) * _softmax(z)[WINNER], -12, 12,
             epsabs=1e-13, epsrel=1e-12)[0]
    dmu = quad(lambda z: _phi(z) * _softmax(z)[WINNER]
               * (1.0 - _softmax(z)[WINNER]), -12, 12, epsabs=1e-13)[0] / p
    dV = quad(lambda z: _phi(z) * z * _softmax(z)[WINNER]
              * (1.0 - _softmax(z)[WINNER]), -12, 12, epsabs=1e-13)[0] / p
    return p, dmu, dV


def test_probability_matches_the_one_dimensional_integral(oracle):
    p, _, _ = oracle
    ll, _, _ = ranking_loglik_and_score(MU[None, :], V1[:, None], [[WINNER]])
    assert abs(np.exp(ll) - p) / p < 1e-6, (np.exp(ll), p)


def test_the_scores_match_it_too(oracle):
    p, dmu_e, dV_e = oracle
    _, dmu, dV = ranking_loglik_and_score(MU[None, :], V1[:, None], [[WINNER]])
    assert abs(dmu[0, WINNER] - dmu_e) / abs(dmu_e) < 1e-5
    # the one that was 88x small
    assert abs(dV[WINNER, 0] - dV_e) / abs(dV_e) < 1e-4, (dV[WINNER, 0], dV_e)


def test_a_smooth_field_keeps_the_cheap_rule():
    """Gauss-Hermite is BETTER below the crossover, and 146 times
    cheaper -- 7 nodes against 1024 -- so it must not be abandoned."""
    V = np.array([0.3, -0.1, 0.2, -0.4])[:, None]
    assert sharpness_bound(V, n=4) < _FACTOR_SOBOL_SHARP
    F, _ = _factor_nodes(1, sharp=sharpness_bound(V, n=4))
    assert len(F) == 7


def test_a_sharp_field_escalates():
    assert sharpness_bound(V1[:, None], n=4) > _FACTOR_SOBOL_SHARP
    F, W = _factor_nodes(1, sharp=sharpness_bound(V1[:, None], n=4))
    assert len(F) == 1024
    assert abs(W.sum() - 1.0) < 1e-12


def test_the_dispatch_is_on_the_effective_loadings():
    """The loadings enter as V/tau, so a large temperature makes a
    sharp field smooth and must take the cheap rule."""
    hot, _, _ = ranking_loglik_and_score(MU[None, :], V1[:, None],
                                         [[WINNER]], temperature=20.0)
    assert np.isfinite(hot)
    # at tau = 20 the effective contrast is 4.78/20, well under the
    # crossover, so the cheap rule is taken -- and the answer still has
    # to be right
    assert sharpness_bound(V1[:, None] / 20.0, n=4) < _FACTOR_SOBOL_SHARP

    def tempered(z):
        a = (MU + V1 * z) / 20.0
        e = np.exp(a - a.max())
        return e / e.sum()

    ref = quad(lambda z: _phi(z) * tempered(z)[WINNER], -12, 12,
               epsabs=1e-13, epsrel=1e-12)[0]
    assert abs(np.exp(hot) - ref) / ref < 1e-6


@pytest.mark.parametrize("scale", [0.2, 0.5, 1.0, 2.0, 3.0])
def test_accuracy_holds_across_the_crossover(scale):
    """The dispatch must not leave a band where neither rule is good."""
    rng = np.random.default_rng(17)
    worst = 0.0
    for _ in range(6):
        mu = rng.standard_normal(4) * 2.0
        v = rng.standard_normal(4) * scale
        w = int(rng.integers(0, 4))
        ex = quad(lambda z: _phi(z) * _softmax(z, mu, v)[w], -12, 12,
                  epsabs=1e-13, epsrel=1e-12)[0]
        if ex < 1e-6:
            continue
        ll, _, _ = ranking_loglik_and_score(mu[None, :], v[:, None], [[w]])
        worst = max(worst, abs(np.exp(ll) - ex) / ex)
    assert worst < 5e-3, (scale, worst)
