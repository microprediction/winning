"""Mixed Plackett-Luce ranking likelihood with analytic score.

The likelihood member the measured ranking bias demanded: consuming
rankings stagewise is exact under the (conditionally) Gumbel model, so
this likelihood is safe for structure learning where the Gaussian-base
stagewise shortcut inflated learned correlation threefold.
"""
import numpy as np

from winning.factor.races import harville_order_logprob
from winning.likelihood import ranking_loglik_and_score


def _world(rng, T, J, s, profile):
    V = s * profile
    f = rng.normal(size=(T, 1))
    G = -np.log(-np.log(rng.random((T, J))))
    U = f @ V.T + G
    orders = [np.argsort(-U[t]) for t in range(T)]
    return orders


def test_score_matches_finite_differences():
    rng = np.random.default_rng(0)
    T, J, r = 5, 5, 2
    mu = rng.normal(size=(T, J))
    V = rng.normal(size=(J, r)) * 0.6
    orders = [rng.permutation(J)[: rng.integers(1, J + 1)] for _ in range(T)]
    ll, dmu, dV = ranking_loglik_and_score(mu, V, orders)
    eps = 1e-6
    for t in (0, 3):
        for j in (0, 4):
            m2 = mu.copy()
            m2[t, j] += eps
            fd = (ranking_loglik_and_score(m2, V, orders)[0] - ll) / eps
            assert abs(fd - dmu[t, j]) < 1e-4
    V2 = V.copy()
    V2[1, 0] += eps
    fd = (ranking_loglik_and_score(mu, V2, orders)[0] - ll) / eps
    assert abs(fd - dV[1, 0]) < 1e-4


def test_common_shift_invariance_and_harville_consistency():
    rng = np.random.default_rng(1)
    T, J = 4, 5
    mu = rng.normal(size=(T, J))
    orders = [rng.permutation(J) for _ in range(T)]
    ll, _, _ = ranking_loglik_and_score(mu, np.zeros((J, 1)), orders)
    ll_shift, _, _ = ranking_loglik_and_score(mu + 3.1, np.zeros((J, 1)),
                                              orders)
    assert abs(ll - ll_shift) < 1e-12
    # V = 0 reduces to independent Harville (max-wins negation)
    ll_h = sum(harville_order_logprob(-mu[t], orders[t]) for t in range(T))
    assert abs(ll - ll_h) < 1e-12


def test_exact_against_monte_carlo():
    # the likelihood IS the mixed-PL order probability: check against
    # 2e5-draw simulation of the generative world (measured at 2e6
    # draws: agreement at the MC noise floor, quadrature converged
    # Qf=7 vs 21 to 1e-7)
    rng = np.random.default_rng(0)
    J = 4
    mu = np.array([0.3, -0.2, 0.5, 0.0])
    V = np.array([[0.9], [0.2], [0.6], [1.1]])
    M = 200_000
    f = rng.normal(size=(M, 1))
    G = -np.log(-np.log(rng.random((M, J))))
    U = mu + f @ V.T + G
    from collections import Counter
    counts = Counter(map(tuple, np.argsort(-U, axis=1)))
    checked = 0
    for perm, cnt in counts.items():
        if cnt < 2000:
            continue
        ll, _, _ = ranking_loglik_and_score(mu[None, :], V,
                                            [np.array(perm)])
        assert abs(np.exp(ll) - cnt / M) < 3e-3
        checked += 1
    assert checked >= 5


def test_correlation_scale_recovered_from_rankings():
    # identification lesson applied to the test itself: a near-common
    # loading profile is choice-irrelevant, so use the family-vs-
    # outsider CONTRAST world (half loaded, half not); rankings through
    # the exact likelihood then pin the scale
    rng = np.random.default_rng(7)
    J, T = 6, 800
    profile = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])[:, None]
    s_true = 0.8
    orders = _world(rng, T, J, s_true, profile)
    mu0 = np.zeros((T, J))
    grid = np.linspace(0.2, 2.0, 19)
    lls = [ranking_loglik_and_score(mu0, s * profile, orders)[0]
           for s in grid]
    s_hat = grid[int(np.argmax(lls))]
    assert abs(s_hat - s_true) < 0.25
