"""A ranking's factor posterior is narrower than a choice's (#1017).

`ranking_loglik_and_score` took the fixed 7-node Gauss-Hermite tensor
whenever the loading sharpness was at most 2 (#272), a threshold measured
on a SINGLE softmax. A k-stage ranking multiplies k softmaxes, so its
factor posterior has width about 1/(sharpness*sqrt(k)) and the prior-
centred tensor misses it: J = 5, rank 1, sharpness 1.91, a full order
came back 0.090 nats high with the loading score about half its true
value, and the error grows with the ranking length (0.32 nats at J = 10).

The fix moves the same tensor to each ranking's posterior mode, scaled
by the posterior curvature (adaptive Gauss-Hermite), when k >= 2 and
sharpness*sqrt(k) > 2. Everything outside that gate is the old rule.
The references below are one-dimensional `quad` integrals of the exact
mixed Plackett-Luce likelihood, so they test the TRUE likelihood, which
finite differences against the rule itself cannot.
"""
import numpy as np
import pytest
from scipy.integrate import quad

import winning.likelihood as L
from winning.likelihood import (_factor_nodes, ranking_loglik_and_score,
                                sharpness_bound)


def _pl_logp(mu, V, order, F, tau=1.0):
    z = (mu[None, :] + F @ V.T) / tau
    alive = np.ones(len(mu), dtype=bool)
    lp = np.zeros(len(F))
    for w in order:
        zz = np.where(alive[None, :], z, -np.inf)
        m = zz.max(axis=1, keepdims=True)
        lp += z[:, w] - (m[:, 0] + np.log(np.exp(zz - m).sum(axis=1)))
        alive[w] = False
    return lp


def _quad_loglik(mu, V, order, tau=1.0):
    def integrand(x):
        return (np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
                * np.exp(_pl_logp(mu, V, order, np.array([[x]]), tau)[0]))
    return np.log(quad(integrand, -12, 12, epsabs=1e-15, epsrel=1e-12,
                       limit=400)[0])


def _fixed_tensor_loglik(mu, V, order, tau=1.0):
    F, W = _factor_nodes(V.shape[1], sharp=0.0)
    a = np.log(W) + _pl_logp(mu, V, order, F, tau)
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def _field(J, seed, sharp=1.9, r=1):
    rng = np.random.default_rng(1000 * J + seed)
    mu = rng.standard_normal(J)
    V = rng.standard_normal((J, r))
    V *= sharp / sharpness_bound(V, n=J)
    return mu, V, list(rng.permutation(J))


@pytest.mark.parametrize("J", [5, 10])
def test_full_rankings_match_quad_worst_over_seeds_1017(J):
    errs, fixed = [], []
    for seed in range(8):
        mu, V, order = _field(J, seed)
        ll, _, _ = ranking_loglik_and_score(mu[None, :], V, [order])
        ref = _quad_loglik(mu, V, order)
        errs.append(abs(ll - ref))
        fixed.append(abs(_fixed_tensor_loglik(mu, V, order) - ref))
    assert max(errs) <= 1e-3, max(errs)
    # the old rule on the same fields: this test can see the bug
    assert max(fixed) > 1e-2, max(fixed)


def test_temperature_enters_the_adapted_rule_1017():
    mu, V, order = _field(6, 3, sharp=1.0)
    tau = 0.55                          # sharpness of V/tau is about 1.8
    assert sharpness_bound(V / tau, n=6) * np.sqrt(6) > 2
    ll, _, _ = ranking_loglik_and_score(mu[None, :], V, [order],
                                        temperature=tau)
    assert abs(ll - _quad_loglik(mu, V, order, tau)) <= 1e-3


def test_score_tracks_the_true_likelihood_1017():
    # The score is the exact derivative of the rule with its nodes held
    # where they were placed. Finite differences also move the nodes (the
    # mode and curvature depend on mu and V); that term is quadrature-
    # error sized. Both comparisons are against the TRUE likelihood too.
    mu, V, order = _field(5, 0)
    mu = mu[None, :]
    _, dmu, dV = ranking_loglik_and_score(mu, V, [order])
    J = 5
    own_v, true_v, own_m, true_m = (np.zeros(J) for _ in range(4))
    for j in range(J):
        e = np.zeros((J, 1))
        e[j, 0] = 1.0
        own_v[j] = (ranking_loglik_and_score(mu, V + 1e-6 * e, [order])[0]
                    - ranking_loglik_and_score(mu, V - 1e-6 * e, [order])[0]
                    ) / 2e-6
        true_v[j] = (_quad_loglik(mu[0], V + 1e-4 * e, order)
                     - _quad_loglik(mu[0], V - 1e-4 * e, order)) / 2e-4
        d = np.zeros_like(mu)
        d[0, j] = 1.0
        own_m[j] = (ranking_loglik_and_score(mu + 1e-6 * d, V, [order])[0]
                    - ranking_loglik_and_score(mu - 1e-6 * d, V, [order])[0]
                    ) / 2e-6
        true_m[j] = (_quad_loglik(mu[0] + 1e-4 * d[0], V, order)
                     - _quad_loglik(mu[0] - 1e-4 * d[0], V, order)) / 2e-4
    for got, own, true in [(dV[:, 0], own_v, true_v), (dmu[0], own_m, true_m)]:
        scale = np.abs(true).max()
        assert np.abs(got - own).max() / scale < 2e-3
        assert np.abs(got - true).max() / scale < 2e-3


def test_rank_two_full_rankings_against_a_dense_grid_1017():
    x = np.linspace(-8, 8, 801)
    h = x[1] - x[0]
    X, Y = np.meshgrid(x, x, indexing="ij")
    F = np.column_stack([X.ravel(), Y.ravel()])
    errs, fixed = [], []
    for seed in range(8):
        mu, V, order = _field(6, seed, sharp=1.9, r=2)
        lw = (-0.5 * (F ** 2).sum(axis=1) - np.log(2 * np.pi)
              + _pl_logp(mu, V, order, F))
        m = lw.max()
        ref = m + np.log(np.exp(lw - m).sum() * h * h)
        ll, _, _ = ranking_loglik_and_score(mu[None, :], V, [order])
        errs.append(abs(ll - ref))
        fixed.append(abs(_fixed_tensor_loglik(mu, V, order) - ref))
    assert max(errs) <= 1e-3, max(errs)
    assert max(fixed) > 1e-2, max(fixed)


def _forbid_adaptive(*args, **kwargs):
    raise AssertionError("adaptive rule taken outside its gate")


@pytest.mark.parametrize("r, sharp, k", [
    (1, 1.95, 1),        # a length-1 order is the choice likelihood
    (2, 1.95, 1),
    (1, 1.25, 2),        # sharp*sqrt(k) = 1.77, below the blend band
    (2, 0.55, 10),       # 1.74; (1.8, 2] blends the two rules (#1063)
    (1, 3.0, 6),         # Sobol branch
    (3, 1.5, 6),         # rank 3: Sobol
    (0, 0.0, 6),         # rank 0: the empty product
])
def test_rule_is_unchanged_outside_the_gate_1017(monkeypatch, r, sharp, k):
    monkeypatch.setattr(L, "_ranking_adaptive_nodes", _forbid_adaptive)
    rng = np.random.default_rng(int(10 * sharp) + r + k)
    J = 10
    mu = rng.standard_normal((3, J))
    V = rng.standard_normal((J, r))
    if r:
        V *= sharp / sharpness_bound(V, n=J)
    orders = [list(rng.permutation(J)[:k]) for _ in range(3)]
    ranking_loglik_and_score(mu, V, orders)


def test_gate_is_per_order_length_1017(monkeypatch):
    # In one batch only the long orders are adapted.
    seen = []
    real = L._ranking_adaptive_nodes

    def spy(mu_t, V, order, tau, X):
        seen.append(len(order))
        return real(mu_t, V, order, tau, X)

    monkeypatch.setattr(L, "_ranking_adaptive_nodes", spy)
    mu, V, order = _field(8, 0, sharp=1.2)    # gate at k > 2.78
    orders = [order[:1], order[:2], order[:3], order]
    ranking_loglik_and_score(np.tile(mu, (4, 1)), V, orders)
    assert seen == [3, 8]


def test_mode_hessian_ignores_a_huge_common_loading():
    # a loading shared by every alternative cancels from each stage; the
    # posterior Hessian must not overflow on it (review on #1061)
    from winning.likelihood import _ranking_mode
    mu = np.zeros(4)
    V0 = np.array([[0, -1], [0, -.5], [0, .5], [0, 1.]])
    V = V0 + np.array([2.0 ** 600, 0.0])
    _, H = _ranking_mode(mu, V, [0, 1, 2, 3], 1.0)
    _, H0 = _ranking_mode(mu, V0, [0, 1, 2, 3], 1.0)
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H, H0, atol=1e-12)
