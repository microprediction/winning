"""The ranking likelihood's node-rule switches are continuous (#1063).

`ranking_loglik_and_score` switched rules with two hard gates: the fixed
Gauss-Hermite tensor to adaptive Gauss-Hermite at sharpness*sqrt(k) = 2
(#1017), and Gauss-Hermite to scrambled Sobol at sharpness = 2 (#272).
The rules disagree slightly, so a loading moved by 1e-9 across either
gate moved the log-likelihood by up to 6e-4 nats per observation, and
the score could not see it. Both are now blended in probability over a
band with a C1 weight whose derivative the score carries.
"""
import numpy as np
import pytest

import winning.likelihood as L
from winning.likelihood import ranking_loglik_and_score, sharpness_bound

J, T = 5, 30


def _field(seed, r=1):
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=(J, r))
    return rng, v0 / sharpness_bound(v0, n=J), rng.normal(size=(T, J))


# (order length, sharpness): both gates and both band ends
POINTS = [(2, 2.0 / np.sqrt(2)), (3, 2.0 / np.sqrt(3)), (2, 1.8 / np.sqrt(2)),
          (1, 2.0), (4, 2.0), (1, 2.2), (4, 2.2)]


@pytest.mark.parametrize("r", [1, 2])
@pytest.mark.parametrize("k,s", POINTS)
def test_value_is_continuous_across_gates_and_band_ends(k, s, r):
    rng, v, mu = _field(7 + k, r)
    orders = [rng.permutation(J)[:k] for _ in range(T)]
    lo = ranking_loglik_and_score(mu, v * s * (1 - 1e-9), orders)[0]
    hi = ranking_loglik_and_score(mu, v * s * (1 + 1e-9), orders)[0]
    # the slope alone moves it by about 1e-8; the old gates jumped by 1e-3
    assert abs(hi - lo) < 1e-7


@pytest.mark.parametrize("seed", range(6))
def test_score_carries_the_blend_derivative(seed):
    # length-1 orders in the Sobol band: neither rule's nodes move with
    # V, so the score is the exact derivative, blend weight included
    rng, v, mu = _field(seed, r=1 + seed % 2)
    orders = [rng.permutation(J)[:1] for _ in range(T)]
    V = v * rng.uniform(2.03, 2.17)
    _, _, dV = ranking_loglik_and_score(mu, V, orders)
    D = rng.normal(size=V.shape)
    h = 1e-6
    num = (ranking_loglik_and_score(mu, V + h * D, orders)[0]
           - ranking_loglik_and_score(mu, V - h * D, orders)[0]) / (2 * h)
    assert abs(num - (dV * D).sum()) < 1e-6 * max(1.0, abs(num))


def test_outside_the_bands_the_rules_are_unblended(monkeypatch):
    # with the band machinery made to fail, fields outside both bands
    # still evaluate: they never reach it
    def boom(*a, **k):
        raise AssertionError("blend reached outside its band")
    monkeypatch.setattr(L, "_sharpness_grad", boom)
    for k, s in [(1, 1.5), (2, 1.0), (2, 1.6), (3, 1.9), (1, 2.3), (4, 3.0)]:
        rng, v, mu = _field(k, 1)
        orders = [rng.permutation(J)[:k] for _ in range(T)]
        ranking_loglik_and_score(mu, v * s, orders)


def test_blend_holds_under_temperature():
    # the gates are on V / tau: the blend must follow the scaled statistic
    rng, v, mu = _field(3, 1)
    orders = [rng.permutation(J)[:3] for _ in range(T)]
    tau = 1.7
    vals = [ranking_loglik_and_score(mu, v * 2.0 * tau * (1 + e), orders,
                                     temperature=tau)[0]
            for e in (-1e-9, 1e-9)]
    assert abs(vals[1] - vals[0]) < 1e-7
