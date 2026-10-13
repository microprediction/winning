"""The own-noise order defaults to 7 below J = 6 and 11 from J = 6 (#173).

Seven own-noise nodes left J = 10 weak-factor fields up to 5.3e-3 nats
off a tight MVN reference; eleven take that to 5.3e-4 for 1.0-1.6x the
cost. Smaller fields must not move at all.
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from winning.likelihood import choice_loglik_and_score, default_qz


def _field(J, r, seed=0):
    g = np.random.default_rng(seed)
    mu = g.normal(size=(5, J))
    V = 0.4 * g.normal(size=(J, r))
    ch = g.integers(J, size=5)
    return mu, V, ch


def test_default_qz_rule():
    assert [default_qz(J) for J in (2, 3, 4, 5, 6, 7, 10, 30)] == \
        [7, 7, 7, 7, 11, 11, 11, 11]


def test_small_fields_are_bitwise_the_seven_node_rule():
    for J in (3, 4, 5):
        for r in (0, 1, 2):
            mu, V, ch = _field(J, r, seed=J + 10 * r)
            a = choice_loglik_and_score(mu, V, ch)
            b = choice_loglik_and_score(mu, V, ch, Qz=7)
            assert a[0] == b[0]
            assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_large_fields_use_eleven_nodes_and_an_explicit_qz_wins():
    for J in (6, 8):
        mu, V, ch = _field(J, 1, seed=J)
        a = choice_loglik_and_score(mu, V, ch)[0]
        assert a == choice_loglik_and_score(mu, V, ch, Qz=11)[0]
        assert a != choice_loglik_and_score(mu, V, ch, Qz=7)[0]


def test_ten_runner_independent_probit_is_accurate():
    mu = np.linspace(-2.0, 2.0, 10)
    V = np.zeros((10, 0))
    for k in (0, 5, 9):
        rivals = np.delete(mu, k)
        ref = np.log(quad(
            lambda z: norm.pdf(z) * np.prod(norm.cdf(mu[k] - rivals + z)),
            -12, 12, epsabs=1e-15, epsrel=1e-13, limit=500)[0])
        new = choice_loglik_and_score(mu[None], V, [k])[0]
        old = choice_loglik_and_score(mu[None], V, [k], Qz=7)[0]
        assert abs(new - ref) < 1e-5
        assert abs(old - ref) > 5 * abs(new - ref)
