"""The softmax special case, exposed analytically.

softmax_probabilities is closed form (no lattice); these tests pin it
against the engine's base="gumbel" lattice path, which is the point:
wherever the race is priced numerically the softmax twin is priced
exactly, making it the standing control variate and comparison line.
Convention under test: temperature tau <-> D = tau^2 pi^2/6.
"""
import numpy as np

from winning import race_probabilities
from winning.factor.races import (abilities_from_softmax,
                                  softmax_probabilities)

PI2_6 = np.pi ** 2 / 6.0


def test_independent_closed_form_equals_lattice_gumbel():
    mu = np.array([-0.8, -0.2, 0.1, 0.5, 1.1])
    for tau in (1.0, 2.0, 0.5):
        p_analytic = softmax_probabilities(mu, temperature=tau)
        p_lattice = race_probabilities(mu, D=np.full(5, PI2_6 * tau * tau),
                                       base="gumbel", points=4001)
        # measured 5.6e-17 at these settings; the bound is lattice
        # resolution, the analytic side is exact
        assert np.abs(p_analytic - p_lattice).max() < 1e-12
        assert abs(p_analytic.sum() - 1.0) < 1e-14


def test_factor_mixture_equals_lattice_gumbel():
    rng = np.random.default_rng(7)
    n = 8
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 0.6
    tau = 1.0
    from winning.factor.core import hermite_nodes
    F, W = hermite_nodes(2, 21)
    p_analytic = softmax_probabilities(mu, temperature=tau, V=V, F=F, W=W)
    p_lattice = race_probabilities(mu, V=V,
                                   D=np.full(n, PI2_6 * tau * tau),
                                   F=F, W=W, base="gumbel", points=4001)
    assert np.abs(p_analytic - p_lattice).max() < 1e-6


def test_inverse_is_exact():
    rng = np.random.default_rng(1)
    mu = rng.normal(size=12)
    mu -= mu.mean()
    for tau in (1.0, 0.7):
        p = softmax_probabilities(mu, temperature=tau)
        mu_back = abilities_from_softmax(p, temperature=tau)
        assert np.abs(mu_back - mu).max() < 1e-13


def test_default_engine_scale_is_one_over_beta():
    # the engine's default D=1 unit-variance Gumbel corresponds to
    # tau = sqrt(6)/pi (the mixed-logit anchor's 1/BETA)
    mu = np.array([-0.5, 0.0, 0.4])
    p_engine = race_probabilities(mu, base="gumbel", points=4001)
    p_analytic = softmax_probabilities(mu, temperature=np.sqrt(6) / np.pi)
    assert np.abs(p_engine - p_analytic).max() < 1e-4
