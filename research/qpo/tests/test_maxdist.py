"""Anchors for the distribution of the maximum.

Separate from the argmax vector, and cheaper: no lattice, O(Q N) per
threshold. Checked against the closed form for independent normals, against
Monte Carlo under a factor model, and in the far tail where the whole point is
that Monte Carlo cannot follow.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pom import (expected_max_factor, hermite_nodes, max_cdf_factor,  # noqa: E402
                 max_sf_factor, sobol_nodes)

EULER = 0.5772156649015329


def test_independent_case_matches_the_closed_form():
    """With no factor the CDF of the max is exactly the product of CDFs."""
    rng = np.random.default_rng(0)
    n = 200
    mu = rng.standard_normal(n) * 0.4
    d = rng.uniform(0.5, 1.5, n)
    t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 5.0])
    got = max_cdf_factor(t, mu, None, d)
    want = np.prod(ndtr((t[:, None] - mu[None, :]) / np.sqrt(d)[None, :]), axis=1)
    assert np.max(np.abs(got - want)) < 1e-12


def test_expected_max_of_k_iid_standard_normals():
    """Against Monte Carlo truth, and separately against the DSR's asymptotic.

    The second comparison is the interesting one: the closed form used by the
    Deflated Sharpe Ratio, (1-gamma) Phi^-1(1-1/K) + gamma Phi^-1(1-1/(Ke)), is
    an asymptotic and carries its own error of a couple of hundredths of a
    Sharpe unit even in the independent case it is exact for. Recorded here so
    that error is never mistaken for ours.
    """
    from scipy.special import ndtri
    rng = np.random.default_rng(11)
    for K in (100, 1000, 10000):
        e = expected_max_factor(np.zeros(K), None, np.ones(K), points=20001)
        M = 200_000
        mc = rng.standard_normal((M, K)).max(axis=1)
        se = mc.std(ddof=1) / np.sqrt(M)
        assert abs(e - mc.mean()) < 5 * se + 1e-3, (K, e, mc.mean(), se)
        approx = ((1 - EULER) * ndtri(1 - 1.0 / K)
                  + EULER * ndtri(1 - 1.0 / (K * np.e)))
        assert abs(e - approx) < 0.03, (K, e, approx)


def test_factor_case_matches_monte_carlo():
    rng = np.random.default_rng(1)
    n, r = 500, 3
    mu = rng.standard_normal(n) * 0.2
    V = rng.standard_normal((n, r)) * 0.5
    d = rng.uniform(0.3, 1.0, n)
    F, W = hermite_nodes(r, Q=25)

    M = 400_000
    Y = mu[None, :] + rng.standard_normal((M, r)) @ V.T \
        + rng.standard_normal((M, n)) * np.sqrt(d)[None, :]
    mx = Y.max(axis=1)
    for t in (0.5, 1.5, 2.5, 3.5):
        emp = float((mx <= t).mean())
        se = np.sqrt(max(emp * (1 - emp), 1e-12) / M)
        got = float(max_cdf_factor(t, mu, V, d, F, W)[0])
        assert abs(got - emp) < 5 * se + 1e-6, (t, got, emp, se)
    assert abs(expected_max_factor(mu, V, d, F, W) - mx.mean()) < 0.02


def test_correlation_lowers_the_maximum():
    """Slepian: more positive correlation cannot raise the max."""
    n, r = 1000, 1
    mu = np.zeros(n)
    prev = None
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        V = np.full((n, 1), np.sqrt(rho))
        d = np.full(n, 1.0 - rho)
        F, W = hermite_nodes(1, Q=61)
        e = expected_max_factor(mu, V, d, F, W)
        if prev is not None:
            assert e < prev + 1e-9, (rho, e, prev)
        prev = e


def test_far_tail_keeps_its_digits():
    """The survival function must stay accurate where 1 - CDF underflows.

    Note which side is fragile. The naive reference n*(1 - ndtr(8)) is wrong by
    7%, because ndtr(8) = 1 - 6.2e-16 and the subtraction keeps almost no
    significant digits. scipy's norm.sf is the honest reference, and the
    log-domain computation matches it.
    """
    from scipy.stats import norm
    n = 100_000
    t = 8.0
    sf = float(max_sf_factor(t, np.zeros(n), None, np.ones(n))[0])
    want = n * float(norm.sf(t))            # accurate: no cancellation
    naive = n * (1 - ndtr(t))               # what the obvious reference gives
    assert abs(sf / want - 1.0) < 1e-6, (sf, want)
    assert abs(naive / want - 1.0) > 0.02   # the naive one really is that bad
    assert 1e-12 < sf < 1e-9


def test_sf_and_cdf_are_consistent():
    rng = np.random.default_rng(3)
    n, r = 300, 2
    mu = rng.standard_normal(n) * 0.3
    V = rng.standard_normal((n, r)) * 0.4
    d = rng.uniform(0.4, 1.0, n)
    F, W = hermite_nodes(r, Q=21)
    t = np.array([0.0, 1.0, 2.0])
    assert np.max(np.abs(max_cdf_factor(t, mu, V, d, F, W)
                         + max_sf_factor(t, mu, V, d, F, W) - 1.0)) < 1e-12
