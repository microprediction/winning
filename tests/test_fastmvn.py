"""Tests for winning.fastmvn (scipy.stats.multivariate_normal.cdf
drop-in for factor-structured covariance)."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from winning.fastmvn import (factorize_covariance, mvn_cdf_fast,
                             mvn_cdf_fast_info)


def test_factor_case_matches_scipy():
    rng = np.random.default_rng(7)
    n = 10
    V = rng.normal(size=(n, 2)) * 0.6
    D = 0.5 + rng.random(n)
    mu = rng.normal(size=n) * 0.3
    b = rng.normal(size=n) + 1
    p, meth = mvn_cdf_fast_info(upper=b, mean=mu, V=V, D=D)
    ps = multivariate_normal(mean=mu, cov=V @ V.T + np.diag(D)).cdf(b)
    assert meth == "factor"
    assert abs(p - ps) < 5e-5     # scipy default tolerance dominates


def test_two_sided_rectangle():
    rng = np.random.default_rng(8)
    n = 8
    V = rng.normal(size=(n, 2)) * 0.6
    D = 0.5 + rng.random(n)
    a = rng.normal(size=n) - 1.5
    b = a + np.abs(rng.normal(size=n)) + 0.5
    p, _ = mvn_cdf_fast_info(lower=a, upper=b, V=V, D=D)
    ps = multivariate_normal(mean=np.zeros(n),
                             cov=V @ V.T + np.diag(D)).cdf(
        b, lower_limit=a)
    assert abs(p - ps) < 5e-5


def test_structured_detection_and_exactness_guarantee():
    rng = np.random.default_rng(9)
    n = 8
    V = rng.normal(size=(n, 1))
    D = 0.3 + rng.random(n)
    fd = factorize_covariance(V @ V.T + np.diag(D))
    assert fd is not None
    V2, D2 = fd
    resid = np.abs(V2 @ V2.T + np.diag(D2)
                   - (V @ V.T + np.diag(D))).max()
    assert resid < 1e-10


def test_independence_product_of_marginals():
    from scipy.stats import norm
    n = 6
    D = 0.5 + np.arange(1, n + 1) / 10
    b = np.linspace(-1, 1.5, n)
    p = mvn_cdf_fast(upper=b, V=np.zeros((n, 1)), D=D)
    assert abs(p - np.prod(norm.cdf(b / np.sqrt(D)))) < 1e-12


def test_deep_tail_recenters():
    rng = np.random.default_rng(7)
    n = 200
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.5 + rng.random(n)
    b = rng.normal(size=n) + 1.5
    p, meth = mvn_cdf_fast_info(upper=b, V=V, D=D)
    assert meth == "factor-recentered"
    assert 0 < p < 1e-8
