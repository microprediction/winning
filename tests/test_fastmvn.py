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


# --- empty and degenerate rectangles (#235) -------------------------------
#
# P(lower <= X <= upper) over a reversed coordinate is an EMPTY event. The
# ports formed the negative conditional cell Phi(0) - Phi(1) = -0.3413...
# and clamped it to the underflow floor, so the answer came back as
# 1e-300: a finite log probability near -690.8 for an impossible
# observation, rather than -inf. On the recentered path it was worse, as
# importance integration then integrated the artificial constant cell.
#
# mvtnorm::pmvnorm -- which r/mvtnormfast is a drop-in for -- raises on
# reversed bounds and returns 0 when a coordinate has lower == upper. All
# four ports now do the same, each pinned in its own suite: here, in
# r/mvtnormfast/tests/testthat/test-pmvnorm-fast.R, and in
# julia/FactorMvNormalCDF/test/runtests.jl.

def _one_by_one():
    return dict(mean=[0.0], V=np.zeros((1, 1)), D=np.ones(1))


def test_reversed_bounds_raise_rather_than_returning_the_floor():
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        mvn_cdf_fast(lower=[1.0], upper=[0.0], **_one_by_one())


def test_the_error_names_the_offending_coordinate():
    with pytest.raises(ValueError, match="coordinate 1"):
        mvn_cdf_fast(lower=[0.0, 1.0, -1.0], upper=[1.0, 0.0, 1.0],
                     mean=np.zeros(3), V=np.zeros((3, 1)), D=np.ones(3))


def test_a_degenerate_coordinate_has_exactly_zero_mass():
    p, meth = mvn_cdf_fast_info(lower=[0.5], upper=[0.5], **_one_by_one())
    assert p == 0.0                       # exactly, not 1e-300
    assert meth == "degenerate-rectangle"
    with np.errstate(divide="ignore"):    # the point: -inf, not -690.8
        assert np.log(p) == -np.inf


def test_one_degenerate_coordinate_among_valid_ones_is_still_zero():
    p, meth = mvn_cdf_fast_info(
        lower=[0.0, 0.3, -1.0], upper=[1.0, 0.3, 1.0],
        mean=np.zeros(3), V=np.zeros((3, 1)), D=np.ones(3))
    assert p == 0.0 and meth == "degenerate-rectangle"


def test_infinite_bounds_that_coincide_are_degenerate_not_full():
    for b in (-np.inf, np.inf):
        p, _m = mvn_cdf_fast_info(lower=[b], upper=[b], **_one_by_one())
        assert p == 0.0, f"P(X == {b}) should be 0"


def test_the_dense_route_agrees_with_the_factor_route():
    """The scipy fallback is a separate branch with its own bounds, and
    scipy returns a NEGATIVE number for a reversed rectangle rather than
    raising, so the check has to sit in front of it too."""
    rng = np.random.default_rng(3)
    sigma = None
    for _ in range(400):
        A = rng.standard_normal((4, 4))
        S = A @ A.T
        S = S / np.sqrt(np.outer(np.diag(S), np.diag(S)))
        if factorize_covariance(S) is None:
            sigma = S
            break
    assert sigma is not None, "no non-factorizable covariance found"
    # the branch really is the fallback for a valid rectangle
    _p, meth = mvn_cdf_fast_info(lower=[-1.0] * 4, upper=[1.0] * 4,
                                 mean=np.zeros(4), sigma=sigma)
    assert meth == "scipy-fallback"
    with pytest.raises(ValueError, match="lower must not exceed upper"):
        mvn_cdf_fast(lower=[1.0, 0.0, 0.0, 0.0], upper=[0.0, 1.0, 1.0, 1.0],
                     mean=np.zeros(4), sigma=sigma)
    p, meth = mvn_cdf_fast_info(lower=[0.2, 0.0, 0.0, 0.0],
                                upper=[0.2, 1.0, 1.0, 1.0],
                                mean=np.zeros(4), sigma=sigma)
    assert p == 0.0 and meth == "degenerate-rectangle"


def test_valid_rectangles_are_untouched_by_the_check():
    rng = np.random.default_rng(11)
    n = 6
    V = rng.normal(size=(n, 2)) * 0.5
    D = 0.5 + rng.random(n)
    mu = rng.normal(size=n) * 0.3
    lo = mu - 1.0 - rng.random(n)
    up = mu + 1.0 + rng.random(n)
    p, meth = mvn_cdf_fast_info(lower=lo, upper=up, mean=mu, V=V, D=D)
    ps = multivariate_normal(mean=mu, cov=V @ V.T + np.diag(D)).cdf(
        up, lower_limit=lo)
    assert meth == "factor"
    assert abs(p - ps) < 5e-5
