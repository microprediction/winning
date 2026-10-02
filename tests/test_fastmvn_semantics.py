"""The structured-MVN rectangle depends on the GAUSSIAN, not on how it is
spelled: representation, units, orientation and reflection must not move
the answer. One regression per reported failure, each run against both
the in-package copy and the standalone python/fastmvn package (they are
byte-identical by contract; importing both proves the package ships the
fix, not just the tree).
"""
import importlib
import os
import sys

import numpy as np
import pytest
from scipy.special import ndtr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STANDALONE = os.path.join(ROOT, "python", "fastmvn", "src")


def _load(name):
    if name == "fastmvn":
        if not os.path.isdir(STANDALONE):
            pytest.skip("standalone fastmvn source not in this tree")
        if STANDALONE not in sys.path:
            sys.path.insert(0, STANDALONE)
    return importlib.import_module(name)


@pytest.fixture(params=["winning.fastmvn", "fastmvn"])
def fm(request):
    return _load(request.param)


# --- #429: V = I, D = 0 is the same Gaussian as V = 0, D = 1 ------------

@pytest.mark.parametrize("b", [-3.0, -4.0, -6.0])
def test_where_the_variance_is_stored_does_not_matter(fm, b):
    exact = float(ndtr(b) ** 2)
    in_v, m = fm.mvn_cdf_fast_info(upper=[b, b], V=np.eye(2), D=np.zeros(2))
    in_d, _ = fm.mvn_cdf_fast_info(upper=[b, b], V=np.zeros((2, 2)),
                                   D=np.ones(2))
    # was 6.3e-305 at b = -4 and 3.5e-309 at b = -6 for the V = I spelling
    assert in_v == pytest.approx(exact, rel=1e-10)
    assert in_d == pytest.approx(exact, rel=1e-10)


# --- #410: a rectangle that misses the singular support is empty -------

@pytest.mark.parametrize("lower,upper,want", [
    ([-np.inf, 1.0], [0.0, np.inf], 0.0),          # disjoint
    ([-np.inf, 0.0], [0.0, np.inf], 0.0),          # touching: F == 0 only
    ([-np.inf, -1.0], [0.0, np.inf], float(ndtr(0.0) - ndtr(-1.0))),
])
def test_singular_support_is_respected(fm, lower, upper, want):
    # X1 = X2 = F exactly
    p, _ = fm.mvn_cdf_fast_info(lower=lower, upper=upper, mean=np.zeros(2),
                                V=np.ones((2, 1)), D=np.zeros(2))
    if want == 0.0:
        assert p == 0.0          # was 6.6e-301 and 1.0e-300
    else:
        assert p == pytest.approx(want, rel=1e-12)


def test_a_loaded_deterministic_row_beside_free_ones(fm):
    from scipy.stats import multivariate_normal
    g = np.random.default_rng(3)
    V = g.normal(size=(5, 2)) * 0.7
    D = np.array([0.0, 0.0, 0.5, 0.7, 1.0])
    mu = g.normal(size=5) * 0.3
    b = g.normal(size=5) + 0.3
    a = b - 1.5 - g.random(5)
    p, _ = fm.mvn_cdf_fast_info(lower=a, upper=b, mean=mu, V=V, D=D)
    # tolerances on the cdf call, not the constructor: the frozen
    # constructor only accepts them on recent scipy (CI's 3.10 job has
    # an older one), while the generator's cdf has taken them throughout
    ref = multivariate_normal.cdf(b, mean=mu, cov=V @ V.T + np.diag(D),
                                  maxpts=10 ** 7, abseps=1e-12, releps=1e-9,
                                  lower_limit=a)
    assert p == pytest.approx(ref, rel=1e-4)


# --- #414: lower == upper on a zero-variance coordinate is an atom ------

def test_an_atom_on_the_bound_carries_its_mass(fm):
    p, _ = fm.mvn_cdf_fast_info(lower=[0.0, -np.inf], upper=[0.0, 0.0],
                                mean=[0.0, 0.0], V=np.zeros((2, 1)),
                                D=[0.0, 1.0])
    assert p == pytest.approx(0.5, abs=1e-15)      # was 0.0
    p1, _ = fm.mvn_cdf_fast_info(lower=[0.0], upper=[0.0], mean=[0.0],
                                 V=[[0.0]], D=[0.0])
    assert p1 == 1.0                                # was 0.0


def test_an_atom_off_the_bound_and_a_continuous_slab_are_empty(fm):
    miss, _ = fm.mvn_cdf_fast_info(lower=[0.0, -np.inf], upper=[0.0, 0.0],
                                   mean=[0.5, 0.0], V=np.zeros((2, 1)),
                                   D=[0.0, 1.0])
    slab, m = fm.mvn_cdf_fast_info(lower=[0.0, -np.inf], upper=[0.0, 0.0],
                                   mean=[0.0, 0.0], V=np.zeros((2, 1)),
                                   D=[1.0, 1.0])
    assert miss == 0.0
    assert slab == 0.0 and m == "degenerate-rectangle"


def test_an_atom_on_the_dense_route(fm):
    S = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    p, _ = fm.mvn_cdf_fast_info(lower=[2.0, -np.inf, -np.inf],
                                upper=[2.0, 0.0, 0.0], mean=[2.0, 0, 0],
                                sigma=S)
    assert p == pytest.approx(0.25 + np.arcsin(0.5) / (2 * np.pi), abs=1e-6)


# --- #422 / #79: the dimension comes from the arguments that carry it ---

def test_one_element_D_is_the_scalar_shorthand(fm):
    V = np.array([[0.3], [-0.2], [0.4]])
    want = fm.mvn_cdf_fast(V=V, D=np.ones(3), upper=0.0)
    for D in (1.0, [1.0], np.ones(1)):
        assert fm.mvn_cdf_fast(V=V, D=D, upper=0.0) == pytest.approx(
            want, abs=1e-15)                       # [1.0] gave Phi(0)
    assert fm.mvn_cdf_fast(V=np.zeros((3, 1)), D=[1.0], upper=0.0) == \
        pytest.approx(0.125, abs=1e-15)


def test_scalar_D_with_every_loading_spelling(fm):
    v = np.array([0.4, -0.2, 0.1, 0.5, -0.3])
    up = np.zeros(5)
    want = fm.mvn_cdf_fast(upper=up, V=v[:, None], D=np.ones(5))
    for V in (v, v[:, None], v[None, :]):
        assert fm.mvn_cdf_fast(upper=up, V=V, D=1.0) == pytest.approx(
            want, abs=1e-15)
    assert fm.mvn_cdf_fast(upper=up, V=0.2, D=1.0) == pytest.approx(
        fm.mvn_cdf_fast(upper=up, V=np.full(5, 0.2), D=np.ones(5)),
        abs=1e-15)


def test_conflicting_lengths_are_refused(fm):
    with pytest.raises(ValueError, match="disagree"):
        fm.mvn_cdf_fast(upper=np.zeros(4), V=np.ones((3, 1)), D=np.ones(3))


# --- #395: one coordinate needs no quadrature ---------------------------

@pytest.mark.parametrize("z", [-4.2, -4.1, -4.0, -1.0, 2.0])
def test_one_coordinate_is_the_normal_cdf(fm, z):
    sd = np.sqrt(1.0 + 1e-6)
    p = fm.mvn_cdf_fast(V=np.array([[1.0]]), D=np.array([1e-6]),
                        upper=[z * sd])
    assert p == pytest.approx(float(ndtr(z)), rel=1e-12)   # was 5.9x


def test_one_hit_on_the_sobol_rule_is_not_an_answer(fm):
    """One node is 1/8192; a rank-two sharp case that lands one or two
    nodes must go to the tail repair rather than return k/8192."""
    sd = np.sqrt(1.0 + 1e-6)
    for z in (-4.0, -3.8, -3.6, -3.0, -2.0):
        p, m = fm.mvn_cdf_fast_info(V=np.eye(2), D=np.full(2, 1e-6),
                                    upper=[z * sd, z * sd])
        assert m == "factor-recentered"
        assert abs(p * 8192 - round(p * 8192)) > 1e-6   # not a node count
        assert p == pytest.approx(float(ndtr(z) ** 2), rel=2e-3)


# --- #196 / #98: the upper tail, and reflection symmetry ----------------

@pytest.mark.parametrize("lo,hi", [(8.0, 9.0), (9.0, 10.0), (-10.0, -9.0)])
def test_tail_interval_masses(fm, lo, hi):
    from scipy.stats import norm
    want = float(norm.sf(lo) - norm.sf(hi)) if lo > 0 else \
        float(norm.cdf(hi) - norm.cdf(lo))
    p = fm.mvn_cdf_fast(lower=[lo], upper=[hi], V=np.zeros((1, 1)),
                        D=np.ones(1))
    assert p == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_rectangle_and_its_reflection(fm, n):
    up = fm.mvn_cdf_fast(lower=9 * np.ones(n), V=np.zeros((n, 1)),
                         D=np.ones(n))
    down = fm.mvn_cdf_fast(upper=-9 * np.ones(n), V=np.zeros((n, 1)),
                           D=np.ones(n))
    assert up == pytest.approx(float(ndtr(-9.0) ** n), rel=1e-9)
    assert up == pytest.approx(down, rel=1e-12)     # was 0 for n >= 2


def test_reflection_with_correlation(fm):
    g = np.random.default_rng(2)
    V = g.normal(size=(4, 1)) * 0.5
    D = 0.5 + g.random(4)
    a = np.full(4, 4.0)
    up = fm.mvn_cdf_fast(lower=a, V=V, D=D)
    down = fm.mvn_cdf_fast(upper=-a, V=-V, D=D)
    assert up > 0
    assert up == pytest.approx(down, rel=1e-6)


# --- #368: units do not change the algorithm ----------------------------

@pytest.mark.parametrize("c", [1e-12, 1e-6, 1.0, 1e6, 1e12])
def test_scaled_covariance_is_the_same_event(fm, c):
    V = np.array([[.5], [-.3], [.4], [.2], [-.4], [.3]])
    D = np.array([.8, 1.1, .9, 1.2, 1., .7])
    S = V @ V.T + np.diag(D)
    mu = np.array([.1, -.2, .3, 0., .4, -.1])
    b = np.array([.5, .2, .8, -.1, .4, .6])
    fd = fm.factorize_covariance(c * S)
    assert fd is not None                            # was None at 1e-12
    Vc, Dc = fd
    err = np.abs(Vc @ Vc.T + np.diag(Dc) - c * S).max() / c
    assert err < 1e-10
    p0, m0 = fm.mvn_cdf_fast_info(mean=mu, upper=b, sigma=S)
    p, m = fm.mvn_cdf_fast_info(mean=np.sqrt(c) * mu, upper=np.sqrt(c) * b,
                                sigma=c * S)
    assert m == m0 == "factor"
    assert p == pytest.approx(p0, rel=1e-12)


# --- #359: sigma must be a covariance -----------------------------------

def test_an_asymmetric_sigma_is_refused(fm):
    S = np.array([[1.0, 0.7, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
    for A in (S, S.T):
        with pytest.raises(ValueError, match="not symmetric"):
            fm.mvn_cdf_fast(upper=np.zeros(3), sigma=A)


@pytest.mark.parametrize("S,match", [
    (np.ones((2, 3)), "square"),
    (np.array([[1.0, np.nan], [np.nan, 1.0]]), "NaN"),
    (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
])
def test_malformed_sigma_is_refused(fm, S, match):
    with pytest.raises(ValueError, match=match):
        fm.mvn_cdf_fast(upper=0.0, sigma=S)


def test_numerical_asymmetry_and_singular_psd_are_accepted(fm):
    S = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]])
    E = S.copy()
    E[0, 1] += 1e-14
    assert fm.mvn_cdf_fast(upper=np.zeros(3), sigma=E) == pytest.approx(
        fm.mvn_cdf_fast(upper=np.zeros(3), sigma=S), abs=1e-6)
    v = np.array([1.0, 2.0, 3.0])
    assert fm.mvn_cdf_fast(upper=np.zeros(3), sigma=np.outer(v, v)) == \
        pytest.approx(0.5, abs=1e-6)


# --- #132: diagonal, and correlation judged on its own scale ------------

def test_a_diagonal_sigma_is_the_independent_product(fm):
    D0 = np.array([1., 1., 1., 1., 1e8])
    b = np.array([10., 10., 10., 10., -5000.])
    V, D = fm.factorize_covariance(np.diag(D0))
    assert V.shape == (5, 0)
    p = fm.mvn_cdf_fast(upper=b, sigma=np.diag(D0))
    assert p == pytest.approx(float(np.prod(ndtr(b / np.sqrt(D0)))),
                              rel=1e-13)


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.9])
def test_a_correlation_is_not_negligible_beside_a_large_variance(fm, rho):
    S = np.array([[1., rho, 0.], [rho, 1., 0.], [0., 0., 1e16]])
    V, D = fm.factorize_covariance(S)
    assert V.shape[1] == 1                       # was rank zero
    p = fm.mvn_cdf_fast(upper=np.zeros(3), sigma=S)
    want = 0.5 * (0.25 + np.arcsin(rho) / (2 * np.pi))
    assert p == pytest.approx(want, rel=1e-6)    # was -41.6% at 0.9


@pytest.mark.parametrize("rho", [0.5, 0.8, 0.85, 0.9])
def test_the_rank_one_rule_resolves_the_orthant(fm, rho):
    """The order rule grew linearly with sharpness, which left 0.2%
    relative error at correlation 0.8 and 0.33% at 0.85."""
    V = np.full((2, 1), np.sqrt(rho))
    D = np.full(2, 1 - rho)
    p = fm.mvn_cdf_fast(upper=np.zeros(2), V=V, D=D)
    assert p == pytest.approx(0.25 + np.arcsin(rho) / (2 * np.pi), rel=1e-7)
