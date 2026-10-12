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


# --- #470: the integrand's width is collective, not rowwise ---

def _ring(n=200):
    th = 2 * np.pi * np.arange(n) / n
    return np.column_stack([np.cos(th), np.sin(th)])


def test_many_mild_rows_make_a_narrow_integrand_470(fm):
    # rowwise sharpness exactly 1, so the fixed rule was 15 x 15 nodes at
    # unit scale; the product of 200 cells is 0.28 wide and the answer
    # came back 36.7% high. Reference: the issue's polar quadrature,
    # 0.0007891258848216968, which a 151 x 151 rule reproduces.
    V = _ring()
    p, method = fm.mvn_cdf_fast_info(upper=np.full(200, 2.0), V=V,
                                     D=np.ones(200))
    assert method == "factor"
    assert p == pytest.approx(0.0007891258848216968, rel=1e-8)


def test_two_sided_random_field_470(fm):
    # an ordinary random rank-two field, two-sided cells
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 2)) * 0.4
    up = 2.0 + rng.normal(size=200) * 0.3
    p = fm.mvn_cdf_fast(lower=-up, upper=up, V=V, D=np.ones(200))
    # reference: tensor Gauss-Hermite at 151^2 and 201^2 nodes, which
    # agree to 1e-14
    assert p == pytest.approx(1.366457537796035e-06, rel=1e-8)


# --- #518 / #457: the answer depends on V V', not on the factor basis ---

def _rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def test_rank_two_rotations_and_sign_flips_518(fm):
    V = np.array([[0.30812945940123476, -1.8936005031221805],
                  [0.05423700319905680, -1.8882188805898732],
                  [-0.04568751096449155, 1.2303149108078115],
                  [-1.2824267666411462, -0.7947487032502422]])
    D = np.array([0.48381594889619123, 0.607241434401239,
                  0.2137039397661559, 0.5038986029743953])
    mu = np.array([1.3435081180060733, 0.48784958537212353,
                   0.20068268880906692, 1.449261738300165])
    up = np.array([0.3670668335351432, -0.9140137229005407,
                   -0.16906974496655425, 0.6067127817398206])
    base = fm.mvn_cdf_fast(upper=up, mean=mu, V=V, D=D)
    # dense reference over four seeds: 1.419761803e-4 .. 1.419762735e-4
    assert base == pytest.approx(1.4197622e-4, rel=2e-6)
    rng = np.random.default_rng(0)
    for rot in [_rot(t) for t in rng.uniform(0, 2 * np.pi, 6)] + [
            np.diag([1.0, -1.0]), np.diag([-1.0, 1.0]),
            np.array([[0.0, 1.0], [1.0, 0.0]])]:
        assert fm.mvn_cdf_fast(upper=up, mean=mu, V=V @ rot, D=D) == \
            pytest.approx(base, rel=1e-12)


def test_rank_three_sign_flips_518(fm):
    V = np.array([
        [2.692520713523368, 1.392980981803484, 0.7503218321328118],
        [-3.1413260603735793, 3.1757401235045286, 4.534002392597087],
        [5.505616602042417, -2.6084117064617693, 0.7630448809882905],
        [2.8733471786655946, -1.889424026920147, -2.6786504937506477]])
    D = np.array([1.354640723776008, 0.07136225300228002,
                  0.4603168051577446, 0.2640963072247931])
    mu = np.array([-0.23242635828353164, 0.1304103908063737,
                   -0.6231001573285849, 0.31079948140401914])
    up = np.array([-0.23773277909955737, -0.2700683791403624,
                   -0.22984373753684728, 0.5581730522783985])
    base = fm.mvn_cdf_fast(upper=up, mean=mu, V=V, D=D)
    for flip in ([1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]):
        assert fm.mvn_cdf_fast(upper=up, mean=mu, V=V * np.array(flip),
                               D=D) == pytest.approx(base, rel=1e-12)
    perm = V[:, [2, 0, 1]]
    assert fm.mvn_cdf_fast(upper=up, mean=mu, V=perm, D=D) == \
        pytest.approx(base, rel=1e-12)
    # 6.8% apart before; the converged value is 0.0286629
    assert base == pytest.approx(0.0286629, rel=5e-3)


def test_zero_and_split_columns_457(fm):
    v = np.array([-1.19009272, 2.32396446, -1.49968514, 2.14990819])
    D = np.array([0.24428964, 0.62490739, 0.25676327, 1.17899935])
    b = np.array([1.21213156, -1.15078308, 1.50271797, 6.29628221])
    ref = 0.11879131458669231            # 1-D quad, from the issue
    base = fm.mvn_cdf_fast(upper=b, V=v[:, None], D=D)
    assert base == pytest.approx(ref, rel=1e-9)
    for V in (np.column_stack([v, np.zeros((4, 2))]),
              np.column_stack([v / np.sqrt(2), v / np.sqrt(2)]),
              np.column_stack([0.6 * v, 0.8 * v, np.zeros(4)])):
        assert fm.mvn_cdf_fast(upper=b, V=V, D=D) == pytest.approx(
            base, rel=1e-12)


# --- #487: an unconstrained coordinate is marginalised out exactly ---

def test_unconstrained_coordinates_leave_the_answer_alone_487(fm):
    v = np.array([-2.145, 0.525, 3.328, 1.261])[:, None]
    D = np.array([1.122, 0.142, 2.040, 0.300])
    mu = np.array([0.702, 0.319, 0.254, -3.958])
    up = np.array([-2.305, 4.151, 1.287, 0.557])
    base = fm.mvn_cdf_fast(upper=up, mean=mu, V=v, D=D)
    assert base == pytest.approx(0.0037867371230754, rel=1e-9)
    for load in (4.0, 3.0, 40.0):
        aug = fm.mvn_cdf_fast(lower=np.r_[np.full(4, -np.inf), -np.inf],
                              upper=np.r_[up, np.inf],
                              mean=np.r_[mu, 123.0],
                              V=np.vstack([v, [[load]]]), D=np.r_[D, 1.0])
        assert aug == base
    S = v @ v.T + np.diag(D)
    S5 = np.zeros((5, 5))
    S5[1:, 1:] = S
    S5[0, 0] = 2.0
    S5[0, 1:] = S5[1:, 0] = 0.3
    full = fm.mvn_cdf_fast(upper=np.r_[np.inf, up], mean=np.r_[0.0, mu],
                           sigma=S5)
    assert full == fm.mvn_cdf_fast(upper=up, mean=mu, sigma=S)
    assert fm.mvn_cdf_fast_info(upper=np.inf, V=v, D=D) == (1.0, "factor")


# --- #480 / #89: exactly one covariance representation ---

@pytest.mark.parametrize("kw", [
    dict(sigma=np.eye(2), V=[10.0, -10.0]),
    dict(sigma=np.eye(2), D=[100.0, 100.0]),
    dict(sigma=np.eye(2), V=[10.0, -10.0], D=[1.0, 1.0]),
    dict(V=[10.0, -10.0]),
    dict(D=[1.0, 1.0]),
    dict(),
])
def test_mixed_covariance_inputs_are_refused_480(fm, kw):
    with pytest.raises(ValueError, match="sigma"):
        fm.mvn_cdf_fast_info(upper=np.zeros(2), **kw)


def test_one_covariance_is_accepted_480(fm):
    assert fm.mvn_cdf_fast(upper=np.zeros(2), sigma=np.eye(2)) == \
        pytest.approx(0.25, abs=1e-15)
    assert fm.mvn_cdf_fast(upper=np.zeros(2), V=[10.0, -10.0],
                           D=[1.0, 1.0]) == pytest.approx(0.0224147254,
                                                          rel=1e-8)


# --- #664: a narrow central interval keeps its mass ---

@pytest.mark.parametrize("eps", [1e-15, 1e-16, 1e-17, 1e-100, 1e-300])
def test_narrow_central_interval_664(fm, eps):
    from scipy.special import erf
    exact = erf(eps / np.sqrt(2.0))
    p = fm.mvn_cdf_fast(lower=[-eps], upper=[eps], V=np.zeros((1, 0)),
                        D=[1.0])
    assert p == pytest.approx(exact, rel=1e-12)
    # inside a product
    p2 = fm.mvn_cdf_fast(lower=[-eps, -np.inf], upper=[eps, 0.0],
                         V=np.zeros((2, 0)), D=[1.0, 1.0])
    assert p2 == pytest.approx(exact / 2, rel=1e-12)
