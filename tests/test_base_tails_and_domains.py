"""Shipped performance bases: stable tails and enforced domains.

#136 logistic left tail, #108 skew-logistic domain/tails, #399
skew-normal standardization, #389/#135 failure base, #103 exponential
power cusp. Every reference is an independent quadrature or a closed
form, run on the pure path (what CI exercises)."""
import numpy as np
import pytest
from scipy.integrate import quad

from winning import rustconfig
from winning.factor.races import race_probabilities


@pytest.fixture(autouse=True)
def _pure():
    was = rustconfig.rust_active()
    rustconfig.use_rust(False)
    yield
    rustconfig.use_rust(was)


def _two_runner_ref(fz, Fz_sf, gap):
    """P(runner at +gap beats runner at 0) for a standardized base:
    integral f(x - gap) S(x) dx, split at the two modes."""
    g = lambda x: fz(x - gap) * Fz_sf(x)
    pts = sorted({0.0, gap})
    lo, hi = -60.0, gap + 60.0
    return sum(quad(g, a, b, epsabs=0, epsrel=1e-12, limit=400)[0]
               for a, b in zip([lo] + pts, pts + [hi]))


# --- #136 ----------------------------------------------------------------

def _logistic_pdf(x):
    c = np.pi / np.sqrt(3.0)
    e = np.exp(-abs(c * x))
    return c * e / (1 + e) ** 2


def _logistic_sf(x):
    c = np.pi / np.sqrt(3.0)
    u = c * x
    return np.exp(-u) / (1 + np.exp(-u)) if u > 0 else 1 / (1 + np.exp(u))


@pytest.mark.parametrize("gap", [20.0, 25.0, 35.0, 50.0])
def test_logistic_rare_winner_matches_quadrature(gap):
    p = race_probabilities(np.array([0.0, gap]), D=np.ones(2), base="logistic",
                           window="span", points=65537)
    ref = _two_runner_ref(_logistic_pdf, _logistic_sf, gap)
    assert p[1] == pytest.approx(ref, rel=1e-6)


def test_logistic_three_runner_tail_matches_the_compiled_values():
    # the compiled kernel's log-domain values quoted in #136
    p = race_probabilities(np.array([0.0, 25.0, 50.0]), D=np.ones(3),
                           base="logistic", window="span", points=65537)
    assert p[1] == pytest.approx(8.99017456e-19, rel=1e-6)
    assert p[2] == pytest.approx(1.86370210e-38, rel=1e-6)


# --- #108 ----------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.001, 0.003, 0.01, 0.0292, 0.03, 0.1, 0.3, 1.0])
def test_skew_logistic_small_alpha_is_finite(alpha):
    from winning.factor.races import skew_logistic_base
    b = skew_logistic_base(alpha)
    assert np.all(np.isfinite(b.span))
    S, f, fp = b(np.linspace(-80, 40, 2001))
    assert np.all(np.isfinite(S)) and np.all(np.isfinite(f)) and np.all(np.isfinite(fp))
    p = race_probabilities([-0.5, 1.1, -0.6], base=b, window="span", points=16385)
    assert np.all(np.isfinite(p)) and abs(p.sum() - 1) < 1e-9


def test_skew_logistic_samples_are_finite_and_standardized():
    from winning.factor.races import skew_logistic_base
    x = skew_logistic_base(0.001).sample(np.random.default_rng(0), 100_000)
    assert np.all(np.isfinite(x))
    x = skew_logistic_base(0.3).sample(np.random.default_rng(1), 400_000)
    assert abs(x.mean()) < 0.01 and abs(x.std() - 1) < 0.01


@pytest.mark.parametrize("gap", [25.0, 45.0])
def test_skew_logistic_alpha_one_is_the_logistic_in_the_tail(gap):
    from winning.factor.races import skew_logistic_base
    kw = dict(window="span", points=16385)
    ref = race_probabilities([0.0, gap], base="logistic", **kw)[1]
    got = race_probabilities([0.0, gap], base=skew_logistic_base(1.0), **kw)[1]
    assert got == pytest.approx(ref, rel=1e-8)


def test_skew_logistic_span_quantile_is_the_stable_one():
    from winning.factor.races import skew_logistic_base
    # the stable standardized left span at alpha = 0.02 quoted in #108
    # (it was inf)
    assert skew_logistic_base(0.02).span[0] == pytest.approx(21.7111, abs=1e-3)


def test_skew_logistic_tail_compiled_matches_pure():
    """The compiled softplus was ln(1 + e^u), which rounds for u << 0;
    skipped where no compiled kernel is installed (CI)."""
    from winning.factor import races
    from winning.factor.races import skew_logistic_base
    if not races._RUST_OK:
        pytest.skip("fastrace not installed")
    kw = dict(window="span", points=16385)
    pure = race_probabilities([0.0, 25.0], base=skew_logistic_base(1.0), **kw)[1]
    rustconfig.use_rust(True)
    try:
        comp = race_probabilities([0.0, 25.0], base=skew_logistic_base(1.0), **kw)[1]
    finally:
        rustconfig.use_rust(False)
    assert comp == pytest.approx(pure, rel=1e-8)


# --- #399 ----------------------------------------------------------------

def test_skew_normal_saturated_shape_is_continuous():
    from winning.factor.races import skew_normal_base
    mu = np.array([-0.6, 0.2, 1.1, 1.7])
    D = np.array([0.25, 1.0, 2.25, 4.0])
    at = lambda a: race_probabilities(mu, D=D, base=skew_normal_base(a),
                                      points=4001, window="span")
    for sign in (1.0, -1.0):
        below = at(sign * 1e154)
        for a in (2e154, 1e200, 1e300):
            np.testing.assert_allclose(at(sign * a), below, atol=1e-12)
    # and the saturated law is the standardized half-normal limit
    b = skew_normal_base(1e200)
    z = np.linspace(-1.2, 6, 200)
    S, f, fp = b(z)
    assert np.all(np.isfinite(f)) and np.all(np.isfinite(fp))
    zz = np.linspace(-1.4, 12, 400001)
    _, ff, _ = b(zz)
    dz = zz[1] - zz[0]
    assert abs(ff.sum() * dz - 1) < 1e-3          # unit mass
    assert abs((ff * zz).sum() * dz) < 1e-3       # mean zero
    assert abs((ff * zz * zz).sum() * dz - 1) < 1e-3   # unit variance


def test_skew_normal_slope_is_finite_at_the_kink():
    from winning.factor.races import skew_normal_base
    b = skew_normal_base(2e154)
    m_sd = b.rust_base[1][1:]
    z0 = -m_sd[0] / m_sd[1]          # the z at which x = 0
    _, _, fp = b(np.array([z0, 0.0, 1.0]))
    assert not np.isnan(fp).any()


@pytest.mark.parametrize("a", [np.inf, -np.inf, np.nan])
def test_skew_normal_refuses_a_non_finite_shape(a):
    from winning.factor.races import skew_normal_base
    with pytest.raises(ValueError, match="finite"):
        skew_normal_base(a)
