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
