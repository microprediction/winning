"""The out-of-the-box base families: standardized, self-consistent, and
each verified against Monte Carlo in an actual race."""
import numpy as np
import pytest

from winning.factor.races import (BASES, failure_base, race_probabilities,
                                  skew_normal_base, student_base)

ALL = [("normal", "normal"), ("gumbel", "gumbel"),
       ("logistic", "logistic"), ("laplace", "laplace"),
       ("student4", student_base(4)), ("skew3", skew_normal_base(3)),
       ("skew-5", skew_normal_base(-5))]


@pytest.mark.parametrize("name,base", ALL)
def test_base_is_standardized_and_consistent(name, base):
    fn = BASES[base] if isinstance(base, str) else base
    x = np.linspace(-40, 40, 400001)
    dx = x[1] - x[0]
    S, f, fp = fn(x)
    assert abs(f.sum() * dx - 1) < 1e-5
    assert abs((x * f).sum() * dx) < 1e-6
    assert abs(((x ** 2) * f).sum() * dx - 1) < 2e-3
    # S' = -f and fp = f' (away from any kink)
    interior = np.abs(x) < 8
    assert np.abs(np.gradient(S, dx) + f)[interior].max() < 1e-4
    assert np.abs(np.gradient(f, dx) - fp)[interior].max() < 1e-4


@pytest.mark.parametrize("name,base", ALL)
def test_base_races_match_monte_carlo(name, base):
    rng = np.random.default_rng(3)
    n, M = 6, 400_000
    mu = np.sort(rng.normal(size=n))
    fn = BASES[base] if isinstance(base, str) else base
    xg = np.linspace(-60, 60, 800001)
    Sg, _, _ = fn(xg)
    u = rng.random((M, n))
    X = mu + np.interp(u, 1.0 - Sg, xg)
    counts = np.bincount(X.argmin(1), minlength=n) / M
    p = race_probabilities(mu, D=np.ones(n), base=base)
    assert 0.5 * np.abs(p - counts).sum() < 4e-3   # MC noise at 4e5 draws
    assert abs(p.sum() - 1) < 1e-9


def test_failure_base_races_match_monte_carlo():
    rng = np.random.default_rng(4)
    n, M, q = 6, 400_000, 0.2
    mu = np.sort(rng.normal(size=n))
    p = race_probabilities(mu, D=np.ones(n), base=failure_base(q))
    perf = mu + rng.standard_normal((M, n))
    broke = rng.random((M, n)) < q
    perf[broke] += 6.0 + 0.35 * rng.standard_normal(int(broke.sum()))
    counts = np.bincount(perf.argmin(1), minlength=n) / M
    assert 0.5 * np.abs(p - counts).sum() < 4e-3
