"""The CDF-gradient evaluator resolves a narrow runner beside a wide one (#955).

One uniform grid over every marginal's +-7 sd put a 0.001-sd runner
between every point of a 1e-6/1/100 variance field: its share came back
exactly zero and the vector lost 18% of its mass. Independent normals, so
the exact shares are one-dimensional integrals.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

pytest.importorskip("jax")
from winning.alternatives.cdf_grad import _union_grid, cdf_gradient_shares  # noqa: E402

MU = np.array([0.0, 0.2, -0.1])
D = np.array([1e-6, 1.0, 100.0])


def _exact(i):
    sd = np.sqrt(D)

    def f(x):
        others = np.prod([norm.cdf((x - MU[j]) / sd[j]) for j in range(3) if j != i])
        return norm.pdf((x - MU[i]) / sd[i]) / sd[i] * others
    lo, hi = MU[i] - 12 * sd[i], MU[i] + 12 * sd[i]
    return quad(f, lo, hi, epsabs=1e-13, epsrel=1e-12, limit=500)[0]


def test_a_narrow_runner_keeps_its_share():
    exact = np.array([_exact(i) for i in range(3)])
    got = cdf_gradient_shares(MU, np.empty((3, 0)), D)
    assert got[0] == pytest.approx(exact[0], abs=1e-6)      # was exactly 0
    assert np.abs(got - exact).max() < 3e-3                  # GHK noise at 512 draws
    assert abs(got.sum() - 1) < 3e-3                         # was 0.819


def test_a_balanced_field_keeps_the_global_grid():
    mu, sd = np.array([0.0, 0.3, -0.2]), np.ones(3)
    g = _union_grid(mu, sd, 96)
    np.testing.assert_array_equal(g, np.linspace((mu - 7).min(), (mu + 7).max(), 96))
