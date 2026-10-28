"""#843: StatePricer must treat NumPy low-precision NaN dividends as missing quotes."""
import numpy as np
import pytest

from winning.research import AbilityCalibrator, Density, StatePricer, UniformLattice

WANT = np.array([4 / 7, 1 / 7, 2 / 7])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_nan_dividend_is_missing_quote(dtype):
    got = StatePricer.prices_from_dividends(np.array([2, np.nan, 4], dtype=dtype), nan_value=8)
    np.testing.assert_allclose(got, WANT, rtol=0, atol=1e-15)


@pytest.mark.parametrize("missing", [float("nan"), None, np.float32("nan")])
def test_python_sequence_missing_quote(missing):
    got = StatePricer.prices_from_dividends([2.0, missing, 4.0], nan_value=8)
    np.testing.assert_allclose(got, WANT, rtol=0, atol=1e-15)


def test_solve_from_dividends_float32_nan_is_finite():
    lat = UniformLattice(L=80, unit=0.2)
    cal = AbilityCalibrator(Density.skew_normal(lat, loc=0, scale=1, a=0), n_iter=2)
    a32 = cal.solve_from_dividends(np.array([2, np.nan, 4], dtype=np.float32), nan_value=8)
    a64 = cal.solve_from_dividends(np.array([2, np.nan, 4], dtype=np.float64), nan_value=8)
    assert np.all(np.isfinite(a32))
    np.testing.assert_array_equal(a32, a64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ordinary_dividends_unchanged(dtype):
    # No missing quote: the float64 path and a plain list agree bitwise.
    d = [2.5, 4.0, 8.0, 16.0]
    got = StatePricer.prices_from_dividends(np.array(d, dtype=dtype))
    inv = np.array([1 / x for x in d])
    np.testing.assert_array_equal(got, inv / inv.sum())
