"""Vectorized standard normal CDF without scipy (numpy has no erf)."""
import numpy as np

# Abramowitz & Stegun 7.1.26 rational approximation to erf, vectorized.
_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)
_P = 0.3275911


def _erf(x):
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + _P * x)
    y = 1.0 - (((((_A[4] * t + _A[3]) * t) + _A[2]) * t + _A[1]) * t + _A[0]) * t * np.exp(-x * x)
    return s * y


def Phi_np(x):
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
