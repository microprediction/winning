"""Performance-noise kernels for ThurstoneRating(base_kernel=...).

The lattice imposes no distributional family on performance noise; these
helpers make the useful ones one-liners. All return odd-length arrays on the
given lattice step size, normalized to unit mass, on the time-like convention
(mass at positive offsets = slower/worse performance).

Measured guidance (research/t_sweep.py and friends): tail effects decompose.
One-sided disaster mass (offday_kernel) is what helps where catastrophes are
real (F1: every metric); SYMMETRIC heavy tails there hurt (t puts miracle
mass on the fast side). A mild student-t (nu ~ 5) is a small free win even on
Gaussian worlds — likelihood tails double as robustness to the update's
approximate opponent marginals — while nu = 2 trades log loss for the best
calibration measured (HK ECE 0.0035 vs 0.0127 Gaussian). Skew rarely earns
its keep on rank data.
"""

from __future__ import annotations

import math

import numpy as np

_SUPPORT_Z = 6.0


def gaussian_kernel(unit: float = 0.1, sd: float = 1.0) -> np.ndarray:
    half = max(1, int(math.ceil(_SUPPORT_Z * sd / unit)))
    x = np.arange(-half, half + 1) * unit
    k = np.exp(-0.5 * (x / sd) ** 2)
    return k / k.sum()


def student_t_kernel(unit: float = 0.1, nu: float = 4.0, scale: float = 1.0) -> np.ndarray:
    """Student-t performance noise: nu is the tail dial (2 = very heavy,
    large = Gaussian). Support is widened with the tails; scale is the t
    scale parameter (variance is scale^2 * nu/(nu-2) for nu > 2)."""
    if nu <= 1:
        raise ValueError("nu must exceed 1")
    tail_widening = max(1.0, 4.0 / math.sqrt(nu))
    half = max(1, int(math.ceil(_SUPPORT_Z * scale * tail_widening / unit)))
    x = np.arange(-half, half + 1) * unit
    t = x / scale
    k = (1.0 + t * t / nu) ** (-(nu + 1.0) / 2.0)
    return k / k.sum()


def offday_kernel(
    unit: float = 0.1, p_off: float = 0.1, wide: float = 3.0, sd: float = 1.0
) -> np.ndarray:
    """Mixture: usual N(0, sd) performance, with probability p_off a bad/wild
    day of sd `wide` — the shape that improved every F1 metric (DNF mass)."""
    k_wide = gaussian_kernel(unit, wide)
    k_core = gaussian_kernel(unit, sd)
    half = (len(k_wide) - 1) // 2
    core = np.zeros_like(k_wide)
    s = half - (len(k_core) - 1) // 2
    core[s : s + len(k_core)] = k_core
    k = (1.0 - p_off) * core + p_off * k_wide
    return k / k.sum()


def skew_kernel(unit: float = 0.1, a: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """Skew-normal noise; a > 0 puts the heavy tail on the slow side."""
    half = max(1, int(math.ceil(_SUPPORT_Z * scale / unit))) + int(2.0 / unit)
    x = np.arange(-half, half + 1) * unit
    t = x / scale
    k = np.exp(-0.5 * t * t) * (1.0 + np.array([math.erf(a * ti / math.sqrt(2.0)) for ti in t]))
    return k / k.sum()
