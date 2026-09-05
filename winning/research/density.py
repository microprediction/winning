from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .lattice import UniformLattice
from .normaldist import normcdf, normpdf


def _cdf_from_pdf(pdf: np.ndarray) -> np.ndarray:
    c = np.cumsum(pdf)
    # clamp to [0,1] monotone non-decreasing
    c = np.maximum.accumulate(np.minimum(c, 1.0))
    c = np.clip(c, 0.0, 1.0)
    return c


def _pdf_from_cdf(cdf: np.ndarray) -> np.ndarray:
    # prepend 0 then diff
    return np.diff(np.insert(cdf, 0, 0.0))


def _normalize(pdf: np.ndarray) -> np.ndarray:
    s = float(np.sum(pdf))
    if s < 0:
        raise ValueError("PDF has negative total mass, which is invalid.")
    if s == 0:
        # Allow zero-mass arrays to pass through (e.g., extreme shifts off the lattice)
        return pdf
    return pdf / s


@dataclass
class Density:
    """
    Lattice-aligned measure. In the typical case sum(p) == 1, but we allow
    sum(p) == 0 as a sentinel for 'off-lattice' runners in extreme offsets.
    Negative total mass is always an error.
    """

    lattice: UniformLattice
    p: np.ndarray  # shape (2L+1,), sum ~ 1

    def __post_init__(self):
        self.lattice.assert_compatible(self.p)
        self.p = _normalize(np.asarray(self.p, dtype=float))

    # ---- statistics ----
    def cdf(self) -> np.ndarray:
        return _cdf_from_pdf(self.p)

    def mean(self) -> float:
        return float(np.dot(self.p, self.lattice.grid))

    def approx_support(self, tol: float = 1e-12):
        return np.where(self.p > tol)[0]

    def approx_support_width(self, tol: float = 1e-12) -> int:
        idx = self.approx_support(tol)
        if len(idx) == 0:
            return 0
        return int(idx.max() - idx.min())

    # ---- transforms ----
    def shift_integer(self, k: int) -> "Density":
        """Shift CDF right by k steps, then re-diff to PDF."""
        c = self.cdf()
        K = len(c)
        if k <= -K:
            c2 = np.ones_like(c)
        elif -K < k < 0:
            c2 = np.concatenate([c[abs(k) :], np.full(abs(k), c[-1])])
        elif 0 < k < K:
            c2 = np.concatenate([np.zeros(k), c[:-k]])
        elif k >= K:
            c2 = np.zeros_like(c)
        else:  # k == 0
            c2 = c
        p2 = _pdf_from_cdf(c2)
        return Density(self.lattice, p2)

    def shift_fractional(self, x: float) -> "Density":
        """Linear blend of neighboring integer shifts (on the CDF)."""
        L = self.lattice.L
        # represent x as convex combo of floor and ceil, but clamp to lattice
        if -L + 2 < x < L - 2:
            l = int(np.floor(x))
            u = int(np.ceil(x))
            r = x - l
            lc, uc = 1.0 - r, r
        elif x >= L - 2:
            l, u, lc, uc = L - 2, L - 1, 1.0, 0.0
        else:  # x <= -L+2
            l, u, lc, uc = -L + 1, -L + 2, 0.0, 1.0
        c2 = lc * self.shift_integer(l).cdf() + uc * self.shift_integer(u).cdf()
        p2 = _pdf_from_cdf(c2)
        return Density(self.lattice, p2)

    def center(self) -> "Density":
        m = self.mean()
        # convert physical mean to lattice steps
        steps = m / self.lattice.unit
        return self.shift_fractional(-steps)

    def convolve(
        self, other: "Density", *, keep_L: Optional[int] = None, pad: bool = False
    ) -> "Density":
        if self.lattice.unit != other.lattice.unit:
            raise ValueError("Units must match for convolution.")
        L = self.lattice.L if keep_L is None else keep_L
        if keep_L is None and self.lattice.L != other.lattice.L:
            raise ValueError("Convolution with differing L; specify keep_L.")
        p = np.convolve(self.p, other.p)
        # ensure odd length
        if len(p) % 2 == 0:
            p = p[:-1]
        # if longer than needed, truncate via CDF
        if len(p) > 2 * L + 1:
            c = _cdf_from_pdf(p)
            n_extra = (len(p) - (2 * L + 1)) // 2
            c = c[n_extra:-n_extra]
            p_mid = _pdf_from_cdf(c)
        elif len(p) < 2 * L + 1:
            if pad:
                n_extra = (2 * L + 1) - len(p)
                left = n_extra // 2
                right = n_extra - left
                p_mid = np.concatenate([np.zeros(left), p, np.zeros(right)])
            else:
                raise ValueError("Resulting convolution too short; set pad=True or increase L.")
        else:
            p_mid = p
        # correct mean drift via fractional shift
        mu_self = self.mean()
        mu_other = other.mean()
        mu_mid = float(np.dot(p_mid, self.lattice.unit * np.linspace(-L, L, 2 * L + 1)))
        mu_diff = mu_mid - (mu_self + mu_other)
        d_mid = Density(UniformLattice(L, self.lattice.unit), p_mid)
        return d_mid.shift_fractional(-mu_diff / self.lattice.unit)

    def dilate(self, unit_ratio: float = 2.0) -> "Density":
        """Move mass as if unit size increased by unit_ratio (coarser lattice)."""
        L = self.lattice.L
        x_idx = np.arange(-L, L + 1)
        out = np.zeros_like(self.p)
        for idx, prob in zip(x_idx, self.p):
            x = idx / unit_ratio
            # fractional placement between floor/ceil
            if -L + 2 < x < L - 2:
                l = int(np.floor(x))
                u = int(np.ceil(x))
                r = x - l
                lc, uc = 1.0 - r, r
            elif x >= L - 2:
                l, u, lc, uc = L - 2, L - 1, 1.0, 0.0
            else:
                l, u, lc, uc = -L + 1, -L + 2, 0.0, 1.0
            li = min(2 * L, max(l + L, 0))
            ui = min(2 * L, max(u + L, 0))
            out[li] += prob * lc
            out[ui] += prob * uc
        return Density(self.lattice, out)

    # ---- constructors ----
    @classmethod
    def from_callable(
        cls,
        lattice: UniformLattice,
        f: Callable[[float], float],
        *,
        center: bool = True,
    ):
        x = lattice.grid
        p = np.array([max(float(f(xi)), 0.0) for xi in x], dtype=float)
        d = cls(lattice, p)
        return d.center() if center else d

    @classmethod
    def skew_normal(cls, lattice: UniformLattice, loc=0.0, scale=1.0, a=0.0):
        def f(x):
            t = (x - loc) / scale
            return 2.0 / scale * normpdf(t) * normcdf(a * t)

        return cls.from_callable(lattice, f, center=True).shift_fractional(loc / lattice.unit)

    @classmethod
    def gumbel_min(cls, lattice: UniformLattice, loc=0.0, scale=1.0):
        """Minimum-Gumbel density. Independent min-wins races with this base
        reproduce the Luce/softmax law exactly: P(i wins) = softmax(-mu_i / scale).
        With factor correlation (see thurstone.correlated) this becomes a
        correlated softmax race with an exact Luce limit."""

        def f(x):
            z = (x - loc) / scale
            return float(np.exp(z - np.exp(z)) / scale)

        return cls.from_callable(lattice, f, center=True).shift_fractional(loc / lattice.unit)
