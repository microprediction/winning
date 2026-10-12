"""Gaussian evidence on the contrast space, shared by every conjugate
ratings node (market prices, margins, scores, team cardinals).

An observation of centred performances lives on the (k-1)-dimensional
space orthogonal to 1. Its covariance M = P C P is singular along 1 by
construction, and possibly along further directions when the model
itself is degenerate. The copies of this computation decided rank by
eigenvalue tests that were absolute (lam > 1e-12, which at ability unit
1e-7 dropped every contrast and returned logZ = 0 for any observation)
or relative to the largest eigenvalue at 1e-10 (which deleted a valid
1e-4 contrast beside a 1e8-noisy one), and they projected away whatever
residual fell outside the kept directions (#651). Here the common
direction is removed exactly by an orthonormal contrast basis, the rank
test is the numerical one (a few ulps of the largest eigenvalue), and
an observation with a material component outside the model's support is
refused rather than given a density.
"""

from __future__ import annotations

import numpy as np

_basis_cache = {}


def contrast_basis(k):
    """Orthonormal (k, k-1) basis of the complement of 1 (Helmert)."""
    B = _basis_cache.get(k)
    if B is None:
        B = np.zeros((k, max(k - 1, 0)))
        for j in range(1, k):
            B[:j, j - 1] = 1.0
            B[j, j - 1] = -float(j)
            B[:, j - 1] /= np.sqrt(j * (j + 1.0))
        _basis_cache[k] = B
    return B


def contrast_gaussian(r, M, scale=None):
    """Evidence of the centred residual r under N(0, M) on the contrast
    space, and the pseudo-inverse of M there.

    Returns (logZ, Minv) with Minv the (k, k) pseudo-inverse restricted
    to the supported contrasts. `scale` is the size of the quantities r
    was formed from, for the support test (defaults to |r|).
    """
    r = np.asarray(r, dtype=float)
    k = r.shape[0]
    if k < 2:
        return 0.0, np.zeros((k, k))
    B = contrast_basis(k)
    Mc = B.T @ M @ B
    Mc = 0.5 * (Mc + Mc.T)
    lam, U = np.linalg.eigh(Mc)
    top = max(float(np.abs(lam).max()), 1e-300)
    keep = lam > 16.0 * k * np.finfo(float).eps * top
    z = U.T @ (B.T @ r)
    if not keep.all():
        ref = float(np.linalg.norm(r)) if scale is None else float(scale)
        off = float(np.linalg.norm(z[~keep]))
        if off > 1e-8 * max(ref, 1e-300) and off > 0.0:
            raise ValueError(
                "observation lies outside the support of its model: a "
                f"contrast with zero modelled variance moved by {off:.3g}. "
                "Its likelihood is zero; add measurement noise (meas_var=, "
                "beta2=) if such contrasts can vary.")
    zk, lk = z[keep], lam[keep]
    logZ = float(-0.5 * (np.sum(zk * zk / lk) + np.sum(np.log(lk))
                         + keep.sum() * np.log(2.0 * np.pi)))
    W = B @ U[:, keep]
    return logZ, (W / lk) @ W.T
