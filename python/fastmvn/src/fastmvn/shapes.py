"""The shape contract at the Python boundary: one spelling rule, one error.

Every public verb in this package that takes factor loadings calls them
``V`` and means the same thing: an ``(n, rank)`` matrix, one ROW per
contestant, for ``Sigma = V V\' + diag(D)``. That contract was previously
re-implemented at twenty-odd call sites in three different ways --
``np.atleast_2d`` alone, ``np.atleast_2d`` plus a hand-rolled transpose,
or an explicit row-count check -- so the same array was accepted here,
rejected there, and silently misread somewhere else.

``np.atleast_2d`` alone is the dangerous one. It reads a bare length-n
vector as ``(1, n)`` -- ONE contestant carrying n factors -- which is a
valid shape for the code that follows, so nothing raises. In
``winning.factor`` the gauge-fix then subtracted the single row from
itself, the loadings became zero, and the race quietly degenerated to
the independent one (issue #66). Elsewhere it surfaced as "not enough
values to unpack" or "tuple index out of range" several frames below the
call.

These two normalisers are the one place that decision is made.
"""
from __future__ import annotations

import numpy as np


def as_loadings(V, n):
    """Factor loadings as the ``(n, rank)`` matrix the kernels contract for.

    Accepted spellings, all of which describe the same race:

    ==================  ==================================================
    ``()`` scalar       one common loading for every contestant
    ``(n,)``            rank-one loadings, one per contestant
    ``(n, rank)``       the contract shape, returned unchanged
    ``(rank, n)``       transposed to the contract shape
    ==================  ==================================================

    Anything else raises ``ValueError`` naming the expected shape, so no
    compiled kernel is reachable with a mis-shaped array. ``(n, 0)`` is
    kept: an empty V is the independent race, as the grammar documents.

    The ``(rank, n)`` transpose is a genuine ambiguity only when
    ``rank == n``, where the contract wins: n rows is n contestants.
    """
    A = np.asarray(V, dtype=float)
    # Finiteness, as as_idio and as_weights already check it. This was
    # the one of the three that let a NaN through, and the loading then
    # travelled to the lattice sizing and came back as "cannot convert
    # float NaN to integer" -- a refusal, so no wrong answer, but one
    # that names nothing the caller passed and sends them looking at the
    # wrong module.
    bad = np.flatnonzero(~np.isfinite(A.ravel()))
    if bad.size:
        raise ValueError(
            f"V has a non-finite entry: V is a loading matrix, and "
            f"{bad.size} of {A.size} entries are not finite (first at "
            f"flat index {int(bad[0])}, value {float(A.ravel()[bad[0]])!r})")
    if A.ndim == 0:
        return np.full((n, 1), float(A))
    if A.ndim == 1 and A.size == n:
        return np.ascontiguousarray(A.reshape(n, 1))
    if A.ndim == 2:
        # ascontiguousarray, not the bare transpose view: BLAS reduces a
        # C-ordered and an F-ordered matrix in different orders, so the
        # (rank, n) spelling otherwise answered the (n, rank) one to
        # 5.5e-17 rather than exactly. Same loadings, same bits.
        if A.shape[0] == n:
            return np.ascontiguousarray(A)
        if A.shape[1] == n:
            return np.ascontiguousarray(A.T)
    raise ValueError(
        f"V must carry one row per contestant: expected a scalar, a "
        f"vector of length {n}, or an ({n}, rank) matrix; got shape "
        f"{A.shape}")


def _as_variance(x, n, name, what, positive=False):
    """A length-n vector of VARIANCES: right length, finite, non-negative.

    The non-negativity is not pedantry, it is the only thing telling a
    variance from a rank-one loading vector. `v` (belief variance) and
    `V` (loadings) differ by case alone in five public signatures, and a
    rank-one V has exactly the same shape as v, so passing one for the
    other used to be silent -- measured at up to 1.09 on the posterior
    mean, and a sqrt of a negative number (NaN, RuntimeWarning only) in
    simulate.correlated_draws.

    Loadings are gauge-fixed to mean zero, so any non-trivial loading
    vector HAS a negative entry, which is what makes this cheap check
    catch the swap. A variance of exactly zero is legal (a perfectly
    known quantity); a negative one never is.
    """
    A = np.asarray(x, dtype=float)
    if A.ndim == 0:
        A = np.full(n, float(A))
    elif A.shape != (n,):
        raise ValueError(
            f"{name} must be one {what} per contestant: expected a scalar "
            f"or shape ({n},); got shape {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} has a non-finite entry: {name} is a "
                         f"{what} and must be finite")
    # A TOLERANCE, not >= 0. A variance that is negative only by
    # round-off IS zero -- a symmetric eigendecomposition routinely
    # leaves a -1e-18 where the true value is 0, and fit_covariance
    # returns idiosyncratic variances down at 1e-6 -- so a strict test
    # would raise on a caller doing nothing wrong. Loading entries are
    # O(0.1-1) once gauge-fixed, so a relative 1e-12 still separates a
    # swapped loading vector from round-off by twelve orders.
    tol = 1e-12 * max(1.0, float(np.abs(A).max()) if A.size else 0.0)
    bad = A < -tol
    if bad.any():
        raise ValueError(
            f"{name} has {int(bad.sum())} negative entr"
            f"{'y' if bad.sum() == 1 else 'ies'} (min {float(A.min()):.4g}, "
            f"tolerance -{tol:.2g}): {name} is a {what} and cannot be "
            f"negative. A loading vector reaching a variance argument is "
            f"the usual cause -- loadings are gauge-fixed to mean zero, so "
            f"they carry negative entries and a variance never does.")
    if (A < 0.0).any():
        A = np.maximum(A, 0.0)      # within tolerance: these ARE zero
    if positive and (A <= 0.0).any():
        # A zero VARIANCE is legal as a belief (a perfectly known
        # quantity) but not as performance noise on the lattice, which
        # divides by the standard deviation: before this check a zero D
        # surfaced as OverflowError from the grid sizing, deep in
        # forward_grid, with no hint of the cause.
        k = int((A <= 0.0).sum())
        raise ValueError(
            f"{name} must be strictly positive here: {k} entr"
            f"{'y is' if k == 1 else 'ies are'} zero. A zero {what} is a "
            f"point mass, and the lattice divides by its standard "
            f"deviation. Use a small positive variance.")
    return A


def as_idio(D, n, positive=False):
    """Idiosyncratic VARIANCES as the length-n vector the kernels contract for.

    A scalar is the same variance for every contestant. A wrong length,
    a non-finite entry or a negative variance raises here rather than
    reaching a broadcast error several frames down, or the compiled
    kernel. ``positive=True`` (the lattice kernels) also rejects an exact
    zero, which the lattice cannot represent.
    """
    return _as_variance(D, n, "D", "idiosyncratic variance", positive=positive)


def as_variance(v, n):
    """BELIEF variances as the length-n vector the ratings verbs contract for.

    The companion to as_loadings on the other side of the same
    signature: `update_winner_correlated(m, v, winner, V)` takes both,
    they have the same shape at rank one, and exchanging them was
    silent before this existed.
    """
    return _as_variance(v, n, "v", "belief variance")

def as_weights(W, name="W"):
    """Quadrature weights, normalised to sum to one.

    They are RELATIVE: W and c*W for positive c describe the same factor
    law, and every verb normalises its own result, so the common scale
    cancels. It did not cancel in the pre-normalisation mass checks,
    which compared an unnormalised row sum against 1 (#208).

    A negative entry is refused, not merely a negative TOTAL. `sum > 0`
    admits a SIGNED rule such as [2, -1], which passed and returned
    "probabilities" of -0.144 and 1.999 (#263). An individual ZERO is
    fine: it is a node that contributes nothing.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional weight vector; "
                         f"got shape {W.shape}")
    if not np.isfinite(W).all():
        raise ValueError(f"{name} has a non-finite entry")
    neg = np.flatnonzero(W < 0.0)
    if neg.size:
        raise ValueError(
            f"{name}[{int(neg[0])}] = {float(W[neg[0]])!r} is negative. "
            "Quadrature weights here are relative and must be "
            "non-negative; a signed rule can return values outside [0, 1].")
    tot = float(W.sum())
    if tot <= 0.0:
        raise ValueError(
            f"{name} must have a positive total; got {tot!r}. They are "
            "relative, so any positive multiple of a valid rule is the "
            "same factor law, but an all-zero rule is not one.")
    return W / tot
