"""Validation of observed outcomes and of scale parameters at the
likelihood doors.

One place decides what a legal order, prefix or temperature is, for the
same reason `winning.shapes` decides parameter shapes: each likelihood
used to coerce its input with `np.asarray(..., dtype=int)` and consume
it, so a duplicate finisher scored as evidence (a prefix probability of
52.6), a negative label silently indexed from the end, 0.9 truncated to
0, and a negative temperature fitted the reversed preference model
(#129, #366, #390).
"""

from __future__ import annotations

import numpy as np


def as_order(order, n, name="order", full=False, min_len=1):
    """A sequence of DISTINCT integer labels in 0..n-1, as an int array.

    full=True requires a complete permutation of the n labels. Raises
    ValueError naming the first bad entry; never truncates, wraps or
    drops one.
    """
    a = np.asarray(order)
    if a.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {a.shape}")
    if a.size and not np.issubdtype(a.dtype, np.integer):
        if a.dtype == bool or not np.issubdtype(a.dtype, np.number):
            raise ValueError(f"{name} must hold integer labels; got {a.dtype}")
        bad = np.flatnonzero(~np.isfinite(a) | (a != np.floor(a)))
        if bad.size:
            raise ValueError(
                f"{name}[{int(bad[0])}] = {a[bad[0]]!r} is not an integer "
                "label; it would be truncated, not rejected")
    a = a.astype(int)
    if a.size < min_len:
        raise ValueError(f"{name} needs at least {min_len} label(s); got {a.size}")
    out = np.flatnonzero((a < 0) | (a >= n))
    if out.size:
        raise ValueError(
            f"{name}[{int(out[0])}] = {int(a[out[0]])} is outside 0..{n - 1} "
            "(a negative label would silently index from the end)")
    if len(np.unique(a)) != a.size:
        vals, counts = np.unique(a, return_counts=True)
        raise ValueError(
            f"{name} repeats label {int(vals[counts > 1][0])}: a runner "
            "cannot finish twice, and a repeat scored as ordinary evidence "
            "(probabilities above one)")
    if full and a.size != n:
        raise ValueError(
            f"{name} must be a complete order of all {n} labels; got {a.size}")
    return a


def as_luce_temperature(temperature, name="temperature"):
    """A Gumbel / softmax scale: finite and strictly positive."""
    tau = float(temperature)
    if not (np.isfinite(tau) and tau > 0.0):
        raise ValueError(
            f"{name} must be finite and positive; got {tau!r} (a negative "
            "scale reverses every comparison, zero divides by zero, and "
            "infinity erases the utilities)")
    return tau


def as_soft_temperature(temperature, name="temperature"):
    """A softening scale where 0 (or None) is the hard-race sentinel:
    finite and non-negative. Negative and NaN values used to be read as
    zero and silently select the hard race."""
    if temperature is None:
        return temperature
    tau = float(temperature)
    if not (np.isfinite(tau) and tau >= 0.0):
        raise ValueError(
            f"{name} must be finite and >= 0 (0 is the hard race); got "
            f"{tau!r}")
    return temperature
