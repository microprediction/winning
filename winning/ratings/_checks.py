"""Input contracts shared by the ratings updates and filters."""
from __future__ import annotations

import numpy as np


def as_noise_variances(x, k, name):
    """Observation-noise VARIANCES broadcast to the k observed rows.

    The ratings updates take their noise as a scalar or one value per
    row: tau2 (market noise), beta2 (performance noise) and meas_var
    (measurement noise). None of them was checked. A NaN or inf
    silently turned an observation into the unchanged prior with
    logZ = 0, and a negative value fitted a covariance model with a
    negative component variance and returned plausible finite output
    (#809). Each must be finite and non-negative; zero stays legal (an
    exact observation is a limit, not an invalid input, #778).
    """
    try:
        A = np.broadcast_to(np.asarray(x, dtype=float), (k,)).astype(float)
    except ValueError:
        raise ValueError(
            f"{name} must be a scalar or one variance per observed row "
            f"({k}); got shape {np.shape(x)}") from None
    if not np.isfinite(A).all():
        raise ValueError(f"{name} must be a finite noise variance; got "
                         f"{float(A[~np.isfinite(A)][0])!r}")
    if (A < 0.0).any():
        raise ValueError(f"{name} must be a non-negative noise variance; "
                         f"got {float(A.min())!r}")
    return A


def require_distinct(ids, name="runners"):
    """Identities within one contest must be distinct.

    A repeated id is an entity racing itself. The tracker priced the
    copies as independent runners and then wrote both posteriors to the
    same key, so the last slot won and a winner could be stored as a
    loser; the history and factor fits built repeated rows (#811).
    Deduplicating is not a repair, because scores, order positions and
    covariate rows are aligned by slot.
    """
    seen = {}
    for pos, r in enumerate(ids):
        if r in seen:
            raise ValueError(
                f"{name} repeats {r!r} at positions {seen[r]} and {pos}; "
                f"the entities in one contest must be distinct")
        seen[r] = pos


def as_labels(labels, name):
    """Categorical labels (entity indices, finishing positions) as an int
    array, validated BEFORE any integer cast.

    np.asarray(labels, dtype=int) turned 0.9 into 0, 2.8 into 2, True into
    1 and "2" into 2, so a fractional, Boolean or string label was silently
    redirected to a different entity or finisher and the fit, likelihood,
    design and prediction came back finite and plausible for a model the
    caller never specified (#915). Labels are names, not quantities: a
    whole number is accepted whether spelled as an int or an exactly
    integral float (2 and 2.0 are the same label, as a float column read
    from a frame is), and anything else -- a fraction, a Boolean, a
    string, None, NaN or inf -- is refused. Range and distinctness are
    the caller's checks, as before; see concentration_matrix (#671) for
    the same rule in the polish grammar.
    """
    raw = np.asarray(labels, dtype=object)
    if raw.ndim > 1:
        raise ValueError(f"{name} must be a flat list of whole-number labels; "
                         f"got shape {raw.shape}")
    out = np.empty(raw.size, dtype=int)
    for pos, v in enumerate(raw.ravel()):
        if isinstance(v, (bool, np.bool_)):
            ok = False
        elif isinstance(v, (int, np.integer)):
            ok = True
        elif isinstance(v, (float, np.floating)):
            ok = bool(np.isfinite(v)) and float(v) == int(v)
        else:
            ok = False
        if not ok:
            raise ValueError(
                f"{name} must be whole-number labels (entity indices or "
                f"positions); got {v!r} at position {pos}. A fractional, "
                f"Boolean or string label would be cast to a different "
                f"label silently")
        out[pos] = int(v)
    return out
