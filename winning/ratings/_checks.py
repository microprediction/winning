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
