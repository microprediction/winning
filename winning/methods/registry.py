"""Method registry: name -> callable(mu, V, D, budget, seed) -> (p, info).

Every method takes the same ``(mu, V, D)``, and each used to unpack the
loadings its own way: a length-n vector was reshaped correctly by two of
them, and produced "not enough values to unpack" or "tuple index out of
range" three frames down in the rest. One signature deserves one
contract, so ``register`` normalises the shapes on the way in and the
twelve bodies see the ``(n, rank)`` matrix they were all written for.
"""
import functools

import numpy as np

from ..shapes import as_idio, as_loadings
from ..factor.races import _as_abilities

METHODS = {}


def as_draws(budget, name="budget"):
    """A simulation budget: a whole number of at least one draw.

    The sampling methods used the budget as an array length or a divisor
    unchecked. Zero draws returned an all-NaN probability vector from
    five of them (log 0 minus log 0, then a normalisation), a fraction
    was truncated without a word, and True ran one draw (#668).
    """
    v = np.asarray(budget)
    if (v.ndim != 0 or v.dtype == bool or not np.issubdtype(v.dtype, np.number)
            or np.iscomplexobj(v) or not np.isfinite(v) or v != np.floor(v)):
        raise ValueError(
            f"{name} must be a whole number of draws; got {budget!r}")
    v = int(v)
    if v < 1:
        raise ValueError(f"{name} must be at least 1 draw; got {v}")
    return v


def register(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(mu, V, D, *args, **kw):
            mu = np.asarray(mu, dtype=float)
            if mu.ndim != 1:
                raise ValueError("mu must be a one-dimensional ability vector")
            # Validate before the covariance bypass and every method shortcut.
            # In particular, argmax can turn NaN into plausible certainty.
            mu = _as_abilities(mu)
            n = len(mu)
            # A method handed the covariance DIRECTLY has no loadings to
            # normalise, and forcing it to supply a factorization is what
            # #302 was: the caller factored, the method rebuilt, and the
            # round trip lost the contrast. Narrow on purpose -- only an
            # explicit cov= takes this door, and the method validates the
            # matrix itself.
            if kw.get("cov") is not None:
                return fn(mu, V, D, *args, **kw)
            return fn(mu, as_loadings(V, n), as_idio(D, n), *args, **kw)
        METHODS[name] = wrapped
        return wrapped
    return deco


def get_method(name):
    return METHODS[name]
