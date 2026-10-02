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

METHODS = {}


def register(name):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(mu, V, D, *args, **kw):
            n = len(np.asarray(mu, dtype=float))
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
