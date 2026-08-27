"""Transparent scipy acceleration, sklearnex-style: after patch_scipy(),
scipy.stats.multivariate_normal.cdf routes factor-structured covariances
through the fast path and everything else through original scipy,
bit-identical API. Factorization attempts are cached by covariance
content so repeated calls with the same matrix pay the eigendecomposition
once."""

from __future__ import annotations

import hashlib

import numpy as np

from .core import _mvn_cdf_impl, factorize_covariance

_ORIGINAL = {}
_FACTOR_CACHE = {}
_CACHE_LIMIT = 128


def _cov_key(cov):
    a = np.ascontiguousarray(np.asarray(cov, dtype=float))
    return hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


def _cached_factorization(cov):
    key = _cov_key(cov)
    if key not in _FACTOR_CACHE:
        if len(_FACTOR_CACHE) >= _CACHE_LIMIT:
            _FACTOR_CACHE.clear()
        _FACTOR_CACHE[key] = factorize_covariance(np.asarray(cov, float))
    return _FACTOR_CACHE[key]


def patch_scipy():
    """Route scipy.stats.multivariate_normal.cdf through fastmvn when the
    covariance is exactly factor-plus-diagonal; identical results
    otherwise via the original implementation."""
    from scipy.stats import _multivariate as m
    if "cdf" in _ORIGINAL:
        return
    _ORIGINAL["cdf"] = m.multivariate_normal_gen.cdf

    def cdf(self, x, mean=None, cov=1, allow_singular=False, maxpts=None,
            abseps=1e-5, releps=1e-5, *, lower_limit=None):
        try:
            c = np.atleast_2d(np.asarray(cov, dtype=float))
            if c.ndim == 2 and c.shape[0] == c.shape[1] and c.shape[0] > 2:
                fd = _cached_factorization(c)
                if fd is not None:
                    V, D = fd
                    n = c.shape[0]
                    mu = np.zeros(n) if mean is None else \
                        np.asarray(mean, dtype=float)
                    lo = None if lower_limit is None else lower_limit
                    p, _ = _mvn_cdf_impl(lo, x, mu, None, V, D)
                    return p
        except Exception:
            pass  # any surprise routes to original scipy below
        return _ORIGINAL["cdf"](self, x, mean=mean, cov=cov,
                                allow_singular=allow_singular,
                                maxpts=maxpts, abseps=abseps,
                                releps=releps, lower_limit=lower_limit)

    m.multivariate_normal_gen.cdf = cdf


def unpatch_scipy():
    from scipy.stats import _multivariate as m
    if "cdf" in _ORIGINAL:
        m.multivariate_normal_gen.cdf = _ORIGINAL.pop("cdf")
