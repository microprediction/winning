"""The general race: one API, distributions and correlation as parameters.

    race_probabilities(mu)                          classic independent race
    race_probabilities(mu, V=V, D=D)                factor probit (Gaussian)
    race_probabilities(mu, base="gumbel")           Luce / softmax, exactly
    race_probabilities(mu, V=V, base="gumbel")      correlated Luce
    race_probabilities(mu, base=my_base)            anything standardized

Min-wins convention throughout. A base is a callable z -> (S, f, fp)
giving survival, density and density derivative of a MEAN-ZERO,
UNIT-VARIANCE law (standardization keeps noise family separate from
noise scale). Zero factors is literally the one-node quadrature, so the
independent race is not a separate code path.

Promoted from research/experiments/exp14_boundaries/run_boundaries.py,
where the general engine was exercised by the paper's substitution
experiments (the Gumbel base's zero-loading case equals softmax to
2.8e-17 there).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .core import hermite_nodes

_EULER = 0.5772156649015329


def _normal(z):
    S = np.maximum(1.0 - ndtr(z), 1e-300)
    f = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return S, f, -z * f


def _gumbel_min(z):
    c = np.pi / np.sqrt(6.0)
    u = np.minimum(z * c - _EULER, 30.0)
    eu = np.exp(u)
    S = np.maximum(np.exp(-eu), 1e-300)
    f = c * eu * S
    return S, f, c * c * eu * S * (1.0 - eu)


BASES = {"normal": _normal, "gumbel": _gumbel_min}
_SPANS = {"normal": (8.0, 8.0), "gumbel": (22.0, 8.0)}   # (left, right) tails


def _setup(mu, V, D, F, W, base):
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    D = np.ones(n) if D is None else np.asarray(D, dtype=float)
    if V is None:
        V = np.zeros((n, 1))
        F, W = np.zeros((1, 1)), np.ones(1)
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if F is None or W is None:
            F, W = hermite_nodes(V.shape[1])
    fn = base if callable(base) else BASES[base]
    left, right = _SPANS.get(base, (12.0, 12.0)) if not callable(base) \
        else (12.0, 12.0)
    return mu, V, D, np.asarray(F, float), np.asarray(W, float), fn, left, right


def race_probabilities(mu, V=None, D=None, F=None, W=None, base="normal",
                       points=501, return_slopes=False):
    """Win probabilities of the general race, all N in one field pass."""
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    x = np.linspace(M_all.min() - left * sd.max(),
                    M_all.max() + right * sd.max(), points)
    dx = x[1] - x[0]
    p = np.zeros(n)
    slope = np.zeros(n)
    chunk = max(1, int(5e6 / (n * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S, f, fp = fn(z)
        f = f / sd[None, :, None]
        logS = np.log(S)
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)
        slope += Wc @ (np.sum(-fp / sd[None, :, None] ** 2 * rest, axis=2) * dx)
    total = p.sum()
    if return_slopes:
        return p / total, slope / total
    return p / total


def abilities_from_race(p, V=None, D=None, F=None, W=None, base="normal",
                        points=501, n_iter=60, tol=1e-8):
    """Invert the general race: mean-zero mu with race_probabilities(mu) = p."""
    target = np.asarray(p, dtype=float)
    if np.any(target <= 0):
        raise ValueError("all target probabilities must be positive")
    target = target / target.sum()
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    for _ in range(n_iter):
        phat, sl = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base,
                                      points=points, return_slopes=True)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        if np.abs(resid).max() < tol:
            break
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        mu = mu - np.clip(resid / dlogp, -2.0, 2.0)
        mu -= mu.mean()
    return mu
