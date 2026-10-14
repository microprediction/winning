"""A fixed battery of ORDINARY inputs, shared by the guard tests.

Ordinary means what a caller does every day: a handful to a few dozen
runners, abilities within a few units, idiosyncratic variances of order
one, factor loadings of moderate size, the shipped bases, default
lattice and node settings, and sample covariances drawn from a true
factor model. Nothing here is near-deterministic, extreme in scale or
adversarial -- those cases have their own tests.

`calls()` yields (name, thunk) pairs. Every thunk is deterministic, so
test_ordinary_inputs_quiet.py can require that none of them warns or
raises, and test_default_drift.py can pin what they return.
"""
import numpy as np

from winning.factor import (abilities_from_race, abilities_from_topk,
                            bottom_k_probabilities, ordered_probabilities,
                            race_probabilities, rank_probabilities,
                            top_k_probabilities)
from winning.factor.core import fit_covariance
from winning.factor.polish import race_jacobian

BASES = ("normal", "gumbel", "logistic", "laplace")


def _fields():
    """(label, mu, V, D): a fixed set of ordinary fields."""
    rng = np.random.default_rng(20261014)
    out = []
    for n in (3, 5, 8, 12, 20):
        for r in (0, 1, 2):
            mu = rng.normal(size=n) * rng.uniform(0.4, 1.5)
            V = None if r == 0 else 0.5 * rng.normal(size=(n, r))
            D = rng.uniform(0.4, 1.8, size=n)
            out.append((f"n{n}r{r}", mu, V, D))
    return out


def _sample_covariances():
    """(label, S): sample covariances of a true factor model, 4n draws."""
    rng = np.random.default_rng(1014)
    out = []
    for n in (6, 12, 24):
        for r in (1, 2, 3):
            V = rng.normal(size=(n, r)) * rng.uniform(0.4, 1.2)
            d = rng.uniform(0.3, 1.5, size=n)
            # drawn straight from the factor model: multivariate_normal
            # factors the covariance through the platform's LAPACK, so
            # the same seed gave different samples on Linux, Windows and
            # macOS
            m = 4 * n
            X = (rng.standard_normal((m, r)) @ V.T
                 + rng.standard_normal((m, n)) * np.sqrt(d))
            out.append((f"n{n}r{r}", np.cov(X.T)))
    return out


def _fitted_covariance(S):
    """V V' + diag(D) of the default fit (warnings still surface)."""
    V, D, _F, _W = fit_covariance(S)
    return V @ V.T + np.diag(D)


def calls():
    for label, mu, V, D in _fields():
        n = len(mu)
        for base in BASES:
            yield (f"race/{base}/{label}",
                   lambda mu=mu, V=V, D=D, b=base:
                   race_probabilities(mu, V=V, D=D, base=b))
            yield (f"top2/{base}/{label}",
                   lambda mu=mu, V=V, D=D, b=base:
                   top_k_probabilities(mu, 2, V=V, D=D, base=b))
            yield (f"bottom1/{base}/{label}",
                   lambda mu=mu, V=V, D=D, b=base:
                   bottom_k_probabilities(mu, 1, V=V, D=D, base=b))
            if n <= 12:
                yield (f"rank/{base}/{label}",
                       lambda mu=mu, V=V, D=D, b=base:
                       rank_probabilities(mu, D=D, base=b, V=V))
        if n <= 8:
            yield (f"ordered2/normal/{label}",
                   lambda mu=mu, V=V, D=D:
                   ordered_probabilities(mu, k=2, V=V, D=D))
        yield (f"jacobian/normal/{label}",
               lambda mu=mu, V=V, D=D: race_jacobian(mu, V=V, D=D))
        # round trips from the library's own forward maps
        p = race_probabilities(mu, V=V, D=D)
        yield (f"inv_race/normal/{label}",
               lambda p=p, V=V, D=D: abilities_from_race(p, V=V, D=D))
        q = top_k_probabilities(mu, 2, V=V, D=D)
        yield (f"inv_top2/normal/{label}",
               lambda q=q, V=V, D=D: abilities_from_topk(q, 2, V=V, D=D))
    for label, S in _sample_covariances():
        # the fit's loadings are identified only up to sign and rotation,
        # which rounding can flip; the fitted covariance is what is pinned
        yield (f"fit_covariance/sample/{label}",
               lambda S=S: _fitted_covariance(S))
        mu = np.linspace(-1.0, 1.0, len(S))
        yield (f"race_cov/sample/{label}",
               lambda mu=mu, S=S: race_probabilities(mu, cov=S))


def flatten(result):
    """Every array or float a call returns, as one float vector."""
    parts = []

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                walk(x[k])
        elif isinstance(x, (tuple, list)):
            for y in x:
                walk(y)
        else:
            try:
                a = np.asarray(x, dtype=float)
            except (TypeError, ValueError):
                return
            parts.append(a.ravel())
    walk(result)
    return np.concatenate(parts) if parts else np.zeros(0)
