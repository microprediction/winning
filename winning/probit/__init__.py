"""Factor multinomial probit, in the language of its literature.

Max-wins convention: HIGHER utility wins, shares sum to one, exactly as
in "Scalable Share Calibration for Factor Multinomial Probit Models"
(papers/factor-probit-transform). This module is the one audited
reflection point onto the package's internal min-wins race (valid
because the Gaussian law is symmetric; non-Gaussian bases live in
winning.factor.races, where the reflection is the caller's business).

    shares(utilities, V=V, D=D)          all N choice probabilities
    utilities_from_shares(p, V=V, D=D)   the paper's calibration
    shares(utilities, Sigma=Sigma, k=2)  supplied covariance: fits the
                                         certified contrast factor model
                                         first (return_fit=True to get
                                         the fitted V, D back)
"""

from __future__ import annotations

import numpy as np

from ..factor.core import (abilities_from_probabilities_factor,
                           factor_model_contrast, hermite_nodes,
                           win_probabilities_factor)

__all__ = ["shares", "utilities_from_shares", "fit_factor_model"]


def fit_factor_model(Sigma, k):
    """Certified contrast-space factor fit: Sigma ~ V V' + diag(D) on the
    choice-relevant quotient. Returns (V, D)."""
    return factor_model_contrast(np.asarray(Sigma, dtype=float), k)


def _prepare(n, V, D, Sigma, k):
    if Sigma is not None:
        if k is None:
            raise ValueError("supply the factor rank k with Sigma")
        V, D = fit_factor_model(Sigma, k)
    if V is None:
        V = np.zeros((n, 1))
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[0] != n:
        raise ValueError(
            f"V has {V.shape[0]} rows but there are {n} alternatives")
    D = np.ones(n) if D is None else np.asarray(D, dtype=float)
    F, W = hermite_nodes(V.shape[1])
    return V, D, F, W


def shares(utilities, V=None, D=None, Sigma=None, k=None, points=501,
           return_fit=False):
    """Choice probabilities of the factor probit model (max-wins)."""
    u = np.asarray(utilities, dtype=float)
    V, D, F, W = _prepare(len(u), V, D, Sigma, k)
    p = win_probabilities_factor(-u, V, D, F, W, points=points)
    return (p, V, D) if return_fit else p


def utilities_from_shares(p, V=None, D=None, Sigma=None, k=None,
                          points=501, tol=1e-6):
    """Mean-zero utilities reproducing the observed shares (max-wins)."""
    p = np.asarray(p, dtype=float)
    V, D, F, W = _prepare(len(p), V, D, Sigma, k)
    a = abilities_from_probabilities_factor(p, V, D, F, W, tol=tol,
                                            points=points)
    return -a
