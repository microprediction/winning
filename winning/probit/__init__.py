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
                                         contrast-space factor model
                                         first (return_fit=True to get
                                         the fitted V, D back)
"""

from __future__ import annotations

import numpy as np

from ..factor.core import (abilities_from_probabilities_factor,
                           factor_model_projected, hermite_nodes,
                           win_probabilities_factor)

__all__ = ["shares", "utilities_from_shares", "calibrate_utilities",
           "removal_shares", "fit_factor_model"]


def fit_factor_model(Sigma, k):
    """Contrast-space factor fit: Sigma ~ V V' + diag(D) on the
    choice-relevant quotient, i.e. minimising ||P (Sigma - V V' - diag D)
    P||_F with P the centring projection. Returns (V, D) with V centred
    (P V = V) and canonicalised (SVD, sign convention), so results are
    reproducible at the covariance level; the fit is a certified
    alternation (factor_model_projected), not a global certificate.

    Until 2026-09-11 this called factor_model_contrast, which applies
    ordinary principal-factor analysis to P Sigma P although the
    idiosyncratic part in contrast space, P diag(D) P, is not diagonal:
    for an INDEPENDENT race (Sigma = I, k = 1) it invented a factor from
    the centring artefact and moved the shares by 0.03-0.05 (issue #27).
    The projected fit returns V = 0, D = diag(Sigma) there, exactly.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(Sigma)
    V, D = factor_model_projected(Sigma, k)
    P = np.eye(n) - np.ones((n, n)) / n
    V = P @ V
    A, sv, _ = np.linalg.svd(V, full_matrices=False)
    V = A[:, :k] * sv[:k]
    for j in range(V.shape[1]):
        i0 = np.argmax(np.abs(V[:, j]))
        if V[i0, j] < 0:
            V[:, j] = -V[:, j]
    return V, D


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


def removal_shares(utilities, V=None, D=None, Sigma=None, k=None,
                   points=501, mass_tol=1e-3):
    """q[i][j] = P(j chosen | i removed), max-wins, rows summing to one.

    Routed through the safe top-level implementation (sixth review): the
    deletion rows out of the shared-field pass are normalized without a
    coverage window, sharpest-runner refinement or mass check, so
    removing a dominant favorite could return a plausible row that the
    lattice never resolved. This entry point inherits all three, and
    raises rather than normalizing a defective row away."""
    from ..factor.races import removal_shares as _removal
    u = np.asarray(utilities, dtype=float)
    V, D, F, W = _prepare(len(u), V, D, Sigma, k)
    return _removal(-u, V=V, D=D, F=F, W=W, points=points,
                    mass_tol=mass_tol)


# canonical calibrate_* family name; calibrate_factors is reserved for
# the outer estimation problem
calibrate_utilities = utilities_from_shares
