"""Rank-r factor approximations of a posterior covariance.

The headline construction is the oracle of the experiment brief: truncate the
eigendecomposition and give the residual back to the diagonal, so that every
candidate keeps its exact marginal variance and only the off-diagonal
dependence is approximated.

    Sigma_r = V_r V_r' + D_r,
    V_r = U[:, :r] Lambda[:r]^{1/2},
    D_r = diag(diag(Sigma) - diag(V_r V_r')).

Because the diagonal is exact by construction, the fraction of the trace
captured by r eigenvalues is the wrong error measure -- most of that trace is
idiosyncratic variance that D_r reproduces exactly. What rank r has to buy is
the off-diagonal, so cov_error reports the off-diagonal residual.

contrast_factor is the same idea run in the choice-relevant quotient space.
Choice probabilities depend on Sigma only through P Sigma P with
P = I - 11'/N: a factor that loads equally on every candidate shifts all of
them together and cannot change the argmax. A molecular similarity matrix has
exactly such a component -- all Tanimoto similarities are positive, so the top
eigenvector is close to the all-ones direction -- and spending the first factor
on it is spending it on nothing.
"""

from __future__ import annotations

import numpy as np


def top_eigen(M: np.ndarray, r_max: int):
    """Descending top-r_max eigenpairs. Cache this: eigh is O(N^3) and the rank
    ladder would otherwise pay for it once per rank."""
    w, U = np.linalg.eigh(np.asarray(M, dtype=float))
    idx = np.argsort(w)[::-1][:r_max]
    return w[idx], U[:, idx]


def eig_factor(Sigma: np.ndarray, r: int, floor_rel: float = 1e-12,
               eig=None):
    """Oracle rank-r factor model with the exact marginal variances.

    eig, if given, is (w, U) from top_eigen(Sigma, r_max) for some r_max >= r.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(Sigma)
    dg = np.diag(Sigma).copy()
    if r <= 0:
        return np.zeros((n, 0)), dg.copy()
    if eig is None:
        w, U = top_eigen(Sigma, r)
    else:
        w, U = eig[0][:r], eig[1][:, :r]
    V = U[:, :r] * np.sqrt(np.maximum(w[:r], 0.0))
    d = dg - np.sum(V ** 2, axis=1)
    d = np.maximum(d, floor_rel * float(np.median(dg)))
    return V, d


def project(M: np.ndarray) -> np.ndarray:
    """P M P for P = I - 11'/N, in O(N^2) rather than two matrix products."""
    M = np.asarray(M, dtype=float)
    row = M.mean(axis=1)
    return M - row[:, None] - row[None, :] + float(row.mean())


def contrast_factor(Sigma: np.ndarray, r: int, floor_rel: float = 1e-9,
                    eig=None):
    """Rank-r factor model fitted to P Sigma P, the part choices can see.

    Argmax probabilities depend on Sigma only through the difference variances
    (e_i - e_j)' Sigma (e_i - e_j), and e_i - e_j sums to zero, so they depend
    on Sigma only through C = P Sigma P. This routine therefore truncates the
    eigendecomposition of C and hands the residual to the diagonal of C:

        V = U_C[:, :r] Lambda_C[:r]^{1/2},  d = diag(C) - diag(V V').

    The returned model does NOT reproduce the marginal variances of Sigma, and
    that is deliberate -- a first attempt that restored them scored worse than
    plain eig_factor, because restoring the marginals puts the common variance
    back on the diagonal, and while a common FACTOR is choice-irrelevant a
    common addition to the idiosyncratic variances is not: it inflates every
    difference variance. Only the factor direction may be dropped.
    """
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(Sigma)
    C = project(Sigma)
    dg = np.diag(C).copy()
    if r <= 0:
        return np.zeros((n, 0)), np.maximum(dg, floor_rel * float(np.median(np.abs(dg))))
    if eig is None:
        w, U = top_eigen(C, r)
    else:
        w, U = eig[0][:r], eig[1][:, :r]
    V = U[:, :r] * np.sqrt(np.maximum(w[:r], 0.0))
    d = dg - np.sum(V ** 2, axis=1)
    d = np.maximum(d, floor_rel * float(np.median(np.abs(dg))))
    return V, d


def cov_error(Sigma: np.ndarray, V: np.ndarray, d: np.ndarray) -> dict:
    """Off-diagonal reconstruction error, absolute and correlation-scaled."""
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(Sigma)
    R = Sigma - (V @ V.T + np.diag(d))
    off = ~np.eye(n, dtype=bool)
    s = np.sqrt(np.diag(Sigma))
    corr_res = R[off] / (s[:, None] * s[None, :])[off]
    corr_true = Sigma[off] / (s[:, None] * s[None, :])[off]
    return {
        "cov_offdiag_fro": float(np.sqrt(np.sum(R[off] ** 2))),
        "cov_offdiag_rel": float(np.sqrt(np.sum(R[off] ** 2) /
                                         max(np.sum(Sigma[off] ** 2), 1e-300))),
        "corr_offdiag_rmse": float(np.sqrt(np.mean(corr_res ** 2))),
        "corr_offdiag_max": float(np.max(np.abs(corr_res))),
        "corr_true_rmse": float(np.sqrt(np.mean(corr_true ** 2))),
        "diag_max_abs_err": float(np.max(np.abs(np.diag(R)))),
    }


def quotient_cov_error(Sigma: np.ndarray, V: np.ndarray, d: np.ndarray) -> dict:
    """Reconstruction error of P Sigma P, the part choices can see.

    This is the error measure that governs the probabilities: two covariances
    with the same P Sigma P give identical argmax probabilities.
    """
    PA = project(np.asarray(Sigma, dtype=float) - (V @ V.T + np.diag(d)))
    PS = project(np.asarray(Sigma, dtype=float))
    return {
        "quot_fro": float(np.linalg.norm(PA)),
        "quot_rel": float(np.linalg.norm(PA) / max(np.linalg.norm(PS), 1e-300)),
    }


def effective_rank(Sigma: np.ndarray, quotient: bool = False) -> dict:
    """Spectral summaries: participation ratio and trace-capture thresholds."""
    Sigma = np.asarray(Sigma, dtype=float)
    M = Sigma
    if quotient:
        row = Sigma.mean(axis=1)
        M = Sigma - row[:, None] - row[None, :] + float(row.mean())
    w = np.linalg.eigvalsh(M)[::-1]
    w = np.maximum(w, 0.0)
    tot = w.sum()
    frac = np.cumsum(w) / tot
    out = {"trace": float(tot),
           "participation_ratio": float(tot ** 2 / np.sum(w ** 2)),
           "erank_shannon": float(np.exp(-np.sum((w / tot) *
                                                 np.log(np.maximum(w / tot, 1e-300)))))}
    for t in (0.5, 0.9, 0.99):
        out[f"r_for_{int(t * 100)}pct_trace"] = int(np.searchsorted(frac, t) + 1)
    return out
