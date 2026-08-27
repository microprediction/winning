"""Conditioning a factor posterior on one new observation, without leaving the
factor family.

Entropy Search and its relatives need the argmax distribution AFTER a
hypothetical observation, for every candidate that might be observed, and that
step is what makes them expensive. Under a factor posterior it is not
expensive, and it is not even approximate.

Observing candidate k with value y under observation noise sigma_n^2 is a
rank-one Gaussian downdate,

    sigma = Sigma e_k,   s = Sigma_kk + sigma_n^2,
    mu'    = mu + (y - mu_k) sigma / s,
    Sigma' = Sigma - sigma sigma' / s.

The useful fact is what this does RESTRICTED TO THE CANDIDATES STILL IN PLAY.
Once k is observed it leaves the field, and off the k-th coordinate the
diagonal part of Sigma contributes nothing to the column:

    sigma_{-k} = V_{-k} v_k                       (no e_k term survives)
    Sigma'_{-k} = V_{-k} (I_r - v_k v_k' / s) V_{-k}' + D_{-k}

So the update is EXACT, keeps the same rank r, and leaves the idiosyncratic
diagonal completely unchanged. The core I - v_k v_k'/s is positive definite
because |v_k|^2 / s = |v_k|^2 / (|v_k|^2 + d_k + sigma_n^2) < 1, so it has a
Cholesky factor L and V_new = V_{-k} L. Cost O(N r^2), no N x N matrix, no
truncation, nothing to floor.

If instead the observed candidate is KEPT in the field, the exact update needs
rank r+1 with an INDEFINITE core (the e_k direction enters with a negative
eigenvalue that the diagonal absorbs), which the V V' + D form cannot hold. An
earlier version of this module tried to keep k and clipped that negative
eigenvalue, which is wrong -- it silently discarded real structure and only
looked correct in the noiseless case, where the negative direction happens to
vanish. condition_keep below is the honest version of that, and says so.

test_condition.py checks both against forming Sigma densely.
"""

from __future__ import annotations

import numpy as np


def sigma_column(V, d, k):
    """Column k of V V' + diag(d), in O(N r)."""
    V = np.atleast_2d(np.asarray(V, dtype=float))
    d = np.asarray(d, dtype=float)
    col = V @ V[k]
    col[k] += d[k]
    return col


def condition_drop(mu, V, d, k, y, noise: float = 0.0):
    """Observe candidate k = y, remove it, return the exact posterior on the rest.

    Exact: same rank, same diagonal, no truncation. O(N r^2).
    """
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    d = np.asarray(d, dtype=float)
    r = V.shape[1]
    vk = V[k].copy()
    s = float(vk @ vk + d[k] + noise)
    if s <= 0:
        raise ValueError("non-positive predictive variance")

    keep = np.arange(len(mu)) != k
    Vk = V[keep]
    mu_new = mu[keep] + (y - mu[k]) * (Vk @ vk) / s

    M = np.eye(r) - np.outer(vk, vk) / s
    M = 0.5 * (M + M.T)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:                 # numerical edge, |v_k|^2 ~ s
        w, U = np.linalg.eigh(M)
        L = U * np.sqrt(np.maximum(w, 0.0))
    return mu_new, Vk @ L, d[keep].copy()


def condition_keep(mu, V, d, k, y, noise: float = 0.0, rank: int | None = None):
    """Same update but keeping k in the field, which costs an approximation.

    The exact updated covariance is A C A' + D with A = [V | e_k] and an
    indefinite core C = diag(I_r, 0) - g g'/s, g = [v_k ; d_k]. A V V' + D model
    cannot represent the negative direction, so the returned model keeps the
    positive part and restores the exact marginal variances through the
    diagonal -- the same contract as the rank-r oracle elsewhere in this
    directory, and exact when noise = 0 because the negative direction vanishes
    there. Prefer condition_drop, which needs no approximation at all.
    """
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    d = np.asarray(d, dtype=float)
    N, r = V.shape
    rank = r if rank is None else rank

    col = sigma_column(V, d, k)
    s = float(col[k] + noise)
    mu_new = mu + (y - mu[k]) * col / s
    diag_new = np.maximum((d + np.sum(V ** 2, axis=1)) - col ** 2 / s, 0.0)

    g = np.concatenate([V[k], [d[k]]])
    C = np.zeros((r + 1, r + 1))
    C[:r, :r] = np.eye(r)
    C -= np.outer(g, g) / s
    G = np.empty((r + 1, r + 1))
    G[:r, :r] = V.T @ V
    G[:r, r] = V[k]
    G[r, :r] = V[k]
    G[r, r] = 1.0

    w, U = np.linalg.eigh(0.5 * (G + G.T))
    keep = w > 1e-12 * max(w.max(), 1e-300)
    Uk, wk = U[:, keep], w[keep]
    Gh = (Uk * np.sqrt(wk)) @ Uk.T
    Gi = (Uk / np.sqrt(wk)) @ Uk.T
    S = Gh @ C @ Gh
    lam, W = np.linalg.eigh(0.5 * (S + S.T))
    order = np.argsort(lam)[::-1][:rank]
    lam_k = np.maximum(lam[order], 0.0)
    coef = Gi @ W[:, order] * np.sqrt(lam_k)
    ek = np.zeros(N)
    ek[k] = 1.0
    V_new = V @ coef[:r] + np.outer(ek, coef[r])

    d_new = diag_new - np.sum(V_new ** 2, axis=1)
    med = float(np.median(diag_new[diag_new > 0])) if np.any(diag_new > 0) else 1.0
    return mu_new, V_new, np.maximum(d_new, 1e-10 * med)
