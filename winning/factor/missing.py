"""Factor estimation from a score matrix with missing cells (#637).

fit_factor_model_missing estimates the per-trial covariance
Lambda = V V' + diag(D) that crn_posterior / crn_posterior_replicates
take as known, from a trials x entrants score matrix in which entrants
were not scored on every trial (NaN). It is maximum likelihood by EM,
not a pairwise-complete covariance followed by a projection, because
the pairwise matrix need not be positive semidefinite, its entries rest
on different trial sets, and a projection of it optimises nothing that
the posterior uses. The pairwise counts are still reported, since they
say which covariance entries the data touch directly.
"""
from __future__ import annotations

import warnings

import numpy as np

_LOG2PI = float(np.log(2.0 * np.pi))


def _ledermann(K):
    """Largest k with no more factor parameters than covariance entries."""
    return int(np.floor((2 * K + 1 - np.sqrt(8 * K + 1)) / 2 + 1e-9))


def _estep(Yz, obs, mu, V, D, k):
    """Observed-data loglik at (mu, V, D) and the per-entrant expected
    sufficient statistics of the regression of y_ti on (1, F_t).

    Vectorised over trials: row t's posterior precision of F_t is
    M_t = I + sum_{i observed} v_i v_i' / D_i, and by the Woodbury and
    determinant lemmas its loglik term uses only M_t and
    B_t = sum_{i observed} r_ti v_i / D_i."""
    T, K = obs.shape
    Ob = obs.astype(float)
    Rw = np.where(obs, Yz - mu, 0.0) / D                # r / D, 0 if missing
    quad = (Rw * np.where(obs, Yz - mu, 0.0)).sum(axis=1)
    logdet = Ob @ np.log(D)
    if k:
        M = np.eye(k) + (Ob @ ((V[:, :, None] * V[:, None, :]).reshape(K, -1)
                              / D[:, None])).reshape(T, k, k)
        L = np.linalg.cholesky(M)
        B = Rw @ V                                      # T x k
        Fh = np.linalg.solve(M, B[:, :, None])[:, :, 0]  # E[F_t | y_t]
        Minv = np.linalg.inv(M)
        quad = quad - (B * Fh).sum(axis=1)
        logdet = logdet + 2.0 * np.log(
            np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
        E2 = (Minv + Fh[:, :, None] * Fh[:, None, :]).reshape(T, -1)
        Sf = Ob.T @ Fh
        Syf = Yz.T @ Fh                                 # Yz is 0 if missing
        Sff = (Ob.T @ E2).reshape(K, k, k)
    else:
        Sf = Syf = np.zeros((K, 0))
        Sff = np.zeros((K, 0, 0))
    ll = -0.5 * float((Ob.sum(axis=1) * _LOG2PI + logdet + quad).sum())
    return ll, Sf, Syf, Sff


def _mstep(n, Sy, Syy, Sf, Syf, Sff, floor):
    K, k = Sf.shape
    mu, V, D = np.empty(K), np.empty((K, k)), np.empty(K)
    for i in range(K):
        A = np.empty((k + 1, k + 1))
        A[0, 0], A[0, 1:], A[1:, 0], A[1:, 1:] = n[i], Sf[i], Sf[i], Sff[i]
        c = np.concatenate([[Sy[i]], Syf[i]])
        beta = np.linalg.solve(A, c)
        mu[i], V[i] = beta[0], beta[1:]
        D[i] = (Syy[i] - beta @ c) / n[i]
    return mu, V, np.maximum(D, floor)


def _canonical(V, D):
    """Rotate so V' D^-1 V is diagonal, decreasing; largest |entry| of
    each column positive. W W' and the likelihood are unchanged."""
    if V.shape[1] == 0:
        return V
    lam, Q = np.linalg.eigh((V.T / D) @ V)
    V = V @ Q[:, np.argsort(lam)[::-1]]
    i0 = np.argmax(np.abs(V), axis=0)
    return V * np.sign(V[i0, np.arange(V.shape[1])])


def _em(Yz, obs, k, mu, V, D, floor, tol, max_iter):
    n = obs.sum(axis=0).astype(float)
    Sy = Yz.sum(axis=0)
    Syy = (Yz * Yz).sum(axis=0)
    trace = []
    converged = False
    for it in range(max_iter):
        ll, Sf, Syf, Sff = _estep(Yz, obs, mu, V, D, k)
        trace.append(ll)
        if it and abs(trace[-1] - trace[-2]) <= tol * (1.0 + abs(ll)):
            converged = True
            break
        mu, V, D = _mstep(n, Sy, Syy, Sf, Syf, Sff, floor)
    if not converged:       # report the loglik of what is returned
        trace.append(_estep(Yz, obs, mu, V, D, k)[0])
    return mu, V, D, trace, converged


def _fit_k(Yz, obs, k, C, floor, tol, max_iter, n_starts, seed):
    K = Yz.shape[1]
    mu0 = Yz.sum(axis=0) / obs.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    for start in range(n_starts):
        D0 = np.maximum(0.5 * np.diag(C), floor)
        if start == 0:
            lam, U = np.linalg.eigh(C - np.diag(D0))
            top = np.argsort(lam)[::-1][:k]
            V0 = U[:, top] * np.sqrt(np.maximum(lam[top], 0.0))
            # a zero column is a fixed point of EM; seed it off zero
            dead = ~(np.abs(V0) > 0).any(axis=0)
            V0[:, dead] = 0.1 * np.sqrt(D0)[:, None] * rng.normal(
                size=(K, int(dead.sum())))
        else:
            V0 = np.sqrt(D0)[:, None] * rng.normal(size=(K, k))
        fit = _em(Yz, obs, k, mu0.copy(), V0, D0, floor, tol,
                  max_iter)
        if best is None or fit[3][-1] > best[3][-1]:
            best = fit
    return best


def fit_factor_model_missing(Y, k=None, *, k_max=None, min_pairs=1,
                             tol=1e-10, max_iter=5000, n_starts=1, seed=0):
    """Maximum-likelihood factor model from a score matrix with NaN cells.

    Y is (n_trials, n_entrants), NaN where an entrant was not scored on a
    trial. ENTRANTS ARE COLUMNS here (one row per trial, as a score sheet
    is laid out); crn_posterior_replicates takes the transpose.

    Model and assumptions. Rows are independent draws

        y_t = mu + V F_t + eps_t,   F_t ~ N(0, I_k),  eps_t ~ N(0, diag D),

    and cells are missing at random: whether a cell is missing may depend
    on the observed scores and on design (which entrants were run on which
    trial), but not on the missing score itself. Missing completely at
    random, and any missingness fixed in advance by the design, qualify.
    Under these assumptions the estimator is the maximum of the observed-
    data likelihood, computed by EM (Dempster, Laird and Rubin 1977;
    Rubin and Thayer 1982 for factor analysis): the E-step takes the
    posterior of F_t given each row's observed cells, the M-step is a
    per-entrant regression of y_ti on (1, F_t) over that entrant's
    observed trials. Each iteration cannot lower the likelihood; the
    trace is returned so this can be checked.

    The fit is an ESTIMATE. A posterior computed with the fitted (V, D)
    treated as known is exact for those values and approximate for the
    true ones; the estimation error is not propagated.

    k is the number of factors, or None to choose it by BIC over
    0..k_max (default min(5, the Ledermann bound), the largest k with no
    more parameters than covariance entries). BIC uses
    p = 2 K + K k - k (k - 1)/2 free parameters and log(n_trials).

    Refusals. min_pairs (default 1): every pair of entrants must share at
    least that many trials, else ValueError naming the pair; pass 0 to
    let the factor structure carry pairs never scored together (their
    covariance is then extrapolated through V, not observed). Each
    entrant needs k + 2 observed trials. k above the Ledermann bound is
    not identified and is refused. A rank that does not converge within
    max_iter emits a RuntimeWarning and report["converged"] is False.

    Deterministic given seed: the first start is the principal-factor
    solution of the pairwise-complete covariance (no randomness unless
    a factor column is exactly zero); further starts (n_starts > 1) are
    seeded random loadings, and the highest likelihood wins.

    Returns (V, D, report). V is (n_entrants, k), rotated so V' D^-1 V is
    diagonal and decreasing (only V V' is identified). report holds
    "mean" (the fitted mu), "loglik", "iterations", "converged", "trace"
    (the loglik at every iteration), "k", "bic" (per k tried, when k is
    None), "pairs" (n_entrants x n_entrants: trials on which each pair
    was BOTH scored, the diagonal is per-entrant counts), "min_pairs"
    (smallest off-diagonal count), "missing_fraction", "heywood"
    (entrants whose D hit the floor 1e-8 x their observed variance, a
    boundary solution) and "estimator".
    """
    Ym = np.asarray(Y, dtype=float)
    if Ym.ndim != 2:
        raise ValueError(f"Y must be (n_trials, n_entrants); got {Ym.shape}")
    if np.isinf(Ym).any():
        raise ValueError("Y has an infinite entry (NaN marks 'not scored')")
    obs = ~np.isnan(Ym)
    rows = obs.any(axis=1)
    Ym, obs = Ym[rows], obs[rows]
    T, K = Ym.shape
    if K < 2:
        raise ValueError("need at least two entrants (columns)")
    nobs = obs.sum(axis=0)
    pairs = obs.T.astype(np.int64) @ obs.astype(np.int64)
    off = np.where(np.eye(K, dtype=bool), np.iinfo(np.int64).max, pairs)
    i, j = np.unravel_index(np.argmin(off), off.shape)
    if pairs[i, j] < min_pairs:
        raise ValueError(
            f"entrants {i} and {j} share {int(pairs[i, j])} scored trials, "
            f"fewer than min_pairs={min_pairs}; their covariance would rest "
            "on the factor model alone. Pass min_pairs=0 to accept that")
    lb = _ledermann(K)
    if k is None:
        ks = list(range(0, min(lb, 5 if k_max is None else int(k_max)) + 1))
    else:
        if int(k) != k or k < 0:
            raise ValueError(f"k must be a non-negative integer; got {k!r}")
        if k > lb:
            raise ValueError(
                f"k={k} factors for {K} entrants exceeds the Ledermann "
                f"bound {lb}: the model is not identified")
        ks = [int(k)]
    need = max(ks) + 2
    if (nobs < need).any():
        e = int(np.flatnonzero(nobs < need)[0])
        raise ValueError(
            f"entrant {e} is scored on {int(nobs[e])} trials; a {max(ks)}-"
            f"factor fit needs at least {need} per entrant")
    Yz = np.where(obs, Ym, 0.0)
    mu0 = Yz.sum(axis=0) / nobs
    R = np.where(obs, Ym - mu0, 0.0)
    var = (R * R).sum(axis=0) / nobs
    if not (var > 0).all():
        e = int(np.flatnonzero(~(var > 0))[0])
        raise ValueError(f"entrant {e} has zero observed variance")
    # pairwise-complete covariance (start values only)
    C = (R.T @ R) / np.maximum(pairs, 1)
    C[pairs < 2] = 0.0
    np.fill_diagonal(C, var)
    floor = 1e-8 * var
    fits, bic = {}, {}
    for kk in ks:
        fits[kk] = _fit_k(Yz, obs, kk, C, floor, tol, max_iter,
                          max(int(n_starts), 1), seed)
        p = 2 * K + K * kk - kk * (kk - 1) / 2
        bic[kk] = -2.0 * fits[kk][3][-1] + p * np.log(T)
    kk = min(bic, key=bic.get)
    mu, V, D, trace, converged = fits[kk]
    slow = [j for j in ks if not fits[j][4]]
    if slow:
        warnings.warn(
            f"fit_factor_model_missing: EM did not converge in {max_iter} "
            f"iterations at k={slow} (the loglik there is a lower bound"
            + (", so BIC may under-select those ranks" if k is None else "")
            + "); raise max_iter or loosen tol",
            RuntimeWarning, stacklevel=2)
    V = _canonical(V, D)
    report = {
        "k": kk, "mean": mu, "loglik": float(trace[-1]),
        "iterations": len(trace), "converged": converged,
        "trace": np.asarray(trace), "pairs": pairs,
        "min_pairs": int(off.min()) if K > 1 else 0,
        "missing_fraction": float(1.0 - obs.mean()),
        "heywood": np.flatnonzero(D <= floor * (1 + 1e-12)),
        "estimator": ("EM maximum likelihood, rows iid Gaussian, cells "
                      "missing at random"),
    }
    if k is None:
        report["bic"] = bic
        report["converged_by_k"] = {j: fits[j][4] for j in ks}
    return V, D, report
