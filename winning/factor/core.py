"""Factor-correlated race transforms (ported from the kinetics research
repository, experiments/raceutil.py, at tag paper-r10; log-domain lattice
kernel, L=501 production default, all identities numerically verified
there). Min-wins convention internally; see winning.methods for the
max-wins arena interface."""


from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr

_TINY = 1e-300
_PFLOOR = 1e-15


def _lattice(mu: np.ndarray, sigma: float, points: int = 3001) -> np.ndarray:
    """Lattice adapted to the field: covers every competitor's density."""
    lo = float(np.min(mu)) - 8.0 * sigma
    hi = float(np.max(mu)) + 8.0 * sigma
    return np.linspace(lo, hi, points)


def win_probabilities(mu: np.ndarray, sigma: float = 1.0,
                      x: np.ndarray | None = None) -> np.ndarray:
    """P(X_i = min_j X_j) for X_i = mu_i + sigma*eps, eps ~ N(0,1) iid."""
    mu = np.asarray(mu, dtype=float)
    if x is None:
        x = _lattice(mu, sigma)
    z = (x[None, :] - mu[:, None]) / sigma
    S = 1.0 - ndtr(z)
    f = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    dx = x[1] - x[0]
    log_S_field = np.sum(np.log(np.maximum(S, _TINY)), axis=0)
    log_rest = log_S_field[None, :] - np.log(np.maximum(S, _TINY))
    rest = np.exp(np.clip(log_rest, -745.0, 0.0))
    p = np.sum(f * rest, axis=1) * dx
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("race integration failed; widen the lattice")
    return p / total  # remove lattice quadrature error


def abilities_from_probabilities(p: np.ndarray, sigma: float = 1.0,
                                 n_iter: int = 500, step: float = 0.5,
                                 tol: float = 1e-9) -> np.ndarray:
    """Invert the race: find mu (mean zero) with win_probabilities(mu) = p.

    Damped fixed point: win probability is decreasing in mu (argmin race), so raise
    the ability of overpriced competitors and lower it for underpriced ones. Residuals
    are clipped so a vanishing model probability cannot destabilize the iteration.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    mu = -sigma * (logp - logp.mean()) / 2.0  # conservative warm start
    for _ in range(n_iter):
        model = np.maximum(win_probabilities(mu, sigma), _PFLOOR)
        resid = np.clip(np.log(model) - logp, -4.0, 4.0)
        mu = mu + step * sigma * resid
        mu -= mu.mean()
        if np.abs(resid).max() < tol:
            break
    return mu


# ---------------------------------------------------------------------------
# Fast transform for CORRELATED fields (program Q6).
#
# Decompose the covariance as Sigma ~= V V^T + D (k factors + idiosyncratic
# diagonal; eigen-truncation leaves a nonnegative diagonal residual and matches
# the diagonal exactly). Conditionally on the k factors the competitors are
# independent, so the multiplicative cavity applies at every quadrature node:
#
#   p_i = E_f [ integral f_i(x|f) * S_field(x|f) / S_i(x|f) dx ],
#
# a k-dimensional Gauss-Hermite quadrature wrapped around the O(N) independent
# transform. The two leave-one-out identities compose: the Gaussian/Schur side
# compresses the coupling into factors; the field product prices the race.
# ---------------------------------------------------------------------------


def factor_model(C: np.ndarray, k: int, n_iter: int = 200,
                 tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Fit C ~= V V^T + diag(D) by iterated principal-factor analysis.

    Unlike naive eigen-truncation (which invents off-diagonal correlation --
    catastrophically so for C near identity), the iteration fits the
    off-diagonals: eigendecompose C - diag(D), re-estimate D from the exact
    diagonal, repeat. V has k columns; D is the idiosyncratic variance.
    """
    C = np.asarray(C, dtype=float)
    D = np.full(len(C), 0.5 * float(np.mean(np.diag(C))))
    V = np.zeros((len(C), k))
    for _ in range(n_iter):
        lam, U = np.linalg.eigh(C - np.diag(D))
        idx = np.argsort(lam)[::-1][:k]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new = np.clip(np.diag(C) - np.sum(V**2, axis=1), 1e-3, None)
        if np.abs(D_new - D).max() < tol:
            D = D_new
            break
        D = D_new
    return V, D


def factor_model_contrast(C: np.ndarray, k: int, n_iter: int = 200,
                          tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Factor fit in the choice-relevant quotient space (third-review fix).

    Choices depend on (mu, Sigma) only through P mu and P Sigma P with
    P = I - 11^T/N: a common factor shock V -> V + 1 c^T shifts every
    conditional location equally and cannot move the argmax (machine-precision
    fact, tested). Fitting raw Sigma can therefore spend scarce factor rank on
    a choice-irrelevant common component (Sigma = tau^2 11^T + b b^T + D with
    large tau: the raw rank-1 fit takes 11^T and misses b b^T entirely).

    The fit is therefore run on the projected matrix C_P = P C P itself ---
    the quantity choice probabilities actually depend on --- and the returned
    (V, D) reproduce C_P, not C. NOTE the asymmetry, caught by a wrong first
    version of this function: only the common FACTOR direction is irrelevant;
    a common addition to the idiosyncratic variances D is choice-relevant
    (it inflates every difference variance), so D must come from the quotient
    fit, not be refit against diag(C). Loadings are centered (P V) and
    canonicalized by SVD with a sign convention, making results reproducible
    at the covariance level rather than the supplied-V level.
    """
    C = np.asarray(C, dtype=float)
    n = len(C)
    P = np.eye(n) - np.ones((n, n)) / n
    V, D = factor_model(P @ C @ P, k, n_iter=n_iter, tol=tol)
    V = P @ V
    A, sv, _ = np.linalg.svd(V, full_matrices=False)
    V = A[:, :k] * sv[:k]
    for j in range(V.shape[1]):
        i0 = np.argmax(np.abs(V[:, j]))
        if V[i0, j] < 0:
            V[:, j] = -V[:, j]
    return V, D


def factor_model_projected(C: np.ndarray, k: int, n_outer: int = 60):
    """Certified quotient-space factor fit (eighth-review construction).

    The contrast heuristic applies principal-factor logic to P C P, but
    P diag(D) P is not diagonal, so that heuristic does not literally
    minimize the projected norm. This routine fits the reduced matrix
    S = B' C B (B an orthonormal basis of the mean-zero subspace) by
    W W' + sum_i D_i b_i b_i', alternating a top-k PSD approximation for W
    with nonnegative least squares for D. On the boundary-experiment
    matrices it changes the quotient residual by under 1% relative to the
    heuristic, so the heuristic is not the binding constraint; this exists
    to certify that.
    """
    from scipy.optimize import nnls

    C = np.asarray(C, dtype=float)
    n = len(C)
    B = np.linalg.qr(np.eye(n) - np.ones((n, n)) / n)[0][:, :n - 1]
    S = B.T @ C @ B
    D = np.full(n, 0.5 * float(np.mean(np.diag(C))))
    M = np.column_stack([np.outer(B[i], B[i]).ravel() for i in range(n)])
    W = np.zeros((n - 1, k))
    for _ in range(n_outer):
        R = S - (B.T * D) @ B
        lam, U = np.linalg.eigh(R)
        idx = np.argsort(lam)[::-1][:k]
        W = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new, _ = nnls(M, (S - W @ W.T).ravel())
        if np.abs(D_new - D).max() < 1e-12:
            D = D_new
            break
        D = D_new
    return B @ W, np.maximum(D, 1e-8)


def hermite_nodes(k: int, Q: int = 15, prune: float = 1e-7):
    """Product Gauss-Hermite rule for E over N(0, I_k); returns (nodes, weights)."""
    x, w = np.polynomial.hermite_e.hermegauss(Q)
    w = w / np.sqrt(2.0 * np.pi)
    if k == 1:
        return x[:, None], w
    grids = np.meshgrid(*([x] * k), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for d in range(k):
        W *= w[np.searchsorted(x, F[:, d])]
    keep = W > prune * W.max()
    return F[keep], W[keep]


def win_probabilities_factor(mu: np.ndarray, V: np.ndarray, D: np.ndarray,
                             F: np.ndarray, W: np.ndarray,
                             keep: np.ndarray | None = None,
                             points: int = 501,
                             return_deletions: bool = False,
                             per_node_interval: bool = False,
                             return_total: bool = False):
    """Win probabilities for X = mu + V f + sqrt(D) eps, argmin wins.

    With return_deletions=True also returns the FULL single-deletion ensemble
    q[i, j] = P(j wins | i removed) from the same conditional field pass --
    the multiplicative cavity, conditionally: divide S_field by S_i (and S_j).

    per_node_interval=True gives each factor node its own lattice
    [min_i m_i(f_q) - 8 sd_max, max_i m_i(f_q) + 8 sd_max] instead of one
    global interval over all retained nodes. Same O(QNL) cost; keeps spatial
    resolution from degrading as Q grows (with a global interval, adding
    factor nodes widens the lattice at fixed L, entangling Q- and L-errors).
    return_total=True also returns the pre-normalization quadrature total
    (|1 - total| is a resolution diagnostic).
    """
    mu = np.asarray(mu, dtype=float)
    if keep is not None:
        mu, V, D = mu[keep], V[keep], D[keep]
    N = len(mu)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T                      # (nodes, N) cond. means
    pad = 8.0 * sd.max()
    grid = np.arange(points) / (points - 1)            # common normalized coord

    p = np.zeros(N)
    q = np.zeros((N, N)) if return_deletions else None
    chunk = max(1, int(5e6 / (N * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]                          # (nc, N)
        Wc = W[a:a + chunk]
        if per_node_interval:
            lo_c = M.min(axis=1) - pad                  # (nc,)
            hi_c = M.max(axis=1) + pad
        else:
            lo_c = np.full(M.shape[0], M_all.min() - pad)
            hi_c = np.full(M.shape[0], M_all.max() + pad)
        x = lo_c[:, None] + (hi_c - lo_c)[:, None] * grid[None, :]   # (nc, L)
        dx = (hi_c - lo_c) / (points - 1)               # (nc,)
        z = (x[:, None, :] - M[:, :, None]) / sd[None, :, None]
        f = np.exp(-0.5 * z**2) / (sd[None, :, None] * np.sqrt(2.0 * np.pi))
        logS = log_ndtr(-z)     # exact log-survival; 1-ndtr underflows at z~8.3
        logSfield = logS.sum(axis=1)                    # (nc, L)
        rest = np.exp(np.clip(logSfield[:, None, :] - logS, -745.0, 0.0))
        p += (Wc * dx) @ np.sum(f * rest, axis=2)       # (nc, N) -> (N,)
        if return_deletions:
            for i in range(N):
                # divide the deleted competitor's survival back out
                rest_i = np.exp(np.clip(
                    logSfield[:, None, :] - logS - logS[:, i:i + 1, :],
                    -745.0, 0.0))
                contrib = np.sum(f * rest_i, axis=2)
                contrib[:, i] = 0.0
                q[i] += (Wc * dx) @ contrib
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise FloatingPointError("factor race integration failed")
    out = p / total
    if return_deletions:
        q = q / q.sum(axis=1, keepdims=True)
        return (out, q, total) if return_total else (out, q)
    return (out, total) if return_total else out


def abilities_from_probabilities_factor(p: np.ndarray, V: np.ndarray,
                                        D: np.ndarray, F: np.ndarray,
                                        W: np.ndarray, n_iter: int = 50,
                                        tol: float = 1e-6,
                                        return_info: bool = False,
                                        points: int = 501):
    """Inverse transform under the factor model, by coordinate-wise Newton.

    Design synthesis (credit where due): the coordinate-Newton-against-a-frozen-
    field structure is the ORIGINAL fast-ability-transform inversion (winning /
    thurstone); the independent-inverse warm start and the observation that the
    choice-space Jacobian is intrinsically well-conditioned are from the
    allocation package (allocation/_thurstone/calibrate.py and
    experiments/preconditioner.py). This version adds: k-factor quadrature,
    analytic per-coordinate slopes dp_i/dmu_i = sum_w W_w int (z f(z)/sd_i)
    rest_i dx computed in the same chunked lattice pass as p_hat, and a
    tail-aware tolerance (convergence is not held hostage by runners whose
    target probability is unidentifiably small). Typical cost: ~10 forward-pass
    equivalents, versus hundreds for the damped Picard iteration it replaces.
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    p = p / p.sum()
    logp = np.log(p)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    D = np.asarray(D, dtype=float)
    sd = np.sqrt(D)
    N = len(p)
    # tail-aware convergence: runners below the floor are matched best-effort
    floor = max(1e-9, 1e-4 / N)
    ident = p > floor
    # warm start: exact INDEPENDENT inversion (allocation's design), using each
    # runner's total sd, via this same Newton with a single zero factor node
    if F.shape[1] >= 1 and np.any(V != 0.0):
        sd_tot = np.sqrt(D + np.sum(V**2, axis=1))
        mu = abilities_from_probabilities_factor(
            p, np.zeros((N, 1)), sd_tot**2, np.zeros((1, 1)), np.ones(1),
            n_iter=n_iter, tol=tol)
    else:
        mu = (logp - logp.mean()) / 2.0
    step_cap = 1.0 * np.sqrt(D + np.sum(V**2, axis=1))
    prev_res = np.inf
    damp = 1.0
    for _ in range(n_iter):
        M_all = mu[None, :] + F @ V.T
        lo = M_all.min() - 8.0 * sd.max()
        hi = M_all.max() + 8.0 * sd.max()
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
        phat = np.zeros(N)
        slope = np.zeros(N)
        chunk = max(1, int(5e6 / (N * len(x))))
        for a in range(0, len(F), chunk):
            M = M_all[a:a + chunk]
            Wc = W[a:a + chunk]
            z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
            f = np.exp(-0.5 * z**2) / (sd[None, :, None] * np.sqrt(2.0 * np.pi))
            logS = log_ndtr(-z)
            logSfield = logS.sum(axis=1)
            rest = np.exp(np.clip(logSfield[:, None, :] - logS, -745.0, 0.0))
            phat += Wc @ (np.sum(f * rest, axis=2) * dx)
            slope += Wc @ (np.sum(z * f / sd[None, :, None] * rest, axis=2) * dx)
        phat = np.maximum(phat / phat.sum(), _PFLOOR)
        resid = np.log(phat) - logp
        res = np.abs(resid[ident]).max() if np.any(ident) else np.abs(resid).max()
        if res < tol:
            break
        if res > prev_res * 1.2:
            damp = max(0.25, damp * 0.5)     # simple safeguard
        prev_res = res
        dlogp = slope / phat                  # negative for min-wins
        dlogp = np.minimum(dlogp, -1e-3 / (sd + 1e-9))
        delta = np.clip(damp * resid / dlogp, -step_cap, step_cap)
        mu = mu - delta                      # Newton: mu <- mu - resid / dlogp
        mu -= mu.mean()
    if return_info:
        return mu, {"iterations": _ + 1, "residual": float(res),
                    "converged": bool(res < tol)}
    return mu


def qmc_nodes(k: int, m: int = 13, seed: int = 0):
    """Scrambled-Sobol nodes for E over N(0, I_k): 2^m equal-weight points.

    Deterministic given the seed; error decays ~n^-1 on smooth integrands vs
    n^-1/2 for plain Monte Carlo. Use for factor ranks beyond the reach of
    product Gauss-Hermite (k >~ 4).
    """
    from scipy.stats import norm, qmc

    F = norm.ppf(qmc.Sobol(k, scramble=True, seed=seed).random_base2(m))
    return F, np.full(len(F), 1.0 / len(F))


def jacobian_vector_product(mu, V, D, F, W, h, points=3001, form="ibp"):
    """(J h)_i for J = d p / d mu of the min-wins factor race, in O(Q N L).

    Uses the integration-by-parts form (referee of the factor-probit paper):
        (J h)_i = E_f int g_i R_i (A - h_i Lambda) dx,
    with hazards lam_j = g_j / S_j, Lambda = sum_j lam_j, A = sum_j h_j lam_j,
    R_i = prod_{j != i} S_j. Everything is computed in the log domain
    (log g analytic, log-hazard = log g - log_ndtr(-z)), so no density is
    floored and no hazard overflows; the earlier density-derivative form
    produced inf/NaN when g underflowed while log S stayed finite.

    Conventions and scope: this is the JVP of the exact min-wins identity,
    for which sum_i p_i = 1 and J^T 1 = J 1 = 0; for the max-wins (argmax
    utility) map with mu = -a the sign flips, (J_max h) = -(J_min h). The
    implemented forward map normalizes its quadrature output; the
    normalization correction (v - p (1^T v))/s is O(quadrature defect)
    because 1^T v = 0 analytically, and this formula matches central finite
    differences of the returned normalized map to ~5e-9 in tests.

    form="ibp" (default) is the integration-by-parts / weighted-Laplacian
    form: the exact derivative of the CONTINUUM map, and a symmetric
    surrogate for the lattice map (gap 1.7e-14 at L=1501, but 2e-5 at L=51).
    form="grid" is the exact derivative of the frozen-grid rectangle sum
    itself (own-term (x - m_i)/D_i instead of the IBP form): it matches
    finite differences of the frozen-grid map to ~1e-10 at EVERY lattice
    resolution, at identical O(QNL) cost. Use "grid" when exact
    differentiation of the returned numerical map matters; "ibp" when the
    symmetric SPD structure matters.
    """
    mu = np.asarray(mu, dtype=float)
    h = np.asarray(h, dtype=float)
    sd = np.sqrt(np.asarray(D, dtype=float))
    N = len(mu)
    M_all = mu[None, :] + F @ V.T
    x = np.linspace(M_all.min() - 8 * sd.max(), M_all.max() + 8 * sd.max(), points)
    dx = x[1] - x[0]
    out = np.zeros(N)
    log_norm = np.log(sd * np.sqrt(2 * np.pi))
    chunk = max(1, int(4e6 / (N * points)))
    for a0 in range(0, len(F), chunk):
        M = M_all[a0:a0 + chunk]
        Wc = W[a0:a0 + chunk]
        for c in range(M.shape[0]):
            z = (x[None, :] - M[c][:, None]) / sd[:, None]
            logS = log_ndtr(-z)
            logg = -0.5 * z**2 - log_norm[:, None]
            haz = np.exp(logg - logS)   # Mills ratio: grows only linearly
            A = (h[:, None] * haz).sum(0)
            gR = np.exp(np.clip(logg + logS.sum(0)[None, :] - logS, -745.0, 700.0))
            if form == "grid":
                own = h[:, None] * (x[None, :] - M[c][:, None]) / np.asarray(D)[:, None]
                integ = gR * (own + A[None, :] - h[:, None] * haz)
            else:
                integ = gR * (A[None, :] - h[:, None] * haz.sum(0)[None, :])
            out += Wc[c] * (integ.sum(1) * dx)
    return out



# package-style name; the research name remains as an alias
abilities_from_win_probabilities = abilities_from_probabilities_factor
