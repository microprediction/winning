"""Full-covariance belief updates: the ADF diagonal loss, repaired.

Measured motivation (bandits lane): composing a market observation with
an outcome observation, each leg individually near-exact, left the
composed posterior off by 0.14 sd -- the market's contrast observation
induces cross-correlations that a diagonal belief projection discards
before the outcome update consumes the prior.

THE CONSTRUCTION, and the standing rule behind it: whatever part of the
belief is DIAGONAL in observation coordinates belongs on the lattice as
idiosyncratic variance, where it is integrated analytically and
exactly; quadrature nodes are spent only on genuinely shared structure.
An earlier version put the whole belief covariance into the loading
matrix by Cholesky, so the belief rode quadrature -- and a corner-space
audit measured all four consequences: marginals ~5x worse than the
diagonal members even at prior ratio 1, posterior variance inflated 7x
under a diffuse prior, a hard overflow crash on the order path past
ratio ~20, and relabelling non-invariance (Cholesky is not
permutation-equivariant). _belief_split repairs all four at once:
S = B B' + diag(psi), psi joins beta2 on the lattice, only B costs
nodes. A diagonal belief then costs ZERO nodes and reproduces the
diagonal member exactly; the split is eigendecomposition-based, hence
permutation-equivariant.

The Gaussian-prior moment identities give the FULL posterior:

    E[s | E]   = m + Sigma grad_m log P(E)
    Cov[s | E] = Sigma + Sigma hess_m log P(E) Sigma

with the gradient the posterior-node-weighted conditional gradient and
the Hessian by central differences of the mixture gradient at steps
scaled to each coordinate's own belief sd (a fixed absolute step is
pure cancellation noise once the prior is diffuse, and |Sigma|^2
amplifies it). Sigma-in, Sigma-out, logZ returned. Diagonal wrappers in
nway.py remain the cheap default.
"""

from __future__ import annotations

import numpy as np

from .nway import _grad_logp_row, _order_pass


def _psd_repair(S, floor_frac=1e-8):
    S = np.asarray(S, dtype=float)
    if not np.isfinite(S).all():
        # diagnosable failure instead of a LAPACK message (bandits
        # report: the order path overflowed into NaN before the repair
        # saw it, surfacing as "Eigenvalues did not converge")
        raise ValueError(
            "non-finite entries in a belief covariance; the update that "
            "produced it overflowed (typically a near-impossible "
            "observation under a very diffuse prior)")
    S = 0.5 * (S + S.T)
    lam, U = np.linalg.eigh(S)
    floor = floor_frac * max(float(np.trace(S)) / len(S), 1e-12)
    return (U * np.maximum(lam, floor)) @ U.T


def _belief_split(S, tol=1e-8, max_rank=8):
    """Represent S = B B' + diag(psi) with psi >= 0.

    This is the load-bearing construction of the full-covariance
    updates. Putting ALL of S into the loading matrix (an earlier
    Cholesky version did) makes the belief ride QUADRATURE, where the
    diagonal members ride the LATTICE analytically -- measurably worse
    marginals at every prior scale, and catastrophic under a diffuse
    prior. Splitting instead sends the diagonal part to the lattice as
    idiosyncratic variance and quadratures only what is genuinely
    correlated: a diagonal belief costs ZERO quadrature dimensions and
    reproduces the diagonal member exactly, and a rank-k-plus-diagonal
    belief (what a filter state usually is) costs k.

    Eigendecomposition-based, hence permutation-equivariant: relabelling
    the field permutes the answer instead of perturbing it, which
    Cholesky could not promise.
    """
    S = np.asarray(S, dtype=float)
    n = len(S)
    dg = np.diag(S).copy()
    scale = max(float(dg.mean()), 1e-300)
    off = S - np.diag(dg)
    if np.abs(off).max() <= tol * scale:
        return np.zeros((n, 0)), np.maximum(dg, 0.0)
    from ..factor.core import factor_model
    for k in range(1, min(max_rank, n - 1) + 1):
        B, psi = factor_model(S, k)
        R = S - B @ B.T - np.diag(psi)
        np.fill_diagonal(R, 0.0)
        if float(np.abs(R).max()) <= tol * scale:
            return np.asarray(B, dtype=float), np.maximum(psi, 0.0)
    # no low-rank split found: fall back to the EXACT symmetric square
    # root (still permutation-equivariant) rather than ship model error
    lam, U = np.linalg.eigh(S)
    lam = np.maximum(lam, 0.0)
    keep = lam > tol * max(float(lam.max()), 1e-300)
    return (U[:, keep] * np.sqrt(lam[keep])), np.zeros(n)


def _mixture_nodes(rank, nodes_log2, Qf=9):
    """Nodes over the augmented factor space: the degenerate one-node
    rule at rank 0 (belief diagonal, no shared factors), Gauss-Hermite
    tensor at rank <= 2 (matching the diagonal members), scrambled
    Sobol beyond."""
    if rank == 0:
        return np.zeros((1, 1)), np.ones(1)
    if rank <= 2:
        from ..factor.core import hermite_nodes
        return hermite_nodes(rank, Q=Qf)
    from ..factor.core import qmc_nodes
    return qmc_nodes(rank, m=nodes_log2)


def _mixture_update_full(m, S, V, beta2, node_logp_grad, nodes_log2=10,
                         eps=None, A=None, eps_rel=0.15, kernel=None):
    """Core of the full-covariance updates.

    A (optional, k x n) maps the belief space to the OBSERVATION space
    -- identity for individual races, the team assignment/weight matrix
    for team races. The observation-space belief covariance A S A' is
    SPLIT (see _belief_split): its diagonal part joins beta2 on the
    lattice, only the correlated part becomes loadings. Moments:
    E[s|E] = m + S A' g, Cov[s|E] = S + S A' H A S, with H by central
    differences of the mixture gradient taken at steps scaled to each
    coordinate's own belief sd (a fixed absolute step is pure
    cancellation noise once the prior is diffuse, and |S|^2 amplifies
    it -- the failure the bandits corner-space audit measured).

    kernel(mu_obs_batch, D) -> (logp (Q,), grad (Q, k)) evaluates every
    factor node at once when supplied; node_logp_grad is the per-node
    fallback.
    """
    m = np.asarray(m, dtype=float)
    S = _psd_repair(np.asarray(S, dtype=float))
    n = len(m)
    if A is None:
        A = np.eye(n)
    else:
        A = np.atleast_2d(np.asarray(A, dtype=float))
    k = A.shape[0]
    Sobs = A @ S @ A.T
    B, psi = _belief_split(Sobs)
    D = psi + np.broadcast_to(np.asarray(beta2, dtype=float),
                              (k,)).astype(float)
    if V is None:
        Vaug = B
    else:
        Vv = np.atleast_2d(np.asarray(V, dtype=float))
        if Vv.shape[0] != k:
            Vv = Vv.T
        Vaug = np.hstack([B, Vv])
    rank = Vaug.shape[1]
    F, W = _mixture_nodes(rank, nodes_log2)
    if rank == 0:
        Vaug = np.zeros((k, 1))
    logW = np.log(np.maximum(W, 1e-300))
    shifts = F @ Vaug.T
    mu_obs = A @ m

    def mixture(mo):
        pts = mo[None, :] + shifts
        if kernel is not None:
            logps, grads = kernel(pts, D, Vaug, F, W)
            if logps is None:            # kernel handled the mixture
                return grads
        else:
            logps = np.empty(len(F))
            grads = np.empty((len(F), k))
            for q in range(len(F)):
                lp, g = node_logp_grad(pts[q], D)
                logps[q] = lp
                grads[q] = g
        a = logW + logps
        astar = a.max()
        if not np.isfinite(astar):
            return np.zeros(k), -np.inf
        pw = np.exp(a - astar)
        logZ = astar + np.log(pw.sum())
        omega = pw / pw.sum()
        return omega @ grads, logZ

    G, logZ = mixture(mu_obs)
    m_new = m + S @ (A.T @ G)
    # coordinate steps in the belief's own units
    sd_obs = np.sqrt(np.maximum(np.diag(Sobs), 0.0))
    steps = eps_rel * np.maximum(sd_obs, np.sqrt(np.maximum(D, 1e-12)))
    if eps is not None:
        steps = np.full(k, float(eps))
    H = np.empty((k, k))
    for j in range(k):
        ej = np.zeros(k)
        ej[j] = steps[j]
        gp, _ = mixture(mu_obs + ej)
        gm, _ = mixture(mu_obs - ej)
        H[j] = (gp - gm) / (2 * steps[j])
    H = 0.5 * (H + H.T)
    SAt = S @ A.T
    S_new = _psd_repair(S + SAt @ H @ SAt.T)
    return m_new, S_new, float(logZ)


def _winner_kernel(winner, k, points=801):
    """Vectorized winner kernel: log P and the gradient row for every
    factor node in ONE shared-field pass."""
    from ..factor.core import (jacobian_vector_product,
                               win_probabilities_factor)

    e = np.zeros(k)
    e[int(winner)] = 1.0

    def kernel(pts, D, Vaug, F, W):
        mo = pts[0] - (F @ Vaug.T)[0]
        a = -mo
        p = win_probabilities_factor(a, Vaug, D, F, W)
        Ji = jacobian_vector_product(a, Vaug, D, F, W, e, form="grid",
                                     points=points)
        pi = max(float(p[int(winner)]), 1e-300)
        return None, (-Ji / pi, float(np.log(pi)))

    return kernel


def _order_kernel(order, base="normal"):
    """Vectorized ordered-statistics kernel across factor nodes."""
    from .nway import _order_pass_batch

    order = np.asarray(order, dtype=int)

    def kernel(pts, D, Vaug, F, W):
        return _order_pass_batch(pts, np.sqrt(np.maximum(D, 1e-300)), order,
                                 base=base)

    return kernel


def update_winner_full(m, S, winner, V=None, beta2=1.0, nodes_log2=10,
                       eps=None, points=801, eps_rel=0.15):
    """Winner observation against a full-covariance belief N(m, S)
    (max-wins). V: optional shared performance factors on top of the
    belief correlation; beta2: idiosyncratic performance noise (scalar
    or per-participant). Returns (m_post, S_post, logZ).

    The belief SPLITS into a lattice-borne diagonal part plus
    quadratured correlation (_belief_split), so a diagonal belief costs
    no quadrature at all and reproduces update_winner_correlated
    exactly; whatever factors remain are integrated in ONE vectorized
    shared-field pass.
    """
    kernel = _winner_kernel(winner, len(np.asarray(m, dtype=float)),
                            points=points)
    return _mixture_update_full(m, S, V, beta2, None, nodes_log2=nodes_log2,
                                eps=eps, eps_rel=eps_rel, kernel=kernel)


def update_order_full(m, S, order, V=None, beta2=1.0, nodes_log2=10,
                      eps=None, eps_rel=0.15, base="normal"):
    """Full-order observation against a full-covariance belief
    (max-wins, order best-first). Returns (m_post, S_post, logZ);
    near-impossible orders degrade like order_loglik.

    Belief split as in update_winner_full (a diagonal belief costs no
    quadrature and reproduces update_order_correlated), and the
    surviving mixture rides _order_pass_batch -- the ordered-statistics
    sweeps vectorized across factor nodes, because online deployments
    are order-heavy.
    """
    kernel = _order_kernel(order, base=base)
    return _mixture_update_full(m, S, V, beta2, None, nodes_log2=nodes_log2,
                                eps=eps, eps_rel=eps_rel, kernel=kernel)


def update_market_full(m, S, p_market, tau2=0.25, invert=None,
                       **market_model):
    """Market prices against a full-covariance belief: the conjugate
    case, exact and closed form. Observation y = P s + eta on the
    contrast space (max-wins; the default invert negates the racing
    engine's min-wins abilities). Returns (m_post, S_post, logZ)."""
    m = np.asarray(m, dtype=float)
    S = np.asarray(S, dtype=float)
    n = len(m)
    if invert is None:
        from ..factor.races import abilities_from_race

        def invert(p):
            return -abilities_from_race(p, **market_model)
    y = np.asarray(invert(np.asarray(p_market, dtype=float)), dtype=float)
    y = y - y.mean()
    tau2 = np.broadcast_to(np.asarray(tau2, dtype=float), (n,)).astype(float)
    P = np.eye(n) - np.ones((n, n)) / n
    Sinv = np.linalg.inv(_psd_repair(S))
    A = Sinv + P @ np.diag(1.0 / tau2) @ P
    S_new = np.linalg.inv(A)
    m_new = S_new @ (Sinv @ m + P @ (y / tau2))
    M = P @ (S + np.diag(tau2)) @ P
    lam, U = np.linalg.eigh(M)
    keep = lam > 1e-12
    r = y - P @ m
    z = U[:, keep].T @ r
    logZ = float(-0.5 * (np.sum(z * z / lam[keep])
                         + np.sum(np.log(lam[keep]))
                         + keep.sum() * np.log(2.0 * np.pi)))
    return m_new, _psd_repair(S_new), logZ
