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
Putting the whole belief covariance into the loading matrix by
Cholesky makes the belief ride quadrature, and a corner-space audit
measures four consequences: marginals ~5x worse than the
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

from ..shapes import as_loadings


from .nway import _recentre_nodes


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
    S = 0.5 * S + 0.5 * S.T        # halved first: see #279
    lam, U = np.linalg.eigh(S)
    # Clip only NEGATIVE eigenvalues (roundoff), and leave a PSD matrix
    # alone. This raised every eigenvalue to floor_frac * trace / n, so a
    # perfectly common component (S = eps I + 11') turned a contrast
    # variance of 2e-12 into ~2e-8 and the update priced a different
    # race (#650). Small and zero eigenvalues are model, not noise;
    # callers that need an inverse use forms that do not (update_market_full).
    if lam.min() >= 0.0:
        return S
    return (U * np.maximum(lam, 0.0)) @ U.T


def _check_full_inputs(m, S, V=None, A=None, beta2=1.0):
    """The shape contract of every full-covariance update, checked up
    front: m a finite (n,) vector, S an (n, n) matrix, A (k, n) when
    given, V loadings for the k observed rows and beta2 broadcastable to
    k. A singleton order returns before any of these is touched, so it
    certified a length-2 "covariance" for a length-3 mean, or a team map
    with the wrong number of players, as an exact posterior (#456)."""
    m = np.asarray(m, dtype=float)
    if m.ndim != 1:
        raise ValueError(f"m must be a vector of belief means; got shape {m.shape}")
    n = m.shape[0]
    S = np.asarray(S, dtype=float)
    if S.shape != (n, n):
        raise ValueError(
            f"S must be the ({n}, {n}) belief covariance of the {n} means; "
            f"got shape {S.shape}")
    k = n
    if A is not None:
        A = np.atleast_2d(np.asarray(A, dtype=float))
        if A.ndim != 2 or A.shape[1] != n:
            raise ValueError(
                f"A must map the {n} players to the observed rows, shape "
                f"(k, {n}); got shape {A.shape}")
        k = A.shape[0]
    if V is not None:
        as_loadings(V, k)
    try:
        np.broadcast_to(np.asarray(beta2, dtype=float), (k,))
    except ValueError:
        raise ValueError(
            f"beta2 must be a scalar or one variance per observed row "
            f"({k}); got shape {np.shape(beta2)}") from None
    return k


def _drop_common_mode(Sobs):
    """Remove a perfectly common component c 11' from the observation-
    space belief, which no winner or order can see (it shifts every
    performance equally), before it is split into loadings.

    Left in, it dominated the split: with S = eps I + 11' the contrast
    variance 2 eps was a 1e-12 remainder of an order-one factor, the
    split's tolerance swallowed it, and the update priced a different
    race (P 0.76 against the exact 0.69 at eps = 1e-12; posterior
    contrast variance 40% off already at eps = 1e-6, #650). c is the
    smallest off-diagonal entry, clipped to [0, c*] where c* is the
    largest c that leaves the belief PSD, so a diagonal or
    contrast-dominated belief (the common filter state) is unchanged."""
    k = Sobs.shape[0]
    if k < 2:
        return Sobs
    off = Sobs[~np.eye(k, dtype=bool)]
    c = float(off.min())
    if not c > 0.0:
        return Sobs
    one = np.ones(k)
    q = float(one @ np.linalg.pinv(Sobs, hermitian=True) @ one)
    if q > 0.0:
        c = min(c, 1.0 / q)
    return Sobs - c * np.outer(one, one)


def _belief_split(S, tol=1e-8, max_rank=8):
    """Represent S = B B' + diag(psi) with psi >= 0.

    The full-covariance updates depend on this construction. Putting
    all of S into the loading matrix (an earlier Cholesky version did)
    makes the belief ride quadrature, where the
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
    Sobs = _drop_common_mode(A @ S @ A.T)
    B, psi = _belief_split(Sobs)
    beta2b = np.broadcast_to(np.asarray(beta2, dtype=float),
                             (k,)).astype(float)
    D = psi + beta2b
    if V is None:
        Vaug = B
    else:
        Vv = as_loadings(V, k)
        Vaug = np.hstack([B, Vv])
    # one representation per factor law, so the nodes cannot see a
    # column sign, a rotation, or which eigenvector sign LAPACK chose
    # for the belief split (#130)
    from .nway import _canonical_loadings
    Vaug = _canonical_loadings(Vaug)
    rank = Vaug.shape[1]
    F, W = _mixture_nodes(rank, nodes_log2)
    if rank == 0:
        Vaug = np.zeros((k, 1))
    logW = np.log(np.maximum(W, 1e-300))
    shifts = F @ Vaug.T
    mu_obs = A @ m
    if rank >= 3 and kernel is not None:
        # scrambled-Sobol branch: recentre and rescale the node cloud on
        # the factor posterior (one kernel pass at the prior nodes), with
        # the importance correction folded into the weights. Under a
        # diffuse dense belief the posterior over the factors sits away
        # from the prior the nodes were drawn for and 1024 nodes carried
        # 13-35 percent effective weight; the verifier's referee measured
        # relative variance errors of 0.13 at prior sd 10 and 0.05 at 3
        # (2026-09-11). Recentred: 0.0075 and 0.03 at the same nodes, the
        # effective count near 80 percent. Kernels that mix internally
        # (the Gaussian winner pass) are left on the prior nodes.
        logps0, _ = kernel(mu_obs[None, :] + shifts, D, Vaug, F, W, psi, beta2b)
        if logps0 is not None:
            F, logW = _recentre_nodes(F, logW, logps0)
            shifts = F @ Vaug.T

    def mixture(mo):
        pts = mo[None, :] + shifts
        if kernel is not None:
            logps, grads = kernel(pts, D, Vaug, F, W, psi, beta2b)
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
    G = G - G.mean()               # contrasts only, as H below (#650)
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
    H = 0.5 * H + 0.5 * H.T        # halved first: see #279
    # Winner and order likelihoods see only contrasts of mu_obs, so the
    # exact H annihilates 1. Projecting the difference-noise out keeps a
    # large common belief component from being multiplied into it: with
    # S = eps I + 11' the lift S A' H A S otherwise moved the posterior
    # contrast variance 100x at eps = 1e-12 (#650).
    Pk = np.eye(k) - np.ones((k, k)) / k
    H = Pk @ H @ Pk
    SAt = S @ A.T @ Pk
    S_new = _psd_repair(S + SAt @ H @ SAt.T)
    return m_new, S_new, float(logZ)


def _winner_kernel(winner, k, points=801, base="normal"):
    """Vectorized winner kernel: log P and the gradient row for every
    factor node in ONE shared-field pass.

    Gaussian only. The shared-field pass and its adjoint run in the log
    domain on analytic Gaussian hazards (log_ndtr), so a general base
    would need those two primitives generalized first. The ORDER kernel
    has no such restriction, which is the path a failure lump wants
    anyway: under winner-only feedback a retirement is just "not the
    winner", which describes most of the field, while under ranked
    feedback it is a last-place finish and carries real information."""
    from ..factor.core import (jacobian_vector_product,
                               win_probabilities_factor)

    e = np.zeros(k)
    e[int(winner)] = 1.0

    def kernel(pts, D, Vaug, F, W, psi=None, beta2=None):
        if k == 2:
            # a pair winner IS the two-runner order, one Gaussian
            # contrast: priced per node in closed form (no lattice, #92)
            # and returned per node, so the mixture -- including the
            # Sobol recentring -- is the order kernel's exactly
            from .nway import _order_pass_batch
            return _order_pass_batch(pts, np.sqrt(np.maximum(D, 1e-300)),
                                     [int(winner), 1 - int(winner)])
        mo = pts[0] - (F @ Vaug.T)[0]
        a = -mo
        p = win_probabilities_factor(a, Vaug, D, F, W)
        Ji = jacobian_vector_product(a, Vaug, D, F, W, e, form="grid",
                                     points=points)
        pi = max(float(p[int(winner)]), 1e-300)
        return None, (-Ji / pi, float(np.log(pi)))

    return kernel


def _order_kernel(order, base="normal"):
    """Vectorized ordered-statistics kernel across factor nodes.

    Non-normal bases price each observation coordinate's predictive as
    the belief-split's diagonal Gaussian remainder psi convolved with
    the base noise (curves built once per update -- psi and beta2 are
    fixed across the mixture's mean shifts), not as the base at
    combined variance: the convolution-shortcut fix, full-covariance
    edition."""
    from .nway import _order_pass_batch, _predictive_curves

    order = np.asarray(order, dtype=int)   # validated by the callers
    cache = {}

    def kernel(pts, D, Vaug, F, W, psi=None, beta2=None):
        curves = None
        if base != "normal" and psi is not None:
            if "curves" not in cache:
                cache["curves"] = _predictive_curves(psi, beta2, base)
            curves = cache["curves"]
        return _order_pass_batch(pts, np.sqrt(np.maximum(D, 1e-300)), order,
                                 base=base, curves=curves)

    return kernel


def update_winner_full(m, S, winner, V=None, beta2=1.0, nodes_log2=10,
                       eps=None, points=801, eps_rel=0.15, base="normal"):
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
    if base != "normal":
        raise NotImplementedError(
            "update_winner_full is Gaussian only: the shared-field winner "
            "pass and its adjoint are analytic in the Gaussian log domain. "
            "For a non-normal base use update_order_full (ranked feedback, "
            "which is where a failure lump carries information) or the "
            "diagonal update_winner, both of which accept base=.")
    kernel = _winner_kernel(winner, len(np.asarray(m, dtype=float)),
                            points=points)
    return _mixture_update_full(m, S, V, beta2, None, nodes_log2=nodes_log2,
                                eps=eps, eps_rel=eps_rel, kernel=kernel)


def update_order_full(m, S, order, V=None, beta2=1.0, nodes_log2=10,
                      eps=None, eps_rel=0.15, base="normal"):
    """Full-order observation against a full-covariance belief
    (max-wins, order lists entrants first to last finisher, best first).
    A rank array (position of entrant i) is the inverse permutation and is
    silently wrong at K >= 3; convert it with order_from_positions (see
    winning.ratings.orders for the convention and a self-test).
    Returns (m_post, S_post, logZ);
    near-impossible orders degrade like order_loglik.

    Belief split as in update_winner_full (a diagonal belief costs no
    quadrature and reproduces update_order_correlated), and the
    surviving mixture rides _order_pass_batch -- the ordered-statistics
    sweeps vectorized across factor nodes, because online deployments
    are order-heavy.
    """
    from ..outcomes import as_order
    _check_full_inputs(m, S, V=V, beta2=beta2)                 # #456
    order = as_order(order, len(np.asarray(m)))                # #129
    if len(order) <= 1:                  # tautology: exact, no mixture
        return (np.asarray(m, dtype=float).copy(),
                np.asarray(S, dtype=float).copy(), 0.0)
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
    S = _psd_repair(S)
    # posterior (S^-1 + P T^-1 P)^-1 in Woodbury form with L = P T^-1/2,
    # so a singular or near-singular belief needs no inverse (#650):
    # S_new = S - S L K^-1 L' S, K = I + L' S L, and
    # m_new = m + S L K^-1 T^-1/2 (y - P m)
    if np.allclose(tau2, tau2[0]):
        it = 1.0 / np.sqrt(tau2)
        Lw = P * it[None, :]
        z = it * (y - P @ m)
    else:
        # The noise P eta has covariance P diag(tau2) P (rank n-1), whose
        # pseudo-inverse is P diag(1/tau2) P only for uniform tau2: the
        # posterior used that while the evidence used the true one, 0.23
        # off in the means for heterogeneous tau2 (#93, as update_market).
        # In an orthonormal contrast basis B, with B' T B = C C', the
        # precision is L L' for L = B C^-T.
        B = np.linalg.svd(P)[0][:, :n - 1]
        C = np.linalg.cholesky(B.T @ (tau2[:, None] * B))
        Lw = np.linalg.solve(C, B.T).T
        z = Lw.T @ (y - P @ m)
    SL = S @ Lw
    K = np.eye(Lw.shape[1]) + Lw.T @ SL
    S_new = S - SL @ np.linalg.solve(K, SL.T)
    m_new = m + SL @ np.linalg.solve(K, z)
    M = P @ (S + np.diag(tau2)) @ P
    # contrast-space evidence with a scale-aware rank test: the absolute
    # lam > 1e-12 dropped every contrast at ability unit 1e-7 and
    # returned logZ = 0 for any prices (#651)
    from ._contrast import contrast_gaussian
    r = y - P @ m
    logZ, _ = contrast_gaussian(
        r, M, scale=float(np.linalg.norm(y)) + float(np.linalg.norm(P @ m)))
    return m_new, _psd_repair(S_new), logZ
