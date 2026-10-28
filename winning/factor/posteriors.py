"""Exact Gaussian posteriors that stay in the race grammar.

The race verbs price N(m, diag(d) + W W') in time linear in the number
of entrants, so a belief that keeps that form after every observation
can be priced after every observation. Three updates keep it at a
known rank and are implemented here exactly (papers/exact_pom,
Section 3):

    crn_posterior             n common-random-number replicates of every
                              entrant (Section 3.1). Rank = the number of
                              shared input variables.
    crn_posterior_replicates  the same with entrants run on DIFFERENT
                              subsets of the shared replicates. Still
                              exact; the rank grows with the number of
                              distinct observation patterns.
    feature_bandit_posterior  per-arm pulls under a shared-feature prior
                              (Section 3.2, the stable form of Eq. 14).
                              Rank = the prior's.

A fourth, observing a contrast a'theta (update_contrast), keeps it only
at a cost in rank: the posterior covariance Sigma - g g'/s is a rank-one
DOWNDATE, which diag(D) + V V' cannot absorb with the same D and rank.
update_contrast represents it exactly with |support(a)| - 1 extra
columns and a smaller D on a's support, or (refit=True) refits it to a
bounded rank with fit_covariance and reports the residual.

Max-wins and min-wins are the caller's affair: these are beliefs about
theta, and race_probabilities(-m, V=-W, D=d) is the max race on them.
"""
from __future__ import annotations

import numpy as np

from ..shapes import as_idio, as_loadings


def _psd_inv_sqrt(H):
    """Symmetric H^{-1/2} of a positive definite rho x rho matrix.

    The SYMMETRIC root, not a Cholesky factor, so that rotating the
    loadings V -> V Q rotates the posterior loadings W -> W Q exactly.
    """
    H = 0.5 * (H + H.T)
    lam, Q = np.linalg.eigh(H)
    return (Q / np.sqrt(lam)) @ Q.T


def _unit(K, *variances):
    """Per-coordinate powers of two near the square root of each
    coordinate's own typical variance, shape (K,).

    The three constructors are equivariant under a per-coordinate change
    of units: scaling coordinate i's locations, loadings and sds by c_i
    (variances by c_i^2) scales m_i and row i of W by c_i and d_i by
    c_i^2. Dividing by an exact power of two per coordinate and
    multiplying the outputs back is therefore exact (m and d bit for bit,
    W W' to rounding, since W's final compression is not row-scale
    equivariant) wherever no intermediate over- or underflows, and keeps
    the precisions (n/D, counts/sigma_n^2) finite when variances sit near
    either end of the float range: at D = 4e-308, n/D overflowed and a
    representable posterior variance of 3.6e-309 came back as 0 (#670).
    One global unit could not serve a field mixing 2e-154 and 1e154
    scales (NaN means); each coordinate now takes its own. Each argument
    is (K,) or (K, R) (NaN or nonpositive entries ignored).
    """
    rows = [np.asarray(x, dtype=float).reshape(K, -1) for x in variances]
    v = np.hstack(rows) if rows else np.zeros((K, 0))
    good = np.isfinite(v) & (v > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(good, np.log2(np.where(good, v, 1.0)), np.nan)
    u = np.ones(K)
    ok = good.any(axis=1)
    if ok.any():
        # the median of the logs: a median of two 1e308 variances
        # averaged them to inf
        med = np.nanmedian(lv[ok], axis=1)
        u[ok] = np.ldexp(1.0, np.round(0.5 * med).astype(int))
    return u


def _pooled_unit(*variances):
    """One power of two near the square root of the median variance of
    the whole field (the pre-#670 global unit): the scale of a latent
    common shift, whose flat prior makes its scale arbitrary, so the
    free column level keeps the latent unit it always had."""
    v = np.concatenate([np.ravel(np.asarray(x, dtype=float))
                        for x in variances])
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return 1.0
    with np.errstate(over="ignore"):
        med = np.median(v)
    lm = np.log2(med) if np.isfinite(med) else np.median(np.log2(v))
    return float(np.ldexp(1.0, int(np.round(0.5 * lm))))


def _positive_vector(x, n, name, what):
    A = np.asarray(x, dtype=float)
    if A.ndim == 0:
        A = np.full(n, float(A))
    elif A.shape != (n,):
        raise ValueError(f"{name} must be a scalar or shape ({n},); "
                         f"got shape {A.shape}")
    if not np.isfinite(A).all() or (A <= 0.0).any():
        raise ValueError(
            f"{name} must be finite and strictly positive ({what}); a zero "
            "is a point mass, which is outside the scope of these updates")
    return A


def _finite_vector(x, n, name):
    A = np.asarray(x, dtype=float)
    if A.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},); got {A.shape}")
    if not np.isfinite(A).all():
        raise ValueError(f"{name} has a non-finite entry")
    return A


def crn_posterior(V, D, n, ybar, s0, m0=None):
    """Exact posterior of the means after n common-random-number replicates.

    Model (paper Section 3.1): replicate r scores every entrant on one
    shared draw of the random inputs,

        Y_ir = theta_i + v_i' F_r + eps_ir,  F_r ~ N(0, I_rho),
        eps_ir ~ N(0, D_i),  theta ~ N(m0, diag(s0^2)),

    with V (n_entrants x rho) and D (replicate noise VARIANCES, > 0)
    known and F_r unrecorded. ybar is the vector of replicate means.
    Returns (m, d, W) with posterior theta ~ N(m, diag(d) + W W'),
    EXACT, W of the prior's rank rho (Equation 12).

    Computed without the double Woodbury's subtraction: integrating
    theta out given the replicate-mean factor fbar ~ N(0, I/n) gives
    tau = s0^2 + D/n, k = s0^2/tau, H = n I + V' diag(1/tau) V and

        d = 1/(1/s0^2 + n/D),  W = diag(k) V H^{-1/2},
        m = m0 + k (ybar - m0 - V fhat),  fhat = H^{-1} V'((ybar - m0)/tau),

    every term a sum of positive quantities. W rotates with V (V -> VQ
    gives W -> WQ); only W W' is identified.

    n is a replicate count >= 0 (n = 0 returns the prior with W = 0).
    An ARRAY of per-entrant counts is accepted only when its entries are
    equal: with unequal counts the per-entrant means are NOT a
    sufficient statistic (a replicate seen by entrant i alone carries
    different information about theta_i than one shared with others,
    because only the shared one lets the factor be differenced out), so
    no function of (n_i, ybar) is the exact posterior. That case is
    crn_posterior_replicates, which takes the replicates themselves.
    """
    yb = np.asarray(ybar, dtype=float)
    if yb.ndim != 1:
        raise ValueError(f"ybar must be a vector; got shape {yb.shape}")
    K = len(yb)
    yb = _finite_vector(yb, K, "ybar")
    Vm = as_loadings(V, K)
    Dv = as_idio(D, K, positive=True)
    s0v = _positive_vector(s0, K, "s0", "a prior sd")
    m0v = np.zeros(K) if m0 is None else _finite_vector(m0, K, "m0")
    nn = np.asarray(n, dtype=float)
    if nn.ndim == 1:
        if nn.shape != (K,):
            raise ValueError(f"n must be a scalar or shape ({K},); "
                             f"got shape {nn.shape}")
        if not (nn == nn[0]).all():
            raise ValueError(
                "unequal per-entrant replicate counts: under common random "
                "numbers the per-entrant means are not sufficient for that "
                "design, so (n_i, ybar) has no exact posterior. Pass the "
                "replicates to crn_posterior_replicates (NaN where an "
                "entrant was not run).")
        nn = nn[0]
    elif nn.ndim != 0:
        raise ValueError(f"n must be a scalar or shape ({K},)")
    if not (np.isfinite(nn) and nn >= 0 and nn == np.floor(nn)):
        raise ValueError(f"n must be a non-negative integer; got {n!r}")
    nr = float(nn)
    rho = Vm.shape[1]
    if nr == 0:
        return m0v.copy(), s0v ** 2, np.zeros((K, rho))
    u = _unit(K, Dv, s0v ** 2)
    Vm, Dv, s0v, yb, m0v = (Vm / u[:, None], Dv / u / u, s0v / u, yb / u,
                            m0v / u)
    s02 = s0v ** 2
    tau = s02 + Dv / nr
    k = s02 / tau
    d = 1.0 / (1.0 / s02 + nr / Dv)
    H = nr * np.eye(rho) + (Vm.T / tau) @ Vm
    W = (k[:, None] * Vm) @ _psd_inv_sqrt(H)
    r = yb - m0v
    fhat = np.linalg.solve(H, Vm.T @ (r / tau)) if rho else np.zeros(0)
    m = m0v + k * (r - Vm @ fhat)
    return m * u, d * u * u, W * u[:, None]


def crn_posterior_replicates(V, D, Y, s0=None, m0=None, *, prior=None,
                             counts=None, groups=None, common=None):
    """Exact CRN posterior when entrants were run on different replicates.

    Y is (n_entrants x R): column r holds replicate r's outputs, NaN for
    an entrant not run on it (same model as crn_posterior; replicate r
    shares one F_r across every entrant it ran). Returns (m, d, W),
    theta ~ N(m, diag(d) + W W') EXACTLY.

    Keyword extensions (#635, #636, #638). Each is exact Gaussian
    conditioning, not an approximation; the general model they share is

        theta = m0 + V0 beta + e,      beta ~ N(0, I_q), e ~ N(0, diag d0),
        Y_ir  = theta_i + v_i' F_r + sum_l u_li' G_{l, g_l(r)}
                + c_r + eps_ir,        eps_ir ~ N(0, D_ir),

    every shock independent N(0, I) except c_r (see common=).

    prior=(V0, d0) or (V0, d0, m0), a TUPLE, or a dense (K, K) Sigma0
        replaces the independent s0^2 I prior (#635); pass s0=None. A factor
        prior keeps its eigenstructure: W rotates with V0 and nothing is
        refit. A dense Sigma0 must be positive definite; it is split
        exactly as Sigma0 = diag(d0) + V0 V0' with d0 = (c/2) diag(Sigma0),
        c the smallest eigenvalue of its correlation matrix (V0 has rank
        n_entrants, so W does too).
    D may be (n_entrants, R): a noise variance per cell (#638). Entries
        at unobserved cells are ignored.
    counts (n_entrants, R): the number of runs averaged into each cell
        (#638). D is then the SINGLE-RUN variance (per entrant, or per
        cell) and cell (i, r) has variance D_ir / counts_ir. This is exact
        when the runs pooled in a cell share that column's shocks and
        differ only in independent noise: the cell mean is then sufficient
        for the cell. Observed cells need counts > 0; an unobserved cell
        must have count 0 or NaN.
    groups=[(labels, U), ...] adds shared shock levels (#636): labels has
        length R, and columns with the same label share one draw G with
        loadings U (n_entrants x rho_l). Trials repeated under shared seeds
        are groups=[(trial_id, U_trial), (seed_id, U_seed)], with V the
        column-level (trial x seed) shock, which may have zero columns.
    common="free" gives each column a level shift c_r with a FLAT prior
        and integrates it out exactly (#636): only within-column contrasts
        then inform theta, the limit of a synthetic common column whose
        variance goes to infinity. A column observed for one entrant then
        carries no information.

    The general path (any keyword above, or V=None for no column-level
    shock) costs O(P^3) in P = q + rho R' + sum_l rho_l n_l (+ R' with
    common="free"), R' the number of distinct columns: columns are merged
    only when their pattern, noise, and group labels coincide. Without
    any keyword the original path below runs unchanged.

    Derivation. Group the replicates by observation pattern S_l (the set
    of entrants run), c_l replicates and per-entrant sums T_l in group
    l. The group's factor mean fbar_l ~ N(0, I/c_l) is independent
    across groups. Given F = (fbar_1..fbar_L), entrants are independent:
    theta_i | F, Y ~ N(b_i - (G F)_i, 1/a_i) with

        a_i = 1/s0_i^2 + n_i/D_i,   n_i = sum_{l: i in S_l} c_l,
        b_i = (m0_i/s0_i^2 + sum_l T_li/D_i)/a_i,
        G[i, block l] = c_l 1[i in S_l] v_i' / (a_i D_i),

    and, integrating theta out, F has posterior precision
    H = blockdiag(c_l (I + V_l' D_l^{-1} V_l)) - Gt' diag(1/a) Gt with
    Gt = diag(a) G (positive definite: it is a Schur complement of the
    joint (theta, F) precision). So the posterior is diag(1/a) + G H^{-1} G',
    a factor form of rank rho * L. With equal counts (L = 1) this is
    crn_posterior. The rank, not the exactness, is what unequal
    observation costs; columns beyond n_entrants are compressed away.

    The general path is the same elimination with theta = m0 + V0 beta + e:
    the latent psi = (beta, every shock) has posterior precision H (the
    Schur complement of e in the joint precision of (e, psi), positive
    definite whenever the posterior is proper) and

        d = 1/a,  a_i = 1/d0_i + s_i,  s_i = sum_r 1/D_ir (observed),
        W = Ghat H^{-1/2},  Ghat = [diag(kappa) V0, -diag(1/a) Gt],
        kappa_i = 1/(1 + d0_i s_i),

    with Gt the e-to-shock block of the joint precision. The beta parts
    of Ghat, of H and of the score are formed subtraction-free through
    kappa, as in feature_bandit_posterior; with no shocks this IS
    feature_bandit_posterior, and with q = 0 it is the derivation above.
    """
    if (prior is not None or counts is not None or groups is not None
            or common is not None or V is None or np.ndim(D) == 2):
        return _crn_general(V, D, Y, s0, m0, prior, counts, groups, common)
    if s0 is None:
        raise ValueError("s0 is required unless prior= is given")
    Ym = np.asarray(Y, dtype=float)
    if Ym.ndim != 2:
        raise ValueError(f"Y must be (n_entrants, R); got shape {Ym.shape}")
    K = Ym.shape[0]
    if np.isinf(Ym).any():
        raise ValueError("Y has an infinite entry (NaN marks 'not run')")
    Vm = as_loadings(V, K)
    Dv = as_idio(D, K, positive=True)
    s0v = _positive_vector(s0, K, "s0", "a prior sd")
    m0v = np.zeros(K) if m0 is None else _finite_vector(m0, K, "m0")
    rho = Vm.shape[1]
    u = _unit(K, Dv, s0v ** 2)
    Vm, Dv, s0v, Ym, m0v = (Vm / u[:, None], Dv / u / u, s0v / u,
                            Ym / u[:, None], m0v / u)
    obs = ~np.isnan(Ym)
    keep = obs.any(axis=0)
    obs, Yz = obs[:, keep], np.where(obs, Ym, 0.0)[:, keep]
    s02 = s0v ** 2
    if obs.shape[1] == 0:
        return m0v * u, s02 * u * u, np.zeros((K, rho))
    pats, inv = np.unique(obs.T, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    L = len(pats)
    c = np.bincount(inv, minlength=L).astype(float)
    n_i = obs.sum(axis=1).astype(float)
    a = 1.0 / s02 + n_i / Dv
    b = (m0v / s02 + Yz.sum(axis=1) / Dv) / a
    # Gt[:, block l] = c_l 1[i in S_l] v_i / D_i
    mask = pats.T.astype(float)                     # K x L
    Gt = ((mask * c)[:, :, None] * (Vm / Dv[:, None])[:, None, :]
          ).reshape(K, L * rho)
    H = -(Gt.T / a) @ Gt
    Vd = Vm / np.sqrt(Dv)[:, None]
    for j in range(L):
        sl = slice(j * rho, (j + 1) * rho)
        Vj = Vd[pats[j]]
        H[sl, sl] += c[j] * (np.eye(rho) + Vj.T @ Vj)
    G = Gt / a[:, None]
    # posterior mean of F. The joint (theta, F) precision has blocks
    # diag(a), Gt, blockdiag(c_l M_l) and linear terms a*b and
    # V' D^{-1} T_l; eliminating theta leaves H Fhat = V' D^{-1} T_l
    # - Gt' b, i.e. block l: V' D^{-1} (T_l - c_l 1_{S_l} b).
    T = np.zeros((K, L))
    np.add.at(T.T, inv, Yz.T)
    resid = T - mask * c * b[:, None]
    score = ((Vm / Dv[:, None]).T @ resid).T.reshape(L * rho)
    Fhat = np.linalg.solve(0.5 * (H + H.T), score)
    m = b - G @ Fhat
    W = G @ _psd_inv_sqrt(H)
    if W.shape[1] > K:
        U, s, _ = np.linalg.svd(W, full_matrices=False)
        W = U * s
    return m * u, (1.0 / a) * u * u, W * u[:, None]


def _crn_prior(prior, s0, m0, K):
    """(m0, V0, d0) of the prior theta ~ N(m0, V0 V0' + diag d0)."""
    if prior is None:
        if s0 is None:
            raise ValueError("s0 is required unless prior= is given")
        s0v = _positive_vector(s0, K, "s0", "a prior sd")
        m0v = np.zeros(K) if m0 is None else _finite_vector(m0, K, "m0")
        return m0v, np.zeros((K, 0)), s0v ** 2
    if s0 is not None:
        raise ValueError("pass either s0 (independent prior) or prior=, "
                         "not both")
    # a TUPLE is the factor form; anything else is read as a dense matrix,
    # so a nested-list 2 x 2 covariance is never taken for (V0, d0)
    if isinstance(prior, tuple):
        if len(prior) not in (2, 3):
            raise ValueError("prior= is (V0, d0), (V0, d0, m0) or a dense "
                             "(K, K) covariance")
        V0 = as_loadings(prior[0], K)
        d0 = as_idio(prior[1], K, positive=True)
        mp = prior[2] if len(prior) == 3 else None
        if mp is not None and m0 is not None:
            raise ValueError("the prior mean was given twice (prior=(V0, "
                             "d0, m0) and m0=)")
        mp = m0 if mp is None else mp
        m0v = np.zeros(K) if mp is None else _finite_vector(mp, K, "m0")
        return m0v, V0, d0
    S = np.asarray(prior, dtype=float)
    if S.shape != (K, K):
        raise ValueError(f"a dense prior must be ({K}, {K}); got {S.shape}")
    from .core import _validate_covariance
    S = _validate_covariance(S, name="prior")
    sd = np.sqrt(np.diag(S))
    if not (sd > 0).all():
        raise ValueError("prior has a zero variance: a point mass, which is "
                         "outside the scope of these updates")
    lam = float(np.linalg.eigvalsh(S / np.outer(sd, sd)).min())
    if not lam > 1e-10:
        raise ValueError(
            f"prior is (numerically) singular: smallest correlation "
            f"eigenvalue {lam:.2e}. A singular prior is a point mass on a "
            "contrast, outside the scope of these updates; pass a factor "
            "prior (V0, d0) with d0 > 0 if that is what is meant")
    # exact split: S - (lam/2) diag(S) is positive definite (its
    # correlation-scale eigenvalues are >= lam/2), so it has a symmetric
    # square root and nothing is clamped
    d0 = 0.5 * lam * np.diag(S)
    V0 = _psd_sqrt(S - np.diag(d0))
    m0v = np.zeros(K) if m0 is None else _finite_vector(m0, K, "m0")
    return m0v, V0, d0


def _crn_cells(D, counts, Ym, obs):
    """Per-cell noise variances (K, R), meaningful where obs."""
    K, R = Ym.shape
    Dm = np.asarray(D, dtype=float)
    if Dm.ndim == 2:
        if Dm.shape != (K, R):
            raise ValueError(f"a D matrix must be ({K}, {R}); got {Dm.shape}")
        Dc = Dm.copy()
    else:
        Dc = np.broadcast_to(as_idio(D, K, positive=True)[:, None],
                             (K, R)).copy()
    if Dm.ndim == 2:
        bad = obs & ~(np.isfinite(Dc) & (Dc > 0))
        if bad.any():
            i, r = np.argwhere(bad)[0]
            raise ValueError(
                f"D[{i}, {r}] = {Dc[i, r]!r} at an observed cell: noise "
                "variances must be finite and > 0 (a zero is a point mass, "
                "outside the scope of these updates)")
    if counts is not None:
        C = np.asarray(counts, dtype=float)
        if C.shape != (K, R):
            raise ValueError(f"counts must be ({K}, {R}); got {C.shape}")
        if (np.isinf(C) | (C < 0)).any():
            raise ValueError("counts must be finite and >= 0 (NaN allowed "
                             "at unobserved cells)")
        if (obs & ~(C > 0)).any():
            i, r = np.argwhere(obs & ~(C > 0))[0]
            raise ValueError(f"cell ({i}, {r}) is observed but counts is "
                             f"{C[i, r]!r}; an observed cell needs counts > 0")
        if (~obs & (C > 0)).any():
            i, r = np.argwhere(~obs & (C > 0))[0]
            raise ValueError(f"cell ({i}, {r}) has counts {C[i, r]!r} but "
                             "Y is NaN there; a cell with runs needs a value")
        Dc = np.where(obs, Dc / np.where(obs, C, 1.0), 1.0)
    return np.where(obs, Dc, 1.0)


def _crn_general(V, D, Y, s0, m0, prior, counts, groups, common):
    """crn_posterior_replicates with a factor/dense prior, per-cell noise,
    shared shock levels and/or a free column level (see its docstring)."""
    Ym = np.asarray(Y, dtype=float)
    if Ym.ndim != 2:
        raise ValueError(f"Y must be (n_entrants, R); got shape {Ym.shape}")
    K, R = Ym.shape
    if np.isinf(Ym).any():
        raise ValueError("Y has an infinite entry (NaN marks 'not run')")
    m0v, V0, d0 = _crn_prior(prior, s0, m0, K)
    Vm = np.zeros((K, 0)) if V is None else as_loadings(V, K)
    if common not in (None, "free"):
        raise ValueError(f"common must be None or 'free'; got {common!r}")
    lev = []
    if groups is not None:
        for j, g in enumerate(groups):
            if not (isinstance(g, (tuple, list)) and len(g) == 2):
                raise ValueError("groups is a list of (labels, loadings) "
                                 "pairs")
            lab = np.asarray(g[0])
            if lab.shape != (R,):
                raise ValueError(f"groups[{j}] labels must have length {R} "
                                 f"(one per column of Y); got {lab.shape}")
            if lab.dtype.kind == "f" and not np.isfinite(lab).all():
                raise ValueError(f"groups[{j}] labels must be finite")
            lev.append((np.unique(lab, return_inverse=True)[1].ravel(),
                        as_loadings(g[1], K)))
    obs = ~np.isnan(Ym)
    Dc = _crn_cells(D, counts, Ym, obs)
    u = _unit(K, np.where(obs, Dc, np.nan), d0)
    u0 = _pooled_unit(Dc[obs], d0)
    uc = u[:, None]
    Ym, m0v, V0, d0, Vm, Dc = (Ym / uc, m0v / u, V0 / uc, d0 / u / u,
                               Vm / uc, Dc / uc / uc)
    lev = [(lab, U / uc) for lab, U in lev]
    keep = obs.any(axis=0)
    obs, Dc = obs[:, keep], Dc[:, keep]
    Yz = np.where(obs, Ym[:, keep], 0.0)
    lev = [(lab[keep], U) for lab, U in lev]
    # merge columns whose pattern, cell noise and group labels coincide:
    # c of them are one column with mean Y, noise D/c and a column shock
    # of precision c (the within-group deviations are free of theta and
    # of every shared shock, so they carry nothing about theta)
    key = np.column_stack([obs.T, np.where(obs, Dc, 0.0).T]
                          + [lab[:, None] for lab, _ in lev])
    if key.shape[0]:
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = np.asarray(inv).ravel()
    else:
        inv = np.zeros(0, dtype=int)
    Lc = int(inv.max()) + 1 if inv.size else 0
    c = np.bincount(inv, minlength=Lc).astype(float)
    first = np.unique(inv, return_index=True)[1]     # one column per group
    T = np.zeros((K, Lc))
    np.add.at(T.T, inv, Yz.T)
    obs_l = obs[:, first]
    w = np.where(obs_l, c / Dc[:, first], 0.0)      # precision of the mean
    r = np.where(obs_l, T / c - m0v[:, None], 0.0)
    # shock levels: (codes per merged column, loadings, prior precision)
    levels = []
    if Vm.shape[1]:
        levels.append((np.arange(Lc), Vm, c))
    if common == "free":
        # one common shift: u0 / u_i in these units, u0 the field's
        # pooled unit (any scale serves under its flat prior; this one
        # keeps the arithmetic of the pre-#670 global unit)
        with np.errstate(over="ignore"):
            info = np.max(w * (u0 / uc) ** 2) if w.size else 0.0
        if not np.isfinite(info):
            # coordinates 1e200 apart in scale: the pooled unit put a
            # loading of 1e200 on the small one and its square overflowed
            # the precision (#670). Then the unit of the coordinate that
            # pins the shift best, so every loading-weighted precision
            # w_ir (u0 / u_i)^2 is at most about one
            pos = w > 0
            r0 = np.min(np.where(pos, uc / np.sqrt(np.where(pos, w, 1.0)),
                                 np.inf))
            u0 = float(np.ldexp(1.0, int(np.round(np.log2(r0)))))
        levels.append((np.arange(Lc), u0 / uc, np.zeros(Lc)))
    for lab, U in lev:
        if U.shape[1]:
            codes = np.unique(lab[first], return_inverse=True)[1].ravel()
            levels.append((codes, U, np.ones(int(codes.max()) + 1
                                             if codes.size else 0)))
    s = w.sum(axis=1)
    a = 1.0 / d0 + s
    kap = 1.0 / (1.0 + d0 * s)
    he = (w * r).sum(axis=1)
    Gts, hz, sizes = [], [], []
    for codes, U, _ in levels:
        n_l = int(codes.max()) + 1 if codes.size else 0
        Wa = np.zeros((K, n_l))
        np.add.at(Wa.T, codes, w.T)
        WRa = np.zeros((K, n_l))
        np.add.at(WRa.T, codes, (w * r).T)
        Gts.append((Wa[:, :, None] * U[:, None, :]).reshape(K, -1))
        hz.append((WRa.T @ U).reshape(-1))
        sizes.append(n_l * U.shape[1])
    Pz = int(sum(sizes))
    q = V0.shape[1]
    Gt = np.hstack(Gts) if Gts else np.zeros((K, 0))
    Bz = np.zeros((Pz, Pz))
    offs = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    for x, (cx, Ux, px) in enumerate(levels):
        nx, rx = sizes[x] // Ux.shape[1], Ux.shape[1]
        idx = np.arange(nx * rx).reshape(nx, rx)
        Bz[offs[x] + idx, offs[x] + idx] += px[:, None]
        for y, (cy, Uy, _) in enumerate(levels):
            if y < x:
                continue
            ny, ry = sizes[y] // Uy.shape[1], Uy.shape[1]
            Tl = np.einsum("il,ia,ib->lab", w, Ux, Uy)
            acc = np.zeros((nx, ny, rx, ry))
            np.add.at(acc, (cx, cy), Tl)
            blk = acc.transpose(0, 2, 1, 3).reshape(nx * rx, ny * ry)
            Bz[offs[x]:offs[x + 1], offs[y]:offs[y + 1]] += blk
            if y != x:
                Bz[offs[y]:offs[y + 1], offs[x]:offs[x + 1]] += blk.T
    P = q + Pz
    H = np.empty((P, P))
    H[:q, :q] = np.eye(q) + (V0.T * (s * kap)) @ V0
    H[:q, q:] = (V0.T * kap) @ Gt
    H[q:, :q] = H[:q, q:].T
    H[q:, q:] = Bz - (Gt.T / a) @ Gt
    H = 0.5 * (H + H.T)
    score = np.concatenate([V0.T @ (kap * he),
                            (np.concatenate(hz) if hz else np.zeros(0))
                            - Gt.T @ (he / a)])
    Ghat = np.hstack([kap[:, None] * V0, -Gt / a[:, None]])
    m = m0v + he / a
    if P:
        lam, Q = np.linalg.eigh(H)
        if not lam[0] > 1e-12 * max(float(lam[-1]), 1e-300):
            raise ValueError(
                "the shared shocks are not identified by these data "
                f"(posterior precision eigenvalue {lam[0]:.2e} against "
                f"{lam[-1]:.2e}); the posterior is improper or numerically "
                "singular, so no exact answer is returned")
        m = m + Ghat @ ((Q.T @ score) / lam @ Q.T)
        W = Ghat @ ((Q / np.sqrt(lam)) @ Q.T)
        if W.shape[1] > K:
            Us, sv, _ = np.linalg.svd(W, full_matrices=False)
            W = Us * sv
    else:
        W = np.zeros((K, 0))
    return m * u, (1.0 / a) * u * u, W * u[:, None]


def feature_bandit_posterior(V0, d0, counts, sums, sigma_n):
    """Exact posterior of the arm means of a shared-feature bandit.

    Prior theta ~ N(0, V0 V0' + diag(d0)) (paper Section 3.2: arm i has
    features v_i, theta_i = v_i' beta + e_i, beta ~ N(0, I)); pulling
    arm i returns theta_i + N(0, sigma_n_i^2). counts and sums are the
    per-arm pull counts and reward totals; sigma_n is a scalar or one
    noise sd per arm. Returns (m, d, W) with
    theta | data ~ N(m, diag(d) + W W'), EXACT, W of the prior's rank,
    via the subtraction-free form of Equation 14:

        lambda = counts / sigma_n^2,  r = 1 / (1 + d0 lambda),
        H = I + V0' diag(lambda r) V0,
        d = d0 r,  W = diag(r) V0 H^{-1/2},  m = (diag(d) + W W') sums/sigma_n^2.

    An arm never pulled has lambda = 0 (and must have sum 0). W rotates
    with V0; only W W' is identified.
    """
    cnt = np.asarray(counts, dtype=float)
    if cnt.ndim != 1:
        raise ValueError(f"counts must be a vector; got shape {cnt.shape}")
    K = len(cnt)
    cnt = _finite_vector(cnt, K, "counts")
    if (cnt < 0).any():
        raise ValueError("counts must be non-negative")
    sm = _finite_vector(sums, K, "sums")
    if (sm[cnt == 0] != 0).any():
        raise ValueError("an arm with count 0 has a nonzero sum")
    Vm = as_loadings(V0, K)
    d0v = as_idio(d0, K, positive=True)
    sn = _positive_vector(sigma_n, K, "sigma_n", "a noise sd")
    u = _unit(K, d0v, sn ** 2)
    Vm, d0v, sn, sm = Vm / u[:, None], d0v / u / u, sn / u, sm / u
    lam = cnt / sn ** 2
    r = 1.0 / (1.0 + d0v * lam)
    rho = Vm.shape[1]
    H = np.eye(rho) + (Vm.T * (lam * r)) @ Vm
    W = (r[:, None] * Vm) @ _psd_inv_sqrt(H)
    d = d0v * r
    h = sm / sn ** 2
    m = d * h + W @ (W.T @ h)
    return m * u, d * u * u, W * u[:, None]


def _psd_sqrt(C):
    """Symmetric square root of a positive semidefinite matrix."""
    lam, Q = np.linalg.eigh(0.5 * (C + C.T))
    return (Q * np.sqrt(np.maximum(lam, 0.0))) @ Q.T


def update_contrast(m, cov, a, mean, var, refit=False):
    """Condition a Gaussian belief on a'theta ~ N(mean, var), exactly.

    The Kalman update for one linear observation: g = Sigma a,
    s = a' Sigma a + var, m' = m + g (mean - a'm)/s,
    Sigma' = Sigma - g g'/s. Typical a: child minus parent (e_i - e_j),
    an external judgment of their difference with its own error
    variance. (update_team_margins_full reaches the same update with
    A = [e_i; e_j], scores=[mean, 0] and beta2 = var/2.)

    cov is a dense Sigma, returning (m', Sigma'), or a factor belief
    (V, D) / (V, D, F, W) with Sigma = V V' + diag(D), D > 0 (F, W:
    quadrature nodes and weights, as fit_covariance returns them),
    returning (m', V', D', F', W', report).

    The factor form is kept EXACTLY, at a cost in rank. Sigma' is a
    rank-one downdate, which diag(D) + V V' cannot absorb with the same
    D and rank (a orthogonal to every loading does not rescue it: then
    g = D a, still off-diagonal once a touches two entrants). Write
    theta = m + V f + e. Given f the observation sees only a'e, so with
    S the support of a, b = V'a, s_e = a'Da + var:

        theta | data = m'' + Vt f + zeta,   Vt = V - D a b'/s_e,
        f | data ~ N(fhat, I - b b'/(b'b + s_e)),
        zeta ~ N(0, diag(D) - (D a)(D a)'/s_e)   (differs from D on S only).

    The S x S block Z of Cov(zeta) splits as c diag(Z) + (Z - c diag(Z))
    with c the smallest eigenvalue of Z's correlation matrix (c > 0
    because var > 0); the second part is PSD of rank |S| - 1. So

        V' = [Vt (I - b b'/(b'b + s_e))^{1/2},  E_S (Z - c diag Z)^{1/2}],
        D' = D off S,  c diag(Z) on S,

    exact, rank rho + |S| - 1: unchanged for a single-entrant
    observation, one more column per two-entrant contrast. D' shrinks
    on S as var -> 0 (the contrast becomes known; var = 0 is a point
    mass and is refused). F, W are passed through when the rank is
    unchanged and given, else a fresh Sobol rule of the new rank.
    report: {"exact": True, "rank": ...}.

    refit=True instead forms Sigma' densely (exactly) and refits it with
    fit_covariance(k = rank of V), returning its residual diagnostics
    in report with report["exact"] False. That bounds the rank, at the
    price of the fit residual (measured: contrast_residual_max 0.26 on
    a 40-entrant rank-2 example), and the fit is to the choice-relevant
    part of Sigma' only, so chain further updates on the exact form.

    a must be finite and nonzero, var finite and > 0. The exact factor
    path costs O(n rho^2 + |S|^3); dense and refit paths O(n^2) memory.
    """
    mv = np.asarray(m, dtype=float)
    if mv.ndim != 1:
        raise ValueError(f"m must be a vector; got shape {mv.shape}")
    K = len(mv)
    mv = _finite_vector(mv, K, "m")
    av = _finite_vector(a, K, "a")
    if not (av != 0).any():
        raise ValueError("a is the zero vector: it observes nothing")
    mean = float(mean)
    if not np.isfinite(mean):
        raise ValueError("mean must be finite")
    var = float(var)
    if not (np.isfinite(var) and var > 0.0):
        raise ValueError(
            f"var must be finite and > 0; got {var!r}. var = 0 is a hard "
            "constraint (a point mass on a'theta), outside the scope of "
            "this update")
    if not isinstance(cov, tuple):
        S = np.asarray(cov, dtype=float)
        if S.shape != (K, K):
            raise ValueError(f"cov must be ({K}, {K}) or a factor tuple; "
                             f"got shape {S.shape}")
        if not np.isfinite(S).all():
            raise ValueError("cov has a non-finite entry")
        # a Kalman update is Gaussian conditioning only on a covariance:
        # an indefinite [[1, 2], [2, 1]] came back as a valid-looking
        # posterior (#634). The shared validator also symmetrises.
        from .core import _validate_covariance
        S = _validate_covariance(S, name="cov")
        g = S @ av
        s = float(av @ g) + var
        m_new = mv + g * ((mean - float(av @ mv)) / s)
        S_new = S - np.outer(g, g) / s
        return m_new, 0.5 * (S_new + S_new.T)
    if len(cov) not in (2, 4):
        raise ValueError("a factor cov is (V, D) or (V, D, F, W)")
    Vm = as_loadings(cov[0], K)
    Dv = as_idio(cov[1], K, positive=True)
    rho = Vm.shape[1]
    b = Vm.T @ av
    Da = Dv * av
    s_e = float(av @ Da) + var
    s = float(b @ b) + s_e
    g = Vm @ b + Da
    m_new = mv + g * ((mean - float(av @ mv)) / s)
    if refit:
        from .core import fit_covariance
        S_new = Vm @ Vm.T + np.diag(Dv) - np.outer(g, g) / s
        S_new = 0.5 * (S_new + S_new.T)
        V_new, D_new, F, Wq, report = fit_covariance(
            S_new, k=max(rho, 1), return_report=True)
        return m_new, V_new, D_new, F, Wq, dict(report, exact=False)
    sup = np.flatnonzero(av)
    Vt = Vm - np.outer(Da, b) / s_e
    cols = [Vt @ _psd_sqrt(np.eye(rho) - np.outer(b, b) / s)]
    # Cov(zeta) on the support, built without cancelling. Its diagonal
    # is D_i (s_e - D_i a_i^2)/s_e, and s_e - D_i a_i^2 is the sum of
    # positive terms var + sum_{j != i} D_j a_j^2; forming it as
    # D_i - (D_i a_i)^2/s_e instead gave exactly 0 once var fell below
    # machine precision relative to a'Da, then 0/0 in the correlation
    # and a NaN in D' (#669).
    q = Da[sup] * av[sup]                           # D_i a_i^2 >= 0
    # s_e - D_i a_i^2 as a sum of the OTHER terms, never a difference
    rest = var + np.array([np.delete(q, i).sum() for i in range(len(sup))])
    with np.errstate(over="ignore", under="ignore", invalid="ignore",
                     divide="ignore"):
        zz = Dv[sup] * rest / s_e                   # diag of Cov(zeta)
        z = np.sqrt(zz)
        Rz = -np.outer(Da[sup], Da[sup]) / s_e / np.outer(z, z)
    if not (np.isfinite(Rz).all() and (zz > 0).all() and np.isfinite(zz).all()):
        # D_i rest_i and D_i a_i D_j a_j overflowed at D = 1e155 (NaN in
        # D'), though both ratios are ordinary (#669). The same quantities
        # with each product taken in its scale-free form: zz by exponent
        # and mantissa, Rz_ij = -w_i w_j with w_i = a_i sqrt(D_i / rest_i)
        mD, eD = np.frexp(Dv[sup])
        mr, er = np.frexp(rest)
        ms, es = np.frexp(s_e)
        zz = np.ldexp(mD * mr / ms, eD + er - es)
        z = np.sqrt(zz)
        w = av[sup] * np.sqrt(Dv[sup] / rest)
        Rz = -np.outer(w, w)
    np.fill_diagonal(Rz, 1.0)
    lam, U = np.linalg.eigh(Rz)
    # the smallest eigenvalue is at least var/s_e (Rz = L (I - w w'/s_e) L
    # with L >= I and w'w = a'Da), which is what keeps D' positive when
    # the computed eigenvalue has rounded to zero or below
    c = max(float(lam[0]), var / s_e)
    D_new = Dv.copy()
    D_new[sup] = c * z ** 2
    if c < np.finfo(float).tiny:
        # var / s_e underflowed before its product with z^2: prior 1e150,
        # var 1e-180 gave D' = 0 for a representable 2.5e-181 (#669)
        D_new[sup] = np.maximum(float(lam[0]) * zz, var * (zz / s_e))
    if len(sup) > 1:
        E = np.zeros((K, len(sup) - 1))
        E[sup] = (z[:, None] * U[:, 1:]) * np.sqrt(np.maximum(lam[1:] - c,
                                                               0.0))
        cols.append(E)
    V_new = np.hstack(cols)
    r = V_new.shape[1]
    if len(cov) == 4 and r == rho:
        F, Wq = cov[2], cov[3]
    elif r == 0:
        F, Wq = np.zeros((1, 0)), np.ones(1)
    else:
        from .core import qmc_nodes
        F, Wq = qmc_nodes(r, m=11, seed=0)
    return m_new, V_new, D_new, F, Wq, {"exact": True, "rank": r}
