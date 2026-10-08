"""Exact Gaussian posteriors that stay in the race grammar.

The race verbs price N(m, diag(d) + W W') in time linear in the number
of entrants, so a belief that keeps that form after every observation
can be priced after every observation. Three updates do keep it, and
are implemented here exactly (papers/exact_pom, Section 3):

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
    s02 = s0v ** 2
    if nr == 0:
        return m0v.copy(), s02.copy(), np.zeros((K, rho))
    tau = s02 + Dv / nr
    k = s02 / tau
    d = 1.0 / (1.0 / s02 + nr / Dv)
    H = nr * np.eye(rho) + (Vm.T / tau) @ Vm
    W = (k[:, None] * Vm) @ _psd_inv_sqrt(H)
    r = yb - m0v
    fhat = np.linalg.solve(H, Vm.T @ (r / tau)) if rho else np.zeros(0)
    m = m0v + k * (r - Vm @ fhat)
    return m, d, W


def crn_posterior_replicates(V, D, Y, s0, m0=None):
    """Exact CRN posterior when entrants were run on different replicates.

    Y is (n_entrants x R): column r holds replicate r's outputs, NaN for
    an entrant not run on it (same model as crn_posterior; replicate r
    shares one F_r across every entrant it ran). Returns (m, d, W),
    theta ~ N(m, diag(d) + W W') EXACTLY.

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
    """
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
    obs = ~np.isnan(Ym)
    keep = obs.any(axis=0)
    obs, Yz = obs[:, keep], np.where(obs, Ym, 0.0)[:, keep]
    s02 = s0v ** 2
    if obs.shape[1] == 0:
        return m0v.copy(), s02.copy(), np.zeros((K, rho))
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
    return m, 1.0 / a, W


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
    lam = cnt / sn ** 2
    r = 1.0 / (1.0 + d0v * lam)
    rho = Vm.shape[1]
    H = np.eye(rho) + (Vm.T * (lam * r)) @ Vm
    W = (r[:, None] * Vm) @ _psd_inv_sqrt(H)
    d = d0v * r
    h = sm / sn ** 2
    m = d * h + W @ (W.T @ h)
    return m, d, W

