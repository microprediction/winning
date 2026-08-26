"""Block, nested and tree races: clustered covariance in the base package.

Promoted from research/pqrace (validation history in research/pqrace/SCHUR.md:
every kernel checked against Monte Carlo to its noise floor, Jacobians against
finite differences, inversions by round trip). Conventions follow the front
door: MIN-wins abilities (lower mu is better), mean-zero gauge, Gaussian base.

Models (Y is a performance TIME; smallest wins):

    block   Y_i = mu_i + v_i a_{c(i)} + sd_i eps_i          a_c iid N(0,1)
    nested  ... + gamma g_i f                               one global factor
    tree    ... + sum_{t in ancestors(i)} lam_t a_t         hierarchy

Across-cluster independence factorizes the survival field by cluster, so cost
is O(N x lattice x nodes) regardless of the number of blocks; the tree adds
one upward and one downward message pass. The optional `fastrace` extension
accelerates the block field pass transparently.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, roots_hermitenorm

TINY = 1e-300

try:
    import fastrace as _fastrace
    _HAVE_RUST = hasattr(_fastrace, "block_race")
except Exception:                                            # pragma: no cover
    _fastrace = None
    _HAVE_RUST = False


def _cluster_nodes(r, qa, seed=0):
    """Quadrature for a cluster's r-dim effect: GH for r = 1, scrambled
    Sobol for r >= 2 (the node economy validated in research/qpo)."""
    if r == 1:
        an, aw = roots_hermitenorm(qa)
        return an.reshape(-1, 1), aw / aw.sum()
    from scipy.stats import qmc
    from scipy.special import ndtri
    m = max(4, int(np.ceil(np.log2(qa ** r))))
    u = qmc.Sobol(r, scramble=True, seed=seed).random_base2(min(m, 10))
    nodes = ndtri(np.clip(u, 1e-9, 1 - 1e-9))
    return nodes, np.full(len(nodes), 1.0 / len(nodes))


def _block_max_r(mu, sd, cluster, V, points, qa):
    """Max-wins kernel for RANK-R blocks: loading matrix V (n, r), each
    cluster with its own independent r-dim effect. Conditional independence
    given the effect makes the field a per-cluster r-dim quadrature."""
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    V = np.atleast_2d(np.asarray(V, float))
    if V.shape[0] != len(mu):
        V = V.T
    r = V.shape[1]
    cluster = np.asarray(cluster)
    n = len(mu)
    _, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, V_o, c_o = mu[order], sd[order], V[order], inv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(c_o) != 0])
    nodes, w = _cluster_nodes(r, qa)
    Q = len(nodes)
    if _HAVE_RUST and hasattr(_fastrace, "block_race_r"):
        p_o = np.asarray(_fastrace.block_race_r(
            np.ascontiguousarray(mu_o), np.ascontiguousarray(sd_o),
            np.ascontiguousarray(V_o), starts.astype(np.int64),
            np.ascontiguousarray(nodes), np.ascontiguousarray(w), points))
        p = np.empty(n); p[order] = p_o
        return np.maximum(p, 0.0)
    tot = np.sqrt(sd_o ** 2 + (V_o ** 2).sum(1))
    lo = float((mu_o - 8 * tot).min()); hi = float((mu_o + 8 * tot).max())
    x = np.linspace(lo, hi, points); dx = x[1] - x[0]
    shift = V_o @ nodes.T                                   # (n, Q)
    z = (x[None, None, :] - mu_o[:, None, None]
         - shift[:, :, None]) / sd_o[:, None, None]
    logF = np.log(np.maximum(ndtr(z), TINY))
    pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))
    S = np.add.reduceat(logF, starts, axis=0)
    G = np.einsum("q,cql->cl", w, np.exp(np.minimum(S, 0.0)))
    logG = np.log(np.maximum(G, TINY))
    rest = np.exp(np.minimum(logG.sum(axis=0)[None, :] - logG, 0.0))
    h = np.einsum("q,nql->nl", w, pdf * np.exp(np.minimum(S[c_o] - logF, 0.0)))
    p_o = (h * rest[c_o]).sum(axis=1) * dx
    p = np.empty(n); p[order] = p_o
    return np.maximum(p, 0.0)


def _block_max(mu, sd, cluster, v, points, qa):
    """Max-wins kernel (numpy reference); public functions negate."""
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    v = np.asarray(v, float); cluster = np.asarray(cluster)
    if v.ndim == 2 and v.shape[-1] > 1:
        return _block_max_r(mu, sd, cluster, v, points, qa)
    v = v.reshape(len(mu))
    n = len(mu)
    _, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, v_o = mu[order], sd[order], v[order]
    starts = np.flatnonzero(np.r_[True, np.diff(inv[order]) != 0])
    an, aw = roots_hermitenorm(qa); aw = aw / aw.sum()
    if _HAVE_RUST:
        p_o = np.asarray(_fastrace.block_race(
            np.ascontiguousarray(mu_o), np.ascontiguousarray(sd_o),
            np.ascontiguousarray(v_o), starts.astype(np.int64),
            np.ascontiguousarray(an), np.ascontiguousarray(aw), points))
    else:
        tot = np.sqrt(sd_o ** 2 + v_o ** 2)
        lo = float((mu_o - 8 * tot).min()); hi = float((mu_o + 8 * tot).max())
        x = np.linspace(lo, hi, points); dx = x[1] - x[0]
        z = (x[None, None, :] - mu_o[:, None, None]
             - v_o[:, None, None] * an[None, :, None]) / sd_o[:, None, None]
        logF = np.log(np.maximum(ndtr(z), TINY))
        pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))
        S = np.add.reduceat(logF, starts, axis=0)
        G = np.einsum("q,cql->cl", aw, np.exp(np.minimum(S, 0.0)))
        logG = np.log(np.maximum(G, TINY))
        rest = np.exp(np.minimum(logG.sum(axis=0)[None, :] - logG, 0.0))
        c_o = inv[order]
        h = np.einsum("q,nql->nl", aw,
                      pdf * np.exp(np.minimum(S[c_o] - logF, 0.0)))
        p_o = (h * rest[c_o]).sum(axis=1) * dx
    p = np.empty(n); p[order] = p_o
    return np.maximum(p, 0.0)


def block_race_probabilities(mu, cluster, loading, D, points=257, qa=9):
    """P(i wins | min-wins) under one private factor per cluster.

    loading[i] is member i's exposure to its cluster's shared effect;
    D[i] its idiosyncratic VARIANCE (as everywhere in winning.factor)."""
    p = _block_max(-np.asarray(mu, float), np.sqrt(np.asarray(D, float)),
                   cluster, loading, points, qa)
    t = p.sum()
    return p / t if t > 0 else p


def nested_race_probabilities(mu, cluster, loading, D, coupling=None,
                              gamma=1.0, points=257, qa=9, qf=15):
    """Block race plus one global factor with per-runner loadings `coupling`;
    gamma interpolates from independent blocks (0) to full coupling (1)."""
    if coupling is None or gamma == 0.0:
        return block_race_probabilities(mu, cluster, loading, D,
                                        points=points, qa=qa)
    mu = np.asarray(mu, float)
    g = np.atleast_2d(np.asarray(coupling, float))
    if g.shape[0] != len(mu):
        g = g.T
    if g.shape[1] == 1:
        fn, fw = roots_hermitenorm(qf)
        fn = fn.reshape(-1, 1); fw = fw / fw.sum()
    else:
        fn, fw = _cluster_nodes(g.shape[1], qf, seed=1)
    p = np.zeros(len(mu))
    for q in range(len(fn)):
        p += fw[q] * block_race_probabilities(mu + gamma * (g @ fn[q]), cluster,
                                              loading, D, points=points, qa=qa)
    t = p.sum()
    return p / t if t > 0 else p


def block_race_jacobian(mu, cluster, loading, D, points=257, qa=9):
    """Exact d p / d mu (min-wins), one pass. Rows sum to zero."""
    mu = np.asarray(mu, float)
    m = -mu
    sd = np.sqrt(np.asarray(D, float))
    v = np.asarray(loading, float); cluster = np.asarray(cluster)
    n = len(mu)
    _, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, v_o, c_o = m[order], sd[order], v[order], inv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(c_o) != 0])
    n_c = len(starts)
    an, aw = roots_hermitenorm(qa); aw = aw / aw.sum()
    tot = np.sqrt(sd_o ** 2 + v_o ** 2)
    lo = float((mu_o - 8 * tot).min()); hi = float((mu_o + 8 * tot).max())
    x = np.linspace(lo, hi, points); dx = x[1] - x[0]
    z = (x[None, None, :] - mu_o[:, None, None]
         - v_o[:, None, None] * an[None, :, None]) / sd_o[:, None, None]
    logF = np.log(np.maximum(ndtr(z), TINY))
    pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))
    S = np.add.reduceat(logF, starts, axis=0)
    G = np.einsum("q,cql->cl", aw, np.exp(np.minimum(S, 0.0)))
    logG = np.log(np.maximum(G, TINY))
    logG_all = logG.sum(axis=0)
    Rc = np.exp(np.minimum(logG_all[None, :] - logG, 0.0))
    h = np.einsum("q,nql->nl", aw, pdf * np.exp(np.minimum(S[c_o] - logF, 0.0)))
    Gall = np.exp(np.minimum(logG_all, 0.0))
    U = h * Rc[c_o] / np.sqrt(np.maximum(Gall, TINY))[None, :] * np.sqrt(dx)
    J = -(U @ U.T)
    for ci in range(n_c):
        a0 = starts[ci]; a1 = starts[ci + 1] if ci + 1 < n_c else n
        idx = np.arange(a0, a1)
        if len(idx) == 1:
            continue
        lo2 = np.exp(np.minimum(S[ci][None, None, :, :]
                                - logF[idx][:, None, :, :]
                                - logF[idx][None, :, :, :], 0.0))
        term = np.einsum("q,ijql,l->ij", aw,
                         pdf[idx][:, None, :, :] * pdf[idx][None, :, :, :] * lo2,
                         Rc[ci]) * dx
        J[np.ix_(idx, idx)] = -term
    np.fill_diagonal(J, 0.0)
    J[np.arange(n), np.arange(n)] = -J.sum(axis=1)
    Jf = np.empty((n, n)); Jf[np.ix_(order, order)] = J
    # chain rule for the front-door convention: p_min(mu) = p_max(-mu), so
    # d p_min / d mu = -J_max(-mu). The kernel above computed J_max at -mu.
    return -Jf


def abilities_from_block_race(p, cluster, loading, D, points=257, qa=9,
                              tol=1e-10, max_iter=25):
    """Invert the block race: centred min-wins mu with probabilities p.

    Sub-resolution targets are BOUNDS, not measurements: they are floored at
    max(1e-14, min-positive/1000) and the returned abilities for those
    entries are upper bounds on quality (lower bounds on mu)."""
    p_t = np.asarray(p, float); p_t = p_t / p_t.sum()
    n = len(p_t)
    floor = max(1e-14, p_t[p_t > 0].min() * 1e-3)
    p_t = np.maximum(p_t, floor); p_t = p_t / p_t.sum()
    lt = np.log(p_t)
    ones = np.ones((n, n)) / n
    forward = lambda m: block_race_probabilities(m, cluster, loading, D,
                                                 points=points, qa=qa)
    # adaptive fixed point into Newton's basin
    mu = -(lt - lt.mean())
    eta = 1.0
    lp = np.log(np.maximum(forward(mu), TINY))
    err = np.abs(lp - lt).max()
    for _ in range(200):
        if err < 0.2:
            break
        mu_n = mu - eta * (lt - lp); mu_n -= mu_n.mean()
        lp_n = np.log(np.maximum(forward(mu_n), TINY))
        e_n = np.abs(lp_n - lt).max()
        if e_n < err:
            mu, lp, err = mu_n, lp_n, e_n
            eta = min(eta * 1.2, 1.5)
        else:
            eta *= 0.5
            if eta < 1e-4:
                break
    for it in range(max_iter):
        pv = np.maximum(forward(mu), TINY); pv = pv / pv.sum()
        r = np.log(pv) - lt
        cur = np.abs(r).max()
        if cur < tol:
            return mu - mu.mean(), float(cur), it
        J = block_race_jacobian(mu, cluster, loading, D, points=points, qa=qa)
        Jl = J / pv[:, None]
        step, *_ = np.linalg.lstsq(Jl + ones, -r, rcond=1e-12)
        nn = np.linalg.norm(step)
        if nn > 5.0:
            step *= 5.0 / nn
        for _ in range(8):
            mu_n = mu + step; mu_n -= mu_n.mean()
            p_n = np.maximum(forward(mu_n), TINY); p_n = p_n / p_n.sum()
            if np.abs(np.log(p_n) - lt).max() < cur:
                mu = mu_n
                break
            step *= 0.5
    pv = np.maximum(forward(mu), TINY); pv = pv / pv.sum()
    return mu - mu.mean(), float(np.abs(np.log(pv) - lt).max()), max_iter


def tree_race_probabilities(mu, cluster, loading, D, parent, strength,
                            points=257, qa=9):
    """P(i wins | min-wins) under a hierarchy of uniform shared effects.

    Leaf clusters keep per-member loadings; each internal node t (indices
    continuing past the leaf clusters) applies strength[t] uniformly to every
    leaf beneath it; parent[t] gives the tree (root's parent = -1). Two
    message passes on the lattice; see research/pqrace/SCHUR.md for the
    validation history (MC to noise floor, common-root invariance 3e-7)."""
    mu = np.asarray(mu, float)
    m = -mu
    sd = np.sqrt(np.asarray(D, float))
    v = np.asarray(loading, float); cluster = np.asarray(cluster)
    parent = np.asarray(parent, int); lam = np.asarray(strength, float)
    n = len(m); nT = len(parent)
    _, inv = np.unique(cluster, return_inverse=True)
    nC = inv.max() + 1
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, v_o, c_o = m[order], sd[order], v[order], inv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(c_o) != 0])
    an, aw = roots_hermitenorm(qa); aw = aw / aw.sum()
    depth_shift = np.zeros(nT)
    for t in range(nT):
        s_, u = 0.0, t
        while parent[u] >= 0:
            s_ += abs(lam[parent[u]]); u = parent[u]
        depth_shift[t] = s_
    tot = np.sqrt(sd_o ** 2 + v_o ** 2)
    pad = 8.0 + 3.5 * (depth_shift[:nC].max() if nC < nT else 0.0)
    lo = float((mu_o - pad * np.maximum(tot, 1.0)).min())
    hi = float((mu_o + pad * np.maximum(tot, 1.0)).max())
    x = np.linspace(lo, hi, points); dx = x[1] - x[0]
    z = (x[None, None, :] - mu_o[:, None, None]
         - v_o[:, None, None] * an[None, :, None]) / sd_o[:, None, None]
    logF = np.log(np.maximum(ndtr(z), TINY))
    pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))
    S = np.add.reduceat(logF, starts, axis=0)
    G = np.empty((nT, points))
    G[:nC] = np.einsum("q,cql->cl", aw, np.exp(np.minimum(S, 0.0)))
    children = [[] for _ in range(nT)]
    root = -1
    for t in range(nT):
        if parent[t] >= 0:
            children[parent[t]].append(t)
        else:
            root = t
    shift_eval = lambda g, delta: np.interp(x, x - delta, g, left=g[0], right=g[-1])
    for t in sorted(range(nC, nT), key=lambda u: -depth_shift[u]):
        acc = np.zeros(points)
        for q in range(qa):
            prod = np.ones(points)
            for c in children[t]:
                prod = prod * shift_eval(G[c], lam[t] * an[q])
            acc += aw[q] * prod
        G[t] = np.maximum(acc, 0.0)
    R = np.ones((nT, points))
    for t in sorted(range(nT), key=lambda u: depth_shift[u]):
        pa = parent[t]
        if pa < 0:
            continue
        sm = np.zeros(points)
        for q in range(qa):
            sm += aw[q] * shift_eval(R[pa], -lam[pa] * an[q])
        prod = np.ones(points)
        for s_ in children[pa]:
            if s_ != t:
                prod = prod * G[s_]
        R[t] = np.maximum(sm * prod, 0.0)
    h = np.einsum("q,nql->nl", aw, pdf * np.exp(np.minimum(S[c_o] - logF, 0.0)))
    p_o = (h * R[c_o]).sum(axis=1) * dx
    p = np.empty(n); p[order] = p_o
    p = np.maximum(p, 0.0)
    t_ = p.sum()
    return p / t_ if t_ > 0 else p
