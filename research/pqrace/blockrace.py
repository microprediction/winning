"""The block race: exact win probabilities under clustered covariance.

    Y_i = mu_i + v_i * a_{c(i)} + sd_i * eps_i,
    a_c ~ N(0,1) independent across clusters, eps independent.

Sigma is block-structured: within cluster c, cov(Y_i, Y_j) = v_i v_j;
across clusters, zero (plus diagonal). This is the geometry a global
low-rank factorization represents worst (many small blocks), and it
factorizes EXACTLY: across-cluster independence splits the field product
by cluster,

    G(x) = prod_c G_c(x),      G_c(x) = E_a[ prod_{j in c} F_j(x - v_j a) ],

so each cluster needs one 1-d quadrature, not a joint dimension. The winner's
own cluster is handled by leave-one-out INSIDE its block -- a cavity division
at the cluster level (the Schur move: condition on the block, integrate it
out):

    p_i = int dx sum_a w_a f_i(x,a) exp(S_{c(i)}(x,a) - logF_i(x,a))
                        * prod_{c' != c(i)} G_{c'}(x)

Cost O(N * L * Q_a): the same order as a rank-1 race, for arbitrarily many
blocks. A global factor on top would add one outer quadrature dimension.
"""
import numpy as np
from scipy.special import ndtr, roots_hermitenorm

TINY = 1e-300


def block_race(mu, sd, cluster, v, points=257, qa=9, span=8.0):
    """p_i = P(Y_i = max_j Y_j) under the nested-effects model.

    mu, sd, v : (N,) means, idiosyncratic sds, cluster-effect loadings
    cluster   : (N,) int cluster ids (singletons allowed: loading irrelevant)
    """
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    v = np.asarray(v, float); cluster = np.asarray(cluster)
    N = len(mu)
    _, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, v_o, c_o = mu[order], sd[order], v[order], inv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(c_o) != 0])
    nC = len(starts)

    tot = np.sqrt(sd_o ** 2 + v_o ** 2)
    lo = float((mu_o - span * tot).min()); hi = float((mu_o + span * tot).max())
    x = np.linspace(lo, hi, points); dx = x[1] - x[0]

    an, aw = roots_hermitenorm(qa)
    aw = aw / aw.sum()

    # logF[j, q, l] = log Phi((x_l - mu_j - v_j a_q) / sd_j)
    z = (x[None, None, :] - mu_o[:, None, None] - v_o[:, None, None] * an[None, :, None]) / sd_o[:, None, None]
    logF = np.log(np.maximum(ndtr(z), TINY))                       # (N, qa, L)
    pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))

    S = np.add.reduceat(logF, starts, axis=0)                      # (nC, qa, L)
    G = np.einsum("q,cql->cl", aw, np.exp(np.minimum(S, 0.0)))     # (nC, L), S<=0
    logG = np.log(np.maximum(G, TINY))
    logG_all = logG.sum(axis=0)                                    # (L,)
    rest = logG_all[None, :] - logG                                # (nC, L) leave-cluster-out

    # per member: within-cluster leave-one-out at each (a, x)
    Sc = S[c_o]                                                    # (N, qa, L)
    inner = pdf * np.exp(np.minimum(Sc - logF, 0.0))               # (N, qa, L)
    inner = np.einsum("q,nql->nl", aw, inner)                      # (N, L)
    p_o = (inner * np.exp(np.minimum(rest[c_o], 0.0))).sum(axis=1) * dx
    p = np.empty(N); p[order] = p_o
    return np.maximum(p, 0.0)


def nested_race(mu, sd, cluster, v, g=None, gamma=1.0, points=257, qa=9,
                qf=15, span=8.0):
    """Depth-2 Schur race: global coupling factor over correlated blocks.

        Y_i = mu_i + gamma * g_i * f + v_i * a_{c(i)} + sd_i * eps_i

    Sigma = gamma^2 g g' + block-diagonal + D: blocks whose CROSS-covariance
    is carried by one rank-1 term -- the race analogue of the Schur
    complementary portfolio construction, where sub-problems stay separable
    and the inter-block coupling enters through a low-rank adjustment.
    gamma interpolates: 0 = independent blocks (the hierarchical/Harville
    end), 1 = the full nested covariance (the Markowitz end).

    Cost: qf outer nodes x the block_race field assembly. Recursing the same
    move (blocks of blocks, one coupling factor per split) gives the tree
    race at O(N L Q log C); this is the first rung.
    """
    from scipy.stats import qmc
    from scipy.special import ndtri
    mu = np.asarray(mu, float)
    if g is None or gamma == 0.0:
        return block_race(mu, sd, cluster, v, points=points, qa=qa, span=span)
    g = np.asarray(g, float)
    fn, fw = roots_hermitenorm(qf)
    fw = fw / fw.sum()
    p = np.zeros(len(mu))
    for q in range(qf):
        p += fw[q] * block_race(mu + gamma * g * fn[q], sd, cluster, v,
                                points=points, qa=qa, span=span)
    return p


def block_abilities_from_probabilities(p, sd, cluster, v, g=None, gamma=1.0,
                                       points=257, qa=9, qf=15, tol=1e-8,
                                       max_iter=200, eta0=1.0):
    """Invert the block/nested race: find centred mu with p(mu) = p.

    The map mu -> log p is smooth and diagonally dominant (raising one
    ability chiefly raises its own win probability), so a damped log-space
    fixed point converges:

        mu <- mu + eta * (log p_target - log p(mu)),   recentred each step,

    with eta halved on any step that worsens the residual (the same scheme
    the winning package's coordinate inversion reduces to when the forward
    map is treated as a black box). Identification: p is invariant to a
    common shift, so mu is returned centred -- the contrast is the estimand,
    exactly as everywhere else in this programme.
    """
    p = np.asarray(p, float)
    p = p / p.sum()
    lt = np.log(np.maximum(p, 1e-300))
    mu = np.log(p) - np.log(p).mean()          # Luce start
    if g is not None and gamma != 0.0:
        def forward(m):
            return nested_race(m, sd, cluster, v, g=g, gamma=gamma,
                               points=points, qa=qa, qf=qf)
    else:
        def forward(m):
            return block_race(m, sd, cluster, v, points=points, qa=qa)
    eta = eta0
    lp = np.log(np.maximum(forward(mu), 1e-300))
    err = np.abs(lp - lt).max()
    for _ in range(max_iter):
        if err < tol:
            break
        mu_new = mu + eta * (lt - lp)
        mu_new -= mu_new.mean()
        lp_new = np.log(np.maximum(forward(mu_new), 1e-300))
        err_new = np.abs(lp_new - lt).max()
        if err_new < err:
            mu, lp, err = mu_new, lp_new, err_new
            eta = min(eta * 1.2, 1.5)
        else:
            eta *= 0.5
            if eta < 1e-4:
                break
    return mu, err


def block_race_jac(mu, sd, cluster, v, points=257, qa=9, span=8.0):
    """Win probabilities AND the exact Jacobian d p / d mu, from one pass.

    The Jacobian inherits the Schur structure of the model:
      same block   J_ij = -int dx sum_a w_a f_i f_j exp(S_c - lF_i - lF_j) R_c(x)
                   (i and j share the block effect at each node)
      cross block  J_ij = -int dx h_i(x) h_j(x) G(x) / (G_c(x) G_d(x))
                   -- a GRAM over lattice points: block-diagonal plus a
                   factored coupling, so Newton can solve blocks locally and
                   correct through the shared field. Rows sum to zero (a
                   common shift moves nothing), which sets the diagonal.
    """
    mu = np.asarray(mu, float); sd = np.asarray(sd, float)
    v = np.asarray(v, float); cluster = np.asarray(cluster)
    N = len(mu)
    _, inv = np.unique(cluster, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    mu_o, sd_o, v_o, c_o = mu[order], sd[order], v[order], inv[order]
    starts = np.flatnonzero(np.r_[True, np.diff(c_o) != 0])
    nC = len(starts)
    tot = np.sqrt(sd_o ** 2 + v_o ** 2)
    lo = float((mu_o - span * tot).min()); hi = float((mu_o + span * tot).max())
    x = np.linspace(lo, hi, points); dx = x[1] - x[0]
    an, aw = roots_hermitenorm(qa); aw = aw / aw.sum()

    z = (x[None, None, :] - mu_o[:, None, None] - v_o[:, None, None] * an[None, :, None]) / sd_o[:, None, None]
    logF = np.log(np.maximum(ndtr(z), TINY))
    pdf = np.exp(-0.5 * z * z) / (sd_o[:, None, None] * np.sqrt(2 * np.pi))
    S = np.add.reduceat(logF, starts, axis=0)
    G = np.einsum("q,cql->cl", aw, np.exp(np.minimum(S, 0.0)))
    logG_all = np.log(np.maximum(G, TINY)).sum(axis=0)
    Rc = np.exp(np.minimum(logG_all[None, :] - np.log(np.maximum(G, TINY)), 0.0))  # (nC, L)

    lo_i = np.exp(np.minimum(S[c_o] - logF, 0.0))            # leave-one-out (N, qa, L)
    h = np.einsum("q,nql->nl", aw, pdf * lo_i)               # (N, L)
    p_o = (h * Rc[c_o]).sum(axis=1) * dx

    # cross-block Gram: U_i(x) = h_i sqrt(dx G_all) / G_c(i)
    Gall = np.exp(np.minimum(logG_all, 0.0))
    U = h * np.sqrt(np.maximum(Gall, TINY) * dx)[None, :] / np.maximum(G[c_o], TINY)
    J = -(U @ U.T)
    # replace same-block entries with the exact shared-node term
    for ci in range(nC):
        a0 = starts[ci]; a1 = starts[ci + 1] if ci + 1 < nC else N
        idx = np.arange(a0, a1)
        if len(idx) == 1:
            J[a0, a0] = 0.0
            continue
        lo2 = np.exp(np.minimum(S[ci][None, None, :, :] - logF[idx][:, None, :, :]
                                - logF[idx][None, :, :, :], 0.0))
        term = np.einsum("q,ijql,l->ij", aw, pdf[idx][:, None, :, :] * pdf[idx][None, :, :, :]
                         * lo2, Rc[ci]) * dx
        Jb = -term
        J[np.ix_(idx, idx)] = Jb
    np.fill_diagonal(J, 0.0)
    J[np.arange(N), np.arange(N)] = -J.sum(axis=1)
    # un-permute
    p = np.empty(N); p[order] = p_o
    Jf = np.empty((N, N)); Jf[np.ix_(order, order)] = J
    return np.maximum(p, 0.0), Jf


def block_invert_newton(p_target, sd, cluster, v, points=257, qa=9,
                        tol=1e-10, max_iter=30):
    """Newton inversion using the exact Jacobian. Gauge fixed by working on
    the centred subspace (J's rows sum to zero; add the ones-projector)."""
    p_t = np.asarray(p_target, float); p_t = p_t / p_t.sum()
    N = len(p_t)
    mu = np.log(np.maximum(p_t, 1e-300)); mu -= mu.mean()
    ones = np.ones((N, N)) / N
    lt = np.log(np.maximum(p_t, 1e-300))
    # globalize with the ADAPTIVE fixed point (backtracking eta) until inside
    # Newton's basin -- measured: Newton contracts to machine precision from
    # |mu err| <= 0.5, and an un-damped fixed point can diverge outright
    # under strong correlation, which is why the globalizer must backtrack
    mu, _ = block_abilities_from_probabilities(p_t, sd, cluster, v,
                                               points=points, qa=qa,
                                               tol=0.2, max_iter=60)
    for it in range(max_iter):
        p, J = block_race_jac(mu, sd, cluster, v, points=points, qa=qa)
        p = np.maximum(p / p.sum(), 1e-300)
        r = np.log(p) - lt                       # log residual: even rows scale
        if np.abs(r).max() < tol:
            return mu - mu.mean(), float(np.abs(r).max()), it
        Jl = J / p[:, None]                      # d log p / d mu
        step, *_ = np.linalg.lstsq(Jl + ones, -r, rcond=1e-12)
        n = np.linalg.norm(step)
        if n > 5.0:                              # trust region on the step
            step *= 5.0 / n
        cur = np.abs(r).max()
        for _ in range(12):                      # damp but never abort
            mu_n = mu + step; mu_n -= mu_n.mean()
            p_n = block_race(mu_n, sd, cluster, v, points=points, qa=qa)
            p_n = np.maximum(p_n / p_n.sum(), 1e-300)
            if np.abs(np.log(p_n) - lt).max() < cur:
                mu = mu_n
                break
            step *= 0.5
        else:
            mu = mu + step * (2.0 ** 0)          # take the tiny step anyway
            mu -= mu.mean()
    p, _ = block_race_jac(mu, sd, cluster, v, points=points, qa=qa)
    p = np.maximum(p / p.sum(), 1e-300)
    return mu - mu.mean(), float(np.abs(np.log(p) - lt).max()), max_iter
