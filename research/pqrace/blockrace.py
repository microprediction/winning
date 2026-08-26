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
