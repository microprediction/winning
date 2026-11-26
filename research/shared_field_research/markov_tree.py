"""Research prototype: all min-winner shares on a Gaussian Markov tree.

Each vertex is a contestant (including internal vertices). Standardized
marginals Z_i are N(0,1); Z_i | Z_parent is N(rho_i Z_parent, 1-rho_i**2).
Unlike winning.factor.structures.Tree, this describes sparse PRECISION,
not additive common-ancestor covariance. See README for derivation.
"""
import numpy as np
from scipy.special import roots_legendre


def _pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _tree(parent, rho, n):
    p = np.asarray(parent, dtype=float)
    r = np.asarray(rho, dtype=float)
    if (p.shape != (n,) or not np.all(np.isfinite(p)) or
            np.any(p != np.floor(p)) or np.any((p < -1) | (p >= n))):
        raise ValueError('parent must contain one integer index per node')
    p = p.astype(int)
    roots = np.flatnonzero(p == -1)
    if (len(roots) != 1 or r.shape != (n,) or not np.all(np.isfinite(r)) or
            np.any(np.abs(r) >= 1) or r[roots[0]] != 0):
        raise ValueError('one root, root rho=0, all other |rho| < 1 required')
    children = [[] for _ in range(n)]
    for i, j in enumerate(p):
        if j >= 0:
            children[j].append(i)
    order = [int(roots[0])]
    for i in order:
        order.extend(children[i])
    if len(order) != n:
        raise ValueError('parent must describe a connected acyclic tree')
    return p, r, children, order


def tree_shares(mu, sd, parent, rho, *, L=64, M=48, span=9.):
    """Raw, nonnegative min-win shares; no forced normalization.

    L: outer Gauss-Legendre threshold order; M: inner order per edge.
    span: truncate every standardized marginal to [-span, span].
    Refuses singular edges, zero sds and malformed input. Extreme
    correlations or very unequal scales may require much larger L,M.
    Returned sum is a useful diagnostic but not an error certificate.
    """
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    n = mu.size
    if (mu.shape != (n,) or n == 0 or sd.shape != (n,) or
            not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sd)) or
            np.any(sd <= 0)):
        raise ValueError('finite mean vector and strictly positive sd vector required')
    p, r, children, order = _tree(parent, rho, n)
    if (any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
            or k < 2 for k in (L, M)) or not np.isfinite(span) or span <= 0):
        raise ValueError('L,M must be integers >=2; span positive and finite')
    if n == 1:
        return np.array([1.])
    # Center and normalize the common utility unit; do not add jitter.
    scale = np.max(sd)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        mu = (mu - mu[order[0]]) / scale
        sd = sd / scale
    if not np.all(np.isfinite(mu)) or np.any(sd == 0):
        raise ValueError('utility dynamic range exceeds prototype arithmetic')
    a = np.min(mu - span * sd)
    b = np.min(mu + span * sd)
    if not np.isfinite(a) or not np.isfinite(b) or not b > a:
        raise ValueError('collapsed or nonfinite threshold window')
    # Two nested quadratures: threshold x shared across all nodes,
    # and a threshold-adapted interval for each standardized vertex.
    qx, wx = roots_legendre(L)
    q, qw = roots_legendre(M)
    x = a + (b-a)/2 * (1+qx)
    wx = wx * (b-a)/2
    cut = (x[:, None] - mu) / sd
    lo = np.clip(cut, -span, span)
    width = (span-lo)/2
    y = lo[:, :, None] + width[:, :, None] * (1+q)
    wy = width[:, :, None] * qw
    # Last node evaluates the cavity at the winner's exact boundary;
    # it is not included in integration over other vertices.
    y = np.concatenate([y, lo[:, :, None]], axis=2)
    one = np.ones((L, M+1))

    def transfer(src, dst, incoming, edge_r):
        s = np.sqrt((1-edge_r)*(1+edge_r))
        z = (y[:, src, None, :M] - edge_r*y[:, dst, :, None])/s
        return ((_pdf(z)/s) @ (incoming[:, :M]*wy[:, src, :])[:, :, None])[:, :, 0]

    # Upward cavity from each child's side, conditioned on its parent.
    up = [None]*n
    for i in reversed(order[1:]):
        field = one.copy()
        for c in children[i]:
            field *= up[c]
        up[i] = transfer(i, p[i], field, r[i])
    # Downward cavities. Prefix/suffix products avoid 0/0 and quadratic
    # work at high-degree vertices. Standard normal edge reversibility
    # supplies the same conditional kernel in the other direction.
    down = [None]*n
    down[order[0]] = one
    shares = np.empty(n)
    for i in order:
        kids = children[i]
        prefix = [down[i]]
        for c in kids:
            prefix.append(prefix[-1]*up[c])
        within = np.abs(cut[:, i]) <= span
        shares[i] = wx @ (_pdf(lo[:, i])/sd[i] * within * prefix[-1][:, -1])
        suffix = one.copy()
        for j in range(len(kids)-1, -1, -1):
            c = kids[j]
            down[c] = transfer(i, c, prefix[j]*suffix, r[c])
            suffix *= up[c]
        down[i] = None
        for c in kids:
            up[c] = None
    if not np.all(np.isfinite(shares)):
        raise FloatingPointError('nonfinite quadrature result')
    return shares


def tree_covariance(sd, parent, rho):
    """Dense covariance for small verification cases, not needed to price."""
    sd = np.asarray(sd, float)
    n = len(sd)
    p, r, _, order = _tree(parent, rho, n)
    A = np.zeros((n, n))
    for i in order:
        if p[i] < 0:
            A[i, i] = 1
        else:
            A[i] = r[i]*A[p[i]]
            A[i, i] = np.sqrt((1-r[i])*(1+r[i]))
    A *= sd[:, None]
    return A @ A.T
