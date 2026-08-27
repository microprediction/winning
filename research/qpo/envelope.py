"""One-factor conditional envelopes for a Gaussian race with arbitrary covariance.

The factor probit in pom.py samples the r-dimensional factor and integrates the
N idiosyncratic dimensions exactly. That needs the conditional covariance to be
diagonal, so it needs Sigma to be low rank plus diagonal. This module does the
opposite, and therefore covers the covariances the other cannot: sample the
(N-1)-dimensional residual exactly, and integrate ONE direction analytically.

    U = mu + eps,   eps ~ N(0, Sigma_c),   Sigma_c = P Sigma P
    Sigma_c = b b' + R                     b from the leading eigenpair
    U_i = c_i + b_i Z,   c_i = mu_i + eta_i,   eta ~ N(0, R),  Z ~ N(0, 1)

Conditional on eta the race is the upper envelope of N straight lines in z, so
the conditional winner probabilities are exact:

    q_{h_k} = Phi(tau_k) - Phi(tau_{k-1})

for the envelope segments h_1..h_K with breakpoints tau. Lines not on the
envelope win with probability zero. Averaging q over residual draws is a
Rao-Blackwellised estimator of p, and unlike the version in pom.py it costs
only O(N log N) per draw on top of the O(N^2) already spent drawing eta.

The conditional Jacobian falls out of the same picture. Only ADJACENT envelope
segments share a boundary, so the photo-finish graph conditional on eta is a
path, and

    J^(m) = sum over boundaries  w_ij (e_i - e_j)(e_i - e_j)',
    w_ij  = phi(tau_ij) / |b_j - b_i|,

a graph Laplacian with N-1 edges at most. It is positive semidefinite with
J 1 = 0 by construction, which is the conditional form of the Jacobian theorem
for these races.

Everything here is checked against brute force in test_envelope.py before use.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_PHI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(x):
    return _PHI * np.exp(-0.5 * np.asarray(x, dtype=float) ** 2)


# --------------------------------------------------------------------------
# the envelope
# --------------------------------------------------------------------------

def upper_envelope(b, c, slope_tol: float = 0.0):
    """Indices and breakpoints of the upper envelope of the lines c_i + b_i z.

    Returns (idx, tau) with idx of length K and tau of length K-1, so that
    line idx[k] is the maximum for tau[k-1] < z < tau[k], with tau[-1] = -inf
    and tau[K-1] = +inf implied.

    Lines are processed in increasing slope; among equal slopes only the
    largest intercept can ever be on the envelope. The stack test pops a line
    when the new line overtakes it at or before the point where it overtook its
    own predecessor, which is the standard upper-hull condition.
    """
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    n = b.size
    if n == 1:
        return np.array([0]), np.empty(0)

    order = np.lexsort((-c, b))              # by slope, then intercept desc
    bs, cs = b[order], c[order]

    # keep only the best intercept within a group of (near-)equal slopes
    if slope_tol > 0:
        keep = np.ones(n, dtype=bool)
        keep[1:] = (bs[1:] - bs[:-1]) > slope_tol
    else:
        keep = np.ones(n, dtype=bool)
        keep[1:] = bs[1:] != bs[:-1]
    order, bs, cs = order[keep], bs[keep], cs[keep]

    stack_i = []          # indices into the filtered arrays
    stack_t = []          # z at which stack_i[k] takes over from stack_i[k-1]
    for k in range(len(bs)):
        while stack_i:
            j = stack_i[-1]
            # z where the new line k overtakes line j (b_k > b_j strictly)
            t = (cs[j] - cs[k]) / (bs[k] - bs[j])
            if stack_t and t <= stack_t[-1]:
                stack_i.pop()
                stack_t.pop()
                continue
            stack_t.append(t)
            break
        stack_i.append(k)
    idx = order[np.asarray(stack_i, dtype=int)]
    return idx, np.asarray(stack_t, dtype=float)


def upper_envelope_batch(b_sorted, C):
    """Upper envelopes for many residual draws at once.

    The slopes b are the same for every draw -- only the intercepts move -- so
    the sort order is shared and the stack algorithm can be run across draws in
    lockstep. That matters: done one draw at a time the envelope costs O(M n)
    Python-level operations and swamps the O(M n^2) of drawing the residuals
    for any n a person would use, which makes the estimator look slow for
    reasons that have nothing to do with the estimator.

    b_sorted must be strictly increasing. C is (M, n) with columns in that same
    order. Returns (stack, tau, top): for draw m, stack[m, :top[m]] are the
    envelope lines left to right and tau[m, t] is where stack[m, t] takes over
    (tau[m, 0] is -inf).
    """
    b = np.asarray(b_sorted, dtype=float)
    C = np.asarray(C, dtype=float)
    M, n = C.shape
    stack = np.zeros((M, n), dtype=np.int64)
    tau = np.full((M, n), -np.inf)
    top = np.zeros(M, dtype=np.int64)
    rows = np.arange(M)

    for k in range(n):
        # pop while the new line overtakes the top one at or before the point
        # where that one overtook its own predecessor
        for _ in range(n):
            live = top > 1
            if not live.any():
                break
            t_idx = np.maximum(top - 1, 0)
            j = stack[rows, t_idx]
            db = b[k] - b[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (C[rows, j] - C[:, k]) / db
            t = np.where(db > 0, t, np.inf)
            pop = live & (t <= tau[rows, t_idx])
            if not pop.any():
                break
            top = top - pop
        t_idx = np.maximum(top - 1, 0)
        j = stack[rows, t_idx]
        db = b[k] - b[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (C[rows, j] - C[:, k]) / db
        t = np.where(top > 0, np.where(db > 0, t, np.inf), -np.inf)
        stack[rows, top] = k
        tau[rows, top] = t
        top = top + 1
    return stack, tau, top


def conditional_shares_batch(b_sorted, C, order=None, n_total=None):
    """Exact conditional winner probabilities for a batch of draws.

    Returns q of shape (M, n) in ORIGINAL index order if `order` is given.
    """
    stack, tau, top = upper_envelope_batch(b_sorted, C)
    M, n = C.shape
    nt = n if n_total is None else n_total
    q = np.zeros((M, nt))
    pos = np.arange(n)
    live = pos[None, :] < top[:, None]                 # (M, n)
    hi = np.where(pos[None, :] + 1 < top[:, None],
                  np.concatenate([tau[:, 1:], np.full((M, 1), np.inf)], axis=1),
                  np.inf)
    seg = np.where(live, ndtr(hi) - ndtr(tau), 0.0)
    cols = stack if order is None else order[stack]
    np.add.at(q, (np.repeat(np.arange(M), n), cols.ravel()), seg.ravel())
    return q, stack, tau, top


def conditional_shares(b, c, slope_tol: float = 0.0, n_total: int | None = None):
    """Exact P(i wins | eta) for the conditional one-dimensional race."""
    n = b.size if n_total is None else n_total
    idx, tau = upper_envelope(b, c, slope_tol)
    q = np.zeros(n)
    lo = np.concatenate([[-np.inf], tau])
    hi = np.concatenate([tau, [np.inf]])
    q[idx] = ndtr(hi) - ndtr(lo)
    return q, idx, tau


def conditional_edges(b, idx, tau):
    """Edges (i, j, w) of the conditional photo-finish Laplacian.

    One edge per envelope boundary, so at most N-1 of them regardless of N.
    """
    b = np.asarray(b, dtype=float)
    if len(idx) < 2:
        return np.empty((0, 2), dtype=int), np.empty(0)
    left, right = idx[:-1], idx[1:]
    w = _norm_pdf(tau) / np.abs(b[right] - b[left])
    return np.stack([left, right], axis=1), w


def laplacian_from_edges(edges, w, n):
    """Dense Laplacian; only for small-n diagnostics."""
    J = np.zeros((n, n))
    for (i, j), wij in zip(edges, w):
        J[i, i] += wij
        J[j, j] += wij
        J[i, j] -= wij
        J[j, i] -= wij
    return J


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------

def project(Sigma):
    """P Sigma P with P = I - 11'/n. Common shocks cannot change the winner."""
    S = np.asarray(Sigma, dtype=float)
    row = S.mean(axis=1)
    return S - row[:, None] - row[None, :] + float(row.mean())


def split_one_factor(Sigma_c, direction: str = "leading", seed: int = 0):
    """Sigma_c = b b' + R. Returns (b, R, info).

    Which direction to integrate is a real question, not a detail: a direction
    with nearly equal loadings shifts every competitor together, leaves the
    winner unchanged, and buys no variance reduction at all. After projecting
    out the common mode the leading eigenvector is the natural first choice
    because it carries the most contrast.
    """
    S = np.asarray(Sigma_c, dtype=float)
    w, U = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    rng = np.random.default_rng(seed)
    if direction == "leading":
        k = 0
    elif direction == "second":
        k = min(1, len(w) - 1)
    elif direction == "random":
        k = int(rng.integers(0, max(1, min(10, len(w)))))
    else:
        raise ValueError(direction)
    b = U[:, k] * np.sqrt(max(w[k], 0.0))
    R = S - np.outer(b, b)
    ew, EU = np.linalg.eigh(0.5 * (R + R.T))
    ew = np.maximum(ew, 0.0)
    A = EU * np.sqrt(ew)                      # R = A A'
    info = {"eigenvalue": float(w[k]),
            "loading_spread": float(np.std(b)),
            "trace_fraction": float(max(w[k], 0.0) / max(w.sum(), 1e-300))}
    return b, R, A, info


def rb_shares_batch(mu, b, A, M: int = 64, seed: int = 0, chunk: int = 512,
                    want_jacobian: bool = False):
    """Rao-Blackwellised shares, vectorised across draws.

    Cost per draw: O(n^2) to form the residual (which raw winner counting pays
    too) plus O(n log n) for the envelope, the latter carried out in numpy
    across the whole chunk. The analytic integration is therefore close to free
    relative to the sampling.
    """
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    order = np.argsort(b, kind="stable")
    b_sorted = np.asarray(b, dtype=float)[order]
    rng = np.random.default_rng(seed)

    acc = np.zeros(n)
    sq = np.zeros(n)
    e_lo, e_hi, e_w = [], [], []
    done = 0
    while done < M:
        m = min(chunk, M - done)
        C = (mu[None, :] + rng.standard_normal((m, A.shape[1])) @ A.T)[:, order]
        q, stack, tau, top = conditional_shares_batch(b_sorted, C, order=order,
                                                      n_total=n)
        acc += q.sum(axis=0)
        sq += (q ** 2).sum(axis=0)
        if want_jacobian:
            pos = np.arange(n - 1)
            live = pos[None, :] + 1 < top[:, None]
            left, right = stack[:, :-1], stack[:, 1:]
            db = np.abs(b_sorted[right] - b_sorted[left])
            with np.errstate(divide="ignore", invalid="ignore"):
                w = np.where(live & (db > 0), _norm_pdf(tau[:, 1:]) / db, 0.0)
            keep = w > 0
            e_lo.append(order[left[keep]])
            e_hi.append(order[right[keep]])
            e_w.append(w[keep])
        done += m
    p = acc / M
    var = np.maximum(sq / M - p ** 2, 0.0) / M
    if not want_jacobian:
        return p, np.sqrt(var)
    lo = np.concatenate(e_lo) if e_lo else np.empty(0, dtype=int)
    hi = np.concatenate(e_hi) if e_hi else np.empty(0, dtype=int)
    ww = np.concatenate(e_w) if e_w else np.empty(0)
    key = np.minimum(lo, hi) * n + np.maximum(lo, hi)
    uniq, inv = np.unique(key, return_inverse=True)
    agg = np.zeros(len(uniq))
    np.add.at(agg, inv, ww)
    edges = np.stack([uniq // n, uniq % n], axis=1).astype(int)
    return p, np.sqrt(var), edges, agg / M


def rb_shares(mu, b, A, M: int = 64, seed: int = 0, slope_tol: float = 0.0,
              want_jacobian: bool = False, chunk: int = 256):
    """Rao-Blackwellised winner shares, and optionally the Jacobian.

    One residual draw costs O(N^2) for the matmul and O(N log N) for the
    envelope, so the analytic integration is nearly free relative to the
    sampling that raw Monte Carlo also has to do.
    """
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    sq = np.zeros(n)
    Jrows = {} if want_jacobian else None
    done = 0
    while done < M:
        m = min(chunk, M - done)
        E = rng.standard_normal((m, A.shape[1])) @ A.T        # (m, n)
        for t in range(m):
            c = mu + E[t]
            q, idx, tau = conditional_shares(b, c, slope_tol, n_total=n)
            acc += q
            sq += q * q
            if want_jacobian:
                edges, w = conditional_edges(b, idx, tau)
                for (i, j), wij in zip(edges, w):
                    Jrows[(i, j)] = Jrows.get((i, j), 0.0) + wij
        done += m
    p = acc / M
    var = np.maximum(sq / M - p ** 2, 0.0) / M
    if not want_jacobian:
        return p, np.sqrt(var)
    edges = np.array(list(Jrows.keys()), dtype=int).reshape(-1, 2)
    w = np.array([Jrows[tuple(e)] for e in edges]) / M
    return p, np.sqrt(var), edges, w


def raw_shares(mu, Sigma_sqrt, M: int = 64, seed: int = 0, chunk: int = 4096):
    """Ordinary winner counting from the same kind of draws, for comparison."""
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        U = mu[None, :] + rng.standard_normal((m, Sigma_sqrt.shape[1])) @ Sigma_sqrt.T
        np.add.at(counts, np.argmax(U, axis=1), 1)
        done += m
    p = counts / M
    return p, np.sqrt(np.maximum(p * (1 - p), 0) / M)


def apply_laplacian(edges, w, v):
    """L v for the accumulated edge list, without forming L."""
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v)
    if len(edges) == 0:
        return out
    i, j = edges[:, 0], edges[:, 1]
    diff = w * (v[i] - v[j])
    np.add.at(out, i, diff)
    np.add.at(out, j, -diff)
    return out
