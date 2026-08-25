"""One-factor conditional envelope kernel (numba) for max-wins Gaussian races.

    U = mu + eps,  eps ~ N(0, Sigma_c),  Sigma_c = b b' + R,  R = A A'
    U_i = c_i + b_i Z,   c = mu + eta,  eta = A z,  Z ~ N(0,1)

Conditional on eta the race is the upper envelope of n lines in Z; the
conditional winner shares q_i(eta) = Phi(tau_k) - Phi(tau_{k-1}) are exact,
and the conditional Jacobian d q / d mu is the path Laplacian with edge
weights phi(tau) / |b_j - b_i| over adjacent envelope segments.

The algorithm is the one in research/qpo/envelope.py (validated there against
brute force); this file re-implements the inner loop in numba because the
experiment needs ~1e5-1e6 envelope evaluations per problem.  Slopes are
sorted once per (b) rather than once per draw, which requires strictly
increasing slopes: `prepare_slopes` jitters exact ties.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

_SQRT2 = math.sqrt(2.0)
_PHI = 1.0 / math.sqrt(2.0 * math.pi)


@njit(cache=True, fastmath=False)
def _ndtr(x):
    return 0.5 * math.erfc(-x / _SQRT2)


@njit(cache=True, fastmath=False)
def envelope_batch(bs, order, C, q_out, J, want_J):
    """Conditional shares (and accumulated Laplacian) for a batch of draws.

    bs:    (n,) slopes sorted strictly ascending
    order: (n,) original index of the k-th sorted slope
    C:     (M, n) intercepts in ORIGINAL index order
    q_out: (M, n) output conditional shares in original index order (zeroed here)
    J:     (n, n) Laplacian accumulator, added to in place if want_J
    Returns the total number of envelope segments over the batch.
    """
    M, n = C.shape
    stack_i = np.empty(n, dtype=np.int64)
    stack_t = np.empty(n, dtype=np.float64)
    total_segments = 0
    for m in range(M):
        for i in range(n):
            q_out[m, i] = 0.0
        top = 0
        for k in range(n):
            ck = C[m, order[k]]
            bk = bs[k]
            while top > 0:
                j = stack_i[top - 1]
                t = (C[m, order[j]] - ck) / (bk - bs[j])
                if top > 1 and t <= stack_t[top - 2]:
                    top -= 1
                    continue
                stack_t[top - 1] = t
                break
            stack_i[top] = k
            top += 1
        total_segments += top
        prev = -np.inf
        for s in range(top):
            if s < top - 1:
                hi = stack_t[s]
            else:
                hi = np.inf
            lo_cdf = 0.0 if prev == -np.inf else _ndtr(prev)
            hi_cdf = 1.0 if hi == np.inf else _ndtr(hi)
            q_out[m, order[stack_i[s]]] = hi_cdf - lo_cdf
            prev = hi
        if want_J:
            for s in range(top - 1):
                i = order[stack_i[s]]
                jn = order[stack_i[s + 1]]
                t = stack_t[s]
                w = _PHI * math.exp(-0.5 * t * t) / (bs[stack_i[s + 1]] - bs[stack_i[s]])
                J[i, i] += w
                J[jn, jn] += w
                J[i, jn] -= w
                J[jn, i] -= w
    return total_segments


def prepare_slopes(b, rel_jitter: float = 1e-9):
    """Sorted, strictly increasing slopes and the index order. Ties are broken
    by a deterministic jitter far below any variance that matters."""
    b = np.asarray(b, dtype=float).copy()
    scale = max(float(np.max(np.abs(b))), 1e-300)
    order = np.argsort(b, kind="stable")
    bs = b[order]
    for _ in range(3):
        d = np.diff(bs)
        bad = d <= 0
        if not np.any(bad):
            break
        bs[1:][bad] = bs[:-1][bad] + rel_jitter * scale
        bs = np.maximum.accumulate(bs + np.arange(len(bs)) * 0.0)
        # ensure strictly increasing after fixing runs of ties
        for k in range(1, len(bs)):
            if bs[k] <= bs[k - 1]:
                bs[k] = bs[k - 1] + rel_jitter * scale
    return bs, order


class OneFactorRace:
    """Sigma_c = b b' + R with R = A A'; conditional shares via the envelope."""

    def __init__(self, Sigma_c, direction: str = "leading"):
        S = 0.5 * (np.asarray(Sigma_c, dtype=float) + np.asarray(Sigma_c, dtype=float).T)
        w, U = np.linalg.eigh(S)
        idx = np.argsort(w)[::-1]
        w, U = w[idx], U[:, idx]
        k = 0 if direction == "leading" else 1
        self.b = U[:, k] * np.sqrt(max(w[k], 0.0))
        self.eigenvalue = float(w[k])
        self.trace_fraction = float(max(w[k], 0.0) / max(w.sum(), 1e-300))
        R = S - np.outer(self.b, self.b)
        ew, EU = np.linalg.eigh(0.5 * (R + R.T))
        ew = np.maximum(ew, 0.0)
        self.A = EU * np.sqrt(ew)          # symmetric square root of R
        self.R = R
        self.n = len(self.b)
        self.bs, self.order = prepare_slopes(self.b)

    def conditional_shares(self, mu, eta, want_J: bool = False):
        """q (M, n) for intercepts mu + eta, eta of shape (M, n); J summed over draws."""
        C = np.ascontiguousarray(np.asarray(mu, dtype=float)[None, :] + eta)
        q = np.empty_like(C)
        J = np.zeros((self.n, self.n)) if want_J else np.zeros((1, 1))
        envelope_batch(self.bs, self.order, C, q, J, want_J)
        return (q, J) if want_J else q

    def eta_from_z(self, z):
        """eta = A z for z (M, n) standard normal."""
        return z @ self.A.T

    def rb_shares(self, mu, M: int, seed: int = 0, chunk: int = 2048,
                  want_J: bool = False, return_se: bool = True):
        rng = np.random.default_rng(seed)
        acc = np.zeros(self.n)
        sq = np.zeros(self.n)
        Jacc = np.zeros((self.n, self.n)) if want_J else None
        done = 0
        while done < M:
            m = min(chunk, M - done)
            eta = self.eta_from_z(rng.standard_normal((m, self.n)))
            if want_J:
                q, J = self.conditional_shares(mu, eta, want_J=True)
                Jacc += J
            else:
                q = self.conditional_shares(mu, eta)
            acc += q.sum(axis=0)
            sq += (q * q).sum(axis=0)
            done += m
        p = acc / M
        se = np.sqrt(np.maximum(sq / M - p ** 2, 0.0) / M)
        out = [p]
        if return_se:
            out.append(se)
        if want_J:
            out.append(Jacc / M)
        return tuple(out) if len(out) > 1 else p


def raw_winner_shares(mu, L, M: int, seed: int = 0, chunk: int = 4096):
    """Plain winner counting for U = mu + L z (max wins)."""
    mu = np.asarray(mu, dtype=float)
    n = mu.size
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        U = mu[None, :] + rng.standard_normal((m, n)) @ L.T
        np.add.at(counts, np.argmax(U, axis=1), 1)
        done += m
    p = counts / M
    return p, np.sqrt(np.maximum(p * (1 - p), 0) / M)


def sym_sqrt(S, floor: float = 0.0):
    S = 0.5 * (np.asarray(S, dtype=float) + np.asarray(S, dtype=float).T)
    w, U = np.linalg.eigh(S)
    return (U * np.sqrt(np.maximum(w, floor))) @ U.T


def project(S):
    S = np.asarray(S, dtype=float)
    row = S.mean(axis=1)
    return S - row[:, None] - row[None, :] + float(row.mean())
