"""Argmax probabilities of a Gaussian Markov chain on a lattice.

    P(X_t = max_s X_s) = int f_t(u) P(X_<t < u | X_t=u) P(X_>t < u | X_t=u) du

Every lattice threshold u_k is run at once: forward messages and
backward survivals are L x L arrays (row k = threshold k), so each step
is one dense matrix product and the whole vector costs O(n L^3).

Two details decide the accuracy:
  * transitions are integrated over each cell (normal CDF differences),
    not sampled as pdf * spacing;
  * a path value in the same cell as the threshold counts as below it
    with weight 1/2. A hard cut (weight 0) drops the near-tie mass and
    the probabilities sum to visibly less than one.
The result is NOT renormalised; its sum is reported as a check.

Chain: X_1 ~ N(m_1, s_1^2), X_{t+1} | X_t ~ N(a_t + b_t X_t, c_t^2).
"""
import numpy as np
from scipy.special import ndtr


def _cell_edges(s):
    ds = s[1] - s[0]
    return np.concatenate([[-np.inf], (s[:-1] + s[1:]) / 2, [np.inf]]), ds


def _transition(s, edges, a, b, c):
    """T[i, j] = P(X' in cell j | X = s_i)."""
    z = (edges[None, :] - (a + b * s)[:, None]) / c
    return np.diff(ndtr(z), axis=1)


def chain_argmax(m1, s1, a, b, c, L=400, lo=None, hi=None):
    """Argmax probabilities for a chain of n = len(a) + 1 variables."""
    a, b, c = (np.asarray(v, float) for v in (a, b, c))
    n = len(a) + 1
    s = np.linspace(lo, hi, L)
    edges, _ = _cell_edges(s)
    pi1 = np.diff(ndtr((edges - m1) / s1))
    T = [_transition(s, edges, a[t], b[t], c[t]) for t in range(n - 1)]
    mask = np.tril(np.ones((L, L)), -1) + 0.5 * np.eye(L)   # [k, j]
    diag = np.arange(L)

    # forward: alpha[t, k] = P(X_<t "<" u_k, X_t in cell k)
    alpha = np.empty((n, L))
    alpha[0] = pi1
    msg = pi1[None, :] * mask
    for t in range(1, n):
        nxt = msg @ T[t - 1]
        alpha[t] = nxt[diag, diag]
        msg = nxt * mask
    # backward: beta[t, k] = P(X_>t "<" u_k | X_t = u_k)
    beta = np.empty((n, L))
    H = np.ones((L, L))
    beta[n - 1] = 1.0
    for t in range(n - 2, -1, -1):
        H = (H * mask) @ T[t].T
        beta[t] = H[diag, diag]
    return (alpha * beta).sum(axis=1)


def ar1_argmax(phi, n, L=400, g=8.0):
    sd = np.sqrt(1 - phi ** 2)
    return chain_argmax(0.0, 1.0, np.zeros(n - 1), np.full(n - 1, phi),
                        np.full(n - 1, sd), L=L, lo=-g, hi=g)


def walk_argmax(n, L=400, g=6.0):
    w = g * np.sqrt(n)
    return chain_argmax(0.0, 1.0, np.zeros(n - 1), np.ones(n - 1),
                        np.ones(n - 1), L=L, lo=-w, hi=w)


def walk_argmax_exact(n):
    """Sparre Andersen: p_t = q_{t-1} q_{n-t}, q_k = C(2k,k)/4^k."""
    q = np.ones(n)
    for k in range(1, n):
        q[k] = q[k - 1] * (2 * k - 1) / (2 * k)
    return np.array([q[t] * q[n - 1 - t] for t in range(n)])
