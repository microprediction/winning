"""Exact Gaussian semi-discrete OT in r=2, and the probit side to compare it to.

REDIRECT vs the original spec (Phases II, III, XI-A/B): no external OT solver,
no source mesh, no [-R,R]^2 truncation. In two dimensions the Gaussian measure
of a Laguerre cell has an exact radial form, so the reference is analytic to
~1e-10 instead of mesh-limited to ~1e-4. Since the whole experiment measures
how a tau -> 0 sequence approaches that reference, a mesh-limited reference
would have swamped the signal.

Radial form. A cell is an intersection of half-planes,
    C_i = {x : n_j'x >= c_j},  n_j = v_i - v_j,  c_j = mu_j - mu_i.
Along the ray x = r u(theta), each constraint is linear in r, so C_i meets the
ray in an interval [r_lo(theta), r_hi(theta)] (possibly empty). Since
int_{r_lo}^{r_hi} r e^{-r^2/2} dr = e^{-r_lo^2/2} - e^{-r_hi^2/2},

    gamma_2(C_i) = (1/2pi) int_0^{2pi} [e^{-r_lo^2/2} - e^{-r_hi^2/2}] dtheta,

a one-dimensional integral of a continuous, piecewise-smooth function.

Conventions. Spec: sites v_i, weights w_i = 2 mu_i + |v_i|^2 (verified: race
labels and power-cell labels agree exactly). All comparisons are centered,
since mu and w are identified only up to an additive constant. The Jacobian is
reported in mu-coordinates throughout; dw = 2 dmu, so dp/dw = (1/2) dp/dmu.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

import fastrace
from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes)

_EPS = 1e-12


# ===================================================== hard side (reference)

def weights_from_mu(mu, V):
    return 2 * np.asarray(mu) + (np.asarray(V) ** 2).sum(1)


def mu_from_weights(w, V):
    return (np.asarray(w) - (np.asarray(V) ** 2).sum(1)) / 2.0


def labels_race(X, mu, V):
    return (mu + np.asarray(X) @ np.asarray(V).T).argmax(1)


def labels_laguerre(X, w, V):
    X = np.asarray(X)
    d = ((X[:, None, :] - np.asarray(V)[None]) ** 2).sum(2) - np.asarray(w)
    return d.argmin(1)


def _angles(n_theta):
    th = (np.arange(n_theta) + 0.5) * (2 * np.pi / n_theta)
    return np.column_stack([np.cos(th), np.sin(th)])


def cell_masses(mu, V, n_theta=1 << 18, U=None):
    """Exact gamma_2 of every Laguerre cell, by radial integration."""
    mu = np.asarray(mu, float)
    V = np.asarray(V, float)
    N = len(mu)
    U = _angles(n_theta) if U is None else U
    out = np.empty(N)
    for i in range(N):
        oth = [j for j in range(N) if j != i]
        n = V[i] - V[oth]                       # (N-1, 2)
        c = mu[oth] - mu[i]                     # (N-1,)
        dot = U @ n.T                           # (T, N-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(dot) > _EPS, c / dot, 0.0)
        r_lo = np.maximum(
            0.0, np.where(dot > _EPS, ratio, -np.inf).max(axis=1))
        r_hi = np.where(dot < -_EPS, ratio, np.inf).min(axis=1)
        bad = ((np.abs(dot) <= _EPS) & (c > _EPS)).any(axis=1)
        ok = (~bad) & (r_hi > r_lo)
        f = np.where(ok, np.exp(-0.5 * r_lo**2) - np.exp(-0.5 * np.minimum(
            r_hi, 1e8) ** 2), 0.0)
        out[i] = f.mean()
    return out


def facet_weight(i, j, mu, V):
    """k_ij = (1/|v_i-v_j|) * int_facet phi_2 dH^1, closed form in r=2."""
    n = V[i] - V[j]
    nn = float(np.linalg.norm(n))
    if nn < _EPS:
        return 0.0
    c = mu[j] - mu[i]
    x0 = c * n / nn**2
    d = np.array([-n[1], n[0]]) / nn
    t_lo, t_hi = -np.inf, np.inf
    for k in range(len(mu)):
        if k in (i, j):
            continue
        A = float((V[i] - V[k]) @ d)
        B = float((V[i] - V[k]) @ x0 - (mu[k] - mu[i]))
        if abs(A) < 1e-14:
            if B < 0:
                return 0.0
        elif A > 0:
            t_lo = max(t_lo, -B / A)
        else:
            t_hi = min(t_hi, -B / A)
    if t_hi <= t_lo:
        return 0.0
    return float(norm.pdf(abs(c) / nn) * (norm.cdf(t_hi) - norm.cdf(t_lo)) / nn)


def laplacian_exact(mu, V):
    """dp/dmu at tau = 0: weighted graph Laplacian of the power diagram."""
    N = len(mu)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            K[i, j] = K[j, i] = facet_weight(i, j, mu, V)
    J = -K
    np.fill_diagonal(J, K.sum(axis=1))
    return J


def solve_ot_weights(p_target, V, n_theta=1 << 18, tol=1e-11, max_iter=60):
    """Semi-discrete OT: damped Newton for prescribed Laguerre-cell masses,
    using the exact mass map and its exact Laplacian Hessian (the
    Kitagawa-Merigot-Thibert scheme, specialized to a Gaussian source in 2D).
    Returns centered mu."""
    V = np.asarray(V, float)
    N = len(V)
    U = _angles(n_theta)
    p_target = np.asarray(p_target, float)
    mu = np.zeros(N)
    p = cell_masses(mu, V, U=U)
    res = np.abs(p - p_target).max()
    info = {"iterations": 0, "history": [res]}
    for it in range(max_iter):
        if res < tol:
            break
        J = laplacian_exact(mu, V)
        step = np.linalg.lstsq(J, p - p_target, rcond=1e-10)[0]
        step -= step.mean()
        alpha = 1.0
        for _ in range(40):
            trial = mu - alpha * step
            trial -= trial.mean()
            p_try = cell_masses(trial, V, U=U)
            r_try = np.abs(p_try - p_target).max()
            if r_try < res or alpha < 1e-6:
                break
            alpha *= 0.5
        mu, p, res = trial, p_try, r_try
        info["history"].append(res)
        info["iterations"] = it + 1
    info["residual"] = float(res)
    info["converged"] = bool(res < 1e-8)
    return mu - mu.mean(), info


# ============================================================== soft side

def gh_nodes(r, Q):
    F, W = hermite_nodes(r, Q=Q)
    return np.ascontiguousarray(F), np.ascontiguousarray(W)


def shares_gh(mu, V, D, F, W, points):
    """Package forward map (max-wins); D is the vector of variances."""
    p, _ = fastrace.win_probabilities_factor(
        -np.asarray(mu, float), np.asarray(V, float), np.asarray(D, float),
        F, W, points)
    return p


def shares_mc(mu, V, D, draws=4_000_000, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.sqrt(np.asarray(D, float))
    counts = np.zeros(len(mu))
    done = 0
    while done < draws:
        m = min(1_000_000, draws - done)
        U = (mu + rng.standard_normal((m, np.shape(V)[1])) @ np.asarray(V).T
             + sd * rng.standard_normal((m, len(mu))))
        counts += np.bincount(U.argmax(1), minlength=len(mu))
        done += m
    return counts / draws


def calibrate(p, V, D, F, W, n_iter=80, tol=1e-10, points=4001):
    """Probit inverse (max-wins), Newton with the fastrace forward+slope pass;
    same iteration as winning.factor.core, kernel from kinetics/rustcal."""
    p = np.asarray(p, float)
    p = p / p.sum()
    logp = np.log(p)
    V = np.atleast_2d(np.asarray(V, float))
    D = np.asarray(D, float)
    sd = np.sqrt(D)
    N = len(p)
    ident = p > max(1e-9, 1e-4 / N)
    a = abilities_from_probabilities_factor(
        p, np.zeros((N, 1)), D + (V**2).sum(1), np.zeros((1, 1)), np.ones(1),
        n_iter=n_iter, tol=tol)
    cap = np.sqrt(D + (V**2).sum(1))
    prev, damp, res = np.inf, 1.0, np.inf
    for it in range(n_iter):
        ph, slope, total = fastrace.forward_and_slopes(a, V, D, F, W, points)
        ph = np.maximum(ph, 1e-300)
        resid = np.log(ph) - logp
        res = np.abs(resid[ident]).max()
        if res < tol:
            break
        if res > prev * 1.2:
            damp = max(0.25, damp * 0.5)
        prev = res
        dl = np.minimum((slope / total) / ph, -1e-3 / (sd + 1e-9))
        a = a - np.clip(damp * resid / dl, -cap, cap)
        a -= a.mean()
    return -a - (-a).mean(), {"residual": float(res), "iterations": it + 1}


def jacobian_fd(shares_fn, mu, h):
    N = len(mu)
    J = np.zeros((N, N))
    for j in range(N):
        mp, mm = mu.copy(), mu.copy()
        mp[j] += h
        mm[j] -= h
        J[:, j] = (shares_fn(mp) - shares_fn(mm)) / (2 * h)
    return J


def conditional_pi(X, mu, V, D, Q=48):
    """pi_i(F) = P(i wins | F), the soft cell membership. X is (P, r)."""
    z, wz = np.polynomial.hermite_e.hermegauss(Q)
    wz = wz / np.sqrt(2 * np.pi)
    m = np.asarray(mu) + np.asarray(X) @ np.asarray(V).T      # (P, N)
    s = np.sqrt(np.asarray(D, float))                          # (N,)
    P, N = m.shape
    out = np.zeros((P, N))
    ms = m / s
    for zk, wk in zip(z, wz):
        # arg[p,i,j] = (m[p,i] + s_i zk - m[p,j]) / s_j
        arg = (m + s * zk)[:, :, None] / s[None, None, :] - ms[:, None, :]
        lg = log_ndtr(arg)                    # (P, N, N)
        tot = lg.sum(axis=2) - np.einsum("pii->pi", lg)
        out += wk * np.exp(tot)
    return out / out.sum(axis=1, keepdims=True)


def contrast_eigs(J):
    """Eigenvalues of J restricted to the contrast space (drop the 1-null)."""
    N = J.shape[0]
    B = np.linalg.qr(np.eye(N) - np.ones((N, N)) / N)[0][:, :N - 1]
    return np.linalg.eigvalsh(B.T @ ((J + J.T) / 2) @ B)
