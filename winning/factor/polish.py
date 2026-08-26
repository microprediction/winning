"""Polish a race onto linear constraints: the concentration primitive.

The portfolio-facing problem (allocation's "transport"): weights ARE race
probabilities, and finance imposes linear constraints on them -- a cap per
name, a cap per sector/group, a floor. Clipping and renormalising breaks
model-consistency; the right object is the NEAREST RACE satisfying the
constraints:

    minimise   ||mu - mu0||^2   over abilities mu (mean-zero gauge)
    subject to a_k . p(mu) <= b_k          for each cap (>= for floors)

where p(mu) = race_probabilities(mu, V, D, ...). The result is exactly a
race of the same model -- same covariance story, same base -- with the
concentration limits active only where they bind. Everything runs on the
exact Jacobian dp/dmu, assembled from the same shared-field pass as the
race itself.

Conventions follow races.py: MIN-wins abilities, base in {normal, gumbel},
V/D/F/W as in race_probabilities.
"""
from __future__ import annotations

import numpy as np

from .races import race_probabilities, abilities_from_race, _setup


def race_jacobian(mu, V=None, D=None, F=None, W=None, base="normal",
                  points=501):
    """Exact J[i, j] = d p_i / d mu_j for the general race, one field pass.

    Off-diagonal, per factor node, J is a GRAM over the lattice:
        J_ij = sum_a w_a int f_i f_j exp(L - logS_i - logS_j) dx,
    with L = sum_k logS_k -- so per node it is U U' with
    U = f * exp(L/2 - logS). Rows sum to zero (a common shift moves
    nothing), which sets the diagonal. Raising mu_j (slower j, min-wins)
    raises every other p_i: off-diagonals are positive.
    """
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    x = np.linspace(M_all.min() - left * sd.max(),
                    M_all.max() + right * sd.max(), points)
    dx = x[1] - x[0]
    J = np.zeros((n, n))
    chunk = max(1, int(5e6 / (n * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S, f, _ = fn(z)
        f = f / sd[None, :, None]
        logS = np.log(S)
        logf = np.log(np.maximum(f, 1e-300))
        L = logS.sum(axis=1)                                   # (chunk, points)
        # pair integrand f_i f_j prod_{k != i,j} S_k, factored STABLY as
        # [f_i prod_{k != i} S_k] * [f_j / S_j]: the first is bounded by f,
        # the second is the hazard, which grows only polynomially. (The
        # symmetric square-root split overflows where S vanishes.)
        P1 = np.exp(np.clip(logf + L[:, None, :] - logS, -745.0, 40.0))
        P2 = np.exp(np.clip(logf - logS, -745.0, 40.0))
        for q in range(P1.shape[0]):
            J += Wc[q] * (P1[q] @ P2[q].T) * dx
    total = race_probabilities(np.asarray(mu, float), V=V, D=D, F=F, W=W,
                               base=base, points=points)
    # J currently holds the off-diagonal integrals (its diagonal entries are
    # int f_i^2 e^{L - 2 logS_i}, which are NOT dp_i/dmu_i): overwrite the
    # diagonal from the zero-row-sum identity, then normalise as p is.
    np.fill_diagonal(J, 0.0)
    np.fill_diagonal(J, -J.sum(axis=1))
    # p was normalised by its lattice total; apply the same projection:
    # d(p/T)/dmu = (J - p 1'J)/T with T ~ 1; the correction is second order
    # and the zero-sum structure is already exact, so return J as is.
    return J


def concentration_matrix(n, name_caps=None, groups=None):
    """Assemble (A, b) rows for caps: A p <= b.

    name_caps : scalar or length-n array -- per-name weight caps (NaN/None
                entries skipped)
    groups    : iterable of (indices, cap) -- group/sector concentration caps
    """
    rows, bs = [], []
    if name_caps is not None:
        caps = np.broadcast_to(np.asarray(name_caps, float), (n,))
        for i in range(n):
            if np.isfinite(caps[i]):
                r = np.zeros(n); r[i] = 1.0
                rows.append(r); bs.append(float(caps[i]))
    if groups is not None:
        for idx, cap in groups:
            r = np.zeros(n); r[np.asarray(idx, int)] = 1.0
            rows.append(r); bs.append(float(cap))
    if not rows:
        return np.zeros((0, n)), np.zeros(0)
    return np.vstack(rows), np.asarray(bs)


def polish_race(p0=None, mu0=None, V=None, D=None, F=None, W=None,
                base="normal", points=501, name_caps=None, groups=None,
                A=None, b=None, tol=1e-9, max_iter=60):
    """Nearest race satisfying concentration constraints.

    Give either the current weights p0 (inverted to abilities internally) or
    abilities mu0 directly. Constraints via name_caps/groups (see
    concentration_matrix) and/or explicit (A, b) with A p <= b. Returns
    (p, mu, info): a probability vector that IS the race at mu, satisfying
    the caps, with mu as close to mu0 as the constraints allow.

    Solved by SLSQP on mu with the exact race Jacobian supplying constraint
    gradients A J(mu); the mean-zero gauge is an equality constraint.
    """
    from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint
    if mu0 is None:
        if p0 is None:
            raise ValueError("give p0 or mu0")
        mu0 = abilities_from_race(np.asarray(p0, float), V=V, D=D, F=F, W=W,
                                  base=base, points=points)
    mu0 = np.asarray(mu0, float) - np.mean(mu0)
    n = len(mu0)
    A0, b0 = concentration_matrix(n, name_caps=name_caps, groups=groups)
    if A is not None:
        A0 = np.vstack([A0, np.atleast_2d(A)])
        b0 = np.concatenate([b0, np.atleast_1d(b)])
    if len(b0) == 0:
        p = race_probabilities(mu0, V=V, D=D, F=F, W=W, base=base, points=points)
        return p, mu0, {"active": [], "nit": 0}

    def p_of(m):
        return race_probabilities(m, V=V, D=D, F=F, W=W, base=base,
                                  points=points)

    def cons_f(m):
        return b0 - A0 @ p_of(m)

    def cons_j(m):
        return -A0 @ race_jacobian(m, V=V, D=D, F=F, W=W, base=base,
                                   points=points)

    res = minimize(lambda m: 0.5 * np.sum((m - mu0) ** 2), mu0,
                   jac=lambda m: m - mu0, method="SLSQP",
                   constraints=[
                       NonlinearConstraint(cons_f, 0.0, np.inf, jac=cons_j),
                       LinearConstraint(np.ones((1, n)), 0.0, 0.0)],
                   options={"maxiter": max_iter, "ftol": tol})
    mu = res.x - res.x.mean()
    p = p_of(mu)
    slack = b0 - A0 @ p
    return p, mu, {"active": list(np.flatnonzero(slack < 1e-6)),
                   "nit": int(res.nit), "max_violation": float(-min(slack.min(), 0.0)),
                   "mu_distance": float(np.linalg.norm(mu - mu0))}
