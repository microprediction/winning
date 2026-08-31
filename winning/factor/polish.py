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

from .races import (race_probabilities, abilities_from_race, _setup,
                    forward_grid)


def _raw_and_derivative(mu, V, D, F, W, fn, sd, x, dx, rows=None):
    """Unnormalized lattice masses a_i and their exact derivative on THIS
    grid: A[i, j] = d a_i / d mu_j.

    Off-diagonal (j != i) the mu_j derivative hits the survival factor,
    d S_j / d mu_j = +f_j, giving the photo-finish integrand
    int f_i f_j prod_{k != i,j} S_k. It is factored stably as
    [f_i prod_{k != i} S_k] * [f_j / S_j]: the first is bounded by f, the
    second is the hazard, which grows only polynomially (the symmetric
    square-root split overflows where S vanishes).

    On the diagonal the derivative hits the density instead,
    d f_i / d mu_i = -f'_i / sd_i^2, so A_ii is INTEGRATED, not imposed
    by a zero-row-sum identity. In the continuum the two agree; on a
    finite lattice they do not, and the difference is exactly what makes
    a zero-sum-imposed diagonal a continuum surrogate rather than the
    derivative of the sum that was computed.

    rows: restrict to these i (the score needs one row). Returns
    (a, A[rows], dT) with a the full mass vector and dT = d(1'a)/dmu, the
    column sums of the FULL A, computed in O(nL) rather than by forming
    the matrix: with G = sum_i g_i the lattice winner density,

        dT/dmu_j = int f_j (G - g_j)/S_j dx - int f'_j/sd_j^2 R_j dx,

    so one restricted row still gets the exact quotient correction.
    """
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    idx = np.arange(n) if rows is None else np.atleast_1d(rows)
    a = np.zeros(n)
    A = np.zeros((len(idx), n))
    dT = np.zeros(n)
    for q in range(len(F)):
        z = (x[None, :] - M_all[q][:, None]) / sd[:, None]
        S, f, fp = fn(z)
        f = f / sd[:, None]
        logS = np.log(np.maximum(S, 1e-300))
        logf = np.log(np.maximum(f, 1e-300))
        L = logS.sum(axis=0)
        # g_i = f_i prod_{k != i} S_k -- the winner integrand itself
        g = np.exp(np.clip(logf + L[None, :] - logS, -745.0, 40.0))
        haz = np.exp(np.clip(logf - logS, -745.0, 40.0))
        R = np.exp(np.clip(L[None, :] - logS, -745.0, 40.0))
        own = -(fp / (sd ** 2)[:, None] * R)                   # d f_i/d mu_i
        a += W[q] * g.sum(axis=1) * dx
        A += W[q] * (g[idx] @ haz.T) * dx
        # own coordinate, differentiated rather than imposed by zero rows
        A[np.arange(len(idx)), idx] += W[q] * dx * (
            own[idx].sum(axis=1) - (g[idx] * haz[idx]).sum(axis=1))
        Gt = g.sum(axis=0)
        dT += W[q] * dx * (((Gt[None, :] - g) * haz).sum(axis=1)
                           + own.sum(axis=1))
    return a, A, dT


def race_jacobian(mu, V=None, D=None, F=None, W=None, base="normal",
                  points=501, window="bulk", delta=1e-12):
    """Fixed-grid-exact J[i, j] = d p_i / d mu_j, one field pass.

    Exact for the normalized rectangle sum CONDITIONAL ON THE SELECTED
    GRID: the lattice comes from races.forward_grid, the same window and
    refinement race_probabilities uses, so both integrate over the same
    x, and the normalization is differentiated rather than assumed away.
    With raw masses a, total T = 1'a and p = a/T,

        J = (A - p 1'A) / T,        A = da/dmu,

    the quotient rule. What it deliberately omits is differentiation of
    the adaptive window itself (the grid-motion term), which is
    measurable on very coarse lattices (6e-4 at L=25) and falls below
    the reported agreement threshold (3e-11) at production resolutions
    (L >= 101). In the continuum 1'A vanishes and rows sum to zero on
    their own; on a lattice they do not, and the residual is the
    difference between a continuum surrogate and the fixed-grid
    derivative of the function being optimized.

    Raising mu_j (slower j, min-wins) raises every other p_i:
    off-diagonals are positive.
    """
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    x, points = forward_grid(M_all, sd, V, fn, left, right, points,
                             window=window, delta=delta)
    dx = x[1] - x[0]
    a, A, dT = _raw_and_derivative(mu, V, D, F, W, fn, sd, x, dx)
    T = float(a.sum())
    p = a / T
    return (A - np.outer(p, dT)) / T


def race_jacobian_row(mu, y, V=None, D=None, F=None, W=None, base="normal",
                      points=257, window="bulk", delta=1e-12):
    """One row of the race Jacobian: d p_y / d mu_j for all j, one field
    pass (the estimation score needs only the observed alternative's row,
    not the full matrix).

    Same construction as race_jacobian restricted to i = y, including
    the shared lattice and the quotient rule, so the score it feeds is
    the fixed-grid-exact gradient of the objective the forward map
    defines (grid motion omitted; see race_jacobian)."""
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    x, points = forward_grid(M_all, sd, V, fn, left, right, points,
                             window=window, delta=delta)
    dx = x[1] - x[0]
    a, A, dT = _raw_and_derivative(mu, V, D, F, W, fn, sd, x, dx,
                                   rows=[int(y)])
    T = float(a.sum())
    p_y = float(a[int(y)]) / T
    return (A[0] - p_y * dT) / T


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
                base="normal", points=257, name_caps=None, groups=None,
                A=None, b=None, tol=1e-9, max_iter=60, structure=None):
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
    if structure is not None:
        forward, jac, invert = _structure_engines(structure, points)
    else:
        forward = lambda m: race_probabilities(m, V=V, D=D, F=F, W=W,
                                               base=base, points=points)
        jac = lambda m: race_jacobian(m, V=V, D=D, F=F, W=W, base=base,
                                      points=points)
        invert = lambda p: abilities_from_race(np.asarray(p, float), V=V,
                                               D=D, F=F, W=W, base=base,
                                               points=points)
    if mu0 is None:
        if p0 is None:
            raise ValueError("give p0 or mu0")
        mu0 = invert(np.asarray(p0, float))
    mu0 = np.asarray(mu0, float) - np.mean(mu0)
    n = len(mu0)
    A0, b0 = concentration_matrix(n, name_caps=name_caps, groups=groups)
    if A is not None:
        A0 = np.vstack([A0, np.atleast_2d(A)])
        b0 = np.concatenate([b0, np.atleast_1d(b)])
    if len(b0) == 0:
        return forward(mu0), mu0, {"active": [], "nit": 0}

    def p_of(m):
        return forward(m)

    def cons_f(m):
        return b0 - A0 @ p_of(m)

    def cons_j(m):
        return -A0 @ jac(m)

    res = minimize(lambda m: 0.5 * np.sum((m - mu0) ** 2), mu0,
                   jac=lambda m: m - mu0, method="SLSQP",
                   constraints=[
                       NonlinearConstraint(cons_f, 0.0, np.inf, jac=cons_j),
                       LinearConstraint(np.ones((1, n)), 0.0, 0.0)],
                   options={"maxiter": max_iter, "ftol": tol})
    mu = res.x - res.x.mean()
    p = p_of(mu)
    slack = b0 - A0 @ p
    if -slack.min() > 1e-6:
        # the analytic Jacobian may be approximate (tree: cross-cluster
        # Gram); if SLSQP converged infeasible, restore feasibility with
        # exact finite-difference constraint gradients from the current
        # point -- the forward map is always exact.
        def cons_j_fd(m, h=1e-6):
            Jn = np.empty((n, n))
            for j in range(n):
                e = np.zeros(n); e[j] = h
                Jn[:, j] = (p_of(m + e) - p_of(m - e)) / (2 * h)
            return -A0 @ Jn
        res = minimize(lambda m: 0.5 * np.sum((m - mu0) ** 2), mu,
                       jac=lambda m: m - mu0, method="SLSQP",
                       constraints=[
                           NonlinearConstraint(cons_f, 0.0, np.inf,
                                               jac=cons_j_fd),
                           LinearConstraint(np.ones((1, n)), 0.0, 0.0)],
                       options={"maxiter": max_iter, "ftol": tol})
        mu = res.x - res.x.mean()
        p = p_of(mu)
        slack = b0 - A0 @ p
    return p, mu, {"active": list(np.flatnonzero(slack < 1e-6)),
                   "nit": int(res.nit), "max_violation": float(-min(slack.min(), 0.0)),
                   "mu_distance": float(np.linalg.norm(mu - mu0))}


def _structure_engines(structure, points):
    """(forward, jacobian, invert) for a declarative covariance structure."""
    from .structures import Independent, Factor, Blocks, Nested, Tree
    from .blocks import (block_race_probabilities, nested_race_probabilities,
                         block_race_jacobian, nested_race_jacobian,
                         abilities_from_block_race)
    if isinstance(structure, (Independent, Factor)):
        V = None if isinstance(structure, Independent) else np.asarray(structure.V, float)
        D = np.asarray(structure.D, float)
        return (lambda m: race_probabilities(m, V=V, D=D, points=points),
                lambda m: race_jacobian(m, V=V, D=D, points=points),
                lambda p: abilities_from_race(p, V=V, D=D, points=points))
    if isinstance(structure, Blocks):
        c, L, D = structure.cluster, structure.loading, structure.D
        return (lambda m: block_race_probabilities(m, c, L, D, points=points),
                lambda m: block_race_jacobian(m, c, L, D, points=points),
                lambda p: abilities_from_block_race(p, c, L, D, points=points)[0])
    if isinstance(structure, Nested):
        c, L, D, g, ga = (structure.cluster, structure.loading, structure.D,
                          structure.coupling, structure.gamma)
        return (lambda m: nested_race_probabilities(m, c, L, D, coupling=g,
                                                    gamma=ga, points=points),
                lambda m: nested_race_jacobian(m, c, L, D, coupling=g,
                                               gamma=ga, points=points),
                None or (lambda p: _invert_generic(
                    p, lambda m: nested_race_probabilities(
                        m, c, L, D, coupling=g, gamma=ga, points=points))))
    if isinstance(structure, Tree):
        from .blocks import tree_race_probabilities, tree_race_jacobian
        c, L, D, pa, lam = (structure.cluster, structure.loading,
                            structure.D, structure.parent, structure.strength)
        fwd = lambda m: tree_race_probabilities(m, c, L, D, pa, lam,
                                                points=points)
        return (fwd,
                lambda m: tree_race_jacobian(m, c, L, D, pa, lam,
                                             points=points),
                lambda p: _invert_generic(p, fwd))
    raise TypeError(f"polish_race: unknown structure "
                    f"{type(structure).__name__}")


def _invert_generic(p, forward, tol=1e-9, max_iter=400):
    p = np.asarray(p, float); p = p / p.sum()
    lt = np.log(np.maximum(p, 1e-300))
    mu = -(lt - lt.mean())
    eta = 1.0
    lp = np.log(np.maximum(forward(mu), 1e-300))
    err = np.abs(lp - lt).max()
    for _ in range(max_iter):
        if err < tol:
            break
        mu_n = mu - eta * (lt - lp); mu_n -= mu_n.mean()
        lp_n = np.log(np.maximum(forward(mu_n), 1e-300))
        e = np.abs(lp_n - lt).max()
        if e < err:
            mu, lp, err = mu_n, lp_n, e
            eta = min(eta * 1.2, 1.5)
        else:
            eta *= 0.5
            if eta < 1e-4:
                break
    return mu
