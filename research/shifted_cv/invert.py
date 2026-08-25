"""Share inversion with fixed coupled draws (sample-average Newton).

Jacobian: by default the EXACT Jacobian of the calibrated reference race,
J_0(nu*), computed once (run_jacobian.py shows it is closer to J_Sigma(mu*)
than any envelope estimate from hundreds of draws, and the coupled Jacobian
correction of Section 13 does not improve on it).  Pass J_fixed=None to use
the per-iteration target envelope Laplacian on M_J fixed draws instead.

    r(mu) = W_Sigma(mu) - p*   estimated by  r_hat(mu) = mean_m c_m(mu)
    J delta = -r_hat            on 1-perp
    mu <- mu + gamma delta

The draws z (residual) and z_J (Jacobian) are fixed across iterations, so
r_hat is a deterministic function of mu and the iteration solves the
sample-average approximation's root problem. The Jacobian is the averaged
one-factor envelope Laplacian of the TARGET race on the z_J draws, the same
for every residual estimator, so that the comparison isolates the residual
estimator. For one-hot estimators r_hat is piecewise constant in mu and the
iteration is a damped fixed-point rather than a true Newton method; it stops
when the L1 residual stops decreasing.
"""

from __future__ import annotations

import time

import numpy as np

from estimators import combine


def solve_centered(J, r, lam, marquardt: float = 0.1):
    """delta on 1-perp with (J + marquardt diag(J) + lam P) delta = -r."""
    n = len(r)
    P = np.eye(n) - 1.0 / n
    A = J + marquardt * np.diag(np.diag(J)) + lam * P + np.ones((n, n)) / n
    d = np.linalg.solve(A, -(r - r.mean()))
    return d - d.mean()


def newton_invert(method, p_star, mu0, M: int, seed: int, M_J: int = 128,
                  max_iter: int = 40, tol_l1: float = 1e-5, gamma0: float = 1.0,
                  beta=None, jac_every: int = 1, step_cap=None, verbose: bool = False,
                  J_fixed=None, noise_mult: float = 2.0):
    t = method.t if hasattr(method, "t") else method.members[0].t
    n = t.n
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((M, n))
    z0 = rng.standard_normal((M, n))
    zJ = rng.standard_normal((M_J, n))
    mu = np.asarray(mu0, dtype=float).copy()
    mu -= mu.mean()
    if step_cap is None:
        step_cap = 1.0 * t.problem.scale

    def resid(m):
        """Soft-thresholded residual: components within noise_mult standard
        errors of zero are treated as zero, so the Newton step only acts on
        residual components the draws actually resolve."""
        raw, ctrls = method.parts(m, z, z0)
        c = combine(raw, ctrls, beta)
        r = c.mean(axis=0) - p_star
        v = np.maximum(c.var(axis=0), 0.0)
        se = np.sqrt(v / M)
        rt = np.sign(r) * np.maximum(np.abs(r) - noise_mult * se, 0.0)
        rt[v == 0.0] = 0.0                  # never observed in any race: no information
        return rt, float(se.sum())

    t0 = time.time()
    r, se_l1 = resid(mu)
    l1 = float(np.abs(r).sum())
    hist = [l1]
    reason = "max_iter"
    samples = M
    J = None
    it = 0
    gamma = gamma0
    stalls = 0
    for it in range(1, max_iter + 1):
        if l1 < tol_l1:
            reason = "tol"
            break
        if l1 <= 0.0:
            reason = "noise_floor"
            break
        if hasattr(method, "on_iterate"):
            method.on_iterate(mu)
            r, se_l1 = resid(mu)          # the control may have been re-anchored
            l1 = float(np.abs(r).sum())
        if J_fixed is not None:
            J = J_fixed
        elif J is None or (it - 1) % jac_every == 0:
            _, J = t.rb.conditional_shares(mu, t.rb.eta_from_z(zJ), want_J=True)
            J /= M_J
            samples += M_J
        lam = 1e-2 * np.trace(J) / n
        delta = solve_centered(J, r, lam)
        delta = np.clip(delta, -step_cap, step_cap)
        accepted = False
        g = gamma
        for _ in range(4):
            mu_try = mu + g * delta
            mu_try -= mu_try.mean()
            r_try, se_try = resid(mu_try)
            samples += M
            l1_try = float(np.abs(r_try).sum())
            if l1_try < l1:
                mu, r, l1, se_l1 = mu_try, r_try, l1_try, se_try
                accepted = True
                break
            g *= 0.5
        hist.append(l1)
        if not accepted:
            stalls += 1
            if stalls >= 2:
                reason = "stalled"
                break
        else:
            stalls = 0
    return {"mu": mu, "iterations": it, "l1_hist": hist, "final_l1": l1, "final_se_l1": se_l1,
            "seconds": time.time() - t0, "samples": samples, "reason": reason,
            "converged": reason in ("tol", "noise_floor")}


def recovery_metrics(mu_hat, problem, M_fresh: int = 20000, seed: int = 777):
    """Ability recovery and fresh-simulation share errors (RB estimate at mu_hat)."""
    mu_star = problem.mu_star
    e = mu_hat - mu_star
    p_fresh, se = problem_rb(problem).rb_shares(mu_hat, M_fresh, seed=seed)
    d = p_fresh - problem.p_star
    ident = problem.p_star > 1e-3
    return {"rmse_mu": float(np.sqrt(np.mean(e ** 2))),
            "rmse_mu_ident": float(np.sqrt(np.mean(e[ident] ** 2))) if ident.any() else np.nan,
            "n_ident": int(ident.sum()),
            "max_abs_mu": float(np.abs(e).max()),
            "corr_mu": float(np.corrcoef(mu_hat, mu_star)[0, 1]),
            "share_l1": float(np.abs(d).sum()), "share_linf": float(np.abs(d).max()),
            "share_l1_floor": float(se.sum())}


_rb_cache = {}


def problem_rb(problem):
    k = problem.key
    if k not in _rb_cache:
        from envelope_fast import OneFactorRace
        _rb_cache[k] = OneFactorRace(problem.Sigma_c)
    return _rb_cache[k]
