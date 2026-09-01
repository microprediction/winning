"""Seed experiment: heterogeneous-buyer sourcing market cleared by the
production-style diagonal solver.

n suppliers post shadow prices psi (min-wins: lower delivered cost wins
the unit of demand). Buyer type b sees delivered cost psi + delta_b +
correlated noise (rank-one factor: a common freight/market shock), and
allocates its demand as the win-probability vector of that race.
Aggregate demand share for supplier r is

    Q_r(psi) = sum_b w_b p_r(psi + delta_b),

and market clearing asks Q(psi) = kappa (capacity shares). Q is the
gradient of sum_b w_b E min_i(psi_i + delta_bi + eps_i), a sum of
concave potentials, so the winning paper's Theorem 1 applies verbatim:
the clearing psi exists and is unique on contrasts for interior kappa.

Solver: the same damped diagonal log-residual iteration the package
ships for inversion, generalized to the mixture -- own slopes summed
across buyer types, all riding the forward passes.

Ground truth by construction: draw psi*, set kappa = Q(psi*), re-solve
from a cold start, and check both the clearing residual and recovery of
psi* up to the gauge constant.
"""
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from winning.factor.races import race_probabilities  # noqa: E402


def aggregate(psi, deltas, weights, V, D, points=257):
    Q = np.zeros_like(psi)
    S = np.zeros_like(psi)
    for delta_b, w_b in zip(deltas, weights):
        p, s = race_probabilities(psi + delta_b, V=V, D=D, points=points,
                                  return_slopes=True)
        Q += w_b * p
        S += w_b * s
    return Q, S


def clear_market(kappa, deltas, weights, V, D, points=257, tol=1e-8,
                 max_iter=500):
    t = np.log(kappa)
    psi = -(t - t.mean()) / 2.0
    alpha = 0.7
    prev = np.inf
    for it in range(1, max_iter + 1):
        Q, S = aggregate(psi, deltas, weights, V, D, points=points)
        # the same guards the shipped inversion uses: a laggard whose
        # aggregate share underflows still needs a finite residual and a
        # finite slope to be pulled toward its target
        Q = np.maximum(Q, 1e-300)
        r = np.log(Q) - t
        worst = float(np.abs(r).max())
        if worst < tol:
            return psi, it, worst
        if worst > prev:
            alpha = max(alpha / 2.0, 0.05)
        prev = worst
        ratio = S / Q
        d = np.where(np.isfinite(ratio), np.minimum(ratio, -1e-6), -1e-6)
        psi = psi - np.clip(alpha * r / d, -2.0, 2.0)
        psi -= psi.mean()
    return psi, max_iter, worst


def run(n, n_types, seed, points=257):
    rng = np.random.default_rng(seed)
    V = 0.4 * np.ones((n, 1))                 # common market/freight shock
    D = 0.5 + rng.random(n)                   # idiosyncratic delivered noise
    psi_star = rng.normal(0.0, 0.8, n)
    psi_star -= psi_star.mean()
    deltas = [rng.normal(0.0, 0.5, n) for _ in range(n_types)]
    weights = rng.random(n_types)
    weights = weights / weights.sum()

    kappa, _ = aggregate(psi_star, deltas, weights, V, D, points=points)

    t0 = time.time()
    psi_hat, iters, resid = clear_market(kappa, deltas, weights, V, D,
                                         points=points)
    wall = time.time() - t0

    recovery = float(np.abs((psi_hat - psi_hat.mean())
                            - (psi_star - psi_star.mean())).max())
    return dict(n=n, n_types=n_types, seed=seed, iters=iters,
                clearing_residual=resid, psi_recovery_max=recovery,
                wall_seconds=round(wall, 3),
                kappa_min=float(kappa.min()), kappa_max=float(kappa.max()))


if __name__ == "__main__":
    results = []
    for n in (1_000, 10_000, 100_000):
        r = run(n, n_types=3, seed=7)
        print(r)
        results.append(r)
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out)
