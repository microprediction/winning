"""The n=2 rollout-pruning free boundary, solved and tested.

Two correlated Brownian trajectories, X_i = x_i + W_i with
d<W_1, W_2> = rho dt, budget b of compute-time, cost lambda per unit
time per ACTIVE path, terminal reward max over survivors, killing
irreversible. The gap Delta = X_1 - X_2 is Brownian with volatility
sigma = sqrt(2(1 - rho)); the leader's level is a martingale under
any keep policy, and a lone survivor earns nothing in expectation
while still paying lambda. So the whole problem lives on the gap:
writing U(delta, b) for the value of optimal play MINUS the current
leader level,

    U(delta, 0) = 0,
    U(delta, b) = max{ 0,
        -2 lambda dt + E U(|delta + sigma sqrt(dt) Z|, b - dt) },

with |.| because the leader identity swaps when the gap crosses zero
-- the reflection is where the option value comes from (keep both to
exhaustion and U = (E|Delta_T| - delta)/2 - lambda b, by
max = (sum + |gap|)/2 and the sum being a martingale).

Solved by backward induction on a (delta, b) grid, Gauss-Hermite
transition, linear interpolation. Measured:
  1. the kill boundary h(b): keep both iff delta < h(b) -- existence
     and monotonicity in remaining budget are the conjecture;
  2. a scaling collapse: Brownian scaling gives
     h(b; sigma, lambda) = (sigma^2/lambda) H(b lambda^2/sigma^2)
     (derivation at the test site below) -- checked by solving at two
     parameter points that the collapse maps onto each other;
  3. Monte Carlo value of the free-boundary policy against the
     static rules the LLM-rollout literature tunes: keep-both-always,
     kill-at-fixed-time, and fixed-threshold kill -- each static rule
     at its OWN best tuning (grid-searched), so the comparison is
     against tuned incumbents, not straw men.
"""
import json
import os
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

DT = 0.0005
Q = 21


def gh():
    z, w = np.polynomial.hermite_e.hermegauss(Q)
    return z, w / w.sum()


def solve(sigma, lam, b_total, n_delta=1200):
    """Backward induction; returns delta grid, b grid, keep boundary.

    The natural units are delta ~ sigma^2/lambda and t ~ (sigma/lambda)^2,
    with the keep region's onset time t* = sigma^2/(32 pi lambda^2); the
    grid must resolve both, and the U > 0 region is small in sigma
    units, so the delta grid stops at 0.6 sigma^2/lambda (with linear
    interpolation clamping U to ~0 beyond).
    """
    z, w = gh()
    steps = int(round(b_total / DT))
    dmax = max(0.6 * sigma ** 2 / lam, 8.0 * sigma * np.sqrt(DT))
    dgrid = np.linspace(0.0, dmax, n_delta)
    U = np.zeros(n_delta)
    h = np.zeros(steps)
    step_sd = sigma * np.sqrt(DT)
    # transition targets |delta + step_sd z| for every (delta, z)
    tgt = np.abs(dgrid[:, None] + step_sd * z[None, :])
    # each step credits the leader-level drift E[(|delta'| - delta)]/2:
    # U is measured relative to the CURRENT leader, and the reflection
    # of the gap at zero is what raises the leader in expectation
    drift = 0.5 * ((tgt @ w) - dgrid)
    for k in range(steps):
        EU = np.interp(tgt, dgrid, U) @ w
        U = np.maximum(0.0, -2.0 * lam * DT + drift + EU)
        keep = U > 0.0
        h[k] = dgrid[keep][-1] if keep.any() else 0.0
    return dgrid, np.arange(1, steps + 1) * DT, h, U


def simulate(policy, sigma, lam, b_total, n_mc, rng, dgrid=None,
             bgrid=None, h=None, param=None):
    """MC value (relative to initial leader) of a policy from a
    standing start delta0 = 0. Policies: 'free', 'always',
    'fixed_time' (kill loser at time param), 'fixed_gap' (kill loser
    first time |gap| > param)."""
    steps = int(round(b_total / DT))
    delta = np.zeros(n_mc)                 # current |gap|
    lead = np.zeros(n_mc)                  # leader level - initial
    alive = np.ones(n_mc, bool)            # both alive?
    value = np.zeros(n_mc)                 # frozen value once killed
    cost = np.zeros(n_mc)
    step_sd = sigma * np.sqrt(DT)
    for k in range(steps):
        b_left = b_total - k * DT
        if policy == "free":
            hk = np.interp(b_left, bgrid, h)
            kill = alive & (delta >= hk)
        elif policy == "always":
            kill = np.zeros(n_mc, bool)
        elif policy == "fixed_time":
            kill = alive & (k * DT >= param)
        else:
            kill = alive & (delta >= param)
        value[kill] = lead[kill] - cost[kill]
        alive &= ~kill
        if not alive.any():
            break
        cost[alive] += 2.0 * lam * DT
        # gap diffuses; leader level: max = (sum + |gap|)/2, and the
        # increment of the max given the reflected gap bookkeeping:
        # track sum and gap separately for correctness
        zg = rng.normal(size=alive.sum())
        # max = (sum + |gap|)/2 and the sum X_1 + X_2 is a martingale
        # for every rho, so it contributes zero to expected value and
        # only the |gap|/2 increments are tracked.
        delta_new = np.abs(delta[alive] + step_sd * zg)
        lead[alive] += 0.5 * (delta_new - delta[alive])
        delta[alive] = delta_new
    value[alive] = lead[alive] - cost[alive]
    return float(value.mean()), float(value.std() / np.sqrt(n_mc))


if __name__ == "__main__":
    results = {}
    t0 = time.time()

    # 1. boundary existence and shape, three correlations
    for rho in (0.0, 0.5, 0.9):
        sigma = np.sqrt(2.0 * (1.0 - rho))
        dgrid, bgrid, h, _ = solve(sigma, lam=1.0, b_total=2.0)
        # monotone in remaining budget?
        mono = bool(np.all(np.diff(h) >= -1e-9))
        results[f"boundary_rho{rho}"] = dict(
            monotone_in_budget=mono,
            h_at_b=[[float(b), float(np.interp(b, bgrid, h))]
                    for b in (0.01, 0.05, 0.2, 1.0, 2.0)])
        print(f"[rho={rho}] boundary monotone={mono}  h(b): "
              + "  ".join(f"{b}:{np.interp(b, bgrid, h):.3f}"
                          for b in (0.01, 0.05, 0.2, 1.0, 2.0)))

    # 2. scaling collapse. Rescaling delta = a delta~, t = c t~ turns
    #    (sigma, lambda) into (sigma sqrt(c)/a, lambda c/a); choosing
    #    c = (sigma/lambda)^2 and a = sigma^2/lambda normalizes both,
    #    so U(delta, b; sigma, lambda)
    #       = (sigma^2/lambda) U1(delta lambda/sigma^2, b lambda^2/sigma^2)
    #    and h(b; sigma, lambda) = (sigma^2/lambda) H(b lambda^2/sigma^2).
    #    Test: doubling sigma at fixed lambda should give
    #    h2(4b) = 4 h1(b).
    d1, b1, h1, _ = solve(np.sqrt(2.0), 1.0, 2.0)
    d2, b2, h2, _ = solve(2.0 * np.sqrt(2.0), 1.0, 8.0)
    probe_b = np.array([0.2, 0.5, 1.0, 2.0])
    ratio = [float(np.interp(4.0 * bb, b2, h2)
                   / np.interp(bb, b1, h1)) for bb in probe_b]
    results["scaling_ratio"] = dict(predicted=4.0, measured=ratio)
    print("[scaling] predicted ratio 4.000  measured "
          + " ".join(f"{r:.3f}" for r in ratio))

    # 3. free boundary vs tuned static rules, MC
    rho, lam, b_total = 0.5, 1.0, 1.0
    sigma = np.sqrt(2.0 * (1.0 - rho))
    dgrid, bgrid, h, U = solve(sigma, lam, b_total)
    rng = np.random.default_rng(3)
    n_mc = 100_000
    v_free, se = simulate("free", sigma, lam, b_total, n_mc, rng,
                          dgrid=dgrid, bgrid=bgrid, h=h)
    v_bellman = float(np.interp(0.0, dgrid, U))
    rows = dict(free=dict(value=v_free, se=se, bellman=v_bellman))
    print(f"[policy rho=0.5 b=1] free {v_free:.4f} (se {se:.4f}) "
          f"vs Bellman {v_bellman:.4f}")
    for pol, grid_vals in (("always", [None]),
                           ("fixed_time", np.linspace(0.05, 1.0, 12)),
                           ("fixed_gap", np.linspace(0.2, 3.0, 12))):
        best = None
        for param in grid_vals:
            v, se_p = simulate(pol, sigma, lam, b_total, 50_000,
                               np.random.default_rng(17), param=param)
            if best is None or v > best[0]:
                best = (v, se_p, param)
        v, se_p, param = best
        v, se_p = simulate(pol, sigma, lam, b_total, n_mc,
                           np.random.default_rng(23), param=param)
        rows[pol] = dict(value=v, se=se_p,
                         best_param=None if param is None
                         else float(param))
        print(f"  {pol:10s} best value {v:.4f} (se {se_p:.4f})"
              + (f" at param {param:.2f}" if param is not None else ""))
    results["policies"] = rows

    results["seconds"] = time.time() - t0
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print(f"done in {results['seconds']:.0f}s; wrote results.json")
