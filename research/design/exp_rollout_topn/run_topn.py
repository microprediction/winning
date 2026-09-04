"""Rollout pruning at n = 16: catchability boundary vs tuned schedules.

Hard-budget accounting, which is the token-budget reality: a budget B
of path-time is consumed at rate (paths alive) per unit time, killing
a path stretches the survivors' horizon, and the terminal reward is
the max over survivors when the budget runs out. (This differs from
the n=2 experiment's priced-computation accounting; there the
alternative to keeping was stopping, here it is running the rest
longer.)

Trajectories are correlated through one factor:
dX_i = v_i dF + sqrt(d_i) dB_i, so the gap of i to leader L has
volatility sigma_iL^2 = (v_i - v_L)^2 + d_i + d_L -- near-duplicates
(same loading) have small pairwise volatility and cannot catch up,
whatever their gap.

Policies, each tuned on its own held-out seed before the race:
  catch     kill i when X_L - X_i > c sigma_iL sqrt(T_aff), with
            T_aff = remaining budget / current survivor count (the
            affordable horizon) -- the n=2 free boundary's shape
            with the correlation-aware pairwise volatility, one
            constant c to tune.
  halving   sequential halving: split the budget into log2(n) equal
            rounds, kill the worse half at each round boundary.
  gambit    fixed interval tau, kill bottom K each interval
            (constant-capacity beam without branching); (tau, K)
            tuned on a grid.
  keepall   no kills (the ToT-with-large-b baseline).

Two configurations at n = 16, B = 16 (keep-all horizon 1):
  spread    loadings v ~ N(0, 0.7), idiosyncratic 0.35 + U(0,1);
  clusters  four clusters of four genuine near-duplicates
            (within-cluster pairwise gap variance ~0.2, cross-cluster
            dominated by the loading gap): the regime where
            correlation awareness should matter most.
Reported: E[max over survivors at budget exhaustion], MC standard
error, and the kill profile (mean survivor count over time).
"""
import json
import os
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

N = 16
B = 16.0
DT = 0.005


def make_config(kind, rng):
    if kind == "spread":
        v = rng.normal(0.0, 0.7, N)
        d = 0.35 + rng.random(N)
    else:
        # genuine near-duplicates: within-cluster pairwise gap
        # variance ~ 2 d = 0.2, cross-cluster ~ (dv)^2 + 0.2
        centers = rng.normal(0.0, 0.8, 4)
        v = np.repeat(centers, 4) + rng.normal(0.0, 0.03, N)
        d = np.full(N, 0.1)
    return v, d


def run_policy(policy, v, d, n_mc, seed, param=None):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_mc, N))
    alive = np.ones((n_mc, N), bool)
    spent = np.zeros(n_mc)
    running = np.ones(n_mc, bool)
    sqrt_dt = np.sqrt(DT)
    sd = np.sqrt(d)
    # sequential-halving schedule: budget fractions at which to halve
    n_rounds = int(np.log2(N))
    halving_marks = B * (np.arange(1, n_rounds + 1) / n_rounds)
    halved = np.zeros((n_mc, n_rounds), bool)
    # gambit: interval tau in budget units, kill K
    gambit_next = np.full(n_mc, param[0]) if policy == "gambit" else None
    survivors_curve = []
    t_wall = 0.0
    while running.any():
        na = alive.sum(1)
        idx = running & (spent + na * DT <= B)
        running = idx
        if not idx.any():
            break
        # diffuse
        m = idx
        k = m.sum()
        F = rng.normal(size=k) * sqrt_dt
        Zb = rng.normal(size=(k, N)) * sqrt_dt
        X[m] += np.where(alive[m], v[None, :] * F[:, None]
                         + sd[None, :] * Zb, 0.0)
        spent[m] += alive[m].sum(1) * DT
        # kills
        Xm = np.where(alive[m], X[m], -np.inf)
        L = Xm.argmax(1)
        XL = Xm[np.arange(k), L]
        if policy == "catch":
            T_aff = (B - spent[m]) / np.maximum(alive[m].sum(1), 1)
            sig = np.sqrt((v[None, :] - v[L][:, None]) ** 2
                          + d[None, :] + d[L][:, None])
            thr = param * sig * np.sqrt(np.maximum(T_aff, 0.0))[:, None]
            kill = alive[m] & ((XL[:, None] - Xm) > thr)
            kill[np.arange(k), L] = False
            alive[m] = alive[m] & ~kill
        elif policy == "halving":
            for r in range(n_rounds):
                due = (spent[m] >= halving_marks[r]) & ~halved[m][:, r]
                if due.any():
                    rows = np.where(m)[0][due]
                    for row in rows:
                        live = np.where(alive[row])[0]
                        if len(live) > 1:
                            order = live[np.argsort(X[row, live])]
                            alive[row, order[: len(live) // 2]] = False
                        halved[row, r] = True
        elif policy == "gambit":
            due = spent[m] >= gambit_next[m]
            if due.any():
                rows = np.where(m)[0][due]
                for row in rows:
                    live = np.where(alive[row])[0]
                    kk = min(int(param[1]), len(live) - 1)
                    if kk > 0:
                        order = live[np.argsort(X[row, live])]
                        alive[row, order[:kk]] = False
                    gambit_next[row] += param[0]
        if len(survivors_curve) < 4000:
            survivors_curve.append(float(alive[m].sum(1).mean()))
        t_wall += DT
    vals = np.where(alive, X, -np.inf).max(1)
    return float(vals.mean()), float(vals.std() / np.sqrt(n_mc)), \
        survivors_curve


if __name__ == "__main__":
    results = {}
    t0 = time.time()
    for kind in ("spread", "clusters"):
        cfg_rng = np.random.default_rng(hash(kind) % 2 ** 31)
        v, d = make_config(kind, cfg_rng)
        rows = {}
        # tune each policy on seed 100, evaluate on seed 200
        best_c = None
        for c in (0.4, 0.7, 1.0, 1.4, 2.0, 2.8):
            val, _, _ = run_policy("catch", v, d, 8000, 100, param=c)
            if best_c is None or val > best_c[0]:
                best_c = (val, c)
        val, se, curve = run_policy("catch", v, d, 60000, 200,
                                    param=best_c[1])
        rows["catch"] = dict(value=val, se=se, c=best_c[1])
        val, se, _ = run_policy("halving", v, d, 60000, 200)
        rows["halving"] = dict(value=val, se=se)
        best_g = None
        for tau in (1.0, 2.0, 4.0):
            for K in (1, 2, 4):
                gval, _, _ = run_policy("gambit", v, d, 8000, 100,
                                        param=(tau, K))
                if best_g is None or gval > best_g[0]:
                    best_g = (gval, tau, K)
        val, se, _ = run_policy("gambit", v, d, 60000, 200,
                                param=(best_g[1], best_g[2]))
        rows["gambit"] = dict(value=val, se=se, tau=best_g[1],
                              K=best_g[2])
        val, se, _ = run_policy("keepall", v, d, 60000, 200)
        rows["keepall"] = dict(value=val, se=se)
        results[kind] = rows
        print(f"[{kind}] " + "  ".join(
            f"{k} {rows[k]['value']:.4f}±{rows[k]['se']:.4f}"
            for k in ("catch", "halving", "gambit", "keepall"))
            + f"  (c={best_c[1]}, gambit tau={best_g[1]} K={best_g[2]})")
    results["seconds"] = time.time() - t0
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print(f"done in {results['seconds']:.0f}s; wrote results.json")
