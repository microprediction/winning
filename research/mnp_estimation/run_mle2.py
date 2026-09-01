"""Round 2: isolate the simulation penalty. T = 50,000 choices (so
statistical error is small and simulation bias binds), known Sigma,
estimate mu only, n = 30, scored on identified alternatives (p > 0.01).

Fifth review: the saturated likelihood has NO finite maximizer when any
count is zero, and two of the first eight replications (seeds 103 and
105) have one -- the smallest alternative has p = 2.5e-5, so its expected
count is 1.27 and it is empty with probability e^-1.27 = 0.28. Each count
is therefore smoothed by one half before any arm sees it, which is the
posterior-mean share vector under the multinomial Jeffreys prior. (It is
NOT the Jeffreys MAP, whose Dirichlet(1/2) posterior has exponents
c_i - 1/2 and leaves empty cells on the boundary; read as a penalized
likelihood, add-half is the Dirichlet(3/2) mode.) ALPHA = 0 reproduces
the original unsmoothed run, which is well defined only on the
replications with full support.

Sixth review follow-up: eight replications is few, and the arms are
PAIRED by construction -- every method sees the same seeds, hence the
same counts -- so the paired difference is far more informative than
three medians compared by eye. This runs the replication loop OUTSIDE
the method loop and checkpoints after each one, so a long run can be
interrupted or resumed and still yields complete paired replications.

    python run_mle2.py --reps 40 --out results_40.json
    python run_mle2.py --reps 40 --out results_40.json --resume

Timings from this script are only meaningful on a quiet machine; see
docs/latex_src/general_inversion/retime.py, which refuses to measure on
a busy one. The RMSE columns are unaffected by machine load.
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.optimize import minimize

from winning.factor.races import race_probabilities
from winning.factor.polish import race_jacobian
import fastrace

n, T = 30, 50_000
ALPHA = 0.5          # Jeffreys smoothing; 0.0 = unsmoothed (may not exist)
METHODS = ("exact", "msl100", "msl1000")

rng0 = np.random.default_rng(3)
mu_star = np.sort(rng0.normal(size=n)) * 0.8
mu_star -= mu_star.mean()
V = rng0.normal(size=(n, 2)) * 0.4
D = 0.6 + 0.5 * rng0.random(n)


def p_exact(mu):
    return race_probabilities(mu, V=V, D=D, points=257)


def negll_exact(mu_free, counts):
    mu = mu_free - mu_free.mean()
    return -counts @ np.log(np.maximum(p_exact(mu), 1e-300))


def grad_exact(mu_free, counts):
    mu = mu_free - mu_free.mean()
    p = np.maximum(p_exact(mu), 1e-300)
    J = race_jacobian(mu, V=V, D=D, points=257)
    g = -(J.T @ (counts / p))
    return g - g.mean()


def negll_ghk(mu_free, counts, R, seed):
    mu = mu_free - mu_free.mean()
    g = np.asarray(fastrace.ghk_all_shares(-mu, V, D, R, seed))
    g = np.maximum(g, 1e-12)
    g = g / g.sum()
    return -counts @ np.log(g)


def fd_grad(f, th, h=1e-4):
    g = np.zeros(len(th))
    for j in range(len(th)):
        e = np.zeros(len(th))
        e[j] = h
        g[j] = (f(th + e) - f(th - e)) / (2 * h)
    return g


def fit(method, counts):
    th0 = np.zeros(n)
    t0 = time.time()
    if method == "exact":
        res = minimize(negll_exact, th0, args=(counts,), jac=grad_exact,
                       method="L-BFGS-B", options={"maxiter": 150})
    else:
        R = 100 if method == "msl100" else 1000
        def f(th):
            return negll_ghk(th, counts, R, 777)
        res = minimize(f, th0, jac=lambda th: fd_grad(f, th),
                       method="L-BFGS-B", options={"maxiter": 150})
    return res.x - res.x.mean(), time.time() - t0


def summarize(recs, ident_count):
    """Medians, standard errors, and the PAIRED comparison against the
    exact arm, which is what eight unpaired medians could not support."""
    print(f"\nidentified alternatives: {ident_count}/{n}")
    print(f"replications: {len(recs)}\n")
    print(f"{'method':10s} {'median rmse':>12s} {'mean +/- se':>18s} "
          f"{'median fit':>12s}")
    for m in METHODS:
        e = np.array([r["err"][m] for r in recs])
        t = np.array([r["time"][m] for r in recs])
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else float("nan")
        print(f"{m:10s} {np.median(e):12.4f} {e.mean():11.4f} +/- {se:.4f} "
              f"{np.median(t):11.2f}s")
    print("\npaired against exact (positive = exact better):")
    ex = np.array([r["err"]["exact"] for r in recs])
    for m in METHODS[1:]:
        o = np.array([r["err"][m] for r in recs])
        d = o - ex
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        wins = int((d > 0).sum())
        print(f"  vs {m:8s} mean diff {d.mean():+.4f} +/- {se:.4f} "
              f"({d.mean()/se:5.1f} SE)   exact wins {wins}/{len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--out", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    p_true = p_exact(mu_star)
    ident = p_true > 0.01
    print(f"identified alternatives: {ident.sum()}/{n}", flush=True)

    recs = []
    if args.resume and args.out and os.path.exists(args.out):
        recs = json.load(open(args.out))["reps"]
        print(f"resuming from {len(recs)} completed replications", flush=True)

    for rep in range(len(recs), args.reps):
        rng = np.random.default_rng(100 + rep)
        counts = rng.multinomial(T, p_true).astype(float) + ALPHA
        rec = {"rep": rep, "seed": 100 + rep, "err": {}, "time": {},
               "empty_cells": int((counts <= ALPHA + 1e-9).sum())}
        for m in METHODS:
            mu_hat, secs = fit(m, counts)
            rec["err"][m] = float(
                np.sqrt(np.mean((mu_hat[ident] - mu_star[ident]) ** 2)))
            rec["time"][m] = float(secs)
        recs.append(rec)
        print(f"rep {rep:3d} (seed {rec['seed']}, {rec['empty_cells']} empty) "
              + "  ".join(f"{m} {rec['err'][m]:.4f}" for m in METHODS),
              flush=True)
        if args.out:                       # checkpoint every replication
            with open(args.out, "w") as fh:
                json.dump({"n": n, "T": T, "alpha": ALPHA,
                           "identified": int(ident.sum()), "reps": recs},
                          fh, indent=1)

    summarize(recs, int(ident.sum()))


if __name__ == "__main__":
    main()
