"""Does removing TuRBO's candidate cap find better optima?

TuRBO caps its Thompson candidate set at min(100d, 5000) because joint
sampling needs an N x N Cholesky. We can lift the cap (sample the latent
instead of the field). This asks the only question that matters: does it help?

Arms differ ONLY in candidate count and how the batch is drawn. Same GP, same
trust-region logic, same evaluation budget.

  cholesky-5k   what TuRBO does today (the cap, joint sampling)
  factor-5k     same cap, our sampling -- CONTROL for the rank-r approximation
  factor-20k    cap lifted 4x
  factor-50k    cap lifted 10x
"""
import sys, time, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from turbo_race import turbo, ackley, levy

HERE = Path(__file__).resolve().parent
ARMS = [("cholesky-5k", "cholesky", 5000),
        ("factor-5k", "factor", 5000),
        ("factor-20k", "factor", 20000),
        ("factor-50k", "factor", 50000)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", default="ackley")
    ap.add_argument("--d", type=int, default=100)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="turbo_cap.csv")
    a = ap.parse_args()
    f = {"ackley": ackley, "levy": levy}[a.fn]
    print(f"{a.fn} d={a.d}, {a.iters} iters x batch {a.batch}, seeds {a.seeds}", flush=True)
    rows = []
    for name, mode, N in ARMS:
        for s in a.seeds:
            t0 = time.time()
            best, nev, t_acq = turbo(f, d=a.d, n_init=2 * a.d if a.d <= 50 else 20,
                                     n_iter=a.iters, batch=a.batch, N_cand=N,
                                     mode=mode, seed=s)
            rows.append({"fn": a.fn, "d": a.d, "arm": name, "mode": mode, "N_cand": N,
                         "seed": s, "best": round(float(best), 5), "n_evals": int(nev),
                         "acq_seconds": round(t_acq, 1), "total_seconds": round(time.time() - t0, 1)})
            print(f"  {name:12s} seed {s}: best {best:8.4f}  ({nev} evals, "
                  f"acq {t_acq:5.1f}s, total {time.time()-t0:5.0f}s)", flush=True)
            with open(HERE / "results" / a.out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import collections
    agg = collections.defaultdict(list)
    for r in rows: agg[r["arm"]].append(r["best"])
    print("\n=== mean best (lower is better) ===", flush=True)
    for name, _, _ in ARMS:
        v = np.array(agg[name])
        if len(v): print(f"  {name:12s} {v.mean():8.4f} +- {v.std()/np.sqrt(len(v)):.4f}  (n={len(v)})", flush=True)
