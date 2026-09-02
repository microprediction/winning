"""Orchestrates the lpRR complete-vector baseline against the engine.

The strongest factor-aware per-winner baseline (manuscript change 2
in ADVERSARIES.md): mvtnorm::lpRR on the k+1-column reduced-rank
difference representation, one call per winner, common Sobol draws.
The engine's shared field prices the same vector in one pass; lpRR's
cost is O(R N^2 (k+1)) for the vector against O(QNL). Total
variation is measured against the engine (exact for this class,
previously certified against Monte Carlo).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                ".."))
from winning.factor import race_probabilities       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def make_instance(n, k, share, rng):
    mu = rng.normal(0.0, 1.0, n)
    u = rng.normal(size=(n, k))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    base = 0.3 + rng.random(n)
    V = u * np.sqrt(share * base)[:, None]
    d = (1.0 - share) * base
    return mu, V, d


if __name__ == "__main__":
    results = {}
    for n in (50, 200, 1000):
        rng = np.random.default_rng(n + 2)
        mu, V, d = make_instance(n, 2, 0.5, rng)
        t0 = time.time()
        p_exact = race_probabilities(-mu, V=-V, D=d)   # max-wins
        t_exact = time.time() - t0
        inst = os.path.join(HERE, f"instance_n{n}.json")
        with open(inst, "w") as f:
            json.dump(dict(mu=mu.tolist(), V=V.tolist(),
                           d=d.tolist()), f)
        row = dict(exact_seconds=t_exact)
        for R in (512, 4096):
            out = subprocess.run(
                ["Rscript", os.path.join(HERE, "run_lprr.R"), inst,
                 str(R)], capture_output=True, text=True, timeout=3600)
            if out.returncode != 0:
                print(out.stderr[-500:])
                raise SystemExit(1)
            res = json.loads(out.stdout)
            p = np.exp(np.array(res["logp"]))
            row[f"lprr_R{R}"] = dict(
                seconds=float(np.atleast_1d(res["seconds"])[0]),
                sobol=bool(np.atleast_1d(res["sobol"])[0]),
                mass=float(p.sum()),
                tv=float(0.5 * np.abs(p - p_exact).sum()),
                tv_normalized=float(0.5 * np.abs(p / p.sum()
                                                 - p_exact).sum()))
        results[f"n{n}"] = row
        print(f"[N={n} k=2] exact {t_exact:.3f}s | " + " | ".join(
            f"lpRR R={R}: tv {row[f'lprr_R{R}']['tv']:.4f} "
            f"(norm {row[f'lprr_R{R}']['tv_normalized']:.4f}) "
            f"{row[f'lprr_R{R}']['seconds']:.2f}s"
            for R in (512, 4096)))
        os.remove(inst)
    with open(os.path.join(HERE, "results.json"), "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
