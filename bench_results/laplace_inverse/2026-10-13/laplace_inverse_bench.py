"""Laplace-inverse timing: abilities_from_race(base="laplace") on seeded
self-generated targets. No standing suite covers the Laplace inverse, so
this micro-benchmark measures the cost change expected from #707 (Newton
own-slopes from the split lattice = a second forward pass per iteration).
Run the SAME script against two checkouts via PYTHONPATH:

    PYTHONPATH=<checkout> python laplace_inverse_bench.py <label> >> records.jsonl

Each line: label, sha, n, base, seconds (best of 3), iterations, residual,
converged, max |mu_hat - mu_true| after centring."""
import json, subprocess, sys, time, os
import numpy as np
import winning
from winning.factor.races import race_probabilities, abilities_from_race

label = sys.argv[1]
root = os.path.dirname(os.path.dirname(winning.__file__))
sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
for base in ("laplace", "normal"):
    for n, sd_spread in ((10, 1.0), (50, 1.0), (200, 1.0), (50, 3.0)):
        rng = np.random.default_rng(1000 + n)
        mu = rng.normal(0, sd_spread, n); mu -= mu.mean()
        p = race_probabilities(mu, base=base)
        best = np.inf; info = None; muh = None
        for rep in range(3):
            t0 = time.perf_counter()
            try:
                muh, info = abilities_from_race(p, base=base, return_info=True)
            except Exception as e:
                info = {"error": repr(e)}; muh = None
            best = min(best, time.perf_counter() - t0)
        rec = {"label": label, "sha": sha, "base": base, "n": n, "spread": sd_spread,
               "seconds": round(best, 5)}
        if muh is not None:
            rec["mu_err"] = float(np.abs((muh - muh.mean()) - mu).max())
            for k in ("converged", "iterations", "max_log_residual"):
                if isinstance(info, dict) and k in info:
                    v = info[k]; rec[k] = v.item() if hasattr(v, "item") else v
        else:
            rec.update(info)
        print(json.dumps(rec, default=str), flush=True)
