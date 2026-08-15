"""Run contestants over the standing problem grid; append records.

    python -m winning.bench.runner [--quick]

Each record: problem id, method, budget, seconds, max_err, backend info.
References are direct-MC with 2e6 draws, cached per problem id.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..methods import METHODS
from ..methods.native import direct_mc

RESULTS = Path(__file__).resolve().parents[2] / "bench_results"


def problem(pid, n, k, spread, seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, spread, n)
    V = rng.normal(0.0, 0.5 / np.sqrt(k), (n, k))
    D = rng.uniform(0.5, 1.5, n)
    return mu, V, D


GRID = [
    ("n50k2", 50, 2, 1.0, 101),
    ("n200k2", 200, 2, 1.0, 102),
    ("n1000k2", 1000, 2, 1.5, 103),
    ("n200k3", 200, 3, 1.0, 104),
]

BUDGETS = {
    "lattice": [None],
    "direct_mc": [10**5, 10**6],
    "sobol_direct": [2**14, 2**17],
    "factor_rqmc": [2**12, 2**14],
    "ghk": [1000],
    "qmc_ghk": [1024, 8192],
    "genz_bretz": [1024, 8192],
    "tilting": [1000],
    "mendell_elston": [None],
    "ep_orthant": [None],
    "smc_orthant": [1000],
}

# per-alternative methods whose cost is quadratic-or-worse in N: excluded
# entirely at N >= 1000 (their scaling story is told at 50 and 200)
HEAVY_AT_LARGE_N = {"tilting", "factor_rqmc", "ep_orthant", "smc_orthant",
                    "genz_bretz", "ghk", "qmc_ghk", "mendell_elston"}


def main(quick=False):
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "records.jsonl"
    grid = GRID[:2] if quick else GRID
    with out.open("a") as fh:
        for pid, n, k, spread, seed in grid:
            mu, V, D = problem(pid, n, k, spread, seed)
            ref_file = RESULTS / f"ref_{pid}.npy"
            if ref_file.exists():
                truth = np.load(ref_file)
            else:
                truth, _ = direct_mc(mu, V, D, 2_000_000, seed=9)
                np.save(ref_file, truth)
            for name, fn in METHODS.items():
                for budget in BUDGETS.get(name, [None]):
                    if n >= 1000 and name in HEAVY_AT_LARGE_N:
                        continue
                    if quick and name in ("tilting", "factor_rqmc") and n > 200:
                        continue
                    t0 = time.perf_counter()
                    p, info = fn(mu, V, D, budget=budget, seed=17)
                    dt = time.perf_counter() - t0
                    rec = {"problem": pid, "n": n, "k": k, "method": name,
                           "budget": budget, "seconds": round(dt, 4),
                           "max_err": float(np.abs(p - truth).max()),
                           "info": info}
                    fh.write(json.dumps(rec) + "\n")
                    print(f"{pid:>8} {name:>12} budget={str(budget):>8}: "
                          f"{dt:7.2f}s err {rec['max_err']:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
