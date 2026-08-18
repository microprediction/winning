"""Experiment 16, Part D: replication over problems (referee item).

The headline benchmark (exp13) used one problem per N and changed the utility
spread from 1.0 to 1.5 above N=200. Here: 10 independent problems per N at
N in {20, 50, 200}, and 3 at N=1000, all at a COMMON spread of 1.0, each with
its own fresh 2e6-draw truth. Reports median and worst max-coordinate lattice
error per N, plus the reference's own replicate-to-replicate noise scale.

Run:  python experiments/exp16_benchmark_addendum/run_replication.py
Output: results_replication.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from run_ghk_benchmark import lattice_shares, make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    rng = np.random.default_rng(77)
    rows = ["N,problem,lattice_max_err,ref_noise_scale,lattice_seconds"]
    for n, reps in ((20, 10), (50, 10), (200, 10), (1000, 3)):
        errs, noises = [], []
        for r in range(reps):
            mu, V, D = make_problem(n, 2, rng, spread=1.0)
            truth_a = mc_shares(mu, V, D, 2_000_000, seed=100 + r)
            truth_b = mc_shares(mu, V, D, 2_000_000, seed=500 + r)
            noise = np.abs(truth_a - truth_b).max()  # two-reference noise scale
            t0 = time.perf_counter()
            p = lattice_shares(mu, V, D)
            dt = time.perf_counter() - t0
            err = np.abs(p - 0.5 * (truth_a + truth_b)).max()
            errs.append(err); noises.append(noise)
            rows.append(f"{n},{r},{err:.3e},{noise:.3e},{dt:.3f}")
        print(f"N={n:>5}: lattice max err median {np.median(errs):.1e}, "
              f"worst {max(errs):.1e} over {reps} problems "
              f"(reference replicate noise median {np.median(noises):.1e})")
    (HERE / "results_replication.csv").write_text("\n".join(rows) + "\n")
    print("wrote results_replication.csv")


if __name__ == "__main__":
    main()
