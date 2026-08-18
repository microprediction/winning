"""Rust calibration wall time at N = 1000, 5000, 10000 (k=2), same
protocol as run_calibration_scaling.py: independent 5e6-draw MC targets,
no inverse crime. Confirms (or corrects) the abstract's linearity claim.

Run:  python experiments/exp19_calibration_scaling/run_rust_scaling.py
Output: rust_results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes  # noqa: E402
from run_ghk_benchmark import make_problem, mc_shares  # noqa: E402
from rustcal import calibrate_rust  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def main():
    rng = np.random.default_rng(SEED)
    F, W = hermite_nodes(2)
    rows = ["N,seconds,iterations,residual,converged"]
    for n in (1000, 5000, 10000):
        mu, V, D = make_problem(n, 2, rng, spread=1.5)
        mu -= mu.mean()
        target = mc_shares(mu, V, D, 5_000_000, seed=90 + n % 97)
        target = np.maximum(target, 1e-9)
        target /= target.sum()
        t0 = time.perf_counter()
        a_hat, info = calibrate_rust(target, V, D, F, W, return_info=True)
        t_cal = time.perf_counter() - t0
        print(f"N={n:>6}: {t_cal:6.1f}s  ({info['iterations']} iterations, "
              f"residual {info['residual']:.1e}, converged {info['converged']})")
        rows.append(f"{n},{t_cal:.1f},{info['iterations']},"
                    f"{info['residual']:.3e},{info['converged']}")
    (HERE / "rust_results.csv").write_text("\n".join(rows) + "\n")
    print("wrote rust_results.csv")


if __name__ == "__main__":
    main()
