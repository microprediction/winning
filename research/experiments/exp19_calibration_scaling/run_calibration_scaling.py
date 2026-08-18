"""Experiment 19: how far does calibration scale?

The paper's headline is calibration at N=1000 in under a minute. This
measures the ceiling: calibration wall time, solver residual, and utility
recovery at N = 1000, 2000, 5000, 10000 (k=2 factors), with targets from
independent 5e6-draw simulations (no inverse crime).

Per N: report calibration seconds, forward-equivalents (calibration time /
one forward-pass time), identified alternatives (target share > 3e-4), share
mass covered, recovery max/median on identified alternatives.

Run:  python experiments/exp19_calibration_scaling/run_calibration_scaling.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    win_probabilities_factor,
)
from run_ghk_benchmark import make_problem, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def main():
    rng = np.random.default_rng(SEED)
    rows = ["N,seconds,forward_equivalents,identified,mass_identified,"
            "recovery_max,recovery_median,forward_match"]
    F, W = hermite_nodes(2)
    for n in (1000, 2000, 5000, 10000):
        mu, V, D = make_problem(n, 2, rng, spread=1.5)
        mu -= mu.mean()
        target = mc_shares(mu, V, D, 5_000_000, seed=90 + n % 97)
        target = np.maximum(target, 1e-9)
        target /= target.sum()

        t0 = time.perf_counter()
        p_fwd = win_probabilities_factor(-mu, V, D, F, W)
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        a_hat = abilities_from_probabilities_factor(target, V, D, F, W)
        t_cal = time.perf_counter() - t0

        back = win_probabilities_factor(a_hat, V, D, F, W)
        ident = target > 3e-4
        fmatch = np.abs(back[ident] - target[ident]).max()
        util = -a_hat
        err = np.abs((util - util[ident].mean()) - (mu - mu[ident].mean()))
        rec_max = err[ident].max()
        rec_med = float(np.median(err[ident]))
        print(f"N={n:>6}: calibrated in {t_cal:6.0f}s "
              f"({t_cal/t_fwd:4.1f} forward-equivalents of {t_fwd:.1f}s); "
              f"identified {ident.sum()}/{n} ({100*target[ident].sum():.1f}% mass); "
              f"recovery max {rec_max:.3f} median {rec_med:.4f}; "
              f"forward-match {fmatch:.1e}")
        rows.append(f"{n},{t_cal:.1f},{t_cal/t_fwd:.2f},{int(ident.sum())},"
                    f"{target[ident].sum():.4f},{rec_max:.4f},{rec_med:.5f},"
                    f"{fmatch:.3e}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
