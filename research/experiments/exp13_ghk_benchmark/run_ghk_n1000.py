"""GHK error constant measured at N=1000 (eighth review round).

The matched-accuracy extrapolation previously used the R=1000 error constant
measured at N=200 (6.8e-3). This measures it at N=1000 directly on exp13's
problem sequence: err 9.2e-3 in 59s. The constant grows mildly with N, so
the N=200 value was favorable to the baseline; the fifty-hour figure is an
order-of-magnitude illustration conditioned on this constant.

Run:  python experiments/exp13_ghk_benchmark/run_ghk_n1000.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ghk_benchmark import ghk_all_shares, make_problem, mc_shares  # noqa: E402

rng = np.random.default_rng(21)
for n in (5, 20, 50, 200):
    make_problem(n, 2, rng)                       # advance rng as exp13 does
mu, V, D = make_problem(1000, 2, rng, spread=1.5)
truth = mc_shares(mu, V, D, 2_000_000, seed=9)
t0 = time.perf_counter()
p = ghk_all_shares(mu, V, D, R=1000)
print(f"GHK R=1000 at N=1000: err {np.abs(p - truth).max():.2e}, "
      f"{time.perf_counter() - t0:.0f}s")
