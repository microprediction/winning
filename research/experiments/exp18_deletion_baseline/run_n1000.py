"""Experiment 18b: top-two deletion baseline at N=1000 (scaling check).

The field ensemble costs O(QN^2L); top-two simulation costs O(N) per draw, so
at matched wall time its draw count grows with N and its error falls ~N^{-1/2}
while the deterministic error stays flat. This measures where the crossover
sits. Scored on the 20 highest-share deletion rows vs an independent
1e8-draw top-two reference.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes, win_probabilities_factor
from run_ghk_benchmark import make_problem
from run_deletion_baseline import top_two_deletion_matrix

rng = np.random.default_rng(21)
n = 1000
mu, V, D = make_problem(n, 2, rng, spread=1.5)
F, W = hermite_nodes(2)

t0 = time.perf_counter()
_, q_field = win_probabilities_factor(-mu, V, D, F, W, return_deletions=True)
t_field = time.perf_counter() - t0
print(f"field ensemble: {t_field:.0f}s for all {n} removals")

probe = 200_000
t0 = time.perf_counter(); top_two_deletion_matrix(mu, V, D, probe, seed=1)
rate = probe / (time.perf_counter() - t0)
n_draws = int(rate * t_field)
t0 = time.perf_counter()
q_mc = top_two_deletion_matrix(mu, V, D, n_draws, seed=2)
print(f"top-two matched: {n_draws/1e6:.0f}M draws in {time.perf_counter()-t0:.0f}s")

q_ref = top_two_deletion_matrix(mu, V, D, 100_000_000, seed=9)
p_menu = win_probabilities_factor(-mu, V, D, F, W)
heavy = np.argsort(p_menu)[::-1][:20]
e_f = max(np.abs(q_field[i] - q_ref[i]).max() for i in heavy)
e_m = max(np.abs(q_mc[i] - q_ref[i]).max() for i in heavy)
print(f"N=1000 max err over top-20 deletion rows: field {e_f:.1e}, top-two MC {e_m:.1e}")
