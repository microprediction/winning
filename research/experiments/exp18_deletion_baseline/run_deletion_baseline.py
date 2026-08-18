"""Experiment 18: the top-two direct-simulation deletion baseline
(fourth referee round).

A single utility draw determines the winner w and runner-up s. Deleting
alternative i changes nothing unless i = w, in which case the winner becomes
s. So ALL N single-removal share vectors follow from one simulation by
counting winners c_j and winner/runner-up pairs r_ij:

    q[i, j] = (c_j + r_ij) / R   for j != i,   q[i, i] = 0.

Cost O(RN + N^2) -- an intelligently reused simulation, and the fair
comparator for the conditional-field removal ensemble (which the paper
previously compared only against naive per-removal recomputation).

Protocol (N=200, k=2, exp13's problem):
  1. Time the conditional-field ensemble (O(QN^2L), deterministic).
  2. Run top-two simulation at wall time matched to (1).
  3. Score both against an independent high-accuracy top-two reference
     (10^8 draws), on the deletion rows of the 20 highest-share
     alternatives, max abs error over row entries.

Run:  python experiments/exp18_deletion_baseline/run_deletion_baseline.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402
from run_ghk_benchmark import make_problem  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def top_two_deletion_matrix(mu, V, D, n_draws, seed, mem_budget_bytes=1.5e9):
    """q[i, j] = P(j wins | i deleted) from winner/runner-up counts."""
    n, k = V.shape
    chunk = max(10_000, int(mem_budget_bytes / (n * 8 * 4)))
    rng = np.random.default_rng(seed)
    c = np.zeros(n)
    r = np.zeros((n, n))
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        f = rng.standard_normal((m, k))
        U = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((m, n))
        part = np.argpartition(-U, 1, axis=1)[:, :2]
        u2 = np.take_along_axis(U, part, axis=1)
        swap = u2[:, 0] < u2[:, 1]
        w = np.where(swap, part[:, 1], part[:, 0])
        sec = np.where(swap, part[:, 0], part[:, 1])
        c += np.bincount(w, minlength=n)
        np.add.at(r, (w, sec), 1.0)
        done += m
    q = (c[None, :] + r) / n_draws
    np.fill_diagonal(q, 0.0)
    return q / q.sum(axis=1, keepdims=True)


def main():
    rng = np.random.default_rng(SEED)
    n = 200
    mu, V, D = make_problem(n, 2, rng, spread=1.0)
    F, W = hermite_nodes(2)
    rows = ["quantity,value"]

    print("1. conditional-field removal ensemble (deterministic)")
    t0 = time.perf_counter()
    _, q_field = win_probabilities_factor(-mu, V, D, F, W, return_deletions=True)
    t_field = time.perf_counter() - t0
    print(f"   {t_field:.1f}s for all {n} removals")
    rows.append(f"field_seconds,{t_field:.2f}")

    print("2. top-two simulation at matched wall time")
    t0 = time.perf_counter()
    probe = 500_000
    top_two_deletion_matrix(mu, V, D, probe, seed=1)
    rate = probe / (time.perf_counter() - t0)
    n_draws = int(rate * t_field)
    t0 = time.perf_counter()
    q_mc = top_two_deletion_matrix(mu, V, D, n_draws, seed=2)
    t_mc = time.perf_counter() - t0
    print(f"   {n_draws/1e6:.1f}M draws in {t_mc:.1f}s")
    rows += [f"toptwo_draws,{n_draws}", f"toptwo_seconds,{t_mc:.2f}"]

    print("3. independent 1e8-draw top-two reference")
    q_ref = top_two_deletion_matrix(mu, V, D, 100_000_000, seed=9)
    p_menu = win_probabilities_factor(-mu, V, D, F, W)
    heavy = np.argsort(p_menu)[::-1][:20]
    e_field = max(np.abs(q_field[i] - q_ref[i]).max() for i in heavy)
    e_mc = max(np.abs(q_mc[i] - q_ref[i]).max() for i in heavy)
    print(f"   max err over top-20 deletion rows: field {e_field:.1e}, "
          f"top-two MC {e_mc:.1e} (reference noise ~1e-4/sqrt(p))")
    rows += [f"field_max_err,{e_field:.3e}", f"toptwo_max_err,{e_mc:.3e}"]

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
