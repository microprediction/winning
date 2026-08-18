"""Experiment 38: the Rao-Blackwellized all-N conditional simulator.

Train's convenient error partitioning conditions on part of the noise
and computes the remaining probability analytically. Under the factor
model the all-N version is: draw f and all idiosyncratic shocks, form
M_{-i} = max_{j != i} U_j (all N of them in O(N) from the top-two
statistics), and average

    p_i^(r) = Phi((mu_i + v_i' f - M_{-i}) / sqrt(D_i)),

which is unbiased for p_i coordinatewise, smooth apart from max-switch
kinks, and O(RN) for the whole vector. This is the strongest simple
simulation comparator for the shared-field lattice; this experiment
measures both it and raw winner frequencies against the lattice map.

Run:  python experiments/exp38_conditional_mc/run_conditional_mc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402
from run_ghk_benchmark import make_problem  # noqa: E402

HERE = Path(__file__).resolve().parent
N, K = 200, 2


def estimators(mu, V, D, R, rng):
    """Winner-frequency and conditional estimates from common draws."""
    sD = np.sqrt(D)
    freq = np.zeros(N)
    cond = np.zeros(N)
    chunk = 20_000
    done = 0
    while done < R:
        m = min(chunk, R - done)
        f = rng.standard_normal((m, K))
        loc = mu[None, :] + f @ V.T
        U = loc + sD[None, :] * rng.standard_normal((m, N))
        # top-two statistics give every M_{-i} in O(N)
        order = np.argpartition(U, N - 2, axis=1)[:, -2:]
        top_idx = order[np.arange(m), np.argmax(
            U[np.arange(m)[:, None], order], axis=1)]
        top_val = U[np.arange(m), top_idx]
        second_val = np.where(
            order[:, 0] == top_idx,
            U[np.arange(m), order[:, 1]],
            U[np.arange(m), order[:, 0]])
        M_minus = np.broadcast_to(top_val[:, None], (m, N)).copy()
        M_minus[np.arange(m), top_idx] = second_val
        freq += np.bincount(top_idx, minlength=N)
        cond += ndtr((loc - M_minus) / sD[None, :]).sum(axis=0)
        done += m
    return freq / R, cond / R


def main():
    rng = np.random.default_rng(38)
    mu, V, D = make_problem(N, K, rng, spread=1.0)
    mu -= mu.mean()
    F, W = hermite_nodes(K)
    ref = win_probabilities_factor(-mu, V, D, F, W, points=1501)

    lines = [f"share range [{ref.min():.2e}, {ref.max():.2e}]"]
    for R in (10_000, 100_000):
        freq, cond = estimators(mu, V, D, R, np.random.default_rng(100 + R))
        for name, p in (("winner-freq", freq), ("conditional", cond)):
            mask = ref > 1e-3
            with np.errstate(divide="ignore"):
                logerr = np.abs(np.log(np.maximum(p[mask], 1e-300))
                                - np.log(ref[mask])).max()
            lines.append(f"R={R:>6} {name:>12}: max|dp|={np.abs(p-ref).max():.2e} "
                         f"TV={0.5*np.abs(p-ref).sum():.2e} "
                         f"max|dlog| (p>1e-3)={logerr:.2e}")
    out = "\n".join(lines) + "\n"
    print(out)
    (HERE / "results.txt").write_text(out)


if __name__ == "__main__":
    main()
