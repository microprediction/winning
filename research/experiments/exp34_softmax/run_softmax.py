"""Experiment 34: the temperature-softmax simulation comparator.

The natural all-N machine-learning competitor to the lattice transform
is E[softmax((mu + Vf + sqrt(D) eps)/T)] with common draws: smooth,
O(RN) for all N outputs, and the construction used by heteroscedastic
classifiers (Collier et al. 2021) and perturbed optimizers (Berthet et
al. 2020). It is biased at any fixed T > 0 and noisy at finite R; this
experiment measures both against the lattice map.

Reference: lattice at L=1501, GH15 (self-converged; exp17/25).
Race is computed min-wins; softmax comparator uses -x/T accordingly.

Run:  python experiments/exp34_softmax/run_softmax.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

HERE = Path(__file__).resolve().parent
N, K = 200, 2


def softmax_mc(a, V, D, T, R, rng):
    acc = np.zeros(N)
    chunk = 20_000
    done = 0
    while done < R:
        m = min(chunk, R - done)
        f = rng.standard_normal((m, K))
        x = a[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((m, N))
        z = -x / T                       # min-wins: smallest x most likely
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        acc += (e / e.sum(axis=1, keepdims=True)).sum(axis=0)
        done += m
    return acc / R


def main():
    rng = np.random.default_rng(34)
    F, W = hermite_nodes(K)
    a = rng.normal(0, 1.0, N)
    a -= a.mean()
    V = rng.normal(0, np.sqrt(0.36 / K), (N, K))
    D = rng.uniform(0.6, 1.2, N)

    ref = win_probabilities_factor(a, V, D, F, W, points=1501)

    lines = [f"share range [{ref.min():.2e}, {ref.max():.2e}]"]
    for T in (1.0, 0.3, 0.1, 0.03):
        for R in (50_000, 200_000):
            p = softmax_mc(a, V, D, T, R, np.random.default_rng(1000 + int(T * 100)))
            mask = ref > 1e-3
            logerr = np.abs(np.log(p[mask]) - np.log(ref[mask])).max()
            abserr = np.abs(p - ref).max()
            tv = 0.5 * np.abs(p - ref).sum()
            lines.append(f"T={T:<4} R={R:>6}: max|dp|={abserr:.2e} "
                         f"TV={tv:.2e} max|dlog| (p>1e-3)={logerr:.2e}")
    out = "\n".join(lines) + "\n"
    print(out)
    (HERE / "results.txt").write_text(out)


if __name__ == "__main__":
    main()
