"""Forward-pass timing and error of race_probabilities on GLiClass-shaped fields.

PR #175 review item 4. A field is N labels with logits ~ 2 N(0,1) (higher
wins, so mu = -logits), unit label embeddings E (dim 16, random) with
cosine Gram K = E E^T, and the HYPOTHESISED covariance

    Sigma = rho * V V^T + diag(1 - rho * rowsum(V^2)),
    V = Q_r sqrt(Lambda_r)   (rank-r eigentruncation of K, NOT K itself)

with rho in {0.5, 0.9} (0.9 is a sharper field that can push the node
rule onto its 8,192-node Sobol side). rho and the unit scale are assumptions to be calibrated,
not outputs of the checkpoint. Error is max |p - p_mc| against a 4e6-draw
Monte Carlo of the same Sigma (MC standard error <= 2.5e-4).

Run from the repo root:  python research/applications/gliclass_head_timing.py
Writes research/applications/gliclass_head_timing.json.
"""
import json
import platform
import time

import numpy as np

import winning
from winning.factor.core import qmc_nodes
from winning.factor.races import race_probabilities
from winning.rustconfig import rust_active

RHOS, DIM, DRAWS = (0.5, 0.9), 16, 4_000_000


def field(n, r, rho, rng):
    E = rng.standard_normal((n, DIM))
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    K = E @ E.T
    lam, Q = np.linalg.eigh(K)
    idx = np.argsort(lam)[::-1][:r]
    V = Q[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
    V = np.sqrt(rho) * V
    D = 1.0 - np.sum(V**2, axis=1)
    logits = 2.0 * rng.standard_normal(n)
    return -logits, V, D


def mc(mu, V, D, rng, chunk=200_000):
    n, r = V.shape
    wins = np.zeros(n)
    for _ in range(DRAWS // chunk):
        X = mu + rng.standard_normal((chunk, r)) @ V.T \
            + rng.standard_normal((chunk, n)) * np.sqrt(D)
        wins += np.bincount(np.argmin(X, axis=1), minlength=n)
    return wins / DRAWS


def main():
    rng = np.random.default_rng(175)
    rows = []
    for rho in RHOS:
      for n, r, reps in [(3, 1, 21), (10, 1, 7), (10, 2, 5), (50, 1, 5), (50, 2, 3)]:
        mu, V, D = field(n, r, rho, rng)
        race_probabilities(mu, V=V, D=D)                      # warm-up
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            p = race_probabilities(mu, V=V, D=D)
            ts.append(time.perf_counter() - t0)
        err = float(np.max(np.abs(p - mc(mu, V, D, rng))))
        rows.append(dict(rho=rho, N=n, rank=r, reps=reps,
                         median_s=float(np.median(ts)), max_abs_err_vs_mc=err))
        print(rows[-1], flush=True)
    # The capped side of the node rule: what a field sharp enough to cross
    # GH_RULE's threshold costs (8,192 scrambled Sobol nodes at rank 2).
    F, W = qmc_nodes(2, m=13)
    for n, reps in [(10, 3), (50, 3)]:
        mu, V, D = field(n, 2, 0.9, rng)
        race_probabilities(mu, V=V, D=D, F=F, W=W)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            p = race_probabilities(mu, V=V, D=D, F=F, W=W)
            ts.append(time.perf_counter() - t0)
        err = float(np.max(np.abs(p - mc(mu, V, D, rng))))
        rows.append(dict(rho=0.9, N=n, rank=2, reps=reps, nodes="sobol8192 forced",
                         median_s=float(np.median(ts)), max_abs_err_vs_mc=err))
        print(rows[-1], flush=True)
    out = dict(backend="rust/fastrace" if rust_active() else "pure numpy/scipy",
               winning_version=getattr(winning, "__version__", "?"),
               python=platform.python_version(), machine=platform.platform(),
               node_rule="race_probabilities default (GH_RULE tensor or 8192 Sobol)",
                mc_draws=DRAWS, rows=rows)
    with open("research/applications/gliclass_head_timing.json", "w") as f:
        json.dump(out, f, indent=2)
    print(out["backend"])


if __name__ == "__main__":
    main()
