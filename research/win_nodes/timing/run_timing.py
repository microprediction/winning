"""Wall time of one win-node evaluation against counting posterior draws.

For a hundred alternatives with a factor-plus-diagonal posterior of
rank r = 1..5, time (a) race_probabilities at its defaults and (b)
drawing M samples from the same posterior and counting argmax
frequencies, and report how many draws fit in the time of one
evaluation. Single thread, median of repeated runs. The posteriors are
those of a linear sampler: arms with r random features, prior
N(0, V V' + 0.25 I), 2,000 pulls spread over the arms.

Reproduce: OMP_NUM_THREADS=1 python research/win_nodes/timing/run_timing.py
Timings are machine dependent; results.json records the platform.
"""
import json
import os
import platform
import time

import numpy as np

from winning.factor.races import race_probabilities

N, PULLS, REPEAT = 100, 2000, 7
HERE = os.path.dirname(os.path.abspath(__file__))


def posterior(r, rng):
    """Bandit posterior after PULLS uniform pulls, in factor form."""
    V = rng.normal(scale=1 / np.sqrt(r), size=(N, r))
    d0 = np.full(N, 0.25)
    counts = rng.multinomial(PULLS, np.full(N, 1 / N)).astype(float)
    theta = V @ rng.normal(size=r) + np.sqrt(d0) * rng.normal(size=N)
    sums = counts * theta + np.sqrt(counts) * rng.normal(size=N)
    lam = counts                                          # sigma_n = 1
    rr = 1.0 / (1.0 + d0 * lam)
    H = np.eye(r) + (V.T * (lam * rr)) @ V
    Hi = np.linalg.inv(H)
    w, U = np.linalg.eigh(Hi)
    B = (rr[:, None] * V) @ (U * np.sqrt(np.maximum(w, 0)))
    d = d0 * rr
    S = np.diag(d) + B @ B.T
    m = S @ sums                                          # prior mean zero
    return m, B, d


def median_time(f):
    ts = []
    for _ in range(REPEAT):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def count_draws(m, B, d, M, rng):
    X = m + rng.normal(size=(M, B.shape[1])) @ B.T + rng.normal(size=(M, N)) * np.sqrt(d)
    return np.bincount(X.argmax(1), minlength=N) / M


def main():
    rng = np.random.default_rng(20261119)
    rows = []
    for r in range(1, 6):
        m, B, d = posterior(r, rng)
        race_probabilities(-m, V=-B, D=d)                 # warm up
        t_eval = median_time(lambda: race_probabilities(-m, V=-B, D=d))
        t_draw = median_time(lambda: count_draws(m, B, d, 20_000, rng)) / 20_000
        rows.append(dict(rank=r, eval_seconds=t_eval, seconds_per_draw=t_draw,
                         draws_in_one_eval=int(t_eval / t_draw),
                         seconds_200k_draws=200_000 * t_draw))
        print(f"rank {r}: eval {t_eval * 1e3:7.1f} ms   200k draws {200_000 * t_draw:6.2f} s"
              f"   draws per eval {int(t_eval / t_draw):,}")
    out = dict(n=N, pulls=PULLS, repeat=REPEAT, rows=rows,
               platform=platform.platform(), processor=platform.processor(),
               python=platform.python_version(), numpy=np.__version__,
               threads=os.environ.get("OMP_NUM_THREADS"))
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
