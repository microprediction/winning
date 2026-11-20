"""Wall time of one win-node evaluation against counting posterior draws.

For a hundred alternatives with a factor-plus-diagonal posterior of
rank r = 1..5, time (a) race_probabilities at its defaults and (b)
drawing M samples from the same posterior and counting argmax
frequencies, and report how many draws fit in the time of one
evaluation. Single thread, median of repeated runs. The posteriors are
those of a linear sampler: arms with r random features, prior
N(0, V V' + 0.25 I), 2,000 pulls spread over the arms.

Reproduce: python research/win_nodes/timing/run_timing.py
The script pins every BLAS/OpenMP pool to one thread before numpy is
imported. Every raw duration goes to a per-run append-only log
(timings_<run>.jsonl) as it is measured; results.json is the summary.
Timings are machine dependent; results.json records the platform.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import json                                                # noqa: E402
import platform                                            # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

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


def median_time(f, log, **meta):
    ts = []
    for i in range(REPEAT):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
        log.write(json.dumps(dict(meta, repeat=i, seconds=ts[-1])) + "\n")
        log.flush()
    return float(np.median(ts))


def blas_threads():
    """Effective thread limits of the loaded BLAS/OpenMP pools."""
    try:
        from threadpoolctl import threadpool_info
        return [dict(api=i.get("user_api"), lib=i.get("internal_api"),
                     threads=i.get("num_threads")) for i in threadpool_info()]
    except ImportError:
        return "threadpoolctl unavailable; env pinned to 1"



def count_draws(m, B, d, M, rng):
    X = m + rng.normal(size=(M, B.shape[1])) @ B.T + rng.normal(size=(M, N)) * np.sqrt(d)
    return np.bincount(X.argmax(1), minlength=N) / M


def main():
    rng = np.random.default_rng(20261119)
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + f"-{os.getpid()}"
    log = open(os.path.join(HERE, f"timings_{run_id}.jsonl"), "x")
    rows = []
    for r in range(1, 6):
        m, B, d = posterior(r, rng)
        race_probabilities(-m, V=-B, D=d)                 # warm up
        t_eval = median_time(lambda: race_probabilities(-m, V=-B, D=d),
                             log, run=run_id, rank=r, what="eval")
        t_draw = median_time(lambda: count_draws(m, B, d, 20_000, rng),
                             log, run=run_id, rank=r,
                             what="20000 draws") / 20_000
        rows.append(dict(rank=r, eval_seconds=t_eval, seconds_per_draw=t_draw,
                         draws_in_one_eval=int(t_eval / t_draw),
                         seconds_200k_draws=200_000 * t_draw))
        print(f"rank {r}: eval {t_eval * 1e3:7.1f} ms   200k draws {200_000 * t_draw:6.2f} s"
              f"   draws per eval {int(t_eval / t_draw):,}")
    out = dict(n=N, pulls=PULLS, repeat=REPEAT, rows=rows,
               platform=platform.platform(), processor=platform.processor(),
               python=platform.python_version(), numpy=np.__version__,
               blas_threads=blas_threads(), run=run_id,
               raw_log=os.path.basename(log.name))
    log.close()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
