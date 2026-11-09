"""A win node observed two ways: as a market price and as a winner.

One factor-correlated race. The belief over skills is N(m, diag v); the
market quotes win prices made from a noisy view of the skills
(prices = race(-(s + eta)), eta ~ N(0, tau2)); the race is then run and
its winner observed. update_race folds in the price by inversion
(abilities_from_race, then a conjugate Gaussian update on the contrast
space) and the winner by the exact n-ary likelihood message. We measure
the posterior over the skill contrasts against the truth under each
evidence set, averaged over seeded races.

Reproduce: OMP_NUM_THREADS=1 python research/win_nodes/exp1_two_evidence/run_two_evidence.py
"""
import json, os, time
from itertools import combinations
import numpy as np

from winning.factor.races import race_probabilities
from winning.ratings.market import update_race

N, RANK, TAU2, BETA2, SEEDS = 8, 1, 0.25, 1.0, 400
HERE = os.path.dirname(os.path.abspath(__file__))


def centred(x):
    x = np.asarray(x, float)
    return x - x.mean()


def write_json_atomic(path, obj):
    """Sibling temp file then rename: a crash mid-write leaves no truncated file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def main():
    rng = np.random.default_rng(20261107)
    err = {"prior": [], "outcome": [], "price": [], "both": []}
    var = {"prior": [], "outcome": [], "price": [], "both": []}
    hit_winner = {"prior": [], "outcome": [], "price": [], "both": []}
    t0 = time.time()
    run_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime(t0))}-{os.getpid()}"
    # raw per-seed rows go to this run's own append-only log as they are
    # produced (created exclusively, so no run ever writes into another's)
    log_path = os.path.join(HERE, f"seeds_{run_id}.jsonl")
    log = open(log_path, "x")
    for seed in range(SEEDS):
        s = rng.normal(size=N)                                   # true skills (max wins)
        V = rng.normal(scale=0.6, size=(N, RANK))                # shared-factor loadings
        D = np.full(N, BETA2)
        m0, v0 = np.zeros(N), np.ones(N)
        # the market: a noisy view of s, priced under the same race model
        eta = rng.normal(scale=np.sqrt(TAU2), size=N)
        p_market = np.asarray(race_probabilities(-(s + eta), V=V, D=D))
        # the race: performances s + V f + sqrt(D) eps, max wins
        f = rng.normal(size=RANK); eps = rng.normal(size=N)
        perf = s + V @ f + np.sqrt(D) * eps
        winner = int(np.argmax(perf))
        posts = {
            "prior": (m0, v0),
            "outcome": update_race(m0, v0, winner=winner, V=V, beta2=BETA2)[:2],
            "price": update_race(m0, v0, p_market=p_market, tau2=TAU2, V=V, beta2=BETA2)[:2],
            "both": update_race(m0, v0, winner=winner, p_market=p_market, tau2=TAU2, V=V, beta2=BETA2)[:2],
        }
        # a second, independent running of the same race to score the posteriors
        f2 = rng.normal(size=RANK); eps2 = rng.normal(size=N)
        winner2 = int(np.argmax(s + V @ f2 + np.sqrt(D) * eps2))
        row = {"run_id": run_id, "seed": seed}
        for k, (m, v) in posts.items():
            e = float(np.sqrt(np.mean((centred(m) - centred(s)) ** 2)))
            vv = float(np.mean(v))
            pred = np.asarray(race_probabilities(-np.asarray(m), V=V, D=D + np.asarray(v)))
            ls = float(np.log(pred[winner2]))
            err[k].append(e); var[k].append(vv); hit_winner[k].append(ls)
            row[k] = {"rmse": e, "var": vv, "log_score": ls}
        log.write(json.dumps(row) + "\n"); log.flush()
    log.close()

    def se(x):
        return float(np.std(np.asarray(x), ddof=1) / np.sqrt(SEEDS))
    samples = {"rmse_contrast": err, "mean_posterior_var": var,
               "mean_log_score_next_winner": hit_winner}
    out = {"n": N, "rank": RANK, "tau2": TAU2, "beta2": BETA2, "seeds": SEEDS,
           "rmse_contrast": {k: float(np.mean(e)) for k, e in err.items()},
           "mean_posterior_var": {k: float(np.mean(x)) for k, x in var.items()},
           "mean_log_score_next_winner": {k: sum(hit_winner[k]) / SEEDS for k in hit_winner},
           "standard_errors": {m: {k: se(x) for k, x in smp.items()} for m, smp in samples.items()},
           "paired_difference_standard_errors": {
               m: {f"{a}_minus_{b}": se(np.asarray(smp[a]) - np.asarray(smp[b]))
                   for a, b in combinations(smp, 2)}
               for m, smp in samples.items()},
           "run_id": run_id, "raw_log": os.path.basename(log_path),
           "seconds": time.time() - t0}
    # never overwrite an existing results.json silently: a rerun gets a stamped name
    path = os.path.join(HERE, "results.json")
    if os.path.exists(path):
        path = os.path.join(HERE, f"results_{run_id}.json")
    write_json_atomic(path, out)
    print("summary ->", path, " raw ->", log_path)
    for k in ("prior", "outcome", "price", "both"):
        print(f"{k:8s} rmse {out['rmse_contrast'][k]:.4f}  var {out['mean_posterior_var'][k]:.4f}"
              f"  next-winner log score {out['mean_log_score_next_winner'][k]:.4f}")
    print(f"{out['seconds']:.1f}s")


if __name__ == "__main__":
    main()
