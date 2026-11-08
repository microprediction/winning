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
import numpy as np

from winning.factor.races import race_probabilities
from winning.ratings.market import update_race

N, RANK, TAU2, BETA2, SEEDS = 8, 1, 0.25, 1.0, 400
HERE = os.path.dirname(os.path.abspath(__file__))


def centred(x):
    x = np.asarray(x, float)
    return x - x.mean()


def main():
    rng = np.random.default_rng(20261107)
    err = {"prior": [], "outcome": [], "price": [], "both": []}
    var = {"prior": [], "outcome": [], "price": [], "both": []}
    hit_winner = {"prior": 0, "outcome": 0, "price": 0, "both": 0}
    t0 = time.time()
    for _ in range(SEEDS):
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
        for k, (m, v) in posts.items():
            err[k].append(float(np.sqrt(np.mean((centred(m) - centred(s)) ** 2))))
            var[k].append(float(np.mean(v)))
            pred = np.asarray(race_probabilities(-np.asarray(m), V=V, D=D + np.asarray(v)))
            hit_winner[k] += float(np.log(pred[winner2]))
    out = {"n": N, "rank": RANK, "tau2": TAU2, "beta2": BETA2, "seeds": SEEDS,
           "rmse_contrast": {k: float(np.mean(e)) for k, e in err.items()},
           "mean_posterior_var": {k: float(np.mean(x)) for k, x in var.items()},
           "mean_log_score_next_winner": {k: hit_winner[k] / SEEDS for k in hit_winner},
           "seconds": time.time() - t0}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    for k in ("prior", "outcome", "price", "both"):
        print(f"{k:8s} rmse {out['rmse_contrast'][k]:.4f}  var {out['mean_posterior_var'][k]:.4f}"
              f"  next-winner log score {out['mean_log_score_next_winner'][k]:.4f}")
    print(f"{out['seconds']:.1f}s")


if __name__ == "__main__":
    main()
