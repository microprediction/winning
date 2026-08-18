"""Season simulation: exact N-way updates versus the pairwise decomposition.

100 players with fixed true skills; each race draws 10 players at random,
performances = skill + N(0, beta2) noise, winner observed. Ratings updated
race by race under (a) exact N-way moments and (b) winner-beats-each-loser
pairwise probit updates. Reported: Spearman correlation and RMSE of rating
means against true skills as the season progresses.

    python -m winning.bench.season
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..ratings import pairwise_update_winner, update_winner


def main(P=100, field=10, races=800, beta2=1.0, seed=7):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.0, P)
    state = {
        "exact": (np.zeros(P), np.full(P, 1.0)),
        "pairwise": (np.zeros(P), np.full(P, 1.0)),
    }
    checkpoints = [50, 100, 200, 400, 800]
    print(f"{'races':>6} {'exact rho':>10} {'pair rho':>9} "
          f"{'exact rmse':>11} {'pair rmse':>10}")
    for r in range(1, races + 1):
        idx = rng.choice(P, size=field, replace=False)
        perf = truth[idx] + np.sqrt(beta2) * rng.standard_normal(field)
        w = int(np.argmax(perf))
        for name, upd in (("exact", update_winner),
                          ("pairwise", pairwise_update_winner)):
            m, v = state[name]
            out = upd(m[idx], v[idx], w, beta2)
            m[idx], v[idx] = out[0], out[1]
        if r in checkpoints:
            row = [r]
            for name in ("exact", "pairwise"):
                m, _ = state[name]
                rho = spearmanr(m, truth).statistic
                rmse = np.sqrt(np.mean((m - m.mean() - (truth - truth.mean()))**2))
                row += [rho, rmse]
            print(f"{row[0]:>6} {row[1]:>10.3f} {row[3]:>9.3f} "
                  f"{row[2]:>11.3f} {row[4]:>10.3f}")


if __name__ == "__main__":
    main()
