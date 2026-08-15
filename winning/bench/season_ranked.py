"""Full-ranking season: exact sequential N-way updates versus reference
TrueSkill on its home turf (complete finish orders observed).

    python -m winning.bench.season_ranked
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..ratings import update_ranking

import trueskill


def main(P=200, field=20, races=1500, beta2=1.0, seed=7):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.0, P)
    m_ex = np.zeros(P); v_ex = np.full(P, 1.0)
    env = trueskill.TrueSkill(mu=0.0, sigma=1.0, beta=np.sqrt(beta2),
                              tau=0.0, draw_probability=0.0)
    ts = [env.create_rating() for _ in range(P)]
    checkpoints = {100, 300, 700, 1500}
    print(f"{'races':>6} {'exact rho':>10} {'TS rho':>7} "
          f"{'exact rmse':>11} {'TS rmse':>8}")
    for r in range(1, races + 1):
        idx = rng.choice(P, size=field, replace=False)
        perf = truth[idx] + np.sqrt(beta2) * rng.standard_normal(field)
        order = np.argsort(-perf)
        m_ex[idx], v_ex[idx] = update_ranking(m_ex[idx], v_ex[idx],
                                              order, beta2)
        groups = [(ts[i],) for i in idx]
        ranks = np.empty(field, dtype=int)
        ranks[order] = np.arange(field)
        new = env.rate(groups, ranks=list(ranks))
        for j, i in enumerate(idx):
            ts[i] = new[j][0]
        if r in checkpoints:
            m_ts = np.array([rt.mu for rt in ts])
            out = [r]
            for m in (m_ex, m_ts):
                out += [spearmanr(m, truth).statistic,
                        np.sqrt(np.mean((m - m.mean()
                                         - (truth - truth.mean())) ** 2))]
            print(f"{out[0]:>6} {out[1]:>10.3f} {out[3]:>7.3f} "
                  f"{out[2]:>11.3f} {out[4]:>8.3f}")


if __name__ == "__main__":
    main()
