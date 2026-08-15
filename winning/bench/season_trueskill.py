"""Season showdown: exact N-way updates versus reference TrueSkill.

Winner-only observations, matched scales: skills ~ N(0,1), performance
noise beta = 1, no dynamics, no draws. TrueSkill (the reference `trueskill`
package, Herbrich-Minka-Graepel model) receives the same winner-only
information as ranks [0, 1, 1, ..., 1]; the exact N-way update uses the
shared-field moments. Also included: the naive winner-beats-each-loser
pairwise decomposition.

    python -m winning.bench.season_trueskill
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..ratings import pairwise_update_winner, update_winner

try:
    import trueskill
except ImportError:                        # pragma: no cover
    trueskill = None


def main(P=200, field=20, races=1500, beta2=1.0, seed=7):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.0, P)
    m_ex = np.zeros(P); v_ex = np.full(P, 1.0)
    m_pw = np.zeros(P); v_pw = np.full(P, 1.0)
    env = trueskill.TrueSkill(mu=0.0, sigma=1.0, beta=np.sqrt(beta2),
                              tau=0.0, draw_probability=0.0)
    ts_ratings = [env.create_rating() for _ in range(P)]
    checkpoints = {100, 300, 700, 1500}
    print(f"{'races':>6} {'exact rho':>10} {'TS rho':>7} {'pair rho':>9} "
          f"{'exact rmse':>11} {'TS rmse':>8} {'pair rmse':>10}")
    for r in range(1, races + 1):
        idx = rng.choice(P, size=field, replace=False)
        perf = truth[idx] + np.sqrt(beta2) * rng.standard_normal(field)
        w = int(np.argmax(perf))
        out = update_winner(m_ex[idx], v_ex[idx], w, beta2)
        m_ex[idx], v_ex[idx] = out[0], out[1]
        out = pairwise_update_winner(m_pw[idx], v_pw[idx], w, beta2)
        m_pw[idx], v_pw[idx] = out[0], out[1]
        groups = [(ts_ratings[i],) for i in idx]
        ranks = [0 if j == w else 1 for j in range(field)]
        new = env.rate(groups, ranks=ranks)
        for j, i in enumerate(idx):
            ts_ratings[i] = new[j][0]
        if r in checkpoints:
            m_ts = np.array([rt.mu for rt in ts_ratings])
            row = [r]
            for m in (m_ex, m_ts, m_pw):
                rho = spearmanr(m, truth).statistic
                rmse = np.sqrt(np.mean(
                    (m - m.mean() - (truth - truth.mean())) ** 2))
                row += [rho, rmse]
            print(f"{row[0]:>6} {row[1]:>10.3f} {row[3]:>7.3f} "
                  f"{row[5]:>9.3f} {row[2]:>11.3f} {row[4]:>8.3f} "
                  f"{row[6]:>10.3f}")


if __name__ == "__main__":
    main()
