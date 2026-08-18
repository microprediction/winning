"""Full-ranking seasons: exact ordered-statistics updates versus TrueSkill.

Two seasons. The homogeneous season is TrueSkill's home model (one global
noise scale): there the exact update reproduces TrueSkill to ~1e-3 per
rating, which validates both -- TrueSkill's EP is essentially exact on its
own generative model, so it cannot be beaten there. The heteroskedastic
season gives half the field consistent noise (sd 0.5) and half erratic
(sd 1.5): the exact update takes per-player scales natively, TrueSkill is
forced to a single beta, and the gap widens with data.

    python -m winning.bench.season_ranked
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..ratings import update_ranking_exact

import trueskill


def _rmse(m, truth):
    return float(np.sqrt(np.mean((m - m.mean() - (truth - truth.mean())) ** 2)))


def season(noise_sd, P=200, field=20, races=1500, seed=7, label=""):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1.0, P)
    noise_sd = np.broadcast_to(np.asarray(noise_sd, dtype=float), (P,))
    m_ex = np.zeros(P)
    v_ex = np.full(P, 1.0)
    beta_ts = float(np.sqrt(np.mean(noise_sd ** 2)))
    env = trueskill.TrueSkill(mu=0.0, sigma=1.0, beta=beta_ts,
                              tau=0.0, draw_probability=0.0)
    ts = [env.create_rating() for _ in range(P)]
    print(f"\n{label}")
    print(f"{'races':>6} {'exact rho':>10} {'TS rho':>7} "
          f"{'exact rmse':>11} {'TS rmse':>8}")
    for r in range(1, races + 1):
        idx = rng.choice(P, size=field, replace=False)
        perf = truth[idx] + noise_sd[idx] * rng.standard_normal(field)
        order = np.argsort(-perf)
        m_ex[idx], v_ex[idx] = update_ranking_exact(
            m_ex[idx], v_ex[idx], list(order), beta2=noise_sd[idx] ** 2)
        groups = [(ts[i],) for i in idx]
        ranks = np.empty(field, dtype=int)
        ranks[order] = np.arange(field)
        new = env.rate(groups, ranks=list(ranks))
        for j, i in enumerate(idx):
            ts[i] = new[j][0]
        if r in (100, 300, 700, 1500):
            m_ts = np.array([t.mu for t in ts])
            print(f"{r:>6} {spearmanr(m_ex, truth).statistic:>10.3f} "
                  f"{spearmanr(m_ts, truth).statistic:>7.3f} "
                  f"{_rmse(m_ex, truth):>11.3f} {_rmse(m_ts, truth):>8.3f}")


def main():
    season(1.0, label="Homogeneous noise (TrueSkill's home model; expect a tie)")
    P = 200
    noise_sd = np.where(np.arange(P) % 2 == 0, 0.5, 1.5)
    season(noise_sd, P=P,
           label="Heteroskedastic noise (per-player scales; TrueSkill "
                 "restricted to one beta)")


if __name__ == "__main__":
    main()
