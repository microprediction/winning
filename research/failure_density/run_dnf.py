"""Does a failure lump in the performance density beat the heuristics?

The bandits lane measured that with MECHANICAL (ability-independent)
retirements, censoring is near-optimal and no new machinery is needed;
but once retirement probability rises as ability falls, naive
(retirement = slow) wins the ordering while censoring wins the
magnitudes, and neither dominates. This asks whether modelling the
failure explicitly -- winning.factor.races.failure_base -- beats both,
using WINNER-ONLY observations where the effect is cleanest: when a
fast car breaks, a slow car wins, and a Gaussian model has no way to
say so except by crediting the winner's ability.

Scored by rank correlation and by RMSE of the centred estimate.
"""
import numpy as np
from scipy.stats import spearmanr

from winning.factor.races import failure_base
from winning.ratings.nway import update_winner

M, RACES, SEEDS = 10, 220, 12
BETA2 = 1.0


def simulate(rng, coupling, base_q):
    s = rng.normal(size=M)
    q = np.clip(base_q - coupling * s, 0.01, 0.85)     # weak cars break more
    winners, fields = [], []
    for _ in range(RACES):
        alive = rng.random(M) > q
        if alive.sum() < 2:
            continue
        perf = s + np.sqrt(BETA2) * rng.normal(size=M)
        perf[~alive] = -np.inf                          # retired: no result
        winners.append(int(np.argmax(perf)))
        fields.append(alive)
    return s, winners, fields


def rate(winners, fields, mode, q_model):
    m, v = np.zeros(M), np.ones(M)
    for w, alive in zip(winners, fields):
        if mode == "censored":
            # only finishers are in the field the model sees
            idx = np.where(alive)[0]
            if len(idx) < 2 or w not in idx:
                continue
            mm, vv, _ = update_winner(m[idx], v[idx], int(np.where(idx == w)[0][0]),
                                      beta2=BETA2)
            m[idx], v[idx] = mm, vv
        elif mode == "naive":
            # everyone in the field; retirement reads as slowness
            m, v, _ = update_winner(m, v, w, beta2=BETA2)
        else:  # failure density
            m, v, _ = update_winner(m, v, w, beta2=BETA2,
                                    base=failure_base(q_model))
    return m


print(f"{'coupling':>9} {'base_q':>7} | {'naive':>16} {'censored':>16} {'failure-lump':>16}")
print(f"{'':>9} {'':>7} | {'rho    rmse':>16} {'rho    rmse':>16} {'rho    rmse':>16}")
for coupling in (0.0, 0.10, 0.20):
    for base_q in (0.25,):
        acc = {k: [] for k in ("naive", "censored", "failure")}
        for seed in range(SEEDS):
            rng = np.random.default_rng(1000 + seed)
            s, winners, fields = simulate(rng, coupling, base_q)
            sc = s - s.mean()
            for mode in acc:
                mh = rate(winners, fields, mode, base_q)
                mh = mh - mh.mean()
                acc[mode].append((spearmanr(mh, sc).statistic,
                                  float(np.sqrt(np.mean((mh - sc) ** 2)))))
        cells = []
        for mode in ("naive", "censored", "failure"):
            A = np.array(acc[mode])
            cells.append(f"{np.mean(A[:,0]):.3f} {np.mean(A[:,1]):.3f}")
        print(f"{coupling:9.2f} {base_q:7.2f} | {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")
