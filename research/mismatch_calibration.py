"""N=2 mismatches: which performance-distribution family gets upsets right?

At N=2 the win probability is the tail of the performance-difference
distribution at the talent gap — at big gaps the FAMILY is the prediction:
probit (TrueSkill) ~ exp(-x^2), logistic (Elo/Glicko) ~ exp(-x), and reality
(off days) usually fatter still. The lattice can carry an explicit off-day
mixture. Chess gives 121k games with wide gap coverage; buckets come from
the site's own predicted win probability (from ratings at game time), and
within each bucket we compare every system's mean predicted probability for
the site favorite against the empirical win rate (decisive games; draws
counted as half-wins in a separate column).

Run:  .venv/bin/python research/mismatch_calibration.py [months...]

Measured (July 2026, 6 months of Lichess 2013, 966k games; mean predicted
P(site favorite) vs empirical, decisive games, top two gap buckets):
  [0.90,0.95) n=35,509: empirical 0.8821 | site 0.9234 | TrueSkill 0.8780 |
              Thurstone 0.8386 | Elo 0.8099 | Glicko-2 0.7916
  [0.95,1.01) n=16,710: empirical 0.9356 | site 0.9683 | TrueSkill 0.9351 |
              Thurstone 0.9035 | Elo 0.8722 | Glicko-2 0.8646
Findings: (1) the site's plug-in ratings overrate big favorites by ~3.3
points at ~17 se — rock solid; a fitted temperature fixes it out-of-sample
(companion analysis) while heavy-tail families do not: attenuation from
rating measurement error, not fat performance tails. (2) TrueSkill with six
months of history calibrates the extremes almost perfectly WITHOUT any
patch — Bayesian uncertainty performs the attenuation endogenously; that is
the principled answer to "the temperature cure breaks consistency".
(3) Our rater still under-favors big favorites at chess data rates (0.9035
vs 0.9356) — adaptation-speed tuning territory, honestly noted.
"""

from __future__ import annotations

import numpy as np

from winning import EloRating, Glicko2Rating, ThurstoneRating
from winning.shims import TrueSkillRating
from winning.thurstonerating import _gauss_kernel

BUCKETS = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.01)]


def offday_kernel(unit=0.1, p_off=0.1, wide=3.0, half=200):
    g1 = np.zeros(2 * half + 1)
    k1 = _gauss_kernel(unit, 1.0)
    s = (len(g1) - len(k1)) // 2
    g1[s:s + len(k1)] = k1
    k2 = _gauss_kernel(unit, wide)
    g2 = np.zeros(2 * half + 1)
    s = (len(g2) - len(k2)) // 2
    g2[s:s + len(k2)] = k2
    k = (1 - p_off) * g1 + p_off * g2
    return k / k.sum()


def main():
    import sys

    from winning.benchmarks.chess import chess_events

    months = tuple(sys.argv[1:]) or ("2013-01",)
    events = chess_events(months=months)
    n_warm = int(len(events) * 0.2)
    print(f"chess {months[0]}..{months[-1]}: {len(events)} games; "
          "mismatch calibration by site-favorite bucket\n")

    systems = {
        "site ratings (reference)": None,
        "Elo (logistic)": EloRating(),
        "Glicko-2 (logistic)": Glicko2Rating(),
        "TrueSkill (probit)": TrueSkillRating(),
        "Thurstone gaussian": ThurstoneRating(),
        "Thurstone off-day mix": ThurstoneRating(base_kernel=offday_kernel()),
    }
    # per system, per bucket: [sum predicted p_fav, sum outcome, n_decisive, sum score_with_draws, n_all]
    acc = {nm: [[0.0, 0.0, 0, 0.0, 0] for _ in BUCKETS] for nm in systems}

    for idx, ev in enumerate(events):
        for sy in systems.values():
            if sy is not None:
                sy.elapse(ev.dt)
        if idx >= n_warm and ev.market is not None:
            fav = 0 if ev.market[0] >= ev.market[1] else 1
            p_site = ev.market[fav]
            b = next((i for i, (lo, hi) in enumerate(BUCKETS) if lo <= p_site < hi), None)
            if b is not None:
                draw = ev.ranks[0] == ev.ranks[1]
                fav_won = (not draw) and ev.ranks[fav] == 1
                score = 0.5 if draw else (1.0 if fav_won else 0.0)
                for nm, sy in systems.items():
                    p = p_site if sy is None else sy.win_probabilities(ev.names)[fav]
                    row = acc[nm][b]
                    row[3] += score
                    row[4] += 1
                    if not draw:
                        row[0] += p
                        row[1] += 1.0 if fav_won else 0.0
                        row[2] += 1
        for sy in systems.values():
            if sy is not None:
                sy.observe(ev.names, ev.ranks, dt=0.0)

    print(f"{'bucket':>12s} {'n':>7s} {'empirical':>10s}", end="")
    for nm in systems:
        print(f"{nm[:14]:>16s}", end="")
    print()
    for i, (lo, hi) in enumerate(BUCKETS):
        n = acc["Elo (logistic)"][i][2]
        emp = acc["Elo (logistic)"][i][1] / max(n, 1)
        print(f"{f'[{lo:.2f},{hi:.2f})':>12s} {n:7d} {emp:10.4f}", end="")
        for nm in systems:
            row = acc[nm][i]
            pred = row[0] / max(row[2], 1)
            print(f"{pred:16.4f}", end="")
        print()
    print("\ncells = mean predicted P(site favorite wins), decisive games;")
    print("compare each column to 'empirical'. Family bias shows in the top buckets.")


if __name__ == "__main__":
    main()
