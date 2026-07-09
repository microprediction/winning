"""Paired per-position statistics for the win+place pipeline comparison,
rescoring from the saved fitted parameters (no refitting).

Measured (July 2026, ~4,800 paired races; positive = second pipeline better):
  pos   luce -> loc (se)      loc -> loc+scale (se)
  1     +0.0005 (0.0010)      -0.0014 (0.0005)   <- small real P1 tax, t~-2.8
  2     +0.0030 (0.0027)      +0.0028 (0.0019)
  3     +0.0163 (0.0037)      +0.0072 (0.0025)   <- certified, t~2.9
  4     +0.0205 (0.0041)      +0.0064 (0.0030)   <- t~2.1
  5     +0.0246 (0.0044)      +0.0039 (0.0034)
  6     +0.0284 (0.0046)      +0.0020 (0.0036)
Thurstone-over-Harville is overwhelming at depth (t to 6.2, replicating the
independent study almost digit-for-digit); the place-funded scale adds a
certified gain exactly at the place boundary (P3-P4) and nothing beyond its
jurisdiction — information lands where the price that funds it speaks.
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
from thurstone import AbilityCalibrator, Density, UniformLattice

from place_probabilities import logloss_at
from win_place_calibration import CLIP, rank_matrix

TARGETS = (1, 2, 3, 4, 5, 6)


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    d = np.load(os.path.expanduser("~/.cache/winning/implied_params.npz"), allow_pickle=True)
    by_race = defaultdict(dict)
    for r, h, lo, sc in zip(d["race"], d["horse"], d["loc"], d["scale"]):
        by_race[int(r)][str(h)] = (float(lo), float(sc))

    events = [e for e in hkracing_events() if e.context and "place_market" in e.context]
    n_warm = int(len(events) * 0.2)
    rng = np.random.default_rng(31)
    cal = AbilityCalibrator(
        Density.skew_normal(UniformLattice(L=150, unit=0.1), loc=0.0, scale=1.0, a=0.0)
    )

    diffs_ls_vs_l = {t: [] for t in TARGETS}   # loc_scale vs loc
    diffs_l_vs_luce = {t: [] for t in TARGETS}  # loc vs luce
    for idx, ev in enumerate(events):
        if idx < n_warm or idx not in by_race:
            continue
        n = len(ev.names)
        win = np.maximum(np.asarray(ev.market, float), CLIP); win /= win.sum()
        pos_idx = {}
        for t in TARGETS:
            holders = [i for i, r in enumerate(ev.ranks) if r == t]
            pos_idx[t] = holders[0] if len(holders) == 1 and t <= n else None

        g = rng.gumbel(size=(4096, n))
        sim = (-(np.log(win) + g)).argsort(axis=1).argsort(axis=1) + 1
        lp = np.zeros((n, n))
        for k in range(1, n + 1):
            lp[k - 1] = (sim == k).mean(axis=0)
        locs0 = np.asarray(cal.solve_from_prices(list(win)), float)
        Z = rng.standard_normal((4096, n))  # SAME draws for both Thurstone rows
        def rm(lo, sc):
            perf = lo[None, :] + sc[None, :] * Z
            s = perf.argsort(axis=1).argsort(axis=1) + 1
            out = np.zeros((n, n))
            for k in range(1, n + 1):
                out[k - 1] = (s == k).mean(axis=0)
            return out
        tp0 = rm(locs0, np.ones(n))
        fitted = by_race[idx]
        lo1 = np.array([fitted[nm][0] for nm in ev.names])
        sc1 = np.array([fitted[nm][1] for nm in ev.names])
        tp1 = rm(lo1, sc1)

        for t in TARGETS:
            if pos_idx[t] is None:
                continue
            l_luce = logloss_at(lp[t - 1], pos_idx[t])
            l_loc = logloss_at(tp0[t - 1], pos_idx[t])
            l_ls = logloss_at(tp1[t - 1], pos_idx[t])
            diffs_l_vs_luce[t].append(l_luce - l_loc)
            diffs_ls_vs_l[t].append(l_loc - l_ls)

    print("paired per-race log-loss differences (positive = second pipeline better)\n")
    print(f"{'pos':>4s} {'luce - loc':>12s} {'se':>7s} {'loc - loc_scale':>17s} {'se':>7s} {'races':>7s}")
    for t in TARGETS:
        a = np.asarray(diffs_l_vs_luce[t]); b = np.asarray(diffs_ls_vs_l[t])
        print(f"{t:4d} {a.mean():+12.4f} {a.std(ddof=1)/np.sqrt(len(a)):7.4f} "
              f"{b.mean():+17.4f} {b.std(ddof=1)/np.sqrt(len(b)):7.4f} {len(b):7d}")


if __name__ == "__main__":
    main()
