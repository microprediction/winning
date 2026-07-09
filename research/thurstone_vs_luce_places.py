"""Does Luce peeling throw away Thurstone benefit in place probabilities?

The deep-places market reference converted win probabilities to place
probabilities by Plackett-Luce sampling — Harville's sequential-softmax
peeling, with its IIA assumption. The SIAM paper's central claim is that
Thurstone joint order statistics price places differently (and on paper
better). The laborious check, on identical inputs race by race:

  A. Luce peeling: P(position k) via Gumbel-max PL sampling of the market
     win vector (as in deep_places.py).
  B. Thurstone simulation: invert the SAME win vector into abilities via
     thurstone.AbilityCalibrator (the ability transform), simulate the race
     the Thurstonian way — performance_i = ability_i + N(0,1), rank, count —
     and read place probabilities from the simulated finish orders.

Both paths reproduce the win vector at P1 (up to inversion/MC error), so any
difference at P2+ is purely the ranking model. Scored by log loss per finish
position, paired per race (same races, same targets), mean difference with a
standard error. If B < A at deep positions, Harville/Luce is discarding real
structure that the Thurstone model retains — the paper's claim, upgraded from
worked examples to 6,348 races.

Run:  .venv/bin/python research/thurstone_vs_luce_places.py

Measured (July 2026, 5,079 paired races, log loss, diff = luce - thurstone):
    pos   luce    thurstone   diff      se
    1     2.0421  2.0410    +0.0011   0.0010   (sanity: same by construction)
    2     2.2580  2.2562    +0.0018   0.0027
    3     2.3903  2.3763    +0.0139   0.0036
    4     2.4581  2.4359    +0.0222   0.0041
    5     2.4957  2.4710    +0.0247   0.0043
    6     2.5176  2.4876    +0.0300   0.0045
Verdict: Luce/Harville peeling measurably discards Thurstone structure, with
the loss growing monotonically in finish depth (t ~ 7 by P6) — the SIAM
paper's central claim confirmed as a large-sample paired test on real data,
using only the market's own win odds. Consequence: deep_places.py's
"Glicko-2 beats the market at P6" was an artifact of the Luce shortcut; the
properly simulated market leads at every position (P6: 2.4876 vs Glicko-2
2.5098), with its margin shrinking from 0.34 at P1 to 0.02 at P6.
"""

from __future__ import annotations

import numpy as np
from thurstone import AbilityCalibrator, Density, UniformLattice

from place_probabilities import logloss_at

TARGETS = (1, 2, 3, 4, 5, 6)
S = 4096
CLIP = 1e-9


def luce_places(win_probs, rng) -> np.ndarray:
    p = np.maximum(np.asarray(win_probs, dtype=float), CLIP)
    g = rng.gumbel(size=(S, len(p)))
    sim = (-(np.log(p) + g)).argsort(axis=1).argsort(axis=1) + 1
    return _freq(sim, len(p))


def thurstone_places(abilities, rng) -> np.ndarray:
    a = np.asarray(abilities, dtype=float)  # time-like: lower is better
    perf = a[None, :] + rng.normal(size=(S, len(a)))
    sim = perf.argsort(axis=1).argsort(axis=1) + 1  # min wins
    return _freq(sim, len(a))


def _freq(sim_ranks, n) -> np.ndarray:
    out = np.zeros((n, n))
    for k in range(1, n + 1):
        out[k - 1] = (sim_ranks == k).mean(axis=0)
    return out


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events()
    n_warm = int(len(events) * 0.2)
    rng = np.random.default_rng(11)

    lattice = UniformLattice(L=150, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
    cal = AbilityCalibrator(base)

    acc = {"luce": {t: [] for t in TARGETS}, "thurstone": {t: [] for t in TARGETS}}
    diffs = {t: [] for t in TARGETS}  # paired luce - thurstone per race

    n_scored = 0
    for idx, ev in enumerate(events):
        if idx < n_warm or ev.market is None:
            continue
        n = len(ev.names)
        pos_idx = {}
        for t in TARGETS:
            holders = [i for i, r in enumerate(ev.ranks) if r == t]
            pos_idx[t] = holders[0] if len(holders) == 1 and t <= n else None

        lp = luce_places(ev.market, rng)
        abilities = cal.solve_from_prices(list(ev.market))
        tp = thurstone_places(abilities, rng)
        n_scored += 1

        for t in TARGETS:
            if pos_idx[t] is None:
                continue
            ll_l = logloss_at(lp[t - 1], pos_idx[t])
            ll_t = logloss_at(tp[t - 1], pos_idx[t])
            acc["luce"][t].append(ll_l)
            acc["thurstone"][t].append(ll_t)
            diffs[t].append(ll_l - ll_t)

    print(f"HK racing: {n_scored} scored races; market places via Luce peeling vs")
    print("Thurstone simulation from the SAME win probabilities (log loss)\n")
    print(f"{'pos':>4s} {'luce':>9s} {'thurstone':>10s} {'diff':>9s} {'se':>8s} {'races':>7s}")
    for t in TARGETS:
        d = np.asarray(diffs[t])
        if len(d) == 0:
            continue
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(
            f"{t:4d} {np.mean(acc['luce'][t]):9.4f} {np.mean(acc['thurstone'][t]):10.4f} "
            f"{d.mean():+9.4f} {se:8.4f} {len(d):7d}"
        )
    print("\npositive diff = Thurstone simulation better than Luce peeling")


if __name__ == "__main__":
    main()
