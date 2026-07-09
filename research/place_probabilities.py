"""HK racing: predict better, judged on win AND place probabilities.

The generative model is unknown, so the task is prediction alone. Each
system's predictive performance distributions (performance_samples) simulate
the field; counting simulated finish positions gives P(finish exactly 2nd)
and P(finish exactly 3rd) per horse, scored by log loss against the realized
runner-up and third — the same prequential protocol as the win metrics.
Includes the per-horse scale-learning rater (whose gain the common-mixture
ablation showed is genuinely idiosyncratic) and a market reference for place
probabilities via Harville's formula on the win odds (uses race t's own odds,
so it is a ceiling-style reference, not a competitor).

Run:  .venv/bin/python research/place_probabilities.py

Measured (July 2026, log loss win / 2nd / 3rd, tv on win):
    Market (+Harville)         2.0409 / 2.2572 / 2.3896
    Glicko-2                   2.3575 / 2.4266 / 2.4818
    Thurstone lattice (1-D)    2.3926 / 2.4525 / 2.4949
    Thurstone scale-learning   2.4003 / 2.4373 / 2.4777   <- best fundamental at 3rd
    TrueSkill                  2.4661 / 2.5216 / 2.5267
    OpenSkill PlackettLuce     2.4701 / 2.9185 / 2.7557   <- place probs collapse
    Elo                        2.4719 / 2.5009 / 2.5075
Findings: (1) the scale mixture trades a hair of win accuracy for the best
3rd-place probabilities of any fundamental system — heavier tails price deep
finish positions better, the distributional-shape argument made empirical;
(2) the fundamental-to-market gap shrinks with depth (0.35 win, 0.29 2nd,
0.09 3rd): the market's private edge is mostly about who WINS; (3) OpenSkill
PL's rank distributions collapse at 2nd/3rd despite tolerable win probs.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from scale_learning import ScaleLearningThurstoneRating
from winning import EloRating, Glicko2Rating, ThurstoneRating
from winning.shims import OpenSkillRating, TrueSkillRating

SAMPLES = 256
CLIP = 1e-9


def rank_probabilities(system, names, size=SAMPLES) -> np.ndarray:
    """(n_ranks x n_names) matrix of P(name finishes at position k) via MC."""
    samples = np.asarray(system.performance_samples(names, size=size), dtype=float)
    sim_ranks = (-samples).argsort(axis=1).argsort(axis=1) + 1
    n = len(names)
    out = np.zeros((n, n))
    for k in range(1, n + 1):
        out[k - 1] = (sim_ranks == k).mean(axis=0)
    return out


def harville_place(probs: List[float], position: int) -> np.ndarray:
    """P(finish at `position`) from win probabilities via Harville's formula."""
    p = np.asarray(probs, dtype=float)
    n = len(p)
    if position == 1:
        return p
    out = np.zeros(n)
    idx = list(range(n))
    if position == 2:
        for j in idx:
            for i in idx:
                if i == j:
                    continue
                rem = 1.0 - p[i]
                if rem > 0:
                    out[j] += p[i] * p[j] / rem
        return out
    if position == 3:
        for j in idx:
            for i in idx:
                if i == j:
                    continue
                for k in idx:
                    if k in (i, j):
                        continue
                    rem1 = 1.0 - p[i]
                    rem2 = 1.0 - p[i] - p[k]
                    if rem1 > 0 and rem2 > 0:
                        out[j] += p[i] * (p[k] / rem1) * (p[j] / rem2)
        return out
    raise ValueError("position must be 1..3")


def logloss_at(prob_row: np.ndarray, actual_idx: int) -> float:
    q = np.maximum(prob_row, CLIP)
    q = q / q.sum()
    return -math.log(q[actual_idx])


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    n_warm = int(len(events) * 0.2)
    print(f"HK racing: {len(events)} races; win/2nd/3rd log loss, prequential\n")

    systems = {
        "Thurstone lattice (1-D)": ThurstoneRating(),
        "Thurstone scale-learning": ScaleLearningThurstoneRating(),
        "Glicko-2": Glicko2Rating(),
        "TrueSkill": TrueSkillRating(),
        "OpenSkill PlackettLuce": OpenSkillRating("PlackettLuce"),
        "Elo (multi-entrant)": EloRating(),
    }
    acc = {nm: {"win": [], "second": [], "third": [], "tv": []} for nm in systems}
    acc["Market (+Harville for places)"] = {"win": [], "second": [], "third": [], "tv": []}

    for idx, ev in enumerate(events):
        for system in systems.values():
            system.elapse(ev.dt)
        if idx >= n_warm:
            n = len(ev.names)
            pos_idx = {}
            for target in (1, 2, 3):
                holders = [i for i, r in enumerate(ev.ranks) if r == target]
                pos_idx[target] = holders[0] if len(holders) == 1 else None
            for label, system in systems.items():
                probs = np.asarray(system.win_probabilities(ev.names), dtype=float)
                if pos_idx[1] is not None:
                    acc[label]["win"].append(logloss_at(probs, pos_idx[1]))
                    if ev.truth is not None:
                        t = np.asarray(ev.truth)
                        acc[label]["tv"].append(0.5 * np.abs(probs / probs.sum() - t).sum())
                rp = rank_probabilities(system, ev.names)
                for target, key in ((2, "second"), (3, "third")):
                    if n >= 3 and pos_idx[target] is not None:
                        acc[label][key].append(logloss_at(rp[target - 1], pos_idx[target]))
            market = np.asarray(ev.market, dtype=float)
            if pos_idx[1] is not None:
                acc["Market (+Harville for places)"]["win"].append(
                    logloss_at(market, pos_idx[1])
                )
            for target, key in ((2, "second"), (3, "third")):
                if n >= 3 and pos_idx[target] is not None:
                    acc["Market (+Harville for places)"][key].append(
                        logloss_at(harville_place(list(market), target), pos_idx[target])
                    )
        for system in systems.values():
            system.observe(ev.names, ev.ranks, dt=0.0)

    print(f"{'system':32s} {'win':>8s} {'2nd':>8s} {'3rd':>8s} {'tv':>8s}")
    for label, d in sorted(acc.items(), key=lambda kv: np.mean(kv[1]["win"])):
        tv = f"{np.mean(d['tv']):8.4f}" if d["tv"] else "       -"
        print(
            f"{label:32s} {np.mean(d['win']):8.4f} {np.mean(d['second']):8.4f} "
            f"{np.mean(d['third']):8.4f} {tv}"
        )


if __name__ == "__main__":
    main()
