"""HK racing: the full finish-order comparison, positions 1 through 6.

Extends place_probabilities.py deeper into the field. Each system's rank
probabilities come from its own predictive performance family (MC, 512
samples); the market reference generalizes Harville to any depth by sampling
Plackett-Luce finish orders via the Gumbel-max trick (argsort of
Gumbel-perturbed log win-odds), 4096 orders per race. Log loss per position,
scored only where the position exists and is dead-heat-free; 'mean' averages
positions 1-6 per system over its scored races.

Run:  .venv/bin/python research/deep_places.py

Measured (July 2026, log loss by position; P1 is MC-sampled so reads ~0.03
above the exact win columns of other tables — within-column comparisons only):
    Glicko-2          2.3821 2.3974 2.4579 2.4783 2.5058 2.5098 | mean 2.4552
    Thurstone 1-D     2.4213 2.4265 2.4712 2.4873 2.5119 2.5125 | mean 2.4718
    scale-learning    2.4263 2.4135 2.4682 2.4905 2.5116 2.5208 | mean 2.4718
    TrueSkill         2.5075 2.4782 2.4936 2.5036 2.5234 2.5250 | mean 2.5052
    Elo               2.4828 2.4849 2.5004 2.5040 2.5202 2.5273 | mean 2.5033
    OpenSkill PL      3.1960 2.7560 2.6635 2.5589 2.5944 2.5806 | mean 2.7249
    Market (PL)       2.0416 2.2589 2.3896 2.4575 2.4968 2.5172 | mean 2.3603
Findings: (1) the market's edge decays monotonically with finish depth —
but the "crosses zero at P5/P6" reading of THIS table is an ARTIFACT of the
PL/Luce place construction: thurstone_vs_luce_places.py shows that Thurstone
simulation from the same win odds prices deep places much better, and the
properly simulated market leads at every position (margin 0.34 at P1
shrinking to 0.02 at P6).
(2) Replication note: the earlier S=256 finding that scale-learning beat
Glicko-2 at 3rd place did NOT survive S=512 — Glicko-2 leads every column;
the robust residue is scale-learning's P2-P3 edge over its own fixed-scale
version, with means tied. (3) OpenSkill PL's belief-implied P1 (3.20) vs its
native formula (2.47) again shows its predictor flattens bad beliefs.
"""

from __future__ import annotations

import numpy as np
from place_probabilities import logloss_at, rank_probabilities
from scale_learning import ScaleLearningThurstoneRating

from winning import EloRating, Glicko2Rating, ThurstoneRating
from winning.shims import OpenSkillRating, TrueSkillRating

TARGETS = (1, 2, 3, 4, 5, 6)
MC_MARKET = 4096
CLIP = 1e-9


def market_rank_probabilities(win_probs, rng) -> np.ndarray:
    """P(position k) matrix under Plackett-Luce with the market win vector."""
    p = np.maximum(np.asarray(win_probs, dtype=float), CLIP)
    logp = np.log(p)
    g = rng.gumbel(size=(MC_MARKET, len(p)))
    sim_ranks = (-(logp + g)).argsort(axis=1).argsort(axis=1) + 1
    n = len(p)
    out = np.zeros((n, n))
    for k in range(1, n + 1):
        out[k - 1] = (sim_ranks == k).mean(axis=0)
    return out


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    n_warm = int(len(events) * 0.2)
    rng = np.random.default_rng(7)
    print(f"HK racing: {len(events)} races; log loss by finish position 1-6\n")

    systems = {
        "Thurstone lattice (1-D)": ThurstoneRating(),
        "Thurstone scale-learning": ScaleLearningThurstoneRating(),
        "Glicko-2": Glicko2Rating(),
        "TrueSkill": TrueSkillRating(),
        "OpenSkill PlackettLuce": OpenSkillRating("PlackettLuce"),
        "Elo (multi-entrant)": EloRating(),
    }
    labels = list(systems) + ["Market (PL-sampled places)"]
    acc = {nm: {t: [] for t in TARGETS} for nm in labels}

    for idx, ev in enumerate(events):
        for system in systems.values():
            system.elapse(ev.dt)
        if idx >= n_warm:
            n = len(ev.names)
            pos_idx = {}
            for t in TARGETS:
                holders = [i for i, r in enumerate(ev.ranks) if r == t]
                pos_idx[t] = holders[0] if len(holders) == 1 and t <= n else None
            for label, system in systems.items():
                rp = rank_probabilities(system, ev.names, size=512)
                for t in TARGETS:
                    if pos_idx[t] is not None:
                        acc[label][t].append(logloss_at(rp[t - 1], pos_idx[t]))
            mp = market_rank_probabilities(ev.market, rng)
            for t in TARGETS:
                if pos_idx[t] is not None:
                    acc["Market (PL-sampled places)"][t].append(
                        logloss_at(mp[t - 1], pos_idx[t])
                    )
        for system in systems.values():
            system.observe(ev.names, ev.ranks, dt=0.0)

    header = f"{'system':32s}" + "".join(f"{('P' + str(t)):>9s}" for t in TARGETS) + f"{'mean':>9s}"
    print(header)
    for label in labels:
        cells = []
        means = []
        for t in TARGETS:
            v = acc[label][t]
            cells.append(f"{np.mean(v):9.4f}" if v else "        -")
            if v:
                means.append(np.mean(v))
        print(f"{label:32s}" + "".join(cells) + f"{np.mean(means):9.4f}")


if __name__ == "__main__":
    main()
