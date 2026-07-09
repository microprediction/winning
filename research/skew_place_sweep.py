"""Skew sweep judged on PLACE probabilities (win/2nd/3rd), HK racing.

The win-metric skew sweep (lab_variants.py) was null — but win probability is
location-dominated. P(2nd) and P(3rd) depend on distributional shape, which
is where a skewed performance density should register if the true one is
asymmetric. Same protocol as place_probabilities.py, ThurstoneRating variants
only (baseline numbers for the other systems live in that script's results).

Run:  .venv/bin/python research/skew_place_sweep.py

Measured (July 2026, log loss win / 2nd / 3rd):
    gaussian (a=0)    2.3926 / 2.4525 / 2.4949
    skew a=-1         2.3983 / 2.4669 / 2.5173
    skew a=1          2.4032 / 2.4799 / 2.5352
    skew a=2          2.4347 / 2.5315 / 2.5855
Verdict: decisively negative — monotonically worse in |a| in every column,
including the shape-sensitive place metrics. HK performances are symmetric as
far as ranks can tell. Contrast with the scale MIXTURE, which does win the
3rd-place column (place_probabilities.py): the useful shape information here
is symmetric tail-fattening from variance mixing, not asymmetry.
"""

from __future__ import annotations

import numpy as np

from lab_variants import skew_kernel
from place_probabilities import logloss_at, rank_probabilities
from winning import ThurstoneRating


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    n_warm = int(len(events) * 0.2)
    print(f"HK racing: {len(events)} races; skew sweep on win/2nd/3rd log loss\n")

    systems = {
        "gaussian (a=0)": ThurstoneRating(),
        "skew a=-1": ThurstoneRating(base_kernel=skew_kernel(-1.0)),
        "skew a=1": ThurstoneRating(base_kernel=skew_kernel(1.0)),
        "skew a=2": ThurstoneRating(base_kernel=skew_kernel(2.0)),
    }
    acc = {nm: {"win": [], "second": [], "third": []} for nm in systems}

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
                rp = rank_probabilities(system, ev.names)
                for target, key in ((2, "second"), (3, "third")):
                    if n >= 3 and pos_idx[target] is not None:
                        acc[label][key].append(logloss_at(rp[target - 1], pos_idx[target]))
        for system in systems.values():
            system.observe(ev.names, ev.ranks, dt=0.0)

    print(f"{'variant':20s} {'win':>8s} {'2nd':>8s} {'3rd':>8s}")
    for label, d in acc.items():
        print(
            f"{label:20s} {np.mean(d['win']):8.4f} {np.mean(d['second']):8.4f} "
            f"{np.mean(d['third']):8.4f}"
        )


if __name__ == "__main__":
    main()
