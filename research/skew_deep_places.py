"""The missing cells: skew kernels scored at finish positions 1 through 6.

The earlier skew sweep (skew_place_sweep.py) stopped at 3rd place and read
monotonically negative; the deep-places table (deep_places.py) went to 6th
but only for Gaussian-based systems. Since winner metrics and shape metrics
have already disagreed about kernels on F1 (Dirac study), skew deserves its
deep-position cells before the book closes on it.

Run:  .venv/bin/python research/skew_deep_places.py

Measured (July 2026, log loss by position, HK):
    gaussian   2.4071 2.4160 2.4725 2.4866 2.5085 2.5205 | mean 2.4685
    skew a=-1  2.4167 2.4139 2.4782 2.4867 2.5097 2.5167 | mean 2.4703
    skew a=1   2.4273 2.4198 2.4880 2.4876 2.5136 2.5189 | mean 2.4758
    skew a=2   2.5047 2.4486 2.4928 2.4996 2.5293 2.5217 | mean 2.4995
Verdict: no depth rescue. Gaussian wins P1-P5 and the mean; the P6 flicker
(both mild skews ~0.003 ahead) is within noise and shows no dose-response
(a=-1 and a=+1 both "help", a=2 does not) — the signature of nothing. Skew
is decisively null on HK at every finish depth, now tested rather than
extrapolated.
"""

from __future__ import annotations

import numpy as np

from place_probabilities import logloss_at, rank_probabilities
from winning import ThurstoneRating, kernels

TARGETS = (1, 2, 3, 4, 5, 6)


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    n_warm = int(len(events) * 0.2)
    print(f"HK racing: {len(events)} races; skew kernels at positions 1-6\n")

    systems = {
        "gaussian (a=0)": ThurstoneRating(),
        "skew a=-1": ThurstoneRating(base_kernel=kernels.skew_kernel(a=-1.0)),
        "skew a=1": ThurstoneRating(base_kernel=kernels.skew_kernel(a=1.0)),
        "skew a=2": ThurstoneRating(base_kernel=kernels.skew_kernel(a=2.0)),
    }
    acc = {nm: {t: [] for t in TARGETS} for nm in systems}

    for idx, ev in enumerate(events):
        for sy in systems.values():
            sy.elapse(ev.dt)
        if idx >= n_warm:
            n = len(ev.names)
            pos_idx = {}
            for t in TARGETS:
                holders = [i for i, r in enumerate(ev.ranks) if r == t]
                pos_idx[t] = holders[0] if len(holders) == 1 and t <= n else None
            for nm, sy in systems.items():
                rp = rank_probabilities(sy, ev.names, size=512)
                for t in TARGETS:
                    if pos_idx[t] is not None:
                        acc[nm][t].append(logloss_at(rp[t - 1], pos_idx[t]))
        for sy in systems.values():
            sy.observe(ev.names, ev.ranks, dt=0.0)

    print(f"{'variant':16s}" + "".join(f"{('P' + str(t)):>9s}" for t in TARGETS) + f"{'mean':>9s}")
    for nm in systems:
        cells = [np.mean(acc[nm][t]) for t in TARGETS]
        print(f"{nm:16s}" + "".join(f"{c:9.4f}" for c in cells) + f"{np.mean(cells):9.4f}")


if __name__ == "__main__":
    main()
