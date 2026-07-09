"""Era-adaptive disaster slab for F1: fit p_dnf prequentially.

DNF rates by decade run 42-59% (1950s-1990s) then collapse to 13-18%
(2010s-2020s) — a fixed 25% slab is wrong for every era. This variant
estimates the disaster mass from a trailing window of observed DNF
frequency (past races only, no leakage) and rebuilds the base kernel every
REBUILD races. Requires knowing which entrants DNF'd: the loader marks them
as the tied-last group, so within observe we treat the tied-last group in
fields where it exists as the retirement pool (an approximation: genuinely
tied finishers are rare in F1 classification).

Compared prequentially: gaussian, fixed slab p=0.25, era-adaptive slab.

Run:  .venv/bin/python research/f1_era_slab.py
"""

from __future__ import annotations

from collections import deque

import numpy as np

from winning import ThurstoneRating
from winning.benchmarks.forward_chain import evaluate
from f1_dirac_disaster import dirac_disaster_kernel


class EraSlabThurstoneRating(ThurstoneRating):
    """Disaster mass tracked from a trailing window of observed DNF shares."""

    def __init__(self, window: int = 60, rebuild: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._dnf_shares = deque(maxlen=window)
        self._since_rebuild = 0
        self._rebuild_every = rebuild
        self.current_p = 0.25

    def observe(self, names, ranks, dt=1.0):
        n = len(ranks)
        worst = max(ranks)
        tied_last = [r for r in ranks if r == worst]
        share = len(tied_last) / n if len(tied_last) >= 2 else 0.0
        self._dnf_shares.append(share)
        self._since_rebuild += 1
        if self._since_rebuild >= self._rebuild_every and len(self._dnf_shares) >= 10:
            p = float(np.clip(np.mean(self._dnf_shares), 0.03, 0.6))
            self.current_p = p
            k = dirac_disaster_kernel(core_sd=1.0, p_disaster=p)
            self._base_kernel = k / k.sum()
            self._since_rebuild = 0
        super().observe(names, ranks, dt=dt)


def main():
    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    print(f"F1: {len(events)} grands prix\n")
    for label, system in [
        ("gaussian", ThurstoneRating()),
        ("fixed slab p=0.25", ThurstoneRating(base_kernel=dirac_disaster_kernel(core_sd=1.0, p_disaster=0.25))),
        ("era-adaptive slab", EraSlabThurstoneRating()),
    ]:
        s = evaluate(system, events, rank_pit=True).summary()
        extra = f"  (final p={system.current_p:.2f})" if hasattr(system, "current_p") else ""
        print(f"{label:22s} log_loss={s['log_loss']:.4f} acc={s['accuracy']:.4f} "
              f"tau={s['kendall_tau']:.4f} ece={s['ece']:.4f} pit_ks={s['rank_pit_ks']:.4f}{extra}")


if __name__ == "__main__":
    main()
