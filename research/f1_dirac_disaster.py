"""F1 kernel: Dirac married to a disaster tail.

Physical premise (Peter's): among cars that finish, order is nearly
deterministic — pace deltas are consistent — so performance noise should be
a near-Dirac spike; rank randomness in F1 is almost all catastrophic
(DNF/mechanical/crash), i.e. a lump of mass far out on the slow side. Kernel:
(1-p) * N(0, core_sd) + p * uniform slab on [3, 12] time units. Swept over
core_sd x p_disaster on the F1 dataset against the Gaussian baseline
(2.2247) and the earlier N(0,1)+slab mixture (2.2090).

Run:  .venv/bin/python research/f1_dirac_disaster.py
"""

from __future__ import annotations

import numpy as np

from winning import ThurstoneRating, kernels
from winning.benchmarks.forward_chain import evaluate


def dirac_disaster_kernel(unit=0.1, core_sd=0.3, p_disaster=0.2, slab=(3.0, 12.0)):
    half = int(slab[1] / unit) + 10
    n = 2 * half + 1
    x = np.arange(-half, half + 1) * unit
    core = np.zeros(n)
    k = kernels.gaussian_kernel(unit, core_sd)
    s = half - (len(k) - 1) // 2
    core[s : s + len(k)] = k
    slab_k = ((x >= slab[0]) & (x <= slab[1])).astype(float)
    slab_k /= slab_k.sum()
    out = (1 - p_disaster) * core + p_disaster * slab_k
    return out / out.sum()


def main():
    from winning.benchmarks.f1 import f1_events

    events = f1_events()
    print(f"F1: {len(events)} grands prix; Dirac-plus-disaster sweep\n")
    print(f"{'kernel':34s} {'log_loss':>9s} {'acc':>7s} {'tau':>7s} {'ece':>7s} {'pit_ks':>8s}")

    def row(label, system):
        s = evaluate(system, events, rank_pit=True).summary()
        print(
            f"{label:34s} {s['log_loss']:9.4f} {s['accuracy']:7.4f} "
            f"{s['kendall_tau']:7.4f} {s['ece']:7.4f} {s['rank_pit_ks']:8.4f}"
        )

    row("gaussian baseline", ThurstoneRating())
    row("N(0,1) + slab p=.25 (earlier)", ThurstoneRating(base_kernel=dirac_disaster_kernel(core_sd=1.0, p_disaster=0.25)))
    for core_sd in (0.5, 0.3, 0.2):
        for p in (0.10, 0.18, 0.25):
            row(
                f"dirac core={core_sd} disaster p={p}",
                ThurstoneRating(base_kernel=dirac_disaster_kernel(core_sd=core_sd, p_disaster=p)),
            )


if __name__ == "__main__":
    main()
