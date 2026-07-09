"""Degrees-of-freedom sweep: student-t performance noise for ThurstoneRating.

The chess mismatch study said tail FAMILY didn't fix rating-gap
overconfidence (temperature did), but F1's DNF mixture said tail MASS
matters where catastrophes are real. The t-family's nu is the single dial
between those regimes: nu=2 very heavy, nu -> inf Gaussian. Swept on the
three most informative datasets: F1 (tails demonstrably real), HK lab
(tv to market truth), synthetic (Gaussian world = misspecification cost).

Run:  .venv/bin/python research/t_sweep.py
"""

from __future__ import annotations

from winning import ThurstoneRating, kernels
from winning.benchmarks.forward_chain import evaluate


def sweep(events, label, extra_metric):
    print(label)
    configs = [("gaussian", None)] + [(f"t nu={nu}", nu) for nu in (2, 3, 5, 10)]
    for name, nu in configs:
        bk = None if nu is None else kernels.student_t_kernel(nu=nu)
        system = ThurstoneRating(base_kernel=bk)
        s = evaluate(system, events, rank_pit=extra_metric == "pit").summary()
        extra = (
            f" tv={s['tv_vs_oracle']:.4f}" if extra_metric == "tv"
            else f" pit_ks={s['rank_pit_ks']:.4f}" if extra_metric == "pit"
            else ""
        )
        print(f"  {name:12s} log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}{extra}")
    print()


def main():
    from winning.benchmarks.events import synthetic_world
    from winning.benchmarks.f1 import f1_events
    from winning.benchmarks.kaggle_datasets import hkracing_events

    sweep(f1_events(), "F1 (tails real: DNFs)", "pit")
    sweep(hkracing_events(oracle_temperature=1.05), "HK racing lab", "tv")
    sweep(
        synthetic_world(num_contestants=200, num_events=3000, seed=17),
        "synthetic Gaussian world (misspecification cost)",
        "tv",
    )


if __name__ == "__main__":
    main()
