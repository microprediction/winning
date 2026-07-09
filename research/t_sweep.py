"""Degrees-of-freedom sweep: student-t performance noise for ThurstoneRating.

The chess mismatch study said tail FAMILY didn't fix rating-gap
overconfidence (temperature did), but F1's DNF mixture said tail MASS
matters where catastrophes are real. The t-family's nu is the single dial
between those regimes: nu=2 very heavy, nu -> inf Gaussian. Swept on the
three most informative datasets: F1 (tails demonstrably real), HK lab
(tv to market truth), synthetic (Gaussian world = misspecification cost).

Run:  .venv/bin/python research/t_sweep.py

Measured (July 2026, log loss unless noted):
  F1:        gaussian 2.2247 | t2 2.3842 | t3 2.3162 | t5 2.2708 | t10 2.2444
  HK lab:    gaussian 2.3801 | t2 2.3942 | t5 2.3784 | t10 2.3776; ECE t2=0.0035
  synthetic: gaussian 1.4990 | t3 1.4981 | t5 1.4929 | t10 1.4940 (t5 best TV too)
Lessons: (1) "heavy tails" decomposes — F1 wants ONE-SIDED disaster mass
(offday/DNF mixture helped, symmetric kurtosis hurts: t puts miracle mass on
the fast side); (2) mild t (nu~5) beats Gaussian even in a purely Gaussian
world — the likelihood's tails act as robustness to the update's approximate
opponent marginals, robust-regression style; (3) nu is a sharpness-vs-
calibration dial (nu=2: best ECE ever measured on HK, 0.0035). Defaults stay
Gaussian (F1 vetoes); kernels docstring carries this guidance.
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
