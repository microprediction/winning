"""Follow-up to run_low_dimension_barrage.py for the independent and
factor classes, whose shipped iteration takes a full step (damping 1)
for n > 2.

For every field and target of the barrage (same seeds) it runs
  shipped, cap 2000   abilities_from_race(..., n_iter=2000): separates
                      slow convergence from a stall or cycle
  damped 0.7          a copy of the shipped loop with the step
                      multiplied by 0.7 at every n (the shipped code
                      does this only at n = 2), caps 60 and 2000
and prints, per class and n, how many of the runs fail under each.

    python run_low_dimension_damped.py                # 20 seeds
    python run_low_dimension_damped.py 20 "factor rank 1"
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "run_low_dimension_barrage.py")).read()
_ns = {}
_argv, sys.argv = sys.argv, sys.argv[:2]
exec(_src[:_src.index('print(f"{SEEDS} seeds')], _ns)   # field(), constants
sys.argv = _argv
field, SIZES, SCALES, SPREADS, CLASSES, SEEDS = (
    _ns[k] for k in ("field", "SIZES", "SCALES", "SPREADS", "CLASSES", "SEEDS"))
race_probabilities = _ns["race_probabilities"]
abilities_from_race = _ns["abilities_from_race"]
WHICH = sys.argv[2:] or ("independent", "factor rank 1", "factor rank 2")
TOL = 1e-8


def damped(target, kw, alpha, cap):
    """The loop of abilities_from_race with a fixed step multiplier."""
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    for it in range(cap):
        phat, sl = race_probabilities(mu, return_slopes=True, **kw)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        if np.abs(resid).max() < TOL:
            return it + 1
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return None


warnings.simplefilter("ignore")
print(f"{SEEDS} seeds per (class, n, scale, spread); round-trip and free targets")
print(f"{'class':14s} {'n':>3s}  runs  shipped: fail@60  fail@2000 | "
      f"damped 0.7: fail@60  fail@2000  sweeps med/max")
for name in WHICH:
    g = CLASSES.index(name)
    for n in SIZES:
        runs = f60 = f2000 = d60 = d2000 = 0
        its = []
        for j in range(len(SCALES)):
            if name == "independent" and j > 0:
                continue
            for k in range(len(SPREADS)):
                for r in range(SEEDS):
                    mu, free, kw = field(g, n, j, k, r)
                    p = race_probabilities(mu, **kw)
                    for target in (p, free / free.sum()):
                        runs += 1
                        _, info = abilities_from_race(target, return_info=True, **kw)
                        if not info["converged"]:
                            f60 += 1
                            _, info = abilities_from_race(
                                target, return_info=True, n_iter=2000, **kw)
                            f2000 += not info["converged"]
                        it = damped(target, kw, 0.7, 2000)
                        d2000 += it is None
                        d60 += it is None or it > 60
                        its.append(2000 if it is None else it)
        print(f"{name:14s} {n:3d}  {runs:4d}  {f60:14d}  {f2000:9d} | "
              f"{d60:17d}  {d2000:9d}  {int(np.median(its)):6d} / {max(its)}",
              flush=True)
