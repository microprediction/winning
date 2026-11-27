"""Candidate step rules for the independent and factor inversion, run on
every field and target of run_low_dimension_barrage.py (same seeds).

The shipped loop of abilities_from_race takes a full step for n > 2 and
oscillates in small fields. Each candidate below is a copy of that loop
with a different step multiplier; nothing else changes (same start,
same preconditioner, same residual-proportional clip, same recentring).

  shipped     multiplier 1 for n > 2, 0.7 at n = 2
  fixed 0.7   multiplier 0.7 at every n
  backtrack   multiplier starts at 1 and halves (floor 0.05) whenever
              the largest residual grows; the rule of the block, tree
              and nested path, started at 1 instead of 0.7
  sign        one multiplier per contestant, starting at 1: halved
              (floor 0.05) when that contestant's residual changes
              sign, multiplied by 1.5 (cap 1) when it does not
  n_eff       multiplier 1 - 1/(2 m), m = 1 / sum(target^2) the
              effective number of contestants: 0.75 for two equal
              contestants, tending to 1 for a large even field

Prints, per class and n, the runs that fail at the 60-sweep cap and at
CAP sweeps, and the median and largest sweep count.

    python run_low_dimension_candidates.py                # 20 seeds
    python run_low_dimension_candidates.py 20 "factor rank 1"
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
WHICH = sys.argv[2:] or ("independent", "factor rank 1", "factor rank 2")
TOL, CAP = 1e-8, 500
RULES = ("shipped", "fixed 0.7", "backtrack", "sign", "n_eff")


def solve(target, kw, rule, cap=CAP):
    """The loop of abilities_from_race under one step rule. Returns the
    sweep count, or None if the residual is above TOL after cap sweeps."""
    n = len(target)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    if rule == "shipped":
        alpha = 1.0 if n > 2 else 0.7
    elif rule == "fixed 0.7":
        alpha = 0.7
    elif rule == "n_eff":
        alpha = 1.0 - 0.5 * float(np.sum(target ** 2))
    elif rule == "sign":
        alpha = np.ones(n)
    else:
        alpha = 1.0
    last, prev = np.inf, None
    for it in range(cap):
        phat, sl = race_probabilities(mu, return_slopes=True, **kw)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        err = float(np.abs(resid).max())
        if err < TOL:
            return it + 1
        if rule == "backtrack":
            if err > last:
                alpha = max(alpha * 0.5, 0.05)
            last = err
        elif rule == "sign":
            if prev is not None:
                flip = resid * prev < 0
                alpha = np.where(flip, np.maximum(alpha * 0.5, 0.05),
                                 np.minimum(alpha * 1.5, 1.0))
            prev = resid
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return None


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print(f"{SEEDS} seeds per (class, n, scale, spread); round-trip and free "
          f"targets; cells are fail@60 / fail@{CAP} / median sweeps / max sweeps")
    print(f"{'class':14s} {'n':>3s} {'runs':>5s}  "
          + "  ".join(f"{r:>18s}" for r in RULES))
    for name in WHICH:
        g = CLASSES.index(name)
        for n in SIZES:
            its = {r: [] for r in RULES}
            for j in range(len(SCALES)):
                if name == "independent" and j > 0:
                    continue
                for k in range(len(SPREADS)):
                    for r in range(SEEDS):
                        mu, free, kw = field(g, n, j, k, r)
                        p = race_probabilities(mu, **kw)
                        for target in (p, free / free.sum()):
                            for rule in RULES:
                                its[rule].append(solve(target, kw, rule))
            cells = []
            for rule in RULES:
                v = its[rule]
                done = [x for x in v if x is not None]
                f_cap = len(v) - len(done)
                f60 = f_cap + sum(x > 60 for x in done)
                w = [CAP if x is None else x for x in v]
                cells.append(f"{f60:4d}/{f_cap:3d}/{int(np.median(w)):3d}/{max(w):4d}")
            print(f"{name:14s} {n:3d} {len(its['shipped']):5d}  "
                  + "  ".join(f"{c:>18s}" for c in cells), flush=True)
