"""Convergence battery for the shipped diagonal inversion.

The paper's inversion (winning.factor.races.abilities_from_race) is a
diagonally preconditioned residual iteration with no convergence proof
and one constructed stall case. This battery characterizes it on random
fields: for each grammar and each strength of shared variation it draws
SEEDS fields, prices them with the shipped forward map, inverts the
result with the default settings (tolerance 1e-8 on the log-probability
residual, at most 60 sweeps), and records the solver's own report.

Per cell it prints the fraction converged, the median and largest sweep
count, and the largest residual reached.

Fields: n contestants, abilities N(0, 1) centred, idiosyncratic
variances uniform on [0.5, 1.5].
  independent     no shared variation (scale column unused)
  factor rank 1   loadings N(0, scale^2)
  factor rank 2   loadings N(0, scale^2) per column
  blocks          n/10 clusters of ten, scalar cluster loadings
                  uniform on [0.5, 1.5] * scale
Seed for (grammar g, scale index j, replicate r): default_rng([2026, g, j, r]).

    python run_convergence_battery.py            # n = 100, 10 seeds per cell
    python run_convergence_battery.py 50 3       # smoke run

Sharp rank-two fields escalate to 8192 Sobol nodes per pass, so those
cells take minutes per inversion; the default run is about half an hour.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import warnings

import numpy as np

from winning.factor.races import abilities_from_race, race_probabilities
from winning.factor.structures import Blocks, dispatch_probabilities

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SCALES = (0.25, 0.5, 1.0, 2.0)
GRAMMARS = ("independent", "factor rank 1", "factor rank 2", "blocks")


def field(g, j, r):
    rng = np.random.default_rng([2026, g, j, r])
    mu = rng.normal(size=N)
    mu -= mu.mean()
    D = 0.5 + rng.random(N)
    scale = SCALES[j]
    name = GRAMMARS[g]
    if name == "independent":
        return mu, dict(D=D)
    if name == "factor rank 1":
        return mu, dict(V=rng.normal(size=(N, 1)) * scale, D=D)
    if name == "factor rank 2":
        return mu, dict(V=rng.normal(size=(N, 2)) * scale, D=D)
    cluster = np.repeat(np.arange(N // 10), 10)
    loading = (0.5 + rng.random(N)) * scale
    return mu, dict(structure=Blocks(cluster=cluster, loading=loading, D=D))


def price(mu, kw):
    if "structure" in kw:
        return dispatch_probabilities(mu, kw["structure"])
    return race_probabilities(mu, **kw)


print(f"n={N}, {SEEDS} seeds per cell, default solver settings")
print(f"{'grammar':14s} {'scale':>5s}  converged  sweeps med/max  "
      f"worst residual  worst |mu err|  seconds")
for g, name in enumerate(GRAMMARS):
    for j, scale in enumerate(SCALES):
        if name == "independent" and j > 0:
            continue
        conv, its, res, err = [], [], [], []
        t0 = time.time()
        for r in range(SEEDS):
            mu, kw = field(g, j, r)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = price(mu, kw)
                mu_hat, info = abilities_from_race(p, return_info=True, **kw)
            conv.append(bool(info["converged"]))
            its.append(int(info["iterations"]))
            res.append(float(info["max_log_residual"]))
            err.append(float(np.abs(mu_hat - mu).max()))
        print(f"{name:14s} {('-' if name == 'independent' else f'{scale:g}'):>5s}  "
              f"{sum(conv):3d}/{SEEDS:<3d}    {int(np.median(its)):3d} / {max(its):<3d}     "
              f"{max(res):.1e}         {max(err):.1e}        {time.time()-t0:.0f}",
              flush=True)
