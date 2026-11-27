"""Low-dimension convergence barrage for the shipped diagonal inversion.

The inversion (winning.factor.races.abilities_from_race) has no
convergence proof. run_convergence_battery.py characterizes it at
n = 100. This script covers small fields, where single contestants
dominate and the shares are most uneven: every n in SIZES, six
covariance classes, four strengths of shared variation, three spreads
of ability, and two kinds of target.

  round trip   the target is the forward map's output at drawn
               abilities, so the ability error is known
  free         the target is a Dirichlet(1,...,1) draw, not produced by
               the model; only the residual can be checked

Default solver settings throughout (tolerance 1e-8 on the
log-probability residual; the solver's own sweep caps).

Fields: abilities N(0, spread^2) centred; idiosyncratic variances
uniform on [0.5, 1.5]; clusters are consecutive pairs.
  independent     no shared variation (scale unused)
  factor rank 1   loadings N(0, scale^2)
  factor rank 2   loadings N(0, scale^2) per column
  blocks          cluster loadings uniform on [0.5, 1.5] * scale
  tree            those blocks; clusters paired under groups, groups
                  under a root; strengths uniform on [0.2, 0.6] * scale
  nested          those blocks under a global factor; couplings uniform
                  on [0, 0.5] * scale, gamma 0.8
Seed for (class g, n, scale index j, spread index k, replicate r):
default_rng([2027, g, n, j, k, r]).

    python run_low_dimension_barrage.py                 # all classes, 20 seeds
    python run_low_dimension_barrage.py 20 "factor rank 2"
    python run_low_dimension_barrage.py 2               # smoke run

Prints one line per (class, n, scale) cell, then every run that did not
converge or raised, then totals.
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
from winning.factor.structures import (Blocks, Nested, Tree,
                                       dispatch_probabilities)

SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
CLASSES = ("independent", "factor rank 1", "factor rank 2", "blocks",
           "tree", "nested")
WHICH = sys.argv[2:] or CLASSES
SIZES = (2, 3, 4, 5, 8, 12, 20)
SCALES = (0.25, 0.5, 1.0, 2.0)
SPREADS = (0.5, 1.0, 2.0)


def field(g, n, j, k, r):
    rng = np.random.default_rng([2027, g, n, j, k, r])
    mu = rng.normal(size=n) * SPREADS[k]
    mu -= mu.mean()
    D = 0.5 + rng.random(n)
    scale = SCALES[j]
    free = rng.dirichlet(np.ones(n))
    name = CLASSES[g]
    if name == "independent":
        return mu, free, dict(D=D)
    if name == "factor rank 1":
        return mu, free, dict(V=rng.normal(size=(n, 1)) * scale, D=D)
    if name == "factor rank 2":
        return mu, free, dict(V=rng.normal(size=(n, 2)) * scale, D=D)
    cluster = np.arange(n) // 2
    n_c = int(cluster.max()) + 1
    loading = (0.5 + rng.random(n)) * scale
    if name == "blocks":
        return mu, free, dict(structure=Blocks(cluster=cluster,
                                               loading=loading, D=D))
    if name == "tree":
        g1 = (n_c + 1) // 2
        parent = np.concatenate([n_c + np.arange(n_c) // 2,
                                 np.full(g1, n_c + g1), [-1]])
        strength = np.concatenate([np.zeros(n_c),
                                   (0.2 + 0.4 * rng.random(g1 + 1)) * scale])
        return mu, free, dict(structure=Tree(cluster=cluster, loading=loading,
                                             D=D, parent=parent,
                                             strength=strength))
    coupling = 0.5 * rng.random(n) * scale
    return mu, free, dict(structure=Nested(cluster=cluster, loading=loading,
                                           D=D, coupling=coupling, gamma=0.8))


def price(mu, kw):
    if "structure" in kw:
        return dispatch_probabilities(mu, kw["structure"])
    return race_probabilities(mu, **kw)


print(f"{SEEDS} seeds per (class, n, scale, spread); spreads {SPREADS}; "
      f"default solver settings")
print(f"{'class':14s} {'n':>3s} {'scale':>5s}  round trip: conv  sweeps med/max  "
      f"worst resid  worst |mu err|  min p   | free: conv  sweeps med/max  "
      f"worst resid   seconds")
bad = []
tot = dict(rt=0, rt_ok=0, fr=0, fr_ok=0)
for name in WHICH:
    g = CLASSES.index(name)
    for n in SIZES:
        for j, scale in enumerate(SCALES):
            if name == "independent" and j > 0:
                continue
            t0 = time.time()
            c1, i1, r1, e1, pm = [], [], [], [], []
            c2, i2, r2 = [], [], []
            for k in range(len(SPREADS)):
                for r in range(SEEDS):
                    key = (name, n, scale, SPREADS[k], r)
                    try:
                        mu, free, kw = field(g, n, j, k, r)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            p = price(mu, kw)
                            mu_hat, info = abilities_from_race(
                                p, return_info=True, **kw)
                            _, info2 = abilities_from_race(
                                free, return_info=True, **kw)
                    except Exception as exc:  # recorded, not hidden
                        bad.append((key, "raised", repr(exc)[:120]))
                        continue
                    c1.append(bool(info["converged"]))
                    i1.append(int(info["iterations"]))
                    r1.append(float(info["max_log_residual"]))
                    e1.append(float(np.abs(mu_hat - mu).max()))
                    pm.append(float(p.min()))
                    c2.append(bool(info2["converged"]))
                    i2.append(int(info2["iterations"]))
                    r2.append(float(info2["max_log_residual"]))
                    if not c1[-1]:
                        bad.append((key, "round trip",
                                    f"resid {r1[-1]:.1e} after {i1[-1]} sweeps, "
                                    f"min p {pm[-1]:.1e}"))
                    if not c2[-1]:
                        bad.append((key, "free",
                                    f"resid {r2[-1]:.1e} after {i2[-1]} sweeps, "
                                    f"min target {free.min():.1e}"))
            tot["rt"] += len(c1); tot["rt_ok"] += sum(c1)
            tot["fr"] += len(c2); tot["fr_ok"] += sum(c2)
            if not c1:
                print(f"{name:14s} {n:3d} {scale:5g}  all runs raised", flush=True)
                continue
            print(f"{name:14s} {n:3d} "
                  f"{('-' if name == 'independent' else f'{scale:g}'):>5s}  "
                  f"{sum(c1):3d}/{len(c1):<3d}  {int(np.median(i1)):3d} / {max(i1):<3d}  "
                  f"{max(r1):.1e}  {max(e1):.1e}  {min(pm):.1e} | "
                  f"{sum(c2):3d}/{len(c2):<3d}  {int(np.median(i2)):3d} / {max(i2):<3d}  "
                  f"{max(r2):.1e}  {time.time() - t0:.0f}", flush=True)

print(f"\nnot converged or raised: {len(bad)}")
for key, kind, note in bad:
    print(f"  {kind:10s} class={key[0]!r} n={key[1]} scale={key[2]} "
          f"spread={key[3]} replicate={key[4]}: {note}")
print(f"\ntotals: round trip {tot['rt_ok']}/{tot['rt']} converged; "
      f"free target {tot['fr_ok']}/{tot['fr']} converged")
