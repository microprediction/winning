"""Inversion round trips at scale for one member of each covariance class.

For n contestants this prices a field under each grammar with the
shipped forward map, inverts the result with the shipped solver at its
default settings, and prints the solver's own report: whether it
converged, the sweeps used, the largest log-probability residual and the
largest ability error. The target is the forward map's own output, so
these are round trips, not comparisons with an external referee.

Classes (abilities N(0, 0.64) centred; idiosyncratic variances uniform
on [0.5, 1.5] except the independent case, which has unit variances):
  independent
  factor r1    loadings N(0, 0.25)
  blocks       clusters of 100, scalar loadings uniform on [0.4, 0.7]
  tree         the same clusters under a three-level tree: groups of ten
               clusters (strength 0.4), groups of ten groups (0.4), root (0.3)
  nested       the same blocks with a global factor, couplings uniform
               on [0, 0.5], gamma 0.8
Seed: default_rng(0) per class, drawn in the order mu, loadings, D,
couplings.

    python invert_structures.py                 # n = 1,000,000, all classes
    python invert_structures.py 10000           # smoke run
    python invert_structures.py 1000000 blocks tree
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import resource
import sys
import time
import warnings

import numpy as np

from winning.factor.races import abilities_from_race, race_probabilities
from winning.factor.structures import (Blocks, Nested, Tree,
                                       dispatch_probabilities)

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
which = sys.argv[2:] or ["independent", "factor r1", "blocks", "tree",
                         "nested"]
assert n % 10_000 == 0, "n must be a multiple of 10,000"


def tree_arrays(n_c):
    """Leaf clusters -> groups of ten -> groups of ten -> root."""
    g1, g2 = n_c // 10, n_c // 100
    parent = np.concatenate([n_c + np.arange(n_c) // 10,
                             n_c + g1 + np.arange(g1) // 10,
                             np.full(g2, n_c + g1 + g2), [-1]])
    strength = np.concatenate([np.zeros(n_c), np.full(g1, 0.4),
                               np.full(g2, 0.4), [0.3]])
    return parent, strength


def case(kind):
    rng = np.random.default_rng(0)
    mu = rng.normal(size=n) * 0.8
    mu -= mu.mean()
    if kind == "independent":
        return mu, dict(D=np.full(n, 1.0))
    if kind == "factor r1":
        V = rng.normal(size=(n, 1)) * 0.5
        return mu, dict(V=V, D=0.5 + rng.random(n))
    n_c = n // 100
    cluster = np.repeat(np.arange(n_c), 100)
    loading = 0.4 + 0.3 * rng.random(n)
    D = 0.5 + rng.random(n)
    if kind == "blocks":
        return mu, dict(structure=Blocks(cluster=cluster, loading=loading, D=D))
    if kind == "tree":
        parent, strength = tree_arrays(n_c)
        return mu, dict(structure=Tree(cluster=cluster, loading=loading, D=D,
                                       parent=parent, strength=strength))
    if kind == "nested":
        coupling = 0.5 * rng.random(n)
        return mu, dict(structure=Nested(cluster=cluster, loading=loading,
                                         D=D, coupling=coupling, gamma=0.8))
    raise SystemExit(f"unknown class {kind!r}")


def price(mu, kw):
    if "structure" in kw:
        return dispatch_probabilities(mu, kw["structure"])
    return race_probabilities(mu, **kw)


print(f"n={n:,d}")
for kind in which:
    mu, kw = case(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        p = price(mu, kw)
        t_fwd = time.time() - t0
        t0 = time.time()
        mu_hat, info = abilities_from_race(p, return_info=True, **kw)
        t_inv = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    print(f"{kind:12s} forward {t_fwd:7.1f} s  inversion {t_inv:8.1f} s  "
          f"converged {info['converged']}  sweeps {info['iterations']:3d}  "
          f"max|log p resid| {info['max_log_residual']:.1e}  "
          f"max|mu err| {np.abs(mu_hat - mu).max():.1e}  "
          f"min p {p.min():.1e}  peak {peak:.1f} GB", flush=True)
