"""Seeded benchmarks behind the tables in 'A General Contest Inversion
Algorithm'. Run on one laptop; wall clock varies, ratios are stable.

  python bench.py ghk     # the against-GHK table (n = 10, 50, 200)
  python bench.py law     # GHK cost-law points (n = 200, 500, 1000, R=1000)
  python bench.py scale   # lattice at n = 1e4, 1e5, 1e6
"""
import sys
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
import fastrace

mode = sys.argv[1] if len(sys.argv) > 1 else "ghk"
rng = np.random.default_rng(4 if mode != "scale" else 1)

if mode == "ghk":
    for n in (10, 50, 200):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 2)) * 0.4
        D = 0.5 + rng.random(n)
        L = np.linalg.cholesky(V @ V.T + np.diag(D))
        z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1)
                          .random_base2(20), 1e-12, 1 - 1e-12)).T
        ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                          minlength=n) / z.shape[1]
        t0 = time.time()
        p = race_probabilities(mu, V=V, D=D, points=257)
        line = (f"n={n:4d}  race {1e3*(time.time()-t0):7.1f} ms "
                f"TV {0.5*np.abs(p-ref).sum():.2e}")
        for R in (1000, 10000):
            t0 = time.time()
            g = np.asarray(fastrace.ghk_all_shares(-mu, V, D, R, 7))
            t_g = time.time() - t0
            g = g / g.sum()
            line += (f"  | GHK R={R}: {1e3*t_g:8.1f} ms "
                     f"TV {0.5*np.abs(g-ref).sum():.2e}")
        print(line, flush=True)
elif mode == "law":
    for n in (200, 500, 1000):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 2)) * 0.4
        D = 0.5 + rng.random(n)
        t0 = time.time()
        fastrace.ghk_all_shares(-mu, V, D, 1000, 7)
        print(f"n={n:5d}  {time.time()-t0:8.2f} s", flush=True)
elif mode == "scale":
    for n in (10_000, 100_000, 1_000_000):
        mu = rng.normal(size=n)
        V = rng.normal(size=(n, 1)) * 0.4
        D = 0.5 + rng.random(n)
        t0 = time.time()
        p = race_probabilities(mu, V=V, D=D, points=257)
        print(f"n={n:9,d}  {time.time()-t0:8.2f} s  (sum {p.sum():.6f})",
              flush=True)
