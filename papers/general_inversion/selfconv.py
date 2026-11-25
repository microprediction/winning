"""Self-convergence of the lattice race on the GHK-table problems.

The paper says self-convergence places the lattice error near 1e-8.
This measures it: the three rank-2 problems of `bench.py ghk` (same
generator, same draw order) are priced at the default settings
(257 lattice points, default factor nodes) and at a refined setting
(1025 lattice points, 41 Gauss-Hermite nodes per factor dimension), and
the two answers are compared. No random numbers beyond the problem draw.

    python selfconv.py
"""
import numpy as np

from winning.factor.core import hermite_nodes
from winning.factor.races import race_probabilities

rng = np.random.default_rng(4)
F, W = hermite_nodes(2, 41)
for n in (10, 50, 200):
    mu = rng.normal(size=n)
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.5 + rng.random(n)
    p = race_probabilities(mu, V=V, D=D, points=257)
    q = race_probabilities(mu, V=V, D=D, F=F, W=W, points=1025)
    print(f"n={n:4d}  TV {0.5*np.abs(p-q).sum():.1e}  "
          f"max|p-q| {np.abs(p-q).max():.1e}  "
          f"max|log p - log q| {np.abs(np.log(p)-np.log(q)).max():.1e}  "
          f"min p {q.min():.1e}")
