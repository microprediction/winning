"""Rank-2 factor nodes: tensor Gauss-Hermite vs scrambled Sobol at equal node counts.

Behind the inversion paper's sentence that at rank two "a tensor
Gauss-Hermite rule beats scrambled Sobol at equal nodes by a wide measured
margin". Same lattice (points=501) throughout, so only the factor rule
differs. Reference: Gauss-Hermite Q=41 (1681 nodes before pruning). Sobol is
reported as the median and worst TV over 8 scrambles at 2^m nodes, set beside
the GH rule with the nearest node count. Seeded; run: python run_gh_vs_sobol.py
"""
import numpy as np
from winning.factor.core import hermite_nodes, qmc_nodes, win_probabilities_factor

rng = np.random.default_rng(24)
for n in (50, 200):
    mu = rng.normal(size=n); V = rng.normal(size=(n, 2)) * 0.4; D = 0.5 + rng.random(n)
    Fr, Wr = hermite_nodes(2, Q=41)
    ref = win_probabilities_factor(mu, V, D, Fr, Wr, points=501)
    print(f"n={n}: reference GH Q=41 ({len(Wr)} nodes after pruning)")
    for Q, m in ((3, 3), (4, 4), (6, 5), (8, 6), (11, 7), (16, 8)):
        F, W = hermite_nodes(2, Q=Q)
        gh = 0.5 * np.abs(win_probabilities_factor(mu, V, D, F, W, points=501) - ref).sum()
        sob = []
        for s in range(8):
            Fs, Ws = qmc_nodes(2, m=m, seed=s)
            sob.append(0.5 * np.abs(win_probabilities_factor(mu, V, D, Fs, Ws, points=501) - ref).sum())
        print(f"  GH Q={Q:2d} ({len(W):3d} nodes) TV {gh:.1e}   |   Sobol 2^{m} ({2**m:3d} nodes) "
              f"TV median {np.median(sob):.1e} worst {max(sob):.1e}", flush=True)
