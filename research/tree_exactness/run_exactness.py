"""Is the tree race's downward pass exact?

CRN Monte Carlo referee at 2^22 paths vs the kernel at high quadrature
(qa=15, points=1001), on trees of increasing depth. If deviations sit at
the MC noise floor at every depth, the pass is exact to quadrature
accuracy; if they plateau above it, the pass carries a bias and we
quantify it for the paper.
"""
import numpy as np
from winning.factor.blocks import tree_race_probabilities

rng = np.random.default_rng(2)

def mc_tree(mu, cluster, loading, D, parent, strength, m):
    n = len(mu)
    nT = len(parent)
    counts = np.zeros(n)
    b = 1 << 18
    done = 0
    while done < m:
        B = min(b, m - done)
        eff = rng.standard_normal((nT, B))
        shift = np.zeros((n, B))
        for i in range(n):
            u = cluster[i]
            shift[i] += loading[i] * eff[u]          # leaf-cluster effect
            t = parent[u]
            while t >= 0:
                shift[i] += strength[t] * eff[t]
                t = parent[t]
        y = mu[:, None] + shift + np.sqrt(D)[:, None] * rng.standard_normal((n, B))
        counts += np.bincount(np.argmin(y, axis=0), minlength=n)
        done += B
    return counts / m, np.sqrt((counts / m) * (1 - counts / m) / m)

def make_tree(nc, depth):
    """balanced-ish binary tree of the nc leaf clusters, given depth."""
    parent = [-1] * nc
    strength = [0.0] * nc
    level = list(range(nc))
    d = 0
    while len(level) > 1 and d < depth:
        nxt = []
        for k in range(0, len(level) - 1, 2):
            parent.append(-1); strength.append(0.35 / (1 + d))
            t = len(parent) - 1
            parent[level[k]] = t; parent[level[k + 1]] = t
            nxt.append(t)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
        d += 1
    # join any remaining forest under one root
    if len(level) > 1:
        parent.append(-1); strength.append(0.3)
        t = len(parent) - 1
        for u in level:
            parent[u] = t
    return np.array(parent), np.array(strength)

M = 1 << 22
for nc, depth, label in [(4, 1, "depth-1 (blocks)"), (4, 2, "depth-2"),
                         (8, 3, "depth-3"), (16, 4, "depth-4")]:
    n = 3 * nc
    mu = rng.normal(size=n)
    cluster = np.repeat(np.arange(nc), 3)
    loading = 0.2 + 0.3 * rng.random(n)
    D = 0.5 + rng.random(n)
    parent, strength = make_tree(nc, depth)
    p = tree_race_probabilities(mu, cluster, loading, D, parent, strength,
                                points=1001, qa=15)
    pm, sd = mc_tree(mu, cluster, loading, D, parent, strength, M)
    z = (p - pm) / np.maximum(sd, 1e-12)
    print(f"{label:16s} n={n:3d} nodes={len(parent):3d}  "
          f"max|p-MC| {np.abs(p-pm).max():.2e}  max|z| {np.abs(z).max():5.2f}  "
          f"rms z {np.sqrt((z*z).mean()):5.2f}", flush=True)
print(f"\nMC noise floor per entry ~ {1/np.sqrt(M):.1e}-scale; "
      "|z| ~ 1 means exact to quadrature accuracy, |z| >> 1 means bias")
