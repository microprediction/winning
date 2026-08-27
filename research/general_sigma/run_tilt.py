"""Group-tilted factor nodes for tail-relative accuracy.

A tail runner is priced by whichever node lands nearest its favourable
corner of factor space. Tilt: for a bucket of tail runners, translate
the factor Gaussian by m* toward the bucket's favourable direction and
reweight exactly (Gaussian importance identity), then price the bucket
from its own tilted call. Measured on the n=500 blocky truth: node-seed
relative drift per probability band, untilted vs tilted, plus a Botev
spot check on the deepest entries.
"""
import json
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)

rng = np.random.default_rng(21)
n = 500
G0 = rng.normal(size=(n, 3)) * [0.55, 0.3, 0.2]
blocks0 = rng.integers(0, 20, size=n)
v0 = 0.35 + 0.2 * rng.random(n)
C = G0 @ G0.T
for c in range(20):
    idx = np.where(blocks0 == c)[0]
    C[np.ix_(idx, idx)] += np.outer(v0[idx], v0[idx])
C += 0.03 * b.onion(n, rng)
C += np.diag(np.maximum(1.0 - np.diag(C), 0.05))
d_ = np.sqrt(np.diag(C)); C = C / np.outer(d_, d_)
mu = np.sort(np.random.default_rng(99).normal(size=n)) * 1.2

Vall, D, _, _ = b.fit_and_nodes(C, n_blocks=20, m=5, log2nodes=12)
r = Vall.shape[1]

def nodes(seed, m_star=None):
    z = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=seed)
                      .random_base2(12), 1e-12, 1 - 1e-12))
    w = np.full(len(z), 1.0 / len(z))
    if m_star is not None:
        z = z + m_star
        w = w * np.exp(-z @ m_star + 0.5 * m_star @ m_star)
    return z, w

def price(seed, m_star=None):
    F, W = nodes(seed, m_star)
    return race_probabilities(mu, V=Vall, D=D, F=F, W=W, points=257)

p_a, p_b = price(3), price(17)
# buckets of tail runners by untilted price
order = np.argsort(p_a)
p_tilt = {3: p_a.copy(), 17: p_b.copy()}
n_buckets = 5
tail_ids = order[:100]
for g in range(n_buckets):
    ids = tail_ids[g::n_buckets]
    # favourable direction: lower the bucket's conditional mean fastest
    d = -Vall[ids].mean(axis=0)
    nd = np.linalg.norm(d)
    if nd < 1e-12:
        continue
    d = d / nd
    for seed in (3, 17):
        best = None
        for scale in (1.0, 2.0, 3.0):
            pt = price(seed, scale * d)
            # keep the tilt that maximizes the bucket's minimum price
            score = np.log(np.maximum(pt[ids], 1e-300)).min()
            if best is None or score > best[0]:
                best = (score, pt)
        p_tilt[seed][ids] = best[1][ids]

def bands(pa, pb, label):
    print(label)
    for lo, hi in [(1e-6, 1e-4), (1e-8, 1e-6), (1e-12, 1e-8), (1e-20, 1e-12)]:
        band = (pa >= lo) & (pa < hi)
        if band.sum():
            rel = np.abs(pb[band] / np.maximum(pa[band], 1e-300) - 1)
            print(f"  p in [{lo:.0e},{hi:.0e}): {band.sum():4d}  "
                  f"median drift {np.median(rel):7.1%}  max {rel.max():9.1%}",
                  flush=True)

bands(p_a, p_b, "untilted (seed 3 vs 17):")
bands(p_tilt[3], p_tilt[17], "tilted   (seed 3 vs 17):")
tiny = order[:3]
json.dump({"mu": mu.tolist(), "C": C.tolist(),
           "idx": [int(t) for t in tiny],
           "p_untilted": [float(p_a[t]) for t in tiny],
           "p_tilted": [float(p_tilt[3][t]) for t in tiny]},
          open("tilt_botev_case.json", "w"))
print("deepest entries:", [(int(t), float(p_a[t]), float(p_tilt[3][t]))
                           for t in tiny])
