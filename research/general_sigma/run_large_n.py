"""Large n: residual-as-extra-factors, one deterministic call vs MC.

Truth (n=2000): rank-3 global + 40 seriated blocks + diag + a genuinely
dense small residual (random orthogonal spectrum). Fit with the
composed pipeline, promote the residual's top eigendirections to extra
factor columns, and price the race in ONE call with Sobol factor nodes.
Score against MC on bulk TV, tail RELATIVE error, and wall clock.
"""
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities

rng = np.random.default_rng(7)
n = 2000

# ---- truth ----------------------------------------------------------------
G0 = rng.normal(size=(n, 3)) * [0.55, 0.3, 0.2]
blocks0 = rng.integers(0, 40, size=n)
v0 = 0.35 + 0.2 * rng.random(n)
A = rng.normal(size=(n, n))
Q0, _ = np.linalg.qr(A)
lamE = rng.dirichlet(np.ones(n) * 2.0) * n * 0.03      # small dense residual
C = G0 @ G0.T
for c in range(40):
    idx = np.where(blocks0 == c)[0]
    C[np.ix_(idx, idx)] += np.outer(v0[idx], v0[idx])
C += Q0 @ np.diag(lamE) @ Q0.T
C += np.diag(np.maximum(1.0 - np.diag(C), 0.05))
d_ = np.sqrt(np.diag(C)); C = C / np.outer(d_, d_)

mu = np.sort(rng.normal(size=n)) * 1.2                  # wide field: real tail

# ---- our estimator: fit + one call ---------------------------------------
t0 = time.time()
w_, U_ = np.linalg.eigh(C)
k = 3
V = U_[:, -k:] * np.sqrt(w_[-k:])
d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
Z = linkage(squareform(d, checks=False), method="average")
cluster = fcluster(Z, 40, criterion="maxclust") - 1
v = np.zeros(n)
R = C - V @ V.T
for c in np.unique(cluster):
    idx = np.where(cluster == c)[0]
    if len(idx) < 2:
        continue
    Rb = R[np.ix_(idx, idx)].copy()
    np.fill_diagonal(Rb, 0.0)
    wb, Ub = np.linalg.eigh(Rb)
    if wb[-1] > 0:
        v[idx] = Ub[:, -1] * np.sqrt(wb[-1])
# residual -> extra factors (block part kept as extra factor columns too:
# each block's loading vector is one sparse factor column)
BDcols = np.zeros((n, len(np.unique(cluster))))
for j, c in enumerate(np.unique(cluster)):
    idx = np.where(cluster == c)[0]
    BDcols[idx, j] = v[idx]
E = C - V @ V.T - BDcols @ BDcols.T
np.fill_diagonal(E, 0.0)
m = 5
wE, UE = np.linalg.eigh(E)
Vres = UE[:, -m:] * np.sqrt(np.maximum(wE[-m:], 0))
Vfull = np.hstack([V, Vres])                            # n x (k+m)
D = np.maximum(np.diag(C) - (Vfull**2).sum(1) - v**2, 1e-3)
t_fit = time.time() - t0
# NOTE: blocks enter exactly via the block kernel? Here: keep it ONE factor
# call by folding blocks in as 40 more (sparse) factor columns.
Vall = np.hstack([Vfull, BDcols])                       # n x (k+m+40)
r = Vall.shape[1]
t0 = time.time()
zq = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=3).random_base2(11),
                   1e-12, 1 - 1e-12))
p_one = race_probabilities(mu, V=Vall, D=D, F=zq,
                           W=np.full(len(zq), 1.0 / len(zq)), points=257)
t_call = time.time() - t0
# self-reference at 4x nodes
zq2 = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=9).random_base2(13),
                    1e-12, 1 - 1e-12))
p_ref_nodes = race_probabilities(mu, V=Vall, D=D, F=zq2,
                                 W=np.full(len(zq2), 1.0 / len(zq2)),
                                 points=257)

# ---- MC -------------------------------------------------------------------
t0 = time.time()
L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
t_chol = time.time() - t0
counts = np.zeros(n)
M_total = 1_000_000
t0 = time.time()
done = 0
while done < M_total:
    B = min(50_000, M_total - done)
    zz = rng.standard_normal((n, B))
    counts += np.bincount(np.argmin(mu[:, None] + L @ zz, axis=0),
                          minlength=n)
    done += B
t_mc = time.time() - t0
p_mc = counts / M_total

# ---- scoring --------------------------------------------------------------
bulk = p_mc > 1e-3                                       # MC-resolvable bulk
tv_bulk = float(np.abs(p_one[bulk] - p_mc[bulk]).sum() * 0.5)
node_conv = float(np.abs(p_one - p_ref_nodes).max())
tail = (p_one < 1e-5) & (p_one > 0)
mc_zero_frac = float((counts[tail] == 0).mean()) if tail.any() else float("nan")
rel_node = float(np.abs(p_one[tail] / np.maximum(p_ref_nodes[tail], 1e-300)
                        - 1).max()) if tail.any() else float("nan")
print(f"n={n} r={r} nodes={len(zq)}")
print(f"fit {t_fit:.1f}s | one-call {t_call:.1f}s | MC: chol {t_chol:.1f}s + "
      f"{M_total//1000}k draws {t_mc:.1f}s")
print(f"bulk TV (one-call vs 1M-draw MC, entries p>1e-3): {tv_bulk:.2e}")
print(f"node self-convergence (2^11 vs 2^13 nodes), max abs: {node_conv:.2e}")
print(f"tail (p<1e-5): {int(tail.sum())} runners; MC saw ZERO wins for "
      f"{100*mc_zero_frac:.0f}% of them; our node-to-node max relative "
      f"drift {rel_node:.1%}")
print(f"smallest resolved probability, one call: {p_one[p_one>0].min():.1e} "
      f"(MC floor at 1M draws: 1e-6)")
