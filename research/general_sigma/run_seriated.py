"""HRP-seriation surrogate: linkage -> blocks -> nested grammar fit.

Fit arbitrary Sigma with OUR grammar via seriation: global rank-1 factor
+ per-block rank-1 effects + diagonal (the Nested race), blocks cut from
the dendrogram. Compare as (a) direct approximation and (b) smoothing
surrogate (positive-part residual), against global rank-k fits.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
from winning.factor.blocks import nested_race_probabilities
import importlib.util
spec = importlib.util.spec_from_file_location("run_cv", "run_cv.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
n, mu = rc.n, rc.mu

def rank_fit(C, k):
    w, U = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    V = U[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    D = np.maximum(np.diag(C) - (V**2).sum(1), 1e-4)
    return V, D

def nested_fit(C, n_blocks):
    """Seriation -> blocks -> global factor + per-block rank-1 + diag."""
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
    # global factor: top eigenvector of C
    w, U = np.linalg.eigh(C)
    g = U[:, -1] * np.sqrt(max(w[-1], 0))
    R = C - np.outer(g, g)
    v = np.zeros(n)
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        if len(idx) == 1:
            continue
        Rb = R[np.ix_(idx, idx)]
        wb, Ub = np.linalg.eigh(Rb)
        if wb[-1] > 0:
            v[idx] = Ub[:, -1] * np.sqrt(wb[-1])
    D = np.maximum(np.diag(C) - g**2 - v**2, 1e-4)
    St = np.outer(g, g) + np.diag(D)
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        St[np.ix_(idx, idx)] += np.outer(v[idx], v[idx])
    return cluster, v, D, g, St

def referee(C):
    L = np.linalg.cholesky(C + 1e-10*np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=5).random_base2(18),
                      1e-12, 1-1e-12)).T
    return np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                       minlength=n) / z.shape[1]

TRUTHS = {"dense_onion": rc.onion_corr(n, rng=np.random.default_rng(11)),
          "sample_T60": rc.factor_world_sample(n, rng=np.random.default_rng(12)),
          "factor_sparse": rc.factor_plus_sparse(n, rng=np.random.default_rng(13))}

for tname, C in TRUTHS.items():
    ref = referee(C)
    rows = []
    for k in (1, 3, 6):
        V, D = rank_fit(C, k)
        p = race_probabilities(mu, V=V, D=D, points=501)
        E = C - V @ V.T - np.diag(D)
        ev = np.linalg.eigvalsh(E)
        rows.append((f"rank{k}", 0.5*np.abs(p-ref).sum(),
                     np.abs(E).sum()/2, float(-ev[ev<0].sum())))
    for nb in (3, 4, 6):
        cluster, v, D, g, St = nested_fit(C, nb)
        p = nested_race_probabilities(mu, cluster, v, D, coupling=g,
                                      gamma=1.0, points=501)
        E = C - St
        ev = np.linalg.eigvalsh(E)
        rows.append((f"nested{nb}", 0.5*np.abs(p-ref).sum(),
                     np.abs(E).sum()/2, float(-ev[ev<0].sum())))
    print(tname)
    for name, tv, e1, negm in rows:
        print(f"   {name:8s} direct TV {tv:.2e}   |resid|_1 {e1:6.2f}  "
              f"negmass {negm:.3f}", flush=True)
