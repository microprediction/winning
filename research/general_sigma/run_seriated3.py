"""Round 3: G G' (rank-k global) + seriated block rank-1 + diag -- the
composed grammar that strictly nests the global rank-k fit. Does the
block layer buy residual/TV beyond rank-3 alone?"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
from winning.factor.blocks import nested_race_probabilities
import importlib.util
spec = importlib.util.spec_from_file_location("rs2", "run_seriated2.py")
import sys
class _stop(Exception): pass
src = open("run_seriated2.py").read()
exec(src[:src.index("TRUTHS =")])          # reuse loaders/fitters

def composed_fit(C, n_blocks, k=3, sweeps=8):
    n = len(C)
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
    V, D0 = rank_fit(C, k)
    G = V.copy()
    v = np.zeros(n)
    for _ in range(sweeps):
        R = C - G @ G.T
        for c in np.unique(cluster):
            idx = np.where(cluster == c)[0]
            if len(idx) == 1:
                v[idx] = 0.0
                continue
            Rb = R[np.ix_(idx, idx)].copy()
            np.fill_diagonal(Rb, 0.0)
            wb, Ub = np.linalg.eigh(Rb)
            v[idx] = (Ub[:, -1] * np.sqrt(max(wb[-1], 0))) if wb[-1] > 0 else 0
        B = C.copy()
        for c in np.unique(cluster):
            idx = np.where(cluster == c)[0]
            B[np.ix_(idx, idx)] -= np.outer(v[idx], v[idx])
        # rank-k fit of B ignoring its diagonal (iterate diag refill)
        M = B.copy()
        for _ in range(10):
            w_, U_ = np.linalg.eigh(M)
            idx_ = np.argsort(-w_)[:k]
            G = U_[:, idx_] * np.sqrt(np.maximum(w_[idx_], 0))
            M = B.copy(); np.fill_diagonal(M, (G**2).sum(1))
    D = np.maximum(np.diag(C) - (G**2).sum(1) - v**2, 1e-4)
    St = G @ G.T + np.diag(D)
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        St[np.ix_(idx, idx)] += np.outer(v[idx], v[idx])
    return cluster, v, D, G, St

TRUTHS = {"french30": load_industry_corr(),
          "sample_T60": None}
import importlib.util as iu
spec2 = iu.spec_from_file_location("run_cv", "run_cv.py")
rc = iu.module_from_spec(spec2); spec2.loader.exec_module(rc)
TRUTHS["sample_T60"] = rc.factor_world_sample(20, rng=np.random.default_rng(12))

for tname, C in TRUTHS.items():
    n = len(C)
    rng = np.random.default_rng(0)
    mu = np.sort(rng.normal(size=n)) * 0.8
    L = np.linalg.cholesky(C + 1e-10*np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=5).random_base2(18),
                      1e-12, 1-1e-12)).T
    ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                      minlength=n) / z.shape[1]
    V, D = rank_fit(C, 3)
    E = C - V @ V.T - np.diag(D)
    ev = np.linalg.eigvalsh(E)
    p = race_probabilities(mu, V=V, D=D, points=501)
    print(f"{tname} (n={n})")
    print(f"   rank3       TV {0.5*np.abs(p-ref).sum():.2e}  "
          f"|resid|_1 {np.abs(E).sum()/2:6.2f}  negmass {-ev[ev<0].sum():.3f}",
          flush=True)
    for nb in (4, 6, 8):
        cluster, v, D2, G, St = composed_fit(C, nb, k=3)
        E = C - St
        ev = np.linalg.eigvalsh(E)
        p = nested_race_probabilities(mu, cluster, v, D2, coupling=G,
                                      gamma=1.0, points=257, qf=9)
        print(f"   rank3+blk{nb} TV {0.5*np.abs(p-ref).sum():.2e}  "
              f"|resid|_1 {np.abs(E).sum()/2:6.2f}  negmass {-ev[ev<0].sum():.3f}",
              flush=True)
