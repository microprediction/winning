"""Round 2: ALS-fitted nested grammar, and a genuinely blocky truth
(full-sample Ken French 30-industry correlation)."""
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

def load_industry_corr():
    raw = open("../hrp_finish/30_Industry_Portfolios_Daily.csv",
               errors="ignore").read().splitlines()
    start = next(i for i, l in enumerate(raw)
                 if l.strip()[:2] in ("19", "20") and "," in l)
    rows = []
    for l in raw[start:]:
        parts = l.split(",")
        if len(parts) != 31 or not parts[0].strip().isdigit():
            break
        vals = np.array([float(x) for x in parts[1:]])
        if (vals <= -99).any():
            continue
        rows.append(vals)
    R = np.array(rows)[-2500:] / 100.0
    return np.corrcoef(R.T)

def rank_fit(C, k):
    w, U = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    V = U[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    D = np.maximum(np.diag(C) - (V**2).sum(1), 1e-4)
    return V, D

def nested_fit_als(C, n_blocks, sweeps=6):
    n = len(C)
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
    w, U = np.linalg.eigh(C)
    g = U[:, -1] * np.sqrt(max(w[-1], 0))
    v = np.zeros(n)
    for _ in range(sweeps):
        # refit block loadings given g (off-diagonal within-block residual)
        R = C - np.outer(g, g)
        for c in np.unique(cluster):
            idx = np.where(cluster == c)[0]
            if len(idx) == 1:
                v[idx] = 0.0
                continue
            Rb = R[np.ix_(idx, idx)].copy()
            np.fill_diagonal(Rb, 0.0)     # fit off-diagonal only
            wb, Ub = np.linalg.eigh(Rb)
            v[idx] = (Ub[:, -1] * np.sqrt(max(wb[-1], 0))) if wb[-1] > 0 else 0
        # refit g given v (off-diagonal cross-block + within residual)
        B = C.copy()
        for c in np.unique(cluster):
            idx = np.where(cluster == c)[0]
            B[np.ix_(idx, idx)] -= np.outer(v[idx], v[idx])
        np.fill_diagonal(B, 0.0)
        wb, Ub = np.linalg.eigh(B)
        # diagonal-free rank-1 fit by a few power iterations with diag refill
        gg = Ub[:, -1] * np.sqrt(max(wb[-1], 0))
        for _ in range(10):
            M = B + np.diag(gg ** 2)
            wb2, Ub2 = np.linalg.eigh(M)
            gg = Ub2[:, -1] * np.sqrt(max(wb2[-1], 0))
        g = gg
    D = np.maximum(np.diag(C) - g**2 - v**2, 1e-4)
    St = np.outer(g, g) + np.diag(D)
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        St[np.ix_(idx, idx)] += np.outer(v[idx], v[idx])
    return cluster, v, D, g, St

TRUTHS = {"french30": load_industry_corr(),
          "sample_T60": rc.factor_world_sample(20, rng=np.random.default_rng(12))}

for tname, C in TRUTHS.items():
    n = len(C)
    rng = np.random.default_rng(0)
    mu = np.sort(rng.normal(size=n)) * 0.8
    L = np.linalg.cholesky(C + 1e-10*np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=5).random_base2(18),
                      1e-12, 1-1e-12)).T
    ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                      minlength=n) / z.shape[1]
    print(f"{tname} (n={n})")
    for k in (1, 3, 6):
        V, D = rank_fit(C, k)
        p = race_probabilities(mu, V=V, D=D, points=501)
        E = C - V @ V.T - np.diag(D)
        ev = np.linalg.eigvalsh(E)
        print(f"   rank{k}    TV {0.5*np.abs(p-ref).sum():.2e}  "
              f"|resid|_1 {np.abs(E).sum()/2:6.2f}  negmass {-ev[ev<0].sum():.3f}",
              flush=True)
    for nb in (4, 6, 8):
        cluster, v, D, g, St = nested_fit_als(C, nb)
        p = nested_race_probabilities(mu, cluster, v, D, coupling=g,
                                      gamma=1.0, points=501)
        E = C - St
        ev = np.linalg.eigvalsh(E)
        print(f"   nested{nb}  TV {0.5*np.abs(p-ref).sum():.2e}  "
              f"|resid|_1 {np.abs(E).sum()/2:6.2f}  negmass {-ev[ev<0].sum():.3f}",
              flush=True)
