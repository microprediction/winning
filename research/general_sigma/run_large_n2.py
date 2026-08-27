"""Stress battery for the one-call general-Sigma estimator."""
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities

def fit_and_nodes(C, k=3, n_blocks=40, m=5, log2nodes=11, seed=3,
                  use_blocks=True):
    n = len(C)
    w_, U_ = np.linalg.eigh(C)
    V = U_[:, -k:] * np.sqrt(np.maximum(w_[-k:], 0))
    v = np.zeros(n)
    cluster = np.zeros(n, int)
    if use_blocks and n_blocks > 1:
        d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
        Z = linkage(squareform(d, checks=False), method="average")
        cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
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
    ncl = len(np.unique(cluster))
    BD = np.zeros((n, ncl))
    for j, c in enumerate(np.unique(cluster)):
        idx = np.where(cluster == c)[0]
        BD[idx, j] = v[idx]
    E = C - V @ V.T - BD @ BD.T
    np.fill_diagonal(E, 0.0)
    cols = [V]
    if m > 0:
        wE, UE = np.linalg.eigh(E)
        cols.append(UE[:, -m:] * np.sqrt(np.maximum(wE[-m:], 0)))
    if use_blocks and n_blocks > 1:
        cols.append(BD)
    Vall = np.hstack(cols)
    D = np.maximum(np.diag(C) - (Vall ** 2).sum(1), 1e-3)
    r = Vall.shape[1]
    zq = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=seed)
                       .random_base2(log2nodes), 1e-12, 1 - 1e-12))
    return Vall, D, zq, np.full(len(zq), 1.0 / len(zq))

def one_call(mu, C, **kw):
    Vall, D, F, W = fit_and_nodes(C, **kw)
    return race_probabilities(mu, V=Vall, D=D, F=F, W=W, points=257)

def big_mc(mu, C, M, rng):
    n = len(C)
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    counts = np.zeros(n)
    done = 0
    while done < M:
        B = min(100_000, M - done)
        zz = rng.standard_normal((n, B))
        counts += np.bincount(np.argmin(mu[:, None] + L @ zz, axis=0),
                              minlength=n)
        done += B
    return counts / M, counts

def zscore_report(tag, p_one, p_mc, counts, M):
    sd = np.sqrt(np.maximum(p_mc * (1 - p_mc), 1e-300) / M)
    seen = counts >= 25
    z = (p_one[seen] - p_mc[seen]) / sd[seen]
    print(f"  {tag:24s} entries(MC-resolvable)={seen.sum():4d}  "
          f"max|z|={np.abs(z).max():5.2f}  frac|z|>3={100*(np.abs(z)>3).mean():4.1f}%  "
          f"rms z={np.sqrt((z**2).mean()):.2f}", flush=True)

def onion(n, rng):
    A = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = rng.dirichlet(np.ones(n) * 2.0) * n
    C = Q @ np.diag(lam) @ Q.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C

def sample_tlt_n(n, T, rng):
    B = rng.normal(size=(n, 3)) * [0.6, 0.35, 0.25]
    X = B @ rng.normal(size=(3, T)) + rng.normal(size=(n, T)) * 0.7
    C = np.corrcoef(X)
    w, U = np.linalg.eigh(C)
    C = U @ np.diag(np.maximum(w, 1e-4)) @ U.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    return C

rng = np.random.default_rng(21)
n = 500
M = 20_000_000
print(f"=== z-score tests, n={n}, MC {M//1_000_000}M draws ===")
TRUTHS = {"blocky": None, "dense_onion": onion(n, rng),
          "sample_T400": sample_tlt_n(n, 400, rng)}
G0 = rng.normal(size=(n, 3)) * [0.55, 0.3, 0.2]
blocks0 = rng.integers(0, 20, size=n)
v0 = 0.35 + 0.2 * rng.random(n)
C = G0 @ G0.T
for c in range(20):
    idx = np.where(blocks0 == c)[0]
    C[np.ix_(idx, idx)] += np.outer(v0[idx], v0[idx])
C += 0.03 * onion(n, rng)
C += np.diag(np.maximum(1.0 - np.diag(C), 0.05))
d_ = np.sqrt(np.diag(C)); TRUTHS["blocky"] = C / np.outer(d_, d_)

MUS = {"wide": np.sort(rng.normal(size=n)) * 1.2,
       "tight": rng.normal(size=n) * 0.05,
       "superstar": np.r_[-4.0, rng.normal(size=n - 1) * 0.5]}

for tname, C in TRUTHS.items():
    for mname, mu in MUS.items():
        p1 = one_call(mu, C, n_blocks=20)
        pmc, counts = big_mc(mu, C, M, np.random.default_rng(5))
        zscore_report(f"{tname}/{mname}", p1, pmc, counts, M)

print("\n=== ablation (blocky/wide) ===")
mu = MUS["wide"]; C = TRUTHS["blocky"]
pmc, counts = big_mc(mu, C, M, np.random.default_rng(5))
for kw, tag in [(dict(k=3, m=0, use_blocks=False), "k3 only"),
                (dict(k=6, m=0, use_blocks=False), "k6 only"),
                (dict(k=3, m=5, use_blocks=False), "k3+m5"),
                (dict(k=3, m=0, use_blocks=True, n_blocks=20), "k3+blocks"),
                (dict(k=3, m=5, use_blocks=True, n_blocks=20), "k3+m5+blocks"),
                (dict(k=3, m=10, use_blocks=True, n_blocks=20), "k3+m10+blocks")]:
    p1 = one_call(mu, C, **kw)
    zscore_report(tag, p1, pmc, counts, M)

print("\n=== tail relative-stability band (blocky/wide) ===")
p_a = one_call(mu, C, n_blocks=20, log2nodes=12, seed=3)
p_b = one_call(mu, C, n_blocks=20, log2nodes=12, seed=17)
for lo, hi in [(1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6), (1e-12, 1e-8),
               (1e-20, 1e-12)]:
    band = (p_a >= lo) & (p_a < hi)
    if band.sum():
        rel = np.abs(p_b[band] / p_a[band] - 1)
        print(f"  p in [{lo:.0e},{hi:.0e}): {band.sum():4d} runners  "
              f"median rel drift {np.median(rel):7.1%}  "
              f"max {rel.max():9.1%}", flush=True)

print("\n=== n scaling (one call, 2^11 nodes) ===")
for nn in (500, 2000, 8000):
    rngn = np.random.default_rng(3)
    Gn = rngn.normal(size=(nn, 3)) * [0.55, 0.3, 0.2]
    Cn = Gn @ Gn.T + 0.5 * np.eye(nn)
    dn = np.sqrt(np.diag(Cn)); Cn = Cn / np.outer(dn, dn)
    mun = np.sort(rngn.normal(size=nn)) * 1.2
    t0 = time.time()
    p1 = one_call(mun, Cn, n_blocks=1, use_blocks=False, m=5)
    print(f"  n={nn:5d}  {time.time()-t0:6.1f}s  (sum {p1.sum():.6f})",
          flush=True)
