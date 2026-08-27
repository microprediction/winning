"""The composed smoothing split, head to head:

  analytic per draw:  rank-3 factor race (cheap, smooth)
  sampled:            blockdiag(v v') + E_pos      (blocks exact: PSD)
  dropped:            E_neg only  (the bias source, ~40% smaller than
                       the global-fit residual's)

vs the global-only split (sampled = E_pos of the rank-3 residual) and
plain MC. Bias vs a 2^18 Sobol referee; variance over replications.
"""
import time
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities

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
    return np.corrcoef(np.array(rows)[-2500:].T / 100.0)

def rank_fit(C, k):
    w, U = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    V = U[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    D = np.maximum(np.diag(C) - (V**2).sum(1), 1e-4)
    return V, D

def composed_fit(C, n_blocks, k=3, sweeps=8):
    n = len(C)
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
    G, _ = rank_fit(C, k)
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
        M = B.copy()
        for _ in range(10):
            w_, U_ = np.linalg.eigh(M)
            idx_ = np.argsort(-w_)[:k]
            G = U_[:, idx_] * np.sqrt(np.maximum(w_[idx_], 0))
            M = B.copy(); np.fill_diagonal(M, (G**2).sum(1))
    D = np.maximum(np.diag(C) - (G**2).sum(1) - v**2, 1e-4)
    return cluster, v, D, G

def sample_cov_sqrt(S):
    w_, U = np.linalg.eigh(S)
    keep = w_ > 1e-12
    return U[:, keep] * np.sqrt(w_[keep])    # (n, m)

def smoother(C, mode, n_blocks=4, k=3):
    """Return (Lsamp, V_analytic, D_analytic, negmass)."""
    n = len(C)
    if mode == "global":
        V, D = rank_fit(C, k)
        E = C - V @ V.T - np.diag(D)
    else:
        cluster, v, D, V = composed_fit(C, n_blocks, k=k)
        BD = np.zeros((n, n))
        for c in np.unique(cluster):
            idx = np.where(cluster == c)[0]
            BD[np.ix_(idx, idx)] = np.outer(v[idx], v[idx])
        E = C - V @ V.T - np.diag(D) - BD
    w_, U = np.linalg.eigh(E)
    pos = w_ > 1e-10
    Epos = (U[:, pos] * w_[pos]) @ U[:, pos].T
    negmass = float(-w_[~pos].sum())
    S = Epos if mode == "global" else Epos + BD
    return sample_cov_sqrt(S), V, D, negmass

def run_estimator(Lsamp, V, D, mu, M, rng, sobol=True):
    m = Lsamp.shape[1]
    if sobol:
        mm = int(np.ceil(np.log2(max(M, 2))))
        z = ndtri(np.clip(qmc.Sobol(m, scramble=True,
                                    seed=int(rng.integers(1 << 30)))
                          .random_base2(mm), 1e-12, 1-1e-12)).T[:, :M]
    else:
        z = rng.standard_normal((m, M))
    W = Lsamp @ z
    p = np.zeros(len(mu))
    for i in range(W.shape[1]):
        p += race_probabilities(mu + W[:, i], V=V, D=D, points=161)
    return p / W.shape[1]

TRUTHS = {"french30": load_industry_corr()}
import importlib.util
spec = importlib.util.spec_from_file_location("run_cv", "run_cv.py")
rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)
TRUTHS["sample_T60"] = rc.factor_world_sample(20, rng=np.random.default_rng(12))
TRUTHS["dense_onion"] = rc.onion_corr(20, rng=np.random.default_rng(11))

REPS, M = 15, 128
for tname, C in TRUTHS.items():
    n = len(C)
    rng0 = np.random.default_rng(0)
    mu = np.sort(rng0.normal(size=n)) * 0.8
    L = np.linalg.cholesky(C + 1e-10*np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=5).random_base2(18),
                      1e-12, 1-1e-12)).T
    ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                      minlength=n) / z.shape[1]
    plain = []
    for r in range(REPS):
        zz = np.random.default_rng(100+r).standard_normal((n, 4096))
        plain.append(np.bincount(np.argmin(mu[:, None] + L @ zz, axis=0),
                                 minlength=n) / 4096)
    sd_p = np.array(plain).std(axis=0).mean()
    for mode in ("global", "composed"):
        Ls, V, D, negm = smoother(C, mode)
        t0 = time.time()
        A = np.array([run_estimator(Ls, V, D, mu, M,
                                    np.random.default_rng(100+r))
                      for r in range(REPS)])
        t_est = (time.time() - t0) / REPS
        sd_s = A.std(axis=0).mean()
        bias = np.abs(A.mean(axis=0) - ref).max()
        tv_bias = 0.5 * np.abs(A.mean(axis=0) / A.mean(axis=0).sum()
                               - ref).sum()
        print(f"{tname:12s} {mode:8s} M={M} sampdims={Ls.shape[1]:2d} "
              f"negmass {negm:.2f}  sd {sd_s:.2e} (plain@4096 {sd_p:.2e}) "
              f"VRF {min((sd_p/max(sd_s,1e-16))**2, 9999):7.1f}  "
              f"maxbias {bias:.1e}  TVbias {tv_bias:.1e}  t {t_est:.1f}s",
              flush=True)
