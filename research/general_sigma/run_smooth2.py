"""Positive-part residual smoothing: keep full D analytic.

Fit Sigma ~ VV' + D (rank k). Residual E = Sigma - VV' - D is small and
indefinite; sample w over its POSITIVE eigenpart only (dropping the
negative part biases, measured here against the Sobol referee), and
average the analytic race p(mu + w; V, D). The integrand stays smooth
because every runner keeps its own idiosyncratic noise analytically.
"""
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
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

def smooth_pos(C, M, rng, k=3, sobol=True):
    V, D = rank_fit(C, k)
    E = C - V @ V.T - np.diag(D)
    w_, U = np.linalg.eigh(E)
    pos = w_ > 1e-10
    m = int(pos.sum())
    Lp = U[:, pos] * np.sqrt(w_[pos])          # (n, m)
    neg_mass = float(-w_[~pos].sum())
    if sobol and m > 0:
        mm = int(np.ceil(np.log2(max(M, 2))))
        z = ndtri(np.clip(qmc.Sobol(m, scramble=True,
                                    seed=int(rng.integers(1 << 30)))
                          .random_base2(mm), 1e-12, 1 - 1e-12)).T[:, :M]
    else:
        z = rng.standard_normal((m, M))
    W = Lp @ z if m else np.zeros((n, M))
    p = np.zeros(n)
    for i in range(W.shape[1]):
        p += race_probabilities(mu + W[:, i], V=V, D=D, points=161)
    return p / W.shape[1], m, neg_mass

TRUTHS = {"dense_onion": rc.onion_corr(n, rng=np.random.default_rng(11)),
          "sample_T60": rc.factor_world_sample(n, rng=np.random.default_rng(12)),
          "factor_sparse": rc.factor_plus_sparse(n, rng=np.random.default_rng(13))}

for tname, C in TRUTHS.items():
    eng = qmc.Sobol(n, scramble=True, seed=5)
    zz = ndtri(np.clip(eng.random_base2(18), 1e-12, 1-1e-12)).T
    L = np.linalg.cholesky(C + 1e-10*np.eye(n))
    ref = np.bincount(np.argmin(mu[:, None] + L @ zz, axis=0),
                      minlength=n) / zz.shape[1]
    REPS = 20
    plain = []
    t0 = time.time()
    for r in range(REPS):
        z = np.random.default_rng(100+r).standard_normal((n, 4096))
        plain.append(np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                                 minlength=n) / 4096)
    t_plain = (time.time() - t0) / REPS
    sd_p = np.array(plain).std(axis=0).mean()
    for k in (3, 6):
        for M in (64, 256):
            sm = []
            t0 = time.time()
            for r in range(REPS):
                p, m, negm = smooth_pos(C, M, np.random.default_rng(100+r), k=k)
                sm.append(p)
            t_sm = (time.time() - t0) / REPS
            A = np.array(sm)
            sd_s = A.std(axis=0).mean()
            bias = np.abs(A.mean(axis=0) - ref).max()
            vrf = (sd_p / max(sd_s, 1e-16)) ** 2
            print(f"{tname:14s} k={k} M={M:4d} resid_dims={m:2d} "
                  f"negmass {negm:.3f}  sd {sd_s:.2e} (plain {sd_p:.2e})  "
                  f"VRF {vrf:8.1f}  maxbias {bias:.1e}  "
                  f"t {t_sm:.2f}s vs {t_plain*1000:.0f}ms", flush=True)
