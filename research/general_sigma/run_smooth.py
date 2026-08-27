"""Rao-Blackwellized general-Sigma estimator vs indicator-CV and plain MC.

Split Sigma = alpha * Sigma_tilde + R with alpha the largest scalar
keeping R PSD (generalized eigenvalue); sample w ~ N(0, R) and average
the ANALYTIC race probability p(mu + w; alpha Sigma_tilde). Smooth
integrand, exact in expectation.
"""
import time
import numpy as np
from scipy.linalg import eigh
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

def smoothed(C, M, rng, k=3):
    V, D = rank_fit(C, k)
    St = V @ V.T + np.diag(D)
    lam = eigh(C, St, eigvals_only=True)
    alpha = float(np.clip(lam.min() * 0.999, 1e-3, 1.0))
    R = C - alpha * St
    w_, U = np.linalg.eigh(R)
    Lr = U * np.sqrt(np.maximum(w_, 0))
    Va, Da = V * np.sqrt(alpha), D * alpha
    W = Lr @ rng.standard_normal((n, M))
    p = np.zeros(n)
    for m in range(M):
        p += race_probabilities(mu + W[:, m], V=Va, D=Da, points=161)
    return p / M, alpha

TRUTHS = {"dense_onion": rc.onion_corr(n, rng=np.random.default_rng(11)),
          "sample_T60": rc.factor_world_sample(n, rng=np.random.default_rng(12)),
          "factor_sparse": rc.factor_plus_sparse(n, rng=np.random.default_rng(13))}
from scipy.stats import qmc
from scipy.special import ndtri

for tname, C in TRUTHS.items():
    eng = qmc.Sobol(n, scramble=True, seed=5)
    zz = ndtri(np.clip(eng.random_base2(18), 1e-12, 1-1e-12)).T
    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    y = mu[:, None] + L @ zz
    ref = np.bincount(np.argmin(y, axis=0), minlength=n) / zz.shape[1]

    # plain MC timing + sd at M=4096
    REPS = 20
    t0 = time.time()
    plain = []
    for r in range(REPS):
        z = np.random.default_rng(100 + r).standard_normal((n, 4096))
        w = np.argmin(mu[:, None] + L @ z, axis=0)
        plain.append(np.bincount(w, minlength=n) / 4096)
    t_plain = (time.time() - t0) / REPS
    plain = np.array(plain)
    sd_p = plain.std(axis=0).mean()

    for M in (64, 256):
        t0 = time.time()
        sm = []
        for r in range(REPS):
            p, alpha = smoothed(C, M, np.random.default_rng(100 + r))
            sm.append(p)
        t_sm = (time.time() - t0) / REPS
        sm = np.array(sm)
        sd_s = sm.std(axis=0).mean()
        bias = np.abs(sm.mean(axis=0) - ref).max()
        vrf = (sd_p / max(sd_s, 1e-16)) ** 2
        vrf_cost = vrf * (t_plain / t_sm)
        print(f"{tname:14s} M={M:4d} alpha={alpha:.2f}  sd {sd_s:.2e} "
              f"(plain {sd_p:.2e})  VRF {vrf:7.1f}  VRF/cost {vrf_cost:7.1f} "
              f" maxbias {bias:.1e}  t {t_sm:.2f}s vs {t_plain:.3f}s", flush=True)
