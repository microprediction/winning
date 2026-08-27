"""Diagonal transfer: the unbiased smoothing split.

E = C - GG' - blockdiag - D is indefinite; instead of dropping E_neg,
transfer delta = |lambda_min(E)| of idiosyncratic variance from the
analytic diagonal into the sampled covariance: sample cov =
blockdiag + E + delta*I  (PSD), analytic D_a = D - delta. Nothing
dropped => exactly unbiased. Price: sharper analytic race (delta
eats idiosyncratic smoothing), handled by the adaptive quadrature.
"""
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
import importlib.util
spec = importlib.util.spec_from_file_location("s3", "run_smooth3.py")
import sys
src = open("run_smooth3.py").read()
exec(src[:src.index("def smoother")])      # loaders + fits
exec(src[src.index("def run_estimator"):src.index("REPS, M = 15, 128")])

TRUTHS = {"french30": load_industry_corr()}
spec2 = importlib.util.spec_from_file_location("run_cv", "run_cv.py")
rc = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rc)
TRUTHS["sample_T60"] = rc.factor_world_sample(20, rng=np.random.default_rng(12))
TRUTHS["dense_onion"] = rc.onion_corr(20, rng=np.random.default_rng(11))

def smoother_transfer(C, n_blocks=4, k=3):
    n = len(C)
    cluster, v, D, V = composed_fit(C, n_blocks, k=k)
    BD = np.zeros((n, n))
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        BD[np.ix_(idx, idx)] = np.outer(v[idx], v[idx])
    E = C - V @ V.T - np.diag(D) - BD
    lam_min = float(np.linalg.eigvalsh(E).min())
    delta = max(0.0, -lam_min) + 1e-9
    if delta >= D.min():
        # can't transfer more than the smallest idiosyncratic variance;
        # cap and accept the (tiny) remaining drop
        delta = 0.95 * D.min()
    S = BD + E + delta * np.eye(n)
    w_, U = np.linalg.eigh(S)
    dropped = float(-w_[w_ < 0].sum())
    keep = w_ > 1e-12
    Ls = U[:, keep] * np.sqrt(w_[keep])
    return Ls, V, D - delta, delta, dropped

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
    Ls, V, Da, delta, dropped = smoother_transfer(C)
    t0 = time.time()
    A = np.array([run_estimator(Ls, V, Da, mu, M,
                                np.random.default_rng(100+r))
                  for r in range(REPS)])
    t_est = (time.time() - t0) / REPS
    sd_s = A.std(axis=0).mean()
    bias = np.abs(A.mean(axis=0) - ref).max()
    print(f"{tname:12s} transfer M={M} delta={delta:.3f} minD_a={Da.min():.3f} "
          f"dropped={dropped:.1e}  sd {sd_s:.2e} (plain {sd_p:.2e})  "
          f"VRF {min((sd_p/max(sd_s,1e-16))**2, 99999):8.1f}  "
          f"maxbias {bias:.1e}  t {t_est:.1f}s", flush=True)
