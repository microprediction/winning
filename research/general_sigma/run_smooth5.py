"""Peter's deconvolution cheat: smooth with FULL D, adjust analytically.

  sample   w ~ N(0, E + delta I)   (PSD, nothing dropped)
  average  p(mu + w; V, D)         (FULL idio -> smooth integrand)
     = price(Sigma + delta I)      (over-smoothed by delta)
  adjust   + p_lattice(mu; V, D) - p_lattice(mu; V, D + delta)
           (the delta-sharpening map, computed exactly under the
            surrogate: two analytic races)

Bias = [pr(Sig)-pr(Sig+dI)] - [pr(Sig~)-pr(Sig~+dI)]: second order.
"""
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
import importlib.util
src = open("run_smooth3.py").read()
exec(src[:src.index("def smoother")])
exec(src[src.index("def run_estimator"):src.index("REPS, M = 15, 128")])

TRUTHS = {"french30": load_industry_corr()}
spec2 = importlib.util.spec_from_file_location("run_cv", "run_cv.py")
rc = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(rc)
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

    for nb, label in ((4, "composed"), (None, "global")):
        if nb is None:
            V, D = rank_fit(C, 3)
            BD = np.zeros((n, n))
        else:
            cluster, v, D, V = composed_fit(C, nb, k=3)
            BD = np.zeros((n, n))
            for c in np.unique(cluster):
                idx = np.where(cluster == c)[0]
                BD[np.ix_(idx, idx)] = np.outer(v[idx], v[idx])
        E = C - V @ V.T - np.diag(D) - BD
        lam_min = float(np.linalg.eigvalsh(E).min())
        delta = max(0.0, -lam_min) + 1e-9
        S = BD + E + delta * np.eye(n)
        w_, U = np.linalg.eigh(S)
        keep = w_ > 1e-12
        Ls = U[:, keep] * np.sqrt(w_[keep])
        # analytic adjustment: the delta-sharpening under the surrogate
        adj = (race_probabilities(mu, V=V, D=D, points=501)
               - race_probabilities(mu, V=V, D=D + delta, points=501))
        t0 = time.time()
        A = np.array([run_estimator(Ls, V, D, mu, M,
                                    np.random.default_rng(100+r)) + adj
                      for r in range(REPS)])
        t_est = (time.time() - t0) / REPS
        sd_s = A.std(axis=0).mean()
        bias = np.abs(A.mean(axis=0) - ref).max()
        print(f"{tname:12s} deconv-{label:8s} delta={delta:.3f} "
              f"sd {sd_s:.2e} (plain {sd_p:.2e})  "
              f"VRF {min((sd_p/max(sd_s,1e-16))**2, 99999):7.1f}  "
              f"maxbias {bias:.1e}  t {t_est:.1f}s", flush=True)
