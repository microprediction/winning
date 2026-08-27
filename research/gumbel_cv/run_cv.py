"""Gumbel-twin control variate for frequency simulation.

Couple the base race and the Gumbel race through common uniforms
(comonotone per coordinate); the Gumbel race's probabilities are
analytic (softmax), so p_cv = p_base_mc - beta (p_gum_mc - softmax).
Measured: VRF per coordinate (optimal scalar beta per coordinate from
the same sample), as the base's GEV shape xi moves away from Gumbel
(xi=0 IS Gumbel), plus the normal base.
"""
import numpy as np

from winning.factor.races import softmax_probabilities

rng = np.random.default_rng(0)
n, M, REPS = 20, 20000, 30
mu = np.sort(rng.normal(size=n)) * 0.8
tau = 1.0
p_exact = softmax_probabilities(mu, temperature=tau)

def gev_min_quantile(u, xi):
    # min-convention GEV: negate the max-GEV of 1-u; xi=0 is min-Gumbel
    if abs(xi) < 1e-12:
        return np.log(-np.log1p(-u))     # min-Gumbel quantile
    return -(((-np.log1p(-u)) ** (-xi)) - 1.0) / xi

def run(base_q, label):
    vrfs, tv_plain, tv_cv = [], [], []
    for rep in range(REPS):
        r = np.random.default_rng(100 + rep)
        U = r.random((M, n))
        G = np.log(-np.log1p(-U))            # min-Gumbel
        X_g = mu + tau * G
        X_b = mu + tau * base_q(U)
        Ig = np.zeros((M, n)); Ib = np.zeros((M, n))
        Ig[np.arange(M), X_g.argmin(1)] = 1
        Ib[np.arange(M), X_b.argmin(1)] = 1
        cov = ((Ib - Ib.mean(0)) * (Ig - Ig.mean(0))).mean(0)
        var_g = Ig.var(0)
        beta = np.where(var_g > 0, cov / np.maximum(var_g, 1e-12), 0.0)
        p_plain = Ib.mean(0)
        p_cv = p_plain - beta * (Ig.mean(0) - p_exact)
        var_plain = Ib.var(0) / M
        var_cv = (Ib - beta * Ig).var(0) / M
        vrfs.append(np.median(np.where(var_cv > 0, var_plain / np.maximum(var_cv, 1e-15), np.inf)))
        tv_plain.append(0.5 * np.abs(p_plain - p_ref).sum())
        tv_cv.append(0.5 * np.abs(p_cv - p_ref).sum())
    print(f"{label:18s} median VRF {np.median(vrfs):8.1f}   "
          f"med TV plain {np.median(tv_plain):.2e}  cv {np.median(tv_cv):.2e}  "
          f"(x{np.median(tv_plain)/np.median(tv_cv):.1f})")

# references: exact for gumbel; big-MC for others
for xi in (0.0, 0.05, 0.1, 0.2, 0.4):
    q = lambda u, xi=xi: gev_min_quantile(u, xi)
    # reference by 4M draws
    r = np.random.default_rng(9)
    U = r.random((4_000_000, n))
    X = mu + tau * q(U)
    p_ref = np.bincount(X.argmin(1), minlength=n) / len(U)
    run(q, f"GEV xi={xi}")

from scipy.special import ndtri
r = np.random.default_rng(9)
U = r.random((4_000_000, n))
sd = tau * np.pi / np.sqrt(6)   # variance-matched normal
X = mu + sd * ndtri(U)
p_ref = np.bincount(X.argmin(1), minlength=n) / len(U)
run(lambda u: (np.pi / np.sqrt(6)) * ndtri(u), "normal (matched)")
