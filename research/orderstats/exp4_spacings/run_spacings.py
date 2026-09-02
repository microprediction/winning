"""Winner margin and range of a factor-correlated field: the teeth test.

Two formulas from SPACINGS.md, both one-dimensional shared-field
integrals under conditional independence (min-race convention, rank-1
factor + diagonal here):

  margin   P(D_1 > g, I = i) = E_z int f_iz(x) prod_{j!=i} S_jz(x+g) dx
           -- the winner integral with the survival field SHIFTED by g;
  range    P(R <= r)         = E_z int sum_i f_iz(x)
                                     prod_{j!=i} [F_jz(x+r) - F_jz(x)] dx
           -- the unique minimum at x, everyone else inside (x, x+r].

Margins g and r are taken as multiples of the lattice step so the
shifted fields are array slices; each additional margin value then
costs one O(nL) pass per factor node.

Referees, in order of strictness:
  1. n=4, general rank-1-plus-diagonal covariance, 4e6-sample Monte
     Carlo: full margin and range curves, plus the winner-specific
     margin decomposition P(D_1 > g, I = i) for every i. This is the
     regime Gupta-Pillai-Steck (Biometrika 1964) could still reach
     with general correlation.
  2. n=50, two-cluster loadings, Monte Carlo again -- past the
     classical range formulas entirely.
  3. Scaling: the same margin curve at n = 2,000 and n = 20,000,
     wall-clock reported (the claim is O(nLQ) per margin value).
"""
import json
import os
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

from scipy.stats import norm  # noqa: E402

QZ = 25          # Gauss-Hermite nodes over the shared factor
L = 1201         # lattice points
TINY = 1e-300


def gh_nodes(q):
    x, w = np.polynomial.hermite_e.hermegauss(q)   # probabilists'
    return x, w / w.sum()


def grids(mu, v, d, z):
    sd = np.sqrt(d)
    lo = (mu[None, :] + np.outer(z, v)).min() - 8.0 * sd.max()
    hi = (mu[None, :] + np.outer(z, v)).max() + 6.0 * sd.max()
    return np.linspace(lo, hi, L)


def margin_and_range(mu, v, d, g_steps, r_steps, want_by_winner=False):
    """P(D_1 > g) for g = g_steps*h and P(R <= r) for r = r_steps*h.

    Returns (h, margin_curve, range_curve, by_winner) where by_winner
    is the (len(g_steps) x n) matrix P(D_1 > g, I = i) if requested.
    """
    z, w = gh_nodes(QZ)
    x = grids(mu, v, d, z)
    h = x[1] - x[0]
    sd = np.sqrt(d)
    marg = np.zeros(len(g_steps))
    rng_ = np.zeros(len(r_steps))
    byw = np.zeros((len(g_steps), len(mu))) if want_by_winner else None
    for zq, wq in zip(z, w):
        m = mu + v * zq                                   # (n,)
        t = (x[None, :] - m[:, None]) / sd[:, None]       # (n, L)
        logS = norm.logsf(t)
        F = norm.cdf(t)
        f = np.exp(norm.logpdf(t)) / sd[:, None]
        for gi, k in enumerate(g_steps):
            # field shifted by g = k*h: columns k: pair with x = :L-k
            logG = logS[:, k:].sum(0)                     # (L-k,)
            integ = (f[:, : L - k]
                     * np.exp(logG[None, :] - logS[:, k:]))
            if want_by_winner:
                byw[gi] += wq * integ.sum(1) * h
            marg[gi] += wq * integ.sum() * h
        for ri, k in enumerate(r_steps):
            B = np.clip(F[:, k:] - F[:, : L - k], TINY, None)
            logBt = np.log(B).sum(0)
            integ = f[:, : L - k] * np.exp(logBt[None, :] - np.log(B))
            rng_[ri] += wq * integ.sum() * h
    return h, marg, rng_, byw


def mc_curves(mu, v, d, g_vals, r_vals, n_mc, seed):
    rng = np.random.default_rng(seed)
    n = len(mu)
    marg = np.zeros(len(g_vals))
    rng_c = np.zeros(len(r_vals))
    byw = np.zeros((len(g_vals), n))
    done = 0
    block = min(n_mc, max(1, 40_000_000 // n))
    while done < n_mc:
        b = min(block, n_mc - done)
        X = (mu[None, :] + rng.normal(size=(b, 1)) * v[None, :]
             + rng.normal(size=(b, n)) * np.sqrt(d)[None, :])
        part = np.partition(X, 1, axis=1)
        d1 = part[:, 1] - part[:, 0]
        win = X.argmin(1)
        span = X.max(1) - X.min(1)
        for gi, g in enumerate(g_vals):
            hit = d1 > g
            marg[gi] += hit.sum()
            byw[gi] += np.bincount(win[hit], minlength=n)
        for ri, r in enumerate(r_vals):
            rng_c[ri] += (span <= r).sum()
        done += b
    return marg / n_mc, rng_c / n_mc, byw / n_mc


def referee(name, n, mu, v, d, n_mc, seed, results):
    g_steps = np.array([0, 4, 10, 20, 40, 80, 140])
    r_steps = (np.array([0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.2])
               * L / 14).astype(int)
    t0 = time.time()
    h, marg, rng_, byw = margin_and_range(mu, v, d, g_steps, r_steps,
                                          want_by_winner=True)
    t_field = time.time() - t0
    g_vals, r_vals = g_steps * h, r_steps * h
    t0 = time.time()
    mc_m, mc_r, mc_b = mc_curves(mu, v, d, g_vals, r_vals, n_mc, seed)
    t_mc = time.time() - t0
    se = np.sqrt(np.maximum(mc_m * (1 - mc_m), 1e-12) / n_mc)
    err_m = np.abs(marg - mc_m)
    err_r = np.abs(rng_ - mc_r)
    err_b = np.abs(byw - mc_b).max()
    print(f"[{name}] n={n}  field {t_field:.2f}s  MC {t_mc:.1f}s")
    print(f"  margin  max|err| {err_m.max():.2e} "
          f"(max MC se {se.max():.1e}); curve "
          + " ".join(f"{p:.4f}" for p in marg))
    print(f"  range   max|err| {err_r.max():.2e}; curve "
          + " ".join(f"{p:.4f}" for p in rng_))
    print(f"  winner-specific margin max|err| {err_b:.2e}")
    results[name] = dict(
        n=n, g=list(g_vals), margin=list(marg), margin_mc=list(mc_m),
        r=list(r_vals), range=list(rng_), range_mc=list(mc_r),
        max_err_margin=float(err_m.max()),
        max_err_range=float(err_r.max()),
        max_err_by_winner=float(err_b),
        t_field=t_field, t_mc=t_mc)


if __name__ == "__main__":
    results = {}
    rng0 = np.random.default_rng(7)

    # 1. n=4, general rank-1+diag: the classical-reach regime
    referee("n4_general", 4,
            np.array([0.0, 0.2, 0.5, 1.0]),
            np.array([0.8, -0.5, 0.3, 0.9]),
            np.array([0.5, 1.2, 0.8, 0.3]),
            n_mc=8_000_000, seed=11, results=results)

    # 2. n=50, two opposed clusters: past the classical formulas
    v50 = np.where(np.arange(50) < 25, 0.7, -0.7)
    referee("n50_clusters", 50,
            rng0.normal(0, 0.6, 50), v50, 0.4 + rng0.random(50),
            n_mc=4_000_000, seed=12, results=results)

    # 3. scaling: same curve at n=2e3 and n=2e4, field only
    for n in (2000, 20000):
        mu = rng0.normal(0, 0.6, n)
        v = rng0.normal(0, 0.5, n)
        d = 0.4 + rng0.random(n)
        g_steps = np.array([0, 4, 10, 20, 40, 80, 140])
        r_steps = (np.array([0.5, 1.0, 1.5]) * L / 14).astype(int)
        t0 = time.time()
        h, marg, rng_, _ = margin_and_range(mu, v, d, g_steps, r_steps)
        dt = time.time() - t0
        print(f"[scale n={n}] {dt:.2f}s  P(D1>0)={marg[0]:.4f} "
              f"(mass check ~1)  margin curve "
              + " ".join(f"{p:.4f}" for p in marg))
        results[f"scale_n{n}"] = dict(n=n, seconds=dt,
                                      mass_check=float(marg[0]),
                                      margin=list(marg),
                                      range=list(rng_))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fjson:
        json.dump(results, fjson, indent=2)
    print("wrote results.json")
