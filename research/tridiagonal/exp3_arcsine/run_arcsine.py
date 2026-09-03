"""Does the transfer-operator argmax recover Levy's arcsine law?

exp2 showed AR(1) (stationary, OU-like) argmax piling at the ends.
The clean classical case is the random walk / Brownian motion: the
TIME OF THE MAXIMUM of B on [0,1] has the arcsine density
  a(tau) = 1 / (pi sqrt(tau (1-tau))),
(Levy; Sparre Andersen for the discrete walk). A random walk is
Markov, so the same forward-backward argmax applies -- increments iid
N(0,1), transition N(x'; x, 1), variance growing with t (non-
stationary, grid widened to +-6 sqrt(n)). If the engine's argmax
profile matches arcsine, it is the discrete engine recovering the
continuum law, and the machinery then also handles the cases arcsine
does NOT cover: finite n, drift, reflecting/absorbing barriers,
non-unit or time-varying increments.
"""
import json
import os

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def walk_argmax(n, L=900):
    g = 6.0 * np.sqrt(n)
    s = np.linspace(-g, g, L)
    ds = s[1] - s[0]
    T = norm.pdf(s[None, :] - s[:, None]) * ds      # increment N(0,1)
    p0 = norm.pdf(s)                                 # X_1 = one step
    P = np.zeros(n)
    for mi, m in enumerate(s):
        if p0[mi] * ds < 1e-14:
            continue
        mask = s < m
        surv = np.empty(n)
        surv[0] = 1.0
        msg = norm.pdf(s - m) * ds * mask            # X_2 given X_1=m
        surv[1] = msg.sum() if n > 1 else 1.0
        for k in range(2, n):
            msg = (msg @ T) * mask
            surv[k] = msg.sum()
        w = p0[mi] * ds
        for t in range(n):
            P[t] += w * surv[t] * surv[n - 1 - t]
    return P / P.sum()                               # normalize grid


def mc_walk_argmax(n, m=800000, seed=0):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(m, n)), axis=1)
    return np.bincount(X.argmax(1), minlength=n) / m


def arcsine_binned(n):
    """Arcsine mass on each of n cells of [0,1]."""
    edges = np.linspace(0, 1, n + 1)
    # CDF of arcsine on [0,1] is (2/pi) arcsin(sqrt(tau))
    cdf = (2 / np.pi) * np.arcsin(np.sqrt(np.clip(edges, 0, 1)))
    return np.diff(cdf)


if __name__ == "__main__":
    results = {}
    for n in (20, 40):
        ex = walk_argmax(n)
        mc = mc_walk_argmax(n)
        ar = arcsine_binned(n)
        err_mc = np.abs(ex - mc).max()
        err_ar = np.abs(ex - ar).max()
        print(f"[random walk n={n}] max|engine-MC| {err_mc:.4f}  "
              f"max|engine-arcsine| {err_ar:.4f}")
        print(f"  ends/mid: engine {ex[0]:.4f}/{ex[n//2]:.4f}  "
              f"arcsine {ar[0]:.4f}/{ar[n//2]:.4f}  MC "
              f"{mc[0]:.4f}/{mc[n//2]:.4f}")
        results[f"n{n}"] = dict(engine=ex.tolist(), mc=mc.tolist(),
                                arcsine=ar.tolist(),
                                err_vs_mc=float(err_mc),
                                err_vs_arcsine=float(err_ar))
    print("If engine ~ arcsine ~ MC, the transfer operator recovers "
          "Levy's law; and it extends to drift/barriers/finite-n where "
          "no closed arcsine holds.")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")
