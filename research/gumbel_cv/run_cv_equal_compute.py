"""Equal-COMPUTE accounting for the Gumbel-twin CV (Peter's objection:
the twin's compute could buy more plain draws instead).

Both estimators get the same wall-clock budget; plain MC spends it all
on draws, the CV arm splits it between base draws and the coupled twin.
Report the net TV improvement at equal time, plus the measured per-draw
cost ratio."""
import time

import numpy as np
from scipy.special import ndtri

from winning.factor.races import softmax_probabilities

rng = np.random.default_rng(0)
n = 20
mu = np.sort(rng.normal(size=n)) * 0.8
tau = 1.0
p_exact = softmax_probabilities(mu, temperature=tau)
sd = tau * np.pi / np.sqrt(6)

def gev_min_quantile(u, xi):
    if abs(xi) < 1e-12:
        return np.log(-np.log1p(-u))
    return -(((-np.log1p(-u)) ** (-xi)) - 1.0) / xi

CASES = [("GEV xi=0.05", lambda u: gev_min_quantile(u, 0.05)),
         ("GEV xi=0.2", lambda u: gev_min_quantile(u, 0.2)),
         ("normal (matched)", lambda u: (np.pi / np.sqrt(6)) * ndtri(u))]

BUDGET_S = 0.25
REPS = 30
for label, q in CASES:
    # reference
    r = np.random.default_rng(9)
    U = r.random((4_000_000, n))
    p_ref = np.bincount((mu + tau * q(U)).argmin(1), minlength=n) / len(U)

    # measure per-batch costs
    r = np.random.default_rng(1)
    U = r.random((200_000, n))
    t0 = time.time(); Xb = mu + tau * q(U); ab = Xb.argmin(1); t_base = time.time() - t0
    t0 = time.time(); Xg = mu + tau * np.log(-np.log1p(-U)); ag = Xg.argmin(1); t_twin = time.time() - t0
    t0 = time.time(); _ = r.random((200_000, n)); t_rng = time.time() - t0
    ratio = (t_rng + t_base + t_twin) / (t_rng + t_base)

    tv_plain, tv_cv = [], []
    for rep in range(REPS):
        r = np.random.default_rng(100 + rep)
        # plain: draw until budget exhausted
        t0 = time.time(); counts = np.zeros(n); m = 0
        while time.time() - t0 < BUDGET_S:
            U = r.random((50_000, n))
            counts += np.bincount((mu + tau * q(U)).argmin(1), minlength=n)
            m += 50_000
        tv_plain.append(0.5 * np.abs(counts / m - p_ref).sum())
        # cv: same budget, coupled twin, beta = 1 (exact at xi=0)
        r = np.random.default_rng(100 + rep)
        t0 = time.time(); diff = np.zeros(n); m = 0
        while time.time() - t0 < BUDGET_S:
            U = r.random((50_000, n))
            G = np.log(-np.log1p(-U))
            ib = (mu + tau * q(U)).argmin(1)
            ig = (mu + tau * G).argmin(1)
            diff += np.bincount(ib, minlength=n) - np.bincount(ig, minlength=n)
            m += 50_000
        tv_cv.append(0.5 * np.abs(p_exact + diff / m - p_ref).sum())
    print(f"{label:18s} cost ratio x{ratio:.2f}   equal-time med TV: "
          f"plain {np.median(tv_plain):.2e}  cv {np.median(tv_cv):.2e}  "
          f"net x{np.median(tv_plain)/np.median(tv_cv):.1f}")
