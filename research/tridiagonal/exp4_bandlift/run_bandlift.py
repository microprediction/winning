"""Band-diagonal by state lifting: the exact max distribution of an
AR(2) chain (precision bandwidth 2) via the lifted transfer operator.

Peter's correction (NOTES 2026-09-03): the track is BAND-DIAGONAL and
tridiagonal is bandwidth 1. The lift realizing that claim: the pair
Z_t = (X_t, X_{t-1}) IS Markov, so the same restricted forward pass
runs on the lifted state. Message m(a, b) = sub-density of
(X_t = a, X_{t-1} = b) with the whole past <= x; update

    m'(c, a) = sum_b N(c; phi1 a + phi2 b, sig^2) m(a, b) 1{c <= x},

one einsum per step against a precomputed (a, c, b) kernel:
O(n L^3) here, O(n L^{b+1}) in general -- linear in n, exponential
only in the bandwidth, exactly as the correction claimed. Stationary
AR(2) with unit marginal variance: sig^2 and the lag-1 correlation
rho1 = phi1/(1 - phi2) follow from Yule-Walker, and the initial
lifted message is the stationary bivariate normal restricted below x.

Run:  python research/tridiagonal/exp4_bandlift/run_bandlift.py
"""
import json
import os
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
G = 6.0


def ar2_params(phi1, phi2):
    """Innovation sd and lag-1 correlation of the UNIT-VARIANCE
    stationary AR(2) (Yule-Walker)."""
    assert abs(phi2) < 1 and phi1 + phi2 < 1 and phi2 - phi1 < 1
    sig2 = (1 + phi2) * ((1 - phi2) ** 2 - phi1 ** 2) / (1 - phi2)
    rho1 = phi1 / (1 - phi2)
    return np.sqrt(sig2), rho1


def max_cdf_lifted(phi1, phi2, n, xs, L=200):
    """P(max_{t<=n} X_t <= x) by the restricted forward pass on the
    lifted state (X_t, X_{t-1})."""
    sig, rho1 = ar2_params(phi1, phi2)
    s = np.linspace(-G, G, L)
    ds = s[1] - s[0]
    # kernel K[a, c, b] = N(s_c; phi1 s_a + phi2 s_b, sig^2) ds
    mu = phi1 * s[:, None] + phi2 * s[None, :]          # (a, b)
    K = norm.pdf((s[None, :, None] - mu[:, None, :]) / sig) / sig * ds
    # stationary bivariate initial density of (X_2, X_1)
    det = 1 - rho1 ** 2
    q = (s[:, None] ** 2 - 2 * rho1 * s[:, None] * s[None, :]
         + s[None, :] ** 2) / det
    p0 = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det)) * ds * ds
    out = np.empty(len(xs))
    for k, x in enumerate(xs):
        mask = s <= x
        m = p0 * mask[:, None] * mask[None, :]
        Kx = K * mask[None, :, None]
        for _ in range(n - 2):
            m = np.einsum("acb,ab->ca", Kx, m)
        out[k] = m.sum()
    return out


def mc_max_cdf(phi1, phi2, n, xs, m=400000, seed=0):
    rng = np.random.default_rng(seed)
    sig, rho1 = ar2_params(phi1, phi2)
    x_prev = rng.normal(size=m)
    x_cur = rho1 * x_prev + np.sqrt(1 - rho1 ** 2) * rng.normal(size=m)
    mx = np.maximum(x_prev, x_cur)
    for _ in range(n - 2):
        x_new = phi1 * x_cur + phi2 * x_prev + sig * rng.normal(size=m)
        mx = np.maximum(mx, x_new)
        x_prev, x_cur = x_cur, x_new
    return np.array([(mx <= xx).mean() for xx in xs])


def exp1_max_cdf(phi, n, xs, L=400):
    """The bandwidth-1 pass from exp1, for the phi2 = 0 embedding."""
    s = np.linspace(-G, G, L)
    ds = s[1] - s[0]
    sd = np.sqrt(1.0 - phi ** 2)
    T = norm.pdf((s[None, :] - phi * s[:, None]) / sd) / sd * ds
    p0 = norm.pdf(s)
    out = np.empty(len(xs))
    for k, x in enumerate(xs):
        mask = s <= x
        v = p0 * mask * ds
        Tx = T * mask[None, :]
        for _ in range(n - 1):
            v = v @ Tx
        out[k] = v.sum()
    return out


if __name__ == "__main__":
    xs = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    n = 50
    results = {}

    # 1. embedding check: phi2 = 0 must reproduce the bandwidth-1 pass
    # ON THE SAME GRID (same L isolates the lift from grid resolution;
    # both restrict the identical event X_1..X_n <= x)
    lift0 = max_cdf_lifted(0.9, 0.0, n, xs, L=200)
    band1 = exp1_max_cdf(0.9, n, xs, L=200)
    emb = np.abs(lift0 - band1).max()
    print(f"embedding phi=(0.9, 0), equal L=200: "
          f"max|lifted - exp1| = {emb:.2e}")
    results["embedding_err"] = float(emb)

    # 2. genuine bandwidth-2 regimes vs Monte Carlo
    for name, (p1, p2) in (("smooth", (0.5, 0.3)),
                           ("persistent", (0.6, 0.35)),
                           ("cyclical", (1.6, -0.8))):
        t0 = time.perf_counter()
        ex = max_cdf_lifted(p1, p2, n, xs)
        dt = time.perf_counter() - t0
        mc = mc_max_cdf(p1, p2, n, xs)
        err = np.abs(ex - mc).max()
        print(f"{name:10s} phi=({p1}, {p2}): max|exact - MC| = {err:.4f} "
              f"({dt * 1000:.0f} ms for {len(xs)} thresholds)")
        results[name] = {"phi": [p1, p2], "max_err": float(err),
                         "ms": dt * 1000,
                         "exact": ex.tolist(), "mc": mc.tolist()}

    # 3. linear in n (the O(n L^3) claim)
    for nn in (50, 200):
        t0 = time.perf_counter()
        max_cdf_lifted(0.5, 0.3, nn, xs[3:4])
        results[f"ms_n{nn}"] = (time.perf_counter() - t0) * 1000
    print(f"timing one threshold: n=50 {results['ms_n50']:.0f} ms, "
          f"n=200 {results['ms_n200']:.0f} ms "
          f"(ratio {results['ms_n200'] / results['ms_n50']:.1f}, linear ~ 4)")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("wrote results.json")
