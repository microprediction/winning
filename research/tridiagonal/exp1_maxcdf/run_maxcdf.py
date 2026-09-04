"""Exact max distribution of a Gauss-Markov (AR(1)) chain via the
forward transfer operator, validated against Monte Carlo.

Stationary AR(1): X_1 ~ N(0,1), X_t = phi X_{t-1} + sqrt(1-phi^2) e_t,
so marginals are N(0,1) and the precision is tridiagonal. The event
{max_t X_t <= x} restricts every state below x, so
  P(max <= x) = 1' (T_x)^{n-1} pi_x,
where pi_x(s) = phi(s) 1{s<=x} is the initial sub-density and
T_x(s'|s) = N(s'; phi s, 1-phi^2) 1{s'<=x} is the transition
restricted below x. One forward pass of sub-probability messages on a
state grid; O(n L) per threshold with the banded transition, linear
in n. This is the transfer-operator / sum-product realization on the
chain -- the exact order statistic a Kalman filter does not give, and
the AR(1) case where GHK's variance is known to blow up.
"""
import json
import os
import time

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))


def max_cdf(phi, n, xs, L=400):
    """P(max_t X_t <= x) for each x in xs. Grid the state on [-g, g]."""
    g = 6.0
    s = np.linspace(-g, g, L)
    ds = s[1] - s[0]
    sd = np.sqrt(1.0 - phi ** 2)
    # transition density T[i, j] = N(s_j; phi s_i, sd^2) * ds
    T = norm.pdf((s[None, :] - phi * s[:, None]) / sd) / sd * ds
    p0 = norm.pdf(s)                      # stationary marginal density
    out = np.empty(len(xs))
    for k, x in enumerate(xs):
        mask = s <= x
        v = p0 * mask * ds                # sub-density of X_1 <= x
        Tx = T * mask[None, :]            # restrict target states <= x
        for _ in range(n - 1):
            v = v @ Tx
        out[k] = v.sum()
    return out


def mc_max_cdf(phi, n, xs, m=400000, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.sqrt(1.0 - phi ** 2)
    x = rng.normal(size=(m,))
    mx = x.copy()
    for _ in range(n - 1):
        x = phi * x + sd * rng.normal(size=m)
        mx = np.maximum(mx, x)
    return np.array([(mx <= xx).mean() for xx in xs])


if __name__ == "__main__":
    xs = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    results = {}
    for phi in (0.0, 0.5, 0.9, 0.99):
        n = 50
        t0 = time.time()
        exact = max_cdf(phi, n, xs)
        t_ex = time.time() - t0
        mc = mc_max_cdf(phi, n, xs)
        err = np.abs(exact - mc).max()
        print(f"[phi={phi} n={n}] max|exact-MC| {err:.4f}  "
              f"(exact {t_ex*1e3:.0f} ms); P(max<=1.5) exact "
              f"{exact[4]:.4f} vs MC {mc[4]:.4f}")
        results[f"phi{phi}"] = dict(xs=xs.tolist(),
                                    exact=exact.tolist(),
                                    mc=mc.tolist(),
                                    max_err=float(err),
                                    seconds=t_ex)
    # linearity in n
    print("scaling (phi=0.9): ", end="")
    for n in (50, 100, 200, 400):
        t0 = time.time()
        max_cdf(0.9, n, xs)
        print(f"n={n}:{(time.time()-t0)*1e3:.0f}ms ", end="")
    print()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")
