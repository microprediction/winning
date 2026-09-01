"""Verify Claim 1 of the Atlas note by simulation, and measure the
at-scale cost of the lattice side.

Claim: for independent Brownian performances X_i(t) = mu_i + sigma W_i,
the expected leading-pair collision local time of a NAMED pair (i, j)
over [0, T] equals 2 sigma^2 times the integral over horizons of the
static tie density of the time-s race:

    E Lambda_T^{ij}  =  2 sigma^2 int_0^T w_ij(s) ds,
    w_ij(s) = dp_i/dmu_j of the Gaussian race with D = sigma^2 s.

Two further checks ride along:
  * the ranked-gap decomposition: summing over named pairs gives the
    rank-1/rank-2 gap local time, and since the Jacobian has zero row
    sums, sum_{i<j} w_ij = -(1/2) tr J -- so the ranked collision rate
    is minus half the trace of the choice Jacobian, and the trace is
    just the own-slopes the forward pass returns anyway;
  * scale: the rate curve at n = 3000 (a Russell-sized field), timed.

Monte Carlo side: occupation estimator
    Lambda_hat = (2 sigma^2 / 2 eps) sum_steps 1{|Y| < eps, pair leads} dt
with dt small enough that the per-step move of Y (sd sigma sqrt(2 dt))
is well under eps.
"""
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from winning.factor.polish import race_jacobian  # noqa: E402
from winning.factor.races import race_probabilities  # noqa: E402


def lattice_integral(mu, sigma, T, i, j, n_s=48):
    """2 sigma^2 int_0^T w_ij(s) ds by the static Jacobian at a horizon
    grid (log-spaced toward 0, where the integrand dies for distinct
    locations)."""
    s_grid = np.geomspace(T / 400.0, T, n_s)
    vals = np.empty(n_s)
    for k, s in enumerate(s_grid):
        J = race_jacobian(mu, D=sigma ** 2 * s * np.ones(len(mu)))
        vals[k] = J[i, j]
    integral = np.trapezoid(vals, s_grid)
    # the [0, s_min] head, bounded by the smallest computed value
    integral += vals[0] * s_grid[0]
    return 2.0 * sigma ** 2 * integral, s_grid, vals


def mc_local_time(mu, sigma, T, i, j, paths, dt, eps, seed=0,
                  block=20000):
    """Occupation estimator of E Lambda_T^{ij} (pair leads = both ahead
    of the rest of the field)."""
    rng = np.random.default_rng(seed)
    n = len(mu)
    steps = int(round(T / dt))
    total = 0.0
    total_sq = 0.0
    done = 0
    others = [k for k in range(n) if k not in (i, j)]
    while done < paths:
        b = min(block, paths - done)
        X = np.tile(mu, (b, 1)).astype(float)
        occ = np.zeros(b)
        for _ in range(steps):
            X += sigma * np.sqrt(dt) * rng.standard_normal((b, n))
            Y = X[:, i] - X[:, j]
            lead = np.maximum(X[:, i], X[:, j]) <= X[:, others].min(axis=1)
            occ += (np.abs(Y) < eps) & lead
        lam = occ * dt * (2.0 * sigma ** 2) / (2.0 * eps)
        total += lam.sum()
        total_sq += (lam ** 2).sum()
        done += b
    mean = total / paths
    se = np.sqrt(max(total_sq / paths - mean ** 2, 0.0) / paths)
    return mean, se


def rate_curve_at_scale(n, sigma, T, n_s=24, seed=11):
    """-(1/2) tr J(s) over a horizon grid at Russell scale, using the
    own-slopes the forward pass returns: the expected rank-1/rank-2
    collision rate profile for the whole field, timed."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.5, n)
    mu -= mu.mean()
    s_grid = np.geomspace(T / 100.0, T, n_s)
    t0 = time.time()
    rate = np.empty(n_s)
    for k, s in enumerate(s_grid):
        _, slopes = race_probabilities(mu, V=np.zeros((n, 1)),
                                       D=sigma ** 2 * s * np.ones(n),
                                       return_slopes=True)
        rate[k] = -2.0 * sigma ** 2 * 0.5 * slopes.sum()
    wall = time.time() - t0
    return s_grid, rate, wall


if __name__ == "__main__":
    out = {}

    # --- Claim 1 verification, n = 5 ---------------------------------
    sigma = 1.0
    T = 0.5
    mu = np.array([-0.3, -0.1, 0.0, 0.25, 0.5])
    i, j = 0, 1                      # the two favourites (min-wins)

    latt, s_grid, vals = lattice_integral(mu, sigma, T, i, j)
    print(f"lattice: 2 sigma^2 int w_ij ds = {latt:.5f}")

    mc, se = mc_local_time(mu, sigma, T, i, j, paths=60_000, dt=1e-4,
                           eps=0.05, seed=1)
    z = (mc - latt) / se
    print(f"MC:      E Lambda = {mc:.5f} +/- {se:.5f}   (z = {z:+.2f})")
    out["claim1"] = dict(lattice=latt, mc=mc, mc_se=se, z=float(z),
                         n=5, T=T, paths=60_000, dt=1e-4, eps=0.05)

    # a second pair, non-adjacent favourites, same run settings
    i2, j2 = 0, 3
    latt2, _, _ = lattice_integral(mu, sigma, T, i2, j2)
    mc2, se2 = mc_local_time(mu, sigma, T, i2, j2, paths=60_000, dt=1e-4,
                             eps=0.05, seed=2)
    z2 = (mc2 - latt2) / se2
    print(f"pair (0,3): lattice {latt2:.5f}  MC {mc2:.5f} +/- {se2:.5f}"
          f"  (z = {z2:+.2f})")
    out["claim1_pair2"] = dict(lattice=latt2, mc=mc2, mc_se=se2,
                               z=float(z2))

    # --- trace identity, same field ----------------------------------
    s_mid = 0.25
    J = race_jacobian(mu, D=sigma ** 2 * s_mid * np.ones(5))
    off_sum = J[np.triu_indices(5, 1)].sum()
    print(f"trace identity at s={s_mid}: sum_(i<j) w_ij = {off_sum:.6f},"
          f"  -tr(J)/2 = {-np.trace(J)/2:.6f}")
    out["trace_identity"] = dict(off_sum=float(off_sum),
                                 half_neg_trace=float(-np.trace(J) / 2))

    # --- Russell-scale rate curve -------------------------------------
    s_grid3, rate3, wall3 = rate_curve_at_scale(3000, sigma, T=1.0)
    print(f"n=3000: full rank-1/2 collision rate curve over 24 horizons"
          f" in {wall3:.2f} s (peak rate {rate3.max():.3f}/unit time)")
    out["russell_scale"] = dict(n=3000, horizons=24,
                                wall_seconds=round(wall3, 2),
                                peak_rate=float(rate3.max()))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
