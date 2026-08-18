"""Multiplicative cavity identity: one field pass prices every competitor.

N competitors with performances X_i = mu_i + eps_i on a lattice; lowest wins.
Naively each win probability integrates against a product of N-1 survival
functions: O(N^2). The fast ability transform builds the field survival once,

    S_field(x) = prod_j S_j(x)

and recovers the field faced by competitor i by division,

    S_{-i}(x) = S_field(x) / S_i(x)

so all N leave-one-out fields cost O(N) total (Cotton 2021; `rest_min_cdf` in
the thurstone package). This is the multiplicative twin of the rank-one cavity
downdate demonstrated in cavity_downdate_demo.py.

Requires only numpy. Run:  python research/demos/race_field_demo.py
"""

import time

import numpy as np


def win_probabilities_naive(mu: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """p_i = int f_i(x) prod_{j != i} S_j(x) dx, product recomputed per competitor."""
    from math import erf

    z = (x[None, :] - mu[:, None]) / sigma
    F = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))
    f = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    S = 1.0 - F
    dx = x[1] - x[0]
    p = np.empty(len(mu))
    for i in range(len(mu)):
        rest = np.ones_like(x)
        for j in range(len(mu)):
            if j != i:
                rest *= S[j]
        p[i] = np.sum(f[i] * rest) * dx
    return p


def win_probabilities_fast(mu: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Same integral; field survival built once, each rest-field by division."""
    from scipy.special import ndtr  # vectorized normal CDF

    z = (x[None, :] - mu[:, None]) / sigma
    S = 1.0 - ndtr(z)
    f = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2.0 * np.pi))
    dx = x[1] - x[0]

    log_S_field = np.sum(np.log(np.maximum(S, 1e-300)), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # S_{-i} = S_field / S_i, in logs for stability
        log_rest = log_S_field[None, :] - np.log(np.maximum(S, 1e-300))
    rest = np.exp(np.clip(log_rest, -700, 0))
    return np.sum(f * rest, axis=1) * dx


def main() -> None:
    rng = np.random.default_rng(11)
    x = np.linspace(-8.0, 8.0, 2001)
    sigma = 1.0

    # --- correctness, small field -------------------------------------------
    N = 20
    mu = rng.normal(0.0, 1.0, size=N)
    p_naive = win_probabilities_naive(mu, x, sigma)
    p_fast = win_probabilities_fast(mu, x, sigma)
    print(f"N={N}: max |fast - naive| = {np.abs(p_fast - p_naive).max():.3e}")
    print(f"       sum of win probabilities (should be ~1): {p_fast.sum():.6f}")

    # --- Monte Carlo sanity check --------------------------------------------
    draws = mu[:, None] + sigma * rng.standard_normal((N, 200_000))
    mc = np.bincount(np.argmin(draws, axis=0), minlength=N) / 200_000
    print(f"       max |fast - Monte Carlo| = {np.abs(p_fast - mc).max():.3e}")

    # --- timing, large field --------------------------------------------------
    for N in (500, 2000):
        mu = rng.normal(0.0, 1.0, size=N)
        t0 = time.perf_counter()
        win_probabilities_naive(mu, x, sigma)
        t_naive = time.perf_counter() - t0
        t0 = time.perf_counter()
        win_probabilities_fast(mu, x, sigma)
        t_fast = time.perf_counter() - t0
        print(f"N={N}: naive {t_naive:.2f}s, fast {t_fast:.3f}s  ({t_naive / t_fast:.0f}x)")


if __name__ == "__main__":
    main()
