"""Why a population of Luce choosers is Thurstonian in aggregate.

Yellott's theorem makes an individual Luce chooser a contest with Gumbel noise. Let
individual log-worths vary across the population,

    log u_i(theta) = mu_i + eps_i(theta),

so that person theta chooses argmax_i (mu_i + eps_i + G_i) with G iid Gumbel. The
population is then a contest whose noise is the convolution eps + G. Two consequences
follow. If individual taste is a sum of many small independent influences, the central
limit theorem makes eps approximately Gaussian. And as the scale of eps grows relative
to the Gumbel scale, the convolution is dominated by its Gaussian part, so the
population should behave as Thurstone's Case V.

This measures the approach to that limit. For each heterogeneity scale sigma, simulate
a population, compute the observed contraction slope lambda, and compare it with the
lambda that Case V implies for the population's own full-menu shares. The ratio starts
at zero, because a homogeneous Luce population does not contract at all, and should
approach one.

The relevance to the paper is that the aggregate caveat becomes the mechanism: the
reason Gaussian transport is the right default for population shares is that
populations are mixtures, and a mixture of Lucians with Gaussian taste variation is
Case V once taste outweighs choice noise.

Usage:  python heterogeneity_limit.py [n_draws]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

# Locations must scale with the noise, b_n / s_n -> a, or the limit is degenerate:
# holding the deterministic part fixed while the noise grows drives every share toward
# 1/K, and the Gaussian race is then recovered only at near-indifference. Setting
# mu = sigma * A keeps the shares put and makes the limit non-trivial.
A = np.array([0.55, 0.25, 0.0, -0.25, -0.55])
SIGMAS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0)


def slope(p, pair):
    """Through-origin contraction slope, favourite-oriented."""
    K = len(p)
    num = den = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            hi, lo = (i, j) if p[i] >= p[j] else (j, i)
            L = np.log(p[hi] / p[lo])
            q = min(max(pair(hi, lo), 1e-9), 1 - 1e-9)
            num += L * (-(np.log(q / (1 - q)) - L))
            den += L * L
    return num / den


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    rng = np.random.default_rng(0)
    K = len(A)
    print("population of heterogeneous Luce choosers, Gumbel scale fixed at 1")
    print(f"{'sigma':>7}{'top share':>11}{'observed':>11}{'Case V':>10}{'ratio':>8}")
    for s in SIGMAS:
        U = s * A + s * rng.standard_normal((n, K)) - np.log(-np.log(rng.random((n, K))))
        p = np.bincount(U.argmax(axis=1), minlength=K) / n
        obs = slope(p, lambda i, j: float((U[:, i] > U[:, j]).mean()))
        a, err = calibrate_np(list(p))
        a = np.asarray(a)
        def cv(i, j):
            w = win_probs_np(a[[i, j]])
            return float(w[0] / w.sum())
        got = slope(p, cv)
        print(f"{s:>7.2f}{p.max():>11.3f}{obs:>11.4f}{got:>10.4f}{obs/got:>8.2f}", flush=True)
    print("\nRatios approach one with the shares held fixed, so the limit is not the\n"
          "near-indifference corner.")


if __name__ == "__main__":
    main()
