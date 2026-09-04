"""The recursive order cavity: specified permutations and the A-matrix.

Three checks of SPACINGS.md's recursion section, min-wins throughout,
rank-1 factor + diagonal:

  1. n=6: EVERY permutation priced by the chain
         H_1 = F_{pi_1},  H_m(x) = int^x f_{pi_m} H_{m-1},
         P(pi) = E_z H_n(inf).
     Invariant: the 720 probabilities sum to one. Referee: 4e6-sample
     Monte Carlo permutation frequencies (max se ~ 5e-4 per cell).
  2. n=6: the top-2 prefix formula A_ij = E_z int f_j F_i
     prod_{l != i,j} S_l, checked three ways -- against the
     permutation sums (marginalizing 720 -> 30 ordered pairs),
     against MC, and rows/columns against the shipped
     race_probabilities / rank_probabilities.
  3. n=200: the matrix-free claim. A is accumulated as an integral of
     outer products u(x) w(x)' (u_i = F_i e^{logG - logS_i},
     w_j = f_j / S_j) -- O(n^2 L) to materialize only because the
     output has n^2 entries; each node's pass is one BLAS matmul.
     Invariants: sum_j A_ij = P(i first) from race_probabilities;
     sum_i A_ij = P(j second) from rank_probabilities column 2.

The lattice window here spans the WHOLE field (all finishers), per
the numerical wrinkle noted in SPACINGS.md.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from scipy.stats import norm                              # noqa: E402
from winning.factor import race_probabilities             # noqa: E402
from winning.factor.topk import rank_probabilities        # noqa: E402

QZ = 25
L = 3001


def gh_nodes(q):
    x, w = np.polynomial.hermite_e.hermegauss(q)
    return x, w / w.sum()


def conditional(mu, v, d, zq, x):
    m = mu + v * zq
    t = (x[None, :] - m[:, None]) / np.sqrt(d)[:, None]
    F = norm.cdf(t)
    f = np.exp(norm.logpdf(t)) / np.sqrt(d)[:, None]
    logS = norm.logsf(t)
    return F, f, logS


def window(mu, v, d, z):
    sd = np.sqrt(d)
    lo = (mu[None, :] + np.outer(z, v)).min() - 8 * sd.max()
    hi = (mu[None, :] + np.outer(z, v)).max() + 8 * sd.max()
    return np.linspace(lo, hi, L)


def cumtrap(y, h):
    out = np.empty_like(y)
    out[0] = 0.0
    np.cumsum(0.5 * h * (y[1:] + y[:-1]), out=out[1:])
    return out


def permutation_probability(perm, F, f, h):
    Hm = F[perm[0]]
    for m in perm[1:]:
        Hm = cumtrap(f[m] * Hm, h)
    return Hm[-1]


def all_permutations_n6(mu, v, d):
    z, w = gh_nodes(QZ)
    x = window(mu, v, d, z)
    h = x[1] - x[0]
    perms = list(itertools.permutations(range(len(mu))))
    P = np.zeros(len(perms))
    for zq, wq in zip(z, w):
        F, f, _ = conditional(mu, v, d, zq, x)
        for pi, perm in enumerate(perms):
            P[pi] += wq * permutation_probability(perm, F, f, h)
    return perms, P


def a_matrix(mu, v, d):
    """A_ij = P(i first, j second), full matrix, one matmul per node."""
    n = len(mu)
    z, w = gh_nodes(QZ)
    x = window(mu, v, d, z)
    h = x[1] - x[0]
    A = np.zeros((n, n))
    for zq, wq in zip(z, w):
        F, f, logS = conditional(mu, v, d, zq, x)
        logG = logS.sum(0)
        u = F * np.exp(np.clip(logG[None, :] - logS, -745.0, 0.0))
        wgt = f * np.exp(np.minimum(-logS, 600.0))
        A += wq * h * (u @ wgt.T)
    np.fill_diagonal(A, 0.0)
    return A


def mc_perms_and_pairs(mu, v, d, n_mc, seed):
    rng = np.random.default_rng(seed)
    n = len(mu)
    counts = {}
    pair = np.zeros((n, n))
    done, block = 0, 1_000_000
    while done < n_mc:
        b = min(block, n_mc - done)
        X = (mu[None, :] + rng.normal(size=(b, 1)) * v[None, :]
             + rng.normal(size=(b, n)) * np.sqrt(d)[None, :])
        order = X.argsort(1)
        for row in order:
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        np.add.at(pair, (order[:, 0], order[:, 1]), 1)
        done += b
    return {k: c / n_mc for k, c in counts.items()}, pair / n_mc


if __name__ == "__main__":
    results = {}
    rng0 = np.random.default_rng(3)

    # --- n=6, every permutation ---
    mu6 = np.array([0.0, 0.15, 0.3, 0.5, 0.8, 1.1])
    v6 = np.array([0.7, -0.4, 0.5, -0.6, 0.3, 0.8])
    d6 = np.array([0.5, 1.1, 0.7, 0.9, 0.4, 1.3])
    t0 = time.time()
    perms, P = all_permutations_n6(mu6, v6, d6)
    t_field = time.time() - t0
    mass = P.sum()
    mc_p, mc_pair = mc_perms_and_pairs(mu6, v6, d6, 4_000_000, 21)
    err = max(abs(P[i] - mc_p.get(perm, 0.0))
              for i, perm in enumerate(perms))
    print(f"[n6 permutations] 720 priced in {t_field:.2f}s  "
          f"sum={mass:.6f}  max|err vs MC|={err:.2e}")
    results["n6_permutations"] = dict(
        seconds=t_field, mass=float(mass), max_err_vs_mc=float(err),
        top5=[{"perm": list(perms[i]), "p": float(P[i])}
              for i in np.argsort(P)[::-1][:5]])

    # --- n=6, A-matrix three ways ---
    A6 = a_matrix(mu6, v6, d6)
    A6_from_perms = np.zeros((6, 6))
    for i, perm in enumerate(perms):
        A6_from_perms[perm[0], perm[1]] += P[i]
    err_int = np.abs(A6 - A6_from_perms).max()
    err_mc = np.abs(A6 - mc_pair).max()
    print(f"[n6 A-matrix] vs perm-sums {err_int:.2e}  vs MC {err_mc:.2e}"
          f"  total {A6.sum():.6f}")
    results["n6_pairs"] = dict(err_vs_perm_sums=float(err_int),
                               err_vs_mc=float(err_mc),
                               total=float(A6.sum()))

    # --- n=200, invariants against the shipped engine ---
    n = 200
    mu = rng0.normal(0, 0.6, n)
    v = rng0.normal(0, 0.5, n)
    d = 0.4 + rng0.random(n)
    t0 = time.time()
    A = a_matrix(mu, v, d)
    t_A = time.time() - t0
    p_first = race_probabilities(mu, V=v.reshape(-1, 1), D=d)
    R = rank_probabilities(mu, V=v.reshape(-1, 1), D=d)
    err_row = np.abs(A.sum(1) - p_first).max()
    err_col = np.abs(A.sum(0) - R[:, 1]).max()
    print(f"[n200 A-matrix] built in {t_A:.2f}s  total {A.sum():.6f}  "
          f"rows-vs-winner {err_row:.2e}  cols-vs-rank2 {err_col:.2e}")
    results["n200_pairs"] = dict(seconds=t_A, total=float(A.sum()),
                                 err_rows_vs_winner=float(err_row),
                                 err_cols_vs_rank2=float(err_col))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
