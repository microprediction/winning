"""Experiment 20: the Chebyshev-separated (low-rank) pass, validated.

Both stages of the per-node O(NL) pass are smooth-kernel sums, and the
kernel matrices are numerically low-rank (measured in
paper/fast-kernel-notes.md). This prototype separates the kernels on a
tensor Chebyshev grid in (location m, scale sigma):

    k(x; m, s) ~= sum_c T_c(m, s) k_c(x),

turning field build and distribute into O(r(N + L)) per node. Barycentric
interpolation gives the runner coefficients; convergence in r is
exponential because the kernels are analytic.

Reported: max share error versus the exact lattice pass, and wall time,
across expansion sizes, at N = 1000 and N = 5000 (k = 2, GH order 15).

Run:  python experiments/exp20_separated_pass/run_separated.py
Output: results.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 21


def cheb_nodes(a, b, r):
    k = np.arange(r)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * r))


def bary_weights(nodes):
    w = np.ones(len(nodes))
    for j in range(len(nodes)):
        w[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    return w


def interp_matrix(nodes, w, q):
    """Barycentric Lagrange interpolation matrix (rows: queries)."""
    d = q[:, None] - nodes[None, :]
    exact = np.abs(d) < 1e-14
    d[exact] = 1.0
    M = w[None, :] / d
    M = M / M.sum(1, keepdims=True)
    M[exact.any(1)] = exact[exact.any(1)].astype(float)
    return M


def separated_shares(mu, V, D, F, W, rm, rs, points=501):
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    lo = M_all.min() - 8 * sd.max()
    hi = M_all.max() + 8 * sd.max()
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    mn = cheb_nodes(M_all.min(), M_all.max(), rm)
    sn = cheb_nodes(sd.min(), sd.max(), rs)
    wm, ws = bary_weights(mn), bary_weights(sn)
    Zc = (x[None, None, :] - mn[:, None, None]) / sn[None, :, None]
    logS_c = log_ndtr(-Zc).reshape(rm * rs, points)
    haz_c = np.exp(-0.5 * Zc**2
                   - np.log(sn[None, :, None] * np.sqrt(2 * np.pi))
                   - log_ndtr(-Zc)).reshape(rm * rs, points)
    Ts = interp_matrix(sn, ws, sd)
    p = np.zeros(n)
    for qi in range(len(W)):
        Tm = interp_matrix(mn, wm, M_all[qi])
        T = (Tm[:, :, None] * Ts[:, None, :]).reshape(n, rm * rs)
        field = T.sum(0) @ logS_c
        wts = np.exp(np.clip(field, -745, 0)) * dx
        p += W[qi] * (T @ (haz_c @ wts))
    return p / p.sum()


def main():
    rng = np.random.default_rng(SEED)
    F, W = hermite_nodes(2)
    rows = ["N,rm,rs,r,max_err_vs_exact,seconds,exact_seconds"]
    for n in (1000, 5000):
        mu = rng.normal(0, 1.5, n)
        V = rng.normal(0, 0.5 / np.sqrt(2), (n, 2))
        D = rng.uniform(0.5, 1.5, n)
        t0 = time.perf_counter()
        p_exact = win_probabilities_factor(mu, V, D, F, W)
        t_exact = time.perf_counter() - t0
        print(f"N={n}: exact pass {t_exact:.2f}s")
        for rm, rs in ((16, 8), (32, 12), (48, 14), (64, 16)):
            t0 = time.perf_counter()
            p_sep = separated_shares(mu, V, D, F, W, rm, rs)
            t_sep = time.perf_counter() - t0
            err = np.abs(p_sep - p_exact).max()
            print(f"  r=({rm}x{rs}={rm*rs:4d}): err {err:.1e}, "
                  f"{t_sep:.2f}s ({t_exact/t_sep:.0f}x)")
            rows.append(f"{n},{rm},{rs},{rm*rs},{err:.3e},{t_sep:.3f},"
                        f"{t_exact:.3f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
