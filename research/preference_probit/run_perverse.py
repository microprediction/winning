"""The perverse incentive, and its repair, as one figure.

Setup from Xu, Dong, Lu, Lam, Wen & Van Roy (arXiv 2312.01057): each prompt
draws K = 4 responses, one from source A and three near-duplicates from source
B. Source B is genuinely better, delta = q_B - q_A > 0. The duplicates share a
prompt-level factor with correlation rho, so they split wins that any one of
them would have taken alone.

Plackett-Luce fitted to winner data is a multinomial logit, so its MLE
reproduces the winner frequencies exactly, and its inferred score gap is a
closed form of the true win probability of the lone response:

    s_A - s_B = log( m p_A / (1 - p_A) ).

As rho rises, p_A rises past 1/(m+1) and the sign FLIPS: Plackett-Luce awards
the lone, worse response the higher reward. That is the perverse incentive as
a sign, with no estimation noise involved -- it is a property of the
likelihood, not of a fit.

The correlated-probit likelihood fitted to the same data recovers delta at
every rho, and (Cherapanamjeri, Daskalakis, Farina & Mohammadpour, arXiv
2510.15839, identifiability of correlation from lists of K >= 3) recovers rho
itself jointly, from choices alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "qpo"))

from likelihood import (loglik_and_grad_sources, plackett_luce_gap,  # noqa: E402
                        true_win_probabilities)
from pom import sobol_nodes  # noqa: E402


def fit_probit_gap(counts, src, rho, n, points=201, iters=400, lr=0.5):
    """MLE of the source scores at a given rho (rho known)."""
    F, W = sobol_nodes(1, m=9, seed=0)
    theta = np.zeros(2)
    for _ in range(iters):
        ll, g = loglik_and_grad_sources(counts, src, theta, rho, 3, F, W,
                                        points=points)
        g = g - g.mean()
        theta = theta + lr * g / max(n, 1)
        theta -= theta.mean()
        if np.max(np.abs(g / max(n, 1))) < 1e-7:
            break
    return float(theta[1] - theta[0]), ll


def fit_probit_gap_and_rho(counts, src, n, points=201, grid=None):
    """Profile likelihood over rho: joint recovery of (delta, rho)."""
    if grid is None:
        grid = np.linspace(0.0, 0.95, 20)
    best = (-np.inf, np.nan, np.nan)
    for rho in grid:
        gap, ll = fit_probit_gap(counts, src, rho, n, points=points)
        if ll > best[0]:
            best = (ll, gap, rho)
    return best[1], best[2]


def main():
    src = [0, 1, 1, 1]
    delta = 0.3
    theta_true = np.array([0.0, delta])
    n = 10_000
    rng = np.random.default_rng(0)

    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        p = true_win_probabilities(src, theta_true, rho)
        pl_gap_inf = plackett_luce_gap(p, src)

        counts = rng.multinomial(n, p / p.sum())
        p_hat = counts / n
        pl_gap_n = plackett_luce_gap(p_hat, src)

        t0 = time.time()
        pr_gap_known, _ = fit_probit_gap(counts, src, rho, n)
        pr_gap_joint, rho_hat = fit_probit_gap_and_rho(counts, src, n)
        secs = time.time() - t0

        rows.append({
            "rho": rho, "delta_true": delta, "p_lone_true": float(p[0]),
            "PL_gap_infinite_data": pl_gap_inf,
            "PL_gap_n10000": pl_gap_n,
            "probit_gap_rho_known": pr_gap_known,
            "probit_gap_rho_fitted": pr_gap_joint,
            "rho_hat": rho_hat, "fit_seconds": secs,
            "PL_perverse": pl_gap_inf > 0,
        })
        print(f"rho={rho:4.2f}  p_A={p[0]:.3f} | PL gap (A minus B): "
              f"inf-data {pl_gap_inf:+.3f}  n=1e4 {pl_gap_n:+.3f} "
              f"{'PERVERSE' if pl_gap_inf > 0 else '        '} | "
              f"probit gap (B minus A): known-rho {pr_gap_known:+.3f} "
              f"fitted-rho {pr_gap_joint:+.3f} (rho_hat {rho_hat:.2f}) "
              f"[{secs:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    dest = HERE / "results" / "perverse_incentive.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    flip = df[df.PL_perverse].rho.min()
    print(f"\nPlackett-Luce flips to the perverse ranking at rho >= {flip:.2f}; "
          f"the correlated probit recovers delta = {delta} at every rho, "
          f"max abs error {np.abs(df.probit_gap_rho_fitted - delta).max():.3f} "
          f"with rho fitted from choices alone.")


if __name__ == "__main__":
    main()
