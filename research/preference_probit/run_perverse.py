"""The perverse incentive, its repair, and what it takes to identify rho.

Setting from Xu, Dong, Lu, Lam, Wen & Van Roy (arXiv 2312.01057): a prompt is
answered by one response from source A and m near-duplicates from source B. B
is genuinely better, delta = q_B - q_A > 0. The duplicates share a
prompt-level factor with correlation rho, so they split votes.

PART A. Plackett-Luce fitted to winner data is a multinomial logit, so its MLE
reproduces the winner frequencies exactly and its inferred per-slot score gap
is closed form:

    s_A - s_B = log( m p_A / (1 - p_A) ).

As rho rises p_A rises past 1/(m+1) and the SIGN FLIPS: Plackett-Luce awards
the lone, worse response the higher reward. No estimation noise is involved --
this is a property of the likelihood, not of a fit.

PART B. The correlated probit likelihood, with rho known, recovers delta at
every rho.

PART C. Identification. With ONE composition the duplicates are exchangeable,
so winner counts carry exactly one free number (p_A) against two parameters
(delta, rho): NOT IDENTIFIED, and a first version of this script duly returned
rho_hat = 0.95 whatever the truth. Varying the composition across prompts --
some with one A and three B, some two and two, some three and one -- gives one
free number per composition and identifies both. That is also what real
preference data looks like. The profile likelihood is reported for both cases
so the failure and the repair are visible side by side.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "qpo"))

from likelihood import win_probability_and_grad  # noqa: E402
from pom import hermite_nodes, pom_fast  # noqa: E402

# rank-1 factor: Gauss-Hermite is far better than Sobol here
F1, W1 = hermite_nodes(1, Q=41)
POINTS = 257


def build(n_a, m, delta, rho):
    """mu, V, d for n_a lone slots (quality 0) and m duplicates (quality delta).

    Duplicates share one factor with loading sqrt(rho); every slot keeps unit
    marginal variance so delta is on a fixed scale.
    """
    K = n_a + m
    mu = np.concatenate([np.zeros(n_a), np.full(m, delta)])
    V = np.zeros((K, 1))
    d = np.ones(K)
    V[n_a:, 0] = np.sqrt(max(rho, 0.0))
    d[n_a:] = 1.0 - max(rho, 0.0)
    return mu, V, d


def true_p(n_a, m, delta, rho):
    mu, V, d = build(n_a, m, delta, rho)
    return pom_fast(mu, V, d, F1, W1, points=POINTS)


def pl_gap(p, n_a, m):
    """Per-slot score gap that Plackett-Luce infers, s_A - s_B. Positive means
    the LONE response is awarded the higher reward: the perverse ranking."""
    p = np.asarray(p, dtype=float)
    pa = p[:n_a].sum() / n_a
    pb = p[n_a:].sum() / m
    return float(np.log(max(pa, 1e-300) / max(pb, 1e-300)))


def loglik(delta, rho, data):
    """Winner-only log-likelihood over a set of (n_a, m, counts) blocks."""
    ll = 0.0
    for n_a, m, counts in data:
        mu, V, d = build(n_a, m, delta, rho)
        for i, c in enumerate(counts):
            if c == 0:
                continue
            p, _ = win_probability_and_grad(i, mu, V, d, F1, W1, points=POINTS)
            ll += c * np.log(max(p, 1e-300))
    return ll


def fit_delta(rho, data, bracket=(-2.0, 2.0)):
    r = minimize_scalar(lambda dl: -loglik(dl, rho, data), bounds=bracket,
                        method="bounded", options={"xatol": 1e-4})
    return float(r.x), float(-r.fun)


def profile_rho(data, grid):
    out = []
    for rho in grid:
        dl, ll = fit_delta(rho, data)
        out.append((rho, dl, ll))
    return out


def main():
    delta_true = 0.3
    m = 3
    n = 20_000
    rng = np.random.default_rng(0)

    # ---- Part A and B -----------------------------------------------------
    rows = []
    print("PART A/B: Plackett-Luce sign flip, and probit recovery at known rho")
    for rho in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        p = true_p(1, m, delta_true, rho)
        gap_inf = pl_gap(p, 1, m)
        counts = rng.multinomial(n, p / p.sum())
        gap_n = pl_gap(counts / n, 1, m)
        t0 = time.time()
        dl_hat, _ = fit_delta(rho, [(1, m, counts)])
        secs = time.time() - t0
        rows.append({"rho": rho, "p_lone": float(p[0]),
                     "PL_gap_infinite_data": gap_inf, "PL_gap_n": gap_n,
                     "PL_perverse": gap_inf > 0,
                     "probit_delta_rho_known": dl_hat,
                     "delta_true": delta_true, "seconds": secs})
        print(f"  rho={rho:4.2f}  p_lone={p[0]:.3f} | PL gap (lone minus dup) "
              f"{gap_inf:+.3f} {'PERVERSE' if gap_inf > 0 else '        '} "
              f"| probit delta_hat {dl_hat:+.3f} (true {delta_true}) [{secs:.1f}s]",
              flush=True)
    pd.DataFrame(rows).to_csv(HERE / "results" / "perverse_incentive.csv",
                              index=False)

    # ---- Part C: identification ------------------------------------------
    rho_true = 0.8
    grid = np.linspace(0.0, 0.95, 20)
    print(f"\nPART C: identification of rho (true rho = {rho_true}, "
          f"delta = {delta_true})")

    p1 = true_p(1, m, delta_true, rho_true)
    single = [(1, m, rng.multinomial(3 * n, p1 / p1.sum()))]

    mixed = []
    for n_a, mm in [(1, 3), (2, 2), (3, 1)]:
        pc = true_p(n_a, mm, delta_true, rho_true)
        mixed.append((n_a, mm, rng.multinomial(n, pc / pc.sum())))

    prof_rows = []
    for name, data in (("single composition (1 vs 3)", single),
                       ("mixed compositions (1v3, 2v2, 3v1)", mixed)):
        prof = profile_rho(data, grid)
        lls = np.array([x[2] for x in prof])
        best = int(np.argmax(lls))
        span = float(lls.max() - lls.min())
        # curvature: how many log-likelihood units separate the peak from rho=0
        sep = float(lls[best] - lls[0])
        print(f"  {name:36s} rho_hat={prof[best][0]:.2f} "
              f"delta_hat={prof[best][1]:+.3f}  "
              f"loglik range over rho grid = {span:8.1f}  "
              f"peak minus rho=0: {sep:8.1f}")
        for rho, dl, ll in prof:
            prof_rows.append({"case": name, "rho": rho, "delta_hat": dl,
                              "loglik": ll, "rho_true": rho_true,
                              "delta_true": delta_true})
    pd.DataFrame(prof_rows).to_csv(HERE / "results" / "identification.csv",
                                   index=False)
    print("\nwrote results/perverse_incentive.csv and results/identification.csv")


if __name__ == "__main__":
    main()
