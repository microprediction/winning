"""Experiment 29: substitution replication with the marginal-variance
confound repaired.

The factorial's factor axis added ||v_i||^2 to each marginal variance,
unevenly across alternatives. Here every loading row is normalized to
||v_i||^2 = rho and the idiosyncratic variance is 1 - rho, so all
marginals are exactly one and the factor axis changes only the
off-diagonal covariance. Twenty seeds; targets from independent
5e6-draw simulations (no inverse crime).

For each seed: delete each resolvable alternative in turn; compare the
true redistribution of its released mass against (a) plain logit (IIA:
proportional reallocation) and (b) factor probit calibrated to the
undeleted shares with the true (V, D), using the removal ensemble.
Misallocation for deletion i = 0.5 * sum_j |qhat_ij - q_ij| / p_i.

Run:  python experiments/exp29_substitution_replication/run_replication.py
Output: results.csv (per seed x stratum medians), summary printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    win_probabilities_factor,
)

HERE = Path(__file__).resolve().parent
N, K, RHO, SEEDS, R = 150, 2, 0.36, 20, 5_000_000
STRATA = ((0.10, 1.01), (0.02, 0.10), (0.005, 0.02), (0.0005, 0.005))


def mc_shares(mu_max, V, D, R, rng):
    p = np.zeros(len(mu_max))
    chunk = max(1, int(2e7 / len(mu_max)))
    done = 0
    while done < R:
        m = min(chunk, R - done)
        f = rng.standard_normal((m, V.shape[1]))
        U = (mu_max[None, :] + f @ V.T
             + np.sqrt(D)[None, :] * rng.standard_normal((m, len(mu_max))))
        p += np.bincount(np.argmax(U, axis=1), minlength=len(mu_max))
        done += m
    return p / R


def main():
    F, W = hermite_nodes(K)
    rows = ["seed,stratum_lo,stratum_hi,n_deletions,med_logit,med_factor"]
    agg = {s: ([], []) for s in STRATA}
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        mu_max = rng.normal(0, 1.0, N)          # max-wins utilities
        Z = rng.normal(0, 1, (N, K))
        V = np.sqrt(RHO) * Z / np.linalg.norm(Z, axis=1, keepdims=True)
        D = np.full(N, 1.0 - RHO)
        target = mc_shares(mu_max, V, D, R, rng)
        target = np.maximum(target, 1e-9); target /= target.sum()
        # factor-probit candidate: calibrate to shares (min-wins: negate)
        a_hat = abilities_from_probabilities_factor(target, V, D, F, W)
        p_fit, q_fit = win_probabilities_factor(a_hat, V, D, F, W,
                                                return_deletions=True)
        # truth deletions from the true model (min-wins mu = -mu_max)
        _, q_true = win_probabilities_factor(-mu_max, V, D, F, W,
                                             return_deletions=True)
        for lo, hi in STRATA:
            eL, eF = [], []
            for i in range(N):
                if not (lo <= target[i] < hi):
                    continue
                mask = np.arange(N) != i
                # logit IIA prediction from observed shares
                qL = target[mask] / (1.0 - target[i])
                qT = q_true[i][mask] / q_true[i][mask].sum()
                qF = q_fit[i][mask] / q_fit[i][mask].sum()
                eL.append(0.5 * np.abs(qL - qT).sum() / target[i])
                eF.append(0.5 * np.abs(qF - qT).sum() / target[i])
            if eL:
                rows.append(f"{seed},{lo},{hi},{len(eL)},"
                            f"{np.median(eL):.4f},{np.median(eF):.4f}")
                agg[(lo, hi)][0].append(np.median(eL))
                agg[(lo, hi)][1].append(np.median(eF))
        print(f"seed {seed} done")
    print(f"\n{'stratum':>14} {'logit med [IQR]':>24} {'factor med [IQR]':>24} {'factor wins':>12}")
    for (lo, hi), (Ls, Fs) in agg.items():
        Ls, Fs = np.array(Ls), np.array(Fs)
        wins = (Fs < Ls).mean()
        print(f"[{lo:.4g},{hi:.4g}) "
              f"{np.median(Ls):.3f} [{np.percentile(Ls,25):.3f},{np.percentile(Ls,75):.3f}]"
              f"   {np.median(Fs):.3f} [{np.percentile(Fs,25):.3f},{np.percentile(Fs,75):.3f}]"
              f"   {wins:.2f}")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("wrote results.csv")


if __name__ == "__main__":
    main()
