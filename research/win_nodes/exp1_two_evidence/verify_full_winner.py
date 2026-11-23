"""Referee for the composed row of Table 1: the exact price posterior
followed by the winner message against the full covariance
(update_market_full, then update_winner_full), checked against
rejection sampling of the generative model on one race drawn as in
run_two_evidence.py. The diagonal composition (update_race) is
reported alongside, so the check can fail for the right reason: it
separates the two families by far more than its own noise.

Reproduce: OMP_NUM_THREADS=1 python research/win_nodes/exp1_two_evidence/verify_full_winner.py
Measured (2026-11-19, 4M draws): contrast means within 0.0046 of the
referee (Monte Carlo se 0.0029), covariance within 0.0053; the
diagonal composition is 0.178 off in the contrast means.
"""
import numpy as np

from winning.factor.races import race_probabilities
from winning.ratings.full import update_market_full, update_winner_full
from winning.ratings.market import update_race

N, TAU2, BETA2, DRAWS, WINNER = 8, 0.25, 1.0, 4_000_000, 3


def main():
    rng = np.random.default_rng(1)
    s = rng.normal(size=N)
    V = rng.normal(scale=0.6, size=(N, 1))
    D = np.full(N, BETA2)
    eta = rng.normal(scale=np.sqrt(TAU2), size=N)
    p = np.asarray(race_probabilities(-(s + eta), V=V, D=D))

    mp, Sp, _ = update_market_full(np.zeros(N), np.eye(N), p, tau2=TAU2, V=V, D=D)
    mb, Sb, _ = update_winner_full(mp, Sp, WINNER, V=V, beta2=BETA2)
    md, _ = update_race(np.zeros(N), np.ones(N), winner=WINNER, p_market=p,
                        tau2=TAU2, V=V, beta2=BETA2)[:2]

    r = np.random.default_rng(7)
    th = mp + r.normal(size=(DRAWS, N)) @ np.linalg.cholesky(Sp).T
    perf = th + r.normal(size=(DRAWS, 1)) @ V.T + np.sqrt(BETA2) * r.normal(size=(DRAWS, N))
    t = th[perf.argmax(1) == WINNER]
    P = np.eye(N) - np.ones((N, N)) / N
    mc_m, mc_S = t.mean(0), np.cov(t.T)
    se = float((t.std(0) / np.sqrt(len(t))).max())

    err_full = float(np.abs(P @ (mb - mc_m)).max())
    err_cov = float(np.abs(Sb - mc_S).max())
    err_diag = float(np.abs(P @ (md - mc_m)).max())
    print(f"accepted {len(t)} of {DRAWS}; Monte Carlo se of a mean {se:.4f}")
    print(f"full-covariance composition: contrast means {err_full:.4f}, covariance {err_cov:.4f}")
    print(f"diagonal composition:        contrast means {err_diag:.4f}")
    assert err_full < 3 * se, "full-covariance winner update disagrees with the referee"
    assert err_diag > 10 * se, "referee cannot tell the two compositions apart"


if __name__ == "__main__":
    main()
