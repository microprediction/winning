"""Active tournament scheduling: does D-optimal field selection in the
photo-finish geometry identify abilities faster than random or
uncertainty-greedy scheduling?

Population of n_pop contestants, true abilities fixed; each round the
scheduler picks a field of size m, the race is simulated winner-only,
and the posterior advances by the exact N-way moment update. The
D-optimal scheduler greedily assembles the field maximizing the
marginal gain of logdet'(H + I_S) where I_S = sum_i J_i^T J_i / p_i is
the winner-observation Fisher information at the current posterior
means, embedded in population coordinates, and H accumulates selected
information on top of the prior precision.
"""
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from winning.factor.polish import race_jacobian  # noqa: E402
from winning.factor.races import race_probabilities  # noqa: E402
from winning.ratings import update_winner_correlated  # noqa: E402


def field_information(mu_hat, var, subset, beta2=1.0):
    """Winner-only Fisher information of racing `subset`, in the
    subset's coordinates."""
    m = mu_hat[subset]
    D = var[subset] + beta2
    p = race_probabilities(m, D=D)
    J = race_jacobian(m, D=D)
    return (J.T / p) @ J, p


def logdet_contrasts(M):
    """log pseudo-determinant on the contrast space (drop the smallest
    eigenvalue, which is the 1-direction zero up to numerics)."""
    w = np.linalg.eigvalsh((M + M.T) / 2)
    return float(np.log(np.maximum(w[1:], 1e-300)).sum())


def pick_d_optimal(mu_hat, var, H, m, rng, n_cand=40, beta2=1.0):
    """Greedy: seed with the pair of highest-variance contestants, then
    add the runner with the largest marginal logdet' gain, evaluating
    the exact field information at each step (fields are small, so the
    repricing is cheap)."""
    n = len(mu_hat)
    order = np.argsort(-var)
    S = list(order[:2])
    while len(S) < m:
        base_val = None
        best, best_val = None, -np.inf
        cands = [j for j in range(n) if j not in S]
        if len(cands) > n_cand:
            cands = list(rng.choice(cands, n_cand, replace=False))
        for j in cands:
            trial = S + [j]
            I, _ = field_information(mu_hat, var, np.array(trial),
                                     beta2=beta2)
            Hf = H.copy()
            ix = np.ix_(trial, trial)
            Hf[ix] += I
            val = logdet_contrasts(Hf)
            if val > best_val:
                best, best_val = j, val
        S.append(best)
    return np.array(S)


def run_full(seed, scheduler, n_pop=30, m=6, rounds=60, beta2=1.0):
    """Same tournament, FULL-covariance posterior: the estimator that
    can actually keep the off-diagonal information the D-optimal design
    buys. Design criterion uses the true posterior precision."""
    from winning.ratings.full import update_winner_full
    rng = np.random.default_rng(seed)
    mu_true = rng.normal(0, 1.0, n_pop)
    mu_true -= mu_true.mean()
    mean = np.zeros(n_pop)
    cov = np.eye(n_pop)
    rmse = []
    for t in range(rounds):
        var = np.diag(cov).copy()
        if scheduler == "random":
            S = rng.choice(n_pop, m, replace=False)
        elif scheduler == "uncertainty":
            S = np.argsort(-var)[:m]
        elif scheduler == "d_optimal":
            H = np.linalg.inv(cov + 1e-9 * np.eye(n_pop))
            S = pick_d_optimal(mean, var, H, m, rng, beta2=beta2)
        else:
            raise ValueError(scheduler)
        perf = mu_true[S] + np.sqrt(beta2) * rng.standard_normal(m)
        winner = int(np.argmin(perf))
        sub = np.ix_(S, S)
        m_s, S_s, _ = update_winner_full(-mean[S], cov[sub], winner,
                                         beta2=beta2)
        # condition the joint belief on the field's update (linear-
        # Gaussian conditioning of the untouched block on the raced one)
        K = cov[:, S] @ np.linalg.inv(cov[sub] + 1e-12 * np.eye(m))
        mean = mean + K @ ((-m_s) - mean[S])
        cov = cov + K @ (S_s - cov[sub]) @ K.T
        cov = (cov + cov.T) / 2
        centered = mean - mean.mean()
        rmse.append(float(np.sqrt(np.mean((centered - mu_true) ** 2))))
    return rmse


def run(seed, scheduler, n_pop=30, m=6, rounds=60, beta2=1.0):
    rng = np.random.default_rng(seed)
    mu_true = rng.normal(0, 1.0, n_pop)
    mu_true -= mu_true.mean()
    mean = np.zeros(n_pop)
    var = np.ones(n_pop)
    H = np.eye(n_pop)                     # prior precision
    V = np.zeros((n_pop, 1))
    rmse = []
    for t in range(rounds):
        if scheduler == "random":
            S = rng.choice(n_pop, m, replace=False)
        elif scheduler == "uncertainty":
            S = np.argsort(-var)[:m]
        elif scheduler == "d_optimal":
            S = pick_d_optimal(mean, var, H, m, rng, beta2=beta2)
        else:
            raise ValueError(scheduler)
        I, _ = field_information(mean, var, S, beta2=beta2)
        H[np.ix_(S, S)] += I
        perf = mu_true[S] + np.sqrt(beta2) * rng.standard_normal(m)
        winner = int(np.argmin(perf))
        m_s, v_s, _ = update_winner_correlated(-mean[S], var[S], winner,
                                               V[S], beta2=beta2)
        mean[S] = -m_s                    # module is max-wins
        var[S] = v_s
        centered = mean - mean.mean()
        rmse.append(float(np.sqrt(np.mean((centered - mu_true) ** 2))))
    return rmse


if __name__ == "__main__":
    import sys
    estimator = sys.argv[1] if len(sys.argv) > 1 else "diag"
    runner = run_full if estimator == "full" else run
    seeds = range(12)
    out = {}
    for scheduler in ("random", "uncertainty", "d_optimal"):
        t0 = time.time()
        curves = np.array([runner(s, scheduler) for s in seeds])
        wall = time.time() - t0
        med = np.median(curves, axis=0)
        out[scheduler] = dict(final=float(med[-1]),
                              half=float(med[len(med) // 2]),
                              wall=round(wall, 1),
                              curve=[round(float(v), 4) for v in med])
        print(f"{scheduler:12s} median RMSE at 30/60 races: "
              f"{med[len(med)//2]:.3f} / {med[-1]:.3f}   ({wall:.0f} s)")
    with open(os.path.join(os.path.dirname(__file__), f"results_{estimator}.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
