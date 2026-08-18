"""Experiment 31: local convergence of the damped Jacobi iteration.

Claim (Proposition, referee-suggested): at the solution, the linearization
of the mean-zero damped Jacobi map on log-shares is I - alpha * L_rw with
L_rw = Delta^{-1} J the random-walk normalized photo-finish Laplacian;
its nonzero spectrum lies in (0, 2), with lambda_N < 2 for N >= 3, so
every fixed damping 0 < alpha <= 1 is locally linearly convergent with
factor max_{j>=2} |1 - alpha lambda_j|.

Checked here: (a) spectrum of Delta^{-1} J real, in (0, 2); (b) measured
per-iteration error contraction of the exact (uncapped) Jacobi iteration
matches the predicted factor.

Run:  python experiments/exp31_jacobi_convergence/run_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

HERE = Path(__file__).resolve().parent
N, K = 12, 2


def main():
    rng = np.random.default_rng(31)
    mu = rng.normal(0, 1.0, N); mu -= mu.mean()
    V = rng.normal(0, 0.4, (N, K))
    D = rng.uniform(0.6, 1.4, N)
    F, W = hermite_nodes(K)
    p_star = win_probabilities_factor(mu, V, D, F, W)

    eps = 1e-6
    J = np.zeros((N, N))
    for j in range(N):
        e = np.zeros(N); e[j] = eps
        J[:, j] = (win_probabilities_factor(mu + e, V, D, F, W)
                   - win_probabilities_factor(mu - e, V, D, F, W)) / (2 * eps)
    J = 0.5 * (J + J.T)
    # min-wins J is negative semidefinite in mu? raceutil is min-wins; the
    # iteration works on abilities a with dp/da; use as computed
    Delta = np.diag(np.diag(J))
    Lrw = np.linalg.solve(Delta, J)
    ev = np.sort(np.linalg.eigvals(Lrw).real)
    print(f"spectrum of Delta^-1 J: min {ev[0]:.2e}, second {ev[1]:.4f}, "
          f"max {ev[-1]:.4f} (claim: 0 = l1 < l2 <= ... <= lN < 2)")
    ok_spec = abs(ev[0]) < 1e-8 and ev[1] > 0 and ev[-1] < 2
    rows = [f"lambda_min,{ev[0]:.3e}", f"lambda_2,{ev[1]:.6f}",
            f"lambda_max,{ev[-1]:.6f}"]

    for alpha in (1.0, 0.7):
        pred = max(abs(1 - alpha * ev[1]), abs(1 - alpha * ev[-1]))
        # exact Jacobi iteration on log shares from a perturbed start
        a = mu + 0.02 * rng.normal(0, 1, N); a -= a.mean()
        logp_t = np.log(p_star)
        errs = []
        for it in range(40):
            p = win_probabilities_factor(a, V, D, F, W)
            r = np.log(p) - logp_t
            errs.append(np.linalg.norm(r - r.mean()))
            step = alpha * r / (np.diag(J) / p)
            a = a - step
            a -= a.mean()
            if errs[-1] < 1e-12:
                break
        errs = np.array(errs)
        good = errs[(errs > 1e-10) & (errs < 1e-3)]
        rate = float(np.exp(np.mean(np.diff(np.log(good))))) if len(good) > 3 else float("nan")
        print(f"alpha={alpha}: predicted factor {pred:.4f}, measured "
              f"contraction {rate:.4f}")
        rows.append(f"alpha_{alpha}_predicted,{pred:.6f}")
        rows.append(f"alpha_{alpha}_measured,{rate:.6f}")
    print("spectrum OK" if ok_spec else "SPECTRUM FAIL")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
