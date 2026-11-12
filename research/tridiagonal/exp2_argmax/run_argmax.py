"""P(X_t is the max) over a Gauss-Markov (AR(1)) chain -- the
order-statistic a Kalman filter does not compute -- by forward-
backward on the chain, validated against Monte Carlo.

Conditional on X_t = m, the Markov property makes the left segment
(X_1..X_{t-1}) and the right segment (X_{t+1}..X_n) independent, so
  P(argmax = t) = int phi(m) A_t(m) B_t(m) dm,
with A_t(m) = P(left < m | X_t=m) and B_t(m) = P(right < m | X_t=m).
Each is the total surviving mass of a constrained sub-density
propagated (t-1) resp. (n-t) steps below m. One constrained
propagation per peak level m records the survival at every length,
so all t are priced together, the sum-product forward-backward with
the order statistic on top. The work is done by chain_argmax.py, which
runs all L thresholds at once with L x L messages: O(n) dense L x L
products, so O(n L^3) for the whole vector -- linear in n at fixed L.
"""
import json
import os

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from chain_argmax import ar1_argmax

HERE = os.path.dirname(os.path.abspath(__file__))


def argmax_probs(phi, n, L=800):
    # corrected lattice: cell-integrated transitions, half-weight tie
    # cell, no renormalisation (see ../chain_argmax.py)
    return ar1_argmax(phi, n, L=L)

def mc_argmax(phi, n, m=400000, seed=0):
    rng = np.random.default_rng(seed)
    sd = np.sqrt(1.0 - phi ** 2)
    X = np.empty((m, n))
    X[:, 0] = rng.normal(size=m)
    for t in range(1, n):
        X[:, t] = phi * X[:, t - 1] + sd * rng.normal(size=m)
    am = X.argmax(1)
    return np.bincount(am, minlength=n) / m


if __name__ == "__main__":
    results = {}
    for phi in (0.0, 0.5, 0.9):
        n = 20
        ex = argmax_probs(phi, n)
        mc = mc_argmax(phi, n)
        err = np.abs(ex - mc).max()
        print(f"[phi={phi} n={n}] sum(exact)={ex.sum():.4f}  "
              f"max|exact-MC|={err:.4f}")
        print(f"  exact ends/mid: t0 {ex[0]:.4f} t_mid {ex[n//2]:.4f} "
              f"tlast {ex[-1]:.4f}  | MC {mc[0]:.4f}/{mc[n//2]:.4f}/"
              f"{mc[-1]:.4f}")
        results[f"phi{phi}"] = dict(exact=ex.tolist(), mc=mc.tolist(),
                                    max_err=float(err),
                                    mass=float(ex.sum()))
    print("iid (phi=0) argmax is uniform 1/n = 0.05 (recovered);\n"
          "  positive correlation pushes the max to the ENDPOINTS --\n"
          "  interior points are squeezed by correlated-high neighbors\n"
          "  on both sides. A stationary Kalman filter gives identical\n"
          "  marginals everywhere and cannot see this.\n"
          "  sum(exact) < 1 by the lattice error only.")
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results.json")
