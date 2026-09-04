"""Feasibility-sampling probe of the quotient completion classes: hit
the constraint manifold with least_squares from many starts and record
the head-to-head values of actual solutions. Sampled ranges are lower
bounds on the true identified intervals.

Measured (2026-09-01, truth Var(X1-X4) = 4.4103, in-grammar truth):
  two cliques   grammar [0.20, 8.11]   unrestricted [0.46, 8.70]
  three cliques grammar [3.11, 5.54]   unrestricted [0.85, 7.13]
The rank-one grammar identifies nothing beyond PSD with two cliques
(the within-clique split between (v_i - v_j)^2 and d_i + d_j is free);
a third overlapping field cuts the grammar ambiguity ~2.6x but does
not collapse it to a point.
"""
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from scipy.optimize import least_squares

CLIQUES3 = [np.array([0, 1, 2]), np.array([2, 3, 4]),
            np.array([1, 3, 4])]
CLIQUES2 = CLIQUES3[:2]


def functionals(Sigma, cliques):
    out = []
    for S in cliques:
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                i, j = S[a], S[b]
                out.append(Sigma[i, i] + Sigma[j, j] - 2 * Sigma[i, j])
    return np.array(out)


def h2h(S):
    return S[0, 0] + S[3, 3] - 2 * S[0, 3]


if __name__ == "__main__":
    rng0 = np.random.default_rng(7)
    v_true = np.array([0.9, -0.3, 0.5, -0.7, 0.2])
    d_true = 0.5 + rng0.random(5)
    St = np.outer(v_true, v_true) + np.diag(d_true)
    truth = h2h(St)
    rng = np.random.default_rng(0)
    tril = np.tril_indices(5)
    Lt = np.linalg.cholesky(St + 1e-9 * np.eye(5))
    for label, cliques in (("two cliques", CLIQUES2),
                           ("three cliques", CLIQUES3)):
        obs = functionals(St, cliques)
        vals_g, vals_u = [], []
        for s in range(300):
            th0 = np.r_[v_true + rng.standard_normal(5),
                        np.sqrt(d_true) + 0.4 * rng.standard_normal(5)]

            def res_g(th):
                S = np.outer(th[:5], th[:5]) \
                    + np.diag(th[5:] ** 2 + 1e-6)
                return functionals(S, cliques) - obs

            r = least_squares(res_g, th0, method="trf", max_nfev=4000)
            if np.abs(r.fun).max() < 1e-9:
                S = np.outer(r.x[:5], r.x[:5]) \
                    + np.diag(r.x[5:] ** 2 + 1e-6)
                vals_g.append(h2h(S))
            th0u = Lt[tril] + rng.standard_normal(15)

            def res_u(th):
                L = np.zeros((5, 5))
                L[tril] = th
                return functionals(L @ L.T, cliques) - obs

            ru = least_squares(res_u, th0u, method="trf", max_nfev=4000)
            if np.abs(ru.fun).max() < 1e-9:
                L = np.zeros((5, 5))
                L[tril] = ru.x
                vals_u.append(h2h(L @ L.T))
        vg, vu = np.array(vals_g), np.array(vals_u)
        print(f"{label}: truth {truth:.4f}")
        print(f"  grammar:      {len(vg)} feasible, h2h in "
              f"[{vg.min():.4f}, {vg.max():.4f}]")
        print(f"  unrestricted: {len(vu)} feasible, h2h in "
              f"[{vu.min():.4f}, {vu.max():.4f}]")
