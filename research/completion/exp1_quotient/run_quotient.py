"""The choice-relevant quotient completion, made concrete.

Two historical fields, {1,2,3} and {3,4,5}. Runners 1 and 4 never
co-raced, but share opponent 3. Race outcomes identify, per field, the
centered within-field block P_S Sigma_SS P_S (Gaussian races see the
covariance only through it), i.e. all within-field difference
covariances -- six numbers total. The head-to-head variance
Var(X_1 - X_4) is a cross-field object: chaining through runner 3
needs Cov(X_1 - X_3, X_3 - X_4), which no single field ever observes.

Measured here: the exact range of Var(X_1 - X_4) over
(a) ALL positive semidefinite completions consistent with the six
    observed quotient functionals, and
(b) completions restricted to the rank-one-factor grammar,
against the in-grammar truth. The claim of note 4: (a) is an interval
a full season cannot shrink; the grammar is what shrinks it.
"""
import json
import os
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

import sys
_CONFIGS = {
    "two": [np.array([0, 1, 2]), np.array([2, 3, 4])],
    "three": [np.array([0, 1, 2]), np.array([2, 3, 4]),
              np.array([1, 3, 4])],
}
CLIQUES = _CONFIGS[sys.argv[1] if len(sys.argv) > 1 else "two"]


def quotient_functionals(Sigma):
    """Within-field difference covariances: the race-identified data."""
    out = []
    for S in CLIQUES:
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                i, j = S[a], S[b]
                out.append(Sigma[i, i] + Sigma[j, j] - 2 * Sigma[i, j])
    return np.array(out)          # 3 per clique: all pairwise diff vars


def head_to_head(Sigma):
    return Sigma[0, 0] + Sigma[3, 3] - 2 * Sigma[0, 3]


def extremize_unrestricted(target_fn, obs, sign, Sigma_feas, seeds=12):
    """min/max target over PSD Sigma = L L' matching the observed
    functionals. SLSQP over the Cholesky factor, seeded AT a feasible
    point (the truth) and its perturbations -- random starts cannot
    reliably satisfy nine equality constraints."""
    n = 5
    tril = np.tril_indices(n)

    def unpack(theta):
        L = np.zeros((n, n))
        L[tril] = theta
        return L @ L.T

    cons = [{"type": "eq",
             "fun": lambda th: quotient_functionals(unpack(th)) - obs}]
    best = None
    rng = np.random.default_rng(0)
    L_feas = np.linalg.cholesky(Sigma_feas + 1e-9 * np.eye(n))
    for s in range(seeds):
        th0 = L_feas[tril] + (0.0 if s == 0
                              else 0.4 * rng.standard_normal(len(tril[0])))
        r = minimize(lambda th: sign * head_to_head(unpack(th)), th0,
                     constraints=cons, method="SLSQP",
                     options=dict(maxiter=800, ftol=1e-12))
        cand = unpack(r.x)
        if np.abs(quotient_functionals(cand) - obs).max() < 1e-6:
            v = head_to_head(cand)
            if best is None or sign * v < sign * best:
                best = v
    return best


def extremize_grammar(obs, sign, seeds=8):
    """Same, over Sigma = v v' + diag(d), d >= 1e-6."""
    def unpack(theta):
        v, d = theta[:5], theta[5:] ** 2 + 1e-6
        return np.outer(v, v) + np.diag(d)

    cons = [{"type": "eq",
             "fun": lambda th: quotient_functionals(unpack(th)) - obs}]
    best = None
    rng = np.random.default_rng(1)
    for s in range(seeds):
        if s == 0:
            th0 = np.r_[V_TRUE, np.sqrt(np.maximum(D_TRUE - 1e-6, 1e-6))]
        else:
            th0 = np.r_[V_TRUE + 0.4 * rng.standard_normal(5),
                        np.sqrt(np.maximum(D_TRUE - 1e-6, 1e-6))
                        + 0.2 * rng.random(5)]
        r = minimize(lambda th: sign * head_to_head(unpack(th)), th0,
                     constraints=cons, method="SLSQP",
                     options=dict(maxiter=400, ftol=1e-12))
        if r.success and np.abs(quotient_functionals(unpack(r.x))
                                - obs).max() < 1e-6:
            v = head_to_head(unpack(r.x))
            if best is None or sign * v < sign * best:
                best = v
    return best


rng0 = np.random.default_rng(7)
V_TRUE = np.array([0.9, -0.3, 0.5, -0.7, 0.2])
D_TRUE = 0.5 + rng0.random(5)

if __name__ == "__main__":
    v_true, d_true = V_TRUE, D_TRUE
    Sigma_true = np.outer(v_true, v_true) + np.diag(d_true)
    obs = quotient_functionals(Sigma_true)
    truth = head_to_head(Sigma_true)

    lo_u = extremize_unrestricted(head_to_head, obs, +1, Sigma_true)
    hi_u = extremize_unrestricted(head_to_head, obs, -1, Sigma_true)
    lo_g = extremize_grammar(obs, +1)
    hi_g = extremize_grammar(obs, -1)

    print(f"truth: Var(X1 - X4) = {truth:.4f}")
    print(f"unrestricted PSD completions: [{lo_u:.4f}, {hi_u:.4f}] "
          f"(width {hi_u - lo_u:.4f})")
    print(f"rank-one grammar completions: [{lo_g:.4f}, {hi_g:.4f}] "
          f"(width {hi_g - lo_g:.4f})")
    out = dict(truth=truth, unrestricted=[lo_u, hi_u],
               grammar=[lo_g, hi_g])
    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results.json")
