"""Why a 40%-wrong covariance is 99.6%-right about the argmax.

The rank ladder showed something that the covariance error does not explain: a
rank-2 factor model whose off-diagonal residual is 40% of the off-diagonal
Frobenius norm reproduces essentially the whole qPO objective. Frobenius norm
is evidently the wrong measure of the residual. This module works out what the
right one is, and checks it numerically.

THE IDENTITY. For X ~ N(mu, Sigma) the Gaussian density satisfies a diffusion
equation in the covariance, which for any expectation p = E[g(X)] gives
Plackett's relation

    d p / d Sigma_ij  =  d^2 p / d mu_i d mu_j      (i != j),
    d p / d Sigma_ii  =  (1/2) d^2 p / d mu_i^2.

THE CONSEQUENCE. The factor construction sets D_r so that the model reproduces
diag(Sigma) exactly. So the residual Delta = Sigma - Sigma_r has ZERO DIAGONAL,
every diagonal term of the expansion drops, and to first order

    p_k(Sigma) - p_k(Sigma_r)  ~=  sum_{i<j} Delta_ij H^(k)_ij,
    H^(k) = Hessian of p_k in the means.

The error is the residual contracted against the Hessian of the win
probability, not its Frobenius norm. H^(k) is supported on pairs of candidates
that are jointly in contention to be the maximum -- for a pair that never
threatens the lead, moving their covariance moves nothing. Frobenius weights
all N(N-1)/2 pairs equally; the argmax weights the few that compete. That gap
is the whole explanation.

COMPUTING IT. Writing Delta = sum_m lambda_m u_m u_m' (its own eigen-
decomposition, lambda of both signs), and using Delta_ii = 0,

    sum_{i<j} Delta_ij H_ij = (1/2) sum_{ij} Delta_ij H_ij
                            = (1/2) sum_m lambda_m (u_m' H u_m)

and u' H^(k) u is the second directional derivative of p_k along u, which is
three evaluations of the forward map. So the prediction costs 3 x (number of
retained directions) forward passes and needs no N x N Hessian.

Everything here is checked against finite differences in test_theory.py before
it is used.
"""

from __future__ import annotations

import numpy as np

from pom import pom_fast


# --------------------------------------------------------------------------
# derivatives of the win probabilities
# --------------------------------------------------------------------------

def second_directional(mu, V, d, F, W, u, h=None, points=257, **kw):
    """D^2_u p = (p(mu + h u) - 2 p(mu) + p(mu - h u)) / h^2, componentwise."""
    mu = np.asarray(mu, dtype=float)
    u = np.asarray(u, dtype=float)
    if h is None:
        # scale the step to the field: h |u| a small fraction of a typical sd
        sd = float(np.sqrt(np.median(d + np.sum(np.atleast_2d(V) ** 2, axis=1))))
        h = 0.05 * sd / max(np.max(np.abs(u)), 1e-300)
    p0 = pom_fast(mu, V, d, F, W, points=points, **kw)
    pp = pom_fast(mu + h * u, V, d, F, W, points=points, **kw)
    pm = pom_fast(mu - h * u, V, d, F, W, points=points, **kw)
    return (pp - 2.0 * p0 + pm) / h ** 2, p0


def cov_perturbation_prediction(mu, V, d, F, W, Delta, n_dirs=64,
                                points=257, h=None, tol=1e-14, **kw):
    """First-order predicted change in p when Sigma moves by Delta.

    Delta must be symmetric; the diagonal need not vanish. The single
    expression (1/2) <Delta, H> covers both cases of Plackett's relation at
    once: an off-diagonal entry appears twice in the double sum and carries no
    half of its own, a diagonal entry appears once and carries the half.
    """
    Delta = np.asarray(Delta, dtype=float)
    lam, U = np.linalg.eigh(0.5 * (Delta + Delta.T))
    order = np.argsort(-np.abs(lam))[:n_dirs]
    lam, U = lam[order], U[:, order]
    keep = np.abs(lam) > tol * max(np.abs(lam).max(), 1e-300)
    lam, U = lam[keep], U[:, keep]

    N = len(mu)
    pred = np.zeros(N)
    p0 = None
    for m in range(len(lam)):
        D2, p0 = second_directional(mu, V, d, F, W, U[:, m], h=h,
                                    points=points, **kw)
        pred += 0.5 * lam[m] * D2
    return pred, p0, {"n_directions": int(len(lam)),
                      "captured": float(np.sum(lam ** 2) /
                                        max(np.sum(np.linalg.eigvalsh(Delta) ** 2),
                                            1e-300))}


# --------------------------------------------------------------------------
# where the sensitivity lives
# --------------------------------------------------------------------------

def contention_weights(mu, V, d, F, W, points=257, **kw):
    """A cheap proxy for where the Hessian mass sits.

    The exact Hessian is N x N per candidate. What the argument needs is only
    that its mass concentrates on pairs that jointly contend for the lead, and
    p_i p_j is the natural proxy: a pair contributes to the argmax decision in
    proportion to the chance that either of them is anywhere near winning.
    Returned as the vector p; the caller forms the outer product it needs.
    """
    return pom_fast(mu, V, d, F, W, points=points, **kw)


def weighted_residual_norms(Delta, p, Sigma=None) -> dict:
    """Compare the Frobenius view of a residual with the contention view.

    frobenius_offdiag   -- what the rank ladder reports, all pairs equal
    contention_weighted -- the same residual with pair (i,j) weighted by p_i p_j
    Both are normalised by the same quantity computed on Sigma itself, so the
    two ratios are directly comparable.
    """
    Delta = np.asarray(Delta, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(p)
    off = ~np.eye(n, dtype=bool)
    Wt = np.outer(p, p)
    out = {"frobenius_offdiag": float(np.sqrt(np.sum(Delta[off] ** 2))),
           "contention_weighted": float(np.sqrt(np.sum((Wt * Delta ** 2)[off])))}
    if Sigma is not None:
        Sigma = np.asarray(Sigma, dtype=float)
        out["frobenius_ratio"] = out["frobenius_offdiag"] / float(
            np.sqrt(np.sum(Sigma[off] ** 2)))
        out["contention_ratio"] = out["contention_weighted"] / float(
            np.sqrt(np.sum((Wt * Sigma ** 2)[off])))
    return out
