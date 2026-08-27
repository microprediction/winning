"""A correlated-probit listwise preference likelihood, with its exact gradient.

The reward-modelling standard is Bradley-Terry / Plackett-Luce: the chosen
response is the argmax of score plus INDEPENDENT Gumbel noise, which is what
makes the likelihood a softmax. Independence is also what makes it wrong when
several candidate responses to the same prompt are near-duplicates: their
latent qualities are correlated, the duplicates split votes, and the learned
reward acquires the perverse incentive documented by Xu, Dong, Lu, Lam, Wen &
Van Roy (arXiv 2312.01057). Cherapanamjeri, Daskalakis, Farina & Mohammadpour
(arXiv 2510.15839) name the correlated probit as the repair and prove the
correlation is identifiable from lists of three or more; what they do not
provide is the estimator. This module is the estimator.

Model per prompt: utilities U = mu + V z + sqrt(D) eps, z ~ N(0, I_r),
eps iid N(0,1); the annotator reports argmax U. The log-likelihood of winner
data is sum over observations of log p_c(mu, V, D), with p from the factor
probit forward map, and the gradient in mu comes from one extra lattice pass:

    d p_i / d mu_j = - E_z int f_i f_j prod_{k != i,j} F_k dx     (j != i)
    d p_i / d mu_i = - sum_{j != i} d p_i / d mu_j                (rows sum to 0)

so grad_mu log p_i is the i-th row of the Jacobian over p_i, at the same
O(Q N L) cost as the forward probability. Every formula is anchored in
tests/test_likelihood.py against finite differences and against the softmax
closed form in the independent-Gumbel limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qpo"))

from pom import _window_adaptive, pom_fast, sobol_nodes  # noqa: E402

_LOG2PI = float(np.log(2.0 * np.pi))


def win_probability_and_grad(i, mu, V, d, nodes=None, weights=None,
                             points: int = 129, max_elements: float = 6e6):
    """p_i and its exact gradient in mu, in one pass.

    Returns (p_i, g) with g_j = d p_i / d mu_j. The row sums to zero: raising
    everyone's quality equally cannot change who wins.
    """
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    N = mu.size
    sd = np.sqrt(d)
    if V is None or (np.ndim(V) == 2 and np.asarray(V).shape[1] == 0):
        M_all = mu[None, :].copy()
        weights = np.ones(1)
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        M_all = mu[None, :] + np.atleast_2d(nodes) @ V.T
        weights = np.asarray(weights, dtype=float)

    lo_all, hi_all = _window_adaptive(M_all, sd)
    grid = np.arange(points) / (points - 1)
    log_norm = np.log(sd) + 0.5 * _LOG2PI

    p_i = 0.0
    g = np.zeros(N)
    chunk = max(1, int(max_elements / (N * points)))
    for a0 in range(0, M_all.shape[0], chunk):
        M = M_all[a0:a0 + chunk]
        Wc = weights[a0:a0 + chunk]
        lo = lo_all[a0:a0 + chunk]
        hi = hi_all[a0:a0 + chunk]
        x = lo[:, None] + (hi - lo)[:, None] * grid[None, :]
        dx = (hi - lo) / (points - 1)
        z = (x[:, None, :] - M[:, :, None]) / sd[None, :, None]     # (C,N,L)
        log_cdf = log_ndtr(z)
        log_pdf = -0.5 * z * z - log_norm[None, :, None]
        log_field = log_cdf.sum(axis=1)                              # (C,L)
        # log of f_i * prod_{k != i} F_k, per lattice point
        log_core = log_pdf[:, i, :] + log_field - log_cdf[:, i, :]   # (C,L)
        p_i += float(np.sum((Wc * dx) * np.exp(np.clip(log_core, -745, 700)).sum(axis=1)))
        # d p_i / d mu_j = - E int f_i f_j prod_{k != i,j} F_k
        integ = np.exp(np.clip(log_core[:, None, :] + log_pdf - log_cdf,
                               -745.0, 700.0))                        # (C,N,L)
        contrib = -(integ.sum(axis=2) * dx[:, None])                  # (C,N)
        gj = Wc @ contrib
        gj[i] = 0.0
        g += gj
    g[i] = -g.sum()
    return p_i, g


def loglik_and_grad_sources(counts, source_of, theta, rho, m_dup,
                            nodes=None, weights=None, points: int = 129):
    """Log-likelihood of winner counts under source-level scores.

    counts[i] = number of prompts where response slot i won. source_of maps
    slot -> source index; theta are per-source scores. The duplicate block
    (all slots of the duplicated source) shares one factor with loading
    sqrt(rho); every slot keeps unit marginal variance.
    """
    theta = np.asarray(theta, dtype=float)
    K = len(source_of)
    mu = theta[np.asarray(source_of)]
    V, d = duplicate_structure(source_of, rho)
    if nodes is None:
        nodes, weights = sobol_nodes(1, m=10, seed=0)
    ll = 0.0
    gmu = np.zeros(K)
    for i in range(K):
        if counts[i] == 0:
            continue
        p, g = win_probability_and_grad(i, mu, V, d, nodes, weights,
                                        points=points)
        ll += counts[i] * np.log(max(p, 1e-300))
        gmu += counts[i] * g / max(p, 1e-300)
    gtheta = np.zeros(theta.size)
    np.add.at(gtheta, np.asarray(source_of), gmu)
    return ll, gtheta


def duplicate_structure(source_of, rho, dup_source=None):
    """V, d for 'one block of near-duplicates': the most common slots share a
    factor with loading sqrt(rho); everything keeps unit marginal variance."""
    src = np.asarray(source_of)
    if dup_source is None:
        vals, cnts = np.unique(src, return_counts=True)
        dup_source = int(vals[np.argmax(cnts)])
    K = len(src)
    V = np.zeros((K, 1))
    d = np.ones(K)
    mask = src == dup_source
    V[mask, 0] = np.sqrt(max(rho, 0.0))
    d[mask] = 1.0 - max(rho, 0.0)
    return V, d


def true_win_probabilities(source_of, theta_true, rho, points: int = 501,
                           m: int = 12):
    """Exact win probabilities under the true correlated model (dense nodes)."""
    from pom import hermite_nodes
    mu = np.asarray(theta_true, dtype=float)[np.asarray(source_of)]
    V, d = duplicate_structure(source_of, rho)
    F, W = hermite_nodes(1, Q=81)
    return pom_fast(mu, V, d, F, W, points=points)


def plackett_luce_gap(p_win, source_of, dup_source=None):
    """The score gap Plackett-Luce infers from winner frequencies.

    Winner-only PL is a multinomial logit, so the MLE matches frequencies
    exactly: s_A - s_B = log(m p_A / (1 - p_A)) for one A against m duplicate
    B slots. Positive means PL awards the lone response the HIGHER reward --
    the perverse incentive, as a sign.
    """
    src = np.asarray(source_of)
    if dup_source is None:
        vals, cnts = np.unique(src, return_counts=True)
        dup_source = int(vals[np.argmax(cnts)])
    p = np.asarray(p_win, dtype=float)
    p_lone = p[src != dup_source].sum()
    m = int((src == dup_source).sum())
    return float(np.log(m * p_lone / max(1.0 - p_lone, 1e-300)))
