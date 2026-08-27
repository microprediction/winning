"""Observation-level likelihood and analytic score for factor
multinomial probit: the estimation core the package lacked.

Model per observation t: utilities U_tj = mu_tj + (V f_t)_j + z_tj with
f_t ~ N(0, I_r), z_tj ~ N(0, D_j), choice = argmax_j U_tj. The
probability of the observed choice, conditional on the factor AND the
chosen alternative's own noise, is a product of univariate normal CDFs,
so the log-likelihood is a low-dimensional smooth integral shared by
every observation and vectorizes into a handful of ndtr calls (the
identity of research/experiments/exp24_factor_rqmc, where it serves as
the methodologically independent referee).

The score is analytic: with posterior node weights omega and Mills
ratios lambda = phi/Phi, d log p / d a = omega * lambda, and utility /
loading gradients follow by the chain rule. Validated against central
finite differences to 1e-8 on both node branches (see the R port
r/mlogitfast, from which this module is translated).

Sharpness rule, third appearance in this repository: past
max_j ||v_j|| / sqrt(min D) of 3 the factor integrand is a near-step,
Gauss-Hermite under-integrates at any order, and an optimizer will
EXPLOIT THE HOLES (observed: a runaway to ||w|| ~ 300 with a fake
20-nat likelihood gain). The evaluation escalates to scrambled Sobol.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import ndtr, ndtri


def _gh1(Q):
    x, w = hermegauss(Q)
    return x, w / w.sum()


def nodes_for_likelihood(r, Qf=7, Qz=7, sharp=0.0):
    """(F, W): nodes over (factor^r, own-noise); Sobol past sharpness 3."""
    if sharp > 3.0:
        from scipy.stats import qmc
        n = 2 ** 10
        u = qmc.Sobol(r + 1, scramble=True, seed=0).random(n)
        return ndtri(np.clip(u, 1e-12, 1 - 1e-12)), np.full(n, 1.0 / n)
    xf, wf = _gh1(Qf)
    xz, wz = _gh1(Qz)
    grids = np.meshgrid(*([xf] * r + [xz]), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for c in range(r):
        W *= wf[np.searchsorted(xf, F[:, c])]
    W *= wz[np.searchsorted(xz, F[:, r])]
    keep = W > 1e-10 * W.max()
    return F[keep], W[keep] / W[keep].sum()


def choice_loglik_and_score(mu, V, choice, D=None, Qf=7, Qz=7):
    """Log-likelihood of observed argmax choices, with analytic score.

    Parameters
    ----------
    mu : (T, J) per-observation utilities.
    V : (J, r) factor loadings.
    choice : (T,) chosen alternative indices in 0..J-1.
    D : (J,) idiosyncratic variances, default all ones.

    Returns
    -------
    loglik : float
    dmu : (T, J) gradient of the log-likelihood in mu.
    dV : (J, r) gradient in the loadings.
    """
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    T, J = mu.shape
    r = V.shape[1]
    D = np.ones(J) if D is None else np.asarray(D, dtype=float)
    s = np.sqrt(D)
    sharp = float(np.max(np.sqrt((V ** 2).sum(axis=1)))
                  / np.sqrt(D.min()))
    F, W = nodes_for_likelihood(r, Qf, Qz, sharp)
    Fq, zq = F[:, :r], F[:, r]
    Q = len(W)
    Vf = Fq @ V.T                                  # (Q, J)

    loglik = 0.0
    dmu = np.zeros((T, J))
    dV = np.zeros((J, r))
    choice = np.asarray(choice)
    for k in range(J):
        idx = np.where(choice == k)[0]
        if len(idx) == 0:
            continue
        rivals = [j for j in range(J) if j != k]
        dmu_k = mu[idx, k][:, None] - mu[idx][:, rivals]   # (Ti, J-1)
        acc = np.zeros((len(idx), Q))
        A = {}
        logPhi = {}
        for c, j in enumerate(rivals):
            shift = Vf[:, k] - Vf[:, j] + zq * s[k]
            A[j] = (dmu_k[:, c][:, None] + shift[None, :]) / s[j]
            logPhi[j] = np.log(np.maximum(ndtr(A[j]), 1e-300))
            acc += logPhi[j]
        m = acc.max(axis=1)
        pw = np.exp(acc - m[:, None]) * W[None, :]
        rs = pw.sum(axis=1)
        loglik += float((m + np.log(np.maximum(rs, 1e-300))).sum())
        omega = pw / rs[:, None]
        for j in rivals:
            lam = np.exp(-0.5 * A[j] ** 2 - 0.5 * np.log(2 * np.pi)
                         - logPhi[j])
            wl = omega * lam / s[j]
            g = wl.sum(axis=1)                     # (Ti,)
            dmu[idx, k] += g
            dmu[idx, j] -= g
            H = wl @ Fq                            # (Ti, r)
            hc = H.sum(axis=0)
            dV[k] += hc
            dV[j] -= hc
    return loglik, dmu, dV
