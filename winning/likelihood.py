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
    # gauge-fix and dispatch on the pairwise-safe bound (eighth review):
    # only loading DIFFERENCES decide a race, and
    # sqrt(2) max_i |(PV)_i|/sqrt(D_i) bounds the pairwise sharpness
    V = V - V.mean(axis=0)
    sharp = float(np.sqrt(2.0) * np.max(np.sqrt((V ** 2).sum(axis=1)))
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


def _factor_nodes(r, Qf=7):
    """(F, W) over the factor space alone (no own-noise dimension): the
    Gumbel members condition only on f, the winner integral being closed
    form. GH tensor for r <= 2, Sobol beyond."""
    if r > 2:
        from scipy.stats import qmc
        n = 2 ** 10
        u = qmc.Sobol(r, scramble=True, seed=0).random(n)
        return ndtri(np.clip(u, 1e-12, 1 - 1e-12)), np.full(n, 1.0 / n)
    xf, wf = _gh1(Qf)
    if r == 1:
        return xf[:, None], wf
    grids = np.meshgrid(*([xf] * r), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for c in range(r):
        W *= wf[np.searchsorted(xf, F[:, c])]
    return F, W / W.sum()


def ranking_loglik_and_score(mu, V, orders, temperature=1.0, Qf=7):
    """Mixed Plackett--Luce log-likelihood of (partial) rankings, with
    analytic score in mu and V: the likelihood member the ranking-bias
    result demanded.

    Max-wins, like choice_loglik_and_score. Conditional on the factor
    the noise is uniform-scale Gumbel, each ranking factorizes exactly
    by Harville's stagewise formula (IIA holds conditionally), and the
    stage scores are the standard PL residuals e_w - softmax; the
    mixture score follows with posterior node weights. Scope: this is
    the GUMBEL-model member (mixed Plackett--Luce), exact when the
    conditional noise is Gumbel and the right likelihood for
    logit-family worlds; for the GAUSSIAN model consume rankings through
    winning.ratings.nway.order_loglik mixed over factor nodes instead
    (measured unbiased where a Gaussian-base stagewise shortcut
    inflated learned correlation threefold). A length-1 order is the
    mixed-logit choice likelihood.

    Parameters
    ----------
    mu : (T, J) per-observation utilities.
    V : (J, r) factor loadings.
    orders : sequence of T index arrays, best first; each may be a full
        ranking or any top-k prefix.
    temperature : Gumbel scale tau (softmax weights exp(mu/tau)).

    Returns
    -------
    loglik : float
    dmu : (T, J)
    dV : (J, r)
    """
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    T, J = mu.shape
    r = V.shape[1]
    tau = float(temperature)
    F, W = _factor_nodes(r, Qf=Qf)
    Q = len(F)
    shift = (F @ V.T)                       # (Q, J)
    loglik = 0.0
    dmu = np.zeros((T, J))
    dV = np.zeros((J, r))
    for t in range(T):
        order = np.asarray(orders[t], dtype=int)
        z = (mu[t][None, :] + shift) / tau  # (Q, J)
        logp_q = np.zeros(Q)
        grad_q = np.zeros((Q, J))
        alive = np.ones(J, dtype=bool)
        for w in order:
            zz = np.where(alive[None, :], z, -np.inf)
            m = zz.max(axis=1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(zz - m).sum(axis=1))
            logp_q += z[:, w] - lse
            sm = np.exp(zz - lse[:, None])
            sm[:, ~alive] = 0.0
            grad_q[:, w] += 1.0
            grad_q -= sm
            alive[w] = False
        mstar = logp_q.max()
        pw = W * np.exp(logp_q - mstar)
        tot = pw.sum()
        omega = pw / tot                     # posterior node weights
        loglik += mstar + np.log(tot)
        g_mu = (omega[:, None] * grad_q).sum(axis=0) / tau
        dmu[t] += g_mu
        dV += ((omega[:, None] * grad_q).T @ F) / tau
    return float(loglik), dmu, dV
