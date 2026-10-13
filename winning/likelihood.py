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

from .outcomes import as_luce_temperature, as_order
from .shapes import as_idio, as_loadings

from numpy.polynomial.hermite_e import hermegauss
from scipy.special import log_ndtr, ndtri


def _gh1(Q):
    x, w = hermegauss(Q)
    return x, w / w.sum()


# Factor orientation is arbitrary: V and V Q price the same law for any
# orthogonal Q, and in particular for a column sign flip or permutation.
# The Gauss-Hermite tensor is invariant under both, but a scrambled Sobol
# cloud is not, so the sharp-branch probability moved by 0.0047 when one
# loading column changed sign (#605). The cloud is therefore the orbit
# of base points under a group of signed permutations of the factor
# coordinates, at about the same node count.
#
# The group order is capped at 8, so at least n/8 base points (and as
# many distinct own-noise values) remain. The full group of 2^r r!
# (48 at r = 3, 384 at r = 4) left ceil(1024/384) = 3 base points: a
# three-point own-noise rule, raw masses summing to 1.216 and a 4.2-point
# error after normalisation (#694). Within the cap: all signed
# permutations at r <= 2, the 2^r sign flips at r = 3, no symmetrisation
# from r = 4 (there a sign group of 16 cost more accuracy than its
# invariance was worth). General rotations are not covered.
_MAX_SYMMETRY_ORDER = 8


def _factor_symmetries(r, n):
    """(perms, signs) of the symmetrising group for r factor columns at
    a budget of n nodes: the largest of {all signed permutations, sign
    flips, identity} whose order is at most min(_MAX_SYMMETRY_ORDER, n)."""
    import itertools
    if r == 0:
        return [()], [np.ones(0)]
    cap = min(_MAX_SYMMETRY_ORDER, n)
    full = (2 ** r) * int(np.prod(np.arange(1, r + 1)))
    if full <= cap:
        perms = list(itertools.permutations(range(r)))
    elif 2 ** r <= cap:
        perms = [tuple(range(r))]
    else:
        return [tuple(range(r))], [np.ones(r)]
    signs = [np.array(sg, dtype=float)
             for sg in itertools.product((1.0, -1.0), repeat=r)]
    return perms, signs


def symmetric_sobol_normal(r, extra, n, seed):
    """Standard-normal scrambled-Sobol cloud of about n points in
    r + extra dimensions whose first r (factor) coordinates are closed
    under the signed permutations of _factor_symmetries; the extra
    coordinates (own noise) ride along unchanged, so they take
    ceil(n / group order) >= n/8 distinct values. Equal weights."""
    import warnings
    from scipy.stats import qmc
    perms, signs = _factor_symmetries(r, n)
    order = len(perms) * len(signs)
    nb = int(np.ceil(n / order))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # balance needs n = 2^m; nb is not
        u = qmc.Sobol(r + extra, scramble=True, seed=seed).random(nb)
    X = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    out = []
    for pm in perms:
        Xp = X[:, list(pm)] if r else X[:, :0]
        for sg in signs:
            out.append(np.hstack([Xp * sg[None, :], X[:, r:]]))
    F = np.vstack(out)
    return F, np.full(len(F), 1.0 / len(F))


def nodes_for_likelihood(r, Qf=7, Qz=7, sharp=0.0):
    """(F, W): nodes over (factor^r, own-noise); Sobol past sharpness 3,
    symmetrised over factor signs and permutations (#605)."""
    if sharp > 3.0:
        return symmetric_sobol_normal(r, 1, 2 ** 10, seed=0)
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


def sharpness_bound(V, D=None, n=None):
    """The pairwise-safe sharpness of a loading matrix, gauge-fixed.

    Only loading DIFFERENCES decide a race, so the statistic must be
    invariant under V -> V + 1c'. Centering first and taking
    `sqrt(2) max_i ||(PV)_i|| / sqrt(min D)` bounds the pairwise
    sharpness `max_ij ||V_i - V_j|| / sqrt(D_i + D_j)` by the triangle
    inequality, and costs nodes rather than accuracy when it overshoots.

    This lives in one place because it was written twice and the copies
    drifted: prediction used `max_i ||V_i||` with no centering and no
    sqrt(2), so a fitted model could optimise under Sobol and then be
    priced on the Hermite tensor (#213). `n` is the field size; it is
    taken from `D` when that is given, and from `V` otherwise, so a
    caller that already holds the count should pass it.
    """
    if n is None:
        n = np.size(D) if D is not None and np.ndim(D) > 0 else np.shape(V)[0]
    V = as_loadings(V, n)          # the same door every other verb uses
    V = V - V.mean(axis=0)
    dmin = 1.0 if D is None else float(np.min(as_idio(D, n)))
    return float(np.sqrt(2.0) * np.max(np.sqrt((V ** 2).sum(axis=1)))
                 / np.sqrt(dmin))


def _log_mills(a):
    """log of phi(a)/Phi(a), the inverse Mills ratio, stable in the far
    left tail where ndtr(a) underflows to exactly zero."""
    return -0.5 * a * a - 0.5 * np.log(2.0 * np.pi) - log_ndtr(a)


# Below this the tilt's curvature term lam (a + lam) is taken from the
# erfcx continued fraction instead of exp(log phi - log Phi); see
# _mills_gap. Equal to scipy's own log_ndtr crossover.
_MILLS_TAIL = -20.0


def _mills_gap(a):
    """(lam, a + lam) for the inverse Mills ratio lam = phi(a)/Phi(a).

    lam (a + lam) lies in (0, 1) -- it is the tilt's curvature term -- but
    in the far left tail lam ~ -a and the sum cancels. Through
    exp(-a^2/2 - log(2 pi)/2 - log_ndtr(a)) lam carries the rounding of
    a^2/2 (2e-7 relative at a = -5.7e4), which the cancellation turns
    into a sum of either sign: the tilt curvature came out at -196 (julia,
    a DomainError in the Fishing fit) or -682 (python, a NaN), from the
    contrasts of an ordinary first BFGS step (#716). Below
    _MILLS_TAIL the continued fraction erfcx(x) = 1/(sqrt(pi) (x + cf)),
    x = -a/sqrt(2), gives both without cancellation: lam = -a + sqrt(2)
    cf and a + lam = sqrt(2) cf. Twelve terms are converged from x = 14.
    Above _MILLS_TAIL this is the old expression, to the bit."""
    a = np.asarray(a, dtype=float)
    with np.errstate(over="ignore"):
        lam = np.exp(-0.5 * a * a - 0.5 * np.log(2 * np.pi) - log_ndtr(a))
    gap = a + lam
    tail = a < _MILLS_TAIL
    if np.any(tail):
        x = -a[tail] / np.sqrt(2.0)
        cf = np.zeros_like(x)
        for k in range(12, 0, -1):
            cf = (0.5 * k) / (x + cf)
        g = np.sqrt(2.0) * cf
        lam[tail] = -a[tail] + g
        gap[tail] = g
    return lam, gap


# The node family switches from the Hermite tensor to scrambled Sobol at
# sharpness 3, and the two rules disagree there: on the binary #420
# fixture a 1e-10 change in a loading moved the returned log-likelihood
# by 0.0253 nats and the loading score from 0.123 to 0.181, so BFGS was
# optimizing a discontinuous objective with a within-branch gradient.
# Across this window the likelihood is the probability-weighted blend
# (1 - w) P_GH + w P_Sobol with a C1 smoothstep w(sharp), and the score
# carries the blend's own derivative. Outside it, each side is the
# single rule it always was (to summation roundoff).
_BLEND_LO, _BLEND_HI = 2.9, 3.1


def _blend_weight(sharp):
    """(w, dw/dsharp): C1 smoothstep from 0 at _BLEND_LO to 1 at _BLEND_HI."""
    t = (sharp - _BLEND_LO) / (_BLEND_HI - _BLEND_LO)
    if t <= 0.0:
        return 0.0, 0.0
    if t >= 1.0:
        return 1.0, 0.0
    return t * t * (3.0 - 2.0 * t), 6.0 * t * (1.0 - t) / (_BLEND_HI - _BLEND_LO)


# The blend's difficulty statistic must be differentiable and symmetric
# in the alternatives. sharpness_bound takes max_i ||(PV)_i||, which has
# a cusp wherever two rows tie, and the score took np.argmax's first tied
# row, so inside the window the gradient depended on alternative labels
# and was not the derivative (left/right differences 0.88/0.26 on the
# tied #420 fixture). A p-norm over the centred row norms is smooth,
# permutation symmetric, at least the max and at most J^(1/p) above it.
_SHARP_P = 32.0


def _smooth_sharpness(V, D):
    """sqrt(2) ||(||(PV)_i||)_i||_p / sqrt(min D), and its V-gradient."""
    C = V - V.mean(axis=0)
    n2 = (C ** 2).sum(axis=1)
    M = float(n2.max())
    scale = np.sqrt(2.0) / np.sqrt(float(np.min(D)))
    if not M > 0.0:
        return 0.0, np.zeros_like(V)
    u = n2 / M
    up = u ** (0.5 * _SHARP_P)
    Su = float(up.sum())
    sn = np.sqrt(M) * Su ** (1.0 / _SHARP_P)
    # d sn / d n2_i = 0.5 sn u_i^(p/2 - 1) / (M sum u^(p/2))
    dn2 = 0.5 * sn * (u ** (0.5 * _SHARP_P - 1.0)) / (M * Su)
    G = 2.0 * dn2[:, None] * C
    G = G - G.mean(axis=0)            # through the centring
    return scale * sn, scale * G


# The own-noise tilt (#128, #270, #352). The nodes sample the chosen
# alternative's own noise z where its PRIOR has mass, but when the
# chosen alternative is behind, the integrand
#
#     h(z) = phi(z) prod_j Phi((c_j + s_k z) / s_j)
#
# has its mass far to the right: at a utility gap of -8 the 7-node rule
# returned log P = -35.3 against the exact -28.9, and -295.8 against
# -146.6 at -20. On the Gauss-Hermite tensor each observation's own-noise
# nodes are therefore moved to the mode m of log h and scaled by the
# curvature there, z = m + sigma x, with the change of measure carried
# in the weights -- log sigma - z^2/2 + x^2/2 -- so the rule integrates
# the same h and is exact in the limit of many nodes. The mode is taken
# with the factors integrated out in the Gaussian approximation (each
# rival's noise widened to t_j^2 = s_j^2 + |v_k - v_j|^2), which keeps
# the tilt honest when loadings are large. log h is strictly concave
# (curvature <= -1), so the mode is unique.
#
# An earlier version solved for m by Newton to convergence and treated
# (m, sigma) as constants in the score, which is then the quadrature of
# the true gradient but not the gradient of the computed objective: an
# end-to-end fit went from 4 s to 160 s and stopped converging. Here m
# is a FIXED smooth map of the utilities and loadings -- four Newton
# steps on a Mills-ratio surrogate (lambda ~ (sqrt(a^2 + 4) - a) / 2),
# then two exact steps -- whose derivatives are carried alongside, and
# sigma comes from the curvature at the last exact point, so the
# analytic score is the exact gradient of the returned log-likelihood.
# Cost: two extra log-CDF evaluations per rival per observation, not
# per node.
#
# The scrambled-Sobol branch (sharp loadings) is NOT tilted: with paired
# (factor, own-noise) points the change-of-measure weight is unbounded
# in x and the low-discrepancy error grew tenfold on the #213 field.
_TILT_SURROGATE_STEPS, _TILT_EXACT_STEPS = 4, 2


def _tilt_mode(c0, t, sk):
    """Smooth approximate mode of g(z) = -z^2/2 + sum_j log Phi((c_j +
    sk z) / t_j), with derivatives.

    c0 (R, n): utility contrasts mu_k - mu_j; t (R, 1): effective rival
    noise sd. Returns m (n,), h (n,) = -g''(m_last) at the last exact
    point, and the total derivatives dm_c, dh_c (R, n) in c_j and
    dm_t, dh_t (R, n) in t_j."""
    R, n = c0.shape
    m = np.zeros(n)
    dm_c = np.zeros((R, n))
    dm_t = np.zeros((R, n))
    inv_t = 1.0 / t
    rho = sk * inv_t
    rho2, rho3 = rho * rho, rho * rho * rho
    for it in range(_TILT_SURROGATE_STEPS + _TILT_EXACT_STEPS):
        a = (c0 + sk * m[None, :]) * inv_t
        if it < _TILT_SURROGATE_STEPS:
            q = np.sqrt(a * a + 4.0)
            lam = 0.5 * (q - a)
            kap = 0.5 * (1.0 - a / q)            # -d lam / da
            kp = -2.0 / (q * q * q)              # d kap / da
        else:
            lam, gap = _mills_gap(a)             # gap = a + lam
            kap = lam * gap                      # -lam'
            kp = lam * (1.0 - kap) - kap * gap
        g1 = -m + (rho * lam).sum(axis=0)
        h = 1.0 + (rho2 * kap).sum(axis=0)
        S3 = (rho3 * kp).sum(axis=0)[None, :]
        # in c_j: da_l/dc_j = delta_lj / t_l + rho_l dm_j
        dg1_c = -(rho * kap) * inv_t - h[None, :] * dm_c
        dh_c = (rho2 * kp) * inv_t + S3 * dm_c
        # in t_j: da_l/dt_j = rho_l dm_j - delta_lj a_l / t_l, and
        # drho_l/dt_j = -delta_lj rho_l / t_l
        dg1_t = rho * (kap * a - lam) * inv_t - h[None, :] * dm_t
        dh_t = -rho2 * (2.0 * kap + kp * a) * inv_t + S3 * dm_t
        step = g1 / h
        sc = (g1 / (h * h))[None, :]
        dm_c = dm_c + dg1_c / h[None, :] - sc * dh_c
        dm_t = dm_t + dg1_t / h[None, :] - sc * dh_t
        m = m + step
    return m, h, dm_c, dh_c, dm_t, dh_t


def _choice_terms(mu, V, choice, s, F, W, tilt=True, score=True):
    """Per-observation log P(choice), d/dmu and d/dV on one node set.
    V is gauge-centred; F holds (factor^r, own-noise) node columns for
    the N(0, I) measure. With tilt, the own-noise column is moved per
    observation (see _tilt_mode). score=False returns (lp, None, None)
    and skips the score's node arrays; lp is the same to the bit."""
    T, J = mu.shape
    r = V.shape[1]
    Fq, xq = F[:, :r], F[:, r]
    Vf = Fq @ V.T                                  # (Q, J)
    logW = np.log(W)
    lp = np.zeros(T)
    dmu = np.zeros((T, J))
    dV = np.zeros((T, J, r))
    half_log2pi = 0.5 * np.log(2 * np.pi)
    for k in range(J):
        idx = np.where(choice == k)[0]
        if len(idx) == 0:
            continue
        Ti = len(idx)
        rv = np.array([j for j in range(J) if j != k])
        sj = s[rv][:, None]                                   # (R, 1)
        rho = s[k] / sj
        c0 = mu[idx, k][None, :] - mu[idx][:, rv].T           # (R, Ti)
        dk = V[k][None, :] - V[rv]                            # (R, r)
        if tilt:
            t = np.sqrt(sj[:, 0] ** 2 + (dk * dk).sum(axis=1))[:, None]
            m, h, dm_c, dh_c, dm_t, dh_t = _tilt_mode(c0, t, s[k])
            sig = 1.0 / np.sqrt(h)
            dsig_c = -0.5 * (sig / h)[None, :] * dh_c          # (R, Ti)
            dsig_t = -0.5 * (sig / h)[None, :] * dh_t
        else:
            m, sig = np.zeros(Ti), np.ones(Ti)
        z = m[:, None] + sig[:, None] * xq[None, :]           # (Ti, Q)
        sh = Vf[:, k][None, :] - Vf[:, rv].T                  # (R, Q)
        # A[j, t, q] = (c0_j + shift_j(f_q) + s_k z_tq) / s_j
        A = (c0[:, :, None] + sh[:, None, :] + s[k] * z[None, :, :]) \
            / sj[:, :, None]
        # log_ndtr, not log(max(ndtr, 1e-300)): the floor made every
        # deep tail the same -690.78 and the Mills ratio a zero score
        # (#270)
        logPhi = log_ndtr(A)
        acc = logPhi.sum(axis=0) + logW[None, :]
        if tilt:
            acc = acc + 0.5 * (xq * xq)[None, :] - 0.5 * z * z
        mx = acc.max(axis=1)
        pw = np.exp(acc - mx[:, None])
        rs = pw.sum(axis=1)
        lp[idx] = np.log(sig) + mx + np.log(np.maximum(rs, 1e-300))
        if not score:
            continue
        lam = np.exp(-0.5 * A * A - half_log2pi - logPhi)
        omega = pw / rs[:, None]
        wl = omega[None, :, :] * lam / sj[:, :, None]          # (R, Ti, Q)
        dc = wl.sum(axis=2)                                    # (R, Ti)
        H = wl @ Fq                                            # (R, Ti, r)
        if tilt:
            # the tilt's own derivative: d log P = dsig/sig
            # + E[g'(z) (dm + x dsig)], g' the true integrand's slope
            gp = -z + (rho[:, :, None] * lam).sum(axis=0)
            Eg = (omega * gp).sum(axis=1)[None, :]
            Egx = (omega * gp * xq[None, :]).sum(axis=1)[None, :]
            dc = dc + dsig_c / sig[None, :] + Eg * dm_c + Egx * dsig_c
            dt = dsig_t / sig[None, :] + Eg * dm_t + Egx * dsig_t
            # t_j = sqrt(s_j^2 + |v_k - v_j|^2): dt_j/dv_k = dk_j / t_j
            H = H + (dt / t)[:, :, None] * dk[:, None, :]
        dmu[idx, k] += dc.sum(axis=0)
        dmu[idx[None, :], rv[:, None]] -= dc
        dV[idx, k] += H.sum(axis=0)
        dV[idx[None, :], rv[:, None]] -= H
    if not score:
        return lp, None, None
    return lp, dmu, dV


def _choice_pair(mu, V, choice, D):
    """J = 2 in closed form: P(k) = Phi(c / s), c = mu_k - mu_j,
    s^2 = D_k + D_j + ||V_k - V_j||^2, with its analytic score. No
    factor or own-noise quadrature, so no tail floor (#128, #270), no
    dependence on an unused loading column (#345), no 1/s_j blow-up
    from a near-deterministic rival (#352) and no node-family switch
    (#420)."""
    T = mu.shape[0]
    r = V.shape[1]
    k = choice
    j = 1 - choice
    rows = np.arange(T)
    c = mu[rows, k] - mu[rows, j]
    d = V[k] - V[j]                                # (T, r)
    s2 = D[k] + D[j] + (d ** 2).sum(axis=1)
    s = np.sqrt(s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(s > 0, c / np.where(s > 0, s, 1.0),
                     np.where(c > 0, np.inf, np.where(c < 0, -np.inf, 0.0)))
    lp = log_ndtr(z)
    with np.errstate(over="ignore", invalid="ignore"):
        lam = np.where(np.isfinite(z),
                       np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - lp),
                       0.0)
    sp = np.where(s > 0, s, 1.0)
    gc = np.where(s > 0, lam / sp, 0.0)            # d lp / d c
    dmu = np.zeros((T, 2))
    dmu[rows, k] = gc
    dmu[rows, j] = -gc
    # d lp / d d = lam * c * (-1/s^2) * d / s
    gd = np.where(s > 0, -lam * c / sp ** 3, 0.0)[:, None] * d
    dV = np.zeros((T, 2, r))
    dV[rows, k] = gd
    dV[rows, j] = -gd
    return lp, dmu, dV


# Own-noise Gauss-Hermite order when the caller gives none. Seven nodes
# carry J <= 5; from J = 6 the chosen alternative's own-noise integrand is
# a product of J - 1 rival CDFs, sharper than seven nodes resolve. On 528
# ordinary fields (J 3-10, r 0-3, sharpness 0.3-6) against a tight scipy
# MVN reference, 11 nodes cut the worst weak-factor error at J = 10 from
# 5.3e-3 to 5.3e-4 nats for 1.0-1.6x the cost (#173). J < 6 is unchanged.
QZ_SMALL, QZ_LARGE, QZ_LARGE_FROM_J = 7, 11, 6


def default_qz(J):
    """The own-noise order used when Qz is None: 7 below J = 6, 11 from
    J = 6 (#173)."""
    return QZ_LARGE if J >= QZ_LARGE_FROM_J else QZ_SMALL


def _choice_logprob_terms(mu, V, D=None, choice=None, Qf=7, Qz=None,
                          score=True):
    """Per-observation log P(choice_t), with its score rows: the one
    probability map that fitting, reporting and prediction share.

    Returns (lp (T,), dmu (T, J), dV (T, J, r)) for the observed
    `choice`. This is the engine under choice_loglik_and_score and
    MNProbit.predict_proba, so both price a model the same way.
    score=False returns (lp, None, None), the same lp without the score.
    """
    mu = np.asarray(mu, dtype=float)
    T, J = mu.shape
    V = as_loadings(V, J)
    r = V.shape[1]
    choice = np.asarray(choice, dtype=int)
    if Qz is None:
        Qz = default_qz(J)
    if J == 2:
        # the closed form needs only D_k + D_j + ||V_k - V_j||^2 > 0, so a
        # deterministic alternative (D_j = 0) is legal here (#352)
        D = np.ones(J) if D is None else as_idio(D, J)          # #84
        lp, dmu, dV = _choice_pair(mu, V, choice, D)
        return (lp, dmu, dV) if score else (lp, None, None)
    # J >= 3: the own-noise integral conditions on each rival through
    # Phi(./sqrt(D_j)); at D_j = 0 that is 0/0 and returned NaN,
    # label-asymmetrically (#352). The central door refuses it (#84).
    D = np.ones(J) if D is None else as_idio(D, J, positive=True)
    s = np.sqrt(D)
    # gauge-fix and dispatch on the pairwise-safe bound (eighth review):
    # only loading DIFFERENCES decide a race, and
    # sqrt(2) max_i |(PV)_i|/sqrt(D_i) bounds the pairwise sharpness; the
    # smooth p-norm form of it, so the blend has no cusp at ties (#420)
    sharp, sharp_grad = _smooth_sharpness(V, D)
    Vc = V - V.mean(axis=0)
    # A loading column that is zero after centring changes no choice
    # probability, but it changed the Sobol dimension, so a zero-padded
    # V priced differently from the unpadded one (#345). Integrate over
    # the live columns only; the score of a null column is exactly zero
    # (its factor's posterior is its prior, which is centred).
    live = np.flatnonzero(np.any(Vc != 0.0, axis=0))
    if live.size < r:
        lp, dmu, dVl = _choice_logprob_terms(mu, V[:, live], D=D,
                                             choice=choice, Qf=Qf, Qz=Qz,
                                             score=score)
        if not score:
            return lp, None, None
        dV = np.zeros((T, J, r))
        dV[:, :, live] = dVl
        return lp, dmu, dV
    # A rival much less noisy than the chosen alternative makes the
    # own-noise integrand a step of width s_j / s_k, which the tilted
    # rule needs more nodes to resolve: at D = [0.1, 1, 1] seven nodes
    # left the probability 0.4% low (3.3% before the tilt, #352). The
    # own-noise order grows with that ratio, capped at 4x; D = 1 (every
    # MNProbit fit) is unaffected, and nearer-deterministic rivals are
    # outside the accuracy envelope.
    ratio = float(np.sqrt(np.max(D) / np.min(D)))
    if ratio > 1.0:
        Qz = int(min(4 * Qz, np.ceil(Qz * ratio)))
    w, dw = _blend_weight(sharp)
    if w == 0.0 or w == 1.0:
        F, Wt = nodes_for_likelihood(r, Qf, Qz, sharp)
        return _choice_terms(mu, Vc, choice, s, F, Wt, tilt=(w == 0.0),
                             score=score)
    # inside the blend window: both rules, mixed in probability
    Fg, Wg = nodes_for_likelihood(r, Qf, Qz, 0.0)
    Fs, Ws = nodes_for_likelihood(r, Qf, Qz, np.inf)
    lg, dmg, dVg = _choice_terms(mu, Vc, choice, s, Fg, Wg, score=score)
    ls, dms, dVs = _choice_terms(mu, Vc, choice, s, Fs, Ws, tilt=False,
                                 score=score)
    lp = np.logaddexp(np.log1p(-w) + lg, np.log(w) + ls)
    if not score:
        return lp, None, None
    a = np.exp(np.log1p(-w) + lg - lp)             # posterior rule weights
    b = np.exp(np.log(w) + ls - lp)
    dmu = a[:, None] * dmg + b[:, None] * dms
    dV = a[:, None, None] * dVg + b[:, None, None] * dVs
    # d lp / d V through w: (P_S - P_GH) / P * dw/dsharp * dsharp/dV
    gw = (np.exp(ls - lp) - np.exp(lg - lp)) * dw          # (T,)
    dV = dV + gw[:, None, None] * sharp_grad[None, :, :]
    return lp, dmu, dV


def check_choice(choice, T, J):
    """The label contract shared by every MNP boundary that consumes
    observed choices: exactly one finite integer label in 0..J-1 per
    observation. Returns the labels as an int array."""
    choice = np.asarray(choice)
    if choice.ndim != 1 or choice.shape[0] != T:
        raise ValueError(
            f"choice must have one entry per observation: got shape "
            f"{choice.shape} for {T} observations")
    if not np.issubdtype(choice.dtype, np.integer):
        if np.issubdtype(choice.dtype, np.floating) and np.isnan(choice).any():
            raise ValueError(
                f"choice has {int(np.isnan(choice).sum())} missing "
                "entries; drop those observations deliberately rather "
                "than letting them score zero")
        if not np.all(np.isfinite(choice)) or not np.all(
                choice == np.floor(choice)):
            raise ValueError("choice must be integer alternative indices")
        choice = choice.astype(int)
    bad = np.flatnonzero((choice < 0) | (choice >= J))
    if bad.size:
        raise ValueError(
            f"choice[{int(bad[0])}] = {int(choice[bad[0]])} is outside "
            f"0..{J - 1}; {bad.size} of {T} observations are. They would "
            "be dropped in silence, which raises the log-likelihood "
            "because there is less of it.")
    return choice.astype(int, copy=False)


def choice_loglik_and_score(mu, V, choice, D=None, Qf=7, Qz=None):
    """Log-likelihood of observed argmax choices, with analytic score.

    Parameters
    ----------
    mu : (T, J) per-observation utilities.
    V : (J, r) factor loadings.
    choice : (T,) chosen alternative indices in 0..J-1.
    D : (J,) idiosyncratic variances, default all ones.
    Qf : Gauss-Hermite order per factor dimension.
    Qz : own-noise Gauss-Hermite order; None (default) takes 7 below
        J = 6 and 11 from J = 6 (#173).

    Returns
    -------
    loglik : float
    dmu : (T, J) gradient of the log-likelihood in mu.
    dV : (J, r) gradient in the loadings.

    J = 2 is the Gaussian contrast in closed form. For J >= 3 the own
    noise and the factors are integrated on the Hermite tensor or, past
    sharpness 3, scrambled Sobol, blended continuously across the switch.
    Known limits at J >= 3: a large observed utility gap puts the mass
    outside the fixed own-noise nodes (#128, #270), and a near-
    deterministic rival makes the integrand a step (#352). The Hermite
    branch tilts the own noise toward that mass; the Sobol branch does
    not, so a rare choice in a sharp field is still understated there
    (mu = [-8, 0, 1], V = [0, 3, -3]: -42.58 against the exact -30.10,
    #270). A product of the Sobol factor cloud with the tilted own-noise
    rule reached -31.10 at 1.6-2.2x the cost of every sharp call, and a
    joint factor/own-noise tilt is not implemented.
    """
    mu = np.asarray(mu, dtype=float)
    T, J = mu.shape
    # Every observation must be accounted for. The loop below walks the
    # LEGAL labels and gathers the rows matching each, so a row whose
    # choice is outside 0..J-1 is never visited: it contributed nothing
    # to the log-likelihood and a zero row to the score, and the fit
    # silently optimised a SUBSET of the data while reporting it as the
    # whole. Five bad rows out of forty moved the log-likelihood from
    # -55.699 to -49.134 -- better, because there was less of it (#194).
    choice = check_choice(choice, T, J)
    lp, dmu, dV = _choice_logprob_terms(mu, V, D=D, choice=choice, Qf=Qf, Qz=Qz)
    return float(lp.sum()), dmu, dV.sum(axis=0)


# Where the Gauss-Hermite tensor stops being the right family for the
# SOFTMAX integrand and the low-discrepancy rule takes over. Measured
# over 220 random fields, worst relative error by sharpness band:
#
#   sharpness   GH Qf=7    Sobol 2^10
#   [0, 1)      3.9e-05    2.6e-04
#   [1, 2)      1.1e-03    2.4e-03
#   [2, 3)      4.7e-02    2.3e-03     <- Gauss-Hermite collapses here
#   [4, 6)      3.9e-01    1.8e-03
#   [6, 20)     1.15       8.5e-05
#
# so Gauss-Hermite is the better rule below 2 -- and 146 times cheaper,
# 7 nodes against 1024 -- and the wrong one above it. The MNP choice
# likelihood dispatches at 3.0, on a product of Gaussian CDFs; a
# softmax is a sharper integrand and turns over sooner, which is why
# these two numbers differ (#272).
_FACTOR_SOBOL_SHARP = 2.0


def _factor_nodes(r, Qf=7, nodes_log2=10, sharp=0.0):
    """(F, W) over the factor space alone (no own-noise dimension): the
    Gumbel members condition only on f, the winner integral being closed
    form. Gauss-Hermite tensor with Qf nodes per dimension at r <= 2;
    2**nodes_log2 scrambled Sobol nodes at r >= 3, where Qf is inert
    (it was silently ignored there until 2026-09-11: the count was
    hard-coded at 1024, so the parameter did nothing in exactly the
    regime a user would reach for it)."""
    if r > 2 or sharp > _FACTOR_SOBOL_SHARP:
        # symmetrised over factor signs and permutations, as the choice
        # likelihood's cloud (#605)
        return symmetric_sobol_normal(r, 0, 2 ** int(nodes_log2), seed=0)
    if r == 0:
        # the EMPTY PRODUCT, as hermite_nodes and choice_loglik_and_score
        # already treat it: rank 1 was special-cased here and rank 0 fell
        # into the rank-2 tensor (#309)
        return np.zeros((1, 0)), np.ones(1)
    xf, wf = _gh1(Qf)
    if r == 1:
        return xf[:, None], wf
    grids = np.meshgrid(*([xf] * r), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for c in range(r):
        W *= wf[np.searchsorted(xf, F[:, c])]
    return F, W / W.sum()


def ranking_loglik_and_score(mu, V, orders, temperature=1.0, Qf=7,
                             nodes_log2=10):
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
    T, J = mu.shape
    V = as_loadings(V, J)
    r = V.shape[1]
    # finite and positive, BEFORE the sharpness dispatch: -1 fitted the
    # reversed preferences (and took the Sobol branch at sharp=inf), 0
    # and NaN returned NaN, +-inf the uniform likelihood with a zero
    # score (#366)
    tau = as_luce_temperature(temperature)
    # one valid ranking per observation, validated before any numerical
    # work: extra rows were ignored, [0, 0] scored 1/6 with an invented
    # gradient, -1 meant alternative J-1 and 0.9 meant 0 (#390, #129)
    if len(orders) != T:
        raise ValueError(
            f"orders must hold one ranking per observation: got "
            f"{len(orders)} for {T} rows of mu")
    orders = [as_order(orders[t], J, name=f"orders[{t}]") for t in range(T)]
    # Dispatch on the loading sharpness, as the choice likelihood does.
    # This took a FIXED Qf=7 tensor whatever the loadings were: on a
    # rank-one field with a centred loading contrast of 4.8 the
    # probability came back 12.9% high and the loading score smaller by
    # a factor of 88 -- derivatives of the true mixed likelihood, not a
    # disagreement with its own rule, so no finite-difference test
    # could see it (#272). The loadings enter as V/tau, so that is what
    # the sharpness is measured on.
    sharp = sharpness_bound(V / tau, n=J)
    F, W = _factor_nodes(r, Qf=Qf, nodes_log2=nodes_log2, sharp=sharp)
    Q = len(F)
    shift = (F @ V.T)                       # (Q, J)
    loglik = 0.0
    dmu = np.zeros((T, J))
    dV = np.zeros((J, r))
    for t in range(T):
        order = orders[t]
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
