"""Top-k membership probabilities: q_i = P(X_i among the k smallest).

The winning event is k = 1. For general k,

    q_i = int f_i(x) P( N_{-i}(x) <= k-1 ) dx,

with N_{-i}(x) the number of OTHER runners below x -- Poisson-binomial
over the per-runner probabilities F_j(x) = 1 - S_j(x) the survival
field already evaluates. The cavity move: build the full-field count
distribution C(x, m) = P(N(x) = m) once per lattice point (one shared
dynamic program, and simultaneously the order-statistic cache, since
P(X_(k) <= x) = P(N(x) >= k)), then remove each runner by DECONVOLUTION
of its Bernoulli factor instead of n separate leave-one-out programs:

    C = Q_i (*) Bernoulli(F_i)  =>
    forward   Q_m = (C_m - F_i Q_{m-1}) / S_i        from m = 0 up,
    backward  Q_m = (C_{m+1} - S_i Q_{m+1}) / F_i    from m = n-1 down.

Either direction alone is unstable (error amplification (F_i/S_i)^steps
forward, its reciprocal backward). The cure is to choose per runner per
lattice point: forward where S_i >= F_i, backward where F_i > S_i, so
the division is always by the larger of the two and initial-condition
error DECAYS along the recursion. tests/test_topk.py measures the
deconvolution against direct leave-one-out programs.

Correlation enters as everywhere in this package: conditional on the
factor draw the race is independent, so the correlated q is the node
mixture of independent ones.

The identity sum_i q_i = k is exact (k slots, each filled), and is
enforced the way the hierarchical kernels enforce unit mass: a material
defect raises rather than being normalized away.

Inversion runs the other way at any depth: abilities_from_topk
calibrates mean-zero locations to a top-k curve (place and show
markets, not just win), and loc_scale_from_topk_pair calibrates the
two-parameter family (mu_i, sigma_i) jointly to TWO curves -- win plus
place identifies per-runner scale, on the exact stacked
(dq/dmu, dq/dsigma) Jacobians below. Cumulative targets only: the
exact-rank marginal is non-monotone in ability and its standalone
inverse is two-branched (see the inversion section).
"""
from __future__ import annotations

import numpy as np

from .races import BASES
from .blocks import TINY, roots_hermitenorm

try:
    import fastrace as _fastrace
    _RUST_OK = hasattr(_fastrace, "top_k")
    _HAVE_RUST = _RUST_OK and __import__("os").environ.get(
        "WINNING_PURE", "").strip() in ("", "0")
except ImportError:                                  # pragma: no cover
    _fastrace = None
    _RUST_OK = False
    _HAVE_RUST = False


def _count_window(mu, sd, k, base_rows, delta=1e-12, pad_sds=2.0):
    """Lattice window for the top-k integrand.

    Below the window at most delta of a single runner has finished
    (sum_j F_j <= delta bounds P(N >= 1)); above it the count exceeds k
    almost surely (Chernoff: mean count mu* with
    mu* - sqrt(2 mu* ln(1/delta)) >= k forces P(N <= k-1) <= delta).
    Both ends by bisection on the monotone mean count, bracketed by
    geometric expansion first, in the node-aware style of the
    hierarchical kernels."""
    smax = max(float(sd.max()), 1e-12)

    def mean_count(x):
        S, _, _ = base_rows((x - mu) / sd)
        return float((1.0 - S).sum())

    lo = float(mu.min()) - 9.0 * smax
    step = 9.0 * smax
    for _ in range(60):
        if mean_count(lo) <= delta:
            break
        lo -= step
        step *= 2.0
    # the Chernoff slack can exceed the saturating mean count n on
    # small fields (the mean count never reaches it, and the bracket
    # doubling runs to infinity); cap just below saturation -- and only
    # JUST below: at k = n-1 the membership factor 1 - prod F_j dies
    # only once every runner is below almost surely, and a cap of
    # n - 0.25 truncated a tail the slot renormalization then masked
    # (the scale-gauge Euler identity exposed it at 3e-7, flat in the
    # point count)
    target_hi = min(k + 2.0 * np.log(1.0 / delta)
                    + np.sqrt(2.0 * (k + 1) * np.log(1.0 / delta)),
                    len(mu) - 1e-4)
    hi = float(mu.max()) + 9.0 * smax
    step = 9.0 * smax
    for _ in range(60):
        if mean_count(hi) >= target_hi:
            break
        hi += step
        step *= 2.0
    a, b = lo, hi
    for _ in range(70):
        m = 0.5 * (a + b)
        if mean_count(m) < delta:
            a = m
        else:
            b = m
    xlo = a
    a, b = xlo, hi
    for _ in range(70):
        m = 0.5 * (a + b)
        if mean_count(m) < target_hi:
            a = m
        else:
            b = m
    return xlo - pad_sds * smax, b + pad_sds * smax


def _count_distribution(F):
    """Full-field Poisson-binomial: C[l, m] = P(exactly m of the n
    runners are below lattice point l). One shared dynamic program,
    O(n^2 L); every column of C also prices an order statistic, since
    P(X_(k) <= x_l) = sum_{m >= k} C[l, m]."""
    L, n = F.shape
    C = np.zeros((L, n + 1))
    C[:, 0] = 1.0
    for j in range(n):
        f = F[:, j:j + 1]
        C[:, 1:j + 2] = C[:, 1:j + 2] * (1.0 - f) + C[:, :j + 1] * f
        C[:, 0] *= (1.0 - f[:, 0])
    return C


def _leave_one_out_cdf(C, F, k, chunk=256):
    """P(N_{-i} <= k-1) for every runner at every lattice point, by
    stable-direction deconvolution of the shared count distribution.
    O(L max(k, n-k)) per runner, vectorized over runner chunks."""
    L, n = F.shape
    out = np.empty((n, L))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Fc = F[:, a:b].T                      # (c, L)
        Sc = 1.0 - Fc
        fwd = Sc >= Fc                        # stable-direction mask
        # forward: accumulate Q_0 .. Q_{k-1}
        Sc_safe = np.maximum(Sc, TINY)
        # each true Q_m is a probability, so clipping the recursion
        # state is exact where the direction is stable and stops the
        # discarded unstable branch from overflowing into warnings
        Q = np.clip(C[None, :, 0] / Sc_safe, 0.0, 1.0)
        acc_f = Q.copy()
        for m in range(1, k):
            Q = np.clip((C[None, :, m] - Fc * Q) / Sc_safe, 0.0, 1.0)
            acc_f += Q
        # backward: accumulate Q_{n-1} down to Q_k, report 1 - tail
        Fc_safe = np.maximum(Fc, TINY)
        Qb = np.clip(C[None, :, n] / Fc_safe, 0.0, 1.0)
        acc_b = Qb.copy()
        for m in range(n - 2, k - 1, -1):
            Qb = np.clip((C[None, :, m + 1] - Sc * Qb) / Fc_safe, 0.0, 1.0)
            acc_b += Qb
        cdf = np.where(fwd, acc_f, 1.0 - acc_b)
        out[a:b] = np.clip(cdf, 0.0, 1.0)
    return out


def _topk_independent(mu, sd, k, base_rows, points, delta=1e-12,
                      is_normal=False):
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta)
    if is_normal and _HAVE_RUST:
        return np.asarray(_fastrace.top_k(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), k, lo, hi, points))
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    C = _count_distribution(F)
    cdf = _leave_one_out_cdf(C, F, k)         # (n, L)
    dens = (f / sd[None, :]).T                # (n, L)
    return (dens * cdf).sum(axis=1) * dx


def _checked_topk(raw, k, kind, mass_tol=5e-3):
    t = float(raw.sum())
    if not np.isfinite(t) or abs(t - k) > mass_tol * k:
        raise RuntimeError(
            f"{kind} captured total membership {t:.4f} where exactly "
            f"{k} slots exist (defect {abs(t-k):.2e}): the window or the "
            "deconvolution missed part of the field. Raise points=, or "
            "report this field.")
    return np.clip(raw * (k / t), 0.0, 1.0)


def top_k_probabilities(mu, k, V=None, D=None, base="normal", points=513,
                        qa=15):
    """P(X_i among the k smallest), for every i, min-wins.

    mu: locations; D: idiosyncratic variances; V: optional factor
    loadings (n, r) -- conditional on the factor draw the race is
    independent, and the result is the Gauss-Hermite mixture of the
    conditional memberships (rank one and two; higher ranks are
    refused, matching the hierarchical kernels' quadrature honesty).
    k = 1 is the win probability. The identity sum_i q_i = k is checked
    and a material defect raises."""
    mu = np.asarray(mu, float)
    n = len(mu)
    if not 1 <= int(k) <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    k = int(k)
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    is_normal = (base == "normal")
    if V is None:
        raw = _topk_independent(mu, sd, k, base_rows, points,
                                is_normal=is_normal)
        return _checked_topk(raw, k, "top-k race")

    Vm = np.asarray(V, float)
    if Vm.ndim == 1:
        Vm = Vm[:, None]
    r = Vm.shape[1]
    if r > 2:
        raise NotImplementedError(
            "top_k_probabilities mixes Gauss-Hermite factor nodes and is "
            "implemented for factor rank <= 2; higher rank needs the "
            "scrambled-Sobol escalation (issue #12).")
    Vm = Vm - Vm.mean(axis=0, keepdims=True)   # common column is gauge
    an, aw = roots_hermitenorm(qa)
    aw = aw / aw.sum()
    if r == 1:
        nodes = an[:, None]
        w = aw
    else:
        nodes = np.array([[a, b] for a in an for b in an])
        w = np.array([u * v for u in aw for v in aw])
        w = w / w.sum()
    raw = np.zeros(n)
    for q in range(len(nodes)):
        shift = Vm @ nodes[q]
        raw += w[q] * _topk_independent(mu + shift, sd, k, base_rows,
                                        points, is_normal=is_normal)
    return _checked_topk(raw, k, "top-k race")


def bottom_k_probabilities(mu, k, V=None, D=None, base="normal",
                           points=513, qa=15):
    """P(X_i among the k largest) -- the complement identity
    P(in the worst k) = 1 - P(in the best n-k), so no reflected base is
    ever needed."""
    n = len(np.asarray(mu))
    if not 1 <= int(k) <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    q = top_k_probabilities(mu, n - int(k), V=V, D=D, base=base,
                            points=points, qa=qa)
    return 1.0 - q


def _loo_pmf(C, F, i):
    """Full leave-one-out pmf Q^{-i}[l, m], m = 0..n-1, by one
    stable-direction deconvolution per lattice point: forward everywhere
    S_i >= F_i, backward everywhere else, so the recursion error decays
    along its whole length."""
    L, n = F.shape
    Fi = F[:, i]
    Si = 1.0 - Fi
    Q = np.empty((L, n))
    fwd = Si >= Fi
    # forward, full length
    if fwd.any():
        s = np.maximum(Si[fwd], TINY)[:, None]
        f = Fi[fwd][:, None]
        Cf = C[fwd]
        Qf = np.empty((fwd.sum(), n))
        Qf[:, 0] = np.clip(Cf[:, 0] / s[:, 0], 0.0, 1.0)
        for m in range(1, n):
            Qf[:, m] = np.clip((Cf[:, m] - f[:, 0] * Qf[:, m - 1])
                               / s[:, 0], 0.0, 1.0)
        Q[fwd] = Qf
    bwd = ~fwd
    if bwd.any():
        f = np.maximum(Fi[bwd], TINY)[:, None]
        s = Si[bwd][:, None]
        Cb = C[bwd]
        Qb = np.empty((bwd.sum(), n))
        Qb[:, n - 1] = np.clip(Cb[:, n] / f[:, 0], 0.0, 1.0)
        for m in range(n - 2, -1, -1):
            Qb[:, m] = np.clip((Cb[:, m + 1] - s[:, 0] * Qb[:, m + 1])
                               / f[:, 0], 0.0, 1.0)
        Q[bwd] = Qb
    return Q


def _pair_pmf_at(Qi, F, i, k, chunk=256):
    """P(N_{-ij} = k-1) for every j != i at every lattice point:
    deconvolve runner j's Bernoulli from the leave-i pmf Qi (L, n),
    stable direction per (j, x), forward k-1 steps or backward n-1-k
    steps since only one coefficient is needed."""
    L, n = F.shape
    out = np.zeros((n, L))
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        Fc = F[:, a:b].T                       # (c, L)
        Sc = 1.0 - Fc
        fwd = Sc >= Fc
        s_safe = np.maximum(Sc, TINY)
        Q = np.clip(Qi[None, :, 0] / s_safe, 0.0, 1.0)
        for m in range(1, k):
            Q = np.clip((Qi[None, :, m] - Fc * Q) / s_safe, 0.0, 1.0)
        f_safe = np.maximum(Fc, TINY)
        Qb = np.clip(Qi[None, :, n - 1] / f_safe, 0.0, 1.0)
        for m in range(n - 3, k - 2, -1):
            Qb = np.clip((Qi[None, :, m + 1] - Sc * Qb) / f_safe,
                         0.0, 1.0)
        out[a:b] = np.where(fwd, Q, Qb)
    out[i] = 0.0
    return out


def top_k_jacobian_row(mu, i, k, D=None, base="normal", points=513):
    """Row i of dq^{(k)}/dmu: the off-diagonals are the cutoff tie
    densities

        w_ij = int f_i(x) f_j(x) P(N_{-ij}(x) = k-1) dx >= 0,

    the same divergence-theorem flux as the win-probability Jacobian
    with the tie constrained to straddle the rank-k boundary (k = 1
    recovers it exactly), and the diagonal follows from translation
    invariance: a common shift of every location moves no membership,
    so the row sums to zero. Independent races only; min-wins."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    lo, hi = _count_window(mu, sd, k, base_rows)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    dens = f / sd[None, :]                     # (L, n)
    C = _count_distribution(F)
    Qi = _loo_pmf(C, F, i)
    pair = _pair_pmf_at(Qi, F, i, k)           # (n, L)
    row = (pair * dens.T * dens[:, i][None, :]).sum(axis=1) * dx
    row[i] = 0.0
    row[i] = -row.sum()
    return row


def top_k_jacobian(mu, k, D=None, base="normal", points=513):
    """The full (n, n) matrix dq^{(k)}/dmu, row by row: symmetric
    nonnegative off-diagonals, zero row sums -- minus a graph Laplacian
    on the rank-k boundary. O(n^2 L min(k, n-k)); intended for moderate
    n (scores, small fields, tests). Independent races only."""
    mu = np.asarray(mu, float)
    n = len(mu)
    J = np.empty((n, n))
    for i in range(n):
        J[i] = top_k_jacobian_row(mu, i, k, D=D, base=base, points=points)
    return J


def top_k_jacobian_row_sigma(mu, i, k, D=None, base="normal", points=513):
    """Row i of both derivatives at once: (dq_i/dmu, dq_i/dsigma).

    The sigma off-diagonals ride the mu computation for one extra
    weighted sum, because differentiating F_j((x - mu_j)/sigma_j) in
    sigma_j inserts the standardized coordinate into the same pair
    integrand:

        dq_i/dsigma_j = int z_j f_i f_j P(N_{-ij} = k-1) dx,  j != i,

    and the own term comes off the forward pass's membership factor,
    dq_i/dsigma_i = -int (z f'(z) + f(z))/sigma_i^2 P(N_{-i} <= k-1) dx.
    The scale gauge gives the exactness check: mu . row_mu +
    sigma . row_sigma = 0, since scaling every performance jointly
    moves no membership."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    lo, hi = _count_window(mu, sd, k, base_rows)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    dens = f / sd[None, :]                       # (L, n)
    C = _count_distribution(F)
    Qi = _loo_pmf(C, F, i)
    pair = _pair_pmf_at(Qi, F, i, k)             # (n, L)
    kern = pair * dens[:, i][None, :]
    row_mu = (kern * dens.T).sum(axis=1) * dx
    row_sd = (kern * (z * dens).T).sum(axis=1) * dx
    row_mu[i] = 0.0
    row_mu[i] = -row_mu.sum()
    cdf_i = Qi[:, :k].sum(axis=1)
    dfdsd = -(z[:, i] * fp[:, i] + f[:, i]) / D[i]
    row_sd[i] = (dfdsd * cdf_i).sum() * dx
    return row_mu, row_sd


def top_k_jacobians(mu, k, D=None, base="normal", points=513):
    """Full (n, n) matrices (dq/dmu, dq/dsigma). The lattice, the
    field rows and the shared count distribution are built ONCE and
    reused across rows -- the row helper rebuilds them per call, which
    at n = 150 spent more time on redundant count programs than on the
    pair terms themselves."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    lo, hi = _count_window(mu, sd, k, base_rows)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    dens = f / sd[None, :]
    zdens = (z * dens).T
    C = _count_distribution(F)
    Jm = np.empty((n, n))
    Js = np.empty((n, n))
    for i in range(n):
        Qi = _loo_pmf(C, F, i)
        pair = _pair_pmf_at(Qi, F, i, k)
        kern = pair * dens[:, i][None, :]
        row_mu = (kern * dens.T).sum(axis=1) * dx
        row_sd = (kern * zdens).sum(axis=1) * dx
        row_mu[i] = 0.0
        row_mu[i] = -row_mu.sum()
        cdf_i = Qi[:, :k].sum(axis=1)
        dfdsd = -(z[:, i] * fp[:, i] + f[:, i]) / D[i]
        row_sd[i] = (dfdsd * cdf_i).sum() * dx
        Jm[i] = row_mu
        Js[i] = row_sd
    return Jm, Js


# ---- Inversion: locations from top-k memberships ----
#
# The cumulative target is the invertible one. dq^{(k)}/dmu is minus a
# graph Laplacian on the rank-k boundary (top_k_jacobian_row), negative
# definite on the mean-zero quotient, so mu -> q^{(k)} is injective up
# to translation and the same damped own-slope iteration that inverts
# the win race inverts any k. The EXACT-rank marginal P(R_i = k) is
# refused as a standalone target on purpose: it is non-monotone in
# mu_i (a favorite and a plodder can share one exactly-2nd
# probability), so its inverse is two-branched per runner. Convert
# instead: exactly-2nd plus win is top-2.


def _topk_with_slopes(mu, sd, k, base_rows, points, delta=1e-12):
    """One numpy forward pass returning the raw memberships AND the own
    translation slopes

        dq_i/dmu_i = -int f'(z_i)/sd_i^2 P(N_{-i}(x) <= k-1) dx,

    the cavity cdf the forward pass already holds against the derivative
    of the runner's own density -- one extra weighted sum, no second
    field pass."""
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    C = _count_distribution(F)
    cdf = _leave_one_out_cdf(C, F, k)          # (n, L)
    dens = (f / sd[None, :]).T
    q = (dens * cdf).sum(axis=1) * dx
    slopes = -((fp / (sd ** 2)[None, :]).T * cdf).sum(axis=1) * dx
    return q, slopes


def _validated_topk_target(q, k, n, target_floor):
    """The abilities_from_race contract, restated for k slots: zeros and
    negatives raise (no finite inverse) unless deliberately floored;
    the slot identity sum q = k is imposed by proportional
    renormalization; and a membership at or above one AFTER that
    renormalization raises, because certainty of placing has no finite
    inverse either."""
    target = np.asarray(q, dtype=float)
    if len(target) != n:
        raise ValueError(f"target has {len(target)} entries for {n} runners")
    if target_floor is not None:
        if not target_floor > 0:
            raise ValueError("target_floor must be positive")
        floored = target < target_floor
        target = np.maximum(target, target_floor)
    else:
        floored = np.zeros(n, dtype=bool)
        if np.any(target <= 0):
            raise ValueError(
                "all target memberships must be positive: a zero top-k "
                "probability has no finite inverse (the supremum is "
                "approached as that runner's contrast diverges). Pass "
                "target_floor= to floor small entries deliberately, or "
                "supply a pseudocount upstream.")
    target = target * (k / target.sum())
    if np.any(target >= 1.0):
        raise ValueError(
            "after renormalizing to k slots, a target membership is >= 1: "
            "certain membership has no finite inverse, and a market vector "
            "this lopsided is outside the model's range (check the "
            "overround treatment and the dead-heat convention of the "
            "place quotes).")
    return target, floored


def _topk_inverse_return(mu, converged, resid_max, iters, floored, tol,
                         return_info, caller):
    if not converged and not return_info:
        import warnings
        warnings.warn(
            f"{caller} did not converge: max |logit residual| "
            f"{resid_max:.2e} after {iters} iterations (tol {tol:.0e}). "
            "The target may sit outside the model's feasible set (place "
            "markets carry overround and dead-heat conventions). Pass "
            "return_info=True for the diagnostics instead of this "
            "warning.", RuntimeWarning, stacklevel=3)
    if return_info:
        return mu, {"converged": bool(converged),
                    "max_logit_residual": float(resid_max),
                    "iterations": int(iters), "floored": floored}
    return mu


def abilities_from_topk(q, k, V=None, D=None, base="normal", points=513,
                        qa=15, n_iter=80, tol=1e-8, target_floor=None,
                        return_info=False):
    """Invert the top-k race: mean-zero mu with
    top_k_probabilities(mu, k) = q. k = 1 recovers abilities_from_race.

    Residuals live in LOGIT space rather than the win-inversion's log
    space: for k >= 2 the favorites saturate toward q = 1, where log
    residuals lose all sensitivity while both dq/dmu and q(1-q) vanish
    together and their ratio stays informative. Same target contract as
    abilities_from_race (zeros raise, target_floor= opts into flooring,
    non-convergence warns unless return_info=True), plus the slot
    identity: targets are renormalized to sum to k, and an entry >= 1
    after that renormalization raises. V= admits factor rank <= 2 by
    the usual node mixture. When k > n/2 the information lives in the
    longshots; inverting the complement (bottom_k_probabilities) is the
    same call at n - k on 1 - q."""
    mu_probe = np.asarray(q, dtype=float)
    n = len(mu_probe)
    k = int(k)
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    target, floored = _validated_topk_target(q, k, n, target_floor)
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    if V is not None:
        Vm = np.asarray(V, float)
        if Vm.ndim == 1:
            Vm = Vm[:, None]
        if Vm.shape[1] > 2:
            raise NotImplementedError(
                "abilities_from_topk mixes Gauss-Hermite factor nodes and "
                "is implemented for factor rank <= 2 (issue #12).")
        Vm = Vm - Vm.mean(axis=0, keepdims=True)
        an, aw = roots_hermitenorm(qa)
        aw = aw / aw.sum()
        if Vm.shape[1] == 1:
            nodes, w = an[:, None], aw
        else:
            nodes = np.array([[a, b] for a in an for b in an])
            w = np.array([u * v for u in aw for v in aw])
            w = w / w.sum()
    else:
        nodes, w, Vm = None, None, None

    logit_t = np.log(target) - np.log1p(-target)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    # N = 2 damping, inherited from the win inversion: K_2 is bipartite
    # and the undamped Jacobi update two-cycles on the quotient.
    alpha = 1.0 if n > 2 else 0.7
    resid_max = np.inf
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        if nodes is None:
            qraw, sl = _topk_with_slopes(mu, sd, k, base_rows, points)
        else:
            qraw = np.zeros(n)
            sl = np.zeros(n)
            for j in range(len(nodes)):
                qj, sj = _topk_with_slopes(mu + Vm @ nodes[j], sd, k,
                                           base_rows, points)
                qraw += w[j] * qj
                sl += w[j] * sj
        qhat = _checked_topk(qraw, k, "top-k inversion")
        resid = (np.log(np.maximum(qhat, 1e-300))
                 - np.log(np.maximum(1.0 - qhat, 1e-300))) - logit_t
        resid_max = float(np.abs(resid).max())
        if resid_max < tol:
            break
        dlogit = np.minimum(sl / np.maximum(qhat * (1.0 - qhat), 1e-300),
                            -1e-6)
        # residual-proportional step cap, as in the win inversion: no
        # coordinate moves much further than its own residual warrants.
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogit, -lim, lim)
        mu -= mu.mean()
    return _topk_inverse_return(mu, resid_max < tol, resid_max, iters,
                                floored, tol, return_info,
                                "abilities_from_topk")


def loc_scale_from_topk_pair(q1, k1, q2, k2, D0=None, base="normal",
                             points=513, n_iter=60, tol=1e-8, ridge=0.0,
                             mu0=None, return_info=False):
    """Joint (loc, scale) calibration from two membership curves: find
    per-runner (mu_i, sigma_i) with top_k_probabilities(mu, k1) = q1 and
    top_k_probabilities(mu, k2) = q2. The market pair is (win, place):
    k1 = 1, k2 = 2 or 3.

    The counting closes exactly. Two curves carry 2n numbers with two
    identities (sum q = k each); the unknowns carry two gauges,
    translation of mu and the joint rescaling (mu, sigma) -> (c mu,
    c sigma) -- the Euler identity mu . row_mu + sigma . row_sigma = 0
    that top_k_jacobian_row_sigma checks IS that second gauge. So the
    Newton system on the double quotient is square, and it runs on the
    exact stacked Jacobians (dq/dmu, dq/dsigma) from top_k_jacobians in
    (mu, log sigma) coordinates, Levenberg-damped, logit residuals as
    in abilities_from_topk. Gauge fix on return: mean-zero mu,
    geometric-mean-one sigma.

    A nearly uniform pair leaves sigma genuinely unidentified (any
    common scale prices the same); the Levenberg floor keeps the step
    finite there and the info dict reports the achieved residual.
    ridge= adds sqrt(ridge) * log sigma_i rows to the residual -- a
    deliberate prior toward a common scale for noisy market boards
    (after the gauge it penalizes only the DISPERSION of log sigma, and
    it biases the otherwise exactly identified solution, so it is off
    by default). Convention: ridge multiplies the SQUARED penalty, so a
    reference that scales the ridge RESIDUALS by w corresponds to
    ridge = w**2 here -- e.g. the T2 pipeline's w = 0.05 is
    ridge = 0.0025 (measured: ridge = 0.05 is a 20x stronger prior that
    biases clean-board parameters by ~0.03). mu0= supplies initial locations in physical units and
    skips the internal warm start.
    Avoid k = n/2 as a depth when the field is nearly level: with a
    symmetric base and equal locations, reflection symmetry makes
    half-field membership exactly scale-blind (q = 1/2 at any spread),
    so that curve carries no scale information at all. A
    market pair outside the family's range converges to its
    least-squares fit and reports converged=False. Independent races
    only, like the Jacobians it stacks. O(n^2 L min(k, n-k)) per
    iteration."""
    t1 = np.asarray(q1, dtype=float)
    n = len(t1)
    k1, k2 = int(k1), int(k2)
    if k1 == k2:
        raise ValueError(
            "k1 == k2 gives one curve twice: scale is unidentified "
            "without a second, different membership depth")
    for kk in (k1, k2):
        if not 1 <= kk <= n - 1:
            raise ValueError(f"k must be in [1, n-1]; got k={kk}, n={n}")
    target1, _ = _validated_topk_target(q1, k1, n, None)
    target2, _ = _validated_topk_target(q2, k2, n, None)
    lt1 = np.log(target1) - np.log1p(-target1)
    lt2 = np.log(target2) - np.log1p(-target2)

    sd = np.ones(n) if D0 is None else np.sqrt(np.asarray(D0, float))
    if mu0 is not None:
        mu = np.asarray(mu0, dtype=float) - np.mean(mu0)
    else:
        ka, ta = (k1, target1) if k1 < k2 else (k2, target2)
        mu, _info0 = abilities_from_topk(ta, ka, D=sd ** 2, base=base,
                                         points=points, return_info=True)
    sqr = float(np.sqrt(max(ridge, 0.0)))

    def logits(m, s):
        qh1 = top_k_probabilities(m, k1, D=s ** 2, base=base, points=points)
        qh2 = top_k_probabilities(m, k2, D=s ** 2, base=base, points=points)
        qh1 = np.clip(qh1, 1e-300, 1.0 - 1e-15)
        qh2 = np.clip(qh2, 1e-300, 1.0 - 1e-15)
        r = np.concatenate([np.log(qh1) - np.log1p(-qh1) - lt1,
                            np.log(qh2) - np.log1p(-qh2) - lt2,
                            sqr * np.log(s)])
        return r, qh1, qh2

    r, qh1, qh2 = logits(mu, sd)
    cost = float(r @ r)
    resid_max = float(np.abs(r[:2 * n]).max())
    lam = 1e-6
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        if resid_max < tol:
            break
        blocks = []
        for kk, qh in ((k1, qh1), (k2, qh2)):
            Jm, Js = top_k_jacobians(mu, kk, D=sd ** 2, base=base,
                                     points=points)
            g = 1.0 / np.maximum(qh * (1.0 - qh), 1e-300)
            # chain to (mu, log sigma) and to logit residuals
            blocks.append(np.hstack([Jm * g[:, None],
                                     (Js * sd[None, :]) * g[:, None]]))
        # ridge rows: d(sqr log sigma)/d(log sigma) = sqr, zero in mu
        blocks.append(np.hstack([np.zeros((n, n)), sqr * np.eye(n)]))
        J = np.vstack(blocks)
        JtJ = J.T @ J
        Jtr = J.T @ r
        accepted = False
        for _ in range(8):
            try:
                step = np.linalg.solve(JtJ + lam * np.eye(2 * n), -Jtr)
            except np.linalg.LinAlgError:
                lam *= 8.0
                continue
            mu_n = mu + step[:n]
            ls_n = np.clip(np.log(sd) + step[n:], -3.0, 3.0)
            # re-gauge exactly: ranks are invariant to (mu, sd) ->
            # (mu - a, sd)/c, so neither move changes the residual
            c = np.exp(ls_n.mean())
            sd_n = np.exp(ls_n - ls_n.mean())
            mu_n = (mu_n - mu_n.mean()) / c
            try:
                r_n, q1_n, q2_n = logits(mu_n, sd_n)
            except RuntimeError:
                lam *= 8.0
                continue
            cost_n = float(r_n @ r_n)
            if cost_n < cost:
                mu, sd, r, qh1, qh2, cost = mu_n, sd_n, r_n, q1_n, q2_n, cost_n
                resid_max = float(np.abs(r[:2 * n]).max())
                lam = max(lam / 3.0, 1e-10)
                accepted = True
                break
            lam *= 8.0
        if not accepted:
            break
    # with a ridge the penalized optimum generally keeps a nonzero fit
    # residual by design: an LM stall there is the answer, not a failure
    converged = resid_max < tol or (sqr > 0.0 and not accepted)
    out = _topk_inverse_return(mu, converged, resid_max, iters,
                               np.zeros(n, dtype=bool), tol, return_info,
                               "loc_scale_from_topk_pair")
    if return_info:
        return out[0], sd, out[1]
    return out, sd


def loc_scale_from_win_and_second(p_win, p_second, D0=None, base="normal",
                                  points=513, n_iter=60, tol=1e-8,
                                  ridge=0.0, mu0=None, return_info=False):
    """The two-marginal transform stated in market terms: win
    probabilities plus EXACTLY-SECOND probabilities, jointly inverted
    for per-runner (mu_i, sigma_i).

    The exact-rank marginal is not invertible alone (two-branched), but
    paired with the win curve it is: P(2nd) + P(win) = P(top-2), and
    (win, top-2) is the well-posed pair loc_scale_from_topk_pair
    solves. Each marginal is renormalized to unit mass first (the
    market overround treatment), so the top-2 target sums to its two
    slots by construction. ridge= and mu0= pass through."""
    p1 = np.asarray(p_win, dtype=float)
    p2 = np.asarray(p_second, dtype=float)
    if len(p1) != len(p2):
        raise ValueError("p_win and p_second must have equal length")
    if np.any(p1 <= 0) or np.any(p2 <= 0):
        raise ValueError(
            "all win and second probabilities must be positive: a zero "
            "entry has no finite inverse (floor small entries upstream)")
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    return loc_scale_from_topk_pair(p1, 1, p1 + p2, 2, D0=D0, base=base,
                                    points=points, n_iter=n_iter, tol=tol,
                                    ridge=ridge, mu0=mu0,
                                    return_info=return_info)


def _rank_marginal_with_jacobian(mu, sd, r, base_rows, points):
    """P(R_i = r) for every i, plus its full mu-Jacobian:

        dP(R_i = r)/dmu_j = int f_i f_j [ P(N_{-ij} = r-1)
                                        - P(N_{-ij} = r-2) ] dx,  j != i

    (the r = 1 case drops the second term and recovers the win
    Jacobian), with the diagonal from translation invariance."""
    n = len(mu)
    lo, hi = _count_window(mu, sd, n - 1, base_rows)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    dens = f / sd[None, :]
    C = _count_distribution(F)
    p = np.empty(n)
    J = np.zeros((n, n))
    for i in range(n):
        Qi = _loo_pmf(C, F, i)
        p[i] = (Qi[:, r - 1] * dens[:, i]).sum() * dx
        hi_pair = _pair_pmf_at(Qi, F, i, r)          # P(N_{-ij} = r-1)
        row = (hi_pair * dens.T * dens[:, i][None, :]).sum(axis=1) * dx
        if r >= 2:
            lo_pair = _pair_pmf_at(Qi, F, i, r - 1)  # P(N_{-ij} = r-2)
            row -= (lo_pair * dens.T
                    * dens[:, i][None, :]).sum(axis=1) * dx
        row[i] = 0.0
        row[i] = -row.sum()
        J[i] = row
    return p, J


def abilities_from_rank_marginal(p, r, mu0=None, D=None, base="normal",
                                 points=513, n_iter=60, tol=1e-8,
                                 return_info=False):
    """Invert one EXACT-rank marginal -- P(finish exactly r-th) -- for
    mean-zero locations at frozen scales, by Levenberg-damped
    Gauss-Newton on log residuals.

    This is the deliberately two-branched problem: P(R_i = r) is
    non-monotone in mu_i for r >= 2, so several ability vectors can
    share one marginal, and mu0= (physical units) selects the branch --
    supply it from the win odds or any prior ordering. Without mu0 the
    solver starts from zeros and converges to SOME consistent field,
    with no promise it is the one you meant. The target is renormalized
    to unit mass (every rank is taken by someone). For the well-posed
    alternative, combine with the win curve: see
    loc_scale_from_win_and_second and abilities_from_topk."""
    target = np.asarray(p, dtype=float)
    n = len(target)
    r = int(r)
    if not 1 <= r <= n:
        raise ValueError(f"rank must be in [1, n]; got r={r}, n={n}")
    if np.any(target <= 0):
        raise ValueError(
            "all rank probabilities must be positive: a zero entry has "
            "no finite inverse (floor small entries upstream)")
    target = target / target.sum()
    logt = np.log(target)
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    mu = (np.zeros(n) if mu0 is None
          else np.asarray(mu0, dtype=float) - np.mean(mu0))

    phat, J = _rank_marginal_with_jacobian(mu, sd, r, base_rows, points)
    resid = np.log(np.maximum(phat, 1e-300)) - logt
    cost = float(resid @ resid)
    resid_max = float(np.abs(resid).max())
    lam = 1e-6
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        if resid_max < tol:
            break
        Jlog = J / np.maximum(phat, 1e-300)[:, None]
        A = Jlog.T @ Jlog
        g = Jlog.T @ resid
        accepted = False
        for _ in range(8):
            try:
                step = np.linalg.solve(A + lam * np.eye(n), -g)
            except np.linalg.LinAlgError:
                lam *= 8.0
                continue
            mu_n = mu + step
            mu_n -= mu_n.mean()
            p_n, J_n = _rank_marginal_with_jacobian(mu_n, sd, r,
                                                    base_rows, points)
            r_n = np.log(np.maximum(p_n, 1e-300)) - logt
            cost_n = float(r_n @ r_n)
            if cost_n < cost:
                mu, phat, J, resid, cost = mu_n, p_n, J_n, r_n, cost_n
                resid_max = float(np.abs(resid).max())
                lam = max(lam / 3.0, 1e-10)
                accepted = True
                break
            lam *= 8.0
        if not accepted:
            break
    return _topk_inverse_return(mu, resid_max < tol, resid_max, iters,
                                np.zeros(n, dtype=bool), tol, return_info,
                                "abilities_from_rank_marginal")


def rank_probabilities(mu, D=None, base="normal", points=513, V=None,
                       qa=15):
    """The full rank marginals: an (n, n) matrix whose (i, r) entry is
    P(contestant i finishes in position r+1), min-wins.

        P(R_i = r) = int f_i(x) P(N_{-i}(x) = r-1) dx,

    the cavity count pmf against the runner's own density -- winner
    pricing uses the zero-finish coefficient, second place the
    one-finish coefficient, and so on down the field. Rows sum to one
    (every runner takes exactly one rank) and columns sum to one (every
    rank is taken), and both identities are enforced. Cumulative row
    sums reproduce top_k_probabilities. O(n^2 L) per factor node."""
    mu = np.asarray(mu, float)
    n = len(mu)
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    def one_node(m):
        lo, hi = _count_window(m, sd, n - 1, base_rows)
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
        z = (x[:, None] - m[None, :]) / sd[None, :]
        S, f, _ = base_rows(z)
        F = np.clip(1.0 - S, 0.0, 1.0)
        dens = f / sd[None, :]
        C = _count_distribution(F)
        P = np.empty((n, n))
        for i in range(n):
            Qi = _loo_pmf(C, F, i)
            P[i] = (Qi * dens[:, i][:, None]).sum(axis=0) * dx
        return P

    if V is None:
        P = one_node(mu)
    else:
        Vm = np.asarray(V, float)
        if Vm.ndim == 1:
            Vm = Vm[:, None]
        if Vm.shape[1] > 2:
            raise NotImplementedError(
                "rank_probabilities mixes Gauss-Hermite factor nodes and "
                "is implemented for factor rank <= 2 (issue #12).")
        Vm = Vm - Vm.mean(axis=0, keepdims=True)
        an, aw = roots_hermitenorm(qa)
        aw = aw / aw.sum()
        if Vm.shape[1] == 1:
            nodes, w = an[:, None], aw
        else:
            nodes = np.array([[a, b] for a in an for b in an])
            w = np.array([u * v for u in aw for v in aw])
            w = w / w.sum()
        P = np.zeros((n, n))
        for q in range(len(nodes)):
            P += w[q] * one_node(mu + Vm @ nodes[q])

    rows = P.sum(axis=1)
    cols = P.sum(axis=0)
    if (not np.isfinite(P).all() or np.abs(rows - 1).max() > 5e-3
            or np.abs(cols - 1).max() > 5e-3):
        raise RuntimeError(
            "rank marginals defective: row-sum error "
            f"{np.abs(rows-1).max():.2e}, column-sum error "
            f"{np.abs(cols-1).max():.2e}. Raise points=, or report this "
            "field.")
    P = P / rows[:, None]
    return np.clip(P, 0.0, 1.0)
