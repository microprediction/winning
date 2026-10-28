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

from ..shapes import (as_factor_law, as_idio, as_loadings, as_nonnegative,
                      as_points, as_tolerance, rescaled_target)


from .races import (_jacobi_sweeps, BASES, _resolution, kink_lattice,
                    _lower_cdf, _tail_fn, _translated_home)
from .blocks import TINY, roots_hermitenorm

from ..rustconfig import load_fastrace

# compiled kernels (rust/fastrace); honours WINNING_PURE and use_rust()
_fastrace, _RUST_OK, _HAVE_RUST = load_fastrace('top_k')


def _count_window(mu, sd, k, base_rows, delta=1e-12, pad_sds=2.0,
                  is_normal=False):
    """Lattice window for the top-k integrand.

    Below the window at most delta of a single runner has finished
    (sum_j F_j <= delta bounds P(N >= 1)); above it the count exceeds k
    almost surely (Chernoff: mean count mu* with
    mu* - sqrt(2 mu* ln(1/delta)) >= k forces P(N <= k-1) <= delta).
    Both ends by bisection on the monotone mean count, bracketed by
    geometric expansion first, in the node-aware style of the
    hierarchical kernels. is_normal=True routes to the compiled search
    when fastrace ships it: the ~260 sequential bracket/bisection
    evaluations dominate small-field solves in python (measured ~2.6 ms
    of a ~5 ms forward at n = 9).

    The search runs in STANDARDISED units -- centred, divided by the
    widest sd -- and the window is mapped back. Every membership is
    invariant under mu -> a + c mu, sd -> c sd, but both searches
    floored the widest sd at an absolute 1e-12, so a field written in
    units of 1e-18 got a window 2e5 times its own width, the 8193-point
    cap undersampled every density, and top-1 raised a "total
    membership 66.9" where race_probabilities returned the right answer
    (#370). In standardised units the floor can never bind (smax = 1).
    """
    mu = np.asarray(mu, dtype=float)
    sd = np.asarray(sd, dtype=float)
    m0 = float(mu.mean())
    c = float(sd.max())
    if not (np.isfinite(c) and c > 0.0):
        raise ValueError("top-k window needs finite positive scales")
    lo, hi = _count_window_std((mu - m0) / c, sd / c, k, base_rows, delta,
                               pad_sds, is_normal)
    return m0 + c * lo, m0 + c * hi


def _count_window_std(mu, sd, k, base_rows, delta, pad_sds, is_normal):
    if is_normal and _HAVE_RUST and hasattr(_fastrace, "top_k_window"):
        return _fastrace.top_k_window(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), int(k), delta, pad_sds)
    # delta is a request. A polynomial tail puts the 1e-12 count quantile
    # so far out that the capped grid cannot resolve the bulk: Student
    # t(2.1) gave a window of 1.2e5 at dx = 14 against a central scale of
    # 0.22, captured 0.2% of the membership and raised (#386). Relax by
    # factors of 100 (to at most 1e-4, as the win race's bulk window
    # does) until the window fits the cap at half the narrowest runner's
    # sd times the base's central scale, and say so.
    res = _resolution(base_rows)
    afford = 0.5 * max(float(np.min(sd)), 1e-300) * res * (_MAX_TOPK_POINTS - 1)
    d = float(delta)
    lo, hi = _count_window_at(mu, sd, k, base_rows, d, pad_sds)
    while hi - lo > afford and d < 1e-4:
        d = min(d * 100.0, 1e-4)
        lo, hi = _count_window_at(mu, sd, k, base_rows, d, pad_sds)
    if d > delta:
        import warnings
        warnings.warn(
            f"top-k window relaxed delta from {delta:.0e} to {d:.0e}: this "
            f"base's tail puts the requested count quantile further out "
            f"than {_MAX_TOPK_POINTS} points can resolve, so the window is "
            f"{hi - lo:.3g} units wide at the relaxed delta and exact there.",
            RuntimeWarning, stacklevel=3)
    return lo, hi


_MAX_TOPK_POINTS = 8193


def _count_window_at(mu, sd, k, base_rows, delta, pad_sds):
    # standardised units: the widest sd is exactly 1, so no floor (#370)
    smax = float(sd.max())

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


def _count_distribution(F, S=None):
    """Full-field Poisson-binomial: C[l, m] = P(exactly m of the n
    runners are below lattice point l). One shared dynamic program,
    O(n^2 L); every column of C also prices an order statistic, since
    P(X_(k) <= x_l) = sum_{m >= k} C[l, m]. S, when given, is the
    survival to use in place of 1 - F (accurate where 1 - F cancels)."""
    L, n = F.shape
    S = 1.0 - F if S is None else S
    C = np.zeros((L, n + 1))
    C[:, 0] = 1.0
    for j in range(n):
        f = F[:, j:j + 1]
        s = S[:, j:j + 1]
        C[:, 1:j + 2] = C[:, 1:j + 2] * s + C[:, :j + 1] * f
        C[:, 0] *= s[:, 0]
    return C


_SYMMETRIC_BASES = (BASES["normal"], BASES["logistic"], BASES["laplace"])


def _cdf_rows(base_rows, z, S):
    """(F, S) for the count program. 1 - S rounds a lower-tail CDF to
    zero (or to a few digits) once S is near one, and those CDFs are
    exactly what a last place or a top-k cavity multiplies together; for
    a base symmetric about zero the CDF is the reflected survival
    S(-z), accurate to full relative precision (#511).

    A named base floors S at 1e-300, a guard for log(0); used as a
    probability here it fabricated rare-place mass (top-1 of a normal
    runner 53 behind: 1.0e-300 for 1.105e-307), so floored points take
    the base's exact log survival instead (#502). Nothing above 1e-300
    moves.

    A callable base has no identity to reflect on; one that carries a
    cdf (or log_cdf) attribute gets its lower tail from that instead,
    and one with neither keeps 1 - S."""
    S = _unfloored(base_rows, z, np.clip(S, 0.0, 1.0))
    F = 1.0 - S
    if base_rows in _SYMMETRIC_BASES:
        lower = S > 0.5
        if lower.any():
            F = np.where(lower, _unfloored(base_rows, -z,
                                           base_rows(-z)[0]), F)
    else:
        cdf = _lower_cdf(base_rows)
        if cdf is not None:
            lower = S > 0.5
            if lower.any():
                F = np.where(lower, cdf(z), F)
    return np.clip(F, 0.0, 1.0), S


def _unfloored(base_rows, z, S):
    lsf = _tail_fn(base_rows, "log_sf")
    if lsf is None:
        return S
    low = S <= 1e-300
    if not low.any():
        return S
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        return np.where(low, np.exp(np.maximum(lsf(z), -1e300)), S)


# Cavity deconvolution with error bounds (#511). Every entry of the count
# distribution C is a sum of nonnegative products formed in at most n+1
# multiply-add steps, so it carries relative error <= 2 (n+1) eps, plus an
# underflow floor once tail products flush below the normal range. Each
# recursion step adds one subtraction and one division; the running
# first-order bound below tracks the absolute error of every recovered
# coefficient, capped at 1 (a clipped probability can be no further off).
#
# The forward pass, Q_m = (C_m - F_i Q_{m-1}) / S_i, is exact at m = 0 (a
# pure product of survivals); the backward pass, Q_m = (C_{m+1} -
# S_i Q_{m+1}) / F_i, is exact at m = n-1 (a pure product of CDFs). The
# old rule picked ONE direction per lattice point (whichever divides by
# the larger factor). That bounds error growth but not the cancellation
# in the subtraction itself: a rare coefficient came back as the
# roundoff residue (~eps x its neighbour) of the other two terms, and
# integrating that residue fabricated exact-place mass -- 62x too much
# last place in a Gumbel field, 1.6e12x at mu = [-12, -4, 4, 12], and a
# 1.8e5x top-1 share at mu = [-14, 0, 0] (#511). Running both passes and
# taking each coefficient from the one with the smaller bound fixes it;
# against a directly multiplied cavity on the same grid every
# coefficient then agrees to ~1e-15 relative.
_EPS = np.finfo(float).eps
_REALMIN = np.finfo(float).tiny


def _count_error(C):
    """Absolute error bound on each count coefficient, with one eps each
    folded in for the subtraction (its subtrahend is at most the entry)
    and the division of the recursion step that consumes it."""
    n = C.shape[-1] - 1
    return (2.0 * (n + 1) + 2.0) * _EPS * C + (n + 1) * _REALMIN


def _deconv(C, Cerr, Fi, Si, forward, stop=0):
    """One deconvolution pass, yielding (m, Q_m, bound_m).

    Vectorised over the shape of Fi (lattice points, or runners x lattice
    points); C[..., j] and Cerr[..., j] must broadcast against it. The
    forward pass runs m = 0 .. n-1-stop, the backward m = n-1 .. stop. A
    pass whose divisor is numerically zero recovers nothing -- C then
    holds no information about the coefficients it would divide out --
    so every bound it reports is the trivial one."""
    n = C.shape[-1] - 1
    if forward:
        dead = Si < TINY
        inv = 1.0 / np.maximum(Si, TINY)
        mul = Fi
        order = [(m, m) for m in range(0, n - stop)]
    else:
        dead = Fi < TINY
        inv = 1.0 / np.maximum(Fi, TINY)
        mul = Si
        order = [(m, m + 1) for m in range(n - 1, stop - 1, -1)]
    q = e = None
    for m, j in order:
        if q is None:
            q = np.minimum(C[..., j] * inv, 1.0)
            e = np.where(dead, 1.0, np.minimum(Cerr[..., j] * inv, 1.0))
        else:
            q = np.minimum(np.maximum((C[..., j] - mul * q) * inv, 0.0),
                           1.0)
            e = np.minimum((Cerr[..., j] + mul * e) * inv, 1.0)
        yield m, q, e


def _cdf_both(Cp, Ep, Fp, Sp, k):
    """Per-coefficient cavity cdf for a flat batch of (runner, lattice
    point) pairs: both passes in full, each coefficient from the pass
    with the smaller bound, then the head sum or one minus the tail sum,
    whichever is better bounded."""
    fwd = list(_deconv(Cp, Ep, Fp, Sp, True))
    head = np.zeros_like(Fp)
    head_e = np.zeros_like(Fp)
    tail = np.zeros_like(Fp)
    tail_e = np.full_like(Fp, _EPS)
    for m, qb, eb in _deconv(Cp, Ep, Fp, Sp, False):
        _, qf, ef = fwd[m]
        better = ef <= eb
        q = np.where(better, qf, qb)
        e = np.where(better, ef, eb)
        if m < k:
            head += q
            head_e += e
        else:
            tail += q
            tail_e += e
    return np.where(head_e <= tail_e, head, 1.0 - tail)


# a cdf whose bound exceeds this fraction of itself is recomputed per
# coefficient; below it the two-estimate answer is already that accurate
_CDF_REL = 1e-6


def _leave_one_out_cdf(C, F, k, chunk=256, S=None):
    """P(N_{-i} <= k-1) for every runner at every lattice point.

    First two estimates per lattice point, each with its error bound: the
    head sum Q_0 + .. + Q_{k-1} by the forward pass, and one minus the
    tail sum Q_k + .. + Q_{n-1} by the backward pass; the smaller bound
    wins. That is the old recursion length, and the old rule chose
    between the same two by the size of F_i alone -- at k = 1, mu =
    [-14, 0, 0] it returned a 3.7e-18 residue for a top-1 probability of
    2.09e-23 that race_probabilities prices right (#511). Where even the
    better bound exceeds 1e-6 of the value (a rare interior cavity), the
    pair is redone with both passes in full and a per-coefficient choice.
    Those pairs are few on ordinary fields; at worst every pair is redone
    and the cost is about three times the old one."""
    L, n = F.shape
    if k >= n:
        return np.ones((n, L))
    out = np.empty((n, L))
    Cerr = _count_error(C)
    Ct, Et = C[None], Cerr[None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Fc = F[:, a:b].T                      # (c, L)
            Sc = 1.0 - Fc if S is None else S[:, a:b].T
            head = np.zeros_like(Fc)
            head_e = np.zeros_like(Fc)
            for _, q, e in _deconv(Ct, Et, Fc, Sc, True, stop=n - k):
                head += q
                head_e += e
            tail = np.zeros_like(Fc)
            tail_e = np.full_like(Fc, _EPS)
            for _, q, e in _deconv(Ct, Et, Fc, Sc, False, stop=k):
                tail += q
                tail_e += e
            use_head = head_e <= tail_e
            cdf = np.where(use_head, head, 1.0 - tail)
            bound = np.where(use_head, head_e, tail_e)
            loose = bound > np.maximum(_CDF_REL * cdf, 1e-280)
            if loose.any():
                ri, li = np.nonzero(loose)
                per = max(1, int(2e6 // (n + 1)))
                for s0 in range(0, len(ri), per):
                    r_, l_ = ri[s0:s0 + per], li[s0:s0 + per]
                    cdf[r_, l_] = _cdf_both(C[l_], Cerr[l_], Fc[r_, l_],
                                            Sc[r_, l_], k)
            out[a:b] = np.clip(cdf, 0.0, 1.0)
    return out


def _loo_pmfs(C, F, S=None, runners=None):
    """Leave-one-out pmfs for several runners at once, yielding
    (i, m, Q^{-i}_m over the lattice) runner-chunk by runner-chunk: both
    passes run in full and each coefficient is taken from the pass whose
    bound is smaller (#511)."""
    L, n = F.shape
    runners = list(range(n)) if runners is None else list(runners)
    Cerr = _count_error(C)
    Ct, Et = C[None], Cerr[None]
    # both passes are held whole, 4 n x chunk x L numbers
    chunk = max(1, min(len(runners), int(2e6 // max(1, n * L))))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for a in range(0, len(runners), chunk):
            idx = runners[a:a + chunk]
            Fc = F[:, idx].T
            Sc = 1.0 - Fc if S is None else S[:, idx].T
            fwd = list(_deconv(Ct, Et, Fc, Sc, True))
            bwd = {m: (q, e) for m, q, e in _deconv(Ct, Et, Fc, Sc, False)}
            for m, qf, ef in fwd:
                qb, eb = bwd[m]
                better = ef <= eb
                yield idx, m, np.where(better, qf, qb), np.where(better, ef, eb)


def _loo_pmf(C, F, i, S=None):
    """Full leave-one-out pmf Q^{-i}[l, m], m = 0..n-1, each coefficient
    from whichever deconvolution direction has the smaller error bound
    at that lattice point (#511), and that bound (L, n)."""
    L, n = F.shape
    Q = np.empty((L, n))
    E = np.empty((L, n))
    for _, m, q, e in _loo_pmfs(C, F, S, runners=[i]):
        Q[:, m] = q[0]
        E[:, m] = e[0]
    return Q, E


def _reflected(base_rows):
    """Rows of the reflected noise -e: survival F(-z) = 1 - S(-z), density
    f(-z), derivative -f'(-z). Used ONLY to place a window -- the
    subtraction is exact enough for that and for nothing finer, except
    where the base carries its own lower tail (the Gumbel, a callable
    with cdf/log_cdf): there 1 - S(-z) rounded a far tail to zero and
    the window treated it as empty (#710)."""
    cdf = _lower_cdf(base_rows)

    def rows(z):
        S, f, fp = base_rows(-z)
        if cdf is not None:
            return np.maximum(cdf(-z), 1e-300), f, -fp
        return 1.0 - S, f, -fp
    # the reflection keeps the base's central scale: without it the
    # bottom-k window budget ignored the declared resolution (#468)
    rows.resolution = _resolution(base_rows)
    return rows


def _kinks(base_rows, kink_split):
    """The base's kink offsets when the split lattice is asked for, else
    (). Smooth bases have none, so kink_split=True is a no-op for them."""
    return tuple(getattr(base_rows, "kinks", ())) if kink_split else ()


def _split_nodes(lo, hi, points, mu, sd, kinks):
    """Nodes and weights of the kink-split lattice (races.kink_lattice)
    over the window [lo, hi] at the given budget: Gauss-Legendre on the
    pieces between runners' kinks. The uniform rectangle sum is only
    second order across a kink, and its slope integrand -f'(z) JUMPS
    there, so a rectangle-sum slope or Jacobian is only first order with
    an error of dx times the jump (which scales as 1/sd^2): a laplace
    own-slope came out 6x too large for a runner whose sd is three
    spacings (#267). On the split lattice both are spectral again. The
    node count stays under about 2 * points (kink_lattice's cap)."""
    return kink_lattice(np.linspace(lo, hi, points),
                        np.asarray(mu, dtype=float), sd, kinks)


def _topk_independent(mu, sd, k, base_rows, points, delta=1e-12,
                      is_normal=False, upper=False, kink_split=False):
    # the lattice is built in the caller's units: at a common offset of
    # 2^49 its points collapse to the float spacing there (#477)
    mu = _translated_home(np.asarray(mu, float), sd * sd)
    if upper:
        # P(i among the k LARGEST) = int f_i(x) P(M_{-i}(x) <= k-1) dx,
        # M_{-i}(x) the number of OTHER runners ABOVE x: the same count
        # program with the survival S_j(x) in the role F_j(x) plays for
        # top-k. The window is the top-k window of the reflected field.
        # The old route, 1 - top_k(n - k), cancelled: at mu = [-12, 0, 0]
        # the last-place probability 7.7e-24 came back as 0 at 513 points
        # and as residue 3.6e-15 -- nine orders too large -- at 8193 (#365).
        lo_r, hi_r = _count_window(-np.asarray(mu, float), sd, k,
                                   _reflected(base_rows), delta=delta)
        lo, hi = -hi_r, -lo_r
        points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
        kinks = _kinks(base_rows, kink_split)
        if kinks:
            x, wq = _split_nodes(lo, hi, points, mu, sd, kinks)
        else:
            x = np.linspace(lo, hi, points)
            dx = x[1] - x[0]
        z = (x[:, None] - mu[None, :]) / sd[None, :]
        S, f, _ = base_rows(z)
        # survival in the role the CDF plays for top-k, and the CDF in
        # the role of the survival -- taken from _cdf_rows, not as 1 - S:
        # below a leader the others' CDFs are tiny, 1 - S rounded them to
        # zero, and at mu = [-30, 0, 0] the last place 4.68e-134 came
        # back 500x low (#511)
        F, S = _cdf_rows(base_rows, z, S)
        C = _count_distribution(S, F)
        cdf = _leave_one_out_cdf(C, S, k, S=F)
        dens = (f / sd[None, :]).T
        if kinks:
            return (dens * cdf) @ wq
        return (dens * cdf).sum(axis=1) * dx
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta,
                           is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    if is_normal and _HAVE_RUST:
        return np.asarray(_fastrace.top_k(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), k, lo, hi, points))
    # kink_split (opt-in): a kinked base on the split lattice; off, this
    # is the uniform rectangle sum exactly as before
    kinks = _kinks(base_rows, kink_split)
    if kinks:
        x, wq = _split_nodes(lo, hi, points, mu, sd, kinks)
    else:
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    C = _count_distribution(F, S)
    cdf = _leave_one_out_cdf(C, F, k, S=S)    # (n, L)
    dens = (f / sd[None, :]).T                # (n, L)
    if kinks:
        return (dens * cdf) @ wq
    return (dens * cdf).sum(axis=1) * dx


_TOPK_QMC_NODES = 1024


def _factor_nodes(V, n, qa, caller, D=None, F=None, W=None):
    """Column-centered loadings plus the factor node mixture (the common
    column is gauge), shared by every correlated entry point so the
    quadrature is identical across them.

    qa=None (the default) takes the GENERAL RACE'S node rule
    (races._setup: Gauss-Hermite order scaled with the centred-loading
    sharpness, escalating to a midpoint-quantile grid at rank one and
    scrambled Sobol at rank two). A fixed 15-node Gauss-Hermite rule was
    the only option here, and on a sharp rank-one field (loadings +-3,
    D = 0.01: sharpness 42) top-1 missed race_probabilities -- the same
    quantity -- by 10.7 percentage points, raising points= changed
    nothing, and both slot sums were exact, so no check fired; at rank
    two a basis rotation V -> VQ, which leaves the covariance unchanged,
    moved a top-2 membership by 1.3 points (#340). An explicit integer
    qa keeps the fixed Gauss-Hermite rule of that order.

    Any rank. Above two the default is the race's rule too -- a pruned
    Gauss-Hermite tensor inside the 1e5-node budget, scrambled Sobol past
    it or past the rank's sharpness threshold -- capped at the same 1024
    Sobol nodes as rank two; it is deterministic (fixed seed), never
    Monte Carlo. An explicit qa at rank > 2 is the pruned product rule
    hermite_nodes(r, qa). The rank-two refusal that stood here (#12)
    only guarded the hand-built tensor below, which ranks <= 2 still use.

    F, W: the caller's own factor law, (nodes, rank) and weights, read
    as race_probabilities reads them (relative weights, zero-weight
    nodes dropped) -- what fit_covariance returns, for instance."""
    Vm = as_loadings(V, n)
    # the shared scale-safe gauge: a plain mean overflowed at a common
    # [1e308] column and left a huge residue at [1e307] (#471)
    from .core import _gauge_center
    Vm = _gauge_center(Vm)
    if F is not None or W is not None:
        if F is None or W is None:
            raise ValueError(
                "supply both F (factor nodes) and W (their weights), or "
                f"neither; got only {'F' if W is None else 'W'}")
        if qa is not None:
            raise ValueError(
                f"{caller}: F, W already give the factor rule; qa= would "
                "give another. Pass one or the other.")
        F = np.asarray(F, dtype=float)
        if F.ndim != 2:
            raise ValueError(
                f"F must be a 2-D (nodes, rank) array; got shape {F.shape}")
        if F.shape[1] != Vm.shape[1]:
            raise ValueError(
                f"F has {F.shape[1]} factor columns but V has rank "
                f"{Vm.shape[1]}; F is (nodes, rank)")
        nodes, w = as_factor_law(F, W)
        return Vm, nodes, w
    if Vm.shape[1] == 0:
        # the EMPTY PRODUCT: rank 1 was special-cased and every other
        # rank fell into the rank-2 tensor, so an (n, 0) matrix -- the
        # documented spelling of "no factors" -- was integrated over a
        # two-dimensional factor space it does not have (#309)
        return Vm, np.zeros((1, 0)), np.ones(1)
    if qa is None:
        from .races import _setup
        _, _, _, nodes, w, _, _, _ = _setup(np.zeros(n), Vm, D, None, None,
                                            "normal")
        if len(w) > _TOPK_QMC_NODES:
            # The race's escalation is 8192 Sobol nodes, priced there by
            # one compiled pass; here every node is a full count-program
            # pass (~0.7 ms at n = 4), so 8192 of them is ~6 s per call
            # and minutes per inversion. 1024 scrambled-Sobol nodes held
            # the #340 rank-two rotation discrepancy to 2.1e-4 (it was
            # 1.3e-2 at the old fixed qa=15) at an eighth of the cost.
            from .core import qmc_nodes
            nodes, w = qmc_nodes(Vm.shape[1], m=10)
        return Vm, np.asarray(nodes, float), np.asarray(w, float)
    if Vm.shape[1] > 2:
        from .core import hermite_nodes
        nodes, w = hermite_nodes(Vm.shape[1], Q=qa)
        return Vm, nodes, w / w.sum()
    an, aw = roots_hermitenorm(qa)
    aw = aw / aw.sum()
    if Vm.shape[1] == 1:
        nodes, w = an[:, None], aw
    else:
        nodes = np.array([[a, b] for a in an for b in an])
        w = np.array([u * v for u in aw for v in aw])
        w = w / w.sum()
    return Vm, nodes, w


def _checked_topk(raw, k, kind, mass_tol=5e-3):
    t = float(raw.sum())
    if not np.isfinite(t) or abs(t - k) > mass_tol * k:
        raise RuntimeError(
            f"{kind} captured total membership {t:.4f} where exactly "
            f"{k} slots exist (defect {abs(t-k):.2e}): the window or the "
            "deconvolution missed part of the field. Raise points=, or "
            "report this field.")
    q = raw * (k / t)
    # The total is ONE scalar, and the clip below is not mass-neutral: a
    # coarse lattice returned raw [0.465, 1.529, 0.0002] -- total 1.9947,
    # inside the tolerance -- and clipping the impossible 1.529 to 1 left
    # memberships summing to 1.467 for k = 2 (#99). A membership is a
    # probability; one materially above 1 is a resolution failure, and
    # the invariants are re-checked on what is actually returned.
    over = float(q.max() - 1.0) if q.size else 0.0
    if over > mass_tol:
        raise RuntimeError(
            f"{kind} produced a membership of {1.0 + over:.4f} > 1: the "
            "lattice did not resolve this field (the total-mass check "
            "cannot see it, since runner-level errors cancel in one "
            "scalar). Raise points=, or report this field.")
    q = np.clip(q, 0.0, 1.0)
    t2 = float(q.sum())
    if abs(t2 - k) > mass_tol * k:
        raise RuntimeError(
            f"{kind} memberships total {t2:.4f} after clipping to [0, 1], "
            f"where exactly {k} slots exist. Raise points=, or report "
            "this field.")
    return q


def _as_rank(r, n, where="r"):
    """An exact finishing rank, 1..n, as a whole number.

    `int(r)` truncated, so r=1.5 and 1.999 silently solved first place
    and 2.5 second, and could report convergence (#317). Same contract as
    _as_depth, which closed that hole for top-k depths, over [1, n]."""
    try:
        rr = float(r)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a whole-number rank; got {r!r}")
    if isinstance(r, (bool, np.bool_)) or not np.isfinite(rr) \
            or rr != int(rr):
        raise ValueError(
            f"{where} must be a whole-number rank; got {r!r}. A finishing "
            "position is a count, so there is no rank 1.5.")
    rr = int(rr)
    if not 1 <= rr <= n:
        raise ValueError(f"rank must be in [1, n]; got r={r}, n={n}")
    return rr


def _as_runner_index(i, n, where="i"):
    """A runner index: a Python or NumPy integer in [0, n), not a bool.

    The single-row Jacobians and the reduced-rank representation indexed
    with it directly, so -1 silently meant the last runner, and int(y)
    read 0.9 as runner 0 and True as runner 1 -- a plausible gradient
    for the wrong alternative (#846)."""
    if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)):
        raise ValueError(
            f"{where} must be an integer runner index; got {i!r}")
    ii = int(i)
    if not 0 <= ii < n:
        raise ValueError(
            f"{where} must be a runner index in [0, n); got {where}={i!r}, "
            f"n={n}")
    return ii


def _is_symmetric(base_rows):
    """S(z) + S(-z) == 1 on a probe set: the noise law is symmetric."""
    z = np.array([-2.3, -1.1, -0.35, 0.6, 1.7])
    try:
        Sp = np.asarray(base_rows(z)[0], float)
        Sm = np.asarray(base_rows(-z)[0], float)
    except Exception:
        return False
    return bool(np.allclose(Sp + Sm, 1.0, rtol=0.0, atol=1e-12))


def _as_depth(k, n, where="k"):
    """The depth of a top-k curve is a COUNT, so it is an integer.

    Every guard here truncated first -- `int(k)`, and `Math.trunc` /
    `as.integer` in the other ports -- and then range-checked the
    truncated value, so a fractional depth passed and was silently
    floored: `top_k_probabilities(mu, 1.5)` returned the top-1 curve
    and `2.5` the top-2 one, with no warning and a mass of 1 or 2
    rather than the 1.5 or 2.5 the caller asked for. k=0, k=n and k>n
    were all refused; only the non-integer slipped through, which is
    the one case the message "k must be in [1, n-1]" reads as
    permitting.
    """
    kk = float(k)
    if kk != int(kk):
        raise ValueError(
            f"{where} must be a whole number of places; got {k!r}. A "
            "top-k curve counts finishers, so there is no top-1.5.")
    kk = int(kk)
    if not 1 <= kk <= n - 1:
        raise ValueError(f"{where} must be in [1, n-1]; got k={k}, n={n}")
    return kk


def top_k_probabilities(mu, k, V=None, D=None, base="normal", points=513,
                        qa=None, F=None, W=None, kink_split=False):
    """P(X_i among the k smallest), for every i, min-wins.

    mu: locations; D: idiosyncratic variances; V: optional factor
    loadings (n, r) of any rank -- conditional on the factor draw the
    race is independent, and the result is the node mixture of the
    conditional memberships. The node rule is the general race's
    (Gauss-Hermite while it is cheap and the field is mild,
    deterministic scrambled Sobol otherwise, capped at 1024 nodes since
    every node here is a full count program); qa=Q forces a
    Gauss-Hermite rule of order Q. F, W: the factor law as
    race_probabilities takes it, (nodes, r) and relative weights, so
    the covariance fit_covariance returns prices directly:
    V, D, F, W = fit_covariance(C); top_k_probabilities(mu, k, V, D,
    F=F, W=W). F and W are keyword arguments, after the existing ones,
    so no positional call changes meaning. Cost is linear in
    the node count; fit_covariance's 2048 nodes cost twice the default
    rule's 1024, and omitting F, W selects the default.

    k = 1 is the win probability. The identity sum_i q_i = k is checked.
    A lattice that fails it (or returns a membership above one) is
    refined, doubling up to four times the requested points; a defect
    that survives that raises, naming points=. The cap keeps a field
    the lattice cannot resolve from costing more than about seven
    ordinary calls before it says so.

    kink_split: opt-in, default False (the uniform lattice, unchanged).
    As in race_probabilities: a base whose density has a kink (laplace,
    exponential_power_base with beta < 2) is integrated on the split
    lattice races.kink_lattice -- Gauss-Legendre between the runners'
    kinks, at most about 2 * points nodes, numpy only -- instead of the
    uniform rectangle sum, which is second order across a kink. Smooth
    bases ignore it.

    base: a named base or a callable z -> (S, f, fp). A callable may
    carry `cdf(z)` (or `log_cdf(z)`) and `log_sf(z)`: the count program
    then takes lower-tail CDFs and underflowing survivals from them, as
    it does for the named bases, instead of 1 - S, which cancels on rare
    places (bottom-k, last place). They are checked once on a probe;
    without them nothing changes. See winning.factor.races."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = _as_depth(k, n)
    if V is None and (F is not None or W is not None):
        raise ValueError(
            "F, W are nodes of the factor law and need loadings V; "
            "without V there is no factor to integrate over")
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    is_normal = (base == "normal")
    if V is None:
        def raw_at(pts):
            return _topk_independent(mu, sd, k, base_rows, pts,
                                     is_normal=is_normal,
                                     kink_split=kink_split)
    else:
        # before the node shifts are added: at 2^49 they would round (#477)
        mu = _translated_home(mu, D)
        Vm, nodes, w = _factor_nodes(V, n, qa, "top_k_probabilities", D,
                                     F=F, W=W)

        def raw_at(pts):
            raw = np.zeros(n)
            for q in range(len(nodes)):
                shift = Vm @ nodes[q]
                raw += w[q] * _topk_independent(mu + shift, sd, k,
                                                base_rows, pts,
                                                is_normal=is_normal,
                                                kink_split=kink_split)
            return raw
    return _refined_topk(raw_at, k, points, "top-k race")


def _refined_topk(raw_at, k, points, kind, check=None):
    """_checked_topk on raw_at(points), refining a lattice that fails it.

    The checks only see a defect; they cannot fix one, and the remedy
    they name -- raise points= -- is mechanical. A field reported to
    raise at the default 513 points priced cleanly at 2049. So a failed
    check doubles the lattice (2p - 1 keeps the old points as a subset)
    and tries again, up to four times the request and never past the
    lattice cap: bounded at about seven ordinary calls, and only ever
    paid by a call that would otherwise have raised. A field that still
    fails raises, naming the points it tried. A call that passes first
    time is untouched.

    check: the validator, raising RuntimeError on a defect; default
    _checked_topk. rank_probabilities passes its own row/column identity
    checks, so the rank marginals share this bounded policy (#819)."""
    points = as_points(points)     # int() read 2.5 and "257" as budgets (#444)
    cap = min(4 * (points - 1) + 1, _MAX_TOPK_POINTS)
    if check is None:
        def check(raw):
            return _checked_topk(raw, k, kind)
    pts = points
    while True:
        try:
            return check(raw_at(pts))
        except RuntimeError as e:
            if pts >= cap:
                if pts == points:
                    raise
                raise RuntimeError(
                    f"{e} (lattice refined from points={points} to "
                    f"{pts} without resolving it; pass a larger points= "
                    f"explicitly, at most {_MAX_TOPK_POINTS}, or report "
                    "this field)") from e
            pts = min(2 * pts - 1, cap)


def bottom_k_probabilities(mu, k, V=None, D=None, base="normal",
                           points=513, qa=None, kink_split=False):
    """P(X_i among the k largest), for every i, min-wins: the worst k.

    Computed DIRECTLY, as the top-k integral with the count of runners
    above x in place of the count below, so a rare last-place
    probability keeps its relative precision. The complement identity
    P(in the worst k) = 1 - P(in the best n-k) is exact in the continuum
    and useless in floating point at the tail this API faces: it returned
    0 for a 7.7e-24 last place, and lattice noise in the near-one top-k
    value fabricated 3.6e-15 when refined (#365). The slot identity
    sum = k is checked as for top-k. kink_split as in
    top_k_probabilities (opt-in, default off)."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = _as_depth(k, n)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    # the same bounded 2p - 1 refinement as top-k (#623): the default 513
    # points threw "membership 1.98" on a Laplace field 1025 resolves (#800)
    if V is None:
        def raw_at(pts):
            return _topk_independent(mu, sd, k, base_rows, pts, upper=True,
                                     kink_split=kink_split)
        return _refined_topk(raw_at, k, points, "bottom-k race")
    mu = _translated_home(mu, D)                                 # (#477)
    Vm, nodes, w = _factor_nodes(V, n, qa, "bottom_k_probabilities", D)

    def raw_at(pts):
        raw = np.zeros(n)
        for q in range(len(nodes)):
            raw += w[q] * _topk_independent(mu + Vm @ nodes[q], sd, k,
                                            base_rows, pts, upper=True,
                                            kink_split=kink_split)
        return raw
    return _refined_topk(raw_at, k, points, "bottom-k race")


def _pair_pmf_at(Qi, F, i, k, S=None, Qerr=None, chunk=256):
    """P(N_{-ij} = k-1) for every j != i at every lattice point:
    deconvolve runner j's Bernoulli from the leave-i pmf Qi (L, n).

    The same bounded two-direction rule as the first level (#511): the
    forward pass to coefficient k-1 and the backward pass down to it
    each carry a running error bound, seeded with Qerr (the bound on Qi
    itself, from _loo_pmf) plus Qi's own rounding, and the better-bounded
    pass wins at each (j, x). The old rule picked one direction by F_j
    alone, and the cancellation in its subtraction came back as residue:
    off-diagonal Jacobian entries 1e5 to 1e75 times too large on
    mu = [-12, -4, 4, 12] and [-20, -6, 0, 3, 9] against a directly
    multiplied pair cavity on the same grid. S is the survival to use in
    place of 1 - F."""
    L, n = F.shape
    out = np.zeros((n, L))
    if k - 1 > n - 2:            # only n-2 others: P(N_{-ij} = k-1) = 0
        return out
    E = _count_error(Qi) + (0.0 if Qerr is None else Qerr)
    Qt, Et = Qi[None], E[None]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            Fc = F[:, a:b].T                       # (c, L)
            Sc = 1.0 - Fc if S is None else S[:, a:b].T
            for _, qf, ef in _deconv(Qt, Et, Fc, Sc, True, stop=n - 1 - k):
                pass
            for _, qb, eb in _deconv(Qt, Et, Fc, Sc, False, stop=k - 1):
                pass
            out[a:b] = np.where(ef <= eb, qf, qb)
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
    k = _as_depth(k, n)
    i = _as_runner_index(i, n)                                   # (#846)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    mu = _translated_home(mu, D)                                 # (#477)
    lo, hi = _count_window(mu, sd, k, base_rows)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    dens = f / sd[None, :]                     # (L, n)
    C = _count_distribution(F, S)
    Qi, Qe = _loo_pmf(C, F, i, S=S)
    pair = _pair_pmf_at(Qi, F, i, k, S=S, Qerr=Qe)   # (n, L)
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
    k = _as_depth(k, n)
    i = _as_runner_index(i, n)                                   # (#846)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    mu = _translated_home(mu, D)                                 # (#477)
    lo, hi = _count_window(mu, sd, k, base_rows)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    dens = f / sd[None, :]                       # (L, n)
    C = _count_distribution(F, S)
    Qi, Qe = _loo_pmf(C, F, i, S=S)
    pair = _pair_pmf_at(Qi, F, i, k, S=S, Qerr=Qe)   # (n, L)
    kern = pair * dens[:, i][None, :]
    row_mu = (kern * dens.T).sum(axis=1) * dx
    row_sd = (kern * (z * dens).T).sum(axis=1) * dx
    row_mu[i] = 0.0
    row_mu[i] = -row_mu.sum()
    cdf_i = Qi[:, :k].sum(axis=1)
    dfdsd = -(z[:, i] * fp[:, i] + f[:, i]) / D[i]
    row_sd[i] = (dfdsd * cdf_i).sum() * dx
    return row_mu, row_sd


def top_k_jacobians(mu, k, D=None, base="normal", points=513, V=None,
                    qa=None, kink_split=False):
    """Full (n, n) matrices (dq/dmu, dq/dsigma). The lattice, the
    field rows and the shared count distribution are built ONCE and
    reused across rows -- the row helper rebuilds them per call, which
    at n = 150 spent more time on redundant count programs than on the
    pair terms themselves.

    V= (any rank) is EXACT given the nodes: conditional on the factor
    draw the race is independent and the shift by V f_q leaves d/dmu and
    d/dsigma untouched, so the correlated Jacobians are the same node
    mixture as the forward pass,
    J = sum_q w_q J_ind(mu + V f_q, sigma).

    kink_split: opt-in, default False. True differentiates the split
    lattice top_k_probabilities(..., kink_split=True) integrates on. For
    a kinked base that is also the continuum Jacobian to spectral
    accuracy, where the uniform rectangle sum of the own-scale term
    (whose integrand jumps at the runner's kink) is only first order;
    the calibrators use it for that reason (#267)."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = _as_depth(k, n)
    if V is not None:
        mu = _translated_home(mu, np.ones(n) if D is None else D)  # (#477)
        Vm, nodes, w = _factor_nodes(V, n, qa, "top_k_jacobians", D)
        Jm = np.zeros((n, n))
        Js = np.zeros((n, n))
        for j in range(len(nodes)):
            Jm_q, Js_q = top_k_jacobians(mu + Vm @ nodes[j], k, D=D,
                                         base=base, points=points,
                                         kink_split=kink_split)
            Jm += w[j] * Jm_q
            Js += w[j] * Js_q
        return Jm, Js
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    mu = _translated_home(mu, D)                                 # (#477)
    lo, hi = _count_window(mu, sd, k, base_rows,
                           is_normal=(base == "normal"))
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    if (base == "normal" and _HAVE_RUST
            and hasattr(_fastrace, "top_k_jacobians")):
        jm, js = _fastrace.top_k_jacobians(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), k, lo, hi, points)
        return (np.asarray(jm, dtype=float).reshape(n, n),
                np.asarray(js, dtype=float).reshape(n, n))
    kinks = _kinks(base_rows, kink_split)
    if kinks:
        x, wq = _split_nodes(lo, hi, points, mu, sd, kinks)
    else:
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    dens = f / sd[None, :]
    zdens = (z * dens).T
    C = _count_distribution(F, S)
    Jm = np.empty((n, n))
    Js = np.empty((n, n))
    for i in range(n):
        Qi, Qe = _loo_pmf(C, F, i, S=S)
        pair = _pair_pmf_at(Qi, F, i, k, S=S, Qerr=Qe)
        kern = pair * dens[:, i][None, :]
        cdf_i = Qi[:, :k].sum(axis=1)
        dfdsd = -(z[:, i] * fp[:, i] + f[:, i]) / D[i]
        if kinks:
            row_mu = (kern * dens.T) @ wq
            row_sd = (kern * zdens) @ wq
        else:
            row_mu = (kern * dens.T).sum(axis=1) * dx
            row_sd = (kern * zdens).sum(axis=1) * dx
        row_mu[i] = 0.0
        row_mu[i] = -row_mu.sum()
        if kinks:
            row_sd[i] = (dfdsd * cdf_i) @ wq
        else:
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


def _resolved_points(lo, hi, sd, points, res=1.0):
    """Points enough to resolve the NARROWEST density on the window.

    The window is set by the widest runner and the grid by `points`, so a
    field with heterogeneous scales can leave the narrowest density
    between samples: sds of 0.093 and 6.02 at the default 513 points gave
    a spacing of 0.174, nearly twice the narrow runner's whole sd, and
    its membership came out 4.4e-3 wrong (#224).

    The mass check cannot see that. It is ONE SCALAR -- the memberships
    sum to k -- and runner-level errors of opposite sign cancel in it:
    the raw total was 3.9944 against a tolerance of 0.02, comfortably
    inside, and the routine then rescaled a wrong vector to sum to four.

    Same rule as the win race (races.forward_grid): about two points per
    narrowest sd, capped at 8193, warning when the cap still leaves the
    lattice coarse -- rounded up to a dyadic count (#515, below).
    """
    # a floor the caller set, validated as every lattice's: int() read
    # -1, 2.5 and "257" as budgets and the adaptive size replaced them (#444)
    points = as_points(points)
    smin = max(float(np.min(sd)), 1e-300) * float(res)
    need = int(np.ceil((hi - lo) / (0.5 * smin))) + 1
    if need > 8193:
        import warnings
        warnings.warn(
            "top-k lattice cannot resolve the narrowest runner even at "
            f"8193 points (min sd x central scale {smin:.1e} over a window of "
            f"{hi - lo:.3g}); memberships may carry percent-level error "
            "the mass check cannot see, since it is one scalar and "
            "runner-level errors of opposite sign cancel in it.",
            RuntimeWarning, stacklevel=3)
    return max(points, _dyadic_points(need))


def _dyadic_points(need):
    """The adaptive count rounded UP to a dyadic lattice 2^m + 1 (capped
    at 8193), so it is piecewise constant over a factor-of-two band of
    the narrowest scale instead of stepping by one point at every ceil.
    A count that moved 611 -> 610 under a 1e-5 relative change of one sd
    jumped a membership by 1.9e-6, so a central difference of the public
    map read 0.30 against a continuum scale Jacobian of 0.003, and the
    loc/scale inverse that consumes that Jacobian stalled at logit
    residual 3e-4 on an exact target (#515). Costs at most twice the
    points, and only when the adaptive count binds."""
    if need <= 2:
        return 2
    if need >= 8193:
        return 8193
    return min((1 << int(np.ceil(np.log2(need - 1)))) + 1, 8193)


def _topk_with_slopes(mu, sd, k, base_rows, points, delta=1e-12,
                      is_normal=False, kink_split=False):
    """One forward pass returning the raw memberships AND the own
    translation slopes

        dq_i/dmu_i = -int f'(z_i)/sd_i^2 P(N_{-i}(x) <= k-1) dx,

    the cavity cdf the forward pass already holds against the derivative
    of the runner's own density -- one extra weighted sum, no second
    field pass. is_normal=True routes through the compiled kernel when
    fastrace ships it (the python window is computed either way, so the
    quadrature grid is identical)."""
    mu = _translated_home(np.asarray(mu, float), sd * sd)      # (#477)
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta,
                           is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    if is_normal and _HAVE_RUST and hasattr(_fastrace, "top_k_slopes"):
        q, sl = _fastrace.top_k_slopes(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), k, lo, hi, points)
        return np.asarray(q, dtype=float), np.asarray(sl, dtype=float)
    kinks = _kinks(base_rows, kink_split)
    if kinks:
        x, wq = _split_nodes(lo, hi, points, mu, sd, kinks)
    else:
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, fp = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    C = _count_distribution(F, S)
    cdf = _leave_one_out_cdf(C, F, k, S=S)     # (n, L)
    dens = (f / sd[None, :]).T
    if kinks:
        return (dens * cdf) @ wq, -((fp / (sd ** 2)[None, :]).T * cdf) @ wq
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
    from ..shapes import as_target
    # finite and one-dimensional first: NaN passed `target <= 0` and died
    # later in the lattice sizing as "cannot convert float NaN to integer"
    # (#110)
    target = as_target(q)
    if len(target) != n:
        raise ValueError(f"target has {len(target)} entries for {n} runners")
    if target_floor is not None:
        # a membership floor, applied to memberships: normalized to k
        # slots first, not in the caller's units (#592)
        from .races import _apply_floor
        target_floor = as_tolerance(target_floor, "target_floor")
        target, floored = _apply_floor(target, float(k), target_floor)
    else:
        floored = np.zeros(n, dtype=bool)
        if np.any(target <= 0):
            raise ValueError(
                "all target memberships must be positive: a zero top-k "
                "probability has no finite inverse (the supremum is "
                "approached as that runner's contrast diverges). Pass "
                "target_floor= to floor small entries deliberately, or "
                "supply a pseudocount upstream.")
    target = rescaled_target(target, k)          # scale-safe (#461)
    if np.any(target >= 1.0):
        raise ValueError(
            "after renormalizing to k slots, a target membership is >= 1: "
            "certain membership has no finite inverse, and a market vector "
            "this lopsided is outside the model's range (check the "
            "overround treatment and the dead-heat convention of the "
            "place quotes).")
    return target, floored


def _retry_on_failure(first, attempt, starts):
    """Bounded deterministic retry for a failed inverse (#779, #781).

    first is the default attempt's (result, info). When it converged it
    is returned as is: a call that converges today is bitwise unchanged
    and costs nothing extra. Otherwise attempt(start) is run from each
    start of the fixed, finite list starts(first) in order, and the
    first converged outcome is returned. If none converges, the attempt
    with the smallest max logit residual is returned, still reporting
    converged=False. Either way info["iterations"] becomes the total
    over every attempt made, so the reported cost is honest. A first
    attempt that made no iteration (n_iter=0) is not retried: that
    budget prices the start and must not move it. A retry that raises
    (say its start's first pricing exhausts the lattice refinement) is
    skipped: it must not discard the completed attempts (#855)."""
    if first[-1]["converged"] or first[-1]["iterations"] == 0:
        # n_iter=0 prices the start and performs no update (#441, #497)
        return first
    best, total = first, first[-1]["iterations"]
    for start in starts(first):
        try:
            out = attempt(start)
        except Exception:
            continue
        total += out[-1]["iterations"]
        if out[-1]["converged"]:
            best = out
            break
        if out[-1]["max_logit_residual"] < best[-1]["max_logit_residual"]:
            best = out
    best[-1]["iterations"] = int(total)
    return best


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
                        qa=None, n_iter=80, tol=1e-8, target_floor=None,
                        return_info=False, kink_split=False):
    """Invert the top-k race: mean-zero mu with
    top_k_probabilities(mu, k) = q. k = 1 recovers abilities_from_race.

    Residuals live in LOGIT space rather than the win-inversion's log
    space: for k >= 2 the favorites saturate toward q = 1, where log
    residuals lose all sensitivity while both dq/dmu and q(1-q) vanish
    together and their ratio stays informative. Same target contract as
    abilities_from_race (zeros raise, target_floor= opts into flooring,
    non-convergence warns unless return_info=True), plus the slot
    identity: targets are renormalized to sum to k, and an entry >= 1
    after that renormalization raises. V= (any rank) enters by the
    usual node mixture. When k > n/2 the information lives in the
    longshots; inverting the complement (bottom_k_probabilities) is the
    same call at n - k on 1 - q.

    kink_split: opt-in, default False -- invert the split-lattice map
    top_k_probabilities(..., kink_split=True) instead of the default
    one. Either way, a kinked base (laplace, exponential_power_base with
    beta < 2) takes its Newton slopes from the split lattice: the
    rectangle-sum slope is first order there and was up to 6x off,
    which stalled exact self-generated targets (#267). The residual is
    always the map being inverted, so the answer is that map's."""
    mu_probe = np.asarray(q, dtype=float)
    n = len(mu_probe)
    k = _as_depth(k, n)
    tol = as_tolerance(tol)                                   # (#551)
    target, floored = _validated_topk_target(q, k, n, target_floor)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    if V is not None:
        Vm, nodes, w = _factor_nodes(V, n, qa, "abilities_from_topk", D)
    else:
        nodes, w, Vm = None, None, None

    logit_t = np.log(target) - np.log1p(-target)
    logt = np.log(target)
    # The field's contrast scale, as abilities_from_race measures it: the
    # warm start, the step cap and the slope floor were in unit-variance
    # units, so the same race written at sd 1e-2 began 100 sd from its
    # answer and stalled at a logit residual of 690, and at sd 1e3 the
    # capped step was a thousandth of the distance (#100).
    _sig_v = 0.0 if Vm is None else float(np.mean((Vm ** 2).sum(axis=1)))
    scale = float(np.sqrt(np.median(D) + _sig_v))
    mu = -(logt - logt.mean()) / 2.0 * scale
    # Damping, as in abilities_from_race: the photo-finish graph of a pair
    # is bipartite and the undamped Jacobi update two-cycles; the same
    # two-cycle appears at any n when two runners hold nearly all the
    # win mass (#151: k = 1 kept `n > 2` after #150 fixed the win race
    # and failed on the same fields), so the k = 1 gate is the win
    # race's top-two share. For k >= 2 the slot identity spreads the
    # mass and the gate is the pair. The sweeps then adapt the damping
    # to the contraction they observe (see races._jacobi_sweeps).
    _top2 = float(np.sort(target)[-2:].sum()) if (k == 1 and n > 2) else 1.0
    alpha = 0.7 if (n == 2 or (k == 1 and _top2 > 0.8)) else 1.0

    split_slopes = bool(getattr(base_rows, "kinks", ())) and not kink_split

    def _with_slopes(m):
        qj, sj = _topk_with_slopes(m, sd, k, base_rows, points,
                                   is_normal=(base == "normal"),
                                   kink_split=kink_split)
        if split_slopes:
            # a kinked base: the slope from the split lattice, the value
            # from the map being inverted (#267)
            sj = _topk_with_slopes(m, sd, k, base_rows, points,
                                   kink_split=True)[1]
        return qj, sj

    def _forward(m):
        if nodes is None:
            qraw, sl = _with_slopes(m)
        else:
            qraw = np.zeros(n)
            sl = np.zeros(n)
            for j in range(len(nodes)):
                qj, sj = _with_slopes(m + Vm @ nodes[j])
                qraw += w[j] * qj
                sl += w[j] * sj
        qhat = _checked_topk(qraw, k, "top-k inversion")
        if float(qraw.max()) * k > float(qraw.sum()):
            # a membership above one, clipped to exactly one: the logit
            # residual there is ~690 and flat, and the sweeps stalled in
            # it (a laplace top-3 target diverged to residual 683, #267).
            # The uniform lattice overshoots one near saturation for a
            # kinked base; refusing the iterate makes the sweeps back off.
            raise RuntimeError("top-k inversion: a membership above one")
        resid = (np.log(np.maximum(qhat, 1e-300))
                 - np.log(np.maximum(1.0 - qhat, 1e-300))) - logit_t
        dlogit = np.minimum(sl / np.maximum(qhat * (1.0 - qhat), 1e-300),
                            -1e-6 / scale)
        return resid, dlogit

    mu, _, resid_max, iters = _jacobi_sweeps(mu, _forward, scale, alpha,
                                             n_iter, tol)
    return _topk_inverse_return(mu, resid_max < tol, resid_max, iters,
                                floored, tol, return_info,
                                "abilities_from_topk")


def loc_scale_from_topk_pair(q1, k1, q2, k2, D0=None, base="normal",
                             points=513, n_iter=60, tol=1e-8, ridge=0.0,
                             mu0=None, V=None, return_info=False,
                             kink_split=False):
    """Joint (loc, scale) calibration from two membership curves: find
    per-runner (mu_i, sigma_i) with top_k_probabilities(mu, k1) = q1 and
    top_k_probabilities(mu, k2) = q2. The market pair is (top-1, top-2):
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
    ridge = w**2 here -- e.g. a reference using w = 0.05 matches
    ridge = 0.0025 (measured: ridge = 0.05 is a 20x stronger prior that
    biases clean-board parameters by ~0.03). mu0= supplies initial locations in physical units and
    skips the internal warm start.
    Avoid k = n/2 as a depth when the field is nearly level: with a
    symmetric base and equal locations, reflection symmetry makes
    half-field membership exactly scale-blind (q = 1/2 at any spread),
    so that curve carries no scale information at all.

    Correlation (V=) is REFUSED here, and the reason is dimension
    counting, not laziness: fixing the loadings destroys the joint
    rescaling gauge (scaling (mu, sigma) by c now changes the mix of
    factor and idiosyncratic noise), so the unknowns after the
    translation gauge number 2n - 1 while two curves carry only
    2n - 2 informative numbers -- a data-dependent flat direction
    survives and the exact fit is a one-parameter family. A third
    membership depth closes it (3n - 3 >= 2n - 1 for n >= 2, an
    overdetermined least-squares fit); until that solver exists,
    calibrate correlated fields at fixed scales with
    abilities_from_topk(V=...), whose Jacobians (top_k_jacobians V=)
    are already exact node mixtures. A
    market pair outside the family's range converges to its
    least-squares fit and reports converged=False. Independent races
    only, like the Jacobians it stacks. O(n^2 L min(k, n-k)) per
    iteration.

    kink_split: opt-in, default False -- fit the split-lattice map
    top_k_probabilities(..., kink_split=True) instead of the default
    one. Either way a kinked base (laplace, exponential_power_base with
    beta < 2) takes its LM Jacobians from the split lattice, where they
    are the continuum derivatives to spectral accuracy; on the uniform
    lattice they are first order across every runner's kink and exact
    self-generated laplace boards stalled at logit residual 2e-3 (#267).
    The residual is always the map being fitted.

    The cold start (neither D0 nor mu0) takes unit scales and fits mu
    to the shallower curve alone, and on some exact boards the solve
    from there stalls far from the answer (#781). Only then, it is
    retried once from a scale-informed start built from both curves
    (_loc_scale_starts), and the retry must meet the same convergence
    test; if it fails too, the better attempt is returned with
    converged=False. A call that converges first time is unchanged,
    bit for bit."""
    kw = dict(base=base, points=points, n_iter=n_iter, tol=tol,
              ridge=ridge, V=V, kink_split=kink_split)
    out = _loc_scale_once(q1, k1, q2, k2, D0, mu0, **kw)
    if D0 is None and mu0 is None:
        out = _retry_on_failure(
            out, lambda d: _loc_scale_once(q1, k1, q2, k2, d, None, **kw),
            lambda f: _loc_scale_starts(q1, k1, q2, k2))
    mu, sd, info = out
    if return_info:
        return mu, sd, info
    mu = _topk_inverse_return(mu, info["converged"],
                              info["max_logit_residual"],
                              info["iterations"], info["floored"], tol,
                              False, "loc_scale_from_topk_pair")
    return mu, sd


def _probit_rational(p):
    """Acklam's rational inverse normal CDF (relative error ~1e-9), the
    browser's invNormalRational: plain arithmetic, so every port computes
    the same retry start. Only a start is built from it."""
    a = (-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924)
    b = (-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857)
    c = (-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878)
    d = (0.00778469570904146, 0.32246712907004, 2.445134137143,
         3.75440866190742)
    pl = 0.02425
    if p < pl:
        q = np.sqrt(-2.0 * np.log(p))
        return ((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4])
                 * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    if p > 1.0 - pl:
        return -_probit_rational(1.0 - p)
    q = p - 0.5
    r2 = q * q
    return ((((((a[0] * r2 + a[1]) * r2 + a[2]) * r2 + a[3]) * r2 + a[4])
             * r2 + a[5]) * q
            / (((((b[0] * r2 + b[1]) * r2 + b[2]) * r2 + b[3]) * r2 + b[4])
               * r2 + 1.0))


def _loc_scale_starts(q1, k1, q2, k2):
    """The retry start of loc_scale_from_topk_pair's cold solve (#781): a
    scale per runner read off BOTH curves. With the depth thresholds
    t_k held fixed, P(top k) ~ Phi((t_k - mu_i) / sigma_i), so the gap
    between a runner's deeper and shallower probits is (t_k2 - t_k1) /
    sigma_i: sigma_i is proportional to its reciprocal. The common
    factor is a gauge; the spread is clipped to the solver's own
    log-scale box. Returned as a D0, so the start's locations come
    from the same warm start the cold solve uses. On hard boards (log
    sigma spread ~1, memberships down to 1e-15) the cold solve failed
    85 times in 800 and this start rescued 25; supplying the true
    scales would rescue 60, so a caller with a scale prior should pass
    D0."""
    n = len(np.asarray(q1, dtype=float))
    (lo, klo), (hi, khi) = sorted(((q1, k1), (q2, k2)), key=lambda t: t[1])
    tlo, _ = _validated_topk_target(lo, klo, n, None)
    thi, _ = _validated_topk_target(hi, khi, n, None)
    gap = np.array([_probit_rational(b) - _probit_rational(a)
                    for a, b in zip(tlo, thi)])
    ls = -np.log(np.maximum(gap, 1e-3))
    ls = np.clip(ls - ls.mean(), -3.0, 3.0)
    return [np.exp(2.0 * ls)]


def _loc_scale_once(q1, k1, q2, k2, D0, mu0, base, points, n_iter, tol,
                    ridge, V, kink_split):
    """One Levenberg solve of loc_scale_from_topk_pair from its start;
    returns (mu, sd, info) and never warns."""
    if V is not None:
        raise NotImplementedError(
            "loc_scale_from_topk_pair with fixed factor loadings is "
            "under-identified: without the joint rescaling gauge, two "
            "curves carry 2n - 2 numbers against 2n - 1 unknowns and a "
            "flat direction survives (see the docstring). Use "
            "abilities_from_topk(V=...) at fixed scales, or wait for "
            "the three-curve least-squares solver.")
    t1 = np.asarray(q1, dtype=float)
    n = len(t1)
    k1 = _as_depth(k1, n, "k1")
    k2 = _as_depth(k2, n, "k2")
    tol = as_tolerance(tol)                                   # (#551)
    # a negative ridge was max(ridge, 0) = 0: the unregularized fit (#558)
    ridge = as_nonnegative(ridge, "ridge")
    if k1 == k2:
        raise ValueError(
            "k1 == k2 gives one curve twice: scale is unidentified "
            "without a second, different membership depth")
    target1, _ = _validated_topk_target(q1, k1, n, None)
    target2, _ = _validated_topk_target(q2, k2, n, None)
    lt1 = np.log(target1) - np.log1p(-target1)
    lt2 = np.log(target2) - np.log1p(-target2)

    # D0 through as_idio, as every other variance here: a scalar D0 was a
    # 0-d array that failed later with "input arrays have different
    # dimensions", and a wrong length broadcast (#254)
    sd = np.ones(n) if D0 is None else np.sqrt(as_idio(D0, n, positive=True))
    if mu0 is not None:
        mu = np.asarray(mu0, dtype=float) - np.mean(mu0)
        # the RETURN gauge (mean-zero mu, geometric-mean-one sigma) applied
        # to the start as well: it was applied only to accepted LM steps,
        # so an exact warm start converged before any step and came back
        # in the caller's physical units -- sd [2, 3, 4, 5] where a cold
        # solve of the same targets returned [0.60, 0.91, 1.21, 1.51]
        # (#360). Ranks are invariant to it, so no probability moves.
        c0 = float(np.exp(np.log(sd).mean()))
        mu, sd = mu / c0, sd / c0
    else:
        ka, ta = (k1, target1) if k1 < k2 else (k2, target2)
        # warm start only: the LM loop refines, so a loose tolerance
        # here buys iterations without moving the final answer
        mu, _info0 = abilities_from_topk(ta, ka, D=sd ** 2, base=base,
                                         points=points, n_iter=20,
                                         tol=1e-3, return_info=True,
                                         kink_split=kink_split)
    sqr = float(np.sqrt(ridge))
    base_rows = BASES[base] if not callable(base) else base
    # the Jacobians of a kinked base always come from the split lattice
    jac_split = kink_split or bool(getattr(base_rows, "kinks", ()))

    def logits(m, s):
        qh1 = top_k_probabilities(m, k1, D=s ** 2, base=base, points=points,
                                  kink_split=kink_split)
        qh2 = top_k_probabilities(m, k2, D=s ** 2, base=base, points=points,
                                  kink_split=kink_split)
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
    last_grad = np.inf
    accepted = True
    # Polish for a kinked base on the DEFAULT lattice (#267). The uniform
    # map is the one being fitted, and it is not the continuum: its
    # window slides with (mu, sigma), so the grid moves across every
    # runner's kink and the rectangle sum's second-order error turns
    # into an O(1) term in the map's DERIVATIVE (a laplace dq/dsigma
    # read 0.62 by central differences against a continuum 0.10, the
    # same at steps 1e-4 and 1e-6). LM on the continuum Jacobian then
    # stalls near the map's own discretization error (logit residual
    # 1e-3 on an exact self-generated board). When it stalls, the
    # Jacobian is taken instead by forward differences of the map
    # itself -- 2n extra evaluations of each curve per iteration, paid
    # only by a kinked base on the default lattice after a stall. The
    # split lattice (kink_split=True) is smooth and never needs it.
    fd_allowed = jac_split and not kink_split
    fd_mode = False
    slow = 0

    def _fd_jacobian(m, s, r0):
        h = 1e-6
        cols = []
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            cols.append((logits(m + e, s)[0] - r0) / h)
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            cols.append((logits(m, s * np.exp(e))[0] - r0) / h)
        return np.column_stack(cols)

    for it in range(n_iter):
        iters = it + 1
        if resid_max < tol:
            break
        if fd_mode:
            try:
                J = _fd_jacobian(mu, sd, r)
            except RuntimeError:
                break
        else:
            blocks = []
            for kk, qh in ((k1, qh1), (k2, qh2)):
                Jm, Js = top_k_jacobians(mu, kk, D=sd ** 2, base=base,
                                         points=points, kink_split=jac_split)
                g = 1.0 / np.maximum(qh * (1.0 - qh), 1e-300)
                # chain to (mu, log sigma) and to logit residuals
                blocks.append(np.hstack([Jm * g[:, None],
                                         (Js * sd[None, :]) * g[:, None]]))
            # ridge rows: d(sqr log sigma)/d(log sigma) = sqr, zero in mu
            blocks.append(np.hstack([np.zeros((n, n)), sqr * np.eye(n)]))
            J = np.vstack(blocks)
        JtJ = J.T @ J
        Jtr = J.T @ r
        last_grad = float(np.abs(Jtr).max())
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
                # crawling: an accepted Gauss-Newton step that does not
                # even halve the residual norm, twice running
                slow = slow + 1 if cost_n > 0.25 * cost else 0
                mu, sd, r, qh1, qh2, cost = mu_n, sd_n, r_n, q1_n, q2_n, cost_n
                resid_max = float(np.abs(r[:2 * n]).max())
                lam = max(lam / 3.0, 1e-10)
                accepted = True
                break
            lam *= 8.0
        if fd_allowed and not fd_mode and (not accepted or slow >= 2):
            fd_mode = True
            lam = 1e-6
            continue
        if not accepted:
            break
    # With a ridge the penalized optimum generally keeps a nonzero fit
    # residual by design, so an LM stall CAN be the answer -- but a stall
    # alone proves nothing. `converged = resid < tol or (ridge > 0 and
    # stalled)` certified every rejected step, including boards no
    # (mu, sigma) can fit: q1 = [.8, .1, .1] against q2 = [.2, .9, .9]
    # reported converged=True at a logit residual of 1.57 and a 35-point
    # repricing miss (#353, #105). A stall now counts only where it is a
    # stationary point of the penalized objective AND the board passes
    # the exact nesting necessity P(top k1) <= P(top k2) for k1 < k2
    # (the top-k1 event is a subset of the top-k2 one), which no market
    # convention can make false for a feasible board. The two facts are
    # reported separately as well: fit_converged (the market residual
    # met tol) and stationary (the penalized optimum was reached).
    lo_t, hi_t = (target1, target2) if k1 < k2 else (target2, target1)
    nested = bool(np.all(lo_t <= hi_t + 1e-12))
    fit_ok = resid_max < tol
    stationary = bool(fit_ok or (not accepted and iters > 0
                                 and last_grad <= 1e-6 * max(1.0, cost)))
    converged = fit_ok or (sqr > 0.0 and stationary and nested)
    out = _topk_inverse_return(mu, converged, resid_max, iters,
                               np.zeros(n, dtype=bool), tol, True,
                               "loc_scale_from_topk_pair")
    info = dict(out[1])
    info.update(fit_converged=bool(fit_ok), stationary=stationary,
                nested=nested)
    return out[0], sd, info


def loc_scale_from_win_and_second(p_win, p_second, D0=None, base="normal",
                                  points=513, n_iter=60, tol=1e-8,
                                  ridge=0.0, mu0=None, return_info=False,
                                  kink_split=False):
    """The two-marginal transform stated in market terms: win
    probabilities plus EXACTLY-SECOND probabilities, jointly inverted
    for per-runner (mu_i, sigma_i).

    The exact-rank marginal is not invertible alone (two-branched), but
    paired with the win curve it is: P(2nd) + P(win) = P(top-2), and
    (win, top-2) is the well-posed pair loc_scale_from_topk_pair
    solves. Each marginal is renormalized to unit mass first (the
    market overround treatment), so the top-2 target sums to its two
    slots by construction. ridge=, mu0= and kink_split= pass through."""
    from ..shapes import as_target
    p1 = as_target(p_win, "p_win")
    p2 = as_target(p_second, "p_second")
    if len(p1) != len(p2):
        raise ValueError("p_win and p_second must have equal length")
    if np.any(p1 <= 0) or np.any(p2 <= 0):
        raise ValueError(
            "all win and second probabilities must be positive: a zero "
            "entry has no finite inverse (floor small entries upstream)")
    p1 = rescaled_target(p1)                     # scale-safe (#483)
    p2 = rescaled_target(p2)
    return loc_scale_from_topk_pair(p1, 1, p1 + p2, 2, D0=D0, base=base,
                                    points=points, n_iter=n_iter, tol=tol,
                                    ridge=ridge, mu0=mu0,
                                    return_info=return_info,
                                    kink_split=kink_split)


def _rank_marginal_with_jacobian(mu, sd, r, base_rows, points,
                                 is_normal=False, kink_split=False,
                                 jacobian=True, with_mass=False):
    """P(R_i = r) for every i, plus its full mu-Jacobian:

        dP(R_i = r)/dmu_j = int f_i f_j [ P(N_{-ij} = r-1)
                                        - P(N_{-ij} = r-2) ] dx,  j != i

    (the r = 1 case drops the second term and recovers the win
    Jacobian), with the diagonal from translation invariance.
    is_normal=True routes through the compiled kernel on the same
    python-computed window. jacobian=False skips the pair terms (J is
    then zero). with_mass=True also returns each runner's density mass
    on this lattice, the row sum rank_probabilities normalises by."""
    n = len(mu)
    mu = _translated_home(np.asarray(mu, float), sd * sd)      # (#477)
    lo, hi = _count_window(mu, sd, n - 1, base_rows, is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    kinks = _kinks(base_rows, kink_split)
    if kinks:
        x, dx = _split_nodes(lo, hi, points, mu, sd, kinks)
    else:
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
    if (is_normal and _HAVE_RUST
            and hasattr(_fastrace, "rank_marginal_jacobian")):
        p, jac = _fastrace.rank_marginal_jacobian(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), int(r), lo, hi, points)
        out = (np.asarray(p, dtype=float),
               np.asarray(jac, dtype=float).reshape(n, n))
        if with_mass:
            dens = base_rows((x[:, None] - mu[None, :]) / sd[None, :])[1]
            out += ((dens / sd[None, :]).sum(axis=0) * dx,)
        return out
    # dx is the uniform spacing, or the split lattice's weight vector,
    # which every reduction below takes as an inner product instead
    if kinks:
        def _wsum(a):
            return a @ dx
    else:
        def _wsum(a):
            return a.sum(axis=-1) * dx
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F, S = _cdf_rows(base_rows, z, S)
    dens = f / sd[None, :]
    C = _count_distribution(F, S)
    p = np.empty(n)
    J = np.zeros((n, n))
    for i in range(n):
        Qi, Qe = _loo_pmf(C, F, i, S=S)
        p[i] = _wsum(Qi[:, r - 1] * dens[:, i])
        if not jacobian:
            continue
        if r <= n - 1:
            hi_pair = _pair_pmf_at(Qi, F, i, r, S=S, Qerr=Qe)  # = r-1
            row = _wsum(hi_pair * dens.T * dens[:, i][None, :])
        else:
            # only n-2 others exist: P(N_{-ij} = n-1) is identically
            # zero, and computing it by deconvolution injects clamped
            # junk that is hypersensitive to the window edge
            row = np.zeros(n)
        if r >= 2:
            lo_pair = _pair_pmf_at(Qi, F, i, r - 1, S=S,
                                   Qerr=Qe)  # P(N_{-ij} = r-2)
            row -= _wsum(lo_pair * dens.T * dens[:, i][None, :])
        row[i] = 0.0
        row[i] = -row.sum()
        J[i] = row
    if with_mass:
        return p, J, _wsum(dens.T)
    return p, J


def abilities_from_rank_marginal(p, r, mu0=None, D=None, base="normal",
                                 points=513, n_iter=60, tol=1e-8,
                                 return_info=False, kink_split=False):
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
    loc_scale_from_win_and_second and abilities_from_topk.

    kink_split: opt-in, default False -- invert the split-lattice map
    rank_probabilities(..., kink_split=True). Either way a kinked base
    takes its Jacobian from the split lattice (#267: on the uniform one
    an exact self-generated laplace target stopped at log residual
    2.7e-3); the residual is always the map being inverted.

    One damped trajectory can stall at a positive-residual stationary
    point of this multi-branched map even from a near-truth mu0 (#779).
    Only then, the solve is retried from a few fixed deterministic
    starts near the original one (_rank_marginal_starts) and the first
    that meets tol is returned; if none does, the best attempt is
    returned with converged=False. A call that converges first time is
    unchanged, bit for bit."""
    kw = dict(D=D, base=base, points=points, n_iter=n_iter, tol=tol,
              kink_split=kink_split)
    first = _rank_marginal_once(p, r, mu0, **kw)
    mu, _, info = _retry_on_failure(
        first, lambda m: _rank_marginal_once(p, r, m, **kw),
        lambda f: _rank_marginal_starts(mu0, f[0], f[1], D))
    return _topk_inverse_return(mu, info["converged"],
                                info["max_logit_residual"],
                                info["iterations"], info["floored"], tol,
                                return_info, "abilities_from_rank_marginal")


def _rank_marginal_starts(mu0, mu_stall, resid, D):
    """The retry starts of abilities_from_rank_marginal (#779), nearest
    the caller's start first. The stalled residual says which runners
    the failed trajectory left mispriced, so the first two starts nudge
    the ORIGINAL start (mu0, or zeros) along it, -0.3 then +0.3 of the
    typical scale (the map is non-monotone for r >= 2, so its sign is
    not known in advance); the third reflects the failed trajectory
    through the start, sending the solver the other way. On a hard
    battery (n 3-8, r 2-4, warm starts off by up to 0.5, and cold) one
    trajectory failed 72 times in 2840 and these three rescued 64."""
    n = len(mu_stall)
    m0 = np.zeros(n) if mu0 is None else np.asarray(mu0, float) - np.mean(mu0)
    u = resid - resid.mean()
    u = u / max(float(np.abs(u).max()), 1e-300)
    D = np.ones(n) if D is None else np.broadcast_to(np.asarray(D, float), (n,))
    h = 0.3 * float(np.sqrt(np.mean(D)))
    return [m0 - h * u, m0 + h * u, 2.0 * m0 - mu_stall]


def _rank_marginal_once(p, r, mu0, D, base, points, n_iter, tol,
                        kink_split):
    """One Levenberg trajectory of abilities_from_rank_marginal from mu0;
    returns (mu, final log residual, info) and never warns."""
    from ..shapes import as_target
    target = as_target(p, "p")
    n = len(target)
    r = _as_rank(r, n)
    tol = as_tolerance(tol)                                   # (#551)
    if np.any(target <= 0):
        raise ValueError(
            "all rank probabilities must be positive: a zero entry has "
            "no finite inverse (floor small entries upstream)")
    target = rescaled_target(target)             # scale-safe (#483)
    logt = np.log(target)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    if mu0 is None and n % 2 == 1 and r == (n + 1) // 2 \
            and _is_symmetric(base_rows):
        # Under a symmetric base, negating every location reverses the
        # finishing order, so the exact MIDDLE rank of an odd field is an
        # even function of mu and its Jacobian at the default zero start
        # is identically zero: J'J and J'r vanish and no damping makes a
        # step. A self-generated target came back as zeros, converged
        # False, 6.8 points off (#378). The two branches mu and -mu are
        # both exact, so only the caller can choose one.
        raise ValueError(
            f"the exact middle rank r={r} of an odd field (n={n}) under a "
            "symmetric base is unchanged by mu -> -mu, so the zero start "
            "is a stationary point with no way off it and the inverse is "
            "two-branched by symmetry. Pass mu0= (e.g. from the win odds) "
            "to choose the branch.")
    mu = (np.zeros(n) if mu0 is None
          else np.asarray(mu0, dtype=float) - np.mean(mu0))

    is_n = base == "normal"
    kinked = bool(getattr(base_rows, "kinks", ()))
    split_jac = kinked and not kink_split

    def _model(m, jac=True):
        """log of the public column and its Jacobian. The public map is
        rank_probabilities: each ROW normalised by its runner's lattice
        mass, and the target column is then renormalised to unit mass
        here, so the model column is too. The raw column the solver
        used to fit is a different map wherever the lattice mass is not
        one -- a laplace row carries up to 1.6e-3 of rectangle-sum error
        across its kink -- and an exact self-generated target sat 1.5e-3
        (log) outside its range (#267)."""
        pv, Jv, mass = _rank_marginal_with_jacobian(
            m, sd, r, base_rows, points, is_normal=is_n,
            kink_split=kink_split, jacobian=jac and not split_jac,
            with_mass=True)
        u = pv / mass
        tot = u.sum()
        logp = np.log(np.maximum(u / tot, 1e-300))
        if not jac:
            return logp, None
        if split_jac:
            # a kinked base: the Jacobian from the split lattice, the
            # value from the map being inverted (#267)
            Jv = _rank_marginal_with_jacobian(m, sd, r, base_rows, points,
                                              kink_split=True)[1]
        # d log(u_i / sum u), the lattice mass held fixed (its continuum
        # derivative is zero)
        Jlog = (Jv / np.maximum(pv, 1e-300)[:, None]
                - ((Jv / mass[:, None]).sum(axis=0) / tot)[None, :])
        return logp, Jlog

    def _fd_jacobian(m, logp):
        # forward differences of the map itself (values only)
        h = 1e-6
        cols = []
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            cols.append((_model(m + e, jac=False)[0] - logp) / h)
        return np.column_stack(cols)

    logp, Jlog = _model(mu)
    resid = logp - logt
    cost = float(resid @ resid)
    resid_max = float(np.abs(resid).max())
    lam = 1e-6
    iters = 0
    # Polish for a kinked base on the default lattice, as in
    # loc_scale_from_topk_pair: the uniform map's window slides across
    # the kinks, so its derivative is not the continuum one, and LM on
    # the continuum Jacobian crawls or stalls near the map's own
    # discretization error. Then the Jacobian is taken by forward
    # differences of the map (n value-only evaluations an iteration).
    fd_allowed = split_jac
    fd_mode = False
    slow = 0
    for it in range(n_iter):
        iters = it + 1
        if resid_max < tol:
            break
        if fd_mode:
            Jlog = _fd_jacobian(mu, logp)
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
            # a smooth base gets its Jacobian with the value, as before; a
            # kinked one's comes from a second lattice, so only an
            # accepted trial pays for it
            lp_n, J_n = _model(mu_n, jac=not (split_jac or fd_mode))
            r_n = lp_n - logt
            cost_n = float(r_n @ r_n)
            if cost_n < cost:
                slow = slow + 1 if cost_n > 0.25 * cost else 0
                mu, logp, resid, cost = mu_n, lp_n, r_n, cost_n
                resid_max = float(np.abs(resid).max())
                lam = max(lam / 3.0, 1e-10)
                accepted = True
                if J_n is not None:
                    Jlog = J_n
                elif not fd_mode and resid_max >= tol:
                    Jlog = _model(mu)[1]
                break
            lam *= 8.0
        if fd_allowed and not fd_mode and (not accepted or slow >= 2):
            fd_mode = True
            lam = 1e-6
            continue
        if not accepted:
            break
    mu, info = _topk_inverse_return(mu, resid_max < tol, resid_max, iters,
                                    np.zeros(n, dtype=bool), tol, True,
                                    "abilities_from_rank_marginal")
    return mu, resid, info


_TINY = 1e-300


def rank_probabilities(mu, D=None, base="normal", points=513, V=None,
                       qa=None, kink_split=False):
    """The full rank marginals: an (n, n) matrix whose (i, r) entry is
    P(contestant i finishes in position r+1), min-wins.

        P(R_i = r) = int f_i(x) P(N_{-i}(x) = r-1) dx,

    the cavity count pmf against the runner's own density -- winner
    pricing uses the zero-finish coefficient, second place the
    one-finish coefficient, and so on down the field. Rows sum to one
    (every runner takes exactly one rank) and columns sum to one (every
    rank is taken), and both identities are enforced; a lattice that
    fails them is refined as in top_k_probabilities, up to four times
    the requested points, before raising. Cumulative row
    sums reproduce top_k_probabilities. O(n^2 L) per factor node.
    kink_split as in top_k_probabilities (opt-in, default off)."""
    mu = np.asarray(mu, float)
    n = len(mu)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    def one_node(m, points):
        m = _translated_home(np.asarray(m, float), sd * sd)    # (#477)
        lo, hi = _count_window(m, sd, n - 1, base_rows,
                               is_normal=(base == "normal"))
        # a LOCAL name: assigning `points` here would shadow the enclosing
        # parameter and leave it unbound on the compiled branch
        pts = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
        if (base == "normal" and _HAVE_RUST
                and hasattr(_fastrace, "rank_marginals")):
            flat = _fastrace.rank_marginals(
                np.ascontiguousarray(m, dtype=float),
                np.ascontiguousarray(sd, dtype=float), lo, hi, pts)
            return np.asarray(flat, dtype=float).reshape(n, n)
        kinks = _kinks(base_rows, kink_split)
        if kinks:
            x, wq = _split_nodes(lo, hi, pts, m, sd, kinks)
        else:
            x = np.linspace(lo, hi, pts)
            dx = x[1] - x[0]
        z = (x[:, None] - m[None, :]) / sd[None, :]
        S, f, _ = base_rows(z)
        F, S = _cdf_rows(base_rows, z, S)
        dens = f / sd[None, :]
        C = _count_distribution(F, S)
        P = np.empty((n, n))
        for idx, r, q, _ in _loo_pmfs(C, F, S):
            if kinks:
                P[idx, r] = (q * dens[:, idx].T) @ wq
            else:
                P[idx, r] = (q * dens[:, idx].T).sum(axis=1) * dx
        return P

    if V is None:
        def raw_at(pts):
            return one_node(mu, pts)
    else:
        mu = _translated_home(mu, D)                             # (#477)
        Vm, nodes, w = _factor_nodes(V, n, qa, "rank_probabilities", D)

        def raw_at(pts):
            P = np.zeros((n, n))
            for q in range(len(nodes)):
                P += w[q] * one_node(mu + Vm @ nodes[q], pts)
            return P

    # Check what is RETURNED, not what was computed. Row normalisation is
    # not neutral: it makes every row exact and moves the columns, so a
    # matrix that passed the raw check could fail the stated identity
    # afterwards and nothing looked. On a field whose sds span 73x
    # (0.19 to 14.3) at 513 points the column excess grew from 4.5e-3 to
    # 5.4e-3 that way, and the cumulative rows stopped reproducing
    # top_k_probabilities by 2.6e-3 (#203).
    #
    # The columns are not forced. Alternating row/column scaling would
    # make both identities exact and would do it by hiding the
    # under-resolution that caused the defect -- the same field at 2001
    # points has a column error of 3.5e-9, so the quadrature, not the
    # normalisation, is what is wrong. Raising says so; a doubly
    # stochastic answer to a question the lattice could not resolve does
    # not.
    # BOTH checks, before and after. They are independent, and #217
    # mistook them for alternatives: it replaced the raw check with the
    # post one, and row normalisation can ERASE a gross raw defect. On a
    # field whose sds span 232x, the raw matrix was off by 0.249 in a row
    # and 0.199 in a column, and dividing by those same row sums left a
    # column defect of 0.0047 -- under the 5e-3 tolerance, so it returned
    # a false success, while top_k_probabilities rejected the same field
    # at 0.056 (#221). The raw check sees the quadrature; the post check
    # sees what the caller gets; neither implies the other.
    def _defects(M):
        return (float(np.abs(M.sum(axis=1) - 1).max()),
                float(np.abs(M.sum(axis=0) - 1).max()))

    def _reject(where, re_, ce):
        raise RuntimeError(
            f"rank marginals defective {where}: row-sum error {re_:.2e}, "
            f"column-sum error {ce:.2e}. Raise points=, or report this "
            "field.")

    def _checked(P):
        re_raw, ce_raw = _defects(P)
        if not np.isfinite(P).all() or re_raw > 5e-3 or ce_raw > 5e-3:
            _reject("before normalisation", re_raw, ce_raw)
        P = np.clip(P / np.maximum(P.sum(axis=1)[:, None], _TINY), 0.0, 1.0)
        re_out, ce_out = _defects(P)
        if not np.isfinite(P).all() or re_out > 5e-3 or ce_out > 5e-3:
            _reject("in the RETURNED matrix after row normalisation",
                    re_out, ce_out)
        return P

    # a defect is refined on nested 2p - 1 lattices, as top-k (#623): the
    # default 513 and 1025 points both rejected a Laplace field that 2049
    # resolves (#819). A call that passes first time is untouched.
    return _refined_topk(raw_at, None, points, "rank marginals",
                         check=_checked)
