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
                      as_tolerance, rescaled_target)


from .races import _jacobi_sweeps, BASES, _resolution
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


def _reflected(base_rows):
    """Rows of the reflected noise -e: survival F(-z) = 1 - S(-z), density
    f(-z), derivative -f'(-z). Used ONLY to place a window -- the
    subtraction is exact enough for that and for nothing finer."""
    def rows(z):
        S, f, fp = base_rows(-z)
        return 1.0 - S, f, -fp
    # the reflection keeps the base's central scale: without it the
    # bottom-k window budget ignored the declared resolution (#468)
    rows.resolution = _resolution(base_rows)
    return rows


def _topk_independent(mu, sd, k, base_rows, points, delta=1e-12,
                      is_normal=False, upper=False):
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
        x = np.linspace(lo, hi, points)
        dx = x[1] - x[0]
        z = (x[:, None] - mu[None, :]) / sd[None, :]
        S, f, _ = base_rows(z)
        G = np.clip(S, 0.0, 1.0)
        C = _count_distribution(G)
        cdf = _leave_one_out_cdf(C, G, k)
        dens = (f / sd[None, :]).T
        return (dens * cdf).sum(axis=1) * dx
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta,
                           is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
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
    Vm = Vm - Vm.mean(axis=0, keepdims=True)
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
                        qa=None, F=None, W=None):
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
    ordinary calls before it says so."""
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
                                     is_normal=is_normal)
    else:
        Vm, nodes, w = _factor_nodes(V, n, qa, "top_k_probabilities", D,
                                     F=F, W=W)

        def raw_at(pts):
            raw = np.zeros(n)
            for q in range(len(nodes)):
                shift = Vm @ nodes[q]
                raw += w[q] * _topk_independent(mu + shift, sd, k,
                                                base_rows, pts,
                                                is_normal=is_normal)
            return raw
    return _refined_topk(raw_at, k, points, "top-k race")


def _refined_topk(raw_at, k, points, kind):
    """_checked_topk on raw_at(points), refining a lattice that fails it.

    The checks only see a defect; they cannot fix one, and the remedy
    they name -- raise points= -- is mechanical. A field reported to
    raise at the default 513 points priced cleanly at 2049. So a failed
    check doubles the lattice (2p - 1 keeps the old points as a subset)
    and tries again, up to four times the request and never past the
    lattice cap: bounded at about seven ordinary calls, and only ever
    paid by a call that would otherwise have raised. A field that still
    fails raises, naming the points it tried. A call that passes first
    time is untouched."""
    points = int(points)
    cap = min(4 * (points - 1) + 1, _MAX_TOPK_POINTS)
    pts = points
    while True:
        try:
            return _checked_topk(raw_at(pts), k, kind)
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
                           points=513, qa=None):
    """P(X_i among the k largest), for every i, min-wins: the worst k.

    Computed DIRECTLY, as the top-k integral with the count of runners
    above x in place of the count below, so a rare last-place
    probability keeps its relative precision. The complement identity
    P(in the worst k) = 1 - P(in the best n-k) is exact in the continuum
    and useless in floating point at the tail this API faces: it returned
    0 for a 7.7e-24 last place, and lattice noise in the near-one top-k
    value fabricated 3.6e-15 when refined (#365). The slot identity
    sum = k is checked as for top-k."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = _as_depth(k, n)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    if V is None:
        raw = _topk_independent(mu, sd, k, base_rows, points, upper=True)
        return _checked_topk(raw, k, "bottom-k race")
    Vm, nodes, w = _factor_nodes(V, n, qa, "bottom_k_probabilities", D)
    raw = np.zeros(n)
    for q in range(len(nodes)):
        raw += w[q] * _topk_independent(mu + Vm @ nodes[q], sd, k,
                                        base_rows, points, upper=True)
    return _checked_topk(raw, k, "bottom-k race")


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
    k = _as_depth(k, n)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    lo, hi = _count_window(mu, sd, k, base_rows)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
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
    k = _as_depth(k, n)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base
    lo, hi = _count_window(mu, sd, k, base_rows)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
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


def top_k_jacobians(mu, k, D=None, base="normal", points=513, V=None,
                    qa=None):
    """Full (n, n) matrices (dq/dmu, dq/dsigma). The lattice, the
    field rows and the shared count distribution are built ONCE and
    reused across rows -- the row helper rebuilds them per call, which
    at n = 150 spent more time on redundant count programs than on the
    pair terms themselves.

    V= (any rank) is EXACT given the nodes: conditional on the factor
    draw the race is independent and the shift by V f_q leaves d/dmu and
    d/dsigma untouched, so the correlated Jacobians are the same node
    mixture as the forward pass,
    J = sum_q w_q J_ind(mu + V f_q, sigma)."""
    mu = np.asarray(mu, float)
    n = len(mu)
    k = _as_depth(k, n)
    if V is not None:
        Vm, nodes, w = _factor_nodes(V, n, qa, "top_k_jacobians", D)
        Jm = np.zeros((n, n))
        Js = np.zeros((n, n))
        for j in range(len(nodes)):
            Jm_q, Js_q = top_k_jacobians(mu + Vm @ nodes[j], k, D=D,
                                         base=base, points=points)
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
    return max(int(points), _dyadic_points(need))


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
                      is_normal=False):
    """One forward pass returning the raw memberships AND the own
    translation slopes

        dq_i/dmu_i = -int f'(z_i)/sd_i^2 P(N_{-i}(x) <= k-1) dx,

    the cavity cdf the forward pass already holds against the derivative
    of the runner's own density -- one extra weighted sum, no second
    field pass. is_normal=True routes through the compiled kernel when
    fastrace ships it (the python window is computed either way, so the
    quadrature grid is identical)."""
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta,
                           is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    if is_normal and _HAVE_RUST and hasattr(_fastrace, "top_k_slopes"):
        q, sl = _fastrace.top_k_slopes(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), k, lo, hi, points)
        return np.asarray(q, dtype=float), np.asarray(sl, dtype=float)
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
        from .races import _floorable
        target_floor = as_tolerance(target_floor, "target_floor")
        target = _floorable(target, float(k))
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
    target = rescaled_target(target, k)          # scale-safe (#461)
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
                        qa=None, n_iter=80, tol=1e-8, target_floor=None,
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
    after that renormalization raises. V= (any rank) enters by the
    usual node mixture. When k > n/2 the information lives in the
    longshots; inverting the complement (bottom_k_probabilities) is the
    same call at n - k on 1 - q."""
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

    def _forward(m):
        if nodes is None:
            qraw, sl = _topk_with_slopes(m, sd, k, base_rows, points,
                                         is_normal=(base == "normal"))
        else:
            qraw = np.zeros(n)
            sl = np.zeros(n)
            for j in range(len(nodes)):
                qj, sj = _topk_with_slopes(m + Vm @ nodes[j], sd, k,
                                           base_rows, points,
                                           is_normal=(base == "normal"))
                qraw += w[j] * qj
                sl += w[j] * sj
        qhat = _checked_topk(qraw, k, "top-k inversion")
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
                             mu0=None, V=None, return_info=False):
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
    iteration."""
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
                                         tol=1e-3, return_info=True)
    sqr = float(np.sqrt(ridge))

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
    last_grad = np.inf
    accepted = True
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
                mu, sd, r, qh1, qh2, cost = mu_n, sd_n, r_n, q1_n, q2_n, cost_n
                resid_max = float(np.abs(r[:2 * n]).max())
                lam = max(lam / 3.0, 1e-10)
                accepted = True
                break
            lam *= 8.0
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
                               np.zeros(n, dtype=bool), tol, return_info,
                               "loc_scale_from_topk_pair")
    if return_info:
        info = dict(out[1])
        info.update(fit_converged=bool(fit_ok), stationary=stationary,
                    nested=nested)
        return out[0], sd, info
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
                                    return_info=return_info)


def _rank_marginal_with_jacobian(mu, sd, r, base_rows, points,
                                 is_normal=False):
    """P(R_i = r) for every i, plus its full mu-Jacobian:

        dP(R_i = r)/dmu_j = int f_i f_j [ P(N_{-ij} = r-1)
                                        - P(N_{-ij} = r-2) ] dx,  j != i

    (the r = 1 case drops the second term and recovers the win
    Jacobian), with the diagonal from translation invariance.
    is_normal=True routes through the compiled kernel on the same
    python-computed window."""
    n = len(mu)
    lo, hi = _count_window(mu, sd, n - 1, base_rows, is_normal=is_normal)
    points = _resolved_points(lo, hi, sd, points, _resolution(base_rows))
    if (is_normal and _HAVE_RUST
            and hasattr(_fastrace, "rank_marginal_jacobian")):
        p, jac = _fastrace.rank_marginal_jacobian(
            np.ascontiguousarray(mu, dtype=float),
            np.ascontiguousarray(sd, dtype=float), int(r), lo, hi, points)
        return (np.asarray(p, dtype=float),
                np.asarray(jac, dtype=float).reshape(n, n))
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
        if r <= n - 1:
            hi_pair = _pair_pmf_at(Qi, F, i, r)      # P(N_{-ij} = r-1)
            row = (hi_pair * dens.T
                   * dens[:, i][None, :]).sum(axis=1) * dx
        else:
            # only n-2 others exist: P(N_{-ij} = n-1) is identically
            # zero, and computing it by deconvolution injects clamped
            # junk that is hypersensitive to the window edge
            row = np.zeros(n)
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
    phat, J = _rank_marginal_with_jacobian(mu, sd, r, base_rows, points,
                                           is_normal=is_n)
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
                                                    base_rows, points,
                                                    is_normal=is_n)
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


_TINY = 1e-300


def rank_probabilities(mu, D=None, base="normal", points=513, V=None,
                       qa=None):
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
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    # scalar, length-n, and no
    # negative or zero variance: a bare asarray made a scalar 0-d, and
    # the compiled kernel then indexed past it and PANICKED, while a
    # wrong length broadcast into a plausible wrong answer (#254)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    def one_node(m):
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
        x = np.linspace(lo, hi, pts)
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
        Vm, nodes, w = _factor_nodes(V, n, qa, "rank_probabilities", D)
        P = np.zeros((n, n))
        for q in range(len(nodes)):
            P += w[q] * one_node(mu + Vm @ nodes[q])

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

    re_raw, ce_raw = _defects(P)
    if not np.isfinite(P).all() or re_raw > 5e-3 or ce_raw > 5e-3:
        _reject("before normalisation", re_raw, ce_raw)
    P = np.clip(P / np.maximum(P.sum(axis=1)[:, None], _TINY), 0.0, 1.0)
    re_out, ce_out = _defects(P)
    if not np.isfinite(P).all() or re_out > 5e-3 or ce_out > 5e-3:
        _reject("in the RETURNED matrix after row normalisation",
                re_out, ce_out)
    return P
