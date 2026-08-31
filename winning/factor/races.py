"""The general race: one API, distributions and correlation as parameters.

    race_probabilities(mu)                          classic independent race
    race_probabilities(mu, V=V, D=D)                factor probit (Gaussian)
    race_probabilities(mu, base="gumbel")           Luce / softmax, exactly
    race_probabilities(mu, V=V, base="gumbel")      correlated Luce
    race_probabilities(mu, base="logistic")         logistic noise
    race_probabilities(mu, base="laplace")          robust double-exponential
    race_probabilities(mu, base=student_base(4))    fat tails, unit variance
    race_probabilities(mu, base=skew_normal_base(2))  the heritage family
    race_probabilities(mu, base=failure_base(0.1))  performance + DNF lump
    race_probabilities(mu, base=my_base)            anything standardized

Min-wins convention throughout. A base is a callable z -> (S, f, fp)
giving survival, density and density derivative of a MEAN-ZERO,
UNIT-VARIANCE law (standardization keeps noise family separate from
noise scale). Zero factors is literally the one-node quadrature, so the
independent race is not a separate code path.

WHAT A DISTRIBUTION IS HERE, and why it matters: a formula, not a
vector. The engine never tabulates a distribution -- it evaluates
S_i and f_i EXACTLY at the lattice points via the base callable, and
the lattice discretizes only the one-dimensional win integral, whose
smooth integrand makes that quadrature spectrally accurate (measured:
machine precision by 33 points on asymmetric near-tied fields). Ties
therefore have probability zero by construction and need no
bookkeeping. The buy-in is that the law must BE a formula: an
empirical atom set (finish times at recorded precision, integer
scores, histogram data) has no native representation here and must
either be smoothed into a callable -- erasing any real dead-heat mass
-- or go to winning.classic, whose primitive is the opposite (the atom
vector IS the distribution, and the multiplicity calculus prices its
genuine ties exactly). The conversion error lives at that format
boundary and only there: sampling a continuous law onto classic's
atoms costs ~1e-3 at 129 points where this engine's quadrature costs
1e-16 (pinned in the tests), and smoothing real atoms into a formula
deletes model mass no refinement recovers.

Promoted from research/experiments/exp14_boundaries/run_boundaries.py,
where the general engine was exercised by the paper's substitution
experiments (the Gumbel base's zero-loading case equals softmax to
2.8e-17 there).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from .core import hermite_nodes

try:                                       # compiled kernels (rust/fastrace)
    import fastrace as _fastrace
    _RUST_OK = hasattr(_fastrace, "forward_and_slopes")
    _HAVE_RUST = _RUST_OK and __import__("os").environ.get(
        "WINNING_PURE", "").strip() in ("", "0")
except ImportError:
    _fastrace = None
    _RUST_OK = False
    _HAVE_RUST = False

_EULER = 0.5772156649015329


def _normal(z):
    S = np.maximum(1.0 - ndtr(z), 1e-300)
    f = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return S, f, -z * f


def _gumbel_min(z):
    c = np.pi / np.sqrt(6.0)
    u = np.minimum(z * c - _EULER, 30.0)
    eu = np.exp(u)
    S = np.maximum(np.exp(-eu), 1e-300)
    f = c * eu * S
    return S, f, c * c * eu * S * (1.0 - eu)


def _logistic(z):
    # standardized logistic: scale s = sqrt(3)/pi gives unit variance
    c = np.pi / np.sqrt(3.0)
    u = np.clip(c * z, -700.0, 700.0)
    S = 1.0 / (1.0 + np.exp(u))
    f = c * S * (1.0 - S)
    # S' = -f, so f' = c S'(1-2S) = -c f (1-2S) (a sign the numeric
    # audit caught in the first draft)
    return np.maximum(S, 1e-300), f, -c * f * (1.0 - 2.0 * S)


def _laplace(z):
    # standardized Laplace: scale b = 1/sqrt(2) gives unit variance
    b = 1.0 / np.sqrt(2.0)
    az = np.abs(z)
    f = np.exp(-az / b) / (2.0 * b)
    S = np.where(z < 0, 1.0 - 0.5 * np.exp(z / b), 0.5 * np.exp(-z / b))
    return np.maximum(S, 1e-300), f, -np.sign(z) * f / b


BASES = {"normal": _normal, "gumbel": _gumbel_min,
         "logistic": _logistic, "laplace": _laplace}
_SPANS = {"normal": (8.0, 8.0), "gumbel": (22.0, 8.0),   # (left, right) tails
          "logistic": (16.0, 16.0), "laplace": (18.0, 18.0)}


def student_base(nu):
    """Standardized Student-t base (unit variance; needs nu > 2): fat
    tails for performances with occasional wild days. Returns a callable
    for base=; the tail span widens with falling nu automatically."""
    nu = float(nu)
    if nu <= 2.0:
        raise ValueError("student_base needs nu > 2 for unit variance")
    from scipy.stats import t as _t
    s = np.sqrt(nu / (nu - 2.0))            # raw x = s * z

    def _student(z):
        x = s * np.asarray(z, dtype=float)
        S = np.maximum(_t.sf(x, nu), 1e-300)
        f = _t.pdf(x, nu) * s
        fp = f * (-(nu + 1.0) * x / (nu + x * x)) * s
        return S, f, fp

    # polynomial tails: place the window edge at the actual 1e-7
    # quantile (heavy tails on an equispaced lattice are intrinsically
    # expensive; below nu ~ 3 expect wide windows and budget points
    # accordingly)
    edge = float(_t.isf(1e-7, nu)) / s
    _student.span = (max(12.0, edge), max(12.0, edge))
    return _student


def skew_normal_base(a):
    """Standardized skew-normal base (the classic engine's heritage
    family), shape a: mean zero, unit variance. Returns a callable for
    base=."""
    a = float(a)
    from scipy.stats import skewnorm as _sn
    delta = a / np.sqrt(1.0 + a * a)
    m = delta * np.sqrt(2.0 / np.pi)
    sd = np.sqrt(1.0 - 2.0 * delta * delta / np.pi)

    def _skew(z):
        x = m + sd * np.asarray(z, dtype=float)
        S = np.maximum(_sn.sf(x, a), 1e-300)
        phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        Phi_ax = ndtr(a * x)
        f = 2.0 * phi * Phi_ax * sd
        fp = 2.0 * (-x * phi * Phi_ax
                    + a * phi * np.exp(-0.5 * a * a * x * x)
                    / np.sqrt(2.0 * np.pi)) * sd * sd
        return S, f, fp

    _skew.span = (10.0, 10.0)
    return _skew


def _setup(mu, V, D, F, W, base):
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    D = np.ones(n) if D is None else np.asarray(D, dtype=float)
    if V is None:
        V = np.zeros((n, 1))
        F, W = np.zeros((1, 1)), np.ones(1)
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        # Gauge-fix the loadings: V -> PV, subtracting each factor's mean
        # loading across contestants. A common loading column c adds the
        # same c'f to every performance and cannot move an argmin, so the
        # centered V prices the identical race -- and every downstream
        # decision (node family, node order, lattice window) becomes
        # invariant under V -> V + 1c', which the uncentered matrix is
        # not. The eighth review's counterexample: rows (2.95,0) x2,
        # (2.95,0.4), (-2.95,0) at D = 1 have max raw row norm 2.977,
        # under the escalation threshold, while the centered rows reach
        # 4.43 -- the race is genuinely sharp, the raw statistic missed
        # it, and the shipped answer carried TV 9.5e-3.
        V = V - V.mean(axis=0)
        if F is None or W is None:
            # adaptive order: when idiosyncratic noise is small relative
            # to the loadings, the conditional race is nearly
            # deterministic and the factor integrand is nearly a step --
            # a fixed 15-node rule silently loses 2-5% (found by the
            # fuzz battery, research/fuzz). Scale the order with the
            # sharpness ratio; identical rule in the R port.
            #
            # The ratio itself: what decides a race is loading
            # DIFFERENCES, and the intrinsic pairwise statistic
            # s_delta = max_{ij} |V_i - V_j| / sqrt(D_i + D_j) is
            # O(n^2 k). The dispatch uses the linear-time bound
            # sqrt(2) * max_i |(PV)_i| / sqrt(D_i) >= s_delta
            # (triangle inequality on centered rows), which cannot miss
            # a sharp pair; its false positives cost nodes, not error.
            sharp = float(np.sqrt(2.0)
                          * np.max(np.sqrt((V ** 2).sum(axis=1))
                                   / np.sqrt(np.maximum(D, 1e-300))))
            r = V.shape[1]
            if r >= 2 and sharp > 3.0:
                # past this sharpness the integrand is a near-step in
                # factor space and Gauss-Hermite converges slowly at ANY
                # order (measured: the 25-node rule still loses ~1e-2 TV
                # at sharp ~ 10, while scrambled Sobol reaches the QMC
                # reference's own noise). Escalate the FAMILY, not the
                # order. See docs/latex_src/general_inversion/break.py,
                # section H; identical rule in the R port (Halton there,
                # to stay dependency-free).
                from .core import qmc_nodes
                F, W = qmc_nodes(r, m=13)
            elif r == 1 and np.ceil(8.0 * sharp) > 201:
                # rank-1 extreme sharpness (gap-stress find): Gauss-
                # Hermite's clustered nodes and wild weights are the
                # wrong family for a near-step integrand -- at sharp 100
                # the 201-node GH rule carried TV 0.65 where an
                # equal-weight midpoint-quantile grid of the SAME size
                # carried 6e-3. Scale the grid with sharpness, capped.
                Q = int(min(np.ceil(8.0 * sharp), 4001))
                u = (np.arange(Q) + 0.5) / Q
                F = ndtri(u)[:, None]
                W = np.full(Q, 1.0 / Q)
            elif 15 ** r > 100_000:
                # high-rank ecology footgun (bandits, 120 horses x 12
                # stable factors): the tensor grid is 15^r nodes BEFORE
                # pruning -- 2.6e9 at r=8, 944 TiB at r=12, a hard kill.
                # Past a 1e5-node tensor budget the rule escalates to
                # scrambled Sobol, as the likelihood module always did.
                from .core import qmc_nodes
                F, W = qmc_nodes(r, m=13)
            else:
                cap = 201 if r == 1 else (41 if r == 2 else 15)
                Q = int(np.clip(np.ceil(8.0 * sharp), 15, cap))
                F, W = hermite_nodes(r, Q=Q)
    fn = base if callable(base) else BASES[base]
    left, right = _SPANS.get(base, (12.0, 12.0)) if not callable(base) \
        else getattr(base, "span", (12.0, 12.0))
    return mu, V, D, np.asarray(F, float), np.asarray(W, float), fn, left, right




def _bulk_window(M_all, sd, points, delta, fn=None):
    """Lattice over the WINNER distribution's bulk, not the ability span.

    G(x) = 1 - prod_j S_j(x) is the winner cdf under the min-race (averaged
    over factor nodes via the extreme conditional locations, a conservative
    envelope); [G^-1(delta), G^-1(1-delta)] carries all but 2*delta of every
    runner's win integrand -- a hopeless runner only wins by running a
    winner-class time. Measured (research/lattice_window): 33 points here
    beat 513 on the ability-span window, whose own truncation floors its
    accuracy near 6e-11. Bisection on a monotone function; cost negligible.

    The envelope uses the CALLER'S base survival (fn), not the normal one.
    An earlier version hardcoded the normal survival for every base, which
    clipped polynomial tails: a Student-t(2.5) race at n=40 lost 5e-3 of
    total variation against the span window (fifth review). With the base
    supplied, the window adapts to the tail it is actually integrating.

    Two things make the quantile claim literally true (sixth review).
    The interval is BRACKETED before it is bisected -- nine sigma holds
    the delta-quantiles of a Gaussian race but not of an arbitrary
    polynomial tail, and bisecting an unbracketed interval converges to
    its endpoint while looking successful. And delta is treated as a
    request, not a guarantee: a t(4) race's 1e-12 winner quantile sits
    1456 standard units out, so honoring it at 257 points would spread
    the lattice to a spacing of 5.7 and lose more to discretization than
    truncation ever cost. When the requested window will not fit the
    point budget, delta is relaxed by factors of a hundred until it
    does, and the achieved delta is reported. The window is then exactly
    what it claims to be at the delta it names.
    """
    S_of = (lambda z: np.maximum(1.0 - _ndtr_local(z), 1e-300)) if fn is None \
        else (lambda z: np.maximum(fn(z)[0], 1e-300))
    mu_lo = M_all.min(axis=0)
    mu_hi = M_all.max(axis=0)
    s = sd

    def G(x):
        # envelope: winner cdf using each runner's most favourable node
        z = (x - mu_lo) / s
        logS = np.log(S_of(z))
        return 1.0 - np.exp(logS.sum())

    def H(x):
        # right edge from the LEAST favourable nodes: no runner's density cut
        z = (x - mu_hi) / s
        logS = np.log(S_of(z))
        return 1.0 - np.exp(logS.sum())

    def _bracket(x0, step0, ok, sign):
        """Push x0 outward until ok(x0), doubling the step (sixth review).

        Nine sigma brackets the delta-quantiles of a Gaussian race and of
        every shipped base, but a caller's own polynomial tail can leave
        G^-1(delta) outside it, and bisection on an unbracketed interval
        converges to the endpoint while looking like it succeeded. So
        bracket first, and say so when even a capped expansion fails
        rather than returning a window whose quantile claim is false.
        """
        step = step0
        for _ in range(60):
            if ok(x0):
                return x0
            x0 += sign * step
            step *= 2.0
        import warnings
        warnings.warn(
            "bulk window could not bracket the requested quantile after 60 "
            "doublings; this base's tail is heavier than the lattice can "
            "span, and the window is truncated rather than quantile-exact.",
            RuntimeWarning, stacklevel=3)
        return x0

    # safety margin beyond the bisected bulk. The envelope now uses the
    # true base, so the pad only absorbs the node-extreme approximation;
    # heavy-tailed bases additionally widen it by their own declared span.
    pad = 2.0 * float(s.max())
    if fn is not None:
        span = getattr(fn, "span", None)
        if span is not None:
            pad = max(pad, 0.25 * float(max(span)) * float(s.max()))

    def window_at(d):
        step0 = max(9.0 * float(s.max()), 1e-12)
        lo0 = _bracket(float(mu_lo.min() - 9.0 * s.max()), step0,
                       lambda x: G(x) <= d, -1.0)
        hi0 = _bracket(float(mu_hi.max() + 9.0 * s.max()), step0,
                       lambda x: H(x) >= 1.0 - d, +1.0)
        a, b = lo0, hi0
        for _ in range(80):
            m = 0.5 * (a + b)
            if G(m) < d:
                a = m
            else:
                b = m
        xlo = a
        a, b = xlo, hi0
        for _ in range(80):
            m = 0.5 * (a + b)
            if H(m) < 1.0 - d:
                a = m
            else:
                b = m
        return xlo - pad, b + pad

    # affordable width: keep the spacing under half the tightest
    # performance sd, the same resolution target the sharpness refinement
    # uses, so relaxation fires only when the tail would genuinely
    # outrun the lattice rather than whenever the budget is modest
    budget = 0.5 * float(s.min()) * max(points - 1, 1)
    d = float(delta)
    lo, hi = window_at(d)
    while hi - lo > budget and d < 1e-4:
        d = min(d * 100.0, 1e-4)
        lo, hi = window_at(d)
    if d > delta:
        import warnings
        warnings.warn(
            f"bulk window relaxed delta from {delta:.0e} to {d:.0e}: this "
            f"base's tail puts the requested quantile further out than "
            f"{points} points can resolve, so the window is "
            f"{hi - lo:.3g} units wide at the relaxed delta and exact "
            "there. Raise points= to tighten it.",
            RuntimeWarning, stacklevel=3)
    return np.linspace(lo, hi, points)


def _ndtr_local(z):
    from scipy.special import ndtr
    return ndtr(z)


def forward_grid(M_all, sd, V, fn, left, right, points, window="bulk",
                 delta=1e-12):
    """THE lattice: window plus any sharpness refinement, in one place.

    The forward map and its derivatives have to integrate on the SAME
    grid or the derivative is not the derivative of the map that was
    evaluated (sixth review). Every caller that differentiates the race
    -- race_jacobian, race_jacobian_row -- builds its lattice here, so
    the only remaining difference between them and race_probabilities is
    the integrand.

    Returns (x, points); points may exceed the request after refinement.
    """
    sd = np.asarray(sd, dtype=float)
    if window == "bulk":
        x = _bulk_window(M_all, sd, points, delta, fn)
    else:
        x = np.linspace(M_all.min() - left * sd.max(),
                        M_all.max() + right * sd.max(), points)
    dx = x[1] - x[0]
    smin = float(sd.min())
    sharp_here = float(np.max(np.sqrt((np.asarray(V, float) ** 2).sum(axis=1)))
                       / max(smin, 1e-300))
    if sharp_here > 25.0 and dx > 0.5 * smin:
        # extreme-sharpness lattice refinement (gap-stress find): with
        # near-deterministic conditional races (same sharp > 25 regime
        # as the node escalation -- an explicit coarse points= budget on
        # ordinary fields is honored untouched) the winner density is
        # narrower than the lattice spacing and the integral is
        # underresolved (TV 5e-2 at conditional sd 1e-3 on the default
        # 257 points; 6e-4 once resolved). Refine to ~2 points per
        # conditional sd, capped; warn when the cap still leaves the
        # lattice coarse.
        need = int(np.ceil((x[-1] - x[0]) / (0.5 * smin))) + 1
        pts2 = min(need, 8193)
        if pts2 > points:
            x = np.linspace(x[0], x[-1], pts2)
            points = pts2
        if need > 8193:
            import warnings
            warnings.warn(
                "conditional races are sharper than the lattice can "
                f"resolve even at 8193 points (min sd {smin:.1e} over a "
                f"window of {x[-1]-x[0]:.3g}); results may carry "
                "percent-level error. This is the near-deterministic "
                "regime; consider larger idiosyncratic variances or "
                "simulation.", RuntimeWarning, stacklevel=2)
    # A correctly bracketed window can still be unusable: a polynomial
    # tail pushes the delta-quantile so far out that the points are
    # spread too thin to resolve the bulk. That used to hide behind a
    # truncated window; now that the window is honest, say so.
    if (x[-1] - x[0]) / max(len(x) - 1, 1) > 0.5 * smin:
        import warnings
        warnings.warn(
            f"lattice spacing {(x[-1]-x[0])/max(len(x)-1,1):.3g} exceeds "
            f"half the smallest performance sd ({smin:.3g}): this base's "
            "tail forced a window wider than the point budget can "
            "resolve. Raise points=, raise delta=, or declare a span on "
            "the base.", RuntimeWarning, stacklevel=2)
    return x, points


def race_probabilities(mu, V=None, D=None, F=None, W=None, base="normal",
                       points=257, temperature=0.0, return_slopes=False,
                       structure=None, window="bulk", delta=1e-12, cov=None):
    """Win probabilities of the general race, all N in one field pass.

    Pass `structure=` (Independent/Factor/Blocks/Nested/Tree from
    winning.factor.structures) to describe the covariance declaratively --
    one race, five grammars; V=/D= remain as sugar for the factor case.
    Pass `cov=` (a dense covariance or correlation matrix) to have it
    fitted to the grammar first via winning.factor.core.fit_covariance
    (approximate: the fit residual is the price of density; see the
    paper's dense-Sigma section for measured accuracy by ensemble).
    temperature > 0 returns the softmin expectation E[softmin(X/tau)],
    computed exactly as the hard race with each base convolved with the
    tau-scaled min-Gumbel kernel."""
    if cov is not None:
        if structure is not None or V is not None or D is not None:
            raise ValueError("cov= replaces structure=/V=/D=; pass one only")
        from .core import fit_covariance
        V, D, F, W, report = fit_covariance(cov, return_report=True)
        if report.get("contrast_residual_max", 0.0) > 0.05:
            import warnings
            warnings.warn(
                "cov= carries a nearly singular contrast the fit did not "
                "hold: the worst pairwise difference variance is off by "
                f"{report['contrast_residual_max']:.2f} of its own size. "
                "Global residual norms cannot see this (choice "
                "probabilities become infinitely sensitive as a contrast "
                "variance approaches zero), so head-to-head probabilities "
                "between the affected pair may be badly wrong even though "
                "the covariance residual looks small.",
                RuntimeWarning, stacklevel=2)
        if report["projected_residual_max"] > 0.05:
            import warnings
            warnings.warn(
                "cov= is imperfectly served by the grammar fit (worst "
                "choice-relevant residual entry "
                f"{report['projected_residual_max']:.2f} of the average "
                "variance; short-length-scale/locality covariances are the "
                "known hard family). Probabilities may carry percent-level "
                "bias; see the paper's dense-covariance section.",
                RuntimeWarning, stacklevel=2)
        elif report["rank"] > 12 and report["sharpness"] > 5:
            import warnings
            warnings.warn(
                "cov= fits well but needs a high-rank, sharp factor "
                f"integral (rank {report['rank']}, sharpness "
                f"{report['sharpness']:.0f}); the default node budget may "
                "leave percent-level quadrature error. Pass more nodes "
                "(fit_covariance(..., nodes_log2=14)) or price by "
                "simulation for near-singular smooth covariances.",
                RuntimeWarning, stacklevel=2)
    if structure is not None:
        from .structures import dispatch_probabilities
        return dispatch_probabilities(mu, structure, base=base,
                                      temperature=temperature,
                                      return_slopes=return_slopes)
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    if temperature and temperature > 0:
        return _race_tempered(mu, V, D, F, W, fn, left, right,
                              float(temperature), points, return_slopes)
    sd = np.sqrt(D)
    n = len(mu)
    if _HAVE_RUST and base == "normal" and n * len(F) > 2e7:
        # at scale, materializing the Q x n conditional-means matrix (only
        # ever used for the window) costs gigabytes and dominates runtime;
        # per-runner extremes over the node set suffice. For GH tensor
        # grids the box hull IS the exact node-set extreme (every sign
        # corner is a node); for Sobol it is a conservative superset.
        fabs = np.abs(F).max(axis=0)
        spread = np.abs(V) @ fabs
        M_lo = (mu - spread)[None, :]
        M_hi = (mu + spread)[None, :]
        if window == "bulk":
            x = _bulk_window(np.vstack([M_lo, M_hi]), sd, points, delta, fn)
        else:
            x = np.linspace(M_lo.min() - left * sd.max(),
                            M_hi.max() + right * sd.max(), points)
        dx = x[1] - x[0]
        p, sl, total = _fastrace.forward_and_slopes(
            np.ascontiguousarray(mu), np.ascontiguousarray(V),
            np.ascontiguousarray(D), np.ascontiguousarray(F),
            np.ascontiguousarray(W), points, float(x[0]), float(x[-1]))
        if return_slopes:
            return np.asarray(p), np.asarray(sl) / total
        return np.asarray(p)
    M_all = mu[None, :] + F @ V.T
    x, points = forward_grid(M_all, sd, V, fn, left, right, points,
                             window=window, delta=delta)
    dx = x[1] - x[0]
    if _HAVE_RUST and base == "normal":
        try:
            p, sl, total = _fastrace.forward_and_slopes(
                np.ascontiguousarray(mu), np.ascontiguousarray(V),
                np.ascontiguousarray(D), np.ascontiguousarray(F),
                np.ascontiguousarray(W), points,
                float(x[0]), float(x[-1]))
            if return_slopes:
                return np.asarray(p), np.asarray(sl) / total
            return np.asarray(p)
        except TypeError:
            pass       # older fastrace without window arguments: numpy path
    p = np.zeros(n)
    slope = np.zeros(n)
    chunk = max(1, int(5e6 / (n * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S, f, fp = fn(z)
        f = f / sd[None, :, None]
        logS = np.log(S)
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)
        slope += Wc @ (np.sum(-fp / sd[None, :, None] ** 2 * rest, axis=2) * dx)
    total = p.sum()
    if return_slopes:
        return p / total, slope / total
    return p / total


def abilities_from_race(p, V=None, D=None, F=None, W=None, base="normal",
                        points=257, temperature=0.0, n_iter=60, tol=1e-8,
                        structure=None, cov=None, target_floor=None,
                        return_info=False):
    """Invert the general race: mean-zero mu with race_probabilities(mu) = p.

    Accepts the same covariance descriptions as the forward call: V=/D=
    factor sugar, structure= for any grammar member, cov= for a dense
    matrix (fitted first, so the inverse is of the fitted race). For
    block/nested/tree structures the update below keeps the exact forward
    map and preconditions with the own-slope of the variance-matched
    independent race (one extra O(nL) pass per iteration).

    Contract on the target (fifth review): zero and negative entries
    RAISE, because a zero share has no finite inverse. target_floor=
    opts into flooring small entries, and the result is then a one-sided
    bound on the floored contrasts, not their inverse; the returned info
    dict (return_info=True) reports which entries were floored, whether
    the iteration converged, and the achieved residual. Non-convergence
    warns rather than returning silently."""
    if cov is not None:
        if structure is not None or V is not None or D is not None:
            raise ValueError("cov= replaces structure=/V=/D=; pass one only")
        from .core import fit_covariance
        V, D, F, W = fit_covariance(cov)
    if structure is not None:
        from .structures import Factor, Independent
        if isinstance(structure, Independent):
            structure, D = None, np.asarray(structure.D, float)
        elif isinstance(structure, Factor):
            V = np.asarray(structure.V, float)
            D = np.asarray(structure.D, float)
            structure = None
    target = np.asarray(p, dtype=float)
    if target_floor is not None:
        if not target_floor > 0:
            raise ValueError("target_floor must be positive")
        floored = target < target_floor
        target = np.maximum(target, target_floor)
    else:
        floored = np.zeros(len(target), dtype=bool)
        if np.any(target <= 0):
            raise ValueError(
                "all target probabilities must be positive: a zero share "
                "has no finite inverse (the supremum is approached as that "
                "contrast diverges). Pass target_floor= to floor small "
                "entries deliberately and read the result as a one-sided "
                "bound on the floored contrasts, or supply a pseudocount "
                "upstream.")
    target = target / target.sum()
    if structure is not None:
        # the grammar path takes the SAME contract (sixth review): the
        # target was validated or floored above, and the iteration reports
        # convergence through the same tail rather than returning silently
        mu, converged, resid_max, iters = _abilities_from_structure(
            target, structure, points=points, n_iter=max(n_iter, 120),
            tol=tol)
        return _inverse_return(mu, converged, resid_max, iters, floored,
                               tol, return_info)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    # N = 2: the photo-finish graph K_2 is bipartite, so the undamped
    # Jacobi update on the mean-zero quotient has eigenvalue 1 - 2 = -1,
    # a local two-cycle. Fixed damping 0.7 restores contraction.
    alpha = 1.0 if len(target) > 2 else 0.7
    resid_max = np.inf
    iters = 0
    for it in range(n_iter):
        iters = it + 1
        phat, sl = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base,
                                      points=points, temperature=temperature,
                                      return_slopes=True)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        resid_max = float(np.abs(resid).max())
        if resid_max < tol:
            break
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        # residual-proportional step cap: a near-certain winner has
        # residual AND own-slope both vanishing, and their ratio is an
        # O(0.1) noise step that recentering sloshes into every other
        # coordinate (measured: heavy-favorite targets in the 1e-4..1e-8
        # window stalled at 200 iterations; capped, they converge in
        # 4-6). No coordinate moves much further than its own residual
        # warrants.
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return _inverse_return(mu, resid_max < tol, resid_max, iters, floored,
                           tol, return_info)


def _inverse_return(mu, converged, resid_max, iters, floored, tol,
                    return_info):
    """One exit for every inversion path, so the contract cannot differ
    by grammar: warn on non-convergence unless the caller asked for the
    diagnostics, and report the iteration actually reached."""
    if not converged and not return_info:
        import warnings
        warnings.warn(
            f"abilities_from_race did not converge: max |log residual| "
            f"{resid_max:.2e} after {iters} iterations (tol {tol:.0e}). "
            "Pass return_info=True for the residual and iteration count "
            "instead of this warning.",
            RuntimeWarning, stacklevel=2)
    if return_info:
        return mu, {"converged": bool(converged),
                    "max_log_residual": float(resid_max),
                    "iterations": int(iters), "floored": floored}
    return mu


def _abilities_from_structure(target, structure, points=257, n_iter=120,
                              tol=1e-8):
    """Generic grammar inversion: exact forward map through the dispatch,
    damped log-residual fixed point preconditioned by the own-slope of the
    independent race at matched total variances. Damping backtracks (halves)
    whenever the residual fails to shrink, so contraction is monitored, not
    assumed.

    Takes an already validated and normalized target (the caller owns the
    contract) and returns (mu, converged, max_log_residual, iterations)."""
    from .structures import dispatch_probabilities, structure_variances
    target = np.asarray(target, dtype=float)
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    totvar = structure_variances(structure)
    alpha, last = 0.7, np.inf
    err, iters = np.inf, 0
    for it in range(n_iter):
        iters = it + 1
        phat = dispatch_probabilities(mu, structure, points=points)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        err = np.abs(resid).max()
        if err < tol:
            break
        if err > last:
            alpha = max(alpha * 0.5, 0.05)
        last = err
        ps, ss = race_probabilities(mu, D=totvar, points=points,
                                    return_slopes=True)
        dlogp = np.minimum(ss / np.maximum(ps, 1e-300), -1e-6)
        lim = np.minimum(2.0, 10.0 * np.abs(resid))
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return mu, bool(err < tol), float(err), iters


# ---------------------------------------------------------------------------
# Finite temperature: E[softmin(X/tau)] as a hard race with a convolved base.
#
# By the Gumbel-argmin identity, E[softmin(X/tau)_i] = P(i = argmin_j
# {X_j + tau g_j}) with g iid standard min-Gumbel (verified against common-
# draw Monte Carlo; see the softmax-thurstone notes). So temperature > 0
# just convolves each runner's noise with the tau-Gumbel kernel and runs
# the identical shared-field engine. tau -> 0 is the hard race; tau -> inf
# flattens toward uniform. Temperature is not identifiable from a single
# race, so inversion treats it as fixed.
# ---------------------------------------------------------------------------


def _tempered_curves(sd_i, tau, fn, left, right, m=4001):
    """Survival, density, density-derivative of sd*e + tau*g on a grid."""
    lo = -left * sd_i - 30.0 * tau
    hi = right * sd_i + 8.0 * tau
    u = np.linspace(lo, hi, m)
    du = u[1] - u[0]
    _, f_base, _ = fn(u[None, None, :] / sd_i)
    f_base = f_base[0, 0] / sd_i
    v = np.exp(np.minimum(u / tau, 30.0))
    f_gum = v * np.exp(-v) / tau                       # min-Gumbel, scale tau
    f_eta = np.convolve(f_base, f_gum, mode="same") * du
    f_eta = np.maximum(f_eta, 0.0)
    total = f_eta.sum() * du
    f_eta /= total
    cdf = np.cumsum(f_eta) * du
    S = np.maximum(1.0 - cdf, 1e-300)
    fp = np.gradient(f_eta, du)
    return u, S, f_eta, fp


def _race_tempered(mu, V, D, F, W, fn, left, right, temperature, points,
                   return_slopes):
    sd = np.sqrt(D)
    n = len(mu)
    curves = [_tempered_curves(sd[i], temperature, fn, left, right)
              for i in range(n)]
    M_all = mu[None, :] + F @ V.T
    pad_lo = max(left * sd.max(), 30.0 * temperature + left * sd.max())
    pad_hi = right * sd.max() + 8.0 * temperature
    x = np.linspace(M_all.min() - pad_lo, M_all.max() + pad_hi, points)
    dx = x[1] - x[0]
    p = np.zeros(n)
    slope = np.zeros(n)
    chunk = max(1, int(5e6 / (n * points)))
    S = np.empty((min(chunk, len(F)), n, points))
    f = np.empty_like(S)
    fp = np.empty_like(S)
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        nc = M.shape[0]
        for i in range(n):
            u, Sg, fg, fpg = curves[i]
            args = (x[None, :] - M[:, i, None]).ravel()
            S[:nc, i, :] = np.interp(args, u, Sg, left=1.0,
                                     right=1e-300).reshape(nc, points)
            f[:nc, i, :] = np.interp(args, u, fg, left=0.0,
                                     right=0.0).reshape(nc, points)
            fp[:nc, i, :] = np.interp(args, u, fpg, left=0.0,
                                      right=0.0).reshape(nc, points)
        logS = np.log(np.maximum(S[:nc], 1e-300))
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f[:nc] * rest, axis=2) * dx)
        slope += Wc @ (np.sum(-fp[:nc] * rest, axis=2) * dx)
    total = p.sum()
    if return_slopes:
        return p / total, slope / total
    return p / total


def tie_densities(mu, V=None, D=None, F=None, W=None, base="normal",
                  points=501):
    """Pairwise photo-finish densities w[i][j]: the tie-for-the-win rate
    between i and j with the field behind — the graph-Laplacian weights,
    and the conductances of the circuit interpretation. O(Q N^2 L)."""
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    x = np.linspace(M_all.min() - left * sd.max(),
                    M_all.max() + right * sd.max(), points)
    dx = x[1] - x[0]
    w = np.zeros((n, n))
    for c in range(len(F)):
        z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
        S, f, _ = fn(z)
        f = f / sd[:, None]
        logS = np.log(S)
        logSfield = logS.sum(0)
        for i in range(n):
            rest = np.exp(np.clip(logSfield[None, :] - logS[i] - logS,
                                  -745.0, 0.0))
            w[i] += W[c] * (f[i] * f * rest).sum(1) * dx
    np.fill_diagonal(w, 0.0)
    return 0.5 * (w + w.T)          # symmetric by theory; average numerics


def removal_shares(mu, V=None, D=None, F=None, W=None, base="normal",
                   points=501, mass_tol=1e-3):
    """The full single-removal ensemble q[i][j] = P(j wins | i removed),
    every row from the same shared field by dividing i's survival back
    out. Rows sum to one. O(Q N^2 L) -- note the ensemble's OUTPUT alone
    is Omega(n^2); a single requested row costs one field pass.

    Grid discipline (fourth review): the removal identity is exact in
    the continuum but the lattice must COVER the post-removal winner,
    whose bulk is the first-or-second-order statistic of the original
    field -- remove a dominant favorite and the new winner lives where
    the old one never did. This function therefore uses the full
    ability-span window (coverage by construction: it contains every
    runner's density), refines the spacing to the sharpest runner so a
    wide field cannot silently dilute resolution, and CHECKS the
    unnormalized row masses, which the continuum identity fixes at one,
    raising if the defect exceeds mass_tol instead of letting the
    normalization hide it. The tight-window alternative is a lattice
    sized to the first- and second-finisher bulks; the second-finisher
    survival is the one-failure term of the SIAM paper's multiplicity
    union calculus, prod_i S_i (1 + sum_i (1-S_i)/S_i), an O(nL)
    byproduct of the field.
    """
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    span = float(M_all.max() - M_all.min()) + (left + right) * float(sd.max())
    need = int(np.ceil(span / (float(sd.min()) / 8.0))) + 1
    pts = int(min(max(points, need), 16385))
    if need > 16385:
        import warnings
        warnings.warn(
            "removal_shares: the ability span is too wide to resolve the "
            f"sharpest runner even at 16385 lattice points (needs {need}); "
            "row masses are checked and will raise if accuracy is lost",
            RuntimeWarning, stacklevel=2)
    x = np.linspace(M_all.min() - left * sd.max(),
                    M_all.max() + right * sd.max(), pts)
    dx = x[1] - x[0]
    q = np.zeros((n, n))
    for c in range(len(F)):
        z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
        S, f, _ = fn(z)
        f = f / sd[:, None]
        logS = np.log(S)
        logSfield = logS.sum(0)
        for i in range(n):
            rest = np.exp(np.clip(logSfield[None, :] - logS[i] - logS,
                                  -745.0, 0.0))
            contrib = (f * rest).sum(1) * dx
            contrib[i] = 0.0
            q[i] += W[c] * contrib
    mass = q.sum(axis=1)
    defect = float(np.abs(mass - 1.0).max())
    if defect > mass_tol:
        raise FloatingPointError(
            f"removal_shares: post-removal winner mass defect {defect:.2e} "
            f"exceeds {mass_tol:.0e} (worst row {int(np.abs(mass-1).argmax())}); "
            "the lattice failed to capture the post-removal race -- raise "
            "points= or mass_tol= deliberately rather than trusting a "
            "renormalization that would hide the missing mass")
    return q / q.sum(axis=1, keepdims=True)


# canonical tier-1 name; calibrate_factors is reserved for the outer
# estimation problem (see the paper's discussion and experiment 30)
calibrate_abilities = abilities_from_race

_GUMBEL_UNIT_D = np.pi ** 2 / 6.0


def softmax_probabilities(mu, temperature=1.0, V=None, F=None, W=None):
    """Luce/softmax as the closed-form special case of the race, exposed.

    Min-wins: p = softmax(-mu/tau). Identical to
    race_probabilities(mu, D=tau^2 pi^2/6, base="gumbel") -- the
    Gumbel-argmin identity -- but analytic: no lattice, no quadrature
    over the winning value (verified against the lattice at machine
    precision in the tests). With factor loadings V (and nodes F, W over
    the factors), performances are conditionally uniform-scale Gumbel
    given f, so the answer is the exact mixture of conditional softmaxes
        p = sum_q w_q softmax(-(mu + V f_q)/tau),
    one closed form per node. Because it is analytic wherever the race
    is priced numerically, it is the natural control variate and the
    permanent point of comparison: same mu, same conditioning, the IIA
    answer next to the correlated one. Control-variate accounting,
    measured in research/gumbel_cv: equal-draw variance reduction 3-12x
    for bases at or near Gumbel, deflating to 1.2-1.8x at equal compute
    because the coupled twin nearly doubles the simulation loop --
    worthwhile if you are simulating a near-Gumbel base anyway, not a
    reason to simulate. Heterogeneous Gumbel scales have no closed
    form; use race_probabilities(..., base="gumbel") there.
    """
    mu = np.asarray(mu, dtype=float)
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("temperature must be positive")
    if V is None:
        z = -mu / tau
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[0] != len(mu):
        V = V.T
    if F is None or W is None:
        D_impl = np.full(len(mu), _GUMBEL_UNIT_D * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    M = -(mu[None, :] + F @ V.T) / tau
    M -= M.max(axis=1, keepdims=True)
    E = np.exp(M)
    P = E / E.sum(axis=1, keepdims=True)
    return W @ P


def harville_order_logprob(mu, order, temperature=1.0, V=None, F=None,
                           W=None):
    """log P(full finishing order) under (mixed) Plackett--Luce: Harville's
    formula, the stagewise winner-of-remaining product that is EXACT for
    the Gumbel base (IIA) and only there.

    Min-wins: stage t contributes z_{o_t} - logsumexp(z_{o_t:}) with
    z = -mu/tau. With factor loadings V and nodes (F, W) the result is
    the mixed Plackett--Luce likelihood log sum_q w_q P(order | f_q),
    each conditional closed form: the safe way to consume full rankings
    under correlation (consuming them stagewise under the GAUSSIAN base
    instead inflates learned correlation threefold; see
    winning.ratings.nway). Racing lore's Henery and Stern corrections
    are exactly the non-Gumbel ordering problem, i.e. the still-open
    shared-noise ranked-moments item.
    """
    mu = np.asarray(mu, dtype=float)
    tau = float(temperature)
    order = np.asarray(order, dtype=int)

    def _one(z):
        rest = order.copy()
        total = 0.0
        for t in range(len(order) - 1):
            zr = z[rest]
            m = zr.max()
            total += z[rest[0]] - (m + np.log(np.exp(zr - m).sum()))
            rest = rest[1:]
        return total

    if V is None:
        return _one(-mu / tau)
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[0] != len(mu):
        V = V.T
    if F is None or W is None:
        D_impl = np.full(len(mu), _GUMBEL_UNIT_D * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    logs = np.array([_one(-(mu + np.asarray(F)[q] @ V.T) / tau)
                     for q in range(len(F))])
    m = logs.max()
    return float(m + np.log(np.dot(np.asarray(W), np.exp(logs - m))))


def harville_place_probabilities(p, k=3):
    """P(finish in the top k) for every runner, from win probabilities,
    by Harville's conditioning: after removing a finisher, the rest
    renormalize (exact under Luce/Gumbel, the classical racing formula;
    its known favorite-longshot bias in real place markets is the
    Gaussian/Gamma ordering effect Henery and Stern model). k in
    {1, 2, 3} (win, place, show)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(p)
    if k == 1:
        return p.copy()
    out = p.copy()
    # the complement 1 - p_j is computed as the SUM OF THE OTHERS, never
    # by subtraction from one: at p_fav = 1 - 1e-13 the subtraction loses
    # three digits to cancellation and the exact identity sum(top-k) = k
    # drifted to k + 3e-3 (caught by the gap-stress battery)
    rest = np.array([p[np.arange(n) != j].sum() for j in range(n)])
    # second place: j first, i second -- P2[j, i] = p_j p_i / rest_j
    P2 = (p / np.maximum(rest, 1e-300))[:, None] * p[None, :]
    np.fill_diagonal(P2, 0.0)
    out += P2.sum(axis=0)
    if k == 2:
        return out
    if k != 3:
        raise ValueError("k must be 1, 2 or 3")
    # third place: j first, l second, i third
    for j in range(n):
        pj = p[j]
        rem1 = rest[j]
        for sec in range(n):
            if sec == j:
                continue
            w = pj * p[sec] / max(rem1, 1e-300)
            denom2 = rem1 - p[sec]
            if denom2 < 1e-8 * rem1:
                # same cancellation one level deeper (two large entries
                # exhausting the field): recompute as the sum of the
                # actual remaining entries, exactly
                mask = np.ones(n, dtype=bool)
                mask[j] = mask[sec] = False
                denom2 = p[mask].sum()
            denom2 = max(denom2, 1e-300)
            contrib = w * p / denom2
            contrib[j] = 0.0
            contrib[sec] = 0.0
            out += contrib
    return out


def abilities_from_softmax(p, temperature=1.0):
    """Exact inverse of the independent softmax race: mean-zero mu with
    softmax_probabilities(mu, temperature) = p. Closed form."""
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    tau = float(temperature)
    logp = np.log(p / p.sum())
    return -tau * (logp - logp.mean())


def failure_base(q, width=0.35, offset=6.0, base="normal",
                 standardize=False):
    """A performance density with a FAILURE LUMP: with probability q the
    contestant does not perform (retires, crashes, times out, refuses,
    emits a parse error) and lands far down the field; otherwise it runs
    the given base.

    Returns a base callable for race_probabilities(base=...) and the
    ratings updates. Min-wins: the lump sits `offset` standard units
    into the SLOW tail, smeared by `width` so everything stays smooth
    and differentiable (a true Dirac would break the lattice, which is
    exactly the 'dirac disaster' the F1 experiments recorded).

    Why this rather than a heuristic: with a lump in the density, Bayes
    SPLITS a catastrophic result between 'slow' and 'broke' according to
    the modelled failure rate, instead of charging all of it to ability
    (the naive treatment, which ranks a retirement as last) or throwing
    the evidence away (censoring). Measured motivation, from the bandits
    lane on synthetic motorsport: with mechanical, ability-independent
    failures censoring is near-optimal (RMSE 0.146 against naive 0.351),
    but once retirement probability rises as ability falls, neither
    heuristic dominates -- naive wins the ordering, censoring wins the
    magnitudes, and no tuning reconciles them, because they answer
    different halves of the question.

    WHEN TO REACH FOR IT (measured, bandits lane, M=8, 40 races, q=0.25,
    25 seeds, paired per-seed comparisons; ability rank correlation and
    RMSE):

      failures INDEPENDENT of ability   spearman   rmse
        naive (retirement ranks last)     0.951    0.488
        censored (drop the retiree)       0.960    0.205   <- best
        lump at the true q                0.957    0.290
        lump at q/2                       0.957    0.279
        lump at 2q                        0.418    1.155
      failures COUPLED to ability (0.12)
        naive                             0.965    0.255
        censored                          0.961    0.214
        lump at the true q                0.957    0.233
        lump at q/2                       0.967    0.203   <- best on both
        lump at 2q                        0.614    0.945

    The crossover sits near a coupling of $0.08$, which in checkable
    terms is: the weakest entrants failing about twice as often as the
    strongest. Below roughly a 1.5x ratio, censor without thinking about
    it. But the rule to remember is the asymmetry around the crossover,
    not the crossover: the two mistakes cost about 5 to 1. Wrongly
    lumping when failures are independent costs 0.072 RMSE (a 3.6-SE
    effect); wrongly censoring when they are coupled costs 0.015 (1.1
    SE). So the operational rule is "censor unless the coupling evidence
    is strong", not "estimate the coupling and pick a side". At strong
    coupling (0.16) the naive treatment becomes competitive again and
    takes the best rank correlation, because a retirement genuinely has
    become evidence of weakness, so the three rules partition by
    coupling rather than one dominating.

    So this is a targeted instrument, not a general replacement for
    censoring. When failures are plausibly ability-independent, censor:
    the lump repairs most of the naive damage but does not reach the
    simple fix, and that gap is not an artifact of the lump's shape (a
    sweep of offset over 3, 6 and 12 moves RMSE only within 0.27-0.30).
    When failures are coupled to ability, an UNDER-WEIGHTED lump is the
    only method measured that wins on ordering and magnitude at once,
    which is the reconciliation the mechanism promises and neither
    heuristic could deliver. At the true q it does not reconcile.

    Censoring in this package is passing the finishing order over the
    runners who finished; the omitted runner is marginalized out exactly
    (see update_ranking_exact).

    GUESS q LOW. Misspecification is violently asymmetric and the
    dangerous direction is overstatement: at 2q the model expects half
    the field to break, so a genuinely slow finisher is attributed to
    failure and the back of the field stops carrying ability information
    at all -- rank correlation collapses from 0.96 to 0.42. Halving q is
    free or better than free in both regimes. Take the low end of the
    plausible range rather than the midpoint, and if q is a guess rather
    than an estimate prefer censoring, which has no q to misspecify.

    USE IT ON RANKED FEEDBACK. Under winner-only feedback a retirement
    is just "not the winner", which describes most of the field too, so
    the lump has almost nothing to explain: on the winner path (M=10, 80
    races, 10 seeds) it is one clear win at a 30 percent independent
    failure rate (RMSE 0.389 against Gaussian's 0.420) and otherwise
    indistinguishable, and the same overstatement that is catastrophic
    on ranked feedback is merely mild there. The information in a
    retirement is in the LAST-PLACE FINISH, which is why the order path
    carries this feature: update_ranking, update_ranking_exact,
    order_loglik, update_order_correlated, update_order_full,
    update_team_order_full, update_race, rate_history and predict_race
    all take base=. update_winner_full is Gaussian only and says so.

    NOT standardized by default, deliberately, and this matters: the
    engine's other bases are mean-zero unit-variance so that `D` is the
    performance variance, but a lump at six sigma inflates the mixture
    variance to 7.5 at q=0.25, and standardizing would then squeeze the
    RUNNING component by a factor of 2.7 -- `D` would silently stop
    meaning what the caller thinks. Measured cost of getting this wrong:
    abilities recovered with the right ORDER (rank correlation 0.85) but
    badly shrunk magnitudes (RMSE 0.58 against 0.27 for the naive
    treatment). Unstandardized, the running component stays N(0,1) in
    `z`, so `D` is the variance of a contestant who actually runs, and
    the lump is a separate event on top. Pass standardize=True only if
    you specifically want the mixture itself normalized.
    """
    q = float(q)
    if not 0.0 <= q < 1.0:
        raise ValueError("failure probability q must lie in [0, 1)")
    fn0 = base if callable(base) else BASES[base]
    w = float(width)
    off = float(offset)
    if standardize:
        m1 = q * off
        var = (1.0 - q) * 1.0 + q * (w * w + off * off) - m1 * m1
        sd = np.sqrt(max(var, 1e-12))
    else:
        m1, sd = 0.0, 1.0

    def _fail(z):
        u = m1 + sd * np.asarray(z, dtype=float)      # de-standardize
        S0, f0, fp0 = fn0(u)
        zl = (u - off) / w
        Sl = np.maximum(1.0 - ndtr(zl), 1e-300)
        fl = np.exp(-0.5 * zl * zl) / (w * np.sqrt(2.0 * np.pi))
        fpl = -zl * fl / w
        S = (1.0 - q) * S0 + q * Sl
        f = ((1.0 - q) * f0 + q * fl) * sd
        fp = ((1.0 - q) * fp0 + q * fpl) * sd * sd
        return np.maximum(S, 1e-300), f, fp

    return _fail
