"""Fast multivariate normal rectangle probabilities for structured
covariance: the scipy.stats.multivariate_normal.cdf drop-in for the
factor-plus-diagonal slice.

P(a <= X <= b) with X ~ N(mu, V V' + diag(D)): conditional on the
r-dimensional factor the coordinates are independent, so the rectangle
probability is an r-dimensional smooth integral of a product of
univariate normal CDFs. Deterministic, milliseconds at dimensions where
Genz-style quadrature needs seconds to minutes (measured: 4.6 s for one
probability at n = 200 via scipy's integrator vs ~2 ms here).

Port of the R package mvtnormfast (r/mvtnormfast in this repository),
whose measured validation carries over: agreement with mvtnorm inside
its own reported error bound, with Botev's minimax tilting to a few
1e-4 relative at probabilities down to 1e-17 (deep tails via Laplace
recentering), and strict refusal on genuinely dense covariance -- an
inexact factorization must never masquerade as the structured case, so
those calls fall back to scipy.stats.multivariate_normal.cdf unchanged.
"""

from __future__ import annotations

import functools

import numpy as np

from .shapes import as_idio, as_loadings

from numpy.polynomial.hermite_e import hermegauss
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import norm


def _halton_unit(r, n):
    primes = (2, 3, 5, 7, 11, 13)[:r]
    out = np.empty((n, r))
    for c, b in enumerate(primes):
        idx = np.arange(1, n + 1) + 20
        h = np.zeros(n)
        f = 1.0 / b
        i = idx.copy()
        while i.max() > 0:
            h += f * (i % b)
            i //= b
            f /= b
        out[:, c] = h
    return out


def _gh_nodes(r, Q):
    # Cached: the rule depends on (r, Q) alone, and rebuilding it was most
    # of the cost of an ordinary call -- hermegauss(201) is an eigensolve
    # of a 201 x 201 companion matrix, 2.4 ms against 0.1 ms for the
    # integral it serves.
    F, W = _gh_nodes_cached(int(r), int(Q))
    return F, W


@functools.lru_cache(maxsize=64)
def _gh_nodes_cached(r, Q):
    # r = 0 is the EMPTY PRODUCT: one node of weight 1 with no columns,
    # so an (n, 0) loading matrix -- which the shape normalizer accepts
    # as "empty = indep" -- reduces exactly to the independent product.
    # np.meshgrid() of nothing gave "need at least one array to
    # concatenate" from inside numpy instead (#68).
    if r == 0:
        F, W = np.zeros((1, 0)), np.ones(1)
    else:
        x, w = hermegauss(Q)
        w = w / w.sum()
        grids = np.meshgrid(*([x] * r), indexing="ij")
        F = np.column_stack([g.ravel() for g in grids])
        W = np.ones(len(F))
        for c in range(r):
            W *= w[np.searchsorted(x, F[:, c])]
        keep = W > 1e-12 * W.max()
        F, W = F[keep], W[keep] / W[keep].sum()
    F.setflags(write=False)
    W.setflags(write=False)
    return F, W


def _nodes_for(V, D):
    r = V.shape[1]
    sharp = float(np.max(np.sqrt((V ** 2).sum(axis=1))
                         / np.sqrt(np.maximum(D, 1e-300))))
    if sharp > 3.0 or r > 2:
        # scrambled Sobol, not Halton: plain Halton's low-dimensional
        # projections degrade badly past three dimensions (measured: a
        # rank-6 auto-detected decomposition gave 1e-4 error on Halton
        # nodes where Sobol reaches quadrature accuracy).
        from scipy.stats import qmc
        n = 2 ** 13
        u = qmc.Sobol(r, scramble=True, seed=0).random(n)
        F = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        return F, np.full(n, 1.0 / n)
    # The order grows like sharpness SQUARED, not 8 x sharpness. A
    # cell's transition in factor space is 1/sharpness wide and
    # Gauss-Hermite resolves it only once its node spacing is well below
    # that; the linear rule gave 0.2% relative error on a two-coordinate
    # orthant at correlation 0.8 (sharpness 2, Q = 16) and 0.7% at 0.9,
    # against closed forms (#132). At sharpness <= 1 this is the old
    # old 15-point floor at rank two, so ordinary inputs pay what they
    # always did, and the rules are cached. Rank one floors at 61: its
    # cost is n x Q evaluations (microseconds once the rule is cached),
    # and 15 points left 2e-6 at correlation 0.5, and 2.6e-3 on a
    # 200-coordinate field whose rows are individually mild but whose
    # product is sharp. The caps (201, 41) and the Sobol escalation past
    # sharpness 3 are unchanged. Measured: correlation 0.8 orthant
    # 4e-3 -> 3e-8; a rank-two sharpness-2.8 field 2e-3 -> 2e-7 at 0.15 ms.
    Q = int(np.clip(np.ceil(15.0 * sharp * sharp), 61 if r == 1 else 15,
                    201 if r == 1 else 41))
    return _gh_nodes(r, Q)


def factorize_covariance(sigma, max_rank=6, tol=1e-11, n_iter=300):
    """Exact V V' + diag(D) decomposition of sigma, if one exists.

    Iterated principal-factor fit for ranks 0..max_rank, accepted only
    after a recomputed-V verification against the final D; returns
    (V, D) or None.

    The search runs on the CORRELATION matrix and scales back. Two
    tolerances used to be absolute or global, and both broke a contract:

    - the residual floor `D >= 1e-12` and the acceptance test
      `err < tol * max|sigma|` made the answer depend on units. The
      exact factor covariance `1e-12 * S` was rejected outright (its
      clipped D cannot reconstruct a diagonal of size 1e-12), so the
      call went to the stochastic scipy fallback while `S` itself took
      the deterministic structured path (#368);
    - the rank-zero test `max|offdiag| <= tol * max|sigma|` used the
      largest variance as the yardstick for every correlation, so a
      0.9 correlation between two unit-variance coordinates was
      "negligible" next to an unrelated 1e16 variance and discarded:
      41.6% low on an orthant whose answer is closed-form (#132).

    In correlation units every tolerance is dimensionless and applies to
    each pair on its own scale. Rank zero is tried first: a diagonal
    sigma is exactly V V' + diag(D) with no factors, and a rank-one
    search fits it by putting a variance into an artificial LOADING
    (9487 on diag([1, 1, 1, 1, 1e8])), which turns an independent
    product into a near-step factor integrand (#132).
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1] \
            or not np.isfinite(sigma).all():
        return None
    n = len(sigma)
    d = np.diag(sigma).copy()
    if np.any(d < 0.0):
        return None
    pos = d > 0.0
    # a zero-variance coordinate is a constant: its whole row must be
    # zero, and it carries neither loading nor residual
    if np.any(sigma[~pos] != 0.0) or np.any(sigma[:, ~pos] != 0.0):
        return None
    m = int(pos.sum())
    sd = np.sqrt(d[pos])
    R = sigma[np.ix_(pos, pos)] / np.outer(sd, sd)

    def _back(Vr, Dr):
        V = np.zeros((n, Vr.shape[1]))
        V[pos] = Vr * sd[:, None]
        D = np.zeros(n)
        D[pos] = Dr * d[pos]
        return V, D

    if m == 0 or np.abs(R - np.diag(np.diag(R))).max() <= tol:
        return np.zeros((n, 0)), d
    dR = np.diag(R)
    for r in range(1, min(max_rank, m - 1) + 1):
        D = np.full(m, 0.5)
        for _ in range(n_iter):
            lam, U = np.linalg.eigh(R - np.diag(D))
            idx = np.argsort(lam)[::-1][:r]
            V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
            D_new = np.maximum(dR - (V ** 2).sum(axis=1), 1e-12)
            if np.abs(D_new - D).max() < 1e-12:
                D = D_new
                break
            D = D_new
        lam, U = np.linalg.eigh(R - np.diag(D))
        idx = np.argsort(lam)[::-1][:r]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        if np.abs(V @ V.T + np.diag(D) - R).max() < tol:
            return _back(V, D)
    return None


def _as_covariance(sigma):
    """A finite, square, symmetric, positive-semidefinite matrix, or a
    ValueError. Returns the symmetrised matrix.

    `sigma` reached scipy's multivariate_normal unchecked whenever the
    factorization declined it, and scipy reads ONE triangle of whatever
    it is given. An asymmetric matrix and its transpose therefore priced
    two different Gaussians -- 0.1732 and 0.2270 on the same orthant,
    the closed-form values of the two triangular completions -- with
    nothing but "scipy-fallback" to say so (#359). Same tolerances as
    winning.factor.core._validate_covariance, the race layer's door.
    """
    C = np.asarray(sigma, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"sigma must be a square matrix; got shape "
                         f"{C.shape}")
    if not np.isfinite(C).all():
        raise ValueError("sigma contains NaN or inf")
    asym = float(np.abs(C - C.T).max()) if C.size else 0.0
    if asym > 1e-8 * max(float(np.abs(C).max()) if C.size else 0.0,
                         1e-300):
        raise ValueError(
            f"sigma is not symmetric (max asymmetry {asym:.2e}); pass "
            "(sigma + sigma.T)/2 if the asymmetry is numerical noise")
    C = 0.5 * C + 0.5 * C.T
    if C.size:
        lam_min = float(np.linalg.eigvalsh(C).min())
        n = len(C)
        if lam_min < -1e-8 * max(float(np.sum(np.diag(C) / n)), 1e-300):
            raise ValueError(
                f"sigma is not positive semidefinite (min eigenvalue "
                f"{lam_min:.2e}); it is not a covariance matrix")
    return C


def mvn_cdf_fast(lower=None, upper=None, mean=None, sigma=None,
                 V=None, D=None):
    """P(lower <= X <= upper), X ~ N(mean, V V' + diag(D)).

    Supply (V, D), or supply sigma and an exact decomposition is
    searched; dense covariance falls back to
    scipy.stats.multivariate_normal.cdf unchanged. Returns a float with
    .method metadata via the companion function mvn_cdf_fast_info.
    """
    p, _ = _mvn_cdf_impl(lower, upper, mean, sigma, V, D)
    return p


def mvn_cdf_fast_info(lower=None, upper=None, mean=None, sigma=None,
                      V=None, D=None):
    """As mvn_cdf_fast, returning (p, method)."""
    return _mvn_cdf_impl(lower, upper, mean, sigma, V, D)


def _check_bounds(lo, up):
    """Refuse a reversed rectangle {lo <= x <= up}.

    A reversed coordinate makes the event EMPTY. Every port used to form
    the negative conditional cell -- Phi(0) - Phi(1) = -0.3413... -- and
    then clamp it to the underflow floor, so an impossible observation
    came back as 1e-300 and, in log-likelihood code, as a finite -690.8
    instead of -inf (#235). `mvtnorm::pmvnorm`, which r/mvtnormfast is a
    drop-in for, raises on reversed bounds; all four ports do the same.

    lower == upper is a degenerate slab of zero mass; `_drop_constants`
    returns it, as mvtnorm::pmvnorm does.
    """
    if np.isnan(lo).any() or np.isnan(up).any():
        raise ValueError("lower and upper must not contain NaN")
    bad = np.nonzero(lo > up)[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            "lower must not exceed upper: coordinate {} has lower={!r} > "
            "upper={!r}, so the rectangle is empty. Check the argument "
            "order.".format(i, float(lo[i]), float(up[i])))


def _per_coordinate(x, n, name, default, finite):
    """A scalar, a one-element container, or n values; never a reshape.
    A one-element container is the scalar shorthand, as in the R port."""
    if x is None:
        return np.full(n, default)
    a = np.asarray(x, dtype=float)
    if a.size == 1:
        a = np.full(n, float(a.reshape(())))
    elif a.shape != (n,):
        raise ValueError(f"{name} must be a scalar or one value per "
                         f"coordinate: expected shape ({n},); got {a.shape}")
    a = a.astype(float)
    if finite and not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _infer_dimension(V, D, mean, lower, upper):
    """The number of coordinates, from the arguments that carry it.

    It used to come from D when D was a vector and from the RAW shape
    of V otherwise. So a scalar D read a transposed (1, n) loading as a
    one-dimensional problem and a scalar V as no dimension at all
    (#79), and a one-element D = [1.0] beside a canonical (3, 1)
    loading redefined a three-coordinate rectangle as one coordinate:
    Phi(0) = 0.5 where Phi(0)^3 = 0.125 was asked (#422).

    Every argument with more than one entry is length-bearing and they
    must agree; a one-element container is the scalar shorthand. Only
    when none of them has a length does V decide, read canonically: an
    (n, rank) matrix has n rows, a vector one entry per coordinate.
    """
    lengths = {}
    for name, x in (("D", D), ("mean", mean), ("lower", lower),
                    ("upper", upper)):
        if x is None:
            continue
        a = np.asarray(x)
        if a.size > 1:
            if a.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional; got "
                                 f"shape {a.shape}")
            lengths[name] = a.size
    if len(set(lengths.values())) > 1:
        raise ValueError("the per-coordinate arguments disagree about the "
                         "dimension: " + ", ".join(
                             f"{k} has {v}" for k, v in lengths.items()))
    if lengths:
        return next(iter(lengths.values()))
    A = np.asarray(V, dtype=float)
    if A.ndim == 0:
        return 1
    if A.ndim == 1:
        return A.size
    return A.shape[0]


def _mvn_cdf_impl(lower, upper, mean, sigma, V, D):
    if V is None or D is None:
        if sigma is None:
            raise ValueError("supply sigma, or V and D")
        sigma = _as_covariance(sigma)
        n = len(sigma)
        mu = _per_coordinate(mean, n, "mean", 0.0, True)
        lo = _per_coordinate(lower, n, "lower", -np.inf, False)
        up = _per_coordinate(upper, n, "upper", np.inf, False)
        _check_bounds(lo, up)
        fd = factorize_covariance(sigma)
        if fd is None:
            return _dense(lo, up, mu, sigma)
        V, D = fd
    else:
        n = _infer_dimension(V, D, mean, lower, upper)
        V = as_loadings(V, n)
        Da = np.asarray(D, dtype=float)
        D = as_idio(Da.reshape(()) if Da.size == 1 else Da, n)
        mu = _per_coordinate(mean, n, "mean", 0.0, True)
        lo = _per_coordinate(lower, n, "lower", -np.inf, False)
        up = _per_coordinate(upper, n, "upper", np.inf, False)
        _check_bounds(lo, up)
    return _structured(lo, up, mu, V, D)


def _drop_constants(var, mu, lo, up):
    """Settle the degenerate cases before any integration.

    lower == upper on any coordinate is a zero-mass slab, as in
    mvtnorm::pmvnorm (that includes -inf == -inf and +inf == +inf). A
    zero-variance coordinate is the constant mu_i: outside its interval
    the probability is exactly 0; inside, it leaves the integral.
    Near-deterministic (Dirac) inputs are otherwise out of scope.

    Returns (keep, status): status is None, or the (p, method) answer.
    """
    if np.any(lo == up):
        return None, (0.0, "degenerate-rectangle")
    const = var == 0.0
    if np.any(const & ((mu < lo) | (mu > up))):
        return None, (0.0, "outside-support")
    keep = ~const
    if not keep.any():
        return keep, (1.0, "factor")
    return keep, None


def _dense(lo, up, mu, sigma):
    keep, done = _drop_constants(np.diag(sigma).copy(), mu, lo, up)
    if done is not None:
        return done
    idx = np.flatnonzero(keep)
    S = sigma[np.ix_(idx, idx)]
    if len(idx) == 1:
        return _univariate(lo[idx], up[idx], mu[idx], S[0, 0])
    from scipy.stats import multivariate_normal
    mvn = multivariate_normal(mean=mu[idx], cov=S, allow_singular=True)
    p = mvn.cdf(up[idx], lower_limit=lo[idx])
    return float(p), "scipy-fallback"


def _univariate(lo, up, mu, var):
    """One coordinate: the marginal is N(mu, var) whatever the factor
    decomposition, so no quadrature is needed. Sending it through the
    8192-node Sobol rule at sharpness above 3 returned 1/8192 -- one
    node -- for a probability of 2.07e-5, 5.9x too high, because a
    single hit cleared the 1e-8 recentering trigger (#395)."""
    s = np.sqrt(float(np.asarray(var).reshape(())))
    lm = _log_interval_mass(np.asarray(up, float).reshape(1) - mu,
                            np.asarray(lo, float).reshape(1) - mu,
                            np.array([s]))
    return float(np.exp(lm[0])), "factor"


def _structured(lo, up, mu, V, D):
    var = (V ** 2).sum(axis=1) + D
    keep, done = _drop_constants(var, mu, lo, up)
    if done is not None:
        return done
    if not keep.all():
        V, D, mu, lo, up = V[keep], D[keep], mu[keep], lo[keep], up[keep]
        var = var[keep]
    if len(D) == 1:
        return _univariate(lo, up, mu, var[0])
    # a factor no coordinate loads on integrates out exactly
    V = V[:, np.any(V != 0.0, axis=0)]
    if V.shape[1] == 0:
        # rank zero is the exact independent product
        return float(np.exp(_log_interval_mass(up - mu, lo - mu,
                                               np.sqrt(D)).sum())), "factor"
    s = np.sqrt(D)
    F, W = _nodes_for(V, D)
    contrib = W * np.exp(_log_cells(F, V, s, mu, lo, up))
    p = float(contrib.sum())
    # The answer is trusted only if it is RESOLVED. On the equal-weight
    # Sobol rule a single node is worth 1/8192 = 1.2e-4, so the 1e-8
    # trigger alone accepted one accidental hit as the probability
    # (#395). The effective number of contributing nodes says whether
    # the rule saw the integrand or a few points of it.
    ess = p * p / max(float((contrib ** 2).sum()), 1e-300) if p > 0 else 0.0
    sobol = len(W) > 1 and np.all(W == W[0]) and len(W) >= 2 ** 13
    if p >= 1e-8 and (not sobol or ess >= 100.0):
        return p, "factor"
    return _recentered(V, s, mu, lo, up), "factor-recentered"


def _log_cells(F, V, s, mu, lo, up):
    """Sum over coordinates of the log conditional cell mass, per node."""
    M = F @ V.T
    hi = (up - mu)[None, :] - M
    lo_ = (lo - mu)[None, :] - M
    return _log_interval_mass(hi, lo_, s).sum(axis=1)


def _recentered(V, s, mu, lo, up):
    """Deep tail: recenter the node set at the mode of the log-integrand
    and importance-reweight (see r/mvtnormfast)."""
    r = V.shape[1]

    def logint(f):
        return float(_log_cells(f[None, :], V, s, mu, lo, up)[0]
                     - 0.5 * f @ f)

    f0 = _ascend(logint, np.zeros(r))
    from scipy.stats import qmc
    nn = 2 ** 13
    Fh = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=1).random(nn),
                       1e-12, 1 - 1e-12))
    tau = 1.5
    Fq = Fh * tau + f0
    logw = -0.5 * (Fq ** 2).sum(axis=1) + 0.5 * (Fh ** 2).sum(axis=1) \
        + r * np.log(tau)
    lt = _log_cells(Fq, V, s, mu, lo, up) + logw
    m = lt.max()
    if not np.isfinite(m):
        return 0.0
    return float(np.exp(m) * np.mean(np.exp(lt - m)))


def _ascend(logint, f0, h=1e-4, iters=100):
    """Gradient ascent with backtracking on the log-integrand.

    The integrand is evaluated in the log domain (#196), so it is smooth
    and finite wherever the rectangle has mass. The old search
    differentiated a log of cells floored at 1e-300, whose gradient is
    exactly zero wherever every cell has underflowed, and a fixed clipped
    step oscillates on a sharp integrand. The tail repair that #395 now
    routes unresolved Sobol estimates to depends on finding the mode."""
    r = len(f0)
    val = logint(f0)
    for _ in range(iters):
        g = np.array([(logint(f0 + h * e) - logint(f0 - h * e)) / (2 * h)
                      for e in np.eye(r)])
        if not np.all(np.isfinite(g)) or np.linalg.norm(g) < 1e-8:
            break
        step = 1.0
        while step > 1e-12:
            cand = f0 + np.clip(step * g, -1.0, 1.0)
            cv = logint(cand)
            if cv > val:
                f0, val = cand, cv
                break
            step *= 0.5
        else:
            break
    return f0


def _log_interval_mass(hi, lo_, s):
    """Per-coordinate log P(lo_ <= sigma Z <= hi), sigma == 0 too.

    In the LOG domain, on the side of zero where nothing cancels. The
    mass used to be ndtr(hi) - ndtr(lo): in the upper tail both round to
    1.0, so P(9 < Z <= 10) = 1.13e-19 became exactly 0 and then the
    1e-300 floor, [8, 9] came out 7% high, and a rectangle and its
    reflection got different probabilities -- Phi(-9)^n on the left,
    zero on the right for n >= 2 (#98, #196). Upper-tail intervals are
    now reflected to the lower tail, and log_ndtr carries the deep tail
    without underflow, so nothing needs a floor: an exactly empty cell
    is -inf and stays -inf.

    A coordinate with zero idiosyncratic variance is DETERMINISTIC given
    the factor draw, so its cell is an INDICATOR, not a gaussian
    interval (#206): `hi` and `lo_` are already shifted by the mean and
    the factor term, so it is inside exactly when lo_ <= 0 <= hi.
    """
    hi, lo_, s = np.broadcast_arrays(np.asarray(hi, float),
                                     np.asarray(lo_, float),
                                     np.asarray(s, float))
    if np.all(lo_ == -np.inf) and np.all(s > 0.0):
        # the common one-sided cell: log Phi(hi/s), which log_ndtr gives
        # stably on both sides of zero; one call instead of six
        return log_ndtr(hi / s)
    out = np.where((lo_ <= 0.0) & (0.0 <= hi), 0.0, -np.inf)
    g = s > 0.0
    if not g.any():
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b = np.where(g, hi / np.where(g, s, 1.0), 0.0)
        a = np.where(g, lo_ / np.where(g, s, 1.0), 0.0)
        # reflect so that b <= |a| side is the lower tail: the interval
        # [a, b] has the mass of [-b, -a]
        flip = a > 0.0
        a, b = np.where(flip, -b, a), np.where(flip, -a, b)
        lb = log_ndtr(b)
        la = log_ndtr(a)
        # both in the lower tail (b <= 0): log(Phi(b) - Phi(a))
        left = lb + np.log1p(-np.exp(la - lb))
        # straddling zero: 1 - Phi(a) - Phi(-b), no cancellation
        mid = np.log1p(-(ndtr(a) + ndtr(-b)))
        lm = np.where(b <= 0.0, left, mid)
        lm = np.where(a >= b, -np.inf, lm)
    return np.where(g, lm, out)
