"""The general race: one API, distributions and correlation as parameters.

    race_probabilities(mu)                          classic independent race
    race_probabilities(mu, V=V, D=D)                factor probit (Gaussian)
    race_probabilities(mu, base="gumbel")           Luce / softmax, exactly
    race_probabilities(mu, V=V, base="gumbel")      correlated Luce
    race_probabilities(mu, base="logistic")         logistic noise
    race_probabilities(mu, base="laplace")          robust double-exponential

    NOTE: D is the base's own dispersion (performance-noise variance).
    Folding a Gaussian ABILITY-belief variance into D (D = v + beta2)
    is exact only for base="normal", the one base stable under
    convolution; for other bases the true predictive is a Gaussian-
    smoothed base, not a wider base, so that shortcut is biased.
    race_probabilities(mu, base=student_base(4))    fat tails, unit variance
    race_probabilities(mu, base=skew_normal_base(2))  the heritage family
    race_probabilities(mu, base=failure_base(0.1))  performance + DNF lump
    race_probabilities(mu, base=my_base)            anything standardized

Min-wins convention throughout. A base is a callable z -> (S, f, fp)
giving survival, density and density derivative of a MEAN-ZERO,
UNIT-VARIANCE law (standardization keeps noise family separate from
noise scale). Zero factors is literally the one-node quadrature, so the
independent race is not a separate code path.

A callable base may also carry optional attributes: `span` (left, right)
tail extents, `resolution` (central scale), `kinks`, `sample`, and the
tails `cdf(z)` (= 1 - S, accurate in the lower tail), `log_cdf(z)` (used
when cdf is absent) and `log_sf(z)`. The count program behind top-k,
bottom-k and rank marginals takes a lower-tail CDF from `cdf`/`log_cdf`
where 1 - S would cancel, and floored survivals take `log_sf`, as the
named bases do internally; without them a wrapper of the normal prices
[-30, 0, 0]'s second place 21 orders low. The tails are checked once on
a probe (callable, finite, monotone, within 1e-6 of the base's own S).

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

import weakref

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.special import log_ndtr as _log_ndtr

from .core import as_idio, as_loadings, as_weights, _gauge_center, hermite_nodes
from ..shapes import as_factor_law, as_points, as_target, as_tolerance

from ..rustconfig import load_fastrace
from ..outcomes import as_luce_temperature, as_order, as_soft_temperature

# compiled kernels (rust/fastrace); honours WINNING_PURE and use_rust()
_fastrace, _RUST_OK, _HAVE_RUST = load_fastrace('forward_and_slopes')

_EULER = 0.5772156649015329


def _normal(z):
    # the upper tail directly: 1 - ndtr(z) cancels to exactly 0 past
    # about 8.3 sd, and a 20-sd longshot came out ~95x too unlikely (#96)
    S = np.maximum(ndtr(-z), 1e-300)
    f = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return S, f, -z * f


def _gumbel_min(z):
    c = np.pi / np.sqrt(6.0)
    u = np.minimum(z * c - _EULER, 30.0)
    eu = np.exp(u)
    S0 = np.exp(-eu)
    # the density and its slope from the unfloored survival: c eu 1e-300
    # was a fabricated right-tail density (#502)
    f = c * eu * S0
    return np.maximum(S0, 1e-300), f, c * c * eu * S0 * (1.0 - eu)


def _logistic(z):
    # standardized logistic: scale s = sqrt(3)/pi gives unit variance
    c = np.pi / np.sqrt(3.0)
    u = c * np.asarray(z, dtype=float)
    # Reflected, from e = exp(-|u|) <= 1: f = c e / (1+e)^2 never
    # cancels. f = c S (1-S) rounded 1-S to zero once S hit 1 on the left
    # tail, so a runner 25 sd behind was priced 19% low and one 50 sd
    # behind 2e5 times low, disagreeing with the compiled kernel (#136).
    e = np.exp(-np.abs(u))
    S = np.where(u > 0, e / (1.0 + e), 1.0 / (1.0 + e))
    f = c * e / (1.0 + e) ** 2
    # S' = -f, so f' = c S'(1-2S) = -c f (1-2S) (a sign the numeric
    # audit caught in the first draft)
    return np.maximum(S, 1e-300), f, -c * f * (1.0 - 2.0 * S)


def _laplace(z):
    # standardized Laplace: scale b = 1/sqrt(2) gives unit variance
    b = 1.0 / np.sqrt(2.0)
    az = np.abs(z)
    f = np.exp(-az / b) / (2.0 * b)
    # exp of the negative magnitude only: np.where evaluates BOTH
    # branches, so exp(z/b) overflows for large positive z even though
    # the overflowing branch is never selected
    half = b * f                        # = 0.5 * exp(-|z|/b)
    S = np.where(z < 0, 1.0 - half, half)
    return np.maximum(S, 1e-300), f, -np.sign(z) * f / b


# the density is not analytic at z = 0: kink_split=True lattices split
# there (kink_lattice); the default uniform lattice ignores this
_laplace.kinks = (0.0,)


# Log survivals of the named bases, exact where S itself is floored at
# 1e-300 (#502). The integrators take log S; the floor is only a guard
# for log(0), and summed as real mass it made a rare winner plateau at
# ~1e-300 (normal mu = [0, 53, 106]: 1.0e-300 for 1.105e-307). Only the
# floored points use these, so every result above that range is
# unchanged bit for bit.
def _gumbel_log_sf(z):
    c = np.pi / np.sqrt(6.0)
    return -np.exp(np.minimum(z * c - _EULER, 30.0))


def _logistic_log_sf(z):
    u = (np.pi / np.sqrt(3.0)) * np.asarray(z, dtype=float)
    return -np.maximum(u, 0.0) - np.log1p(np.exp(-np.abs(u)))


def _laplace_log_sf(z):
    b = 1.0 / np.sqrt(2.0)
    az = np.abs(z)
    return np.where(z < 0, np.log1p(-0.5 * np.exp(-az / b)),
                    np.log(0.5) - az / b)


def _gumbel_cdf(z):
    # the min-Gumbel is asymmetric, so its lower tail cannot be the
    # reflected survival; 1 - S rounded a representable CDF to zero once
    # S hit one (last place 35 sd ahead: 0 for 2.05e-39, #710)
    c = np.pi / np.sqrt(6.0)
    return -np.expm1(-np.exp(np.minimum(z * c - _EULER, 30.0)))


_normal.log_sf = lambda z: _log_ndtr(-z)
_gumbel_min.log_sf = _gumbel_log_sf
_gumbel_min.cdf = _gumbel_cdf
_logistic.log_sf = _logistic_log_sf
_laplace.log_sf = _laplace_log_sf


# Optional tail attributes of a CALLABLE base (#511 follow-up). The named
# bases carry log_sf, and the top-k count program reflects the symmetric
# ones for a lower-tail CDF that 1 - S cannot give. A callable has no
# identity to key that on, so it may carry the tails itself:
#   cdf(z)      F(z) = 1 - S(z), accurate in the lower tail
#   log_cdf(z)  log F(z), the same (used when cdf is absent)
#   log_sf(z)   log S(z), accurate where S underflows
# Each is checked once per base on a probe before first use: callable,
# finite, the right shape, monotone, and consistent with the base's own S.
_TAIL_ATTRS = ("cdf", "log_cdf", "log_sf")
_TAIL_PROBE = np.linspace(-6.0, 6.0, 25)[:, None]
# fn -> ids of the checked attributes (bound methods by their function),
# so a cache entry holds no reference that would keep fn alive
_TAIL_CHECKED = weakref.WeakKeyDictionary()


def _check_tails(fn):
    attrs = tuple(getattr(fn, a, None) for a in _TAIL_ATTRS)
    key = tuple(id(getattr(a, "__func__", a)) for a in attrs)
    try:
        if _TAIL_CHECKED.get(fn) == key:
            return
    except TypeError:               # not weak-referenceable: check again
        pass
    if all(a is None for a in attrs):
        return
    z = _TAIL_PROBE
    S = None
    for name, a in zip(_TAIL_ATTRS, attrs):
        if a is None:
            continue
        if not callable(a):
            raise TypeError(f"base.{name} must be callable z -> array, "
                            f"got {type(a).__name__}")
        try:
            v = np.asarray(a(z), dtype=float)
        except Exception as e:  # noqa: BLE001 -- name the attribute
            raise ValueError(f"base.{name} failed on a probe of z in "
                             f"[-6, 6]: {e}") from e
        if v.shape != z.shape or not np.all(np.isfinite(v)):
            raise ValueError(f"base.{name} must return a finite array of "
                             "the input's shape on z in [-6, 6]")
        step = np.diff(v[:, 0])
        if name == "log_sf":
            ok = np.all(v <= 1e-12) and np.all(step <= 1e-12)
        elif name == "log_cdf":
            ok = np.all(v <= 1e-12) and np.all(step >= -1e-12)
        else:
            ok = (np.all(v >= 0.0) and np.all(v <= 1.0)
                  and np.all(step >= -1e-12))
        if not ok:
            raise ValueError(
                f"base.{name} is not a valid {name} on z in [-6, 6]: it "
                "must be " + ("non-increasing" if name == "log_sf"
                              else "non-decreasing")
                + (" and <= 0" if name != "cdf" else " and in [0, 1]"))
        if S is None:
            S = np.asarray(fn(z)[0], dtype=float)
        F = np.exp(v) if name == "log_cdf" else v
        gap = np.abs(np.exp(v) - S) if name == "log_sf" \
            else np.abs(F - (1.0 - S))
        if not np.all(gap <= 1e-6):
            raise ValueError(
                f"base.{name} disagrees with the base's own survival S by "
                f"{float(gap.max()):.2e} on z in [-6, 6]; it must describe "
                "the same law (min-wins, mean zero, unit variance)")
    try:
        _TAIL_CHECKED[fn] = key
    except TypeError:
        pass


def _is_named(fn):
    return any(fn is b for b in _NAMED_BASES)


def _tail_fn(fn, name):
    """The base's validated tail attribute `name` ('log_sf', 'cdf',
    'log_cdf'), or None. The named bases are trusted as written."""
    a = getattr(fn, name, None)
    if a is None or _is_named(fn):
        return a
    _check_tails(fn)
    return a


def _lower_cdf(fn):
    """z -> F(z) from a callable base's cdf or log_cdf attribute, or None
    when it carries neither. Of the named bases only the asymmetric
    Gumbel carries one (#710); the symmetric ones reflect."""
    if _is_named(fn):
        return getattr(fn, "cdf", None)
    cdf = _tail_fn(fn, "cdf")
    if cdf is not None:
        return lambda z: np.clip(np.asarray(cdf(z), dtype=float), 0.0, 1.0)
    lcdf = _tail_fn(fn, "log_cdf")
    if lcdf is not None:
        def from_log(z):
            with np.errstate(under="ignore", over="ignore",
                             invalid="ignore"):
                return np.minimum(np.exp(np.maximum(
                    np.asarray(lcdf(z), dtype=float), -1e300)), 1.0)
        return from_log
    return None


def _log_survival(fn, z, S):
    """log S for the race integrators: np.log(S), except at points where
    a named base floored S at 1e-300, which take the base's exact log
    survival (#502); a callable base's validated log_sf likewise."""
    logS = np.log(S)
    lsf = _tail_fn(fn, "log_sf")
    if lsf is not None:
        low = S <= 1e-300
        if low.any():
            # bounded below: an overflowing -z^2 (mu = 1e308) gave -inf,
            # and -inf - -inf is a NaN race; the density is exactly zero
            # wherever this bound binds, so it prices nothing
            with np.errstate(over="ignore", invalid="ignore"):
                exact = np.maximum(lsf(z), -1e300)
            logS = np.where(low, exact, logS)
    return logS


_NAMED_BASES = (_normal, _gumbel_min, _logistic, _laplace)
BASES = {"normal": _normal, "gumbel": _gumbel_min,
         "logistic": _logistic, "laplace": _laplace}


def _canonical_base(base):
    """The exported normal callable is the named law: map it to "normal" so
    it takes the same exact dispatch (pair closed form, Rust, GHK rescue)
    as the string. A wrapper or any other callable is left alone (#858)."""
    return "normal" if base is _normal else base
_SPANS = {"normal": (8.0, 8.0), "gumbel": (22.0, 8.0),   # (left, right) tails
          "logistic": (16.0, 16.0), "laplace": (18.0, 18.0)}
_RUST_BASE_IDS = {"gumbel": 1, "logistic": 2, "laplace": 3}


def exponential_power_base(beta):
    """Standardized exponential-power (generalised normal, Subbotin)
    base: density proportional to exp(-|x/a|^beta), with the scale a
    fixed for unit variance. beta = 1 is the laplace base, beta = 2 is
    exactly the normal, and large beta approaches the uniform while
    keeping full support -- so log S stays finite and the lattice
    machinery keeps its shape (issue 13: the tail exponent controls how
    fast the maximum of K draws grows, (log K)^(1/beta), which is the
    knob on whether systematic components keep deciding large fields).
    The CDF is the regularised incomplete gamma, evaluated from the
    tail side for precision. beta must be >= 1: below it the density
    has a cusp the lattice cannot resolve (#103). For beta < 2 the density has a kink at
    zero (its second derivative is singular), and the factory says so
    to the moment updates via the fd_eps attribute -- the laplace
    lesson. Very large beta makes near-edges the lattice must resolve;
    accuracy is measured in the tests at beta = 12."""
    beta = float(beta)
    # beta < 1 is refused, not approximated (#103). There the density
    # has a cusp, f' ~ |z|^(beta-1) is unbounded at the mode, and the
    # standardized centre is a spike of scale a = sqrt(G(1/b)/G(3/b))
    # (0.091 at beta = 0.5) under stretched-exponential tails, which no
    # uniform lattice over the race window resolves: beta = 0.5 carried
    # TV 0.058 at the default points and still 1.4e-3 at 4001, its own
    # slopes came out POSITIVE and disagreed with finite differences of
    # the shipped map in sign, and abilities_from_race did not converge;
    # beta = 0.7 was 1e-2 off with a wrong-signed slope. Supporting it
    # needs cusp-aware quadrature, which the lattice engines do not have.
    # beta in [1, 2) has only a kink. The default uniform lattice is
    # second order across it (laplace 4e-4 off at 257 points);
    # kink_split=True splits there (kink_lattice) and is ~1e-14.
    # beta = inf (the uniform limit) has no density this family can
    # evaluate and came back as NaN rows (#134); it is refused too.
    if not np.isfinite(beta) or beta < 1.0:
        raise ValueError(
            "exponential_power_base needs a finite beta >= 1; got " + repr(beta)
            + ". Below 1 the density has a cusp the uniform race lattice "
            "cannot resolve (accuracy and slope signs fail); use "
            "student_base or skew_logistic_base for heavier tails")
    from scipy.special import gamma as _gamma, gammaincc
    a = np.sqrt(_gamma(1.0 / beta) / _gamma(3.0 / beta))
    c = beta / (2.0 * a * _gamma(1.0 / beta))

    def _expo(z):
        z = np.asarray(z, dtype=float)
        t = np.power(np.abs(z) / a, beta)
        f = c * np.exp(-np.minimum(t, 745.0))
        half_tail = 0.5 * gammaincc(1.0 / beta, t)  # mass beyond |z|
        S = np.where(z >= 0.0, half_tail, 1.0 - half_tail)
        with np.errstate(divide="ignore", invalid="ignore"):
            fp = -f * beta * np.power(np.abs(z) / a, beta - 1.0) \
                * np.sign(z) / a
        fp = np.where(np.isfinite(fp), fp, 0.0)
        return np.maximum(S, 1e-300), f, fp

    from scipy.special import gammainccinv
    edge = a * float(gammainccinv(1.0 / beta, 2e-9)) ** (1.0 / beta)
    _expo.span = (max(10.0, edge + 2.0), max(10.0, edge + 2.0))
    _expo.rust_base = (4, [beta, a])
    if beta < 2.0:
        # kinked density: lattice gradients carry O(dx) noise, and the
        # moment updates must widen their differencing step exactly as
        # they do for the named laplace base
        _expo.fd_eps = 5e-2
        _expo.kinks = (0.0,)
    _expo.log_concave = beta >= 1.0

    def _expo_sample(rng, size=None):
        # density proportional to exp(-|x/a|^beta): the generalised
        # normal with shape beta and scale a (min-wins; symmetric)
        from scipy.stats import gennorm
        return gennorm.rvs(beta, scale=a, size=size, random_state=rng)
    _expo.sample = _expo_sample
    return _expo


def skew_logistic_base(alpha):
    """Standardized Type I generalized logistic base: CDF of the raw
    variable is (1 + e^{-x})^{-alpha}, so both the density and the
    distribution function are elementary, the density is smooth and
    log-concave for every alpha > 0, and alpha = 1 is exactly the
    logistic base. alpha < 1 skews the race toward heavy LEFT tails
    (occasional disasters, min-wins), alpha > 1 toward heavy right
    tails. Standardized to zero mean and unit variance in closed form:
    mean psi(alpha) - psi(1), variance psi'(alpha) + psi'(1). Returns a
    callable for base=; a natural one-parameter skew dial for joint
    market calibration."""
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("skew_logistic_base needs alpha > 0")
    from scipy.special import digamma, polygamma
    m = digamma(alpha) - digamma(1.0)
    v = polygamma(1, alpha) + polygamma(1, 1.0)
    c = np.sqrt(v)                          # raw x = m + c z

    def _softplus(u):
        # log(1 + e^u) without overflow or the 1 + tiny cancellation
        return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))

    def _skew_logistic(z):
        # Log domain throughout (#108). S = 1 - (1+e^{-x})^{-alpha} is
        # -expm1(-alpha * softplus(-x)); the subtraction from one lost the
        # whole right tail (alpha = 1, the named logistic, was 14% low at
        # a 25 sd gap and 52% low at 45), and the e^{-x} clip at 700 was a
        # second wall on the left.
        x = m + c * np.asarray(z, dtype=float)
        sp = _softplus(-x)                   # log(1 + e^{-x})
        S = np.maximum(-np.expm1(-alpha * sp), 1e-300)
        f = alpha * c * np.exp(-x - (alpha + 1.0) * sp)
        # 1 - sigmoid(x) directly, as sigmoid(-x): the subtraction lost
        # it near x ~ log(alpha), where the bracket below is a difference
        # of O(1) terms, and at alpha = 1e17 the own slopes came out
        # POSITIVE and the inverse walked away from its target (#499)
        sig_c = np.exp(-_softplus(x))         # 1 - logistic cdf of x
        # d/dz f: chain rule through x = m + c z
        fp = f * c * ((alpha + 1.0) * sig_c - 1.0)
        return S, f, fp

    def _log_expm1(y):
        # log(e^y - 1) for y > 0, finite for every y (no expm1 overflow)
        y = np.asarray(y, dtype=float)
        return y + np.log(-np.expm1(-y))

    # exponential tails on both sides; the left tail fattens as alpha
    # falls (raw left quantile ~ log of the alpha-th root), so place the
    # span at the actual 1e-9 quantiles. log(expm1(.)) overflowed to inf
    # below alpha ~ 0.029 and every race came back NaN (#108).
    # Both raw quantiles in closed form, signed and overflow-free (#499):
    # lower, G(x) = 1e-9 at x = -log(expm1(log(1e9) / alpha)); upper,
    # 1 - G(x) = 1e-9 at x = -log(expm1(-log1p(-1e-9) / alpha)), which is
    # log(alpha) + log(1e9) to working precision once alpha is large.
    # The old forms took -abs() of the lower one (the wrong side for
    # alpha > 1: a 1092-sd window at alpha = 1e299) and divided alpha by
    # 1e-9, which overflowed to an infinite span and NaN races from
    # alpha = 2e299.
    x_lo = -float(_log_expm1(np.log(1e9) / alpha))
    x_hi = (-float(_log_expm1(-np.log1p(-1e-9) / alpha)) if alpha < 1e6
            else float(np.log(alpha) + np.log(1e9)))
    lo_edge = abs(x_lo - m) / c + 2.0
    hi_edge = abs(x_hi - m) / c + 2.0
    _skew_logistic.span = (max(14.0, lo_edge), max(14.0, hi_edge))
    _skew_logistic.rust_base = (7, [alpha, m, c])

    def _skew_logistic_sample(rng, size=None):
        # raw x has CDF (1 + e^{-x})^{-alpha}: invert, then standardize.
        # -log(expm1(y)) overflowed to -inf whenever u < exp(-alpha*709),
        # half of all draws at alpha = 0.001 (#108).
        u = rng.random(size)
        x = -_log_expm1(-np.log(u) / alpha)
        return (x - m) / c
    _skew_logistic.sample = _skew_logistic_sample
    return _skew_logistic


def student_base(nu):
    """Standardized Student-t base (unit variance; needs nu > 2): fat
    tails for performances with occasional wild days. Returns a callable
    for base=; the tail span widens with falling nu automatically."""
    nu = float(nu)
    if np.isnan(nu) or nu <= 2.0:
        raise ValueError("student_base needs nu > 2 for unit variance")
    if np.isinf(nu):
        # s = sqrt(inf/inf) is NaN and every row came back NaN, while the
        # compiled kernel panicked (#134). The limit is the normal base.
        raise ValueError("student_base needs a finite nu; use base='normal' "
                         "for the nu -> inf limit")
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
    # central scale of the standardized law (#385): its density at the
    # mode is that of a scale-1/s t, so resolution is set by 1/s
    _student.resolution = 1.0 / s
    _student.rust_base = (5, [nu, s])
    _student.sample = lambda rng, size=None: rng.standard_t(nu, size) / s
    return _student


def skew_normal_base(a):
    """Standardized skew-normal base (the classic engine's heritage
    family), shape a: mean zero, unit variance. Returns a callable for
    base=."""
    a = float(a)
    if not np.isfinite(a):
        raise ValueError("skew_normal_base needs a finite shape a; got " + repr(a))
    from scipy.stats import skewnorm as _sn
    # hypot, not sqrt(1 + a*a): a*a overflows past |a| ~ 1.34e154 and
    # delta became 0, so a finite, saturated shape jumped from the
    # standardized half-normal to an unstandardized law -- the leader's
    # share moved 0.317 between a = 1e154 and 2e154 (#399)
    delta = a / np.hypot(1.0, a)
    m = delta * np.sqrt(2.0 / np.pi)
    sd = np.sqrt(1.0 - 2.0 * delta * delta / np.pi)

    def _skew(z):
        x = m + sd * np.asarray(z, dtype=float)
        S = np.maximum(_sn.sf(x, a), 1e-300)
        phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        Phi_ax = ndtr(a * x)
        f = 2.0 * phi * Phi_ax * sd
        # (a x)^2 may overflow to inf, whose exp(-inf/2) is the correct
        # 0; a*a*x*x at x = 0 was inf*0 = NaN
        with np.errstate(over="ignore", invalid="ignore"):
            ax2 = np.square(a * x)
            kink = np.where(np.isfinite(ax2), a * np.exp(-0.5 * ax2), 0.0)
        kink = np.where(x == 0.0, a, kink)
        fp = 2.0 * (-x * phi * Phi_ax
                    + kink * phi / np.sqrt(2.0 * np.pi)) * sd * sd
        return S, f, fp

    _skew.span = (10.0, 10.0)
    _skew.rust_base = (6, [a, m, sd])

    def _skew_sample(rng, size=None):
        x = _sn.rvs(a, size=size, random_state=rng)
        return (x - m) / sd
    _skew.sample = _skew_sample
    return _skew


# The factor-node rule, per rank: (Gauss-Hermite order cap, the sharpness
# past which even that order loses to scrambled Sobol at 2^13). `sharp` is
# the pairwise bound computed in _setup -- how many idiosyncratic standard
# deviations the factor swings the field by -- so it says how close the
# conditional race is to a step, which is what decides the family.
#
# One threshold of 3.0 for every rank used to stand here, and at rank 3 it
# was defending against a Gauss-Hermite cap of 15 that was itself WORSE
# than the Sobol rule it escalated to: at sharp 4, Q=15 carries total
# variation 4.4e-4 against Sobol's 1.3e-4, while Q=31 -- 4067 nodes after
# pruning, half of Sobol's 8192, and 29791 before it, inside the same
# budget -- carries 1.8e-5. The cap was the defect, not the family.
#
# Each threshold is the last sharpness at which Gauss-Hermite beat Sobol on
# EVERY field, over 8 seeds, against the mean of three 2^15 scrambles
# (n = 50; ratio = GH error / Sobol error, so below 1 is a GH win):
#
#   rank 2, 413 GH nodes            rank 3, 4067 GH nodes
#   sharp  median  worst   wins     sharp  median  worst   wins
#    2.7    0.12   0.16     8/8      3.1    0.15   0.22     8/8
#    3.2    0.15   0.23     8/8      3.7    0.17   0.23     8/8
#    3.7    0.18   0.75     8/8      4.2    0.26   0.35     8/8
#    4.3    0.38   1.73     7/8      4.7    0.40   0.73     8/8
#                                    5.3    0.62   1.35     6/8
#
# Field-to-field spread at fixed sharpness is wide -- at rank 3, sharp 5.3
# the median says Gauss-Hermite wins by 1.6x while the worst field loses by
# 1.35x -- so these are set by the worst case, not the median. An earlier
# cut of this table used 3-seed medians and put rank 3 at 6.0; a one-seed
# regression test found a field losing at 5.6.
#
# Rank >= 4 keeps the old 3.0: its tensor is 10929 nodes at Q=15, already
# dearer than Sobol's 8192, so there is no cheap side to reach for. At
# sharp 4, n = 250 it buys total variation 3.3e-4 against 5.0e-4 for 3.49s
# against 2.58s -- a trade, not a win, and not one to make silently.
#
# Rank 1 never escalates on sharpness (the branch is guarded r >= 2); it
# hands over to an equal-weight midpoint-quantile grid at Q > 80 instead.
#
# What this is worth, measured at n = 250: a rank-3 field at sharp 4 takes
# 1.50s where the escalation took 3.04s, for total variation 2.1e-5 where
# it was 1.6e-4. A realistic correlated field lands there -- a caller
# reported a calibration costing the same at rank 2 and rank 3 because
# both escalated -- and it is ONE runner's exposure that decides it, the
# statistic being a max.
#
# Ranks 1-3 are where the traffic is, which is why only they are tabled
# here. `r` is the rank of the V handed in, and that is normally a fitted
# factor model: a caller studying a five-factor market still races under
# a k-factor estimate with k of 1, 2 or 3, so the population's rank is
# not the one that reaches this rule.
GH_RULE = {1: (201, float("inf")), 2: (41, 3.75), 3: (31, 4.75)}
GH_RULE_DEFAULT = (15, 3.0)

# The rank-one handover from Gauss-Hermite to the midpoint-quantile grid
# was a hard switch at ceil(8 sharp) > 80, and the two rules disagree
# there: a 1e-7 change in one runner's sd moved a top-2 membership by
# 31bp (#783). Across (lo, hi] the rule is the weighted union of both,
# (1 - w) GH + w midpoint, with w a C1 smoothstep in the sharpness, the
# blend #420 uses for the likelihood's family switch. Each order is
# frozen at the value its side uses at the band's edge -- GH at 8 lo,
# the midpoint grid at 8 hi + 1, its first order past the switch -- so
# the band meets the unchanged rule continuously at both ends. The band
# sits on the Gauss-Hermite side: there the grid is the more accurate
# rule (GH 1e-3 to 6e-3 off at sharpness 9-10 on six fields where the
# grid at the same order was 2e-4 or better), so the blend leans the GH
# side toward the better rule, while the grid's own side is left exactly
# as it was.
# The cost inside the band is the two rules together, 72 + 81 nodes.
_R1_BLEND_LO, _R1_BLEND_HI = 9.0, 10.0


def _rank1_blend_rule(sharp):
    """The rank-one rule on (_R1_BLEND_LO, _R1_BLEND_HI] (#783)."""
    t = (sharp - _R1_BLEND_LO) / (_R1_BLEND_HI - _R1_BLEND_LO)
    w = t * t * (3.0 - 2.0 * t)
    Fg, Wg = hermite_nodes(1, Q=int(8 * _R1_BLEND_LO))
    Q = int(8 * _R1_BLEND_HI) + 1
    Fm = ndtri((np.arange(Q) + 0.5) / Q)[:, None]
    return (np.concatenate([Fg, Fm]),
            np.concatenate([(1.0 - w) * Wg, np.full(Q, w / Q)]))


# Below the band the rank-one rule was Gauss-Hermite of order
# ceil(8 sharp), and GH prices alternate with the parity of the order: on
# a near-step integrand GH(Q) and GH(Q + 1) err by similar amounts of
# opposite sign. At every 1/8 of sharpness the price jumped, by up to
# 2.2bp at sharpness 2, 40bp at 5 and 90bp just below 9 on seven fields
# (#850). Half of each cancels the alternation: the pair GH(m-1)/2 +
# GH(m)/2 at m = 5 sharp (at most 34) had 2-11x less worst-case error
# than GH(ceil 8 sharp) against a 4001-node midpoint reference over
# sharpness 4.5 to 8.875, and from 8.4 up it has fewer nodes. With
# x = 8 sharp:
#   x <= 15       GH(15), as before;
#   15 < x <= 24  (1 - w) GH(m) + w GH(m+1), m = floor x, w = S(x - m):
#                 the old orders interpolated, and the old rule itself at
#                 every integer x, where those orders are accurate enough
#                 that a pair would cost more than it gains;
#   24 < x <= 25  GH(24) takes on its partner GH(23), up to half weight;
#   x > 25        the pair at y = 5 sharp, held in [24, 34]: GH(m-1)/2 +
#                 GH(m)/2 at integer y, its GH(m-1) half moving to GH(m+1)
#                 as y crosses (m, m+1);
#   71 < x <= 72  handed back to GH(72), the band's lower edge, so the
#                 band is met continuously and is unchanged.
# S is the band's C1 smoothstep. The orders keep the cost within 2x of
# the old rule: interpolation pays 2m + 1 nodes where the old rule paid
# m + 1, the sliding pair 3m at m <= 5 sharp where it paid 8 sharp, and
# the cap at 34 keeps the hand-back (the pair plus GH(72)) at 139 nodes.
_R1_PAIR_FROM, _R1_PAIR_PER_SHARP, _R1_PAIR_CAP = 24, 5.0, 34


def _rank1_hermite_rule(sharp):
    """The rank-one rule on 15/8 < sharp <= _R1_BLEND_LO (#850)."""
    def S(t):
        return t * t * (3.0 - 2.0 * t)
    x = 8.0 * sharp
    lo = _R1_PAIR_FROM
    if x <= lo:
        m = int(np.floor(x))
        w = S(x - m)
        parts = [(m, 1.0 - w), (m + 1, w)]
    elif x <= lo + 1:
        w = S(x - lo)
        parts = [(lo - 1, 0.5 * w), (lo, 1.0 - 0.5 * w)]
    else:
        y = min(max(_R1_PAIR_PER_SHARP * sharp, float(lo)),
                float(_R1_PAIR_CAP))
        m = int(np.floor(y))
        w = S(y - m)
        parts = [(m - 1, 0.5 * (1.0 - w)), (m, 0.5), (m + 1, 0.5 * w)]
        top = int(8 * _R1_BLEND_LO)
        if x > top - 1:
            v = S(x - (top - 1))
            parts = [(q, c * (1.0 - v)) for q, c in parts] + [(top, v)]
    rules = [(hermite_nodes(1, Q=q), c) for q, c in parts if c > 0.0]
    return (np.concatenate([F for (F, _), _ in rules]),
            np.concatenate([c * W for (_, W), c in rules]))


def _translated_home(mu, D):
    """mu moved next to the origin when a common offset dwarfs the field.

    The origin is a gauge: every race probability depends on differences
    only. But the lattice is built in the caller's units, and at an
    offset of 1e15 its points collapse to the float spacing there and
    every share came back NaN (#477). Shifting by a middle entry (an
    element, so no sum can overflow) is exact for such offsets; below
    a million field scales nothing moves, so ordinary inputs stay
    bit-identical."""
    off = float(np.sort(mu)[len(mu) // 2])
    scale = float(mu.max() - mu.min()) + float(np.sqrt(np.max(D)))
    if abs(off) > 1e6 * scale:
        return mu - off
    return mu


def _as_abilities(mu):
    """mu as a float array, refused if empty or non-finite. mu was the one
    argument nobody checked: D goes through as_idio, V through
    as_loadings, W through as_weights, and the abilities themselves went
    straight to the lattice. A NaN or inf there came back as NaN
    probabilities that sum to nothing, silently. R and julia already
    refused it, which is how the cross-port divergence scan found it. The
    closed-form Luce helpers share this door (#799)."""
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    if n == 0:
        raise ValueError("mu is empty; a race needs at least one contestant")
    bad = np.flatnonzero(~np.isfinite(mu))
    if bad.size:
        raise ValueError(
            f"mu[{int(bad[0])}] = {float(mu[bad[0]])!r} is not finite "
            f"({bad.size} of {n} are); an ability is a finite location on "
            "the performance scale")
    return mu


def _setup(mu, V, D, F, W, base):
    caller_rule = F is not None
    mu = _as_abilities(mu)
    n = len(mu)
    D = np.ones(n) if D is None else as_idio(D, n, positive=True)
    mu = _translated_home(mu, D)                     # (#477)
    if (F is None) != (W is None):
        # A factor rule is a PAIR. Either half alone was silently replaced
        # by the automatic Gaussian rule below, so a caller's node vanished
        # and the default race came back (a 0.58 share move on a one-node
        # law), with the inverse calibrating a different model (#290).
        raise ValueError(
            "supply both F (factor nodes) and W (their weights), or "
            f"neither; got only {'F' if W is None else 'W'}")
    if V is None:
        V = np.zeros((n, 1))
        F, W = np.zeros((1, 1)), np.ones(1)
    else:
        V = as_loadings(V, n)
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
        V = _gauge_center(V)                          # scale-safe (#471)
        if (F is None) != (W is None):
            # A lone F or W used to be discarded silently and BOTH
            # regenerated (#75): the caller's rule never ran.
            raise ValueError(
                "supply both F (factor nodes) and W (their weights), or "
                f"neither; got only {'F' if W is None else 'W'}")
        if F is not None:
            # A supplied rule is checked before any backend dispatch: a
            # short W or a wrong-rank F used to reach the compiled kernel
            # and come back as an index PanicException, or as a late
            # matmul ValueError on the pure path (#75).
            F = np.asarray(F, dtype=float)
            if F.ndim != 2:
                raise ValueError(
                    f"F must be a 2-D (nodes, rank) array; got shape {F.shape}")
            W_arr = np.asarray(W, dtype=float)
            if W_arr.ndim != 1:
                raise ValueError(
                    f"W must be a 1-D weight vector; got shape {W_arr.shape}")
            if F.shape[1] != V.shape[1]:
                raise ValueError(
                    f"F has {F.shape[1]} factor columns but V has rank "
                    f"{V.shape[1]}; F is (nodes, rank)")
            if F.shape[0] != W_arr.shape[0]:
                raise ValueError(
                    f"F has {F.shape[0]} nodes but W has {W_arr.shape[0]} "
                    "weights; they must match one-to-one")
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
            with np.errstate(over="ignore"):     # inf is sharp (#86)
                sharp = float(np.sqrt(2.0)
                              * np.max(np.sqrt((V ** 2).sum(axis=1))
                                       / np.sqrt(np.maximum(D, 1e-300))))
            r = V.shape[1]
            cap, sharp_max = GH_RULE.get(r, GH_RULE_DEFAULT)
            if r >= 2 and sharp > sharp_max:
                # Past this sharpness the integrand is a near-step in
                # factor space and Gauss-Hermite loses to scrambled
                # Sobol even at the order the tensor budget affords.
                # Escalate the FAMILY, not the order. See GH_RULE for
                # the measurements behind each rank's number, and
                # papers/general_inversion/break.py section H; identical
                # rule in the R and browser ports (Halton there, to stay
                # dependency-free).
                from .core import qmc_nodes
                F, W = qmc_nodes(r, m=13)
            elif r == 1 and _R1_BLEND_LO < sharp <= _R1_BLEND_HI:
                F, W = _rank1_blend_rule(sharp)               # (#783)
            elif r == 1 and np.ceil(8.0 * sharp) > 80:
                # rank-1 sharpness (gap-stress find): Gauss-Hermite's
                # clustered nodes and wild weights are the wrong family
                # for a near-step integrand -- at sharp 100 the 201-node
                # GH rule carried TV 0.65 where an equal-weight midpoint-
                # quantile grid of the SAME size carried 6e-3. The
                # handover used to be at Q > 201 (sharp > 25); measured
                # on an 8-runner field against 3M-path truth, GH was
                # already 2.8e-3 off at sharp 15 (Q = 120) and 4.0e-3 at
                # sharp 21 (Q = 169) -- erratic, not slowly worsening --
                # while the midpoint grid at the same Q sat at the truth
                # floor (3.4e-4) throughout. GH is sound to about Q = 85
                # (sharp 10.5: 8.3e-4), so the grid takes over at Q > 80.
                # Scale the grid with sharpness, capped.
                Q = int(min(np.ceil(8.0 * sharp), 4001))
                u = (np.arange(Q) + 0.5) / Q
                F = ndtri(u)[:, None]
                W = np.full(Q, 1.0 / Q)
            elif r == 1 and 8.0 * sharp > 15:
                F, W = _rank1_hermite_rule(sharp)             # (#850)
            elif cap ** r > 100_000:
                # high-rank ecology footgun (bandits, 120 horses x 12
                # stable factors): the tensor grid is cap^r nodes BEFORE
                # pruning -- 2.6e9 at r=8, 944 TiB at r=12, a hard kill.
                # Past a 1e5-node tensor budget the rule escalates to
                # scrambled Sobol, as the likelihood module always did.
                # (The budget is measured on the order actually reachable,
                # the cap, so raising a cap cannot quietly breach it.)
                from .core import qmc_nodes
                F, W = qmc_nodes(r, m=13)
            else:
                Q = int(np.clip(np.ceil(8.0 * sharp), 15, cap))
                F, W = hermite_nodes(r, Q=Q)
    fn = base if callable(base) else BASES[base]
    left, right = _SPANS.get(base, (12.0, 12.0)) if not callable(base) \
        else getattr(base, "span", (12.0, 12.0))
    # W is a RELATIVE weight. W and c*W describe the same factor law, and
    # every verb normalises its own result, so the common scale cancels
    # -- except in the pre-normalisation mass checks, which compared an
    # unnormalised row sum against 1. removal_shares and
    # ordered_probabilities therefore REJECTED W = [5, 5] while accepting
    # the identical law W = [0.5, 0.5], with a "mass defect 9.00e+00"
    # that is just the weight total minus one (#208). Every rule built
    # above already sums to one, so this changes nothing the library
    # generates; it makes the caller's spelling not matter, which is the
    # contract everywhere else.
    # as_factor_law also drops zero-weight nodes, which are no-ops for
    # the integral but would otherwise widen every window built from F
    # (#416). Negligible caller nodes go too; the library's own rules
    # are left exactly as built.
    F, W = as_factor_law(F, W, negligible=caller_rule)
    return mu, V, D, F, W, fn, left, right




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
    Hardcoding the normal survival for every base clips polynomial tails:
    a Student-t(2.5) race at n=40 loses 5e-3 of total variation against
    the span window (fifth review). With the base supplied, the window
    adapts to the tail it is actually integrating.

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
    # the normal base needs only its survival here; calling _normal would
    # also build the density and its slope, discarded, ~160 times (#112).
    # The upper tail directly, not 1 - ndtr(z), which cancels past ~8 sd (#96)
    mu_lo = M_all.min(axis=0)
    mu_hi = M_all.max(axis=0)
    s = sd
    if fn is None or fn is _normal:
        # ~110 evaluations per call run in Python before any kernel, so
        # each is a fixed chain of in-place ufuncs on one buffer (#112).
        # Bit-identical to ndtr(-((x - m) / s)): IEEE subtraction and
        # division are sign-symmetric, so (m - x) / s == -((x - m) / s)
        # exactly, signed zeros included where ndtr is even.
        from scipy.special import ndtr as _ndtr
        buf = np.empty(np.broadcast(mu_lo, s).shape)
        sub, div, mx, log, exp = (np.subtract, np.divide, np.maximum,
                                  np.log, np.exp)

        def _envelope(x, m):
            sub(m, x, out=buf)
            div(buf, s, out=buf)
            _ndtr(buf, out=buf)
            mx(buf, 1e-300, out=buf)
            log(buf, out=buf)
            return 1.0 - exp(buf.sum())
    else:
        def _envelope(x, m):
            z = (x - m) / s
            logS = np.log(np.maximum(fn(z)[0], 1e-300))
            return 1.0 - np.exp(logS.sum())

    def G(x):
        # envelope: winner cdf using each runner's most favourable node
        return _envelope(x, mu_lo)

    def H(x):
        # right edge from the LEAST favourable nodes: no runner's density cut
        return _envelope(x, mu_hi)

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
            if m == a or m == b:
                break               # adjacent floats: the rest are no-ops (#112)
            if G(m) < d:
                a = m
            else:
                b = m
        xlo = a
        a, b = xlo, hi0
        for _ in range(80):
            m = 0.5 * (a + b)
            if m == a or m == b:
                break               # adjacent floats: the rest are no-ops (#112)
            if H(m) < 1.0 - d:
                a = m
            else:
                b = m
        return xlo - pad, b + pad

    # affordable width: keep the spacing under half the tightest
    # performance sd, the same resolution target the sharpness refinement
    # uses, so relaxation fires only when the tail would genuinely
    # outrun the lattice rather than whenever the budget is modest
    budget = 0.5 * float(s.min()) * _resolution(fn) * max(points - 1, 1)
    d = _as_delta(delta)
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
    if fn is None or fn is _normal:
        lo, hi, points = _rare_winner_edges(mu_lo, mu_hi, s, lo, hi, points)
    return np.linspace(lo, hi, points)


# Relative tail mass a runner's win integrand may leave outside the bulk
# window, and the most the window's point count may grow to cover it.
_EDGE_REL_TOL = 1e-6
_EDGE_POINTS_CAP = 4


def _edge_terms(x, m, s):
    """At the points x (shape (K,)): every runner's log win-integrand
    log g_i = log f_i + sum_{j != i} log S_j and its slope, for normal
    performances at locations m, sds s. Shapes (K, n); O(K n)."""
    z = (np.asarray(x, dtype=float)[:, None] - m[None, :]) / s[None, :]
    logS = _log_ndtr(-z)
    logphi = -0.5 * z * z - 0.5 * np.log(2.0 * np.pi)
    haz = np.exp(logphi - logS) / s[None, :]
    logg = logphi - np.log(s)[None, :] + logS.sum(axis=1, keepdims=True) - logS
    slope = -z / s[None, :] - haz.sum(axis=1, keepdims=True) + haz
    return logg, slope


def _rare_winner_edges(mu_lo, mu_hi, s, lo, hi, points):
    """Post-check the bulk window's edges against each runner's OWN win
    integrand, and widen within a bounded cap (#97).

    [G^-1(delta), H^-1(1 - delta)] holds all but 2 delta of the WINNER
    distribution, which bounds every share's error by delta absolutely
    but not relatively: at mu = [0, 20, 40] the middle runner wins near
    x = 10, past the right edge, and came back at 0.099 of its 1.04e-45.
    For a log-concave (normal) integrand, the mass beyond an edge is at
    most g(edge) / |slope| (tangent bound) and the mass inside is at
    least the chord integral over any inner stretch next to the edge
    (the best of a geometric ladder of lengths is used); a runner whose
    ratio exceeds _EDGE_REL_TOL (or whose integrand still rises at the
    edge) is truncated. The edge then moves out until every runner
    passes, with the point count scaled to keep the spacing, up to
    _EDGE_POINTS_CAP times the request; past the cap the window stops
    there and a warning says so. One O(13 n) probe per edge; ordinary
    fields pass it and are unchanged."""
    width0 = hi - lo
    if not (width0 > 0) or len(s) < 2:
        return lo, hi, points
    spacing = width0 / max(points - 1, 1)
    max_width = spacing * (_EDGE_POINTS_CAP * points - 1)
    ws = 0.5 * width0 * 0.25 ** np.arange(6)
    log_ws = np.log(ws)[:, None]

    def bad(edge, m, sign):
        # sign +1: right edge (integrand must FALL outward), -1: left
        logg_all, slope_all = _edge_terms(np.r_[edge, edge - sign * ws], m, s)
        logg, out_slope = logg_all[0], sign * slope_all[0]
        logg_in = logg_all[1:]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            tail = logg - np.log(np.maximum(-out_slope, 1e-300))
            d = np.abs(logg_in - logg[None, :])
            big = np.maximum(logg_in, logg[None, :])
            # log of the chord integral w (e^A - e^B) / (A - B)
            c = np.where(d < 1e-12, logg[None, :] + log_ws,
                         big + log_ws + np.log(-np.expm1(-d))
                         - np.log(np.maximum(d, 1e-300)))
            ratio = tail - c.max(axis=0)
        # NaN (an unrepresentable field) is not evidence of truncation
        return (out_slope >= 0) | (ratio >= np.log(_EDGE_REL_TOL))

    new_lo, new_hi = lo, hi
    for sign, m in ((+1, mu_hi), (-1, mu_lo)):
        edge = new_hi if sign > 0 else new_lo
        if not bad(edge, m, sign).any():
            continue
        step = float(s.max())
        for _ in range(60):
            cand = edge + sign * step
            if (max(new_hi, cand) - min(new_lo, cand)) > max_width:
                edge = (new_lo + max_width) if sign > 0 else (new_hi - max_width)
                import warnings
                warnings.warn(
                    "bulk window: a rare winner's integrand extends past "
                    f"the window even at {_EDGE_POINTS_CAP}x the point "
                    "budget, so its share is truncated (relatively; the "
                    "absolute error stays below delta). Raise points= or "
                    "use window='span' for relative accuracy on such "
                    "shares.", RuntimeWarning, stacklevel=4)
                break
            edge = cand
            if not bad(edge, m, sign).any():
                break
            step *= 2.0
        if sign > 0:
            new_hi = edge
        else:
            new_lo = edge
    if (new_lo, new_hi) == (lo, hi):
        return lo, hi, points
    pts = int(np.ceil((new_hi - new_lo) / spacing)) + 1
    return new_lo, new_hi, max(points, min(pts, _EDGE_POINTS_CAP * points))


def _resolution(fn):
    """The base's central scale in standardized units: the spacing target
    is half of it times the performance sd. A unit-variance law can
    still have a narrow centre -- Student-t(nu) concentrates on
    sqrt((nu-2)/nu), 0.218 at nu = 2.1 -- and measuring resolution in
    sd alone let the window widen until about two points crossed it
    (#385). Bases declare it as .resolution; the default is 1."""
    r = getattr(fn, "resolution", 1.0) if fn is not None else 1.0
    return float(r) if np.isfinite(r) and r > 0 else 1.0


# Above this many n * Q entries the Rust front door skips the Q x n
# location matrix and windows from per-runner extremes (a module
# constant so tests can move the threshold, #117).
_LARGE_DISPATCH_ENTRIES = 2e7


def kink_lattice(x, M, sd, kinks):
    """Nodes and weights over the window of the uniform lattice x, split
    at every runner's kink and Gauss-Legendre on each piece. Used only
    on the opt-in kink_split=True path.

    A uniform rectangle sum is spectrally accurate only for a smooth
    integrand. A base whose density has a kink (laplace, and the
    exponential power family below beta = 2) makes the winner integrand
    kinked at every runner's location, and the rectangle sum is then
    second order: a four-runner laplace race was 4e-4 off at 257 points
    and 2e-6 off at 4097, against winning.certified. Each piece between
    kinks is analytic, so Gauss-Legendre on the pieces is spectral again,
    and at the same point budget the error falls to the window's own
    truncation (~1e-12 at the default delta).

    M: this node's runner locations. The budget len(x) is shared in
    proportion to piece length, the two outer pieces (exponential tails)
    at a quarter weight, with at least four nodes per piece.

    Cost bound: at most len(x) // 4 interior kinks are kept. A larger
    field's kinks are merged onto that many equal-width bins (one kink
    kept per bin), so the node count stays under about 2 * len(x) however
    many runners there are; a merged kink is integrated through at the
    rectangle rule's order, which is the default path's accuracy anyway.
    Small fields (the ones the split is for) are never merged.
    """
    a, b = float(x[0]), float(x[-1])
    L = len(x)
    K = (M[:, None] + np.asarray(kinks, dtype=float)[None, :]
         * sd[:, None]).ravel()
    K = np.unique(K[(K > a) & (K < b)])
    cap = max(1, L // 4)
    if len(K) > cap:
        width = (b - a) / cap
        _, first = np.unique(np.floor((K - a) / width), return_index=True)
        K = K[first]
    cuts = np.concatenate([[a], K, [b]])
    lo, hi = cuts[:-1], cuts[1:]
    eff = hi - lo
    eff[0] *= 0.25
    eff[-1] *= 0.25
    count = np.maximum(4, np.round(L * eff / eff.sum()).astype(int))
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    xs, ws = [], []
    for m in np.unique(count):
        sel = count == m
        t, w = _legendre(int(m))
        xs.append((mid[sel, None] + half[sel, None] * t[None, :]).ravel())
        ws.append((half[sel, None] * w[None, :]).ravel())
    return np.concatenate(xs), np.concatenate(ws)


_LEGENDRE = {}


def _legendre(m):
    if m not in _LEGENDRE:
        _LEGENDRE[m] = np.polynomial.legendre.leggauss(m)
    return _LEGENDRE[m]


def _as_window(window):
    """The lattice window: "bulk" or "span", nothing else. Every other
    value used to mean span, so window="bulkk" moved a share by 39 points
    against the bulk default it misspelled (#582)."""
    if not isinstance(window, str) or window not in ("bulk", "span"):
        raise ValueError(f'window must be "bulk" or "span"; got {window!r}')
    return window


def _as_delta(delta):
    """The bulk window's tail mass: a real number in (0, 0.5). It was
    unvalidated: delta=-1 made the relaxation loop multiply a negative
    number by 100 forever (a hang), delta=1 or 2 put both quantiles past
    each other and priced a different race, and NaN returned NaN (#445)."""
    if (isinstance(delta, (bool, np.bool_))
            or not isinstance(delta, (int, float, np.integer, np.floating))):
        raise ValueError(f"delta must be a number in (0, 0.5); got {delta!r}")
    d = float(delta)
    if not (np.isfinite(d) and 0.0 < d < 0.5):
        raise ValueError(f"delta must be a number in (0, 0.5); got {delta!r}")
    return d


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
    _as_window(window)
    delta = _as_delta(delta)                          # (#445)
    if window == "bulk":
        x = _bulk_window(M_all, sd, points, delta, fn)
    else:
        x = np.linspace(M_all.min() - left * sd.max(),
                        M_all.max() + right * sd.max(), points)
    return _refine_grid(x, sd, V, points, stacklevel=3, fn=fn)


def _refine_grid(x, sd, V, points, stacklevel=2, fn=None):
    """Sharpness refinement and the spacing warning, for ANY window.

    Split out of forward_grid so that the memory-saving large-field
    Rust branch of race_probabilities, which builds its window from
    per-runner extremes instead of the Q x n location matrix, applies
    the same discretization rule (#117). It used to call _bulk_window
    and go straight to the kernel with the requested points, so
    crossing the n*Q threshold silently dropped the refinement (TV
    2.5e-2 on a scaled sharp field). Returns (x, points)."""
    sd = np.asarray(sd, dtype=float)
    # the window may arrive longer than requested: the rare-winner edge
    # check widens it at the requested spacing (#97)
    points = max(points, len(x))
    dx = x[1] - x[0]
    smin = float(sd.min())
    res = _resolution(fn)
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
                "simulation.", RuntimeWarning, stacklevel=stacklevel)
    # A correctly bracketed window can still be unusable: a polynomial
    # tail pushes the delta-quantile so far out that the points are
    # spread too thin to resolve the bulk. A truncated window hides
    # that; an honest one has to say so.
    if (x[-1] - x[0]) / max(len(x) - 1, 1) > 0.5 * smin * res:
        import warnings
        warnings.warn(
            f"lattice spacing {(x[-1]-x[0])/max(len(x)-1,1):.3g} exceeds "
            f"half the smallest performance sd times the base's central "
            f"scale ({smin:.3g} x {res:.3g}): this base's "
            "tail forced a window wider than the point budget can "
            "resolve. Raise points=, raise delta=, or declare a span on "
            "the base.", RuntimeWarning, stacklevel=stacklevel)
    return x, points


def _fit_cov(cov, structure, V, D, stacklevel=2, will_route=False):
    """Fit a dense cov= to the factor grammar, with the accuracy warnings.

    Shared by race_probabilities and ordered_probabilities so the two
    front doors accept the same covariance descriptions on the same
    terms -- and warn about the same fits. Returns (V, D, F, W, degraded):
    `degraded` is True when either failure class is present -- the fit
    reproduces cov badly (residual warnings), or it reproduces cov by
    degenerating to full rank with D on its floor (quadrature cannot
    resolve it). A caller that can route around a degraded fit passes
    will_route=True and gets no warnings, because it will not use the fit.
    """
    if structure is not None or V is not None or D is not None:
        raise ValueError("cov= replaces structure=/V=/D=; pass one only")
    from .core import fit_covariance
    import warnings
    with warnings.catch_warnings():
        # this function words the contrast warning itself, and says
        # nothing when will_route sends the call elsewhere
        warnings.filterwarnings("ignore", message="fit_covariance did not "
                                "hold", category=RuntimeWarning)
        V, D, F, W, report = fit_covariance(cov, return_report=True)
    C = np.asarray(cov, dtype=float)
    n = len(C)
    diag = np.diag(C)
    floor = 1e-6 * np.maximum(diag, 1e-6 * float(diag.mean()))
    bound = int((D <= 2.0 * floor).sum())
    degenerate = bool(bound or report["rank"] >= n)
    misfit = (report.get("contrast_residual_max", 0.0) > 0.05
              or report["projected_residual_max"] > 0.05)
    degraded = degenerate or misfit
    if will_route and degraded:
        return V, D, F, W, True
    if degenerate:
        # The residual checks below judge how well V V' + D reproduces
        # cov, and they do fire on a dense cov the grammar fits badly. The
        # case they MISS is the opposite one: the fit reproduces cov to
        # ~1e-6 by going to full rank with D on its floor, and the
        # conditional race is then a near-step the factor nodes cannot
        # resolve. An exactly 3-factor correlation at n = 8 does this
        # (k + blocks + m directions sum to n) and prices 6.6e-3 off its
        # own exact V/D answer with every residual check silent. The
        # signal for that is the fit's shape, not its residual.
        warnings.warn(
            f"cov= fit degenerated: rank {report['rank']} of {n}"
            + (f", idiosyncratic floor bound on {bound} of {n} entries"
               if bound else "")
            + ". The conditional race is near-deterministic there and the "
            "factor quadrature cannot resolve it; the residual checks are "
            "satisfied and do not see this. Expect percent-level error in "
            "the probabilities (6.6e-3 measured on an exactly 3-factor "
            "correlation at n=8). The forward normal race and "
            "abilities_from_race route this case to GHK automatically; "
            "this call cannot (slopes, ordered prefixes, a temperature or "
            "a non-normal base need the factor form). For win "
            "probabilities alone: winning.methods.get_method('qmc_ghk')"
            "(-mu, numpy.linalg.cholesky(cov), numpy.zeros(n)) (max-wins).",
            RuntimeWarning, stacklevel=stacklevel)
    if report.get("contrast_residual_max", 0.0) > 0.05:
        warnings.warn(
            "cov= carries a nearly singular contrast the fit did not "
            "hold: the worst pairwise difference variance is off by "
            f"{report['contrast_residual_max']:.2f} of its own size. "
            "Global residual norms cannot see this (choice "
            "probabilities become infinitely sensitive as a contrast "
            "variance approaches zero), so head-to-head probabilities "
            "between the affected pair may be badly wrong even though "
            "the covariance residual looks small.",
            RuntimeWarning, stacklevel=stacklevel)
    if report["projected_residual_max"] > 0.05:
        warnings.warn(
            "cov= is imperfectly served by the grammar fit (worst "
            "choice-relevant residual entry "
            f"{report['projected_residual_max']:.2f} of the average "
            "variance; short-length-scale/locality covariances are the "
            "known hard family). Probabilities may carry percent-level "
            "bias; see the paper's dense-covariance section.",
            RuntimeWarning, stacklevel=stacklevel)
    elif report["rank"] > 12 and report["sharpness"] > 5:
        warnings.warn(
            "cov= fits well but needs a high-rank, sharp factor "
            f"integral (rank {report['rank']}, sharpness "
            f"{report['sharpness']:.0f}); the default node budget may "
            "leave percent-level quadrature error. Pass more nodes "
            "(fit_covariance(..., nodes_log2=14)) or price by "
            "simulation for near-singular smooth covariances.",
            RuntimeWarning, stacklevel=stacklevel)
    return V, D, F, W, degraded


def _race_dense(mu, cov, budget=4096, seed=0, return_slopes=False,
                log=False):
    """The dense-covariance race by scrambled-Sobol GHK: exact structure not
    required, deterministic at a fixed seed, smooth in its inputs.

    GHK conditions the runners sequentially, so its error depends on the
    order they are listed in, and a fixed point set made the answer
    label-dependent (#162: 8.8e-3 between two labelings of one 8-runner
    field). The runners are therefore sorted into a canonical order --
    by ability, ties by covariance row -- before the estimate and the
    answer is mapped back, which makes the route exactly permutation-
    equivariant. That order also happens to be the accurate one: on the
    #162 field, versus 12M-path Monte Carlo, 5.6e-3 as listed became
    2.4e-3 sorted at 1024 nodes; 4096 nodes gives 7.2e-4 in 13 ms
    (2^14: 1.6e-4 in 46 ms), so 4096 is the budget, well inside the
    2e-3 the #118 pins hold it to. Earlier measurement on dense
    correlations at n=16 / n=30 against 4M-path Monte Carlo: 4.7e-4 /
    5.0e-4 at 1024 nodes where the fitted lattice was 8.9e-3 / 6.5e-3
    and no amount of extra fitting closed the gap (more factors send D
    to its floor and the quadrature error returns).
    The methods are max-wins, so the min-wins mu is negated and the
    own-slopes (return_slopes=True: dp_i/dmu_i, negative) with it.
    log=True returns (log p, d log p_i / d mu_i) instead, finite where p
    underflows: on a near-singular n=30 correlation a runner displaced
    from its near-duplicates has log p = -16000 at the inverse's warm
    start, and that is a real value the Newton step recovers from, not
    a zero.
    """
    from ..methods.native import qmc_ghk
    C = np.asarray(cov, dtype=float)
    m = np.asarray(mu, dtype=float)
    n = len(C)
    order = np.lexsort((C.sum(axis=1), m))
    Cs = C[np.ix_(order, order)]
    # The Cholesky is a POSITIVE-DEFINITENESS TEST here, not a change of
    # variables: the covariance goes to GHK as itself. Factoring it and
    # letting qmc_ghk rebuild `L @ L.T` lost the choice-relevant
    # eigenvalue whenever an unidentifiable common mode dominated --
    # `I + a 11'` is the same race as `I` for every `a`, and the winner
    # moved by 0.033 at a = 4e15 while the contrast variance stayed
    # exactly 2.0 (#302). The R port always formed the contrast from
    # the original matrix.
    try:
        np.linalg.cholesky(Cs)
    except np.linalg.LinAlgError:
        Cs = Cs + 1e-10 * float(np.trace(Cs)) / n * np.eye(n)
    ps, info = qmc_ghk(-m[order], None, None, budget=int(budget),
                       seed=int(seed), return_slopes=return_slopes, cov=Cs)
    p = np.empty(n)
    p[order] = np.asarray(ps, dtype=float)
    if log:
        lp = np.empty(n)
        lp[order] = np.asarray(info["logp"], dtype=float)
        if not return_slopes:
            return lp
        dl = np.empty(n)
        dl[order] = -np.asarray(info["dlogp"], dtype=float)
        return lp, dl
    if return_slopes:
        sl = np.empty(n)
        sl[order] = -np.asarray(info["slopes"], dtype=float)
        return p, sl
    return p


def _factor_of_structure(structure, verb):
    """The (V, D) of a structure that the factor kernels can price directly.

    Independent and Factor ARE the factor form. The block/nested/tree
    kernels are separate O(N)-per-point recursions with no (V, D) of
    matching rank, so a verb that has only the factor lattice refuses
    them by name instead of pricing a different race (issue #66 asked
    why the grammars are not uniform across the family; this says where
    the boundary is).
    """
    from .structures import Factor, Independent
    if isinstance(structure, Independent):
        return None, np.asarray(structure.D, float)
    if isinstance(structure, Factor):
        return (np.asarray(structure.V, float),
                np.asarray(structure.D, float))
    raise NotImplementedError(
        f"{verb} prices the factor lattice, so structure="
        f"{type(structure).__name__} is not available on it -- only "
        "Independent and Factor, which are the factor form itself. The "
        "block/nested/tree kernels are win-race recursions with no "
        "equivalent ordered-prefix pass; fit the structure to a factor "
        "model (winning.factor.core.fit_covariance on its covariance) "
        "if you need ordered prefixes under it.")


def _as_temperature(temperature):
    """0 is the hard-race sentinel; a positive finite value tempers.

    Every branch read `if temperature and temperature > 0`, so a NEGATIVE
    or NaN temperature fell through to the hard race and returned the
    tau = 0 answer byte for byte, and +inf reached the tempered path and
    died in the grid sizing (#424). A bad calibration parameter should
    not survive as a silently different model.
    """
    if temperature is None:          # None is the hard race too (#366)
        return 0.0
    try:
        tau = float(temperature)
    except (TypeError, ValueError):
        raise ValueError(f"temperature must be a number; got {temperature!r}")
    if not np.isfinite(tau) or tau < 0.0:
        raise ValueError(
            f"temperature must be finite and non-negative (0 is the hard "
            f"race); got {temperature!r}")
    return tau


def _massless_lattice(mu, V, D, base, nodes_given, return_slopes):
    """The factor lattice captured NO winner mass (total 0 or non-finite).

    That happens when every conditional race is far sharper than the
    lattice spacing: a posterior whose idiosyncratic variances have
    collapsed to ~1e-17 under O(1) loadings (the #669 field) leaves each
    of the 8,192 conditional winner densities ~1e-9 wide, and whether
    any of them lands near a lattice point is luck. Normalizing then
    divided 0 by 0 and returned NaN for every runner -- on Linux and
    not on macOS, because the luck depends on the last bits of the
    window. A normal race under the library's own factor rule is still
    an ordinary Gaussian race, so it is priced on the dense GHK route
    (the one cov= uses) with a warning; anything else refuses. Only
    reached when the lattice would otherwise return NaN, so every field
    that priced before prices identically."""
    import warnings
    if isinstance(base, str) and base == "normal" and not nodes_given:
        warnings.warn(
            "race_probabilities: the factor lattice captured no winner "
            "mass (conditional races far sharper than the lattice: min "
            f"idiosyncratic sd {float(np.sqrt(np.min(D))):.1e}); priced "
            "by the dense-covariance GHK route instead.",
            RuntimeWarning, stacklevel=3)
        V = np.asarray(V, dtype=float).reshape(len(mu), -1)
        cov = V @ V.T + np.diag(np.asarray(D, dtype=float))
        return _race_dense(mu, cov, return_slopes=return_slopes)
    raise FloatingPointError(
        "race_probabilities: the factor lattice captured no winner mass "
        f"(min idiosyncratic sd {float(np.sqrt(np.min(D))):.1e} against "
        "O(1) loadings), so every share would be 0/0 = NaN. This is the "
        "near-deterministic regime; raise D, or pass the dense covariance "
        "as cov= for a normal base.")


def race_probabilities(mu, V=None, D=None, F=None, W=None, base="normal",
                       points=257, temperature=0.0, return_slopes=False,
                       structure=None, window="bulk", delta=1e-12, cov=None,
                       kink_split=False):
    """Win probabilities of the general race, all N in one field pass.

    Pass `structure=` (Independent/Factor/Blocks/Nested/Tree from
    winning.factor.structures) to describe the covariance declaratively --
    one race, five grammars; V=/D= remain as sugar for the factor case.
    V is (n, rank), one row per contestant; a scalar, a length-n vector
    (rank one) and an (rank, n) matrix are all normalised to it, and any
    other shape raises rather than reaching the compiled kernel. A square
    (n, n) V is ALWAYS read as (n, rank): no shape rule can tell it from
    its transpose, and V.T describes the different covariance V'V, so the
    transposed shorthand is available only when rank != n (#71).
    Pass `cov=` (a dense covariance or correlation matrix) to have it
    fitted to the grammar first via winning.factor.core.fit_covariance
    (approximate: the fit residual is the price of density; see the
    paper's dense-Sigma section for measured accuracy by ensemble).
    temperature > 0 returns the softmin expectation E[softmin(X/tau)],
    computed exactly as the hard race with each base convolved with the
    tau-scaled min-Gumbel kernel.

    Cost is set by the FIELD as much as by the rank. The node rule
    (GH_RULE, above _setup) picks a Gauss-Hermite tensor while it fits a
    1e5-node budget and the field is not too sharp, and scrambled Sobol
    at 8,192 nodes otherwise, so the node count caps and the cost stops
    growing with rank: measured at K = 8 on a mild field, a forward pass
    takes 12 ms at rank 3, 86 ms at rank 4 and 65 ms at ranks 5 and 6,
    where the budget caps it.

    The rank here is the rank of the V you PASS, which is usually a
    FITTED one -- and a fitted factor model is nearly always low rank
    even when the population it came from is not. A caller whose market
    has five factors, but who races under a k-factor estimate from a
    short panel, is running at rank 1, 2 or 3 and lives entirely on this
    rule's cheap side. Read the table for the rank you race at, not the
    rank of the thing you are modelling; it is easy to size this wrong
    in the pessimistic direction (done, in review, by the author).

    The other axis is `sharp`, how many idiosyncratic standard deviations
    the factor swings the field by. A realistic correlated field crosses
    its rank's threshold and takes the capped 8,192 nodes whatever its
    rank -- a caller reported a calibration costing the same at rank 2
    and rank 3 for exactly this reason, and it is ONE runner's exposure
    that decides it, since the statistic is a max. So rank is the wrong
    thing to optimise for real inputs; if the capped cost is too high,
    pass F, W from winning.factor.core.qmc_nodes(r, m=11) for a quarter
    of the nodes at about five times the error. Inversion runs the
    forward pass per Newton step, so it scales the same way.

    kink_split: opt-in, default False (the uniform lattice, unchanged).
    A base whose density has a kink -- laplace, and exponential_power_base
    with beta < 2 -- makes the winner integrand kinked at every runner's
    location, and the uniform rectangle sum is then only second order: a
    four-runner laplace race is 4e-4 off at the default 257 points.
    kink_split=True integrates on kink_lattice instead (Gauss-Legendre on
    the pieces between kinks), which brings that race to ~4e-15 at the
    same budget. At most points // 4 kinks are kept (a larger field's
    are merged onto that many bins), so the node count stays under about
    2 * points however many runners there are. The path is numpy only
    (no compiled kernel): measured without fastrace it costs 1.0-1.7x
    the default, from four runners to a thousand at rank two, and more
    than that against a compiled default. Smooth bases ignore it, as
    does a tempered race (the softmin kernel smooths the kink); with
    structure= it raises."""
    base = _canonical_base(base)                             # (#858)
    # the lattice mode is one of two names: every value but "bulk" used
    # to mean span, so window="bulkk" moved a share by 39 points (#582)
    if not (isinstance(window, str) and window in ("bulk", "span")):
        raise ValueError(
            f"window must be 'bulk' or 'span'; got {window!r}")
    # finite and >= 0; negative and NaN used to select the hard race
    # silently, and +inf failed in grid sizing (#366)
    temperature = as_soft_temperature(temperature)
    nodes_given = F is not None          # the caller's nodes, not a fit's
    temperature = _as_temperature(temperature)
    points = as_points(points)
    window = _as_window(window)                      # (#582)
    delta = _as_delta(delta)                          # (#445)
    if cov is not None:
        # The forward normal race with no slopes is the one case that can
        # be answered without the fit at all; everything else (slopes for
        # the inverter, a non-normal base, a tempered race) needs the
        # factor form and keeps the fit with its warnings.
        routable = (base == "normal" and not temperature and not return_slopes)
        V, D, F, W, degraded = _fit_cov(cov, structure, V, D, stacklevel=3,
                                        will_route=routable)
        if degraded and routable:
            return _race_dense(mu, cov)
    if structure is not None:
        # structure= DESCRIBES the covariance; V/D/F/W describe it too,
        # and passing both asked two questions and answered one of them
        # silently. Forward dropped the caller's V/D while the inverse
        # could keep V and replace D from the structure, so a target
        # inverted one way did not reprice the other (#89).
        conflicting = [n for n, v in (("V", V), ("D", D), ("F", F),
                                      ("W", W)) if v is not None]
        if conflicting and cov is None:
            raise ValueError(
                f"structure= already describes the covariance; "
                f"{', '.join(conflicting)}= would describe it again. "
                "Pass one or the other.")
        if kink_split:
            raise ValueError("kink_split applies to the factor lattice "
                             "(V=/D=), not to structure=")
        from .structures import dispatch_probabilities
        # points/window/delta are the NUMERICAL controls, and dropping
        # them meant a caller asking for more resolution got the
        # default: points=17, 257 and 2049 returned the identical
        # answer on a Blocks race, and a points= too coarse to resolve
        # the field never raised the mass defect it should have (#89).
        return dispatch_probabilities(mu, structure, base=base,
                                      temperature=temperature,
                                      return_slopes=return_slopes,
                                      points=points, window=window,
                                      delta=delta)
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    if temperature > 0:
        return _race_tempered(mu, V, D, F, W, fn, left, right,
                              float(temperature), points, return_slopes)
    sd = np.sqrt(D)
    n = len(mu)
    if base == "normal" and n == 2:
        # A two-runner normal race is a SINGLE Gaussian contrast, so the
        # lattice has a closed form and does not need to be built. With
        # Sigma = V V' + diag(D), X0 - X1 is normal with variance
        # Sig00 + Sig11 - 2 Sig01, and min-wins gives
        # p0 = Phi((mu1 - mu0) / sd_d).
        #
        # This is not only cheaper but TIGHTER: measured against this
        # closed form the 257-point path carries ~1e-9 of discretization
        # error. It also skips _bulk_window, whose 160 scalar bisections
        # run in Python BEFORE any compiled kernel is reached -- on the
        # head-to-head filter (every contest is a pair) that window
        # search, not the forward pass, was the single largest cost.
        #
        # Under the nodes the caller supplied, F, W ARE the factor law,
        # which need not be normal: the pair is then the exact mixture
        # over the nodes, as the general lattice computes it (#742: a
        # centred two-point law priced 0.5 for an exact 0.75 when only
        # its mean and covariance were matched). The default rule's
        # nodes, and a covariance fit's, stand for the standard normal
        # itself, so the closed form below uses the identity covariance,
        # not the pruned tensor's (3e-5 short of it: a fitted
        # equicorrelated pair priced 3.6e-5 off its analytic value
        # through the nodes).
        dV = V[0] - V[1]
        if nodes_given:
            p, dens = _pair_mixture(mu[1] - mu[0], dV, D[0], D[1], F, W)
            if return_slopes:
                return p, np.array([-dens, -dens])
            return p
        var_d = None
        shift = 0.0
        # The contrast sd in units of the largest term, so it neither
        # overflows (D = [9e307, 9e307] summed to inf and priced [.5, .5])
        # nor meets an absolute floor (max(var, 1e-300) changed the model
        # below contrast sd 1e-150: 0.540 for an exact 0.760, #100).
        if var_d is None:
            with np.errstate(over="ignore"):
                shared = float(dV @ dV)
            if np.isfinite(shared):
                sd_d = _contrast_sd(shared, D[0], D[1])
            else:
                # finite loadings whose squared contrast overflows
                # (V = +-1e155): the norm in units of its largest term,
                # never squared at full scale; [1, 0] for an exact
                # [0.69, 0.31] before (#86)
                m = max(float(np.abs(dV).max()), float(np.sqrt(D.max())))
                big_unit = (m, float(np.sqrt(np.sum((dV / m) ** 2)
                                             + D[0] / m / m + D[1] / m / m)))
                with np.errstate(over="ignore"):
                    sd_d = big_unit[0] * big_unit[1]
        else:
            sd_d = np.sqrt(max(var_d, 0.0))
        if not sd_d > 0.0:
            sd_d = 1e-150     # an exactly deterministic contrast: as before
        if var_d is None and not np.isfinite(shared):
            # the sd itself may exceed the float range: u in its unit
            u = float((mu[1] / big_unit[0] - mu[0] / big_unit[0]
                       - shift / big_unit[0]) / big_unit[1])
        else:
            # runner i performs mu_i + V_i . F + e_i, so the node mean
            # raises the contrast X1 - X0 by -(V0 - V1) . E[F] = -shift.
            # It was added: a caller's node law with mean 1.43 priced an
            # exact [0.128, 0.872] as [0.936, 0.064], and the closed-form
            # inverse (which had the sign right) did not round-trip.
            u = float((mu[1] - mu[0] - shift) / sd_d)
        # Each tail directly: 1 - ndtr(u) rounds the loser to exactly 0 at
        # a 9 sd contrast where the true value is 1.1e-19, and the pair
        # then depends on which runner is listed first. ndtr(u), ndtr(-u)
        # keep full relative precision in both tails and are exactly
        # permutation-equivariant (found in review).
        p = np.array([float(ndtr(u)), float(ndtr(-u))])
        if return_slopes:
            # dp_i/dmu_i, the own-slope the inverter preconditions with;
            # equal for both runners on a pair, and negative because the
            # race is min-wins.
            dens = float(np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)) / sd_d
            return p, np.array([-dens, -dens])
        return p
    if _HAVE_RUST and base == "normal" and n * len(F) > _LARGE_DISPATCH_ENTRIES:
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
        # the same refinement/warning forward_grid applies: this branch
        # is a storage optimization, not a different discretization
        x, points = _refine_grid(x, sd, V, points, fn=fn)
        p, sl, total = _fastrace.forward_and_slopes(
            np.ascontiguousarray(mu), np.ascontiguousarray(V),
            np.ascontiguousarray(D), np.ascontiguousarray(F),
            np.ascontiguousarray(W), points, float(x[0]), float(x[-1]))
        if not (np.isfinite(total) and total > 0):
            return _massless_lattice(mu, V, D, base, nodes_given,
                                     return_slopes)
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
            if not (np.isfinite(total) and total > 0):
                return _massless_lattice(mu, V, D, base, nodes_given,
                                         return_slopes)
            if return_slopes:
                return np.asarray(p), np.asarray(sl) / total
            return np.asarray(p)
        except TypeError:
            pass       # older fastrace without window arguments: numpy path
    spec = _RUST_BASE_IDS.get(base) if isinstance(base, str) \
        else getattr(base, "rust_base", None)
    # kink_split (opt-in): a kinked base on the split lattice
    # (kink_lattice). The compiled kernel integrates on the uniform
    # lattice only, so the opt-in path stays in numpy; the default keeps
    # main's dispatch and arithmetic exactly.
    kinks = getattr(fn, "kinks", ()) if kink_split else ()
    if _HAVE_RUST and spec is not None and not kinks \
            and hasattr(_fastrace, "forward_and_slopes_base"):
        bid, prm = (spec, []) if isinstance(spec, int) else spec
        p, sl, total = _fastrace.forward_and_slopes_base(
            np.ascontiguousarray(mu), np.ascontiguousarray(V),
            np.ascontiguousarray(D), np.ascontiguousarray(F),
            np.ascontiguousarray(W), points,
            float(x[0]), float(x[-1]), int(bid), [float(v) for v in prm])
        if not (np.isfinite(total) and total > 0):
            return _massless_lattice(mu, V, D, base, nodes_given,
                                     return_slopes)
        if return_slopes:
            return np.asarray(p), np.asarray(sl) / total
        return np.asarray(p)
    p = np.zeros(n)
    slope = np.zeros(n)
    if kinks:
        # one split lattice per factor node: the kinks sit at that
        # node's locations
        for q in range(len(F)):
            xq, wq = kink_lattice(x, M_all[q], sd, kinks)
            z = (xq[None, :] - M_all[q][:, None]) / sd[:, None]
            S, f, fp = fn(z)
            f = f / sd[:, None]
            logS = _log_survival(fn, z, S)
            rest = np.exp(np.clip(logS.sum(axis=0)[None, :] - logS,
                                  -745.0, 0.0))
            p += W[q] * ((f * rest) @ wq)
            slope += W[q] * ((-fp / (sd ** 2)[:, None] * rest) @ wq)
        total = p.sum()
        if not (np.isfinite(total) and total > 0):
            return _massless_lattice(mu, V, D, base, nodes_given,
                                     return_slopes)
        if return_slopes:
            return p / total, slope / total
        return p / total
    chunk = max(1, int(5e6 / (n * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S, f, fp = fn(z)
        f = f / sd[None, :, None]
        logS = _log_survival(fn, z, S)
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)
        slope += Wc @ (np.sum(-fp / sd[None, :, None] ** 2 * rest, axis=2) * dx)
    total = p.sum()
    if not (np.isfinite(total) and total > 0):
        return _massless_lattice(mu, V, D, base, nodes_given,
                                 return_slopes)
    if return_slopes:
        return p / total, slope / total
    return p / total


def abilities_from_race(p, V=None, D=None, F=None, W=None, base="normal",
                        points=257, temperature=0.0, n_iter=None, tol=1e-8,
                        structure=None, cov=None, target_floor=None,
                        return_info=False, kink_split=False):
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
    warns rather than returning silently.

    kink_split: opt-in, default False -- invert the split-lattice map
    race_probabilities(..., kink_split=True) instead of the default one.
    Either way a kinked base (laplace, exponential_power_base with
    beta < 2) in a hard race takes its Newton own-slopes from the split
    lattice. The uniform rectangle sum of the slope integrand -f'(z),
    which JUMPS at the runner's kink, is only first order: dx times the
    jump, which grows as 1/sd^2, so a runner whose sd is three lattice
    spacings had an own-slope 6x too large and exact self-generated
    targets failed to converge, missing by 0.099 (#267). The residual is
    always the map being inverted, so the answer is that map's."""
    base = _canonical_base(base)                             # (#858)
    # finite and >= 0; negative and NaN used to select the hard race
    # silently, and +inf failed in grid sizing (#366)
    temperature = as_soft_temperature(temperature)
    dense = None
    if (F is None) != (W is None):
        # the forward refuses a half rule (#290); so does its inverse,
        # which read a lone F with uniform weights for its scale while
        # its iterative forward discarded that same F
        raise ValueError(
            "supply both F (factor nodes) and W (their weights), or "
            f"neither; got only {'F' if W is None else 'W'}")
    nodes_given = F is not None          # the caller's nodes, not a fit's
    temperature = _as_temperature(temperature)       # (#424)
    points = as_points(points)                       # (#444)
    tol = as_tolerance(tol)                          # (#551)
    if structure is not None:
        # One covariance description, as the forward door insists (#89):
        # Independent/Factor replaced D (and V) here while a caller's V
        # survived, so a target inverted under (structure, V) did not
        # reprice under the identical forward call, which refuses it.
        conflicting = [nm for nm, v in (("V", V), ("D", D), ("F", F),
                                        ("W", W)) if v is not None]
        if conflicting and cov is None:
            raise ValueError(
                f"structure= already describes the covariance; "
                f"{', '.join(conflicting)}= would describe it again. "
                "Pass one or the other.")
    if cov is not None:
        # The forward race routes a degraded fit to GHK (#161); the inverse
        # has to invert THAT map or the two front doors describe different
        # races (#164: 4e-3 to 8e-3 apart on exact rank-1/3/5 fixtures).
        # The fit still serves: its lattice inverse is the warm start and
        # its own-slopes the Newton preconditioner, and the sweeps below
        # then polish mu against the dense map itself until the residual
        # is met. Where the fit is healthy or the call is not routable
        # (non-normal base, temperature) the fit IS the model, with its
        # warnings -- it used to call fit_covariance directly and say
        # nothing.
        routable = (base == "normal" and not temperature)
        V, D, F, W, degraded = _fit_cov(cov, structure, V, D, stacklevel=2,
                                        will_route=routable)
        if degraded and routable:
            dense = np.asarray(cov, dtype=float)
    if structure is not None:
        from .structures import Factor, Independent
        if isinstance(structure, Independent):
            structure, D = None, np.asarray(structure.D, float)
        elif isinstance(structure, Factor):
            V = np.asarray(structure.V, float)
            D = np.asarray(structure.D, float)
            structure = None
    # finite and one-dimensional BEFORE the sign test: NaN passes
    # `target <= 0`, and the pair closed form then certified NaN
    # abilities as converged with residual 0 (#110)
    target = as_target(p)
    if target_floor is not None:
        target_floor = as_tolerance(target_floor, "target_floor")
        # the floor is a PROBABILITY: normalize (scale-safely) first. Floored
        # in the caller's units, [c, 0] at c = 1e-6 / 1 / 1e6 returned gaps
        # of 0 / 4.75 / 7.03 for one and the same relative target (#592)
        target, floored = _apply_floor(target, 1.0, target_floor)
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
    # Rescale before summing ONLY when the sum would overflow. A market
    # law is defined up to a positive factor, so [1e308]*3 is the same
    # law as [1]*3 -- but the sum of the first is inf, the division gave
    # NaN, and both ratings filters then stored NaN means and NaN
    # evidence from input every entry of which was finite (#300). The
    # finiteness checks at the filters' doors cannot see this: they look
    # at the entries, and the entries are fine.
    #
    # Conditional so that every ordinary input is bit-identical: the
    # branch is taken only where the current arithmetic is already
    # broken. Dividing by the max first leaves entries in (0, 1], so the
    # sum is at most n. The trial sum is allowed to overflow: it is the
    # test, and under np.seterr(over="raise") it raised before the
    # rescue could run (#300).
    with np.errstate(over="ignore"):
        _tot = target.sum()
    if not np.isfinite(_tot):
        target = target / target.max()
        _tot = target.sum()
    target = target / _tot
    if structure is not None:
        # the grammar path takes the SAME contract (sixth review): the
        # target was validated or floored above, and the iteration reports
        # convergence through the same tail rather than returning silently.
        # The hierarchical kernels are Gaussian hard races: a base= or
        # temperature= the forward door would refuse was dropped here and
        # a Gaussian race calibrated in its place (#89).
        bad = [nm for nm, flag in (("base", base != "normal"),
                                   ("temperature", temperature > 0)) if flag]
        if bad:
            raise NotImplementedError(
                f"{type(structure).__name__} races do not support "
                f"{', '.join(bad)}: the block/nested/tree kernels are "
                "Gaussian hard races. Use structure=Factor (or V=/D=) for "
                "a non-normal base or a finite temperature.")
        # n_iter is the caller's budget (#89): it used to be raised to at
        # least 120, so an n_iter=1 diagnostic call ran 120 sweeps.
        mu, converged, resid_max, iters = _abilities_from_structure(
            target, structure, points=points,
            n_iter=120 if n_iter is None else n_iter, tol=tol)
        return _inverse_return(mu, converged, resid_max, iters, floored,
                               tol, return_info)
    logt = np.log(target)
    n_t = len(target)
    # The field's contrast scale. The warm start below and the step cap in
    # the loop were written in unit-variance units; on a field whose
    # performance sd is 0.1-0.2 (a pair of near-identical high loadings,
    # or D ~ 0.005 at n = 2-3) that start is many sd off, the forward
    # saturates to [1, 0, ...], and a capped step of 2 is 10-20 sd -- the
    # inverse diverged (gap 1.28 for an exact 0.17 at loadings 0.99).
    # Scaling both by the typical contrast sd fixes every such case and
    # leaves fields near unit scale exactly as they were.
    #
    # The factor part is measured on the factor distribution actually
    # represented: with caller-supplied nodes, the covariance of F under W
    # (#149: (V, F) -> (V/c, cF) is the same forward map, and reading the
    # scale off V alone put the start 33x off and diverged at c = 0.03).
    # The default nodes have unit covariance, where this is mean ||V_i||^2.
    _Dn = np.ones(n_t) if D is None else as_idio(D, n_t)
    _Vn = np.zeros((n_t, 1)) if V is None else as_loadings(V, n_t)
    _Vc = _gauge_center(_Vn)                       # scale-safe (#471)
    if V is not None and nodes_given:
        _Fq = np.asarray(F, dtype=float).reshape(len(F), -1)
        _Wq = (np.ones(len(_Fq)) / len(_Fq) if W is None
               else as_weights(W))   # via max(W): sum(W) can overflow (#415)
        _Fm = _Wq @ _Fq
        _CovF = (_Fq - _Fm).T @ ((_Fq - _Fm) * _Wq[:, None])
    else:
        _CovF = np.eye(_Vc.shape[1])
    _SigV = _Vc @ _CovF @ _Vc.T
    # median's midpoint of two finite variances near the top of the range
    # overflows to inf; the pair's closed form below does not use it (#100)
    with np.errstate(over='ignore'):
        scale = float(np.sqrt(np.median(_Dn) + np.diag(_SigV).mean()))
    mu_start = None
    if n_t == 2 and base == "normal" and not temperature and dense is None:
        # A pair is a single Gaussian contrast, so the inverse is closed
        # form (the mirror of the forward closed form): with
        # Sigma = V V' + diag(D), p0 = Phi((mu1 - mu0 + shift) / sd_d),
        # so mu1 - mu0 = sd_d Phi^-1(p0) - shift, mean-zero; the shift is
        # the factor rule's weighted MEAN, (v1 - v0) . E[F] (#374).
        # in units of the largest term: no 1e-300 floor, no overflow of
        # the variance sum at D = 9e307 (#100)
        sd_d = _contrast_sd(_SigV[0, 0] + _SigV[1, 1] - 2.0 * _SigV[0, 1],
                            _Dn[0], _Dn[1])
        shift = (float((_Vc[1] - _Vc[0]) @ _Fm)
                 if (V is not None and nodes_given) else 0.0)
        # Invert the SMALLER share, Phi^-1(p0) = -Phi^-1(p1): for
        # [1, 1e-16] the normalized first share rounds to exactly 1, and
        # ndtri(1) = inf was certified as converged while the
        # permutation [1e-16, 1] gave the finite answer (#412).
        if target[0] <= target[1]:
            gap = sd_d * float(ndtri(target[0])) - shift
        else:
            gap = -sd_d * float(ndtri(target[1])) - shift
        if V is not None and nodes_given:
            # the forward prices the exact mixture over the caller's nodes
            # (#742), which the closed form matches only to two moments:
            # invert THAT map by a monotone 1-D solve from this start
            g = _pair_mixture_gap(target, _Vc[0] - _Vc[1], _Dn[0], _Dn[1],
                                  _Fq, _Wq, gap, sd_d)
            if g is not None and np.isfinite(g):
                # certified by the very map the forward evaluates (the
                # forward's pruning of null nodes moves it by <= 1e-15)
                mu = np.array([-0.5 * g, 0.5 * g])
                ph = _pair_mixture(g, _Vc[0] - _Vc[1], _Dn[0], _Dn[1],
                                   _Fq, _Wq)[0]
                # no 1e-300 floor: a subnormal target is a real share
                # the solve reprices exactly (#91); an exact 0 is inf
                with np.errstate(divide="ignore"):
                    r0 = float(np.abs(np.log(ph) - logt).max())
                if r0 < tol:
                    return _inverse_return(mu, True, r0, 0, floored, tol,
                                           return_info)
        mu = np.array([-0.5 * gap, 0.5 * gap])
        ok = bool(np.isfinite(mu).all())
        if not nodes_given or not ok:
            return _inverse_return(mu, ok, 0.0 if ok else np.inf, 0, floored,
                                   tol, return_info)
        # A caller's rule need not be Gaussian (a centred two-point law
        # has the right mean and variance and a different pair map), and
        # the forward integrates the rule itself: the closed form is a
        # START, certified only by the forward it claims to invert. It
        # used to be returned as exact with residual 0 and missed by up
        # to 21.8 points (#374).
        ph = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base,
                                points=points)
        r0 = float(np.abs(np.log(np.maximum(ph, 1e-300)) - logt).max())
        if r0 < tol:
            return _inverse_return(mu, True, r0, 0, floored, tol, return_info)
        mu_start = mu
    mu = (-(logt - logt.mean()) / 2.0 * scale if mu_start is None
          else mu_start)
    # N = 2: the photo-finish graph K_2 is bipartite, so the undamped
    # Jacobi update on the mean-zero quotient has eigenvalue 1 - 2 = -1,
    # a local two-cycle. Fixed damping 0.7 restores contraction.
    #
    # The same two-cycle appears at ANY N when two runners hold nearly all
    # the mass: the tail barely couples to them, so the pair's Jacobi
    # eigenvalue tends to -1 as the tail vanishes and the two favourites
    # overshoot each other every sweep. Measured (normal base, 500
    # points): two dominant runners with the rest at 1e-4 fail at N = 3,
    # 10 and 30 (residual 1e-2 after 60 sweeps; 500 sweeps is WORSE),
    # while three substantive runners converge in 15-28. Keying the
    # damping on the target's top-two share fixes those in 19-20 sweeps
    # and leaves every other field's sweep count unchanged (a share of
    # 0.86 took 40 undamped sweeps and 14 damped; below 0.8 the undamped
    # step is already fastest). "0.7 always" would triple the sweeps on
    # a 150-runner field.
    _top2 = float(np.sort(target)[-2:].sum()) if len(target) > 2 else 1.0
    alpha = 0.7 if (len(target) == 2 or _top2 > 0.8) else 1.0
    if n_iter is None:
        n_iter = 60

    if dense is not None:
        # Invert the routed map itself (#164), by the same own-slope
        # Newton sweeps as the lattice: GHK accumulates dp_i/dmu_i in its
        # conditioning pass (see methods.native._ghk_prob), so the routed
        # map has slopes of its own. The degraded fit is used for nothing
        # here -- its own-slopes are useless with D on its floor (the
        # conditional race is a near-step: the lattice inverse of the
        # rank-1 fixture ran to a log residual of 690), and a fit lifted
        # off the floor stalls at 0.19 on a dense n=30 correlation. The
        # scale is read off the covariance directly.
        scale = float(np.sqrt(np.mean(np.diag(dense))))
        mu = -(logt - logt.mean()) / 2.0 * scale

        def _dense_fwd(m):
            lp, dl = _race_dense(m, dense, return_slopes=True, log=True)
            return lp - logt, np.minimum(dl, -1e-6 / scale)

        mu, converged, resid_max, iters = _jacobi_sweeps(
            mu, _dense_fwd, scale, alpha, max(n_iter, 120), tol)
        return _inverse_return(mu, converged, resid_max, iters, floored,
                               tol, return_info)

    _fn = BASES.get(base) if isinstance(base, str) else base
    split_slopes = (not temperature and not kink_split
                    and bool(getattr(_fn, "kinks", ())))

    def _lattice(m):
        phat, sl = race_probabilities(m, V=V, D=D, F=F, W=W, base=base,
                                      points=points, temperature=temperature,
                                      return_slopes=True,
                                      kink_split=kink_split)
        if split_slopes:
            # a kinked base: the slope from the split lattice, the value
            # from the map being inverted (#267)
            sl = race_probabilities(m, V=V, D=D, F=F, W=W, base=base,
                                    points=points, return_slopes=True,
                                    kink_split=True)[1]
        phat = np.maximum(phat, 1e-300)
        # the slope floor in the field's units: d log p / d mu scales as
        # 1 / scale, so an absolute -1e-6 took over past scale 1e6 and the
        # same race at D = 1e20 stopped 0.062 of a share short (#100)
        return np.log(phat) - logt, np.minimum(sl / phat, -1e-6 / scale)

    mu, converged, resid_max, iters = _jacobi_sweeps(
        mu, _lattice, scale, alpha, n_iter, tol)
    return _inverse_return(mu, converged, resid_max, iters, floored,
                           tol, return_info)


def _pair_mixture(gap, dV, d0, d1, F, W):
    """A two-runner normal race under a caller's node law, exactly (#742).

    Runner i performs mu_i + V_i . f + e_i (lowest wins), so given the node
    f runner 0 wins with probability Phi((gap - (V0 - V1) . f) / sd),
    gap = mu1 - mu0, sd = sqrt(D0 + D1); the race is the W-mixture of
    those. Returns ([p0, p1], dp0/dgap), the second being the own-slope
    magnitude both runners share. Each tail is summed directly, so the
    loser keeps full relative precision and swapping the runners swaps
    the answer exactly. O(Q).
    """
    Wn = as_weights(W)           # relative weights, as the lattice reads them
    sd = _contrast_sd(0.0, d0, d1)
    if not sd > 0.0:
        sd = 1e-150              # an exactly deterministic contrast: as before
    u = (float(gap) - np.asarray(F, dtype=float) @ dV) / sd
    p = np.array([float(Wn @ ndtr(u)), float(Wn @ ndtr(-u))])
    p = p / p.sum()
    dens = float(Wn @ np.exp(-0.5 * u * u)) / np.sqrt(2.0 * np.pi) / sd
    return p, dens


def _pair_mixture_gap(target, dV, d0, d1, F, W, g0, h):
    """The gap mu1 - mu0 at which _pair_mixture prices `target` (#742).

    p0 is increasing in the gap, so this is a monotone 1-D root of
    log p_small(gap) - log target_small, taken on the SMALLER share so a
    1e-16 target keeps its digits (#412); log p is a log-sum-exp of
    log_ndtr terms, finite where p itself underflows. Newton from g0 (the
    closed form's gap, already right for a normal node law), safeguarded
    by the bracket the iterates establish: a step leaving it bisects, and
    while one side is open the step is capped at h, doubling. Returns
    None if it does not converge.
    """
    from scipy.special import log_ndtr
    Wn = as_weights(W)
    with np.errstate(divide="ignore"):
        logw = np.log(Wn)
    sd = _contrast_sd(0.0, d0, d1)
    if not sd > 0.0:
        sd = 1e-150
    s = np.asarray(F, dtype=float) @ dV
    small = 0 if target[0] <= target[1] else 1
    sign = 1.0 if small == 0 else -1.0       # p_small increases in sign*gap
    logt = float(np.log(target[small]))

    def lse(a):
        m = float(a.max())
        if not np.isfinite(m):
            return m
        return m + float(np.log(np.exp(a - m).sum()))

    if not (np.isfinite(g0) and np.isfinite(h) and h > 0.0):
        return None
    g, lo, hi = float(g0), -np.inf, np.inf
    for _ in range(200):
        u = sign * (g - s) / sd
        p = float(Wn @ ndtr(u))
        if p > 1e-250:
            # the direct sums, half the cost of the log domain
            logp = float(np.log(p))
            ratio = float(Wn @ np.exp(-0.5 * u * u)) / np.sqrt(2.0 * np.pi) / p
        else:
            logp = lse(logw + log_ndtr(u))
            ratio = float(np.exp(lse(logw - 0.5 * u * u)
                                 - 0.5 * np.log(2.0 * np.pi) - logp))
        f = sign * (logp - logt)              # increasing in g
        if f == 0.0:
            return g
        if not np.isfinite(f):
            return None
        if f > 0.0:
            hi = g
        else:
            lo = g
        # d f / d g = sum_w phi(u) / (sd p_small) = ratio / sd > 0
        gn = g - f * sd / ratio if ratio > 0.0 else np.nan
        if abs(f) < 1e-8 and lo < gn < hi:
            # quadratic convergence: from a relative error of 1e-8 the
            # step lands at rounding, and the caller certifies it
            return float(gn)
        # While one side is open a Newton step is capped at h too: on a
        # mixture plateau the ratio is ~1e-66 and the uncapped step
        # (-6.7e64) set a bracket end bisection never came back from (#830).
        is_open = not (np.isfinite(lo) and np.isfinite(hi))
        if not (lo < gn < hi) or (is_open and abs(gn - g) > h):
            if not is_open:
                gn = 0.5 * (lo + hi)
            else:
                gn = g - h if f > 0.0 else g + h
                h *= 2.0
        if abs(gn - g) <= 2.0 * np.finfo(float).eps * max(abs(g), sd):
            return float(gn)
        g = float(gn)
    return None


def _contrast_sd(shared, d0, d1):
    """sqrt(shared + d0 + d1), the sd of a pair's Gaussian contrast, formed
    in units of the largest term so a finite huge variance cannot overflow
    the sum and a tiny one meets no absolute floor (#100). Exact zero is
    zero."""
    lo, hi = sorted((float(d0), float(d1)))   # order-free: a permuted
    big = max(abs(float(shared)), hi)          # pair is the same contrast
    if not big > 0.0:
        return 0.0
    v = float(shared) / big + (lo / big + hi / big)
    return float(np.sqrt(big) * np.sqrt(max(v, 0.0)))


def _jacobi_sweeps(mu, forward, scale, alpha, n_iter, tol):
    """Own-slope-preconditioned coordinate (Jacobi) sweeps on the mean-zero
    quotient: mu <- mu - alpha * resid / dres, capped, recentered.

    `forward(mu)` returns (resid, dres): the residual in whatever space
    the caller inverts in (log p for the win race, logit q for top-k),
    model minus target, and its own-slopes, negative and bounded away
    from zero. Convergence is max |resid| < tol. The step cap is
    residual-proportional: a near-certain winner has residual AND
    own-slope both vanishing, and their ratio is an O(0.1) noise step
    that recentering sloshes into every other coordinate (measured:
    heavy-favorite targets in the 1e-4..1e-8 window stalled at 200
    iterations; capped, they converge in 4-6). No coordinate moves much
    further than its own residual warrants, in the field's scale.

    Damping adapts to the contraction actually observed (#149), and it
    separates two things the first cut of this conflated (#178). The
    fixed gates in the callers (a pair, a dominant pair) catch the
    two-cycle they were written for, but heterogeneous variances produce
    the same negative Jacobi eigenvalue with no share threshold to see
    it: p = [.6, .2, .2], D = [.03, .03, 1] has a top-two share of 0.8
    exactly and contracted by only 0.92 a sweep undamped (60 sweeps,
    1.7e-4 short) against 0.3 a sweep damped (17 sweeps). Consecutive
    steps estimate the dominant mode: their normalised inner product rho
    is the factor 1 - alpha (1 - lambda) that mode contracts by, so
    lambda = 1 - (1 - rho) / alpha, and the Richardson choice for a
    spectrum spanning [lambda, 0] is alpha = 2 / (2 - lambda), clipped
    to [0.1, 1] -- 2/3 for the pair's -1, which is why the fixed 0.7 was
    right where it applied.

    That Richardson value is a PERSISTENT fact about the Jacobian, so it
    is held in `base`. A sweep that contracts neither the max nor the rms
    residual is a different event -- a nonlinear transient the mode
    estimate does not describe -- and is met by undoing the sweep and
    halving a separate `penalty`, which each contracting sweep then
    restores a third of the way back toward 1. Multiplying the two keeps
    caution temporary. One number for both could only ratchet down: on a
    dense n = 30 correlation the residual fell to 5.7e-1 by sweep 10,
    rose back to 1.1 by sweep 30 as the floor took hold, then crawled at
    0.981 a sweep and was still 3.2e-1 short at 120, while the same
    sweeps undamped converged in 93. Restoring `alpha_base` on a good sweep
    instead is worse than either: it fights the Richardson value, which
    the contraction did not repeal, and the heterogeneous cases above
    stop converging at all.

    A monotone iteration riding one mode is summed rather than waited
    out: when consecutive steps are collinear (cosine > 0.999) and their
    norms decay geometrically by a factor in (0.5, 0.999), the remaining
    steps are a geometric series, so the step is scaled by
    1 / (1 - ratio) and the pair is measured fresh afterwards. Only
    while the residual is still a thousand tolerances out: within reach
    of `tol` the steps are small enough that the ratio is noise, and a
    hundredfold extrapolation of noise overshoots -- the top-k pair at
    n = 10 stalled at 1.3e-6 that way, converging at 1e-8 once gated. That is
    Aitken extrapolation in the iterate, and it is what a near-duplicate
    pair needs: two runners 1e-5 apart in a dense field have a contrast
    sd of 0.014, so their difference is nearly unidentified and the
    own-slope preconditioner -- which sees each runner's own marginal,
    not the contrast -- understates the step by the same factor every
    sweep. That mode is monotone, not oscillating (measured: cosine
    between consecutive steps exactly 1.0), so damping was the wrong
    medicine for it and extrapolation is the right one. Dense n = 16 /
    30 / 40 take 31 / 66 / 53 sweeps where they took 120 without
    converging.

    The extrapolated iterate is a bounded, checked TRIAL (#583).
    Collinearity and a ratio in (0.5, 0.999) do not show the iteration
    is in its linear regime: an exact normal top-1 target at residual 20
    had ratio 0.998, so the unchecked jump moved a step of norm 1.8 by
    539x, to abilities in [-897, 235], and backtracking never recovered
    (logit residual 715 after 120 sweeps at 257 points). So the summed
    tail is capped in the max norm at twice the field scale -- an
    ordinary step's own cap -- and repriced: a trial whose forward
    raises, is non-finite, or whose max residual grows tenfold is
    dropped for the ordinary step. Anything milder is left to the
    backtracking above. Requiring the trial to CONTRACT is too strict:
    on the dense n = 30 correlation the residual rises transiently
    (0.53 to 3.5) while the near-duplicate mode is being summed, and
    that fixture stalled at 4e-7 instead of converging in 65. The 583
    fixture reaches 4e-7 in 120 sweeps and converges at 513 / 1025
    points. An accepted trial's forward is reused by the next sweep, so
    the check costs a forward only when it rejects."""
    resid_max = np.inf
    resid_rms = np.inf
    iters = 0
    prev = None
    prev_step = None
    alpha_base = float(alpha)  # the Richardson value: persistent
    penalty = 1.0              # caution after a bad sweep: transient
    cached = None              # forward of an accepted Aitken trial
    backoffs = 0
    for it in range(n_iter):
        iters = it + 1
        if cached is not None:
            (resid, dlogp), cached = cached, None
        else:
            try:
                resid, dlogp = forward(mu)
            except RuntimeError:
                # The forward refused this iterate (a lattice defect, or
                # a membership above one -- the uniform lattice's mass
                # for a kinked base can exceed one near saturation,
                # #267). Before, that ended the inverse with the error
                # although the last priced iterate was fine. Back off
                # halfway toward it, with the transient caution halved,
                # a bounded number of times. The very first forward still
                # raises; a persistent refusal returns the last priced
                # iterate, reported as not converged.
                if prev is None:
                    raise
                if backoffs >= 20:
                    mu, resid_max = prev[0], prev[3]
                    return mu, resid_max < tol, resid_max, iters
                backoffs += 1
                mu = prev[0] + 0.5 * (mu - prev[0])
                penalty = max(0.5 * penalty, 0.1)
                prev_step = None
                continue
        resid_max = float(np.abs(resid).max())
        resid_rms = float(np.sqrt(np.mean(resid * resid)))
        if resid_max < tol:
            break
        if prev is not None and resid_max >= prev[3] and resid_rms >= prev[4]:
            if penalty > 0.1:
                penalty = max(0.5 * penalty, 0.1)
                mu, resid, dlogp, resid_max, resid_rms = prev
                prev_step = None
        elif prev is not None and penalty < 1.0:
            penalty = min(1.0, penalty / 0.75)
        prev = (mu, resid, dlogp, resid_max, resid_rms)
        alpha = alpha_base * penalty
        lim = np.minimum(2.0, 10.0 * np.abs(resid)) * scale
        step = np.clip(alpha * resid / dlogp, -lim, lim)
        step -= step.mean()
        if prev_step is not None:
            na = float(np.linalg.norm(prev_step))
            nb = float(np.linalg.norm(step))
            if na > 0.0:
                rho = float(step @ prev_step) / (na * na)
                cos = float(step @ prev_step) / (na * nb) if nb > 0 else 0.0
                ratio = nb / na
                if rho < 0.0:
                    lam = 1.0 - (1.0 - rho) / alpha
                    alpha_base = float(np.clip(2.0 / (2.0 - lam), 0.1, 1.0))
                elif (cos > 0.999 and 0.5 < ratio < 0.999
                      and resid_max > 1e3 * tol):
                    trial = _aitken_trial(mu, step, ratio, scale, forward,
                                          resid_max)
                    if trial is not None:   # sum the geometric tail
                        mu, cached = trial
                        prev_step = None
                        continue
        prev_step = step
        mu = mu - step
    if not resid_max < tol:
        # The budget ran out after a step no sweep priced: the residual
        # above belongs to the iterate BEFORE it, so the diagnostics
        # described a different mu, and n_iter=0 reported inf for a
        # finite initializer (#497). Price the returned iterate -- one
        # forward, only on exhaustion -- and keep the better of the two.
        last = (mu, resid_max) if prev is None else (prev[0], prev[3])
        if cached is not None:
            fwd = cached
        else:
            try:
                fwd = forward(mu)
            except RuntimeError:
                # The last step crossed the forward's refusal guard: return
                # the last priced iterate as not converged, as the in-loop
                # backoff does. With nothing priced it still raises (#823).
                if prev is None:
                    raise
                return prev[0], False, prev[3], iters
        r_end = np.abs(np.asarray(fwd[0], dtype=float)).max()
        if prev is not None and not r_end < last[1]:
            mu = last[0]
        else:
            resid_max = float(r_end)
    return mu, resid_max < tol, resid_max, iters


def _aitken_trial(mu, step, ratio, scale, forward, resid_max):
    """The Aitken-extrapolated iterate as a bounded, checked trial (#583):
    the summed tail step / (1 - ratio), capped in the max norm at twice
    the field scale (never below the ordinary step), and repriced. Returns
    (mu, forward(mu)), or None -- take the ordinary step -- when the
    forward raises, is non-finite, or its max residual grows tenfold."""
    jump = step / (1.0 - ratio)
    big = float(np.abs(jump).max())
    cap = max(2.0 * float(np.max(scale)), float(np.abs(step).max()))
    if big > cap:
        jump = jump * (cap / big)
    cand = mu - jump
    try:
        fc = forward(cand)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError):
        return None
    rc = np.asarray(fc[0], dtype=float)
    if not (np.all(np.isfinite(rc)) and np.all(np.isfinite(fc[1]))):
        return None
    if float(np.abs(rc).max()) > 10.0 * resid_max:
        return None
    return cand, fc


def _floorable(target, mass):
    """A target about to be floored, as a law of total `mass`: the floor is
    a probability, so it applies to normalized entries, never to the
    caller's arbitrary units (#592). Negative entries have no reading as
    mass and raise; dividing by the max first keeps the sum finite."""
    if np.any(target < 0):
        raise ValueError(
            "target entries must be non-negative: target_floor floors "
            "small and zero shares, not negative ones")
    target = target / target.max()
    return target * (mass / target.sum())


def _apply_floor(target, mass, floor):
    """Floor a target at `floor` IN ITS FINAL NORMALISED SCALE (#592).

    Normalise to `mass` (1 for a race, k slots for top-k), raise the
    entries below the floor to exactly the floor, and take that mass
    proportionally from the others -- repeating if that pushes another
    entry under the floor. Clamping and then renormalising scaled the
    floor itself back down: [.7, .3, 0] at floor .2 repriced its floored
    runner at 0.167. A floor no target of this mass can meet (n * floor
    >= mass) raises. Returns (target, floored mask)."""
    t = _floorable(target, mass)
    floored = t < floor
    for _ in range(len(t)):
        free = mass - floor * int(floored.sum())
        rest = t[~floored].sum()
        if not (free > 0 and rest > 0):
            raise ValueError(
                f"target_floor={floor:g} is infeasible: flooring "
                f"{int(floored.sum())} of {len(t)} entries at it leaves no "
                f"mass for the rest of a target of total {mass:g}")
        scaled = np.where(floored, floor, t * (free / rest))
        newly = ~floored & (scaled < floor)
        if not newly.any():
            break
        floored = floored | newly
    return scaled, floored


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
    totvar = structure_variances(structure)
    # in the field's units: a dimensionless start and an absolute [-2, 2]
    # step made the same Nested/Tree race fail by 0.40 of a share when
    # written at c = 1e-3 and 0.25 at c = 1e3 (#100)
    scale = float(np.sqrt(np.median(totvar)))
    mu = -(logt - logt.mean()) / 2.0 * scale
    # Trial states are priced under the hierarchical kernels' looser trial
    # mass tolerance: they only steer, and the strict 0.5% check rejected
    # a self-generated target at its own warm start (#656). The state
    # returned as converged is re-priced under the strict check below.
    from .blocks import _trial_mass_tolerance
    with _trial_mass_tolerance():
        mu, err, iters = _structure_sweeps(mu, logt, structure, totvar,
                                           scale, points, n_iter, tol)
    if err < tol:
        phat = dispatch_probabilities(mu, structure, points=points)
        err = float(np.abs(np.log(np.maximum(phat, 1e-300)) - logt).max())
    return mu, bool(err < tol), float(err), iters


def _structure_sweeps(mu, logt, structure, totvar, scale, points, n_iter,
                      tol):
    from .structures import dispatch_probabilities
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
        dlogp = np.minimum(ss / np.maximum(ps, 1e-300), -1e-6 / scale)
        lim = np.minimum(2.0, 10.0 * np.abs(resid)) * scale
        mu = mu - np.clip(alpha * resid / dlogp, -lim, lim)
        mu -= mu.mean()
    return mu, float(err), iters


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
    """Survival, density, density-derivative of sd*e + tau*g on a grid.

    The kernel gets its own grid, symmetric about zero. numpy's
    mode="same" keeps the central slice of the full convolution, which
    aligns the kernel's MIDDLE SAMPLE with zero lag, so a kernel
    evaluated on the signal's own asymmetric grid displaces the result
    by that grid's midpoint. The min-Gumbel needs 30 tau to the left
    and 8 to the right, and on [-left sd - 30 tau, right sd + 8 tau]
    the midpoint is -11 tau: the convolved base came out shifted by
    +11 tau. A common shift cancels in a race, which is why the win
    probabilities were nearly right, but the shifted density is then
    truncated against a grid that does not reach it, and that does not
    cancel. It bit hardest where 12 sd < 3 tau put the mode off the
    grid entirely: at sd 0.1, tau 1 the convolved mean read 8.16
    against an exact -0.577.
    """
    lo = -left * sd_i - 30.0 * tau
    hi = right * sd_i + 8.0 * tau
    u = np.linspace(lo, hi, m)
    du = u[1] - u[0]
    _, f_base, _ = fn(u[None, None, :] / sd_i)
    f_base = f_base[0, 0] / sd_i
    half = int(np.ceil(30.0 * tau / du))
    k = np.arange(-half, half + 1) * du                # symmetric, zero-centred
    v = np.exp(np.minimum(k / tau, 30.0))
    f_gum = v * np.exp(-v) / tau                       # min-Gumbel, scale tau
    # full convolution, sliced where the kernel's zero lag sits: entry j
    # of the full product is at u[0] + k[0] + j du, so u[i] is j = i +
    # half. Slicing by hand rather than by mode="same" also survives a
    # kernel longer than the signal, which happens once tau exceeds the
    # base's own width.
    f_eta = np.convolve(f_base, f_gum, mode="full")[half:half + m] * du
    f_eta = np.maximum(f_eta, 0.0)
    total = f_eta.sum() * du
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"base sd {sd_i:g} is too narrow to resolve on the tempered "
            f"grid at temperature {tau:g}: the grid spans the Gumbel "
            f"kernel, so a base this much narrower than tau falls between "
            f"its samples. At that ratio the race is the softmax in the "
            f"limit; use winning.factor.races.softmax_probabilities, which "
            f"is the closed form, rather than this quadrature.")
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
    # These densities are not normalised and carry no mass check, so the
    # grid must RESOLVE the narrowest runner on the span the widest one
    # and the farthest mean set: a runner 100 units behind a close pair
    # cut that pair's density by 10.3% at 501 points (#349). Spacing held
    # to half the smallest sd, as the standalone JS engine now does.
    lo = float(M_all.min() - left * sd.max())
    hi = float(M_all.max() + right * sd.max())
    need = int(np.ceil((hi - lo) / (0.5 * float(sd.min())))) + 1
    if need > 16385:
        import warnings
        warnings.warn(
            "tie_densities: the field is too wide to resolve the sharpest "
            f"runner even at 16385 lattice points (needs {need})",
            RuntimeWarning, stacklevel=2)
    x = np.linspace(lo, hi, int(min(max(points, need), 16385)))
    dx = x[1] - x[0]
    w = np.zeros((n, n))
    for c in range(len(F)):
        z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
        S, f, _ = fn(z)
        f = f / sd[:, None]
        logS = _log_survival(fn, z, S)
        logSfield = logS.sum(0)
        for i in range(n):
            rest = np.exp(np.clip(logSfield[None, :] - logS[i] - logS,
                                  -745.0, 0.0))
            w[i] += W[c] * (f[i] * f * rest).sum(1) * dx
    np.fill_diagonal(w, 0.0)
    return 0.5 * w + 0.5 * w.T      # symmetric by theory; average numerics (#279)


_REMOVAL_CAP = 16385
# per-runner lattice budget for the capped-field refinement (points x n),
# the same order as ordered_probabilities' budget
_REMOVAL_POINT_BUDGET = 4_000_000


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
    points = as_points(points)
    mass_tol = as_tolerance(mass_tol, "mass_tol")   # (#590)
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    n = len(mu)
    if n < 2:
        raise ValueError(
            "removal_shares needs at least two runners: removing the only "
            "one leaves no race to win")
    if n == 2:
        # Removing either of two runners leaves one, who wins with
        # probability one: the answer is the constant permutation matrix
        # whatever mu, V, F, W, D or base. Integrating it anyway let an
        # IRRELEVANT ability gap decide whether the call succeeded -- a
        # Laplace pair 40 apart overshot the cusp to a row mass of
        # 1.0026 and raised (#411).
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    sd = np.sqrt(D)
    M_all = mu[None, :] + F @ V.T
    span = float(M_all.max() - M_all.min()) + (left + right) * float(sd.max())
    need = int(np.ceil(span / (float(sd.min()) / 8.0))) + 1
    pts = int(min(max(points, need), _REMOVAL_CAP))
    lo_x = M_all.min() - left * sd.max()
    hi_x = M_all.max() + right * sd.max()

    def _accumulate(pts):
        x = np.linspace(lo_x, hi_x, pts)
        dx = x[1] - x[0]
        q = np.zeros((n, n))
        for c in range(len(F)):
            z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
            S, f, _ = fn(z)
            f = f / sd[:, None]
            logS = _log_survival(fn, z, S)
            logSfield = logS.sum(0)
            for i in range(n):
                rest = np.exp(np.clip(logSfield[None, :] - logS[i] - logS,
                                      -745.0, 0.0))
                contrib = (f * rest).sum(1) * dx
                contrib[i] = 0.0
                q[i] += W[c] * contrib
        return q

    q = _accumulate(pts)
    # A capped lattice cannot be policed by the row masses: cell errors of
    # opposite sign cancel in each row sum. The #269 fixture (sds 1e-2,
    # 1e-4, 1e-4 over a span of 4.6) had raw row mass 1.00025 -- inside
    # mass_tol -- while P(1 wins | 0 removed) was 0.7848 against an exact
    # 0.7340, five percentage points. So, as ordered_probabilities does,
    # refine by doubling and watch the CELLS until they stop moving, and
    # raise if the budget runs out first.
    if need > pts:
        import warnings
        warnings.warn(
            "removal_shares: the ability span is too wide to resolve the "
            f"sharpest runner at {pts} lattice points (needs {need}); "
            "refining until the shares stop moving. The row masses are NOT "
            "the check -- cell errors of opposite sign cancel in them "
            "(#269).", RuntimeWarning, stacklevel=2)
        ceiling = max(pts, min(need, int(_REMOVAL_POINT_BUDGET // n)))
        prev = q / q.sum(axis=1, keepdims=True)
        moved = np.inf
        while pts < ceiling:
            pts = min(2 * pts - 1, ceiling)
            q = _accumulate(pts)
            cur = q / q.sum(axis=1, keepdims=True)
            moved = float(np.abs(cur - prev).max())
            prev = cur
            if moved <= mass_tol:
                break
        if moved > mass_tol:
            raise FloatingPointError(
                f"removal_shares: the field needs {need} lattice points to "
                f"resolve the sharpest runner and the budget stops at "
                f"{ceiling}; refining to there still moved a share by "
                f"{moved:.2e}, above mass_tol={mass_tol:.0e}. The row "
                "masses are NOT evidence here -- cell errors of opposite "
                "sign cancel in them. Narrow the field or widen mass_tol "
                "deliberately.")
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
    mu = _as_abilities(mu)                                   # (#799)
    # finite and positive: NaN passed the old `tau <= 0` test, and +inf
    # collapsed every field to uniform (#366)
    tau = as_luce_temperature(temperature)
    if V is None:
        z = -mu / tau
        z -= z.max()
        w = np.exp(z)
        return w / w.sum()
    V = as_loadings(V, len(mu))
    if F is None or W is None:
        D_impl = np.full(len(mu), _GUMBEL_UNIT_D * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    else:
        # _setup is where the relative-weight rule is decided, and this
        # branch skips it because the caller supplied the law. Without
        # this the answer scaled with sum(W): the same law spelled
        # [5, 5] returned TEN TIMES the probabilities, and the ordered
        # log-probability log(10) higher (#263).
        W = as_weights(W)
    F = np.asarray(F, dtype=float)
    W = np.asarray(W, dtype=float)
    M = -(mu[None, :] + F @ V.T) / tau
    M -= M.max(axis=1, keepdims=True)
    E = np.exp(M)
    P = E / E.sum(axis=1, keepdims=True)
    return W @ P


def plackett_luce_order_logprob(mu, order, temperature=1.0, V=None, F=None,
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
    mu = _as_abilities(mu)                                   # (#799)
    tau = as_luce_temperature(temperature)                     # #366
    # a complete order of distinct labels: [0, 0] scored p = 0.5 (#129)
    order = as_order(order, len(mu), full=True)

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
    V = as_loadings(V, len(mu))
    if F is None or W is None:
        D_impl = np.full(len(mu), _GUMBEL_UNIT_D * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    else:
        # _setup is where the relative-weight rule is decided, and this
        # branch skips it because the caller supplied the law. Without
        # this the answer scaled with sum(W): the same law spelled
        # [5, 5] returned TEN TIMES the probabilities, and the ordered
        # log-probability log(10) higher (#263).
        W = as_weights(W)
    logs = np.array([_one(-(mu + np.asarray(F)[q] @ V.T) / tau)
                     for q in range(len(F))])
    m = logs.max()
    return float(m + np.log(np.dot(np.asarray(W), np.exp(logs - m))))


def plackett_luce_topk_probabilities(p, k=3):
    """P(finish in the top k) for every runner, from win probabilities,
    by Harville's conditioning: after removing a finisher, the rest
    renormalize (exact under Luce/Gumbel, the classical racing formula;
    its known favorite-longshot bias in real top-k markets is the
    Gaussian/Gamma ordering effect Henery and Stern model). k in
    {1, 2, 3} (top 1, 2, 3)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    n = len(p)
    if int(k) != k or k < 1:
        raise ValueError("k must be a positive integer")
    # Every runner is in the top k of a field of at most k, whatever the
    # weights. An exact zero share -- which stable softmax produces by
    # underflow -- reached a zero remaining denominator at the last
    # place and lost its membership: softmax([1000, 0, 0]) gave top-3
    # [0, 1, 1], summing to 2 (#384).
    if k >= n:
        return np.ones(n)
    if k == 1:
        return p.copy()
    if k not in (2, 3):
        raise ValueError("k must be 1, 2 or 3")
    out = p.copy()
    # Simplex boundary: when the runners still unplaced all have zero
    # weight, the next place goes to one of them uniformly at random,
    # the limit of equal vanishing weights. Without it the place was
    # paid to nobody and sum(top-k) fell below k.
    # the complement 1 - p_j is computed as the SUM OF THE OTHERS, never
    # by subtraction from one: at p_fav = 1 - 1e-13 the subtraction loses
    # three digits to cancellation and the exact identity sum(top-k) = k
    # drifted to k + 3e-3 (caught by the gap-stress battery)
    rest = np.array([p[np.arange(n) != j].sum() for j in range(n)])
    # second: j first, i second -- P2[j, i] = p_j p_i / rest_j
    # as p_j * (p_i / rest_j): p_i <= rest_j off the diagonal, so the
    # ratio is at most one and every POSITIVE rest is a valid divisor. A
    # floor of 1e-300 under rest underpaid subnormal longshots, so
    # [1, 1e-301 x3] gave top-2 memberships summing to 1.3, not 2 (#384).
    exhausted = rest <= 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        P2 = p[:, None] * (p[None, :] / np.where(exhausted, 1.0, rest)[:, None])
    P2[exhausted, :] = (p[exhausted] / (n - 1))[:, None]
    np.fill_diagonal(P2, 0.0)
    out += P2.sum(axis=0)
    if k == 2:
        return out
    # third: j first, l second, i third
    for j in range(n):
        rem1 = rest[j]
        for sec in range(n):
            if sec == j:
                continue
            w = P2[j, sec]
            if w == 0.0:
                continue
            denom2 = rem1 - p[sec]
            if denom2 < 1e-8 * rem1:
                # same cancellation one level deeper (two large entries
                # exhausting the field): recompute as the sum of the
                # actual remaining entries, exactly
                mask = np.ones(n, dtype=bool)
                mask[j] = mask[sec] = False
                denom2 = p[mask].sum()
            if denom2 <= 0.0:
                contrib = np.full(n, w / (n - 2))
            else:
                # w * (p / denom2): the product w * p underflows first
                # for subnormal longshots (#384)
                with np.errstate(over="ignore", invalid="ignore"):
                    contrib = w * (p / denom2)
            contrib[j] = 0.0
            contrib[sec] = 0.0
            out += contrib
    return out


def abilities_from_softmax(p, temperature=1.0):
    """Exact inverse of the independent softmax race: mean-zero mu with
    softmax_probabilities(mu, temperature) = p. Closed form."""
    # nonempty and finite first: NaN passed `p <= 0` and came back as NaN
    # abilities, and [] came back as [] (#799)
    p = as_target(p, "p")
    if np.any(p <= 0):
        raise ValueError("all target probabilities must be positive")
    # tau = 0 mapped EVERY target to equal abilities, which the forward
    # map then refused to price (#366)
    tau = as_luce_temperature(temperature)
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
    base = _canonical_base(base)                             # (#858)
    q = float(q)
    if not 0.0 <= q < 1.0:
        raise ValueError("failure probability q must lie in [0, 1)")
    fn0 = base if callable(base) else BASES[base]
    w = float(width)
    off = float(offset)
    # width is the lump's Gaussian scale: zero divided every race into
    # NaN and a negative width reversed the lump survival, so the
    # probabilities stayed finite and normalised while every own-slope
    # turned positive (#389)
    if not (np.isfinite(w) and w > 0.0):
        raise ValueError("failure_base needs a finite width > 0; got " + repr(width))
    if not np.isfinite(off):
        raise ValueError("failure_base needs a finite offset; got " + repr(offset))
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
        Sl = np.maximum(ndtr(-zl), 1e-300)
        fl = np.exp(-0.5 * zl * zl) / (w * np.sqrt(2.0 * np.pi))
        fpl = -zl * fl / w
        S = (1.0 - q) * S0 + q * Sl
        f = ((1.0 - q) * f0 + q * fl) * sd
        fp = ((1.0 - q) * fp0 + q * fpl) * sd * sd
        return np.maximum(S, 1e-300), f, fp

    # A window that covers BOTH components (#135). Without a span the
    # finite-temperature path and the ratings lattices used the generic
    # (-12, 12), cut the lump off at offset > 12 and renormalised what
    # was left: offset 20 priced [0.678, 0.249, 0.073] against the
    # subset-enumeration [0.515, 0.300, 0.185] at every resolution. The
    # running base keeps its own span (at least the generic 12) and the
    # lump is covered to 9 widths, both in standardized units; at the
    # default offset 6 this is the old (12, 12).
    L0, R0 = (getattr(base, "span", (12.0, 12.0)) if callable(base)
              else _SPANS.get(base, (12.0, 12.0)))
    L0, R0 = max(12.0, float(L0)), max(12.0, float(R0))
    lump_lo = (off - 9.0 * w - m1) / sd
    lump_hi = (off + 9.0 * w - m1) / sd
    _fail.span = (max(L0, (L0 + m1) / sd, -lump_lo),
                  max(R0, (R0 - m1) / sd, lump_hi))
    if base == "normal":
        _fail.rust_base = (8, [q, w, off, m1, sd])

    def _fail_sample(rng, size=None):
        # min-wins: with probability q the lump N(off, w^2) in the slow
        # tail, otherwise the running base; then the (optional)
        # standardization u = m1 + sd z inverted
        from ..ratings.simulate import sample_min
        u0 = sample_min(rng, base, size)
        lump = off + w * rng.standard_normal(np.shape(u0))
        u = np.where(rng.random(np.shape(u0)) < q, lump, u0)
        return (u - m1) / sd
    _fail.sample = _fail_sample
    return _fail


# Deprecated aliases (Plackett--Luce is the preferred name).
harville_order_logprob = plackett_luce_order_logprob
harville_place_probabilities = plackett_luce_topk_probabilities
