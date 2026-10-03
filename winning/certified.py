"""Certified win probabilities for the independent and factor races.

Every other path in winning returns a float that is accurate in
practice, measured against reference computations. This module
returns a ball, a midpoint and a radius whose interval provably
contains the true probability, computed in Arb ball arithmetic
(python-flint) with outward rounding throughout. Ask for 30 digits and
either the radius is below 1e-30 or the call raises; it never hands
back an uncertified answer.

The race is the engine's own, min-wins:

    X_i = mu_i + sqrt(D_i) Z_i,   Z_i iid from a standardized base,
    p_i = P(X_i < X_j for all j != i) = int f_i(x) H_i(x) dx,
    H_i = prod_{j != i} S_j.

Two certificates, which share nothing but the base formulas:

  certified_race_probabilities   Arb's rigorous Gauss-Legendre
      integration of p_i over a finite window, plus tail masses bounded
      in closed form. The bases are analytic in a strip (the Laplace
      base piecewise, split at its kinks), so the digits come cheaply:
      thirty of them in well under a second for small fields.

  bracket_race_probabilities     the monotone bracket. H_i is
      decreasing and F_i increasing, so on a lattice cell [a, b]

          H_i(b) dF_i  <=  int_a^b H_i dF_i  <=  H_i(a) dF_i,

      and the two tails are bracketed the same way. It needs nothing
      but monotone survival functions: no density, no smoothness, no
      integrator. The price is first order, width ~ 1/cells. It is the
      independent check, not the source of digits.

The factor race, X_i = mu_i + V_i f + sqrt(D_i) Z_i with f ~ N(0, I_k),
has its own certificate:

  certified_factor_race_probabilities   p_i is an integral over (x, f)
      in k + 1 dimensions, done by an adaptive tensor Gauss-Legendre
      rule whose error on each box is bounded by the Bernstein-ellipse
      theorem (Trefethen, ATAP Thm 19.3): the integrand is bounded
      rigorously on complex boxes covering each ellipse, so the bound
      needs only evaluations of the formulas, never inner integrals.
      Mass outside the truncation box is bounded by the base's own
      tails, because prod S <= 1.

All N probabilities come from one shared field per lattice point, as in
the float engine: S_j and f_j are evaluated once at every point and
every runner's product of the others is read off prefix and suffix
products (no division by S, which may vanish on a complex box).

Requires python-flint (pip install python-flint); importing this
module without it raises ImportError with that instruction.
"""

from __future__ import annotations

import itertools
import math

try:
    from flint import acb, arb, ctx
except ImportError as exc:  # pragma: no cover - exercised only without flint
    raise ImportError(
        "winning.certified needs python-flint: pip install python-flint"
    ) from exc

__all__ = ["BASES", "certified_race_probabilities",
           "certified_factor_race_probabilities",
           "bracket_race_probabilities", "contains"]


# ---------------------------------------------------------------------------
# Standardized bases, as formulas valid on real and complex balls.
#
# Each base gives survival S(z) and density f(z) of a mean-zero,
# unit-variance law, matching winning.factor.races exactly except that
# constants are exact here (Euler's gamma, pi) where the float engine
# rounds them. `kinks` lists the points where the formulas stop being
# analytic; integration is split there, and `piece` picks the branch
# valid on a segment from a real point inside it.
# ---------------------------------------------------------------------------


class _Base:
    kinks = ()

    def S(self, z, piece=0):
        raise NotImplementedError

    def f(self, z, piece=0):
        raise NotImplementedError

    def piece(self, x):
        return 0


class _Normal(_Base):
    name = "normal"

    def S(self, z, piece=0):
        return (z / arb(2).sqrt()).erfc() / 2

    def f(self, z, piece=0):
        return (-z * z / 2).exp() / (2 * arb.pi()).sqrt()


class _GumbelMin(_Base):
    # S(z) = exp(-exp(c z - gamma)), c = pi/sqrt(6): mean 0, variance 1
    name = "gumbel"

    @staticmethod
    def _u(z):
        return z * arb.pi() / arb(6).sqrt() - arb.const_euler()

    def S(self, z, piece=0):
        return (-self._u(z).exp()).exp()

    def f(self, z, piece=0):
        u = self._u(z)
        return arb.pi() / arb(6).sqrt() * (u - u.exp()).exp()


class _Logistic(_Base):
    # S(z) = 1/(1 + exp(c z)), c = pi/sqrt(3); poles at c z = i pi (2k+1)
    name = "logistic"

    @staticmethod
    def _c():
        return arb.pi() / arb(3).sqrt()

    # The same analytic functions, reflected where Re z > 0. On a wide
    # ball exp(c z) is huge there and 1 + exp(c z) is a ball containing
    # zero, which bounds nothing; exp(-c z) is small and does not.

    @staticmethod
    def _right(z):
        return float(z.real.mid() if isinstance(z, acb) else z.mid()) > 0

    def S(self, z, piece=0):
        if self._right(z):
            e = (-self._c() * z).exp()
            return e / (1 + e)
        return 1 / (1 + (self._c() * z).exp())

    def f(self, z, piece=0):
        e = ((-1 if self._right(z) else 1) * self._c() * z).exp()
        return self._c() * e / (1 + e) ** 2


class _Laplace(_Base):
    # scale b = 1/sqrt(2); analytic on each side of the kink at 0
    name = "laplace"
    kinks = (0,)

    @staticmethod
    def _r():
        return arb(2).sqrt()            # 1/b

    def piece(self, x):
        return 0 if x < 0 else 1

    def S(self, z, piece=0):
        if piece == 0:
            return 1 - (self._r() * z).exp() / 2
        return (-self._r() * z).exp() / 2

    def f(self, z, piece=0):
        if piece == 0:
            return self._r() * (self._r() * z).exp() / 2
        return self._r() * (-self._r() * z).exp() / 2


BASES = {b.name: b for b in (_Normal(), _GumbelMin(), _Logistic(), _Laplace())}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ball(x):
    """Exact conversion: a float is a dyadic rational and becomes a point
    ball; a string becomes Arb's enclosure of the decimal it spells."""
    if isinstance(x, (arb, str)):
        return arb(x)
    return arb(float(x))


def _field(mu, D):
    mu = [_ball(m) for m in mu]
    n = len(mu)
    if n < 2:
        raise ValueError("a race needs at least two contestants")
    if D is None:
        D = [1] * n
    elif isinstance(D, (int, float, str, arb)):
        D = [D] * n
    if len(D) != n:
        raise ValueError(f"D has {len(D)} entries for {n} contestants")
    sd = []
    for d in D:
        d = _ball(d)
        if not d > 0:
            raise ValueError("every D must be provably positive")
        sd.append(d.sqrt())
    return mu, sd


def _base(base):
    if isinstance(base, _Base):
        return base
    try:
        return BASES[base]
    except KeyError:
        raise ValueError(f"base must be one of {sorted(BASES)}") from None


def contains(ball, x):
    """True when the ball provably contains the float or ball x."""
    return ball.contains(_ball(x))


class _Precision:
    def __init__(self, digits):
        self.prec = int(digits * 3.33) + 64

    def __enter__(self):
        self.saved = ctx.prec
        ctx.prec = self.prec
        return self

    def __exit__(self, *exc):
        ctx.prec = self.saved


def _window(mu, sd, base, eps):
    """A finite window [L, R] outside which every runner's own mass is
    below eps, found by doubling. The tails are bounded rigorously by
    the caller whatever window comes back; this only keeps them small."""
    lo = min(float(m.mid()) for m in mu)
    hi = max(float(m.mid()) for m in mu)
    smax = max(float(s.mid()) for s in sd)
    t = 4.0
    for _ in range(40):
        L, R = arb(lo - t * smax), arb(hi + t * smax)
        left = max(_cdf(base, (L - m) / s).upper() for m, s in zip(mu, sd))
        right = max(_sf(base, (R - m) / s).upper() for m, s in zip(mu, sd))
        if left < eps and right < eps:
            return L, R
        t *= 1.5
    raise ArithmeticError("could not find a window holding the field")


def _sf(base, z):
    return base.S(z, base.piece(float(z.mid())))


def _cdf(base, z):
    return 1 - _sf(base, z)


def _tails(i, mu, sd, base, L, R):
    """Rigorous enclosure of int f_i H_i over (-inf, L] and [R, inf).

    H_i decreases from 1, so on the left the mass lies between
    F_i(L) H_i(L) and F_i(L); on the right between 0 and S_i(R) H_i(R).
    """
    def H(x):
        h = arb(1)
        for j, (m, s) in enumerate(zip(mu, sd)):
            if j != i:
                h *= _sf(base, (x - m) / s)
        return h

    FL = _cdf(base, (L - mu[i]) / sd[i])
    SR = _sf(base, (R - mu[i]) / sd[i])
    left = arb.union(FL * H(L), FL)
    right = arb.union(arb(0), SR * H(R))
    return left + right


# ---------------------------------------------------------------------------
# certificate one: rigorous integration
# ---------------------------------------------------------------------------


def certified_race_probabilities(mu, D=None, base="normal", digits=30):
    """Win probabilities of the independent min-wins race as Arb balls.

    Returns a list of `flint.arb`, one per contestant, each provably
    containing p_i, with radius below 10**-digits (the call raises
    ArithmeticError otherwise). Use `float(b.mid())` for the value and
    `b.rad()` for the certified error; `contains(b, x)` checks a float.

    mu and D follow race_probabilities: D is the performance variance,
    scalar or per runner. Floats are taken as the exact binary numbers
    they are; pass strings ("0.1") to mean decimals.
    """
    base = _base(base)
    with _Precision(digits) as p:
        mu, sd = _field(mu, D)
        eps = 10.0 ** (-digits - 3)
        L, R = _window(mu, sd, base, eps)
        tol = arb(10) ** (-digits - 3)

        # segment ends: the window plus every runner's kinks inside it
        cuts = {float(L.mid()), float(R.mid())}
        for m, s in zip(mu, sd):
            for k in base.kinks:
                x = float((m + k * s).mid())
                if float(L.mid()) < x < float(R.mid()):
                    cuts.add(x)
        # a segment end must sit exactly on its kink, so the kink's
        # position must be a point (Laplace's, at x = mu, always is for
        # float mu, whatever D)
        if any((m + k * s).rad() > 0 for m, s in zip(mu, sd)
               for k in base.kinks):
            raise ValueError("a kinked base needs kinks at exact points: "
                             "pass float mu, not decimal strings")
        cuts = sorted(cuts)
        # exact kink positions, so no segment straddles a branch change
        exact = {float(L.mid()): L, float(R.mid()): R}
        for m, s in zip(mu, sd):
            for k in base.kinks:
                exact.setdefault(float((m + k * s).mid()), m + k * s)
        ends = [exact[c] for c in cuts]

        out = []
        for i in range(len(mu)):
            total = _tails(i, mu, sd, base, L, R)
            for a, b in zip(ends[:-1], ends[1:]):
                xm = (float(a.mid()) + float(b.mid())) / 2
                pieces = [base.piece((xm - float(m.mid())) / float(s.mid()))
                          for m, s in zip(mu, sd)]

                def g(x, analytic, i=i, pieces=pieces):
                    z = (x - mu[i]) / sd[i]
                    v = base.f(z, pieces[i]) / sd[i]
                    for j, (m, s) in enumerate(zip(mu, sd)):
                        if j != i:
                            v *= base.S((x - m) / s, pieces[j])
                    return v

                seg = acb.integral(g, acb(a), acb(b), abs_tol=tol,
                                   rel_tol=tol)
                if not seg.imag.contains(0):
                    raise ArithmeticError("integral left the real line")
                total += seg.real
            if not total.rad() < arb(10) ** (-digits):
                raise ArithmeticError(
                    f"runner {i}: radius {total.rad().str(3)} exceeds "
                    f"1e-{digits} at {p.prec} bits; refusing to return it")
            out.append(arb.intersection(total, arb("[0.5 +/- 0.5]")))
        return out


# ---------------------------------------------------------------------------
# certificate two: the monotone bracket
# ---------------------------------------------------------------------------


def bracket_race_probabilities(mu, D=None, base="normal", cells=2000,
                               digits=20):
    """Win probabilities as Arb balls from the monotone Stieltjes bracket.

    Independent of the integrator: only the monotonicity of the survival
    functions is used, so the enclosure is valid for any continuous base
    whose S and F can be evaluated rigorously. It is first order, so the
    radius is ~ 1/cells. Use it to check certified_race_probabilities,
    not to replace it.

    One shared field: S_j and F_j are evaluated once at each of the
    cells + 1 lattice points, and every runner's bracket is read off it.
    """
    base = _base(base)
    with _Precision(digits):
        mu, sd = _field(mu, D)
        n = len(mu)
        L, R = _window(mu, sd, base, 10.0 ** (-digits))
        h = (R - L) / cells
        grid = [L + k * h for k in range(cells + 1)]
        S = [[_sf(base, (x - m) / s) for x in grid]
             for m, s in zip(mu, sd)]

        def Hminus(i, k):
            v = arb(1)
            for j in range(n):
                if j != i:
                    v *= S[j][k]
            return v

        out = []
        for i in range(n):
            # interior cells: dF_i = S_i(a) - S_i(b)
            lo = arb(0)
            hi = arb(0)
            Hk = [Hminus(i, k) for k in range(cells + 1)]
            for k in range(cells):
                dF = S[i][k] - S[i][k + 1]
                lo += Hk[k + 1] * dF
                hi += Hk[k] * dF
            # tails: left in [F_i(L) H_i(L), F_i(L)], right in [0, S_i(R) H_i(R)]
            FL = 1 - S[i][0]
            lo += FL * Hk[0]
            hi += FL + S[i][cells] * Hk[cells]
            ball = arb.union(lo, hi)
            out.append(arb.intersection(ball, arb("[0.5 +/- 0.5]")))
        return out


# ---------------------------------------------------------------------------
# certificate three: the factor race, adaptive tensor Gauss-Legendre
# ---------------------------------------------------------------------------


_GL = {}
_PAD = 1.0 + 1e-9


def _gauss_legendre(n, prec):
    """Rigorous nodes and weights of the n-point rule on [-1, 1]."""
    key = (n, prec)
    if key not in _GL:
        _GL[key] = [arb.legendre_p_root(n, j, weight=True) for j in range(n)]
    return _GL[key]


def _loadings(V, n):
    try:
        rows = [list(r) for r in V]
    except TypeError:
        raise ValueError("V must be (n, k) loadings") from None
    if len(rows) != n:
        raise ValueError(f"V has {len(rows)} rows for {n} contestants")
    if not all(isinstance(r, list) for r in rows) or not rows[0]:
        raise ValueError("V must be (n, k) loadings")
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        raise ValueError("V rows must all have the same length")
    return [[_ball(v) for v in r] for r in rows], k


def _factor_field(base, mu, sd, V, x, f):
    """g_i(x, f) for every runner at one point or box: the k-variate
    normal density of f, runner i's conditional density at x, and the
    others' conditional survivals, by prefix and suffix products."""
    n = len(mu)
    S, dens = [], []
    for j in range(n):
        m = mu[j]
        for vjm, fm in zip(V[j], f):
            m = m + vjm * fm
        u = (x - m) / sd[j]
        S.append(base.S(u))
        dens.append(base.f(u) / sd[j])
    q = f[0] * f[0]
    for fm in f[1:]:
        q = q + fm * fm
    phi = (-q / 2).exp() / (2 * arb.pi()) ** (arb(len(f)) / 2)
    pre = [None] * n
    acc = 1
    for j in range(n):
        pre[j] = acc
        acc = acc * S[j]
    out = [None] * n
    acc = 1
    for j in reversed(range(n)):
        out[j] = phi * dens[j] * pre[j] * acc
        acc = acc * S[j]
    return out


def _box_ball(lo, hi):
    """A ball containing [lo, hi] exactly, padded outward: built from the
    endpoints in Arb, never from a float midpoint and radius, which
    round."""
    pad = 1e-12 * (abs(lo) + abs(hi) + 1.0)
    return arb.union(arb(lo - pad), arb(hi + pad))


def _sup_on_ellipse(field, box, m, rho, pieces=8, others=4):
    """Rigorous upper bounds, per runner, of |g_i| over the Bernstein
    ellipse with parameter rho about box[m] (complex), times the real
    box in every other coordinate. The filled ellipse is covered by
    `pieces` complex rectangles along its real axis, each as tall as the
    ellipse above it; the other coordinates are cut into `others` pieces
    each to keep the interval dependency small."""
    lo, hi = box[m]
    c, h = (lo + hi) / 2, (hi - lo) / 2
    A = h * (rho + 1 / rho) / 2
    B = h * (rho - 1 / rho) / 2
    edges = [c - A + 2 * A * t / pieces for t in range(pieces + 1)]
    cover = []
    for p0, p1 in zip(edges[:-1], edges[1:]):
        near = 0.0 if p0 <= c <= p1 else min(abs(p0 - c), abs(p1 - c))
        height = B * max(0.0, 1 - (near / A) ** 2) ** 0.5
        cover.append(acb(_box_ball(p0, p1),
                         _box_ball(-height * (1 + 1e-9), height * (1 + 1e-9))))
    cuts = []
    for d, (a, b) in enumerate(box):
        if d == m:
            continue
        step = (b - a) / others
        cuts.append([_box_ball(a + t * step, a + (t + 1) * step)
                     for t in range(others)])
    sup = None
    for z in cover:
        for rest in itertools.product(*cuts):
            point = list(rest)
            point.insert(m, z)
            vals = [_upper(v) for v in field(point)]
            sup = vals if sup is None else [max(a, b) for a, b in zip(sup, vals)]
    return sup


def _upper(v):
    """|v| bounded above as a float, rounded up; an indeterminate ball
    (NaN, or infinite on a wide complex box) is an infinite bound. The
    max over the cover must never see a NaN: comparisons with one are
    False, and a dropped piece would make the bound optimistic."""
    u = abs(v).upper()
    if not u.is_finite():
        return math.inf
    x = float(u)
    return math.nextafter(x, math.inf) if math.isfinite(x) else math.inf


def _gl_box(field, box, n, prec, n_out):
    """The n^d tensor Gauss-Legendre sum over a real box, per runner."""
    rule = _gauss_legendre(n, prec)
    axes = []
    for lo, hi in box:
        # in Arb: a float centre and half-width round, and the rule then
        # integrates a box a few ulps off, leaving gaps and overlaps
        # between neighbours that no bound accounts for
        c, h = (arb(lo) + arb(hi)) / 2, (arb(hi) - arb(lo)) / 2
        axes.append([(c + h * t, h * w) for t, w in rule])
    total = [arb(0)] * n_out
    for combo in itertools.product(*axes):
        point = [t for t, _ in combo]
        weight = arb(1)
        for _, w in combo:
            weight = weight * w
        vals = field(point)
        total = [s + weight * v for s, v in zip(total, vals)]
    return total


def certified_factor_race_probabilities(mu, V, D=None, base="normal",
                                        digits=20, orders=(8, 16, 24, 32),
                                        max_boxes=4000):
    """Win probabilities of the factor min-wins race as Arb balls.

    X_i = mu_i + V_i f + sqrt(D_i) Z_i, f ~ N(0, I_k), Z_i iid from the
    base. Returns one `flint.arb` per runner, each provably containing
    p_i, with radius below 10**-digits, or raises ArithmeticError.

    The work is an adaptive cubature in k + 1 dimensions, so it is for
    small fields and low rank. Measured: five runners at rank one take
    1 s (normal), 3 s (gumbel) and 8 s (logistic) at 16 digits; three
    runners at rank two take 11 s at 8 digits and 24 s at 12. Its
    purpose is to check the float engine and to supply reference
    values, not to replace race_probabilities.

    Analytic bases only (normal, gumbel, logistic). The Laplace kink
    moves with the factors, so it is refused rather than mishandled.
    """
    base = _base(base)
    if base.kinks:
        raise NotImplementedError(
            f"the {base.name} base has kinks that move with the factors; "
            "the factor certificate takes analytic bases only")
    with _Precision(digits) as P:
        mu, sd = _field(mu, D)
        n = len(mu)
        V, k = _loadings(V, n)
        target = arb(10) ** (-digits)
        eps = arb(10) ** (-digits - 2)

        # truncation: f in [-T, T]^k, x in [L, R]
        T = 4.0
        while not k * (arb(T) / arb(2).sqrt()).erfc() < eps:
            T += 0.5
        tl = tr = 4.0
        while not _cdf(base, arb(-tl)) < eps:
            tl *= 1.25
        while not _sf(base, arb(tr)) < eps:
            tr *= 1.25
        spread = [sum((abs(v) * T for v in row), arb(0)) for row in V]
        smax = max(float(s.upper()) for s in sd)
        L = min(float((m - w).lower()) for m, w in zip(mu, spread)) - tl * smax
        R = max(float((m + w).upper()) for m, w in zip(mu, spread)) + tr * smax
        # outward, past any float rounding in the two lines above
        L -= 1e-9 * (1.0 + abs(L))
        R += 1e-9 * (1.0 + abs(R))
        tail = (k * (arb(T) / arb(2).sqrt()).erfc()
                + _cdf(base, arb(-tl)) + _sf(base, arb(tr)))

        def field(point):
            return _factor_field(base, mu, sd, V, point[0], point[1:])

        root = [(L, R)] + [(-T, T)] * k
        volume = (R - L) * (2 * T) ** k
        budget = float((target / 4).upper())
        core = [arb(0)] * n
        err = [arb(0)] * n
        stack = [root]
        boxes = 0
        while stack:
            box = stack.pop()
            boxes += 1
            if boxes > max_boxes:
                raise ArithmeticError(
                    f"factor certificate needed more than {max_boxes} boxes "
                    f"for 1e-{digits}; ask for fewer digits or a smaller field")
            vol = 1.0
            for a, b in box:
                vol *= b - a
            allow = budget * vol / volume
            # per-dimension, per-rho sup bounds, then the best order
            best = None
            for rho in (2.0, 4.0):
                sups = [_sup_on_ellipse(field, box, m, rho)
                        for m in range(len(box))]
                for order in orders:
                    E = 64.0 / (15.0 * (rho * rho - 1.0) * rho ** (2 * order))
                    per_dim = []
                    for m, (a, b) in enumerate(box):
                        other = vol / (b - a)
                        h = (b - a) / 2
                        per_dim.append([other * h * E * s for s in sups[m]])
                    # the bound is assembled in floats: pad it so their
                    # rounding cannot make it optimistic
                    bound = [_PAD * sum(col) for col in zip(*per_dim)]
                    worst = max(bound)
                    if worst <= allow:
                        if best is None or order < best[0]:
                            best = (order, bound)
                        break
                if best is None and rho == 4.0:
                    worst_dim = [max(per_dim[m]) for m in range(len(box))]
                    if all(math.isinf(w) for w in worst_dim):
                        # unbounded everywhere: the blow-up may come from
                        # any coordinate's width, so cut the widest
                        split = max(range(len(box)),
                                    key=lambda m: box[m][1] - box[m][0])
                    else:
                        split = max(range(len(box)),
                                    key=lambda m: worst_dim[m])
            if best is None:
                a, b = box[split]
                mid = (a + b) / 2
                if not a < mid < b or b - a < 1e-9 * (1.0 + abs(a)):
                    raise ArithmeticError(
                        "could not bound the integrand on a box of width "
                        f"{b - a:.1e}; the {base.name} base is too wild "
                        "here for this certificate")
                left, right = list(box), list(box)
                left[split] = (a, mid)
                right[split] = (mid, b)
                stack += [left, right]
                continue
            order, bound = best
            vals = _gl_box(field, box, order, P.prec, n)
            core = [c + v for c, v in zip(core, vals)]
            err = [e + arb(b) for e, b in zip(err, bound)]
        out = []
        for i in range(n):
            ball = core[i] + arb(0, err[i].upper()) + arb.union(arb(0), tail)
            if not ball.rad() < target:
                raise ArithmeticError(
                    f"runner {i}: radius {ball.rad().str(3)} exceeds "
                    f"1e-{digits}; refusing to return it")
            out.append(arb.intersection(ball, arb("[0.5 +/- 0.5]")))
        return out
