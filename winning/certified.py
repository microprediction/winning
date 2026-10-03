"""Certified win probabilities for the independent race.

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

All N probabilities come from one shared field per lattice in the
bracket, as in the float engine: S_j and F_j are evaluated once at
every lattice point and every runner's H_i is read off them.

Requires python-flint (pip install python-flint); importing this
module without it raises ImportError with that instruction.
"""

from __future__ import annotations

try:
    from flint import acb, arb, ctx
except ImportError as exc:  # pragma: no cover - exercised only without flint
    raise ImportError(
        "winning.certified needs python-flint: pip install python-flint"
    ) from exc

__all__ = ["BASES", "certified_race_probabilities",
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

    def S(self, z, piece=0):
        return 1 / (1 + (self._c() * z).exp())

    def f(self, z, piece=0):
        e = (self._c() * z).exp()
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
