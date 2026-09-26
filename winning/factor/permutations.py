"""Ordered finishing prefixes -- exacta and trifecta probabilities -- from the
same shared field pass as the win race.

For independent (or factor-conditional) min-wins performances the joint
probability that i finishes first, j second and k third is

    q_ijk = Int f_k(z) prod_{m not in ijk} S_m(z) [ Int_{y<z} f_j(y) F_i(y) dy ] dz,

the removal identity of removal_shares taken one level deeper: divide i's and
j's survival back out of the field product. One lattice, every ordered prefix,
O(Q n^3 L); the exacta is the same with one removal, and k=1 is the win race.

temperature > 0 prices the base convolved with the tau-scaled min-Gumbel kernel
(the same curves race_probabilities uses), so a softened race and its ordered
prefixes come from one consistent procedure. base= and V/D= are honoured as
everywhere else in winning.factor.

Grid discipline follows removal_shares: full ability-span window, spacing
refined to the sharpest runner, and the total mass CHECKED against the
continuum identity sum over all ordered prefixes = 1, raising rather than
renormalising away a lattice that failed to capture the field.
"""
import numpy as np

from .core import as_loadings
from ..rustconfig import load_fastrace
from .races import _fit_cov, _factor_of_structure, _setup, _tempered_curves

# this module's own ceiling (#113): it used to borrow races' flags, and
# use_rust(True) defaults a missing _RUST_OK to True, which reported a
# kernel this module cannot call on an extension that lacks it
_fastrace, _RUST_OK, _HAVE_RUST = load_fastrace("ordered_prefixes")


# The single-shot lattice cap. Past this the field is refined and the
# CELLS are watched for convergence rather than the scalar total (#269).
_ORDERED_CAP = 16385
# Refinement ceiling as n * points: the pass holds several (n, points)
# float arrays, so this is tens of MB, and it keeps a wide field from
# turning into an unbounded run.
_ORDERED_POINT_BUDGET = 4_000_000


def ordered_probabilities(mu, k=3, V=None, D=None, F=None, W=None,
                          base="normal", points=501, temperature=0.0,
                          mass_tol=1e-3, structure=None, cov=None):
    """Joint probability of every ordered k-prefix, k in {1, 2, 3}.

    Returns an array of shape (n,)*k; out[i, j, l] = P(i first, j second,
    l third), zero on repeated indices. k=1 reproduces race_probabilities;
    summing out[i, :, :] + out[:, i, :] + out[:, :, i] reproduces
    top_k_probabilities(mu, 3). Under base="gumbel" (D = tau^2 pi^2/6) the
    result is Harville's stagewise product, exp(plackett_luce_prefix_logprob).

    Covariance descriptions: V=/D= sugar, structure=Independent/Factor,
    or cov= for a dense matrix fitted to the factor grammar first -- so a
    covariance written once for the inverse can be reused here without
    being re-expressed. structure=Blocks/Nested/Tree RAISES: those are
    separate win-race recursions with no ordered-prefix pass, and this
    call would otherwise silently price a different race than
    race_probabilities does with the same argument.
    """
    if k not in (1, 2, 3):
        raise ValueError("k must be 1, 2 or 3")
    if cov is not None:
        V, D, F, W, _ = _fit_cov(cov, structure, V, D, stacklevel=3)
    elif structure is not None:
        if V is not None or D is not None:
            raise ValueError("structure= replaces V=/D=; pass one only")
        V, D = _factor_of_structure(structure, "ordered_probabilities")
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    tau = float(temperature)
    M_all = mu[None, :] + F @ V.T
    pad_lo = left * sd.max() + (30.0 * tau if tau > 0 else 0.0)
    pad_hi = right * sd.max() + (8.0 * tau if tau > 0 else 0.0)
    span = float(M_all.max() - M_all.min()) + pad_lo + pad_hi
    need = int(np.ceil(span / (float(sd.min()) / 8.0))) + 1
    pts = int(min(max(points, need), _ORDERED_CAP))
    lo_x, hi_x = M_all.min() - pad_lo, M_all.max() + pad_hi
    def _accumulate(pts):
        """One pass over the lattice at this resolution. Everything that
        defines the problem is fixed above; only the number of points
        varies, which is what makes a convergence check possible."""
        x = np.linspace(lo_x, hi_x, pts)
        dx = x[1] - x[0]
        if (tau <= 0 and base == "normal" and _HAVE_RUST
                and hasattr(_fastrace, "ordered_prefixes")):
            flat, total = _fastrace.ordered_prefixes(
                np.ascontiguousarray(mu), np.ascontiguousarray(V),
                np.ascontiguousarray(D), np.ascontiguousarray(F),
                np.ascontiguousarray(W), pts, float(x[0]), float(x[-1]), int(k))
            return np.asarray(flat).reshape((n,) * k), float(total)
        curves = ([_tempered_curves(sd[i], tau, fn, left, right) for i in range(n)]
                  if tau > 0 else None)
        out = np.zeros((n,) * k)
        for c in range(len(F)):
            if tau > 0:
                S = np.empty((n, pts)); f = np.empty((n, pts))
                for i in range(n):
                    u, Sg, fg, _ = curves[i]
                    args = x - M_all[c, i]
                    S[i] = np.interp(args, u, Sg, left=1.0, right=1e-300)
                    f[i] = np.interp(args, u, fg, left=0.0, right=0.0)
            else:
                z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
                S, f, _ = fn(z)
                f = f / sd[:, None]
            logS = np.log(np.maximum(S, 1e-300))
            Fc = 1.0 - S
            logSfield = logS.sum(0)
            if k == 1:
                rest = np.exp(np.clip(logSfield[None, :] - logS, -745.0, 0.0))
                out += W[c] * (f * rest).sum(1) * dx
                continue
            for i in range(n):
                if k == 2:
                    rest_i = np.exp(np.clip(logSfield[None, :] - logS[i] - logS,
                                            -745.0, 0.0))
                    contrib = (f * Fc[i][None, :] * rest_i).sum(1) * dx
                    contrib[i] = 0.0
                    out[i] += W[c] * contrib
                    continue
                for j in range(n):
                    if j == i:
                        continue
                    g = f[j] * Fc[i]
                    inner = (np.cumsum(g) - 0.5 * g) * dx
                    rest_ij = np.exp(np.clip(logSfield[None, :] - logS[i] - logS[j]
                                             - logS, -745.0, 0.0))
                    contrib = (f * rest_ij * inner[None, :]).sum(1) * dx
                    contrib[i] = 0.0
                    contrib[j] = 0.0
                    out[i, j] += W[c] * contrib
        return out, float(out.sum())

    out, total = _accumulate(pts)

    # The scalar mass identity CANNOT police a capped lattice. Cell
    # errors of opposite sign cancel in one number: the #269 fixture
    # (sds of 22.0, 0.029 and 0.011) totalled 1.00038 -- inside the
    # default 1e-3 -- while the (1, 2) cell was 2.1% high, and the
    # normalisation then spread the residual into a plausible answer.
    # `points=` could not help, because min(max(points, need), cap) is
    # the cap whenever need exceeds it.
    #
    # So when the lattice is capped, refine it and watch the CELLS,
    # which is where the error lives. Doubling until they stop moving
    # resolves that fixture exactly at 32769 points. If the budget runs
    # out first, raise: an unresolvable field is a deliberate error,
    # never a renormalised guess.
    if need > pts:
        import warnings
        warnings.warn(
            "ordered_probabilities: the ability span is too wide to resolve "
            f"the sharpest runner at {pts} lattice points (needs {need}); "
            "refining until the prefix probabilities stop moving, which "
            "costs another pass or several. The total mass is NOT the "
            "check -- cell errors of opposite sign cancel in it (#269).",
            RuntimeWarning, stacklevel=2)
        ceiling = max(pts, min(need, int(_ORDERED_POINT_BUDGET // max(n, 1))))
        prev = out / total if total > 0 else out
        moved = np.inf
        while pts < ceiling:
            pts = min(2 * pts - 1, ceiling)
            out, total = _accumulate(pts)
            cur = out / total if total > 0 else out
            moved = float(np.abs(cur - prev).max())
            prev = cur
            if moved <= mass_tol:
                break
        if moved > mass_tol:
            raise FloatingPointError(
                f"ordered_probabilities: the field needs {need} lattice "
                f"points to resolve the sharpest runner and the budget "
                f"stops at {ceiling}; refining to there still moved a "
                f"prefix probability by {moved:.2e}, above mass_tol="
                f"{mass_tol:.0e}. The total mass is NOT evidence here -- "
                "cell errors of opposite sign cancel in it. Narrow the "
                "field, widen mass_tol deliberately, or raise the budget.")

    # the tempered curves are interpolated (as in _race_tempered, which
    # renormalises a ~3e-3 defect silently); hold the softened race to the
    # same standard the win race already accepts
    tol = mass_tol if tau <= 0 else max(mass_tol, 1e-2)
    if abs(total - 1.0) > tol:
        raise FloatingPointError(
            f"ordered_probabilities: total mass over ordered {k}-prefixes is "
            f"{total:.6f}, defect {abs(total-1):.2e} exceeds {tol:.0e}; "
            "the lattice failed to capture the field -- raise points= or "
            "mass_tol= deliberately rather than trusting a renormalization")
    return out / total


def plackett_luce_prefix_logprob(mu, prefix, temperature=1.0, V=None, F=None,
                            W=None):
    """log P(the first len(prefix) finishers are `prefix`, in that order)
    under (mixed) Plackett--Luce, with every stage's denominator over ALL
    runners still standing. plackett_luce_order_logprob takes its order as the
    complete field; this is the prefix (exacta, trifecta) version.
    Exact for the Gumbel base and only there."""
    mu = np.asarray(mu, dtype=float)
    tau = float(temperature)
    prefix = np.asarray(prefix, dtype=int)
    n = len(mu)

    def _one(z):
        standing = np.ones(n, dtype=bool)
        total = 0.0
        for o in prefix:
            zr = z[standing]
            m = zr.max()
            total += z[o] - (m + np.log(np.exp(zr - m).sum()))
            standing[o] = False
        return total

    if V is None:
        return _one(-mu / tau)
    V = as_loadings(V, len(mu))
    if F is None or W is None:
        D_impl = np.full(len(mu), (np.pi ** 2 / 6.0) * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    else:
        # _setup is where the relative-weight rule is decided, and this
        # branch skips it because the caller supplied the law. Without
        # this the returned log-probability shifted by log(sum W): the
        # same factor law spelled W = [5, 5] instead of [0.5, 0.5] read
        # -1.5071 as +0.7955, exactly log(10) apart (#208).
        W = np.asarray(W, float)
        wtot = float(W.sum())
        if not np.isfinite(wtot) or wtot <= 0.0:
            raise ValueError(
                f"W must be positive weights: they total {wtot!r}. They "
                "are relative, so any positive multiple of a valid rule "
                "is the same factor law, but a zero or negative total is "
                "not one.")
        W = W / wtot
    logs = np.array([_one(-(mu + np.asarray(F)[q] @ V.T) / tau)
                     for q in range(len(F))])
    m = logs.max()
    return float(m + np.log(np.dot(np.asarray(W), np.exp(logs - m))))


# Deprecated alias (Plackett--Luce is the preferred name).
harville_prefix_logprob = plackett_luce_prefix_logprob
