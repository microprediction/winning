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

from .races import _setup, _tempered_curves, _HAVE_RUST, _fastrace


def ordered_probabilities(mu, k=3, V=None, D=None, F=None, W=None,
                          base="normal", points=501, temperature=0.0,
                          mass_tol=1e-3):
    """Joint probability of every ordered k-prefix, k in {1, 2, 3}.

    Returns an array of shape (n,)*k; out[i, j, l] = P(i first, j second,
    l third), zero on repeated indices. k=1 reproduces race_probabilities;
    summing out[i, :, :] + out[:, i, :] + out[:, :, i] reproduces
    top_k_probabilities(mu, 3). Under base="gumbel" (D = tau^2 pi^2/6) the
    result is Harville's stagewise product, exp(plackett_luce_prefix_logprob).
    """
    if k not in (1, 2, 3):
        raise ValueError("k must be 1, 2 or 3")
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    tau = float(temperature)
    M_all = mu[None, :] + F @ V.T
    pad_lo = left * sd.max() + (30.0 * tau if tau > 0 else 0.0)
    pad_hi = right * sd.max() + (8.0 * tau if tau > 0 else 0.0)
    span = float(M_all.max() - M_all.min()) + pad_lo + pad_hi
    need = int(np.ceil(span / (float(sd.min()) / 8.0))) + 1
    pts = int(min(max(points, need), 16385))
    if need > 16385:
        import warnings
        warnings.warn(
            "ordered_probabilities: the ability span is too wide to resolve "
            f"the sharpest runner even at 16385 lattice points (needs {need}); "
            "the total mass is checked and will raise if accuracy is lost",
            RuntimeWarning, stacklevel=2)
    x = np.linspace(M_all.min() - pad_lo, M_all.max() + pad_hi, pts)
    dx = x[1] - x[0]
    if (tau <= 0 and base == "normal" and _HAVE_RUST
            and hasattr(_fastrace, "ordered_prefixes")):
        flat, total = _fastrace.ordered_prefixes(
            np.ascontiguousarray(mu), np.ascontiguousarray(V),
            np.ascontiguousarray(D), np.ascontiguousarray(F),
            np.ascontiguousarray(W), pts, float(x[0]), float(x[-1]), int(k))
        out = np.asarray(flat).reshape((n,) * k)
        if abs(total - 1.0) > mass_tol:
            raise FloatingPointError(
                f"ordered_probabilities: total mass over ordered {k}-prefixes is "
                f"{total:.6f}, defect {abs(total-1):.2e} exceeds {mass_tol:.0e}; "
                "the lattice failed to capture the field -- raise points= or "
                "mass_tol= deliberately rather than trusting a renormalization")
        return out / total
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
    total = float(out.sum())
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
    V = np.atleast_2d(np.asarray(V, dtype=float))
    if V.shape[0] != len(mu):
        V = V.T
    if F is None or W is None:
        D_impl = np.full(len(mu), (np.pi ** 2 / 6.0) * tau * tau)
        _, _, _, F, W, _, _, _ = _setup(mu, V, D_impl, F, W, "gumbel")
    logs = np.array([_one(-(mu + np.asarray(F)[q] @ V.T) / tau)
                     for q in range(len(F))])
    m = logs.max()
    return float(m + np.log(np.dot(np.asarray(W), np.exp(logs - m))))


# Deprecated alias (Plackett--Luce is the preferred name).
harville_prefix_logprob = plackett_luce_prefix_logprob
