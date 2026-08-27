"""The general race: one API, distributions and correlation as parameters.

    race_probabilities(mu)                          classic independent race
    race_probabilities(mu, V=V, D=D)                factor probit (Gaussian)
    race_probabilities(mu, base="gumbel")           Luce / softmax, exactly
    race_probabilities(mu, V=V, base="gumbel")      correlated Luce
    race_probabilities(mu, base=my_base)            anything standardized

Min-wins convention throughout. A base is a callable z -> (S, f, fp)
giving survival, density and density derivative of a MEAN-ZERO,
UNIT-VARIANCE law (standardization keeps noise family separate from
noise scale). Zero factors is literally the one-node quadrature, so the
independent race is not a separate code path.

Promoted from research/experiments/exp14_boundaries/run_boundaries.py,
where the general engine was exercised by the paper's substitution
experiments (the Gumbel base's zero-loading case equals softmax to
2.8e-17 there).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

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


BASES = {"normal": _normal, "gumbel": _gumbel_min}
_SPANS = {"normal": (8.0, 8.0), "gumbel": (22.0, 8.0)}   # (left, right) tails


def _setup(mu, V, D, F, W, base):
    mu = np.asarray(mu, dtype=float)
    n = len(mu)
    D = np.ones(n) if D is None else np.asarray(D, dtype=float)
    if V is None:
        V = np.zeros((n, 1))
        F, W = np.zeros((1, 1)), np.ones(1)
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if F is None or W is None:
            # adaptive order: when idiosyncratic noise is small relative
            # to the loadings, the conditional race is nearly
            # deterministic and the factor integrand is nearly a step --
            # a fixed 15-node rule silently loses 2-5% (found by the
            # fuzz battery, research/fuzz). Scale the order with the
            # sharpness ratio; identical rule in the R port.
            sharp = float(np.max(np.sqrt((V ** 2).sum(axis=1))
                                 / np.sqrt(np.maximum(D, 1e-300))))
            r = V.shape[1]
            cap = 201 if r == 1 else (41 if r == 2 else 15)
            Q = int(np.clip(np.ceil(8.0 * sharp), 15, cap))
            F, W = hermite_nodes(r, Q=Q)
    fn = base if callable(base) else BASES[base]
    left, right = _SPANS.get(base, (12.0, 12.0)) if not callable(base) \
        else (12.0, 12.0)
    return mu, V, D, np.asarray(F, float), np.asarray(W, float), fn, left, right




def _bulk_window(M_all, sd, points, delta):
    """Lattice over the WINNER distribution's bulk, not the ability span.

    G(x) = 1 - prod_j S_j(x) is the winner cdf under the min-race (averaged
    over factor nodes via the extreme conditional locations, a conservative
    envelope); [G^-1(delta), G^-1(1-delta)] carries all but 2*delta of every
    runner's win integrand -- a hopeless runner only wins by running a
    winner-class time. Measured (research/lattice_window): 33 points here
    beat 513 on the ability-span window, whose own truncation floors its
    accuracy near 6e-11. Bisection on a monotone function; cost negligible.
    """
    from .races import _setup  # noqa -- self-module, for clarity only
    mu_lo = M_all.min(axis=0)
    mu_hi = M_all.max(axis=0)
    s = sd

    def G(x):
        # envelope: winner cdf using each runner's most favourable node
        z = (x - mu_lo) / s
        logS = np.log(np.maximum(1.0 - _ndtr_local(z), 1e-300))
        return 1.0 - np.exp(logS.sum())

    lo0 = float(mu_lo.min() - 9.0 * s.max())
    hi0 = float(mu_hi.max() + 9.0 * s.max())
    a, b = lo0, hi0
    for _ in range(80):
        m = 0.5 * (a + b)
        if G(m) < delta:
            a = m
        else:
            b = m
    xlo = a
    a, b = xlo, hi0
    # right edge from the LEAST favourable nodes so no runner's density is cut
    def H(x):
        z = (x - mu_hi) / s
        logS = np.log(np.maximum(1.0 - _ndtr_local(z), 1e-300))
        return 1.0 - np.exp(logS.sum())
    for _ in range(80):
        m = 0.5 * (a + b)
        if H(m) < 1.0 - delta:
            a = m
        else:
            b = m
    # base-agnostic safety margin: the bisection envelope uses the normal
    # survival, and other bases (gumbel) have different tails -- pad both
    # edges by 2 sd so no base's density is clipped. Costs ~15% width,
    # preserves the ~4x narrowing.
    pad = 2.0 * float(s.max())
    return np.linspace(xlo - pad, b + pad, points)


def _ndtr_local(z):
    from scipy.special import ndtr
    return ndtr(z)


def race_probabilities(mu, V=None, D=None, F=None, W=None, base="normal",
                       points=257, temperature=0.0, return_slopes=False,
                       structure=None, window="bulk", delta=1e-12):
    """Win probabilities of the general race, all N in one field pass.

    Pass `structure=` (Independent/Factor/Blocks/Nested/Tree from
    winning.factor.structures) to describe the covariance declaratively --
    one race, five grammars; V=/D= remain as sugar for the factor case.
    temperature > 0 returns the softmin expectation E[softmin(X/tau)],
    computed exactly as the hard race with each base convolved with the
    tau-scaled min-Gumbel kernel."""
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
    M_all = mu[None, :] + F @ V.T
    if window == "bulk":
        x = _bulk_window(M_all, sd, points, delta)
    else:
        x = np.linspace(M_all.min() - left * sd.max(),
                        M_all.max() + right * sd.max(), points)
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
                        points=257, temperature=0.0, n_iter=60, tol=1e-8):
    """Invert the general race: mean-zero mu with race_probabilities(mu) = p."""
    target = np.asarray(p, dtype=float)
    if np.any(target <= 0):
        raise ValueError("all target probabilities must be positive")
    target = target / target.sum()
    logt = np.log(target)
    mu = -(logt - logt.mean()) / 2.0
    # N = 2: the photo-finish graph K_2 is bipartite, so the undamped
    # Jacobi update on the mean-zero quotient has eigenvalue 1 - 2 = -1,
    # a local two-cycle. Fixed damping 0.7 restores contraction.
    alpha = 1.0 if len(target) > 2 else 0.7
    for _ in range(n_iter):
        phat, sl = race_probabilities(mu, V=V, D=D, F=F, W=W, base=base,
                                      points=points, temperature=temperature,
                                      return_slopes=True)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        if np.abs(resid).max() < tol:
            break
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        mu = mu - np.clip(alpha * resid / dlogp, -2.0, 2.0)
        mu -= mu.mean()
    return mu


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
                   points=501):
    """The full single-removal ensemble q[i][j] = P(j wins | i removed),
    every row from the same shared field by dividing i's survival back
    out. Rows sum to one. O(Q N^2 L)."""
    mu, V, D, F, W, fn, left, right = _setup(mu, V, D, F, W, base)
    sd = np.sqrt(D)
    n = len(mu)
    M_all = mu[None, :] + F @ V.T
    x = np.linspace(M_all.min() - left * sd.max(),
                    M_all.max() + right * sd.max(), points)
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
    return q / q.sum(axis=1, keepdims=True)


# canonical tier-1 name; calibrate_factors is reserved for the outer
# estimation problem (see the paper's discussion and experiment 30)
calibrate_abilities = abilities_from_race
