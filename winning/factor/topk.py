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
"""
from __future__ import annotations

import numpy as np

from .races import BASES
from .blocks import TINY, roots_hermitenorm


def _count_window(mu, sd, k, base_rows, delta=1e-12, pad_sds=2.0):
    """Lattice window for the top-k integrand.

    Below the window at most delta of a single runner has finished
    (sum_j F_j <= delta bounds P(N >= 1)); above it the count exceeds k
    almost surely (Chernoff: mean count mu* with
    mu* - sqrt(2 mu* ln(1/delta)) >= k forces P(N <= k-1) <= delta).
    Both ends by bisection on the monotone mean count, bracketed by
    geometric expansion first, in the node-aware style of the
    hierarchical kernels."""
    smax = max(float(sd.max()), 1e-12)

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
    # doubling runs to infinity); cap just below saturation, where the
    # whole field is below x almost surely
    target_hi = min(k + 2.0 * np.log(1.0 / delta)
                    + np.sqrt(2.0 * (k + 1) * np.log(1.0 / delta)),
                    len(mu) - 0.25)
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


def _topk_independent(mu, sd, k, base_rows, points, delta=1e-12):
    lo, hi = _count_window(mu, sd, k, base_rows, delta=delta)
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]
    z = (x[:, None] - mu[None, :]) / sd[None, :]
    S, f, _ = base_rows(z)
    F = np.clip(1.0 - S, 0.0, 1.0)
    C = _count_distribution(F)
    cdf = _leave_one_out_cdf(C, F, k)         # (n, L)
    dens = (f / sd[None, :]).T                # (n, L)
    return (dens * cdf).sum(axis=1) * dx


def _checked_topk(raw, k, kind, mass_tol=5e-3):
    t = float(raw.sum())
    if not np.isfinite(t) or abs(t - k) > mass_tol * k:
        raise RuntimeError(
            f"{kind} captured total membership {t:.4f} where exactly "
            f"{k} slots exist (defect {abs(t-k):.2e}): the window or the "
            "deconvolution missed part of the field. Raise points=, or "
            "report this field.")
    return np.clip(raw * (k / t), 0.0, 1.0)


def top_k_probabilities(mu, k, V=None, D=None, base="normal", points=513,
                        qa=15):
    """P(X_i among the k smallest), for every i, min-wins.

    mu: locations; D: idiosyncratic variances; V: optional factor
    loadings (n, r) -- conditional on the factor draw the race is
    independent, and the result is the Gauss-Hermite mixture of the
    conditional memberships (rank one and two; higher ranks are
    refused, matching the hierarchical kernels' quadrature honesty).
    k = 1 is the win probability. The identity sum_i q_i = k is checked
    and a material defect raises."""
    mu = np.asarray(mu, float)
    n = len(mu)
    if not 1 <= int(k) <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    k = int(k)
    D = np.ones(n) if D is None else np.asarray(D, float)
    sd = np.sqrt(D)
    base_rows = BASES[base] if not callable(base) else base

    if V is None:
        raw = _topk_independent(mu, sd, k, base_rows, points)
        return _checked_topk(raw, k, "top-k race")

    Vm = np.asarray(V, float)
    if Vm.ndim == 1:
        Vm = Vm[:, None]
    r = Vm.shape[1]
    if r > 2:
        raise NotImplementedError(
            "top_k_probabilities mixes Gauss-Hermite factor nodes and is "
            "implemented for factor rank <= 2; higher rank needs the "
            "scrambled-Sobol escalation (issue #12).")
    Vm = Vm - Vm.mean(axis=0, keepdims=True)   # common column is gauge
    an, aw = roots_hermitenorm(qa)
    aw = aw / aw.sum()
    if r == 1:
        nodes = an[:, None]
        w = aw
    else:
        nodes = np.array([[a, b] for a in an for b in an])
        w = np.array([u * v for u in aw for v in aw])
        w = w / w.sum()
    raw = np.zeros(n)
    for q in range(len(nodes)):
        shift = Vm @ nodes[q]
        raw += w[q] * _topk_independent(mu + shift, sd, k, base_rows,
                                        points)
    return _checked_topk(raw, k, "top-k race")


def bottom_k_probabilities(mu, k, V=None, D=None, base="normal",
                           points=513, qa=15):
    """P(X_i among the k largest) -- the complement identity
    P(in the worst k) = 1 - P(in the best n-k), so no reflected base is
    ever needed."""
    n = len(np.asarray(mu))
    if not 1 <= int(k) <= n - 1:
        raise ValueError(f"k must be in [1, n-1]; got k={k}, n={n}")
    q = top_k_probabilities(mu, n - int(k), V=V, D=D, base=base,
                            points=points, qa=qa)
    return 1.0 - q
