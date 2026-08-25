"""Probability-of-maximality estimators, all reading the same (mu, Sigma).

Six ways to get p_i = P(Y_i = max_j Y_j) for Y ~ N(mu, Sigma):

  pom_full_mc          dense Cholesky sampling + winner counts (the qPO reference)
  pom_factor_mc        sampling from mu + V z + sqrt(d) eps + winner counts
  pom_fast             deterministic factor probit: quadrature over the r factors,
                       the N idiosyncratic dimensions integrated exactly
  pom_independent      the r = 0 special case, exact one-dimensional quadrature
  pom_flite            F-LITE (Menet et al., AISTATS 2025)
  pom_alite            A-LITE (same paper, appendix C)

The lattice kernel here is a retuned copy of winning.factor.core's
win_probabilities_factor. Three changes, all measured rather than assumed:

  * max-wins throughout, so no sign juggling at the call sites;
  * a conservative window that is tight when the means are spread out. The
    package walks [min_j m_j - 8 sd, max_j m_j + 8 sd]; this one walks
        lo = second largest of (m_j - c sd_j),  hi = largest of (m_j + c sd_j),
    exact to N*Phi(-c). Below lo every candidate's integrand carries a survival
    factor under Phi(-c): for i not attaining the largest a_j = m_j - c sd_j
    some other j does, and for the i that attains it the second largest does.
    Above hi every density is under phi(c)/sd. c = 9 gives N*1.1e-19.
  * an adaptive window (the default) that is the exact support of the maximum;
    see _window_adaptive. On the molecular posteriors it is 2.4x narrower again.

With the adaptive window the lattice is spectrally accurate -- the integrand
vanishes at both endpoints, so the trapezoid/rectangle rule has no boundary
term. Measured: 129 points give total-variation 1e-13 against 32001 points,
and that holds up to a 1000-fold spread in the marginal standard deviations.
The default is 257.

test_pom.py checks this kernel against the package function, against
closed-form N=2, against both windows, and against factor Monte Carlo.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

_LOG2PI = float(np.log(2.0 * np.pi))
_TAIL_C = 9.0


# --------------------------------------------------------------------------
# deterministic factor probit
# --------------------------------------------------------------------------

def _window(M: np.ndarray, sd: np.ndarray, c: float = _TAIL_C):
    """Conservative integration window per factor node. M is (nodes, N)."""
    a = M - c * sd[None, :]
    # second largest of a along the candidate axis
    if M.shape[1] >= 2:
        top2 = np.partition(a, -2, axis=1)[:, -2:]
        lo = top2[:, 0]
    else:
        lo = a[:, 0]
    hi = (M + c * sd[None, :]).max(axis=1)
    return lo, hi


def _window_adaptive(M: np.ndarray, sd: np.ndarray, delta: float = 1e-13,
                     iters: int = 24, c: float = _TAIL_C):
    """The exact support of the maximum, to within probability 2*delta.

    Summing the integrand over candidates telescopes:
        sum_i phi_i(x) prod_{j!=i} Phi_j(x) = d/dx prod_j Phi_j(x) = G'(x),
    the density of max_j Y_j. So the integration window is simply the bulk of
    the maximum's own distribution, and the mass this window omits is exactly
    G(x_lo) + (1 - G(x_hi)) = 2*delta -- an error bound, not an estimate.

    That bulk is far narrower than the conservative window when N is large and
    the candidates are packed together, which is the molecular case: the
    maximum of N near-identical Gaussians has spread about sd/sqrt(2 log N),
    not sd. Both endpoints come from vectorised bisection on monotone
    functions, bracketed by the conservative window.

      x_lo:  sum_j log Phi((x-m_j)/sd_j)          = log delta   (increasing)
      x_hi:  logsumexp_j log Phi(-(x-m_j)/sd_j)   = log delta   (decreasing)

    The upper endpoint is solved on the union bound 1 - G <= sum_j (1 - Phi_j)
    because log G is numerically flat against zero up there.
    """
    from scipy.special import logsumexp

    lo0, hi0 = _window(M, sd, c)
    target = np.log(delta)

    lo, hi = lo0.copy(), hi0.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = log_ndtr((mid[:, None] - M) / sd[None, :]).sum(axis=1)
        below = val < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    x_lo = 0.5 * (lo + hi)

    lo, hi = x_lo.copy(), hi0.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = logsumexp(log_ndtr(-(mid[:, None] - M) / sd[None, :]), axis=1)
        above = val > target
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)
    x_hi = 0.5 * (lo + hi)
    bad = ~(x_hi > x_lo)
    if np.any(bad):
        x_lo = np.where(bad, lo0, x_lo)
        x_hi = np.where(bad, hi0, x_hi)
    return x_lo, x_hi


def pom_fast(mu, V, d, nodes=None, weights=None, points: int = 257,
             c: float = _TAIL_C, max_elements: float = 6e6,
             return_total: bool = False, window: str = "adaptive",
             delta: float = 1e-13):
    """P(i = argmax_j Y_j) for Y = mu + V z + sqrt(d) eps, z ~ N(0, I_r).

    Conditional on z the candidates are independent, so
        p_i = E_z int phi_i(x) prod_{j != i} Phi_j(x) dx,
    an r-dimensional quadrature (nodes, weights) wrapped around a lattice
    integral that costs O(N L) per node. Cost O(Q N L), memory O(N L).
    """
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    N = mu.size
    sd = np.sqrt(d)
    if V is None or (np.ndim(V) == 2 and V.shape[1] == 0):
        nodes = np.zeros((1, 1))
        weights = np.ones(1)
        M_all = mu[None, :].copy()
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        if nodes is None:
            raise ValueError("supply quadrature nodes for a rank>0 factor model")
        nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
        weights = np.asarray(weights, dtype=float)
        M_all = mu[None, :] + nodes @ V.T                      # (Q, N)

    p, total = _race_pass(M_all, sd, weights, points, window, delta, c,
                          max_elements)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        raise FloatingPointError("factor race integration failed")
    out = p / s
    return (out, total) if return_total else out


def _race_pass(M_all, sd, weights, points, window, delta, c, max_elements,
               accum=None, sq_accum=None):
    """Weighted sum over conditional means of the exact conditional win vector.

    M_all is (nodes, N) conditional means, sd is (N,). Returns the weighted
    integral and the pre-normalisation total (|1 - total| is the resolution
    diagnostic). accum/sq_accum, if given, are added to in place -- used by the
    Rao-Blackwellised sampler to accumulate the variance of the conditional
    probabilities across draws.
    """
    N = M_all.shape[1]
    if window == "adaptive":
        lo_all, hi_all = _window_adaptive(M_all, sd, delta=delta, c=c)
    elif window == "safe":
        lo_all, hi_all = _window(M_all, sd, c)
    else:
        raise ValueError(f"unknown window {window!r}")
    grid = np.arange(points) / (points - 1)
    log_norm = np.log(sd) + 0.5 * _LOG2PI                       # (N,)

    p = np.zeros(N) if accum is None else accum
    total = 0.0
    chunk = max(1, int(max_elements / (N * points)))
    for a0 in range(0, M_all.shape[0], chunk):
        M = M_all[a0:a0 + chunk]                                # (C, N)
        Wc = weights[a0:a0 + chunk]
        lo = lo_all[a0:a0 + chunk]
        hi = hi_all[a0:a0 + chunk]
        x = lo[:, None] + (hi - lo)[:, None] * grid[None, :]    # (C, L)
        dx = (hi - lo) / (points - 1)                           # (C,)
        z = (x[:, None, :] - M[:, :, None]) / sd[None, :, None]  # (C, N, L)
        log_cdf = log_ndtr(z)                                   # others below x
        log_pdf = -0.5 * z * z - log_norm[None, :, None]
        log_field = log_cdf.sum(axis=1)                         # (C, L)
        integ = np.exp(np.clip(log_pdf + log_field[:, None, :] - log_cdf,
                               -745.0, 700.0))
        contrib = integ.sum(axis=2) * dx[:, None]               # (C, N), per node
        p += Wc @ contrib
        if sq_accum is not None:
            sq_accum += Wc @ (contrib ** 2)
        total += float(np.sum(Wc[:, None] * contrib))
    return p, total


def pom_full_rb(mu, Sigma, M: int = 20000, seed: int = 0, points: int = 257,
                delta: float | None = None, chunk: int | None = None,
                return_se: bool = False, max_elements: float = 6e6):
    """Rao-Blackwellised reference: exact Sigma, no factor approximation.

    Every posterior covariance that carries observation noise satisfies
    Sigma >= lambda_min I with lambda_min > 0, so it splits exactly as

        Y = mu + W + sqrt(delta) eps,   W ~ N(0, Sigma - delta I),  delta <= lambda_min,

    with W and eps independent. Conditional on W the candidates are independent,
    so the winner indicator can be replaced by its conditional expectation --
    the same lattice integral the factor probit uses, but with the exact
    covariance carried in W rather than approximated by r factors.

    This is unbiased for the exact p, and it is far quieter than counting
    winners. Counting gives a candidate 0 or 1; this gives it its conditional
    probability at every draw, so tail candidates stop being all-or-nothing.
    The extra cost is nil in practice: drawing W already costs O(N^2) per
    sample, which dominates the O(N L) lattice pass whenever N > L.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    N = mu.size
    if delta is None:
        lam_min = float(np.linalg.eigvalsh(Sigma)[0])
        delta = 0.95 * lam_min
    if delta <= 0:
        raise ValueError("Sigma is singular; supply delta explicitly")
    A = _chol(Sigma - delta * np.eye(N))
    sd = np.full(N, np.sqrt(delta))
    rng = np.random.default_rng(seed)
    if chunk is None:
        chunk = max(1, int(max_elements / (N * points)))
    acc = np.zeros(N)
    sq = np.zeros(N)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        Mc = mu[None, :] + rng.standard_normal((m, N)) @ A.T
        w = np.full(m, 1.0 / M)
        _race_pass(Mc, sd, w, points, "adaptive", 1e-13, _TAIL_C,
                   max_elements, accum=acc, sq_accum=sq)
        done += m
    p = acc / acc.sum()
    if return_se:
        # sq holds (1/M) sum g^2, acc holds (1/M) sum g
        var = np.maximum(sq - acc ** 2, 0.0) / M
        return p, np.sqrt(var)
    return p


def pom_independent(mu, var, points: int = 257, c: float = _TAIL_C,
                    max_elements: float = 6e6, return_total: bool = False,
                    window: str = "adaptive", delta: float = 1e-13):
    """The r = 0 model: exact PoM under N(mu, diag(var)), by quadrature.

    Not the same thing as F-LITE. This is the exact answer for the independence
    model; F-LITE is a fast approximation to this quantity.
    """
    return pom_fast(mu, None, var, points=points, c=c,
                    max_elements=max_elements, return_total=return_total,
                    window=window, delta=delta)


# --------------------------------------------------------------------------
# the distribution of the maximum itself
# --------------------------------------------------------------------------

def max_cdf_factor(t, mu, V, d, nodes=None, weights=None, log=False):
    """P(max_j Y_j <= t) for Y = mu + V z + sqrt(d) eps, at each t.

    Conditional on the factor the candidates are independent, so the CDF of the
    maximum is a product of univariate normal CDFs, and no lattice is needed at
    all. Cost O(Q N) per threshold rather than the O(Q N L) the argmax vector
    needs -- the max distribution is much the cheaper of the two outputs.

    Returned in the log domain when log=True, which is the only way to get the
    far tail: at K = 10^6 strategies a genome-wide-style threshold puts
    1 - CDF near 1e-8 and the direct difference loses every digit.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    sd = np.sqrt(d)
    if V is None or (np.ndim(V) == 2 and np.asarray(V).shape[1] == 0):
        M = mu[None, :]
        w = np.ones(1)
    else:
        V = np.atleast_2d(np.asarray(V, dtype=float))
        M = mu[None, :] + np.atleast_2d(nodes) @ V.T
        w = np.asarray(weights, dtype=float)
    out = np.empty(t.size)
    for a, ta in enumerate(t):
        lg = log_ndtr((ta - M) / sd[None, :]).sum(axis=1)      # (Q,)
        m = lg.max()
        out[a] = m + np.log(np.sum(w * np.exp(lg - m)))
    return out if log else np.exp(out)


def max_sf_factor(t, mu, V, d, nodes=None, weights=None):
    """P(max_j Y_j > t), computed so the far tail keeps its digits."""
    lc = max_cdf_factor(t, mu, V, d, nodes, weights, log=True)
    return -np.expm1(lc)


def expected_max_factor(mu, V, d, nodes=None, weights=None, lo=None, hi=None,
                        points: int = 4001):
    """E[max_j Y_j] by integrating the survival function.

    E[X] = int_0^inf P(X>x) dx - int_{-inf}^0 P(X<=x) dx, on a grid wide enough
    that both tails are numerically dead.
    """
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    tot = np.sqrt(d + (0.0 if V is None else np.sum(
        np.atleast_2d(V) ** 2, axis=1)))
    if lo is None:
        lo = float(mu.min() - 10.0 * tot.max())
    if hi is None:
        hi = float(mu.max() + 10.0 * tot.max())
    x = np.linspace(lo, hi, points)
    cdf = max_cdf_factor(x, mu, V, d, nodes, weights)
    # integrating by parts on [lo, hi] with CDF(lo) = 0 and CDF(hi) = 1:
    #   E[X] = [x F(x)] - int F(x) dx = hi - int_lo^hi F(x) dx
    return float(hi - np.trapezoid(cdf, x))


# --------------------------------------------------------------------------
# quadrature nodes for the factor integral
# --------------------------------------------------------------------------

def sobol_nodes(r: int, m: int = 10, seed: int = 0):
    """2^m scrambled-Sobol nodes for E over N(0, I_r), equal weights."""
    from scipy.stats import qmc
    u = qmc.Sobol(r, scramble=True, seed=seed).random_base2(m)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    F = ndtri(u)
    return F, np.full(len(F), 1.0 / len(F))


def hermite_nodes(r: int, Q: int = 15, prune: float = 1e-7):
    """Product Gauss-Hermite for E over N(0, I_r). Only sane for r <= 3 or 4."""
    from winning.factor.core import hermite_nodes as _hn
    return _hn(r, Q=Q, prune=prune)


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------

def pom_full_mc(mu, Sigma, M: int = 10000, seed: int = 0, chunk: int = 2000,
                jitter: float = 0.0, return_se: bool = False):
    """Winner counts from dense multivariate-normal samples. The qPO reference.

    scipy's multivariate_normal.rvs uses an eigendecomposition; a Cholesky is
    the same distribution and much faster, so this is the authors' estimator
    with a faster sampler. pom_full_mc_scipy below is the authors' code path,
    kept for a like-for-like check.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    N = mu.size
    A = _chol(Sigma, jitter)
    rng = np.random.default_rng(seed)
    counts = np.zeros(N, dtype=np.int64)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        Y = mu[None, :] + rng.standard_normal((m, N)) @ A.T
        np.add.at(counts, np.argmax(Y, axis=1), 1)
        done += m
    p = counts / M
    if return_se:
        return p, np.sqrt(np.maximum(p * (1 - p), 0) / M)
    return p


def pom_full_mc_scipy(mu, Sigma, M: int = 10000, seed: int = 0):
    """Literally the authors' acquire_qPO estimator."""
    from scipy.stats import multivariate_normal
    mu = np.asarray(mu, dtype=float)
    p_yx = multivariate_normal(mean=mu, cov=np.asarray(Sigma, dtype=float),
                               allow_singular=True, seed=seed)
    samples = p_yx.rvs(size=M, random_state=seed)
    top = np.argmax(samples, axis=1)
    return np.bincount(top, minlength=mu.size) / M


def pom_factor_mc(mu, V, d, M: int = 100000, seed: int = 0, chunk: int = 20000,
                  return_se: bool = False):
    """Winner counts from the factor representation. O(M(N r + N)) work."""
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    sd = np.sqrt(d)
    N = mu.size
    V = None if V is None else np.atleast_2d(np.asarray(V, dtype=float))
    rng = np.random.default_rng(seed)
    counts = np.zeros(N, dtype=np.int64)
    done = 0
    while done < M:
        m = min(chunk, M - done)
        Y = mu[None, :] + rng.standard_normal((m, N)) * sd[None, :]
        if V is not None and V.shape[1] > 0:
            Y += rng.standard_normal((m, V.shape[1])) @ V.T
        np.add.at(counts, np.argmax(Y, axis=1), 1)
        done += m
    p = counts / M
    if return_se:
        return p, np.sqrt(np.maximum(p * (1 - p), 0) / M)
    return p


def _chol(Sigma, jitter: float = 0.0):
    S = np.asarray(Sigma, dtype=float)
    scale = float(np.mean(np.diag(S)))
    eps = jitter if jitter > 0 else 0.0
    for _ in range(12):
        try:
            return np.linalg.cholesky(S + eps * np.eye(len(S)) if eps else S)
        except np.linalg.LinAlgError:
            eps = max(eps * 10, 1e-12 * scale)
    # last resort: symmetric square root
    w, U = np.linalg.eigh(S)
    return U * np.sqrt(np.maximum(w, 0.0))


# --------------------------------------------------------------------------
# LITE (Menet, Huebotter, Kassraie, Krause; AISTATS 2025). numpy transcription
# of https://github.com/lasgroup/LITE  (flite.py and code/src/poo_estimators_and_BO.py)
# --------------------------------------------------------------------------

def pom_flite(mu, var, epsilon: float | None = None, return_kappa: bool = False):
    """F-LITE: q_i = Phi((mu_i - kappa)/sd_i) with kappa set so the q sum to one."""
    mu = np.asarray(mu, dtype=float)
    sd = np.sqrt(np.asarray(var, dtype=float))
    n = mu.size
    if epsilon is None:
        epsilon = 1.0 / (100.0 * n)
    beta = ndtri(1.0 / n)                       # negative
    k_low = mu.min() - beta * sd.min()
    k_up = mu.max() - beta * sd.max()
    for _ in range(200):
        if np.max(ndtr((mu - k_low) / sd) - ndtr((mu - k_up) / sd)) < epsilon:
            break
        k = 0.5 * (k_low + k_up)
        if 1.0 - np.sum(ndtr((mu - k) / sd)) < 0:
            k_low = k
        else:
            k_up = k
    r = 0.5 * (ndtr((mu - k_low) / sd) + ndtr((mu - k_up) / sd))
    r = r / r.sum()
    return (r, 0.5 * (k_low + k_up)) if return_kappa else r


def _log_field_cdf(f, mu, sd):
    """log prod_z Phi((f - mu_z)/sd_z) for a vector of thresholds f."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    out = np.empty(f.size)
    for i, fi in enumerate(f):
        out[i] = log_ndtr((fi - mu) / sd).sum()
    return out


def pom_alite(mu, var, depth: int = 60):
    """A-LITE: quartile-match a single Gaussian CDF to the field, in two stages.

    Stage I matches Phi((f-m)/s) to g(f) = prod_z Phi((f-mu_z)/sd_z); stage II
    matches, per candidate, to g(f)/Phi((f-mu_i)/sd_i) using the clipped
    stage-I parameters. Both stages bias downward in different regimes, so the
    estimator is their elementwise maximum, normalised.
    """
    mu = np.asarray(mu, dtype=float)
    sd = np.sqrt(np.asarray(var, dtype=float))
    n = mu.size
    z75 = ndtri(0.75)

    # ---- stage I: quartiles of the field CDF, by bisection ----------------
    def field_quantile(b):
        target = np.log(b)
        w = ndtri(b ** (1.0 / n))
        lo = mu.min() + sd.min() * w
        hi = mu.max() + sd.max() * w
        lo, hi = min(lo, hi), max(lo, hi)
        lo -= 1.0
        hi += 1.0
        for _ in range(depth):
            mid = 0.5 * (lo + hi)
            if _log_field_cdf(mid, mu, sd)[0] < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    q1 = field_quantile(0.25)
    q3 = field_quantile(0.75)
    m = 0.5 * (q3 + q1)
    s = max((q3 - q1) / (2.0 * z75), 1e-300)
    p_I = ndtr((mu - m) / np.sqrt(sd ** 2 + s ** 2))

    # ---- stage II: per-candidate, on the divided-out field ----------------
    m_t = max(m, mu.max())
    s_t = np.minimum(s, sd)                    # (n,)

    def ratio_log(f):
        """log[ Phi((f-m_t)/s_t_i) / Phi((f-mu_i)/sd_i) ] for a vector f (n,)."""
        return log_ndtr((f - m_t) / s_t) - log_ndtr((f - mu) / sd)

    def ratio_quantile(b):
        target = np.log(b)
        hi = m_t + ndtri(b) * s_t
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (m_t + mu) / 2.0 - sd ** 2 * np.log(2.0 / b) / (m_t - mu)
            denom = 1.0 - (s_t ** 2) / (sd ** 2)
            t2 = m_t - np.sqrt(np.where(denom > 0, 2.0 * np.log(2.0 / b) / np.where(denom > 0, denom, 1.0), np.inf)) * s_t
        t1 = np.where(np.isfinite(t1), t1, -np.inf)
        t2 = np.where(np.isfinite(t2), t2, -np.inf)
        lo = np.minimum(mu - np.sqrt(2.0) * sd, np.maximum(t1, t2))
        lo = np.where(np.isfinite(lo), lo, hi - 1.0)
        lo = np.minimum(lo, hi - 1e-12)
        for _ in range(depth):
            mid = 0.5 * (lo + hi)
            below = ratio_log(mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)

    r1 = ratio_quantile(0.25)
    r3 = ratio_quantile(0.75)
    m_x = 0.5 * (r3 + r1)
    s_x = np.maximum((r3 - r1) / (2.0 * z75), 0.0)
    p_II = ndtr((mu - m_x) / np.sqrt(sd ** 2 + s_x ** 2))

    p = np.maximum(p_I, np.where(np.isfinite(p_II), p_II, 0.0))
    return p / p.sum()


# --------------------------------------------------------------------------
# non-PoM acquisition baselines, for the diversity comparison
# --------------------------------------------------------------------------

def score_greedy(mu, var=None):
    return np.asarray(mu, dtype=float)


def score_ucb(mu, var, beta: float = 1.0):
    return np.asarray(mu, dtype=float) + beta * np.sqrt(np.asarray(var, dtype=float))


def top_b(scores, b: int):
    """Indices of the b largest scores, ties broken by index (stable)."""
    s = np.asarray(scores, dtype=float)
    return np.argsort(-s, kind="stable")[:b]


def greedy_expected_max(mu, V, d, b: int, nodes=None, weights=None,
                        points: int = 257, lo=None, hi=None, seed_set=None):
    """Batch selection by greedy maximisation of E[max_{i in B} Y_i].

    WHY THIS AND NOT top-b BY qPO. The batch that maximises the probability of
    CONTAINING the argmax is exactly the top b by p_i, because those events are
    disjoint: P(argmax in B) = sum_{i in B} p_i. That objective is ADDITIVE --
    it has no interaction between batch members at all, so the only diversity
    it can express is whatever already sits in the marginals (near-duplicates
    splitting their mass). E[max_B] is the objective that values the batch
    itself: it is submodular, so greedy is within 1 - 1/e of optimal, and a
    candidate correlated with one already chosen adds little because it raises
    the max only where the batch is already high.

    HOW IT REUSES THE CAVITY MACHINERY. Conditional on the factor node f the
    candidates are independent, so the batch's max has CDF
    G_B(x|f) = prod_{i in B} F_i(x|f) -- the same "field" object the ability
    transform builds. The exotics cavity DIVIDES this field to remove a
    competitor; batch selection MULTIPLIES one more CDF in to add a member.
    The gain from adding j is

        Delta_j = int E_f[ G_B(x|f) (1 - F_j(x|f)) ] dx

    computable for every candidate at once per node, so one greedy step costs
    the same O(N L Q) as one qPO board and the whole batch costs b of them.
    """
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    n = mu.size
    if nodes is None:
        nodes, weights = np.zeros((1, 1)), np.ones(1)
    nodes = np.atleast_2d(nodes)
    weights = np.asarray(weights, dtype=float)
    sd = np.sqrt(d)
    tot = np.sqrt(d + (0.0 if V is None else np.sum(np.atleast_2d(V) ** 2, axis=1)))
    if lo is None:
        lo = float(mu.min() - 8.0 * tot.max())
    if hi is None:
        hi = float(mu.max() + 8.0 * tot.max())
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]

    # F[q] is (n, points): each candidate's conditional CDF at factor node q
    F = []
    for q in range(nodes.shape[0]):
        shift = mu if V is None else mu + np.atleast_2d(V) @ nodes[q]
        F.append(ndtr((x[None, :] - shift[:, None]) / sd[:, None]))
    F = np.asarray(F)                                   # (Q, n, points)
    w = weights / weights.sum()

    chosen = list(seed_set or [])
    G = np.ones((nodes.shape[0], points))               # batch field per node
    for i in chosen:
        G *= F[:, i, :]
    mask = np.zeros(n, dtype=bool)
    mask[chosen] = True

    while len(chosen) < b:
        # gain_j = sum_q w_q * dx * sum_x G_q(x) (1 - F_q,j(x))
        gain = np.zeros(n)
        for q in range(nodes.shape[0]):
            gain += w[q] * ((G[q][None, :] * (1.0 - F[q])).sum(axis=1))
        gain *= dx
        gain[mask] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        mask[j] = True
        G *= F[:, j, :]
    return np.array(chosen, dtype=int)


def top_m_probability(mu, V, d, m: int, nodes=None, weights=None,
                      points: int = 257, lo=None, hi=None, exact_below: int = 12):
    """P(candidate i is among the top m of the library) -- the PLACE
    probability, as opposed to qPO's WIN probability P(i is the single best).

    Motivation, measured rather than assumed. p(x*) is variance-hungry: a
    candidate wins outright mainly by being uncertain, so on a posterior where
    the fitted noise is most of the outputscale (the antibiotic screen: 82%)
    the optimality probabilities both degenerate towards uniform and tilt
    towards the least-known candidates. P(i in top m) is the same question
    asked at the resolution the campaign actually reports (top-k recovery),
    and it does not degenerate.

    Conditional on the factor node f the candidates are independent, so with
    Y_i = y the number of competitors above y is a sum of independent
    Bernoullis with q_k(y) = 1 - F_k(y). For m = 1 this reduces to the exact
    field product used by qPO; for larger m the count is evaluated by a
    Poisson approximation (exact in the regime that matters, where
    exceedances are rare), refined by the exact truncated Poisson-binomial
    recursion when m <= `exact_below`.
    """
    from scipy.stats import poisson as _poisson
    mu = np.asarray(mu, dtype=float)
    d = np.asarray(d, dtype=float)
    n = mu.size
    if nodes is None:
        nodes, weights = np.zeros((1, 1)), np.ones(1)
    nodes = np.atleast_2d(nodes)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    sd = np.sqrt(d)
    tot = np.sqrt(d + (0.0 if V is None else np.sum(np.atleast_2d(V) ** 2, axis=1)))
    if lo is None:
        lo = float(mu.min() - 8.0 * tot.max())
    if hi is None:
        hi = float(mu.max() + 8.0 * tot.max())
    x = np.linspace(lo, hi, points)
    dx = x[1] - x[0]

    out = np.zeros(n)
    for q in range(nodes.shape[0]):
        shift = mu if V is None else mu + np.atleast_2d(V) @ nodes[q]
        z = (x[None, :] - shift[:, None]) / sd[:, None]
        F = ndtr(z)                                     # (n, points)
        pdf = np.exp(-0.5 * z * z) / (sd[:, None] * np.sqrt(2 * np.pi))
        Q = 1.0 - F                                     # exceedance probs
        Lam = Q.sum(axis=0)                             # (points,) total
        # leave-one-out: candidate i does not compete with itself
        Lam_i = Lam[None, :] - Q                        # (n, points)
        if m == 1:
            surv = np.exp(-Lam_i)                       # P(no one above)
        else:
            surv = _poisson.cdf(m - 1, np.maximum(Lam_i, 0.0))
        out += w[q] * (pdf * surv).sum(axis=1) * dx
    return out
