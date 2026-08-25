"""Residual estimators for p_Sigma(mu) - p* built from coupled draws.

Every method exposes  parts(mu, z, z0) -> (raw, [(ctrl_k, mean_k), ...])
with raw (M, n) the per-draw target contribution (one-hot winner, or the
Rao-Blackwellised conditional share vector) and each control (M, n) a
per-draw vector with KNOWN mean mean_k (n,).  The share estimate is

    p_hat = mean_m [ raw_m - sum_k beta_k (ctrl_km - mean_k) ]

which is unbiased for every beta, and tr Cov(p_hat) = (1/M) sum_i Var(c_i)
for the per-draw combination c.  Diagnostics therefore only need per-draw
first and second moments plus the winner-agreement count.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from envelope_fast import OneFactorRace
from references import GaussianReference, LogitReference, procrustes, target_sqrt, cayley_perturb


def onehot(idx, n):
    M = len(idx)
    E = np.zeros((M, n))
    E[np.arange(M), idx] = 1.0
    return E


class Target:
    """The hard race: U = mu + L z, z ~ N(0, I_n), max wins."""

    def __init__(self, problem, sqrt_kind: str = "sym"):
        self.problem = problem
        self.n = problem.n
        self.Sigma_c = problem.Sigma_c
        self.L = target_sqrt(problem.Sigma_c, sqrt_kind)
        self.sd = np.sqrt(np.maximum(np.diag(problem.Sigma_c), 1e-300))
        self.rb = OneFactorRace(problem.Sigma_c)

    def eps(self, z):
        return z @ self.L.T

    def winners(self, mu, z):
        return np.argmax(mu[None, :] + self.eps(z), axis=1)


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------

class Raw:
    name = "raw"
    smooth = False
    cost = 1.0            # matmuls per draw relative to raw winner counting

    def __init__(self, target: Target):
        self.t = target

    def parts(self, mu, z, z0=None):
        return onehot(self.t.winners(mu, z), self.t.n), []


class OneHotCV:
    """e_W - (e_V - q(nu)) with the reference winner V coupled to W.

    nu_fn(mu) gives the reference location; q_fn(nu) its exact shares.
    coupling in {'indep', 'common', 'procrustes', 'Q'} for Gaussian references,
    {'indep', 'commonz', 'rank'} for the logit reference.
    """
    smooth = False
    cost = 2.0

    def __init__(self, target: Target, ref, nu_fn, q_fn, coupling: str = "common",
                 Q=None, name: str | None = None):
        self.t, self.ref, self.nu_fn, self.q_fn, self.coupling = target, ref, nu_fn, q_fn, coupling
        self.is_logit = isinstance(ref, LogitReference)
        if not self.is_logit:
            if coupling == "procrustes":
                Q = procrustes(target.L, ref.L0)
            elif coupling in ("common", "indep"):
                Q = np.eye(target.n)
            self.Q = Q
            self.L0Q = ref.L0 @ Q
        self.name = name or f"cv[{ref.name},{coupling}]"

    def ref_shocks(self, z, z0):
        if self.is_logit:
            if self.coupling == "rank":
                u = ndtr(self.t.eps(z) / self.t.sd[None, :])
            elif self.coupling == "commonz":
                u = ndtr(z)
            else:
                u = ndtr(z0)
            return self.ref.gumbel_from_uniform(u)
        if self.coupling == "indep":
            return z0 @ self.L0Q.T
        return z @ self.L0Q.T

    def parts(self, mu, z, z0=None):
        n = self.t.n
        W = self.t.winners(mu, z)
        nu = self.nu_fn(mu)
        V = np.argmax(nu[None, :] + self.ref_shocks(z, z0), axis=1)
        return onehot(W, n), [(onehot(V, n), self.q_fn(nu))]


class RB:
    """Rao-Blackwellised conditional shares of the target (one-factor envelope)."""
    name = "rb"
    smooth = True
    cost = 1.0

    def __init__(self, target: Target):
        self.t = target

    def parts(self, mu, z, z0=None):
        return self.t.rb.conditional_shares(mu, self.t.rb.eta_from_z(z)), []


class RBCV:
    """q_Sigma(eta; mu) - (q_0(eta0; nu) - q(nu)), eta0 = A0 Q z coupled to eta = A z."""
    smooth = True
    cost = 2.0

    def __init__(self, target: Target, ref: GaussianReference, nu_fn, q_fn,
                 coupling: str = "common", Q=None, name: str | None = None):
        self.t, self.ref, self.nu_fn, self.q_fn, self.coupling = target, ref, nu_fn, q_fn, coupling
        if coupling == "procrustes":
            Q = procrustes(target.rb.A, ref.rb.A)
        elif coupling in ("common", "indep"):
            Q = np.eye(target.n)
        self.Q = Q
        self.A0Q = ref.rb.A @ Q
        self.name = name or f"rbcv[{ref.name},{coupling}]"

    def parts(self, mu, z, z0=None):
        q = self.t.rb.conditional_shares(mu, self.t.rb.eta_from_z(z))
        nu = self.nu_fn(mu)
        zz = z0 if self.coupling == "indep" else z
        q0 = self.ref.rb.conditional_shares(nu, zz @ self.A0Q.T)
        return q, [(q0, self.q_fn(nu))]


class Combined:
    """Several controls with fitted scalar coefficients beta_k."""
    cost = None

    def __init__(self, members, beta=None, name="combined"):
        self.members = members
        self.smooth = all(m.smooth for m in members)
        self.beta = beta
        self.name = name
        self.cost = 1.0 + sum(m.cost - 1.0 for m in members)

    def parts(self, mu, z, z0=None):
        raw = None
        ctrls = []
        for m in self.members:
            r, c = m.parts(mu, z, z0)
            if raw is None:
                raw = r
            ctrls.extend(c)
        return raw, ctrls


def combine(raw, ctrls, beta=None):
    c = raw.copy()
    for k, (C, mean) in enumerate(ctrls):
        b = 1.0 if beta is None else beta[k]
        c -= b * (C - mean[None, :])
    return c


def fit_beta(raw, ctrls):
    """Least-squares beta minimising sum_i Var(raw_i - sum_k beta_k C_ki)."""
    K = len(ctrls)
    if K == 0:
        return np.zeros(0)
    Rc = raw - raw.mean(axis=0)
    Cs = [C - C.mean(axis=0) for C, _ in ctrls]
    G = np.array([[np.sum(Cs[k] * Cs[l]) for l in range(K)] for k in range(K)])
    g = np.array([np.sum(Cs[k] * Rc) for k in range(K)])
    return np.linalg.solve(G + 1e-12 * np.trace(G) / K * np.eye(K), g)


# ---------------------------------------------------------------------------
# diagnostics: per-draw contribution moments and winner agreement
# ---------------------------------------------------------------------------

def per_draw_stats(method, mu, M: int, seed: int, beta=None, chunk: int = 2048):
    """Returns dict with tr_var (sum_i Var of per-draw contribution),
    agreement P(W=V) (one-hot methods; for RB methods the mean squared
    difference of the two conditional share vectors is reported instead),
    the mean contribution (= estimated residual offset) and its se."""
    n = method.t.n if hasattr(method, "t") else method.members[0].t.n
    rng = np.random.default_rng(seed)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    agree = 0.0
    sqdiff = 0.0
    done = 0
    while done < M:
        m = min(chunk, M - done)
        z = rng.standard_normal((m, n))
        z0 = rng.standard_normal((m, n))
        raw, ctrls = method.parts(mu, z, z0)
        c = combine(raw, ctrls, beta)
        s1 += c.sum(axis=0)
        s2 += (c * c).sum(axis=0)
        if ctrls:
            C0 = ctrls[0][0]
            if not method.smooth:
                agree += float(np.sum(np.argmax(raw, axis=1) == np.argmax(C0, axis=1)))
            sqdiff += float(np.sum((raw - C0) ** 2))
        done += m
    mean = s1 / M
    var = np.maximum(s2 / M - mean ** 2, 0.0)
    return {"tr_var": float(var.sum()), "mean": mean,
            "se": np.sqrt(var / M),
            "agreement": agree / M if ctrls and not method.smooth else np.nan,
            "sqdiff": sqdiff / M if ctrls else np.nan}


def pilot_beta(method, mu, M: int, seed: int):
    n = method.t.n if hasattr(method, "t") else method.members[0].t.n
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((M, n))
    z0 = rng.standard_normal((M, n))
    raw, ctrls = method.parts(mu, z, z0)
    return fit_beta(raw, ctrls)


# ---------------------------------------------------------------------------
# orthogonal coupling hill-climb (winner agreement / RB squared difference)
# ---------------------------------------------------------------------------

def optimize_Q(make_method, Q0, mu, n, pilot_M: int = 4096, steps: int = 40,
               k: int = 8, step: float = 0.3, seed: int = 0, objective: str = "agreement"):
    """Random-rotation hill climb over orthogonal Q starting from Q0.

    make_method(Q) builds the method for a candidate Q. The objective is the
    pilot winner agreement (one-hot) or minus the mean squared conditional
    share difference (RB), on a FIXED pilot sample so the comparison between
    candidates is a common-random-number comparison.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((pilot_M, n))
    z0 = rng.standard_normal((pilot_M, n))

    def score(Q):
        m = make_method(Q)
        raw, ctrls = m.parts(mu, z, z0)
        C0 = ctrls[0][0]
        if objective == "agreement":
            return float(np.mean(np.argmax(raw, axis=1) == np.argmax(C0, axis=1)))
        return -float(np.mean(np.sum((raw - C0) ** 2, axis=1)))

    Q = Q0.copy()
    best = score(Q)
    history = [best]
    accepted = 0
    for s in range(steps):
        cand = cayley_perturb(Q, rng, k, step)
        val = score(cand)
        if val > best:
            Q, best, accepted = cand, val, accepted + 1
        history.append(best)
    return Q, best, history, accepted
