"""Test problems: covariance families x ability regimes, with high-accuracy
reference shares p* = W_Sigma(mu*) (max-wins) cached on disk.

Every sampler in this experiment uses the PROJECTED covariance
Sigma_c = P Sigma P (P = I - 11'/n): a common shock moves all utilities
together and cannot change the winner, so the winner law is unchanged, and
removing it stops couplings/factor fits wasting effort on a direction that
does not matter.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from envelope_fast import OneFactorRace, project, raw_winner_shares, sym_sqrt

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

FAMILIES = ["dense", "clustered", "clustered5", "spectral0.25", "spectral0.5", "spectral1",
            "spectral2", "factor", "nearsingular", "asymmetric"]
REGIMES = ["diffuse", "moderate", "rare", "extreme"]


def _haar(n, rng):
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))


def make_covariance(family: str, n: int, rng) -> np.ndarray:
    if family == "dense":
        A = rng.standard_normal((n, n))
        S = A @ A.T / n + 0.1 * np.eye(n)
    elif family in ("clustered", "clustered5"):
        # 'clustered': clusters of 10 (rank grows with n); 'clustered5': 5 clusters
        # whatever n (effective rank stays ~5, the regime the low-rank surrogate targets)
        size = max(2, min(10, n // 4)) if family == "clustered" else int(np.ceil(n / 5))
        k = int(np.ceil(n / size))
        lab = np.repeat(np.arange(k), size)[:n]
        rho_in, rho_out = 0.8, 0.2
        C = np.where(lab[:, None] == lab[None, :], rho_in, rho_out)
        np.fill_diagonal(C, 1.0)
        sd = np.exp(0.4 * rng.standard_normal(n))
        S = C * np.outer(sd, sd)
    elif family.startswith("spectral"):
        alpha = float(family[len("spectral"):])
        lam = np.arange(1, n + 1, dtype=float) ** (-alpha)
        lam *= n / lam.sum()
        U = _haar(n, rng)
        S = (U * lam) @ U.T
    elif family == "factor":
        B = rng.standard_normal((n, 3)) * np.array([1.0, 0.7, 0.5])
        R = np.exp(0.5 * rng.standard_normal(n)) * 0.5
        S = B @ B.T + np.diag(R)
    elif family == "nearsingular":
        r = max(1, n // 2)
        B = rng.standard_normal((n, r)) / np.sqrt(r)
        S = B @ B.T + 1e-8 * np.eye(n)
        w = np.linalg.eigvalsh(S)
        S *= 1.0 / w.max()          # cond ~ 1e8 after scaling
    elif family == "asymmetric":
        # first half strongly correlated (0.95), second half independent,
        # heteroskedastic across two decades: far from any substitution-symmetric model
        h = n // 2
        C = np.eye(n)
        C[:h, :h] = 0.95
        np.fill_diagonal(C, 1.0)
        sd = np.exp(np.linspace(-1.15, 1.15, n))
        rng.shuffle(sd)
        S = C * np.outer(sd, sd)
    else:
        raise ValueError(family)
    return 0.5 * (S + S.T)


def ability_scale(Sigma_c) -> float:
    n = len(Sigma_c)
    return float(np.sqrt(np.trace(Sigma_c) / (n - 1)))


def make_abilities(regime: str, n: int, s: float, rng) -> np.ndarray:
    if regime == "diffuse":
        mu = 0.3 * s * rng.standard_normal(n)
    elif regime == "moderate":
        mu = 0.6 * s * rng.standard_normal(n)
        top = rng.choice(n, size=min(5, n), replace=False)
        mu[top] += 1.5 * s
    elif regime == "rare":
        mu = 2.5 * s * rng.standard_normal(n)
    elif regime == "extreme":
        mu = 0.5 * s * rng.standard_normal(n)
        mu[rng.integers(n)] += 4.0 * s
    else:
        raise ValueError(regime)
    return mu - mu.mean()


@dataclass
class Problem:
    family: str
    regime: str
    n: int
    seed: int
    Sigma: np.ndarray
    Sigma_c: np.ndarray
    mu_star: np.ndarray
    p_star: np.ndarray
    p_star_se: np.ndarray
    p_star_M: int
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.family}_{self.regime}_n{self.n}_s{self.seed}"

    @property
    def scale(self):
        return ability_scale(self.Sigma_c)


def reference_shares(mu, Sigma_c, M: int, seed: int):
    """High-accuracy p*: Rao-Blackwellised one-factor envelope average over M
    independent residual draws (unbiased for every M; rare winners get exact
    conditional mass rather than zero counts)."""
    race = OneFactorRace(Sigma_c)
    p, se = race.rb_shares(mu, M, seed=seed)
    return p, se


def get_problem(family: str, regime: str, n: int, seed: int = 0,
                M_ref: int | None = None, force: bool = False) -> Problem:
    key = f"{family}_{regime}_n{n}_s{seed}"
    path = os.path.join(CACHE, key + ".npz")
    if os.path.exists(path) and not force:
        d = np.load(path, allow_pickle=True)
        return Problem(family, regime, n, seed, d["Sigma"], d["Sigma_c"],
                       d["mu_star"], d["p_star"], d["p_star_se"], int(d["p_star_M"]),
                       meta=d["meta"].item())
    rng = np.random.default_rng([seed, n, hash(family) % 10007, hash(regime) % 10007])
    Sigma = make_covariance(family, n, rng)
    Sigma_c = project(Sigma)
    s = ability_scale(Sigma_c)
    mu = make_abilities(regime, n, s, rng)
    if M_ref is None:
        M_ref = int(min(400_000, max(100_000, 2e8 // (n * n))))
    t0 = time.time()
    p, se = reference_shares(mu, Sigma_c, M_ref, seed=10_000 + seed)
    # the extreme regime is defined by max p > 0.5: bump the favourite if needed
    bumps = 0
    while regime == "extreme" and p.max() < 0.5 and bumps < 6:
        mu[np.argmax(mu)] += 0.5 * s
        mu -= mu.mean()
        p, se = reference_shares(mu, Sigma_c, M_ref, seed=10_000 + seed)
        bumps += 1
    # entrants that never touched an envelope have zero RB mass; the target
    # must be strictly positive for the log-domain reference inversions, so
    # floor far below anything identifiable and renormalise
    floor = 1e-10
    n_floored = int((p < floor).sum())
    p = np.maximum(p, floor)
    p /= p.sum()
    w = np.linalg.eigvalsh(Sigma)
    meta = {"n_floored": n_floored, "ref_seconds": time.time() - t0, "scale": s,
            "cond": float(w.max() / max(w.min(), 1e-300)),
            "p_max": float(p.max()), "n_below_1e-4": int((p < 1e-4).sum()),
            "n_below_1e-6": int((p < 1e-6).sum()), "bumps": bumps,
            "sum_p2": float((p ** 2).sum())}
    np.savez(path, Sigma=Sigma, Sigma_c=Sigma_c, mu_star=mu, p_star=p, p_star_se=se,
             p_star_M=M_ref, meta=meta)
    return Problem(family, regime, n, seed, Sigma, Sigma_c, mu, p, se, M_ref, meta)


if __name__ == "__main__":
    import sys
    fam = sys.argv[1] if len(sys.argv) > 1 else "dense"
    for reg in REGIMES:
        for n in (20, 50):
            pr = get_problem(fam, reg, n)
            print(pr.key, pr.meta)
