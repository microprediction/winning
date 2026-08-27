"""Covariance families for the general-covariance race experiments.

Deliberately not only the easy factor case. The point of the one-factor
envelope estimator is that it does not assume Sigma is low rank, so it has to
be tested on covariances that a small factor model cannot represent: a slowly
decaying spectrum, clusters of near substitutes, and badly conditioned
matrices.

Every family is returned already projected onto the mean-zero subspace, since
a common shock cannot change the winner and carrying it around only distorts
conditioning numbers.
"""

from __future__ import annotations

import numpy as np

from envelope import project


def _sym(S):
    return 0.5 * (S + S.T)


def random_dense(n, seed=0, delta=0.5):
    """A: Sigma = A A' / n + delta I."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return project(A @ A.T / n + delta * np.eye(n))


def factor_plus_diagonal(n, seed=0, r=5, diag_lo=0.3, diag_hi=1.0):
    """B: Sigma = B B' + D. The control case a low-rank method should win."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, r)) / np.sqrt(r)
    d = rng.uniform(diag_lo, diag_hi, n)
    return project(B @ B.T + np.diag(d))


def decaying_spectrum(n, seed=0, alpha=0.5):
    """C: Haar basis with lambda_k proportional to k^-alpha.

    Small alpha is the adversarial case: the spectrum decays so slowly that no
    small factor model comes close.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = (np.arange(1, n + 1)) ** (-float(alpha))
    lam = lam / lam.mean()
    return project(_sym((Q * lam) @ Q.T))


def clustered(n, seed=0, n_clusters=10, rho_within=0.9, rho_between=0.1,
              hetero=True):
    """D: blocks of near substitutes. The regime where correlation matters most."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, n_clusters, n)
    C = np.full((n, n), rho_between)
    same = lab[:, None] == lab[None, :]
    C[same] = rho_within
    np.fill_diagonal(C, 1.0)
    if hetero:
        s = np.sqrt(rng.uniform(0.5, 1.5, n))
        C = C * np.outer(s, s)
    # make it a genuine covariance: shrink toward the diagonal until PSD
    w = np.linalg.eigvalsh(_sym(C))
    if w.min() < 1e-8:
        C = C + (1e-8 - w.min()) * np.eye(n)
    return project(_sym(C)), lab


def ill_conditioned(n, seed=0, log10_cond=6.0):
    """E: condition number spanning 1e2 to 1e8 after projection."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.logspace(0, -float(log10_cond), n)
    lam = lam / lam.mean()
    return project(_sym((Q * lam) @ Q.T))


FAMILIES = {
    "random_dense": lambda n, seed: random_dense(n, seed),
    "factor_r5": lambda n, seed: factor_plus_diagonal(n, seed, r=5),
    "factor_r20": lambda n, seed: factor_plus_diagonal(n, seed, r=20),
    "decay_a0.25": lambda n, seed: decaying_spectrum(n, seed, 0.25),
    "decay_a0.5": lambda n, seed: decaying_spectrum(n, seed, 0.5),
    "decay_a1": lambda n, seed: decaying_spectrum(n, seed, 1.0),
    "decay_a2": lambda n, seed: decaying_spectrum(n, seed, 2.0),
    "clustered": lambda n, seed: clustered(n, seed)[0],
    "illcond_1e4": lambda n, seed: ill_conditioned(n, seed, 4.0),
    "illcond_1e8": lambda n, seed: ill_conditioned(n, seed, 8.0),
}

ABILITY = {"flat": 0.25, "moderate": 1.0, "unequal": 3.0}


def abilities(n, scale, seed=0):
    rng = np.random.default_rng(seed + 991)
    mu = rng.standard_normal(n) * scale
    return mu - mu.mean()


def sqrt_psd(S):
    """Symmetric square root, used for raw sampling."""
    w, U = np.linalg.eigh(_sym(np.asarray(S, dtype=float)))
    return U * np.sqrt(np.maximum(w, 0.0))
