"""Easy reference races (max-wins), their exact forward/inverse maps, and the
couplings that tie their noise to the target race's noise.

All Gaussian references are fitted to the projected target covariance and
expose a full n x n square root L0 so that eps0 = L0 Q z can be coupled to
the target's eps = L z through a common standard normal z.
"""

from __future__ import annotations

import sys
import os

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from winning.factor.core import (  # noqa: E402
    abilities_from_probabilities, abilities_from_probabilities_factor,
    hermite_nodes, qmc_nodes, win_probabilities, win_probabilities_factor)

from envelope_fast import OneFactorRace, project, sym_sqrt  # noqa: E402


def factor_fit(Sigma_c, r: int):
    """Sigma_c ~= B B' + D by iterated principal factors on the projected
    covariance (fits the off-diagonals rather than truncating eigenvalues)."""
    S = np.asarray(Sigma_c, dtype=float)
    n = len(S)
    if r == 0:
        return np.zeros((n, 0)), np.diag(S).copy() * n / (n - 1)
    D = np.full(n, 0.5 * float(np.mean(np.diag(S))))
    for _ in range(200):
        lam, U = np.linalg.eigh(S - np.diag(D))
        idx = np.argsort(lam)[::-1][:r]
        B = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new = np.clip(np.diag(S) - np.sum(B ** 2, axis=1), 1e-3 * np.mean(np.diag(S)), None)
        if np.abs(D_new - D).max() < 1e-10:
            D = D_new
            break
        D = D_new
    return B, D


def factor_nodes(r: int):
    if r == 0:
        return np.zeros((1, 1)), np.ones(1)
    if r == 1:
        return hermite_nodes(1, Q=31)
    if r == 2:
        return hermite_nodes(2, Q=15)
    if r == 3:
        return hermite_nodes(3, Q=11)
    return qmc_nodes(r, m=9 if r <= 8 else 10, seed=0)


class GaussianReference:
    """V = nu + eps0, eps0 ~ N(0, Sigma0), Sigma0 = B B' + diag(D)."""

    def __init__(self, B, D, name: str, points: int = 501):
        self.B = np.atleast_2d(np.asarray(B, dtype=float))
        if self.B.shape[0] != len(D):
            self.B = self.B.T
        self.D = np.asarray(D, dtype=float)
        self.n = len(self.D)
        self.rank = self.B.shape[1]
        self.name = name
        self.points = points
        self.Sigma0 = self.B @ self.B.T + np.diag(self.D)
        self.F, self.W = factor_nodes(self.rank)
        if self.rank == 0:
            self.B = np.zeros((self.n, 1))
        self._L0 = None
        self._rb = None

    # exact maps (min-wins core: p_max(mu) = p_min(-mu))
    def forward(self, nu):
        nu = np.asarray(nu, dtype=float)
        if self.rank == 0 and np.allclose(self.D, self.D[0]):
            return win_probabilities(-nu, float(np.sqrt(self.D[0])))
        return win_probabilities_factor(-nu, self.B, self.D, self.F, self.W,
                                        points=self.points)

    def invert(self, p):
        p = np.asarray(p, dtype=float)
        nu, info = abilities_from_probabilities_factor(p, self.B, self.D, self.F, self.W,
                                                       n_iter=60, tol=1e-7,
                                                       return_info=True, points=self.points)
        self.last_invert_info = info
        return -nu

    @property
    def L0(self):
        if self._L0 is None:
            self._L0 = sym_sqrt(project(self.Sigma0))
        return self._L0

    @property
    def rb(self):
        if self._rb is None:
            self._rb = OneFactorRace(project(self.Sigma0))
        return self._rb

    def sample_cost(self):
        return "O(n^2) per draw with dense coupling; O(n r) if sampled natively"


def iid_reference(Sigma_c):
    n = len(Sigma_c)
    s2 = float(np.trace(Sigma_c) / (n - 1))
    return GaussianReference(np.zeros((n, 0)), np.full(n, s2), "iid")


def diag_reference(Sigma_c):
    n = len(Sigma_c)
    return GaussianReference(np.zeros((n, 0)), np.diag(Sigma_c) * n / (n - 1), "diag")


def lowrank_reference(Sigma_c, r: int):
    B, D = factor_fit(Sigma_c, r)
    return GaussianReference(B, D, f"lowrank{r}")


class LogitReference:
    """V = nu + G, G_i iid Gumbel(0, tau): q = softmax(nu / tau), nu* = tau log p*."""

    def __init__(self, tau: float):
        self.tau = float(tau)
        self.name = f"logit(tau={tau:.3g})"
        self.rank = None

    def forward(self, nu):
        x = np.asarray(nu, dtype=float) / self.tau
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def invert(self, p):
        nu = self.tau * np.log(np.asarray(p, dtype=float))
        return nu - nu.mean()

    def jacobian(self, nu):
        q = self.forward(nu)
        return (np.diag(q) - np.outer(q, q)) / self.tau

    def gumbel_from_uniform(self, u):
        u = np.clip(u, 1e-300, 1 - 1e-16)
        return -self.tau * np.log(-np.log(u))


def logit_tau0(Sigma_c) -> float:
    """Pairwise scale match: Var(G_i - G_j) = pi^2 tau^2 / 3 against the mean
    pairwise Gaussian difference variance 2 tr(Sigma_c)/(n-1)."""
    n = len(Sigma_c)
    pair_var = 2.0 * float(np.trace(Sigma_c) / (n - 1))
    return float(np.sqrt(3.0 * pair_var) / np.pi)


# ---------------------------------------------------------------------------
# couplings
# ---------------------------------------------------------------------------

def target_sqrt(Sigma_c, kind: str = "sym"):
    if kind == "sym":
        return sym_sqrt(Sigma_c)
    if kind == "chol":
        n = len(Sigma_c)
        return np.linalg.cholesky(Sigma_c + 1e-10 * np.trace(Sigma_c) / n * np.eye(n))
    if kind == "eig":
        w, U = np.linalg.eigh(Sigma_c)
        return U * np.sqrt(np.maximum(w, 0.0))
    raise ValueError(kind)


def procrustes(L, L0):
    """Q = argmin_Q ||L - L0 Q||_F over orthogonal Q: SVD of L0' L."""
    U, _, Vt = np.linalg.svd(L0.T @ L, full_matrices=False)
    return U @ Vt


def cayley_perturb(Q, rng, k: int, step: float):
    """Q exp(step * A) with A a random skew matrix supported on k random
    coordinates: a cheap random rotation for the hill-climb."""
    n = Q.shape[0]
    idx = rng.choice(n, size=min(k, n), replace=False)
    A = rng.standard_normal((len(idx), len(idx)))
    A = A - A.T
    A /= max(np.linalg.norm(A, 2), 1e-300)
    w, V = np.linalg.eigh(1j * A * step)
    R = (V * np.exp(-1j * w)) @ V.conj().T
    Qn = Q.copy()
    Qn[:, idx] = Qn[:, idx] @ R.real
    return Qn
