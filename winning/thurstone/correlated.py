"""Correlated races via latent Gaussian factors.

Performance of competitor i:

    X_i = mu_i + v_i . f + e_i,       f ~ N(0, I_k),   e_i ~ base_i (independent)

with the package's min-wins convention (lower performance wins; lower ability is
stronger). Conditionally on the factors f the competitors are independent, so the
fast field product / divide-one-out identity applies at every quadrature node:

    p_i = E_f [ integral pdf_i(x | f) * S_field(x | f) / S_i(x | f) dx ]

The factor expectation uses product Gauss-Hermite nodes for small k and
scrambled-Sobol quasi-Monte Carlo (via scipy, if installed) or seeded plain Monte
Carlo beyond. The transform is deterministic and smooth in the abilities, which is
what the inverse (fixed-point) calibration requires, and the full single-deletion
("scratch") ensemble comes from the same conditional field pass.

Notes on semantics: scratching a competitor is a MARGINAL operation -- the factor
structure of the survivors is unchanged. Conditioning/pinning is a different
intervention. See kinetics.microprediction.org/semantics.html.

Special cases:
  * loadings = 0 reduces to the ordinary independent race (Race.state_prices).
  * base = Density.gumbel_min and loadings = 0 gives the Luce/softmax law
    exactly: p_i = softmax(-mu_i / scale). With nonzero loadings this becomes a
    correlated softmax race -- a non-IIA generalization with an exact Luce limit.

The construction, its validation, and its limits (kinked kernels at large N) are
documented at https://kinetics.microprediction.org (experiment 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from .density import Density, _pdf_from_cdf

_TINY = 1e-300
_PFLOOR = 1e-15


# ---------------------------------------------------------------------------
# Factor-model fitting and quadrature nodes
# ---------------------------------------------------------------------------


def factor_model(C: np.ndarray, k: int, n_iter: int = 200, tol: float = 1e-10):
    """Fit C ~= V V^T + diag(D) by iterated principal-factor analysis.

    Unlike naive eigen-truncation (which invents off-diagonal correlation,
    catastrophically so near C = I), the iteration fits the off-diagonals:
    eigendecompose C - diag(D), re-estimate D from the exact diagonal, repeat.

    Returns (V, D): loadings (N x k) and idiosyncratic variances (N,).
    """
    C = np.asarray(C, dtype=float)
    D = np.full(len(C), 0.5 * float(np.mean(np.diag(C))))
    V = np.zeros((len(C), k))
    for _ in range(n_iter):
        lam, U = np.linalg.eigh(C - np.diag(D))
        idx = np.argsort(lam)[::-1][:k]
        V = U[:, idx] * np.sqrt(np.maximum(lam[idx], 0.0))
        D_new = np.clip(np.diag(C) - np.sum(V**2, axis=1), 1e-3, None)
        if np.abs(D_new - D).max() < tol:
            return V, D_new
        D = D_new
    return V, D


def hermite_nodes(k: int, Q: Optional[int] = None, prune: float = 1e-7):
    """Product Gauss-Hermite rule for E over N(0, I_k); returns (nodes, weights)."""
    if Q is None:
        Q = {1: 21, 2: 15, 3: 11, 4: 9}.get(k, 7)
    x, w = np.polynomial.hermite_e.hermegauss(Q)
    w = w / np.sqrt(2.0 * np.pi)
    if k == 1:
        return x[:, None], w
    grids = np.meshgrid(*([x] * k), indexing="ij")
    F = np.column_stack([g.ravel() for g in grids])
    W = np.ones(len(F))
    for d in range(k):
        W *= w[np.searchsorted(x, F[:, d])]
    keep = W > prune * W.max()
    return F[keep], W[keep]


def gaussian_nodes(k: int, n: int = 4096, seed: int = 0):
    """Equal-weight nodes for E over N(0, I_k), for k beyond Gauss-Hermite reach.

    Uses scrambled-Sobol quasi-Monte Carlo when scipy is available (error ~ n^-1
    on smooth integrands), otherwise seeded plain Monte Carlo (~ n^-1/2). Both
    are deterministic given the seed.
    """
    try:
        from scipy.stats import norm, qmc  # soft dependency

        m = int(np.ceil(np.log2(max(n, 2))))
        F = norm.ppf(qmc.Sobol(k, scramble=True, seed=seed).random_base2(m))
    except ImportError:
        F = np.random.default_rng(seed).standard_normal((n, k))
    return F, np.full(len(F), 1.0 / len(F))


def default_nodes(k: int):
    """Gauss-Hermite while affordable, Sobol/Monte Carlo beyond."""
    return hermite_nodes(k) if k <= 4 else gaussian_nodes(k)


# ---------------------------------------------------------------------------
# The correlated race
# ---------------------------------------------------------------------------


def _shifted_cdfs(base_cdfs: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Right-shift each competitor's CDF by a fractional number of lattice steps.

    base_cdfs: (N, L) per-competitor CDFs on the shared lattice.
    steps:     (m, N) shifts in lattice units (positive = weaker under min-wins).
    Returns (m, N, L). Matches Density.shift_fractional's interpolation.
    """
    m, N = steps.shape
    L = base_cdfs.shape[1]
    k0 = np.floor(steps).astype(int)
    frac = steps - k0
    idx = np.arange(L)[None, None, :]
    out = np.empty((m, N, L))
    for shift, weight in ((k0, 1.0 - frac), (k0 + 1, frac)):
        j = idx - shift[:, :, None]
        jc = np.clip(j, 0, L - 1)
        vals = np.take_along_axis(np.broadcast_to(base_cdfs, (m, N, L)), jc, axis=2)
        vals = np.where(j < 0, 0.0, vals)
        vals = np.where(j > L - 1, base_cdfs[None, :, -1:], vals)
        if weight is frac:
            out += weight[:, :, None] * vals
        else:
            out = weight[:, :, None] * vals
    return out


@dataclass
class FactorRace:
    """A race with latent-factor correlation: X_i = mu_i + v_i . f + e_i.

    bases:     one shared Density, or one per competitor (the idiosyncratic
               noise e_i; any shape -- Gaussian, skew-normal, Gumbel, ...).
    abilities: (N,) location offsets mu in PHYSICAL units (lower = stronger).
    loadings:  (N, k) Gaussian factor loadings in physical units. Zero rows
               recover the independent race.
    nodes, weights: quadrature over the factors; default via default_nodes(k).
    """

    bases: Union[Density, Sequence[Density]]
    abilities: np.ndarray
    loadings: np.ndarray
    nodes: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        self.abilities = np.asarray(self.abilities, dtype=float)
        self.loadings = np.atleast_2d(np.asarray(self.loadings, dtype=float))
        n = len(self.abilities)
        if self.loadings.shape[0] != n:
            raise ValueError("loadings must have one row per competitor")
        base_list: List[Density] = (
            list(self.bases) if isinstance(self.bases, (list, tuple)) else [self.bases] * n
        )
        if len(base_list) != n:
            raise ValueError("need one base density per competitor (or one shared)")
        lat = base_list[0].lattice
        for d in base_list:
            if d.lattice.L != lat.L or d.lattice.unit != lat.unit:
                raise ValueError("all bases must share the same lattice")
        self._lattice = lat
        self._base_cdfs = np.stack([d.cdf() for d in base_list])
        if self.nodes is None or self.weights is None:
            self.nodes, self.weights = default_nodes(self.loadings.shape[1])

    # -- internals ----------------------------------------------------------

    def _conditional_cdfs(self, keep: np.ndarray, node_slice: slice) -> np.ndarray:
        F = self.nodes[node_slice]
        shifts = (self.abilities[keep][None, :] + F @ self.loadings[keep].T) / self._lattice.unit
        return _shifted_cdfs(self._base_cdfs[keep], shifts)

    def _accumulate(self, keep: np.ndarray, want_deletions: bool):
        n = len(keep)
        L = self._base_cdfs.shape[1]
        p = np.zeros(n)
        q = np.zeros((n, n)) if want_deletions else None
        chunk = max(1, int(5e6 / (n * L)))
        for a in range(0, len(self.nodes), chunk):
            sl = slice(a, a + chunk)
            cdfs = self._conditional_cdfs(keep, sl)  # (m, n, L)
            W = self.weights[sl]
            pdf = np.diff(np.concatenate([np.zeros(cdfs.shape[:2] + (1,)), cdfs], axis=2), axis=2)
            logS = np.log(np.maximum(1.0 - cdfs, _TINY))
            logSfield = logS.sum(axis=1)  # (m, L)
            # rest-field CDF for each competitor, by division of survivals
            rest_cdf = 1.0 - np.exp(np.clip(logSfield[:, None, :] - logS, -745.0, 0.0))
            rest_pdf = np.diff(
                np.concatenate([np.zeros(cdfs.shape[:2] + (1,)), rest_cdf], axis=2), axis=2
            )
            payoff = pdf * (1.0 - rest_cdf) + 0.5 * pdf * rest_pdf  # win + half-tie
            p += W @ payoff.sum(axis=2)
            if want_deletions:
                for i in range(n):
                    r_cdf = 1.0 - np.exp(
                        np.clip(logSfield[:, None, :] - logS - logS[:, i : i + 1, :], -745.0, 0.0)
                    )
                    r_pdf = np.diff(
                        np.concatenate([np.zeros(cdfs.shape[:2] + (1,)), r_cdf], axis=2), axis=2
                    )
                    pay = pdf * (1.0 - r_cdf) + 0.5 * pdf * r_pdf
                    contrib = W @ pay.sum(axis=2)
                    contrib[i] = 0.0
                    q[i] += contrib
        total = p.sum()
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("correlated race integration failed")
        p = p / total
        if want_deletions:
            q = q / q.sum(axis=1, keepdims=True)
        return p, q

    # -- public API -----------------------------------------------------------

    def state_prices(self, keep: Optional[Sequence[int]] = None) -> np.ndarray:
        """Winning probabilities; keep restricts to a surviving subset (scratch)."""
        idx = np.arange(len(self.abilities)) if keep is None else np.asarray(keep, dtype=int)
        p, _ = self._accumulate(idx, want_deletions=False)
        return p

    def deletion_ensemble(self) -> np.ndarray:
        """q[i, j] = P(j wins | i scratched), for all pairs, from one field pass."""
        idx = np.arange(len(self.abilities))
        _, q = self._accumulate(idx, want_deletions=True)
        return q


def solve_abilities(
    bases: Union[Density, Sequence[Density]],
    loadings: np.ndarray,
    target_prices: Sequence[float],
    n_iter: int = 800,
    tol: float = 1e-4,
    nodes: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inverse transform under correlation: abilities whose FactorRace matches target.

    Damped fixed point on log-prices. The step is scaled by the smallest effective
    pairwise noise: strong correlation makes prices hyper-sensitive to ability gaps,
    and a fixed step silently diverges.
    """
    p = np.asarray(target_prices, dtype=float)
    if np.any(p <= 0):
        raise ValueError("all target prices must be positive")
    p = p / p.sum()
    V = np.atleast_2d(np.asarray(loadings, dtype=float))
    # Step from the TOTAL correlation (factors + idiosyncratic base variance):
    # the factor-part correlation alone can be 1 while competitors still carry
    # plenty of independent noise, and an unnecessarily small step stalls.
    base_list = list(bases) if isinstance(bases, (list, tuple)) else [bases] * len(p)
    idio_var = np.array([float(np.dot(b.p, (b.lattice.grid - b.mean()) ** 2)) for b in base_list])
    Sig = V @ V.T + np.diag(idio_var)
    d = np.sqrt(np.diag(Sig))
    rho = Sig / np.maximum(np.outer(d, d), _TINY)
    rho_max = float(np.max(rho - np.diag(np.diag(rho)))) if len(p) > 1 else 0.0
    step = 0.5 * np.sqrt(max(2.0 * (1.0 - min(rho_max, 1.0)), 1e-4))
    logp = np.log(p)
    mu = (logp - logp.mean()) / 2.0  # min-wins: overpriced -> raise (worsen) ability
    race = FactorRace(bases, mu, V, nodes=nodes, weights=weights)
    for _ in range(n_iter):
        model = np.maximum(race.state_prices(), _PFLOOR)
        resid = np.clip(np.log(model) - logp, -4.0, 4.0)
        mu = mu + step * resid
        mu -= mu.mean()
        race.abilities = mu
        if np.abs(resid).max() < tol:
            break
    return mu


def gaussian_factor_race(
    lattice,
    correlation: np.ndarray,
    k: int,
    abilities: Sequence[float],
    scale: float = 1.0,
    nodes: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> FactorRace:
    """Convenience: Gaussian race with a target correlation matrix.

    Fits correlation ~= V V^T + diag(D) by factor analysis, builds per-competitor
    normal bases with standard deviation scale * sqrt(D_i), and loadings scale * V.
    """
    V, D = factor_model(np.asarray(correlation, dtype=float), k)
    bases = [
        Density.skew_normal(lattice, loc=0.0, scale=float(scale * np.sqrt(di)), a=0.0) for di in D
    ]
    return FactorRace(bases, np.asarray(abilities, dtype=float), scale * V, nodes, weights)
