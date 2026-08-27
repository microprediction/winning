"""One race, five covariance grammars.

Every model in this package is the SAME Gaussian min-race, Y = mu + noise;
these dataclasses are declarative descriptions of the noise covariance that
admit O(N)-per-lattice-point evaluation. Pass any of them as `structure=` to
the front-door verbs (race_probabilities, calibrate_abilities, race_jacobian,
polish_race):

    Independent(D)                       Sigma = diag(D)
    Factor(V, D)                         Sigma = V V' + diag(D)
    Blocks(cluster, loading, D)          block-diagonal rank-1 + diag
    Nested(cluster, loading, D,
           coupling, gamma=1.0)          Factor(1) x Blocks: gamma dials the
                                         coupling from 0 (independent blocks)
                                         to 1 (fully coupled)
    Tree(cluster, loading, D,
         parent, strength)               hierarchy of uniform shared effects

Containments: Independent = Blocks with zero loadings = Factor with empty V;
Blocks = Tree of depth 1; Nested = Tree with a rank-1 root IF the coupling is
uniform, and strictly more general when it is not. D is always the
idiosyncratic VARIANCE, as everywhere in winning.factor.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Independent:
    D: object

    @property
    def n(self):
        return len(np.asarray(self.D))


@dataclass(frozen=True)
class Factor:
    V: object
    D: object

    @property
    def n(self):
        return len(np.asarray(self.D))


@dataclass(frozen=True)
class Blocks:
    cluster: object
    loading: object
    D: object

    @property
    def n(self):
        return len(np.asarray(self.D))


@dataclass(frozen=True)
class Nested:
    cluster: object
    loading: object
    D: object
    coupling: object
    gamma: float = 1.0

    @property
    def n(self):
        return len(np.asarray(self.D))


@dataclass(frozen=True)
class Tree:
    cluster: object
    loading: object
    D: object
    parent: object
    strength: object

    @property
    def n(self):
        return len(np.asarray(self.D))

    @classmethod
    def from_linkage(cls, Z):
        """The tree race whose implied correlation IS the cophenetic matrix
        of a scipy linkage (HRP's implicit covariance), exactly.

        Each leaf is its own cluster; internal node t (the k-th merge, at
        cophenetic distance h) carries lam_t^2 = rho_t - rho_parent(t) with
        rho_t = 1 - 2 h^2 (increments nonnegative by linkage monotonicity);
        D_i = 1 - rho at the leaf's first merge. Unit total variance per
        runner; see tests/test_race_invariants.py for the exactness proof."""
        Z = np.asarray(Z, float)
        n = len(Z) + 1
        nT = 2 * n - 1
        parent = -np.ones(nT, int)
        rho = np.zeros(nT)
        for k in range(len(Z)):
            a, b, h = int(Z[k, 0]), int(Z[k, 1]), Z[k, 2]
            t = n + k
            parent[a] = t; parent[b] = t
            # the tree race cannot represent NEGATIVE dependence (its
            # shared effects contribute nonnegative correlation), so
            # cophenetic correlations are floored at zero: merges above
            # the h = 1/sqrt(2) horizon leave their branches independent.
            # Without the floor, clipping the negative root increment
            # silently inflates every other implied correlation.
            rho[t] = max(1.0 - 2.0 * h * h, 0.0)
        lam = np.zeros(nT)
        for t in range(n, nT):
            pa = parent[t]
            lam2 = rho[t] - (rho[pa] if pa >= 0 else 0.0)
            lam[t] = np.sqrt(max(lam2, 0.0))
        D = np.array([1.0 - rho[parent[i]] for i in range(n)])
        return cls(cluster=np.arange(n), loading=np.zeros(n),
                   D=np.maximum(D, 1e-10), parent=parent, strength=lam)


def dispatch_probabilities(mu, structure, points=257, qa=9, qf=15, **kw):
    from .races import race_probabilities as _rp
    from .blocks import (block_race_probabilities, nested_race_probabilities,
                         tree_race_probabilities)
    if isinstance(structure, Independent):
        return _rp(mu, V=None, D=np.asarray(structure.D, float), **kw)
    if isinstance(structure, Factor):
        return _rp(mu, V=np.asarray(structure.V, float),
                   D=np.asarray(structure.D, float), **kw)
    if isinstance(structure, Blocks):
        return block_race_probabilities(mu, structure.cluster,
                                        structure.loading, structure.D,
                                        points=points, qa=qa)
    if isinstance(structure, Nested):
        return nested_race_probabilities(mu, structure.cluster,
                                         structure.loading, structure.D,
                                         coupling=structure.coupling,
                                         gamma=structure.gamma,
                                         points=points, qa=qa, qf=qf)
    if isinstance(structure, Tree):
        return tree_race_probabilities(mu, structure.cluster,
                                       structure.loading, structure.D,
                                       structure.parent, structure.strength,
                                       points=points, qa=qa)
    raise TypeError(f"unknown structure {type(structure).__name__}")
