"""One race, five covariance grammars.

Every model in this package is the SAME Gaussian min-race, Y = mu + noise;
these dataclasses are declarative descriptions of the noise covariance that
admit O(N)-per-lattice-point evaluation. Pass any of them as `structure=` to
the front-door verbs (race_probabilities, calibrate_abilities, race_jacobian,
polish_race):

    Independent(D)                       Sigma = diag(D)
    Factor(V, D)                         Sigma = V V' + diag(D); the keyword
                                         names are V and D, and V is (n, rank)
    Blocks(cluster, loading, D)          block-diagonal rank-1 + diag
    Nested(cluster, loading, D,
           coupling, gamma=1.0)          Factor(1) x Blocks: gamma dials the
                                         coupling from 0 (independent blocks)
                                         to 1 (fully coupled)
    Tree(cluster, loading, D,
         parent, strength)               hierarchy of uniform shared effects
                                         (leaf loadings are SCALAR: rank-one)

Containments: Independent = Blocks with zero loadings = Factor with empty V;
rank-one Blocks = Tree of depth 1; Nested = Tree with a rank-1 root IF the coupling is
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
    """Sigma = V V' + diag(D). The field names ARE the keywords.

    Factor(V=..., D=...), not Factor(loadings=..., idio=...) -- the
    natural guess raises TypeError (issue #66). V is (n, rank), one row
    per contestant; a bare length-n vector is accepted as rank-one
    loadings wherever the race verbs normalise it. D is the
    idiosyncratic VARIANCE, as everywhere in winning.factor.
    """

    V: object
    D: object

    @property
    def n(self):
        # from V, not D: a scalar D is one variance for every contestant
        # and carries no count (#510)
        V = np.asarray(self.V)
        return len(V) if V.ndim else len(np.asarray(self.D))


@dataclass(frozen=True)
class Blocks:
    cluster: object
    loading: object
    D: object

    @property
    def n(self):
        # from the labels, not D: a scalar D is one variance for every
        # contestant and carries no count (#510)
        return len(np.asarray(self.cluster))


@dataclass(frozen=True)
class Nested:
    cluster: object
    loading: object
    D: object
    coupling: object
    gamma: float = 1.0

    @property
    def n(self):
        # from the labels, not D: a scalar D is one variance for every
        # contestant and carries no count (#510)
        return len(np.asarray(self.cluster))


@dataclass(frozen=True)
class Tree:
    cluster: object
    loading: object
    D: object
    parent: object
    strength: object

    @property
    def n(self):
        # from the labels, not D: a scalar D is one variance for every
        # contestant and carries no count (#510)
        return len(np.asarray(self.cluster))

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
        # d[t] = 1 - rho_t, the cophenetic DISTANCE-variance 2 h^2 at node
        # t, kept directly rather than as 1 - rho: for near-duplicate
        # leaves 1 - (1 - 2h^2) cancels to a few ulps (h = 1e-8 gives
        # 2.2e-16 for an exact 2e-16), and every quantity below is a
        # difference of these. A node with no parent has d = 1 (rho = 0).
        d = np.ones(nT)
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
            d[t] = min(2.0 * h * h, 1.0)
        lam = np.zeros(nT)
        # "increments nonnegative by linkage monotonicity" is a PREMISE
        # of this construction, and it was never checked. centroid and
        # median linkage routinely produce inversions -- 9 of 40 random
        # 8-point centroid linkages here -- and `max(lam2, 0.0)` then
        # clipped the negative increment away, returning a tree whose
        # implied covariance is NOT the cophenetic one it promises. The
        # gap reached 0.029 in covariance and 1.4 percentage points on a
        # head-to-head price, silently (#133).
        bad = []
        for t in range(n, nT):
            pa = parent[t]
            lam2 = (d[pa] if pa >= 0 else 1.0) - d[t]   # rho_t - rho_pa
            if lam2 < -1e-9:
                bad.append((t, lam2))
            lam[t] = np.sqrt(max(lam2, 0.0))
        if bad:
            t0, d0 = min(bad, key=lambda x: x[1])
            raise ValueError(
                f"this linkage is not monotonic: node {t0} merges "
                f"{-d0:.3g} BELOW its parent, and {len(bad)} node(s) do. "
                "A tree race is a nested variance decomposition, so an "
                "inversion has no representation in it -- clipping the "
                "negative increment would return a different covariance "
                "from the cophenetic one this promises. Use a monotonic "
                "method (average, complete, ward), check with "
                "scipy.cluster.hierarchy.is_monotonic first, or build "
                "the Tree with explicit parent/strength.")
        # D_i = 1 - rho at the leaf's first merge = 2 h^2, EXACTLY. This
        # was max(D, 1e-10): an absolute floor that changed every
        # near-duplicate branch -- a two-leaf linkage at h = 1e-6 has
        # D = 2e-12, the floor made it 1e-10, and the pair priced 0.556
        # where the cophenetic model gives Phi(1) = 0.841 at every h,
        # with off-unit marginal variances (#430). A positive residual is
        # kept as it is; exactly zero (a merge at height 0: coincident
        # leaves) has no representation as a lattice noise and is refused
        # by name, as a zero D is everywhere else.
        D = np.array([d[parent[i]] if parent[i] >= 0 else 1.0
                      for i in range(n)])
        zero = np.flatnonzero(D <= 0.0)
        if zero.size:
            raise ValueError(
                f"leaf {int(zero[0])} merges at height 0 ({zero.size} "
                "leaf/leaves do): coincident leaves have zero idiosyncratic "
                "variance, which a tree race cannot price. Merge the "
                "duplicates, or perturb them deliberately.")
        return cls(cluster=np.arange(n), loading=np.zeros(n),
                   D=D, parent=parent, strength=lam)


def dispatch_probabilities(mu, structure, points=257, qa=9, qf=15, **kw):
    from .races import race_probabilities as _rp
    from .blocks import (block_race_probabilities, nested_race_probabilities,
                         tree_race_probabilities)
    # `points` is a NAMED parameter here, so it is not in **kw and these
    # two branches used to recurse into the race without it -- the
    # caller's resolution dropped a second time, after the front door
    # had already dropped it (#89).
    if isinstance(structure, Independent):
        return _rp(mu, V=None, D=np.asarray(structure.D, float),
                   points=points, **kw)
    if isinstance(structure, Factor):
        return _rp(mu, V=np.asarray(structure.V, float),
                   D=np.asarray(structure.D, float), points=points, **kw)
    # the hierarchical kernels are Gaussian, hard-race, probabilities-only.
    # Accepting and discarding base=, temperature= or return_slopes=
    # would answer a different question than the caller asked
    # (sixth review), so they are refused.
    # window= and delta= join the refusal list rather than the forward
    # list: the hierarchical kernels do their own windowing and take
    # neither argument, so forwarding them here would have dropped them
    # silently -- the same defect being fixed one level up (#89). The
    # Independent and Factor branches above DO forward them, through
    # **kw, because race_probabilities takes them.
    unsupported = [k for k, bad in
                   (("base", kw.get("base") not in (None, "normal")),
                    ("temperature", bool(kw.get("temperature"))),
                    ("return_slopes", bool(kw.get("return_slopes"))),
                    ("window", kw.get("window") not in (None, "bulk")),
                    ("delta", kw.get("delta") is not None
                     and float(kw["delta"]) != 1e-12))
                   if bad]
    if unsupported:
        lattice = [k for k in unsupported if k in ("window", "delta")]
        model = [k for k in unsupported if k not in ("window", "delta")]
        why = []
        if model:
            why.append(
                f"{', '.join(model)}: the block/nested/tree kernels are "
                "Gaussian hard races returning probabilities only")
        if lattice:
            why.append(
                f"{', '.join(lattice)}: these kernels choose their own "
                "lattice window per cluster")
        raise NotImplementedError(
            f"{type(structure).__name__} races do not support "
            + "; ".join(why)
            + ". Use structure=Factor (or V=/D=) for a non-normal base, "
            "finite temperature, slopes, or an explicit lattice window.")
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


def _loading_var(loading, n):
    """Per-runner shared variance from a loading that may be a scalar
    per runner (rank one) or an (n, r) matrix (rank r): the rank-r
    inversion crashed here on a broadcast before this existed."""
    L = np.asarray(loading, float)
    if L.ndim == 2:
        if L.shape[0] != n:
            L = L.T
        return (L ** 2).sum(axis=1)
    return L ** 2


def structure_variances(structure):
    """Total per-runner variance implied by a grammar structure (shared
    effects plus idiosyncratic): the marginal the generic inverter's
    independent surrogate preconditioner matches. A scalar D is the same
    variance for every contestant, as at every other boundary: it was
    taken as a zero-dimensional array and len(D) raised TypeError, so a
    scalar-D Blocks/Nested target priced by the forward could not be
    inverted through the same structure (#510)."""
    from ..shapes import as_idio
    if isinstance(structure, Independent):
        return np.array(np.asarray(structure.D, float), ndmin=1)
    D = as_idio(structure.D, structure.n)
    if isinstance(structure, Factor):
        V = np.asarray(structure.V, float)
        return D + (V ** 2).sum(axis=1)
    if isinstance(structure, Blocks):
        return D + _loading_var(structure.loading, len(D))
    if isinstance(structure, Nested):
        tot = D + _loading_var(structure.loading, len(D))
        if structure.coupling is not None and structure.gamma:
            g = np.atleast_2d(np.asarray(structure.coupling, float))
            if g.shape[0] != len(D):
                g = g.T
            tot = tot + (float(structure.gamma) ** 2) * (g ** 2).sum(axis=1)
        return tot
    if isinstance(structure, Tree):
        from .blocks import _scalar_loading
        tot = D + _scalar_loading(structure.loading, "tree races") ** 2
        parent = np.asarray(structure.parent, int)
        strength = np.asarray(structure.strength, float)
        # Cluster labels are arbitrary comparable values, and the forward
        # dispatch says so: every tree/block kernel in blocks.py remaps
        # them with np.unique(..., return_inverse=True). This cast them
        # to int and used them as node IDs, so a tree labelled 10/20
        # priced fine and then inverted with "index 10 is out of bounds
        # for axis 0 with size 3", and string labels died in int()
        # (#146). Same remapping here, so a label means the same thing
        # on both paths.
        _labels, cluster = np.unique(np.asarray(structure.cluster),
                                     return_inverse=True)
        anc = np.zeros(len(strength))
        for c in range(len(strength)):
            u, s = c, 0.0
            while parent[u] >= 0:
                s += strength[parent[u]] ** 2
                u = parent[u]
            anc[c] = s
        return tot + anc[cluster]
    raise TypeError(f"unknown structure {type(structure).__name__}")
