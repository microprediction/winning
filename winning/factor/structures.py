"""One race, five covariance grammars.

Every grammar describes the SAME min-race, Y = mu + noise; these
dataclasses are declarative descriptions of the noise covariance that admit
O(N)-per-lattice-point evaluation. With the normal base they are Gaussian
covariance grammars; Independent and Factor also accept any supported smooth
base, while Blocks, Nested and Tree are Gaussian only. Pass any of them as `structure=` to
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
         parent, strength, leaves=None)  hierarchy of uniform shared effects
                                         (leaf loadings are SCALAR: rank-one;
                                         leaves[j] = the label of leaf node
                                         j, default sorted labels, #372)

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


def _as_linkage(Z):
    """A scipy-style linkage, checked at the constructor's own boundary.

    Row k merges two DISTINCT clusters that already exist and are still
    active (ids below n + k, each merged at most once) at a finite,
    non-negative height. Unchecked, a self-reference [[3, 0, .5, 2], ...]
    built parent = [3, 4, 4, 3, -1], a cycle (#328); fractional ids were
    truncated; and a negative height was squared into the same tree as
    its absolute value (#459). Mirrors scipy's is_valid_linkage."""
    try:
        Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("a linkage must be a numeric (n-1) x 4 matrix") \
            from None
    if Z.size == 0:
        return Z.reshape(0, 4)
    if Z.ndim != 2 or Z.shape[1] < 3:
        raise ValueError(
            f"a linkage must be an (n-1) x 4 matrix; got shape {Z.shape}")
    n = len(Z) + 1
    used = np.zeros(2 * n - 1, bool)
    for k in range(len(Z)):
        h = Z[k, 2]
        if not np.isfinite(h) or h < 0:
            raise ValueError(
                f"linkage row {k} merges at height {h}: merge heights are "
                "finite non-negative distances (#459)")
        ids = Z[k, :2]
        for c in ids:
            if not np.isfinite(c) or c != np.floor(c):
                raise ValueError(
                    f"linkage row {k} names cluster {c}: cluster ids are "
                    "whole numbers (#328)")
            if not 0 <= c < n + k:
                raise ValueError(
                    f"linkage row {k} uses cluster {int(c)} before it is "
                    f"formed: row {k} may merge only ids below {n + k} "
                    "(#328)")
            if used[int(c)]:
                raise ValueError(
                    f"linkage row {k} reuses cluster {int(c)}, already "
                    "merged by an earlier row (#328)")
        if ids[0] == ids[1]:
            raise ValueError(
                f"linkage row {k} merges cluster {int(ids[0])} with itself "
                "(#328)")
        used[int(ids[0])] = used[int(ids[1])] = True
    return Z


@dataclass(frozen=True)
class Tree:
    cluster: object
    loading: object
    D: object
    parent: object
    strength: object
    # leaves[j] is the cluster label carried by leaf node j. None keeps
    # the sorted-label order, so labels 0..nC-1 are the leaf nodes; with
    # unequal-depth leaves any other spelling should say which label is
    # which leaf (#372). See blocks._tree_leaf_index.
    leaves: object = None

    def __post_init__(self):
        # whole node indices at the constructor: [2.7, 2, -1] used to be
        # stored and later truncated to [2, 2, -1] by pricing (#328)
        from .blocks import _as_parent, _tree_leaf_index
        _as_parent(self.parent)
        if self.leaves is not None:
            _tree_leaf_index(self.cluster, self.leaves)

    @property
    def leaf_index(self):
        """Each runner's leaf node index (#372)."""
        from .blocks import _tree_leaf_index
        return _tree_leaf_index(self.cluster, self.leaves)

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
        Z = _as_linkage(Z)
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
    if np.size(mu) == 2:
        # A two-runner Gaussian race is one contrast under every grammar;
        # Independent and Factor reach the closed form above, the
        # hierarchical kernels integrated a lattice and priced a 14-sd
        # pair tail at 0.107x its exact 1.04e-45 (#86). The same race as
        # a factor loading matrix takes the closed form too.
        V2, D2 = _pair_factor(structure)
        return _rp(mu, V=V2, D=D2, points=points)
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
        return tree_race_probabilities(mu, structure.leaf_index,
                                       structure.loading, structure.D,
                                       structure.parent, structure.strength,
                                       points=points, qa=qa)
    raise TypeError(f"unknown structure {type(structure).__name__}")


def _pair_factor(structure):
    """(V, D) of a two-runner Blocks, Nested or Tree race as a factor
    model with the same covariance, after the grammar's own validation:
    one column block per cluster (the member loadings), Nested's coupling
    scaled by gamma, and one column per internal tree node holding its
    strength for every runner beneath it."""
    from ..shapes import as_idio, as_loadings
    from .blocks import (_as_parent, _scalar_loading, _validate_tree)
    n = 2
    D = as_idio(structure.D, n, positive=True)
    if isinstance(structure, Tree):
        leaf = np.asarray(structure.leaf_index)
        if leaf.shape != (n,):
            # a three-leaf Tree priced a two-runner mu with its third
            # leaf silently dropped (#86), as Blocks and Nested refuse
            raise ValueError(f"cluster must have one label per runner; got "
                             f"{np.shape(structure.cluster)}")
        parent = _as_parent(structure.parent)
        lam = np.asarray(structure.strength, float)
        v = _scalar_loading(structure.loading, "tree races")
        if v.shape != (n,):
            v = np.broadcast_to(v, (n,)).astype(float)
        _, inv = np.unique(leaf, return_inverse=True)
        nC = int(inv.max()) + 1
        _validate_tree(parent, lam, nC)
        nT = len(parent)
        V = np.zeros((n, nC + nT))
        for i in range(n):
            c = int(inv[i])
            V[i, c] = v[i]
            u = c
            while parent[u] >= 0:
                V[i, nC + int(parent[u])] = lam[parent[u]]
                u = int(parent[u])
        return V, D
    L = as_loadings(structure.loading, n)
    _, inv = np.unique(np.asarray(structure.cluster), return_inverse=True)
    inv = np.asarray(inv).ravel()
    if inv.shape != (n,):
        raise ValueError(f"cluster must have one label per runner; got "
                         f"{np.shape(structure.cluster)}")
    r = L.shape[1]
    V = np.zeros((n, (int(inv.max()) + 1) * r))
    for i in range(n):
        V[i, inv[i] * r:(inv[i] + 1) * r] = L[i]
    if (isinstance(structure, Nested) and structure.coupling is not None
            and structure.gamma != 0.0):
        g = np.atleast_2d(np.asarray(structure.coupling, float))
        if g.shape[0] != n:
            g = g.T
        V = np.hstack([V, float(structure.gamma) * g])
    return V, D


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
            ga = float(structure.gamma)
            with np.errstate(over="ignore", under="ignore",
                             invalid="ignore"):
                cv = (ga ** 2) * (g ** 2).sum(axis=1)
            if not (np.all(np.isfinite(cv))
                    and np.finfo(float).tiny <= ga ** 2 < np.inf):
                # only gamma g is identified: coupling 1e307 g with gamma
                # 1e-307 squared each factor alone, 0 * inf (#722)
                cv = ((ga * g) ** 2).sum(axis=1)
            tot = tot + cv
        return tot
    if isinstance(structure, Tree):
        from .blocks import _scalar_loading
        tot = D + _scalar_loading(structure.loading, "tree races") ** 2
        from .blocks import (_without_common_shock, _as_parent,
                             _validate_tree)
        parent = _as_parent(structure.parent)
        # Cluster labels are remapped below; a common ancestor's shock
        # cancels and is no part of the surrogate's variance (#489)
        cluster = structure.leaf_index                  # (#146, #372)
        n_clusters = int(cluster.max()) + 1
        # the pricers' topology door: a forest (two roots) got variances
        # as if a zero-strength common root joined it, and a cycle spun
        # the walk below forever (#822)
        _validate_tree(parent, np.asarray(structure.strength, float),
                       n_clusters)
        strength = _without_common_shock(
            parent, np.asarray(structure.strength, float), n_clusters)
        # Cluster labels are arbitrary comparable values, and the forward
        # dispatch says so: every tree/block kernel in blocks.py remaps
        # them with np.unique(..., return_inverse=True). This cast them
        # to int and used them as node IDs, so a tree labelled 10/20
        # priced fine and then inverted with "index 10 is out of bounds
        # for axis 0 with size 3", and string labels died in int()
        # (#146). Same remapping here, so a label means the same thing
        # on both paths: Tree.leaf_index is that remapping, or the
        # caller's explicit leaves= map (#372).
        anc = np.zeros(len(strength))
        for c in range(len(strength)):
            u, s = c, 0.0
            while parent[u] >= 0:
                s += strength[parent[u]] ** 2
                u = parent[u]
            anc[c] = s
        return tot + anc[cluster]
    raise TypeError(f"unknown structure {type(structure).__name__}")
