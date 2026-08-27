"""Bug hunt: differential + invariant + metamorphic fuzzing of every kernel.

Random configurations across stressed regimes; every violation prints a
reproducible seed. Run: python fuzz_races.py [n_cases]
"""
import sys
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import winning.factor.races as races
import winning.factor.blocks as blocks
from winning.factor.races import race_probabilities, abilities_from_race
from winning.factor.blocks import (block_race_probabilities,
                                   nested_race_probabilities,
                                   tree_race_probabilities,
                                   block_race_jacobian,
                                   abilities_from_block_race)
from winning.factor.structures import Tree

N_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
FAILS = []

def report(name, seed, val, tol):
    if val > tol:
        FAILS.append((name, seed, val, tol))
        print(f"VIOLATION {name} seed={seed}  {val:.3e} > {tol:.0e}", flush=True)

def random_case(rng):
    n = int(rng.integers(2, 26))
    regime = rng.choice(["mild", "spread", "hopeless", "tinyD", "bigD",
                         "clumped"])
    mu = rng.normal(size=n)
    if regime == "spread":
        mu = mu * rng.uniform(3, 8)
    if regime == "hopeless":
        k = max(1, n // 4)
        mu[rng.choice(n, k, replace=False)] += rng.uniform(8, 20)
    D = 0.5 + rng.random(n)
    if regime == "tinyD":
        D = D * rng.uniform(0.01, 0.1)
    if regime == "bigD":
        D = D * rng.uniform(5, 30)
    if regime == "clumped":
        mu = np.round(mu, 1)          # exact ties
    r = int(rng.integers(0, 3))
    V = rng.normal(size=(n, max(r, 1))) * rng.uniform(0.1, 0.8) if r else None
    return n, mu, D, V

def mc_reference(mu, Sig, m=2 ** 14, seed=0):
    n = len(mu)
    L = np.linalg.cholesky(Sig + 1e-10 * np.eye(n))
    z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=seed)
                      .random_base2(int(np.log2(m))), 1e-12, 1 - 1e-12)).T
    return np.bincount(np.argmin(mu[:, None] + L @ z, axis=0),
                       minlength=n) / z.shape[1]

for case in range(N_CASES):
    seed = 10_000 + case
    rng = np.random.default_rng(seed)
    n, mu, D, V = random_case(rng)

    # --- factor family -----------------------------------------------------
    p = race_probabilities(mu, V=V, D=D, points=257)
    report("nonneg", seed, float(-p.min()), 1e-15)
    report("sums_to_one", seed, abs(p.sum() - 1), 1e-12)
    # shift invariance
    p2 = race_probabilities(mu + rng.uniform(-3, 3), V=V, D=D, points=257)
    report("shift_invariance", seed, np.abs(p - p2).max(), 5e-11)
    # permutation equivariance
    perm = rng.permutation(n)
    p3 = race_probabilities(mu[perm], V=None if V is None else V[perm],
                            D=D[perm], points=257)
    report("permutation", seed, np.abs(p[perm] - p3).max(), 5e-12)
    # rust vs numpy differential
    if races._HAVE_RUST:
        races._HAVE_RUST = False
        p_py = race_probabilities(mu, V=V, D=D, points=257)
        races._HAVE_RUST = True
        report("rust_vs_numpy", seed, np.abs(p - p_py).max(), 1e-11)
    # common scaling of (mu, sd) is a no-op
    c = rng.uniform(0.2, 5)
    p4 = race_probabilities(mu * c, V=None if V is None else V * c,
                            D=D * c * c, points=257)
    report("scale_invariance", seed, np.abs(p - p4).max(), 5e-9)
    # hopeless runner moves nobody
    mu_h = np.append(mu, mu.max() + 12 * np.sqrt(D.max()))
    D_h = np.append(D, D.mean())
    V_h = None if V is None else np.vstack([V, np.zeros(V.shape[1])])
    p5 = race_probabilities(mu_h, V=V_h, D=D_h, points=257)
    report("hopeless_others", seed, np.abs(p5[:n] - p).max(), 1e-5)
    report("hopeless_self", seed, float(p5[n]), 1e-6)
    # MC referee on a subsample
    if case % 20 == 0:
        Sig = (np.diag(D) if V is None else V @ V.T + np.diag(D))
        ref = mc_reference(mu, Sig, seed=seed)
        report("mc_referee_tv", seed, 0.5 * np.abs(p - ref).sum(), 8e-3)

    # --- clustered family --------------------------------------------------
    nc = int(rng.integers(1, max(2, n // 2) + 1))
    cluster = rng.integers(0, nc, size=n)
    ld = rng.uniform(0.05, 0.7, size=n)
    pb = block_race_probabilities(mu, cluster, ld, D, points=257)
    report("blk_sums", seed, abs(pb.sum() - 1), 1e-12)
    # relabeling clusters is a no-op
    relab = rng.permutation(1000)[cluster]
    pb2 = block_race_probabilities(mu, relab, ld, D, points=257)
    report("blk_relabel", seed, np.abs(pb - pb2).max(), 1e-12)
    # zero loading = independent
    pb0 = block_race_probabilities(mu, cluster, np.zeros(n), D, points=257)
    pi = race_probabilities(mu, D=D, points=257)
    report("blk_zero_is_indep", seed, np.abs(pb0 - pi).max(), 5e-9)
    # nested gamma=0 = blocks
    pn0 = nested_race_probabilities(mu, cluster, ld, D, coupling=ld,
                                    gamma=0.0, points=257)
    report("nested_g0", seed, np.abs(pn0 - pb).max(), 1e-14)
    # depth-1 tree with zero strengths = blocks
    nT = nc + 1
    parent = np.full(nT, nT - 1); parent[-1] = -1
    lam = np.zeros(nT)
    pt = tree_race_probabilities(mu, cluster, ld, D, parent, lam, points=257)
    report("tree_flat_is_blk", seed, np.abs(pt - pb).max(), 5e-9)
    # jacobian: rows sum to zero, off-diag sign, FD spot check
    if case % 10 == 0:
        J = block_race_jacobian(mu, cluster, ld, D, points=257)
        report("blkJ_rowsum", seed, np.abs(J.sum(1)).max(), 1e-9)
        off = J - np.diag(np.diag(J))
        report("blkJ_offdiag_sign", seed, float(-off.min()), 1e-12)
        j = int(rng.integers(0, n))
        h = 1e-5
        e = np.zeros(n); e[j] = h
        fd = (block_race_probabilities(mu + e, cluster, ld, D, points=257)
              - block_race_probabilities(mu - e, cluster, ld, D,
                                         points=257)) / (2 * h)
        report("blkJ_fd", seed, np.abs(J[:, j] - fd).max(), 5e-5)
    # inversion round trip
    if case % 10 == 5:
        tgt = rng.dirichlet(np.ones(n) * rng.uniform(0.5, 5))
        out = abilities_from_block_race(tgt, cluster, ld, D, points=257)
        report("blk_invert_residual", seed, float(out[1]), 1e-6)

    # --- random trees ------------------------------------------------------
    if case % 5 == 0:
        # random binary-ish tree over the clusters
        nodes = list(range(nc)); parent_t = [-1] * nc; lam_t = [0.0] * nc
        rng2 = np.random.default_rng(seed + 1)
        while len(nodes) > 1:
            i, j = rng2.choice(len(nodes), 2, replace=False)
            a, b = nodes[max(i, j)], nodes[min(i, j)]
            parent_t.append(-1); lam_t.append(float(rng2.uniform(0, 0.8)))
            t = len(parent_t) - 1
            parent_t[a] = t; parent_t[b] = t
            nodes = [x for x in nodes if x not in (a, b)] + [t]
        pt2 = tree_race_probabilities(mu, cluster, ld, D,
                                      np.array(parent_t), np.array(lam_t),
                                      points=257)
        report("tree_sums", seed, abs(pt2.sum() - 1), 1e-12)
        report("tree_nonneg", seed, float(-pt2.min()), 1e-15)
        # implied correlation of any from_linkage tree is in [0,1] and PSD
        Craw = np.corrcoef(rng2.normal(size=(n, max(n + 2, 8))))
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
        Z = linkage(squareform(np.sqrt(np.clip(0.5 * (1 - Craw), 0, 1)),
                               checks=False), method="average")
        tr = Tree.from_linkage(Z)
        lam2 = np.asarray(tr.strength) ** 2
        par = np.asarray(tr.parent)
        anc = []
        for i in range(n):
            a_ = set(); u = i
            while par[u] >= 0:
                a_.add(par[u]); u = par[u]
            anc.append(a_)
        imp = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                imp[i, j] = imp[j, i] = sum(lam2[t] for t in anc[i] & anc[j])
        report("linkage_corr_le_1", seed, float(imp.max() - 1.0), 1e-10)
        report("linkage_psd", seed,
               float(-np.linalg.eigvalsh(imp).min()), 1e-10)

print(f"\n{N_CASES} cases; {len(FAILS)} violations")
for f in FAILS[:20]:
    print("  ", f)
