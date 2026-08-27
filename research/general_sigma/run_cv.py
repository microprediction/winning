"""General Sigma: can the race grammars serve as control variates?

Estimand: all-n win probabilities under a DENSE Sigma (no grammar).
Estimator: p_hat = p_analytic(Sigma_tilde) + mean_m [win_m(Sigma) -
win_m(Sigma_tilde)] with common random numbers through both square
roots. Unbiased for any surrogate; the question is variance reduction.

Surrogates: factor rank-1, factor rank-3 (eigen fits), cophenetic tree
(linkage + Tree.from_linkage -- NEW since the last attempt), and
equicorrelation (weak baseline). Couplings: Cholesky vs symmetric sqrt.
Truths: dense random corr; sample corr of a factor world at T=60;
factor + sparse off-grammar links.
"""
import sys, json
import numpy as np
from scipy.linalg import sqrtm, cholesky
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from winning.factor.races import race_probabilities
from winning.factor.blocks import tree_race_probabilities
from winning.factor.structures import Tree

rng = np.random.default_rng(11)
n = 20
mu = np.sort(rng.normal(size=n)) * 0.8

def onion_corr(n, eta=2.0, rng=rng):
    """Dense random correlation: random orthogonal frame, Dirichlet
    spectrum scaled to trace n, diagonal renormalized."""
    A = rng.normal(size=(n, n))
    Q, _ = np.linalg.qr(A)
    lam = rng.dirichlet(np.ones(n) * eta) * n
    C = Q @ np.diag(lam) @ Q.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def factor_world_sample(n, T=60, rng=rng):
    B = rng.normal(size=(n, 3)) * [0.6, 0.35, 0.25]
    X = B @ rng.normal(size=(3, T)) + rng.normal(size=(n, T)) * 0.7
    return np.corrcoef(X)

def factor_plus_sparse(n, rng=rng):
    B = rng.normal(size=(n, 2)) * [0.55, 0.3]
    S = B @ B.T
    for _ in range(6):                       # sparse off-grammar links
        i, j = rng.choice(n, 2, replace=False)
        S[i, j] += 0.35; S[j, i] = S[i, j]
    d = np.sqrt(np.diag(S) + 0.5)
    C = (S + 0.5 * np.eye(n)) / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    w, U = np.linalg.eigh(C)
    C = U @ np.diag(np.maximum(w, 1e-4)) @ U.T
    d = np.sqrt(np.diag(C)); C = C / np.outer(d, d)
    return C

def rank_fit(C, k):
    w, U = np.linalg.eigh(C)
    idx = np.argsort(-w)[:k]
    V = U[:, idx] * np.sqrt(np.maximum(w[idx], 0))
    D = np.maximum(np.diag(C) - (V**2).sum(1), 1e-4)
    return V, D

def surrogates(C):
    out = {}
    for k, name in ((1, "factor1"), (3, "factor3")):
        V, D = rank_fit(C, k)
        Ct = V @ V.T + np.diag(D)
        p = race_probabilities(mu, V=V, D=D, points=501)
        out[name] = (Ct, p)
    Z = linkage(squareform(np.sqrt(np.clip(0.5*(1-C), 0, 1)), checks=False),
                method="average")
    tr = Tree.from_linkage(Z)
    coph = np.eye(n)
    lam2 = tr.strength**2
    anc = []
    for i in range(n):
        a = set(); u = i
        while tr.parent[u] >= 0:
            a.add(tr.parent[u]); u = tr.parent[u]
        anc.append(a)
    for i in range(n):
        for j in range(i+1, n):
            coph[i, j] = coph[j, i] = sum(lam2[t] for t in anc[i] & anc[j])
    p = tree_race_probabilities(mu, tr.cluster, tr.loading, tr.D,
                                tr.parent, tr.strength, points=501)
    out["cophtree"] = (coph, p)
    rbar = (C.sum() - n) / (n * (n - 1))
    Ce = np.full((n, n), rbar); np.fill_diagonal(Ce, 1.0)
    V = np.full((n, 1), np.sqrt(max(rbar, 1e-6)))
    p = race_probabilities(mu, V=V, D=np.full(n, 1 - max(rbar, 1e-6)),
                           points=501)
    out["equicorr"] = (Ce, p)
    return out

def sqrt_of(C, kind):
    Cj = C + 1e-10 * np.eye(n)
    return cholesky(Cj, lower=True) if kind == "chol" else np.real(sqrtm(Cj))

def winners(L, z):
    y = mu[:, None] + L @ z
    return np.argmin(y, axis=0)

def estimate(C, surr, kind, M, rep_rng):
    L = sqrt_of(C, kind)
    z = rep_rng.standard_normal((n, M))
    w = winners(L, z)
    plain = np.bincount(w, minlength=n) / M
    est = {"plain": plain}
    for name, (Ct, p_an) in surr.items():
        Lt = sqrt_of(Ct, kind)
        wt = winners(Lt, z)
        diff = (np.bincount(w, minlength=n) - np.bincount(wt, minlength=n)) / M
        est[name] = p_an / p_an.sum() + diff
    return est

TRUTHS = {"dense_onion": onion_corr(n),
          "sample_T60": factor_world_sample(n),
          "factor_sparse": factor_plus_sparse(n)}
M, REPS = 4096, 40
results = {}
if __name__ != "__main__":
    TRUTHS = {}
for tname, C in TRUTHS.items():
    # reference: big sobol under chol
    from scipy.stats import qmc
    from scipy.special import ndtri
    eng = qmc.Sobol(n, scramble=True, seed=5)
    zz = ndtri(np.clip(eng.random_base2(18), 1e-12, 1-1e-12)).T
    ref = np.bincount(winners(sqrt_of(C, "chol"), zz), minlength=n) / zz.shape[1]
    surr = surrogates(C)
    for kind in ("chol", "sqrtm"):
        runs = {k: [] for k in ["plain"] + list(surr)}
        for r in range(REPS):
            est = estimate(C, surr, kind, M, np.random.default_rng(1000 + r))
            for k, v in est.items():
                runs[k].append(v)
        row = {}
        for k, v in runs.items():
            A = np.array(v)
            sd = A.std(axis=0).mean()
            rmse = np.sqrt(((A.mean(axis=0) - ref) ** 2).mean())
            row[k] = (sd, rmse)
        results[(tname, kind)] = row
        sd0 = row["plain"][0]
        line = "  ".join(f"{k}: VRF {((sd0/max(v[0],1e-15))**2):6.1f}"
                         for k, v in row.items() if k != "plain")
        print(f"{tname:14s} {kind:5s}  plain sd {sd0:.2e} | {line}", flush=True)
print("\n(VRF = variance reduction factor vs plain MC at equal draws; "
      "unbiased by construction)")
