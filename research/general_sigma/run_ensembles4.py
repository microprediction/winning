"""Ninth-review rebuild of the ensemble study.

Fixes actioned here:
  (7) twenty seeds per named ensemble, medians/q90/worst reported;
      randomcov pinned at commit 0d27a51.
  (5) the projected arm is the package one-call fit_covariance:
      quotient factor fit for the IDENTIFIED objective, blocks/promotion
      on the PROJECTED residual, closing (P.P) d = diag(P R P) solve.
  (8) kernel split into RBF vs Matern-3/2, stratified by length scale,
      at promoted residual ranks m in {5, 12}.

Referee: 1M-draw MC per (ensemble, seed), resolvable entries only.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
from winning.factor.core import factor_model_projected
import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)
from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod

n, M, SEEDS = 300, 1_000_000, 20
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2

def build(C, space, m=5):
    if space == "projected":
        # the package one-call: certified quotient factor fit, projected
        # residual for blocks/promotion, closing (P.P) d solve
        from winning.factor.core import fit_covariance
        Vall, D, F, W = fit_covariance(C, k=3, m=m, blocks=20)
        return race_probabilities(mu, V=Vall, D=D, F=F, W=W, points=257)
    w_, U_ = np.linalg.eigh(C)
    V = U_[:, -3:] * np.sqrt(np.maximum(w_[-3:], 0))
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, 20, criterion="maxclust") - 1
    v = np.zeros(n)
    R = C - V @ V.T
    for c in np.unique(cluster):
        idx = np.where(cluster == c)[0]
        if len(idx) < 2:
            continue
        Rb = R[np.ix_(idx, idx)].copy()
        np.fill_diagonal(Rb, 0.0)
        wb, Ub = np.linalg.eigh(Rb)
        if wb[-1] > 0:
            v[idx] = Ub[:, -1] * np.sqrt(wb[-1])
    ncl = len(np.unique(cluster))
    BD = np.zeros((n, ncl))
    for j, c in enumerate(np.unique(cluster)):
        idx = np.where(cluster == c)[0]
        BD[idx, j] = v[idx]
    E = C - V @ V.T - BD @ BD.T
    np.fill_diagonal(E, 0.0)
    wE, UE = np.linalg.eigh(E)
    Vres = UE[:, -m:] * np.sqrt(np.maximum(wE[-m:], 0))
    Vall = np.hstack([V, Vres, BD])
    D = np.maximum(np.diag(C) - (Vall ** 2).sum(1), 1e-3)
    r = Vall.shape[1]
    zq = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=3)
                       .random_base2(11), 1e-12, 1 - 1e-12))
    return race_probabilities(mu, V=Vall, D=D, F=zq,
                              W=np.full(len(zq), 1.0 / len(zq)), points=257)

def kernel_case(kind, ls):
    def gen(n, rng):
        X = rng.random((n, 2))
        r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) / ls
        if kind == "rbf":
            C = np.exp(-0.5 * r * r)
        else:
            C = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
        np.fill_diagonal(C, 1.0)
        w_, U_ = np.linalg.eigh(C)
        C = (U_ * np.maximum(w_, 1e-8)) @ U_.T
        dd = np.sqrt(np.diag(C)); C = C / np.outer(dd, dd)
        return C
    return gen

CASES = []
KW = {CorrMethod.SPECTRUM: [{"kind": "marchenko_pastur"}, {"kind": "spiked"}],
      CorrMethod.FACTOR: [{"sparse_links": 30}]}
for method, gen in CORR_GENERATORS.items():
    for kw in KW.get(method, [{}]):
        tag = method.value + ("" if not kw else ":" + str(list(kw.values())[0]))
        if method.value == "kernel":
            continue
        CASES.append((tag, gen, kw, 5))
for kind in ("rbf", "matern32"):
    for ls in (0.08, 0.2, 0.4):
        for m in (5, 12):
            CASES.append((f"kernel-{kind}-ls{ls}-m{m}",
                          kernel_case(kind, ls), {}, m))

import csv
outf = open("results_ensembles4.csv", "w", buffering=1)
out = csv.writer(outf)
out.writerow(["ensemble", "space", "seed", "tv", "linf", "med"])
for tag, gen, kw, m in CASES:
    rows = {"raw": [], "projected": []}
    for seed in range(SEEDS):
        try:
            rr = np.random.default_rng(1000 + seed)
            try:
                C = np.asarray(gen(n=n, rng=1000 + seed, **kw))
            except TypeError:
                try:
                    C = np.asarray(gen(n, rr))
                except TypeError:
                    np.random.seed(1000 + seed)
                    C = np.asarray(gen(n=n, **kw))
            pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(seed))
            seen = counts >= 25
            for space in ("raw", "projected"):
                p1 = build(C, space, m=m)
                err = np.abs(p1 - pmc)
                tv = 0.5 * err.sum(); li = err[seen].max()
                md = np.median(err[seen])
                rows[space].append((tv, li, md))
                out.writerow([tag, space, seed, tv, li, md])
        except Exception as e:
            print(f"{tag} seed {seed}: {type(e).__name__}: {e}", flush=True)
    for space in ("raw", "projected"):
        A = np.array(rows[space])
        if len(A):
            print(f"{tag:26s} {space:9s} med(TV) {np.median(A[:,0]):.2e} "
                  f"q90 {np.quantile(A[:,0],0.9):.2e} worst {A[:,0].max():.2e} "
                  f"| med(med abs) {np.median(A[:,2]):.1e}", flush=True)
