"""Item 1 rebuild: named-ensemble battery with stratified metrics and
raw-vs-contrast fitting compared head to head.

For each randomcov ensemble (n=300, seed 11, one draw; 2M-draw MC
referee restricted to resolvable entries, counts >= 25):

  raw      : the original pipeline (rank-3 fit of C + seriated blocks +
             m=5 residual eigencolumns, all in RAW correlation space)
  contrast : identical pipeline with the global factor stage replaced by
             factor_model_contrast (fit of P C P, the choice-relevant
             quotient; blocks and residual stages then work on the
             projected remainder)

Metrics per ensemble: half-L1 (TV), L-infinity, median/max abs err, and
abs err stratified by referee probability magnitude.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
from winning.factor.core import factor_model_contrast
import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)
from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod

def fit(C, space, k=3, n_blocks=20, m=5, log2nodes=11):
    n = len(C)
    if space == "contrast":
        V, D0 = factor_model_contrast(C, k)
        V = np.asarray(V, float)
    else:
        w_, U_ = np.linalg.eigh(C)
        V = U_[:, -k:] * np.sqrt(np.maximum(w_[-k:], 0))
    d = np.sqrt(np.clip(0.5 * (1 - C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    cluster = fcluster(Z, n_blocks, criterion="maxclust") - 1
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
                       .random_base2(log2nodes), 1e-12, 1 - 1e-12))
    return race_probabilities(mu, V=Vall, D=D, F=zq,
                              W=np.full(len(zq), 1.0 / len(zq)), points=257)

n, M = 300, 2_000_000
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2
KW = {CorrMethod.SPECTRUM: [{"kind": "marchenko_pastur"}, {"kind": "spiked"}],
      CorrMethod.FACTOR: [{"sparse_links": 30}]}
print(f"{'ensemble':24s} {'space':8s} {'TV':>9s} {'Linf':>9s} {'med':>8s} "
      f"{'p>1e-2':>8s} {'1e-3..1e-2':>10s} {'<1e-3':>8s}", flush=True)
for method, gen in CORR_GENERATORS.items():
    for kw in KW.get(method, [{}]):
        tag = method.value + ("" if not kw else ":" + str(list(kw.values())[0]))
        try:
            try:
                C = np.asarray(gen(n=n, rng=11, **kw))
            except TypeError:
                np.random.seed(11)
                C = np.asarray(gen(n=n, **kw))
            pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(4))
            seen = counts >= 25
            for space in ("raw", "contrast"):
                p1 = fit(C, space)
                err = np.abs(p1 - pmc)
                bands = [pmc > 1e-2, (pmc > 1e-3) & (pmc <= 1e-2),
                         seen & (pmc <= 1e-3)]
                bs = [f"{np.median(err[bd]):.1e}" if bd.any() else "--"
                      for bd in bands]
                print(f"{tag:24s} {space:8s} {0.5*err.sum():9.2e} "
                      f"{err[seen].max():9.2e} {np.median(err[seen]):8.1e} "
                      f"{bs[0]:>8s} {bs[1]:>10s} {bs[2]:>8s}", flush=True)
        except Exception as e:
            print(f"{tag:24s} ERROR: {type(e).__name__}: {e}", flush=True)
