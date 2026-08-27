"""Adaptive residual rank: m chosen from the residual spectrum (capture
85 percent of the positive eigenmass, m in [3, 16]) instead of fixed 5.
Rerun the named-ensemble battery; the kernel row is the one to watch.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)

from randomcov.corrgens.allcorrgens import CORR_GENERATORS, CorrMethod

def one_call_adaptive(mu, C, n_blocks=20, k=3, log2nodes=11, capture=0.85):
    n = len(C)
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
    pos = np.maximum(wE, 0)
    order = np.argsort(-pos)
    cum = np.cumsum(pos[order])
    tot = cum[-1] if cum[-1] > 0 else 1.0
    m = int(np.clip(np.searchsorted(cum / tot, capture) + 1, 3, 16))
    sel = order[:m]
    Vres = UE[:, sel] * np.sqrt(pos[sel])
    Vall = np.hstack([V, Vres, BD])
    D = np.maximum(np.diag(C) - (Vall ** 2).sum(1), 1e-3)
    r = Vall.shape[1]
    zq = ndtri(np.clip(qmc.Sobol(r, scramble=True, seed=3)
                       .random_base2(log2nodes), 1e-12, 1 - 1e-12))
    p = race_probabilities(mu, V=Vall, D=D, F=zq,
                           W=np.full(len(zq), 1.0 / len(zq)), points=257)
    return p, m

n, M = 300, 2_000_000
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2
KW = {CorrMethod.SPECTRUM: [{"kind": "marchenko_pastur"}, {"kind": "spiked"}],
      CorrMethod.FACTOR: [{"sparse_links": 30}]}
for method, gen in CORR_GENERATORS.items():
    for kw in KW.get(method, [{}]):
        tag = method.value + ("" if not kw else ":" + str(list(kw.values())[0]))
        try:
            try:
                C = np.asarray(gen(n=n, rng=11, **kw))
            except TypeError:
                np.random.seed(11)
                C = np.asarray(gen(n=n, **kw))
            p1, m = one_call_adaptive(mu, C)
            pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(4))
            seen = counts >= 25
            abs_err = np.abs(p1[seen] - pmc[seen])
            print(f"{tag:24s} m={m:2d}  resolvable={seen.sum():3d}  "
                  f"median abs err={np.median(abs_err):.1e}  "
                  f"max abs err={abs_err.max():.1e}", flush=True)
        except Exception as e:
            print(f"{tag:24s} ERROR: {type(e).__name__}: {e}", flush=True)
