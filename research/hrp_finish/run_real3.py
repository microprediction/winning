"""Last hypothesis: the left-out corr is real but SLOW.

Tree from the fast T=60 window (adapts); residual E = C_long - coph
from a LONG trailing window (250d, 1250d) where a persistent
cross-branch link is actually measurable. Full and rank-1 versions.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from winning.factor.races import race_probabilities, abilities_from_race

T_EST, HOLD, T_LONGS = 60, 21, (250, 1250)

raw = open("30_Industry_Portfolios_Daily.csv", errors="ignore").read().splitlines()
start = next(i for i, l in enumerate(raw) if l.strip()[:2] in ("19", "20") and "," in l)
rows = []
for l in raw[start:]:
    parts = l.split(",")
    if len(parts) != 31 or not parts[0].strip().isdigit():
        break
    vals = np.array([float(x) for x in parts[1:]])
    if (vals <= -99).any():
        continue
    rows.append(vals)
R = np.array(rows)[-13000:] / 100.0
n = R.shape[1]

def hrp_weights(cov):
    std = np.sqrt(np.diag(cov)); C = cov/np.outer(std, std)
    d = np.sqrt(np.clip(0.5*(1-C), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    def leaves(Z, n):
        order = [2*n-2]
        while max(order) >= n:
            t = max(order); i = order.index(t)
            order[i:i+1] = [int(Z[t-n,0]), int(Z[t-n,1])]
        return order
    order = leaves(Z, n)
    w = np.ones(n); clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1: continue
            h = len(cl)//2; a, b = cl[:h], cl[h:]
            def cvar(it):
                sub = cov[np.ix_(it, it)]
                iv = 1/np.diag(sub); iv /= iv.sum()
                return iv @ sub @ iv
            va, vb = cvar(a), cvar(b)
            al = 1 - va/(va+vb)
            for i in a: w[i] *= al
            for i in b: w[i] *= 1-al
            nxt += [a, b]
        clusters = nxt
    return w/w.sum(), Z, C

def coph_corr(Z, n):
    c = 1.0 - 2.0*squareform(cophenet(Z))**2
    np.fill_diagonal(c, 1.0)
    return c

def factor_approx(M, rank=3):
    w_, U = np.linalg.eigh((M+M.T)/2)
    idx = np.argsort(-w_)[:rank]
    V = U[:, idx]*np.sqrt(np.maximum(w_[idx], 0))
    D = np.maximum(np.diag(M) - (V**2).sum(1), 1e-4)
    return V, D

def resid_topk(E, k):
    w_, U = np.linalg.eigh((E+E.T)/2)
    idx = np.argsort(-np.abs(w_))[:k]
    return (U[:, idx]*w_[idx]) @ U[:, idx].T

ARMS = ["HRP"] + [f"L{TL}-{kind}" for TL in T_LONGS for kind in ("full", "r1")]
rets = {a: [] for a in ARMS}
t = max(T_EST, max(T_LONGS))
while t + HOLD <= len(R):
    X = R[t-T_EST:t]
    S_hat = np.cov(X.T)
    try:
        w_hrp, Z, C_hat = hrp_weights(S_hat)
        coph = coph_corr(Z, n)
        V0, D0 = factor_approx(coph)
        mu = abilities_from_race(np.maximum(w_hrp, 1e-9)/np.maximum(w_hrp,1e-9).sum(),
                                 V=V0, D=D0, points=161)
        hold = R[t:t+HOLD]
        rets["HRP"].append(hold @ w_hrp)
        for TL in T_LONGS:
            XL = R[t-TL:t]
            CL = np.corrcoef(XL.T)
            E = CL - coph
            for kind in ("full", "r1"):
                add = E if kind == "full" else resid_topk(E, 1)
                Vg, Dg = factor_approx(coph + add)
                w = race_probabilities(mu, V=Vg, D=Dg, points=161)
                rets[f"L{TL}-{kind}"].append(hold @ w)
    except Exception:
        pass
    t += HOLD
print(f"{len(rets['HRP'])} rebalances")
hr = np.std(np.concatenate(rets["HRP"]))*np.sqrt(252)
for a in ARMS:
    r = np.concatenate(rets[a]); vol = np.std(r)*np.sqrt(252)
    print("  %-11s ann.vol %.4f   vs HRP %+.2f%%" % (a, vol, 100*(vol-hr)/hr))
