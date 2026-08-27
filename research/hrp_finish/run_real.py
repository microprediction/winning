"""Real data: Ken French 30 industry portfolios, daily, walk-forward OOS.

Every claim so far was synthetic (and the generative model favored the
structured estimator by construction). This is the out-of-sample test on
half a century of real returns: monthly rebalance, estimation window T
days, hold 21 days, score realized annualized vol of each strategy over
the whole walk-forward. Long-only throughout.
"""
import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from winning.factor.races import race_probabilities, abilities_from_race

T_EST = int(sys.argv[1]) if len(sys.argv) > 1 else 60
HOLD = 21

raw = open("30_Industry_Portfolios_Daily.csv", errors="ignore").read().splitlines()
start = next(i for i, l in enumerate(raw) if l.strip()[:2] in ("19", "20") and "," in l)
rows, dates = [], []
for l in raw[start:]:
    parts = l.split(",")
    if len(parts) != 31 or not parts[0].strip().isdigit():
        break
    vals = np.array([float(x) for x in parts[1:]])
    if (vals <= -99).any():
        continue
    dates.append(parts[0].strip()); rows.append(vals)
R = np.array(rows) / 100.0
R = R[-13000:]                     # ~ last 50 years
n = R.shape[1]
print(f"data: {R.shape[0]} days x {n} industries")

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
    return w/w.sum(), Z, C, std

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

def lo_minvar(S):
    w = np.linalg.solve(S + 1e-10*np.eye(len(S)), np.ones(len(S)))
    w = np.clip(w, 0, None)
    return w/w.sum()

strats = ["EW", "HRP", "MinVar", "MinVar-LW", "MinVar-struct",
          "finish-g0.5", "finish-g1"]
rets = {s: [] for s in strats}
t = T_EST
while t + HOLD <= len(R):
    X = R[t-T_EST:t]
    S_hat = np.cov(X.T)
    try:
        w_hrp, Z, C_hat, std = hrp_weights(S_hat)
        coph = coph_corr(Z, n)
        V0, D0 = factor_approx(coph)
        mu = abilities_from_race(np.maximum(w_hrp, 1e-9)/np.maximum(w_hrp,1e-9).sum(),
                                 V=V0, D=D0, points=161)
        w = {"EW": np.ones(n)/n, "HRP": w_hrp}
        Sr = S_hat + 1e-4*np.trace(S_hat)/n*np.eye(n)
        w["MinVar"] = lo_minvar(Sr)
        w["MinVar-LW"] = lo_minvar(LedoitWolf().fit(X).covariance_)
        Vf, Df = factor_approx(C_hat)
        w["MinVar-struct"] = lo_minvar(np.outer(std, std)*(Vf@Vf.T + np.diag(Df)))
        for g, key in ((0.5, "finish-g0.5"), (1.0, "finish-g1")):
            Vg, Dg = factor_approx((1-g)*coph + g*C_hat)
            w[key] = race_probabilities(mu, V=Vg, D=Dg, points=161)
        hold = R[t:t+HOLD]
        for s in strats:
            rets[s].append(hold @ w[s])
    except Exception as e:
        pass
    t += HOLD
print(f"T_est={T_EST}, {len(rets['HRP'])} rebalances")
hr = np.std(np.concatenate(rets["HRP"]))*np.sqrt(252)
for s in strats:
    r = np.concatenate(rets[s])
    vol = np.std(r)*np.sqrt(252)
    print("  %-14s ann.vol %.4f   vs HRP %+.2f%%" % (s, vol, 100*(vol-hr)/hr))
