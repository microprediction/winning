"""How much finishing does HRP need, and with which matrix?

Peter's hypothesis: finish with the covariance HRP did NOT use. Formalized as
transport: invert w_HRP to abilities under HRP's own belief (cophenetic
matrix), re-price under belief_gamma = (1-gamma) coph + gamma R_hat.
gamma = 0 is the identity (null check); gamma = 1 prices under the full
sample covariance (all the unused information, estimation noise included).
Score: realized volatility under the TRUE covariance, 150 trials.
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from winning.factor.races import race_probabilities, abilities_from_race

rng = np.random.default_rng(7)
import sys
N_ASSETS, T_OBS, TRIALS = 30, int(sys.argv[1]) if len(sys.argv)>1 else 90, 150
SECTORS = np.repeat(np.arange(3), 10)

def true_cov(rng):
    beta = 0.16*rng.uniform(0.8, 1.2, N_ASSETS)
    sec = 0.10*rng.uniform(0.7, 1.3, N_ASSETS)
    idio = 0.15*rng.uniform(0.5, 1.5, N_ASSETS)
    S = np.outer(beta, beta)
    for s in range(3):
        m = SECTORS == s
        S[np.ix_(m, m)] += np.outer(sec[m], sec[m])
    S[np.diag_indices(N_ASSETS)] += idio**2
    return S

def hrp_weights(cov):
    std = np.sqrt(np.diag(cov))
    R = cov/np.outer(std, std)
    d = np.sqrt(np.clip(0.5*(1-R), 0, 1))
    Z = linkage(squareform(d, checks=False), method="average")
    # quasi-diag order
    def leaves(Z, n):
        order = [2*n-2]
        while max(order) >= n:
            t = max(order); i = order.index(t)
            a, b = int(Z[t-n,0]), int(Z[t-n,1])
            order[i:i+1] = [a, b]
        return order
    order = leaves(Z, N_ASSETS)
    w = np.ones(N_ASSETS)
    clusters = [order]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1: continue
            h = len(cl)//2
            a, b = cl[:h], cl[h:]
            def cvar(items):
                sub = cov[np.ix_(items, items)]
                ivp = 1/np.diag(sub); ivp /= ivp.sum()
                return ivp @ sub @ ivp
            va, vb = cvar(a), cvar(b)
            alpha = 1 - va/(va+vb)
            for i in a: w[i] *= alpha
            for i in b: w[i] *= 1-alpha
            nxt += [a, b]
        clusters = nxt
    return w/w.sum(), Z, R, std

def coph_corr(Z):
    c = 1.0 - 2.0*squareform(cophenet(Z))**2
    np.fill_diagonal(c, 1.0)
    return c

def factor_approx(M, rank=3):
    """rank-k + exact diagonal approx of a correlation-like matrix."""
    w_, U = np.linalg.eigh((M+M.T)/2)
    idx = np.argsort(-w_)[:rank]
    V = U[:, idx]*np.sqrt(np.maximum(w_[idx], 0))
    D = np.maximum(np.diag(M) - (V**2).sum(1), 1e-4)
    return V, D

GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0]
res = {g: [] for g in GAMMAS}
for g in GAMMAS: res[('lw', g)] = []
res['HRP'] = []; res['EW'] = []; res['MinVar'] = []
for tr in range(TRIALS):
    S_true = true_cov(rng)
    L = np.linalg.cholesky(S_true)
    X = rng.standard_normal((T_OBS, N_ASSETS)) @ L.T
    S_hat = np.cov(X.T)
    w_hrp, Z, R_hat, std = hrp_weights(S_hat)
    coph = coph_corr(Z)
    def vol(w): return np.sqrt(w @ S_true @ w)
    res['HRP'].append(vol(w_hrp))
    res['EW'].append(vol(np.ones(N_ASSETS)/N_ASSETS))
    Sr = S_hat + 1e-4*np.trace(S_hat)/N_ASSETS*np.eye(N_ASSETS)
    mv = np.linalg.solve(Sr, np.ones(N_ASSETS)); mv = np.clip(mv, 0, None)
    res['MinVar'].append(vol(mv/mv.sum()))
    # transport: invert under coph belief, re-price under blended belief
    V0, D0 = factor_approx(coph)
    mu = abilities_from_race(w_hrp, V=V0, D=D0, points=201)
    # Ledoit-Wolf-style shrink of the sample correlation toward identity
    lw = 0.3
    R_lw = (1-lw)*R_hat + lw*np.eye(N_ASSETS)
    for g in GAMMAS:
        Vg, Dg = factor_approx((1-g)*coph + g*R_hat)
        res[g].append(vol(race_probabilities(mu, V=Vg, D=Dg, points=201)))
        Vs, Ds = factor_approx((1-g)*coph + g*R_lw)
        res[('lw', g)].append(vol(race_probabilities(mu, V=Vs, D=Ds, points=201)))
print('realized volatility under TRUE covariance (lower is better), %d trials:' % TRIALS)
hr = np.array(res['HRP'])
for k in ['EW', 'HRP', 'MinVar'] + GAMMAS + [('lw', g) for g in GAMMAS]:
    v = np.array(res[k])
    lbl = ('gamma=%.2f' % k) if isinstance(k, float) else ('lw g=%.2f' % k[1] if isinstance(k, tuple) else k)
    dv = 100*(v-hr)/hr
    print('  %-11s %.4f +- %.4f   vs HRP: %+0.2f%%' % (lbl, v.mean(), v.std()/np.sqrt(TRIALS), dv.mean()))
