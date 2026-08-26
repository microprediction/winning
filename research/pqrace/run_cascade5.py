"""v5: the full Schur structure -- global coupling + blocks.

v3 (global rank-4 only): 47.9 / 78.0 / 181.2
v4 (blocks only):        55.5 / 88.9 / 207.0
Each holds half the correlation: the g(sim) curve has a diffuse component
from sim ~ 0.55 (global-ish) AND a tight component above 0.7 (blocky).
v5 races with nested_race: leading eigenvector of the estimated residual
covariance as the coupling, leader clusters on what remains as the blocks.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent
from blockrace import nested_race

N_Q, TOP, RACE_TOP, SEED = 400, 2048, 512, 7
rng = np.random.default_rng(SEED)

fps = np.asarray(np.load(HERE.parent / "qpo" / "snapshots" / "qm9_fps.npy", mmap_mode="r"))
Fn = fps / np.linalg.norm(fps, axis=1, keepdims=True)
P = rng.standard_normal((fps.shape[1], 128)).astype(np.float32) / np.sqrt(fps.shape[1])
X = fps @ P; X /= np.linalg.norm(X, axis=1, keepdims=True)
qi = rng.choice(len(X), N_Q, replace=False)
mask = np.ones(len(X), bool); mask[qi] = False
Qp, Qf = X[qi], Fn[qi]
DBp, DBf = X[mask], Fn[mask]
N = len(DBp)

n_cal = 48
cal = rng.choice(N_Q, n_cal, replace=False)
px, tx, qid = [], [], []
for k, i in enumerate(cal):
    s_p = DBp @ Qp[i]
    cand = np.argpartition(-s_p, TOP)[:TOP]
    px.append(s_p[cand]); tx.append(DBf[cand] @ Qf[i]); qid.append(np.full(TOP, k))
px, tx, qid = map(np.concatenate, (px, tx, qid))
nb = 40
qs = np.quantile(px, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
bi = np.clip(np.searchsorted(qs, px) - 1, 0, nb - 1)
bm = np.array([tx[bi == b].mean() for b in range(nb)])
bc = 0.5 * (qs[:-1] + qs[1:])
resid = tx - bm[bi]
for k in range(n_cal):
    m = qid == k
    resid[m] -= resid[m].mean()
bs = np.array([resid[bi == b].std() for b in range(nb)])
sims, rprod = [], []
for k, i in enumerate(cal[:24]):
    m = qid == k
    s_p = px[m]; r = resid[m]
    top = np.argsort(-s_p)[:256]
    sc = DBp[np.argpartition(-(DBp @ Qp[i]), TOP)[:TOP]][top]
    S = sc @ sc.T
    iu = np.triu_indices(len(top), 1)
    sd_t = np.interp(s_p[top], bc, bs)
    sims.append(S[iu]); rprod.append(np.outer(r[top], r[top])[iu] / max(np.outer(sd_t, sd_t).mean(), 1e-12))
sims, rprod = np.concatenate(sims), np.concatenate(rprod)
gb = np.linspace(sims.min(), sims.max() + 1e-9, 16)
gi = np.clip(np.searchsorted(gb, sims) - 1, 0, 14)
gcurve = np.array([np.clip(rprod[gi == b].mean(), 0.0, 0.95) for b in range(15)])
gcent = 0.5 * (gb[:-1] + gb[1:])
rho_of = lambda s: np.clip(np.interp(s, gcent, gcurve), 0.0, 0.9)
sim_cut = float(np.interp(0.25, gcurve, gcent))

def mu_sd(s):
    return np.interp(s, bc, bm), np.maximum(np.interp(s, bc, bs), 1e-4)

ev = np.setdiff1d(np.arange(N_Q), cal)
DELTAS = [0.10, 0.05, 0.01]
res = {d: {"m": [], "hit": []} for d in DELTAS}
t_race = 0.0
for i in ev:
    s_p = DBp @ Qp[i]
    cand = np.argpartition(-s_p, TOP)[:TOP]
    t_true = DBf[cand] @ Qf[i]
    best_local = int(np.argmax(t_true))
    mu, sd = mu_sd(s_p[cand])
    p_ind = pom_independent(mu, sd ** 2, points=129)
    p_ind = np.maximum(p_ind, 0); p_ind /= p_ind.sum()
    topr = np.argsort(-p_ind)[:RACE_TOP]
    t1 = time.time()
    E = DBp[cand[topr]]
    S = E @ E.T
    Sig = np.outer(sd[topr], sd[topr]) * rho_of(S)
    np.fill_diagonal(Sig, sd[topr] ** 2)
    # global coupling: leading eigenpair by power iteration
    u = rng.standard_normal(RACE_TOP)
    for _ in range(30):
        u = Sig @ u; u /= np.linalg.norm(u)
    lam1 = float(u @ Sig @ u)
    gvec = u * np.sqrt(max(lam1, 0.0))
    # cap so residual variance stays positive
    cap = 0.85 * sd[topr]
    gvec = np.sign(gvec) * np.minimum(np.abs(gvec), cap)
    var_rem = sd[topr] ** 2 - gvec ** 2
    # blocks on what the coupling does not explain
    cl = -np.ones(RACE_TOP, int); lead = []
    for j in range(RACE_TOP):
        done = False
        for li, l in enumerate(lead):
            if S[j, l] > sim_cut:
                cl[j] = li; done = True; break
        if not done:
            cl[j] = len(lead); lead.append(j)
    rho_res = np.zeros(RACE_TOP)
    for j in range(RACE_TOP):
        l = lead[cl[j]]
        if l != j:
            explained = gvec[j] * gvec[l] / max(sd[topr][j] * sd[topr][l], 1e-12)
            rho_res[j] = np.clip(rho_of(S[j, l]) - explained, 0.0, 0.85)
        else:
            rho_res[j] = 0.6
    v = np.sqrt(var_rem * np.clip(rho_res, 0, 0.85))
    d_id = np.sqrt(np.maximum(var_rem - v ** 2, 1e-8))
    p_n = nested_race(mu[topr], d_id, cl, v, g=gvec, gamma=1.0, points=161, qa=7, qf=9)
    p_n = np.maximum(p_n, 0)
    out_mass = p_ind[np.setdiff1d(np.arange(TOP), topr)].sum()
    p_n = p_n / max(p_n.sum(), 1e-12) * (1 - out_mass)
    t_race += time.time() - t1
    o = np.argsort(-p_n); csum = np.cumsum(p_n[o])
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        res[d]["m"].append(m); res[d]["hit"].append(bool(best_local in set(topr[o[:m]].tolist())))
print(f"nested race {1000*t_race/len(ev):.0f} ms/query", flush=True)
print(f"{'target':>7} {'mean m':>7} {'med':>5} {'p90':>5} {'coverage':>9}   (v3: 47.9/78.0/181.2  v4: 55.5/88.9/207.0  indep: 50.7/86.0/231.8)")
for d in DELTAS:
    md = np.array(res[d]["m"])
    print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):5.0f} {np.quantile(md,0.9):5.0f} {np.mean(res[d]['hit']):9.3f}")
