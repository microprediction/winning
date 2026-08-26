"""v4: the block race in the cascade.

v3 raced with a global rank-4 factorization of the estimated residual
covariance and tightened shortlists 6-22% -- limited, we argued, by clustered
correlation being the geometry global low-rank represents worst. v4 tests
that argument directly: cluster the plausible set by proxy similarity
(greedy leader clustering), set within-cluster loadings from the calibrated
g(sim) curve, and race with block_race. Same coverage rule, same targets.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent
from blockrace import block_race

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

# ---- calibration (identical to v2/v3) --------------------------------------
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
# correlation-vs-similarity curve (as v3)
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

def mu_sd(s):
    return np.interp(s, bc, bm), np.maximum(np.interp(s, bc, bs), 1e-4)

# leader clustering at the similarity where g(sim) becomes material
G_THRESH = 0.25
sim_cut = float(np.interp(G_THRESH, gcurve, gcent))
rho_of = lambda s: np.clip(np.interp(s, gcent, gcurve), 0.0, 0.9)
print(f"cluster at proxy-sim > {sim_cut:.3f} (g >= {G_THRESH})", flush=True)

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
    # greedy leader clustering on similarity
    cl = -np.ones(RACE_TOP, int); lead = []
    for j in range(RACE_TOP):
        done = False
        for li, l in enumerate(lead):
            if S[j, l] > sim_cut:
                cl[j] = li; done = True; break
        if not done:
            cl[j] = len(lead); lead.append(j)
    # within-cluster loading: v_j = sd_j * sqrt(rho with its leader)
    rho = np.array([rho_of(S[j, lead[cl[j]]]) if lead[cl[j]] != j else
                    rho_of(0.9) for j in range(RACE_TOP)])
    # a cluster's members share the effect; singleton loading is irrelevant
    v = sd[topr] * np.sqrt(np.clip(rho, 0, 0.9))
    p_b = block_race(mu[topr], np.sqrt(np.maximum(sd[topr]**2 - v**2, 1e-8)),
                     cl, v, points=193, qa=7)
    p_b = np.maximum(p_b, 0)
    out_mass = p_ind[np.setdiff1d(np.arange(TOP), topr)].sum()
    p_b = p_b / max(p_b.sum(), 1e-12) * (1 - out_mass)
    t_race += time.time() - t1
    o = np.argsort(-p_b); csum = np.cumsum(p_b[o])
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        res[d]["m"].append(m); res[d]["hit"].append(bool(best_local in set(topr[o[:m]].tolist())))
print(f"block race {1000*t_race/len(ev):.0f} ms/query", flush=True)
print(f"{'target':>7} {'mean m':>7} {'med':>5} {'p90':>5} {'coverage':>9}   (v3 corr: 47.9/78.0/181.2; indep: 50.7/86.0/231.8)")
for d in DELTAS:
    md = np.array(res[d]["m"])
    print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):5.0f} {np.quantile(md,0.9):5.0f} {np.mean(res[d]['hit']):9.3f}")
