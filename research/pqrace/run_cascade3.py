"""v3 of the cascade: the CORRELATED race.

The residual inflation left after removing common mode, far-pair geometry and
query effects is within-query correlation: similar candidates share proxy
error, so the independence model overstates the effective competition and
over-covers. The fix is the correlated race -- the thesis of this entire
repository, arrived at from a retrieval problem:

  calibration: estimate g(sim) = corr(resid_i, resid_j | proxy-sim_ij) in bins
  per query:   Sigma_ij = s_i s_j g(sim_ij) on the plausible set, factorized
               rank-r; p from pom_fast; same smallest-covering-set rule.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent, pom_fast, sobol_nodes
from factorize import contrast_factor

N_Q, TOP, RACE_TOP, SEED = 400, 2048, 512, 7
RANK = 4
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

# ---- calibration: mean curve, query effects, residual sd, and g(sim) ------
n_cal = 48
cal = rng.choice(N_Q, n_cal, replace=False)
px, tx, qid = [], [], []
sims, rprod = [], []
resids_by_q = []
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
# correlation vs candidate-candidate proxy similarity, from the top of each query
for k, i in enumerate(cal[:24]):
    m = qid == k
    s_p = px[m]; r = resid[m]
    top = np.argsort(-s_p)[:256]
    # candidate embeddings for this query's top: recompute indices
    sc = DBp[np.argpartition(-(DBp @ Qp[i]), TOP)[:TOP]][top]
    S = sc @ sc.T
    iu = np.triu_indices(len(top), 1)
    sims.append(S[iu]); rprod.append(np.outer(r[top], r[top])[iu] /
                                     max(np.outer(np.interp(s_p[top],bc,bs), np.interp(s_p[top],bc,bs)).mean(), 1e-12))
sims, rprod = np.concatenate(sims), np.concatenate(rprod)
gb = np.linspace(sims.min(), sims.max() + 1e-9, 16)
gi = np.clip(np.searchsorted(gb, sims) - 1, 0, 14)
gcurve = np.array([np.clip(rprod[gi == b].mean(), 0.0, 0.95) for b in range(15)])
gcent = 0.5 * (gb[:-1] + gb[1:])
print(f"g(sim) curve: {gcurve.round(2)}", flush=True)

def mu_sd(s):
    return np.interp(s, bc, bm), np.maximum(np.interp(s, bc, bs), 1e-4)

ev = np.setdiff1d(np.arange(N_Q), cal)
DELTAS = [0.10, 0.05, 0.01]
res = {mode: {d: {"m": [], "hit": []} for d in DELTAS} for mode in ("indep", "corr")}
t_race = {"indep": 0.0, "corr": 0.0}
nd, wt = sobol_nodes(RANK, m=6, seed=0)
for i in ev:
    s_p = DBp @ Qp[i]
    cand = np.argpartition(-s_p, TOP)[:TOP]
    t_true = DBf[cand] @ Qf[i]
    best_local = int(np.argmax(t_true))
    mu, sd = mu_sd(s_p[cand])
    # plausible set for the correlated race
    t1 = time.time()
    p_ind = pom_independent(mu, sd ** 2, points=129)
    p_ind = np.maximum(p_ind, 0); p_ind /= p_ind.sum()
    t_race["indep"] += time.time() - t1
    t1 = time.time()
    topr = np.argsort(-p_ind)[:RACE_TOP]
    E = DBp[cand[topr]]
    G = np.interp(E @ E.T, gcent, gcurve)
    Sig = np.outer(sd[topr], sd[topr]) * G
    np.fill_diagonal(Sig, sd[topr] ** 2)
    V, D = contrast_factor(Sig, RANK)
    p_c = pom_fast(mu[topr], V, D, nd, wt, points=129)
    p_c = np.maximum(p_c, 0)
    # mass outside the race set, from the independent model
    out_mass = p_ind[np.setdiff1d(np.arange(TOP), topr)].sum()
    p_c = p_c / p_c.sum() * (1 - out_mass)
    t_race["corr"] += time.time() - t1
    for mode, p_use, idx_map in (("indep", p_ind, np.arange(TOP)), ("corr", p_c, topr)):
        o = np.argsort(-p_use); csum = np.cumsum(p_use[o])
        for d in DELTAS:
            m = int(np.searchsorted(csum, 1 - d) + 1)
            chosen = set(idx_map[o[:m]].tolist())
            res[mode][d]["m"].append(m)
            res[mode][d]["hit"].append(bool(best_local in chosen))
for mode in ("indep", "corr"):
    print(f"\n--- {mode} race ({1000*t_race[mode]/len(ev):.0f} ms/query) ---", flush=True)
    print(f"{'target':>7} {'mean m':>7} {'med':>5} {'p90':>5} {'coverage':>9}")
    for d in DELTAS:
        md = np.array(res[mode][d]["m"])
        print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):5.0f} {np.quantile(md,0.9):5.0f} {np.mean(res[mode][d]['hit']):9.3f}")
