"""Two-stage cascade with an EMPIRICAL noise model: does calibrated depth
survive when the noise must be estimated rather than derived?

Stage 1 (cheap): 128-d random-projection L2 score.
Stage 2 (expensive): exact similarity in the full 2048-d fingerprint space.
Goal: shortlist, per query, the smallest set that contains the TRUE stage-2
best with probability 1 - delta, paying stage-2 evaluations only on the
shortlist. This is the bi-encoder -> cross-encoder shape: the noise model is
a heteroskedastic regression fitted on NEAR pairs (the v2 lesson), no
algebra available.
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent

N_Q, TOP, SEED = 400, 2048, 7
rng = np.random.default_rng(SEED)

fps = np.asarray(np.load(HERE.parent / "qpo" / "snapshots" / "qm9_fps.npy", mmap_mode="r"))
Fn = fps / np.linalg.norm(fps, axis=1, keepdims=True)        # exact space
P = rng.standard_normal((fps.shape[1], 128)).astype(np.float32) / np.sqrt(fps.shape[1])
X = fps @ P
X /= np.linalg.norm(X, axis=1, keepdims=True)                # proxy space
qi = rng.choice(len(X), N_Q, replace=False)
mask = np.ones(len(X), bool); mask[qi] = False
Qp, Qf = X[qi], Fn[qi]
DBp, DBf = X[mask], Fn[mask]
N = len(DBp)
print(f"N={N}, {N_Q} queries; proxy=128d projection, truth=2048d cosine", flush=True)

# ---- empirical noise model on near pairs: true_sim = f(proxy_sim) + noise --
n_cal = 48
cal = rng.choice(N_Q, n_cal, replace=False)
px, tx = [], []
for i in cal:
    s_p = DBp @ Qp[i]
    cand = np.argpartition(-s_p, TOP)[:TOP]
    px.append(s_p[cand]); tx.append(DBf[cand] @ Qf[i])
qid = np.concatenate([np.full(TOP, k) for k in range(len(cal))])
px, tx = np.concatenate(px), np.concatenate(tx)
# binned heteroskedastic regression, monotone mean
nb = 40
qs = np.quantile(px, np.linspace(0, 1, nb + 1)); qs[-1] += 1e-9
bi = np.clip(np.searchsorted(qs, px) - 1, 0, nb - 1)
bm = np.array([tx[bi == b].mean() for b in range(nb)])
# QUERY FIXED EFFECTS: per-query hardness shifts every similarity of that
# query together and cannot change its argmax; counting it as noise is the
# common-mode mistake for the third time. Remove each calibration query's
# mean residual (about the binned mean) before measuring the spread.
resid = tx - bm[bi]
qmean = np.zeros(len(cal))
for k in range(len(cal)):
    qmean[k] = resid[qid == k].mean()
resid = resid - qmean[qid]
bs = np.array([resid[bi == b].std() for b in range(nb)])
bc = 0.5 * (qs[:-1] + qs[1:])
print(f"noise model: {len(px)} near pairs, corr(proxy, true) = {np.corrcoef(px, tx)[0,1]:.3f}, "
      f"resid sd {bs.mean():.4f}", flush=True)

def mu_sd(s):
    return np.interp(s, bc, bm), np.maximum(np.interp(s, bc, bs), 1e-4)

ev = np.setdiff1d(np.arange(N_Q), cal)
DELTAS = [0.10, 0.05, 0.01]
depth = {d: [] for d in DELTAS}; hit = {d: [] for d in DELTAS}
ranks = []
t_race = 0.0
for i in ev:
    s_p = DBp @ Qp[i]
    cand = np.argpartition(-s_p, TOP)[:TOP]
    t_true = DBf[cand] @ Qf[i]                       # stage-2 on the pool, for scoring only
    best_local = int(np.argmax(t_true))
    order_proxy = np.argsort(-s_p[cand])
    ranks.append(int(np.where(order_proxy == best_local)[0][0]))
    t1 = time.time()
    mu, sd = mu_sd(s_p[cand])
    p = pom_independent(mu, sd ** 2, points=129)
    p = np.maximum(p, 0); p /= p.sum()
    o = np.argsort(-p); csum = np.cumsum(p[o])
    t_race += time.time() - t1
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        depth[d].append(m); hit[d].append(bool(best_local in set(o[:m].tolist())))
ranks = np.array(ranks)
print(f"race overhead {1000*t_race/len(ev):.1f} ms/query", flush=True)
print(f"\n{'target':>7} {'mean m':>7} {'med':>5} {'p90':>5} {'max':>6} {'coverage':>9}")
for d in DELTAS:
    md = np.array(depth[d])
    print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):5.0f} {np.quantile(md,0.9):5.0f} {md.max():6.0f} {np.mean(hit[d]):9.3f}")
print("\nfixed-depth coverage (proxy order):", "  ".join(f"m={m}: {(ranks<m).mean():.3f}" for m in (8, 32, 128, 512, 1024)))
print("oracle fixed depth:", "  ".join(f"{1-d:.2f}: m={int(np.quantile(ranks, 1-d, method='higher'))+1}" for d in DELTAS))
