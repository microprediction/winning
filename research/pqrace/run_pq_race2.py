"""v2: race on CONTRASTS -- decompose the ADC error into a query-common part,
a code-shared part, and an idiosyncratic residual, and let only the residual
discriminate.

v1 treated the whole per-candidate error variance as independent noise and
over-covered grotesquely (asked 95%, delivered 100% at mean depth 289 vs a
fixed depth of 32 giving 98.2%). The diagnosis is the racing lesson: a common
shift of every score cannot change the argmin, and errors shared through
codebook cells shift GROUPS together. Only the idiosyncratic residual makes
races close.

Error model, fitted on held-out (query, vector) pairs per block:
    e[q, i] = a_b[q] + g_b[q, code_i] + eps,     per block b
so for one query the discriminating variance of candidate i is
    v_i = sum_b Var(eps | block b, code_i)
with the code-shared g absorbing what v1 wrongly counted. (The g term still
correlates same-code candidates -- treating races across DIFFERENT codes as
independent underestimates nothing to first order because the g's difference
enters both sides; v2 keeps g's variance out of v and notes the residual
correlation as the v3 refinement.)
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent

M_BLOCKS, DSUB, KCODE = 16, 8, 256
N_Q = 500
SEED = 7
rng = np.random.default_rng(SEED)

t0 = time.time()
fps = np.asarray(np.load(HERE.parent / "qpo" / "snapshots" / "qm9_fps.npy", mmap_mode="r"))
P = rng.standard_normal((fps.shape[1], M_BLOCKS * DSUB)).astype(np.float32) / np.sqrt(fps.shape[1])
X = fps @ P
X /= np.linalg.norm(X, axis=1, keepdims=True)
q_idx = rng.choice(len(X), N_Q, replace=False)
mask = np.ones(len(X), bool); mask[q_idx] = False
Q, DB = X[q_idx], X[mask]
N = len(DB)

def kmeans(Z, k, iters=15, seed=0):
    r = np.random.default_rng(seed)
    C = Z[r.choice(len(Z), k, replace=False)].copy()
    for _ in range(iters):
        d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = Z[m].mean(0)
    return C

tr = DB[rng.choice(N, 20000, replace=False)]
books = [kmeans(tr[:, b*DSUB:(b+1)*DSUB], KCODE, seed=b) for b in range(M_BLOCKS)]
codes = np.empty((N, M_BLOCKS), np.uint8)
for b, C in enumerate(books):
    Zb = DB[:, b*DSUB:(b+1)*DSUB]
    for a in range(0, N, 20000):
        d = ((Zb[a:a+20000, None, :] - C[None]) ** 2).sum(-1)
        codes[a:a+20000, b] = d.argmin(1)
print(f"setup {time.time()-t0:.0f}s", flush=True)

# --- error decomposition per block: query-common, code-shared, residual -----
t0 = time.time()
n_fit, n_fq = 4000, 64
fit_db = rng.choice(N, n_fit, replace=False)
fq = Q[rng.choice(N_Q, n_fq, replace=False)]
bias = np.zeros((M_BLOCKS, KCODE)); v_resid = np.zeros((M_BLOCKS, KCODE))
for b, C in enumerate(books):
    qb = fq[:, b*DSUB:(b+1)*DSUB]
    xb = DB[fit_db, b*DSUB:(b+1)*DSUB]
    cb = C[codes[fit_db, b]]
    err = ((qb[:, None, :] - xb[None]) ** 2).sum(-1) - ((qb[:, None, :] - cb[None]) ** 2).sum(-1)
    err = err - err.mean(axis=1, keepdims=True)            # remove query-common a_b[q]
    cd = codes[fit_db, b]
    for c in range(KCODE):
        m = cd == c
        if m.sum() >= 3:
            E = err[:, m]
            g = E.mean(axis=1, keepdims=True)               # code-shared, per query
            bias[b, c] = E.mean()
            v_resid[b, c] = (E - g).var()                    # idiosyncratic only
        else:
            bias[b, c] = err.mean(); v_resid[b, c] = err.var()
print(f"error model {time.time()-t0:.0f}s", flush=True)
BIAS = bias[np.arange(M_BLOCKS)[None, :], codes].sum(1)
VAR = np.maximum(v_resid[np.arange(M_BLOCKS)[None, :], codes].sum(1), 1e-12)
print(f"median race sd: v2 {np.sqrt(np.median(VAR)):.4f}  (v1 counted code-shared too)", flush=True)

t0 = time.time()
truth = np.empty(N_Q, np.int64)
for i in range(N_Q):
    truth[i] = np.argmin(((DB - Q[i]) ** 2).sum(1))
print(f"truth {time.time()-t0:.0f}s", flush=True)

def adc(qv):
    tabs = [((qv[b*DSUB:(b+1)*DSUB][None, :] - books[b]) ** 2).sum(1) for b in range(M_BLOCKS)]
    return np.sum([tabs[b][codes[:, b]] for b in range(M_BLOCKS)], axis=0)

DELTAS = [0.10, 0.05, 0.01]
depth = {d: [] for d in DELTAS}; hit = {d: [] for d in DELTAS}
ranks = []
TOP = 4096
t_race = 0.0
for i in range(N_Q):
    sd_ = adc(Q[i]) + BIAS
    cand = np.argpartition(sd_, TOP)[:TOP]
    order_all = cand[np.argsort(sd_[cand])]
    r = int(np.where(order_all == truth[i])[0][0]) if truth[i] in cand else TOP
    ranks.append(r)
    t1 = time.time()
    p = pom_independent(-sd_[cand], VAR[cand], points=129)
    p = np.maximum(p, 0); p /= p.sum()
    o = np.argsort(-p); csum = np.cumsum(p[o])
    t_race += time.time() - t1
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        depth[d].append(m); hit[d].append(bool(truth[i] in set(cand[o[:m]])))
ranks = np.array(ranks)
print(f"\nrace overhead: {1000*t_race/N_Q:.1f} ms/query", flush=True)
print(f"{'target':>7} {'mean m':>7} {'med m':>6} {'p90':>5} {'max':>6} {'coverage':>9}")
for d in DELTAS:
    md = np.array(depth[d])
    print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):6.0f} {np.quantile(md,0.9):5.0f} {md.max():6.0f} {np.mean(hit[d]):9.3f}")
print("\nfixed-depth coverage:", "  ".join(f"m={m}: {(ranks<m).mean():.3f}" for m in (4, 8, 16, 32, 64, 128)))
print("depth of an ORACLE fixed rule at each target:",
      "  ".join(f"{1-d:.2f}: m={int(np.quantile(ranks, 1-d, method='higher'))+1}" for d in DELTAS))
