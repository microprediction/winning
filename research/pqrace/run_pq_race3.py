"""v3: the race view tells you which two scalars to store.

The ADC error is exactly, per block,
    e_ib = (||x_b||^2 - ||c_b||^2)  -  2 <q_b, r_ib>,     r_ib = x_b - c_ib.
The first term is a per-candidate CONSTANT: computable at index time, one
stored float removes it exactly (a known trick). The second is zero-mean and
linear in the query, with per-candidate variance proportional to the stored
residual energy rho_i = sum_b ||r_ib||^2 -- the second stored float. So:

    debiased score  s_i = ADC_i + const_i          (exact expectation)
    race variance   v_i = kappa * rho_i            (kappa calibrated once)

v1/v2 estimated variances from random (query, vector) pairs, which are far
pairs, and far-pair error geometry is nothing like the near-neighbour region
that decides races -- hence 20x over-coverage. v3's variance is per-candidate
and query-scaled by construction. Storage cost: 2 floats on top of 16 code
bytes.
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

t0 = time.time()
tr = DB[rng.choice(N, 20000, replace=False)]
books = [kmeans(tr[:, b*DSUB:(b+1)*DSUB], KCODE, seed=b) for b in range(M_BLOCKS)]
codes = np.empty((N, M_BLOCKS), np.uint8)
CONST = np.zeros(N); RHO = np.zeros(N)
for b, C in enumerate(books):
    Zb = DB[:, b*DSUB:(b+1)*DSUB]
    for a in range(0, N, 20000):
        d = ((Zb[a:a+20000, None, :] - C[None]) ** 2).sum(-1)
        codes[a:a+20000, b] = d.argmin(1)
    cb = C[codes[:, b]]
    CONST += (Zb ** 2).sum(1) - (cb ** 2).sum(1)
    RHO += ((Zb - cb) ** 2).sum(1)
print(f"index {time.time()-t0:.0f}s  (stored: 16 code bytes + 2 floats per vector)", flush=True)

# calibrate kappa on held-out queries: var(-2<q,R>) = kappa * rho
fq = Q[rng.choice(N_Q, 32, replace=False)]
fit_db = rng.choice(N, 3000, replace=False)
errs = np.zeros((len(fq), len(fit_db)))
for b, C in enumerate(books):
    qb = fq[:, b*DSUB:(b+1)*DSUB]
    rb = DB[fit_db, b*DSUB:(b+1)*DSUB] - C[codes[fit_db, b]]
    errs += -2.0 * (qb @ rb.T)
kappa = float(np.mean(errs ** 2 / RHO[fit_db][None, :]))
print(f"kappa = {kappa:.4f}  (isotropic prediction ~ 4/{M_BLOCKS*DSUB} * ||q||^2 = {4/128:.4f})", flush=True)

t0 = time.time()
truth = np.empty(N_Q, np.int64)
for i in range(N_Q):
    truth[i] = np.argmin(((DB - Q[i]) ** 2).sum(1))
print(f"truth {time.time()-t0:.0f}s", flush=True)

def adc(qv):
    tabs = [((qv[b*DSUB:(b+1)*DSUB][None, :] - books[b]) ** 2).sum(1) for b in range(M_BLOCKS)]
    return np.sum([tabs[b][codes[:, b]] for b in range(M_BLOCKS)], axis=0)

VAR = np.maximum(kappa * RHO, 1e-12)
DELTAS = [0.10, 0.05, 0.01]
depth = {d: [] for d in DELTAS}; hit = {d: [] for d in DELTAS}
ranks = []
TOP = 4096
t_race = 0.0
for i in range(N_Q):
    s = adc(Q[i]) + CONST                      # exact debias
    cand = np.argpartition(s, TOP)[:TOP]
    oa = cand[np.argsort(s[cand])]
    ranks.append(int(np.where(oa == truth[i])[0][0]) if truth[i] in cand else TOP)
    t1 = time.time()
    p = pom_independent(-s[cand], VAR[cand], points=129)
    p = np.maximum(p, 0); p /= p.sum()
    o = np.argsort(-p); csum = np.cumsum(p[o])
    t_race += time.time() - t1
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        depth[d].append(m); hit[d].append(bool(truth[i] in set(cand[o[:m]])))
ranks = np.array(ranks)
print(f"\nrace overhead {1000*t_race/N_Q:.1f} ms/query", flush=True)
print(f"{'target':>7} {'mean m':>7} {'med m':>6} {'p90':>5} {'max':>6} {'coverage':>9}")
for d in DELTAS:
    md = np.array(depth[d])
    print(f"{1-d:7.2f} {md.mean():7.1f} {np.median(md):6.0f} {np.quantile(md,0.9):5.0f} {md.max():6.0f} {np.mean(hit[d]):9.3f}")
print("\ndebiased fixed-depth coverage:", "  ".join(f"m={m}: {(ranks<m).mean():.3f}" for m in (4, 8, 16, 32, 64)))
print("oracle fixed depth:", "  ".join(f"{1-d:.2f}: m={int(np.quantile(ranks, 1-d, method='higher'))+1}" for d in DELTAS))
