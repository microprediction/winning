"""Product quantization search as a race: calibrated per-query rerank depth.

PQ search scores every database vector by a table-lookup (ADC) distance that
is wrong by a quantization error, then reranks the top m exactly. The rerank
depth m is a hand-tuned global constant in every production ANN system. But
"is the true nearest neighbour inside my shortlist?" is a PLACE question in a
race over N horses with noisy abilities and an estimable noise model:

    true_dist_i = adc_i + bias(codes_i) + noise_i,
    p_i = P(i is the true argmin),   coverage(S) = sum_{i in S} p_i

(the events are disjoint, so coverage is additive -- the same identity qPO
uses). Per query, take the smallest shortlist whose summed win probability
reaches the target 1 - delta. Clear-winner queries get short lists; ambiguous
queries in dense regions get long ones.

Data: QM9 fingerprints (133,885 x 2048) projected to 128 dims by a Gaussian
random projection -- real similarity structure, not synthetic clusters.
Error model fitted per (block, code) on a held-out training sample: the mean
of the per-block ADC error gives a DEBIASED score, its variance the race sd.
Errors treated independent across candidates (v1; code-sharing induces
correlation we ignore -- noted, testable later).
"""
import sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent / "qpo"))
from pom import pom_independent

M_BLOCKS, DSUB, KCODE = 16, 8, 256
N_TRAIN_KM = 20000
N_Q = 500
SEED = 7
rng = np.random.default_rng(SEED)

t0 = time.time()
fps = np.asarray(np.load(HERE.parent / "qpo" / "snapshots" / "qm9_fps.npy", mmap_mode="r"))
P = rng.standard_normal((fps.shape[1], M_BLOCKS * DSUB)).astype(np.float32) / np.sqrt(fps.shape[1])
X = fps @ P                                        # (N, 128) real-structured
X /= np.linalg.norm(X, axis=1, keepdims=True)
q_idx = rng.choice(len(X), N_Q, replace=False)
mask = np.ones(len(X), bool); mask[q_idx] = False
Q, DB = X[q_idx], X[mask]
N = len(DB)
print(f"data {time.time()-t0:.0f}s: DB {N} x {DB.shape[1]}, {N_Q} queries", flush=True)

# --- k-means per block ------------------------------------------------------
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
tr = DB[rng.choice(N, N_TRAIN_KM, replace=False)]
books = [kmeans(tr[:, b*DSUB:(b+1)*DSUB], KCODE, seed=b) for b in range(M_BLOCKS)]
print(f"codebooks {time.time()-t0:.0f}s", flush=True)

t0 = time.time()
codes = np.empty((N, M_BLOCKS), np.uint8)
for b, C in enumerate(books):
    Zb = DB[:, b*DSUB:(b+1)*DSUB]
    for a in range(0, N, 20000):
        d = ((Zb[a:a+20000, None, :] - C[None]) ** 2).sum(-1)
        codes[a:a+20000, b] = d.argmin(1)
print(f"encode {time.time()-t0:.0f}s", flush=True)

# --- error model per (block, code): bias and variance of the ADC error ------
t0 = time.time()
n_fit = 4000
fit_db = rng.choice(N, n_fit, replace=False)
fit_q = Q[rng.choice(N_Q, 64, replace=False)]
bias = np.zeros((M_BLOCKS, KCODE)); var = np.zeros((M_BLOCKS, KCODE))
for b, C in enumerate(books):
    qb = fit_q[:, b*DSUB:(b+1)*DSUB]
    xb = DB[fit_db, b*DSUB:(b+1)*DSUB]
    cb = C[codes[fit_db, b]]
    d_true = ((qb[:, None, :] - xb[None]) ** 2).sum(-1)       # (64, n_fit)
    d_adc = ((qb[:, None, :] - cb[None]) ** 2).sum(-1)
    err = d_true - d_adc
    for c in range(KCODE):
        m = codes[fit_db, b] == c
        if m.sum() >= 3:
            bias[b, c] = err[:, m].mean()
            var[b, c] = err[:, m].var()
        else:
            bias[b, c] = err.mean(); var[b, c] = err.var()
print(f"error model {time.time()-t0:.0f}s", flush=True)
BIAS = bias[np.arange(M_BLOCKS)[None, :], codes].sum(1)       # (N,)
VAR = np.maximum(var[np.arange(M_BLOCKS)[None, :], codes].sum(1), 1e-12)

# --- search -----------------------------------------------------------------
t0 = time.time()
truth = np.empty(N_Q, np.int64)
for i in range(N_Q):
    truth[i] = np.argmin(((DB - Q[i]) ** 2).sum(1))
print(f"exact truth {time.time()-t0:.0f}s", flush=True)

def adc(qv):
    tabs = [((qv[b*DSUB:(b+1)*DSUB][None, :] - books[b]) ** 2).sum(1) for b in range(M_BLOCKS)]
    return np.sum([tabs[b][codes[:, b]] for b in range(M_BLOCKS)], axis=0)

DELTAS = [0.10, 0.05, 0.01]
depth_ad = {d: [] for d in DELTAS}; hit_ad = {d: [] for d in DELTAS}
ranks_raw = []; ranks_deb = []
t0 = time.time()
TOP = 4096                      # race run over the ADC top-4096 only
for i in range(N_Q):
    s = adc(Q[i])
    ranks_raw.append(int(np.where(np.argsort(s) == truth[i])[0][0]) if True else 0)
    sd = s + BIAS
    cand = np.argpartition(sd, TOP)[:TOP]
    ranks_deb.append(int((sd[cand[np.argsort(sd[cand])]] < sd[truth[i]]).sum()) if truth[i] not in cand else
                     int(np.where(cand[np.argsort(sd[cand])] == truth[i])[0][0]))
    p = pom_independent(-sd[cand], VAR[cand], points=129)
    p = np.maximum(p, 0); p /= p.sum()
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    for d in DELTAS:
        m = int(np.searchsorted(csum, 1 - d) + 1)
        S = cand[order[:m]]
        depth_ad[d].append(m)
        hit_ad[d].append(bool(truth[i] in S))
print(f"race search {time.time()-t0:.0f}s", flush=True)

ranks_raw = np.array(ranks_deb)   # rank of truth under debiased adc ordering
print("\n=== calibrated per-query rerank depth (adaptive) vs fixed depth ===")
print(f"{'target':>8} {'mean m':>8} {'p90 m':>7} {'max m':>7} {'coverage':>9} | fixed-depth m for same coverage")
for d in DELTAS:
    md = np.array(depth_ad[d]); hd = np.mean(hit_ad[d])
    # fixed depth achieving the same empirical coverage
    need = int(np.quantile(ranks_raw, hd, method="higher")) + 1
    print(f"{1-d:8.2f} {md.mean():8.1f} {np.quantile(md,0.9):7.0f} {md.max():7.0f} {hd:9.3f} | {need}")
fixed_cov = [(m, float((ranks_raw < m).mean())) for m in (8, 32, 128, 512, 2048)]
print("\nfixed-depth coverage:", "  ".join(f"m={m}: {c:.3f}" for m, c in fixed_cov))
print(f"\nADC top-1 == truth (no rerank): {float((ranks_raw==0).mean()):.3f}")
