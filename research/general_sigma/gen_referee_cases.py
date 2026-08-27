"""Emit referee cases: our one-call p for runners across probability
bands, plus (mu, Sigma) for independent adjudication in R."""
import json
import numpy as np

src = open("run_large_n2.py").read()
exec(src[:src.index('rng = np.random.default_rng(21)')])

rng = np.random.default_rng(21)
n = 500
G0 = rng.normal(size=(n, 3)) * [0.55, 0.3, 0.2]
blocks0 = rng.integers(0, 20, size=n)
v0 = 0.35 + 0.2 * rng.random(n)
C = G0 @ G0.T
for c in range(20):
    idx = np.where(blocks0 == c)[0]
    C[np.ix_(idx, idx)] += np.outer(v0[idx], v0[idx])
C += 0.03 * onion(n, rng)
C += np.diag(np.maximum(1.0 - np.diag(C), 0.05))
d_ = np.sqrt(np.diag(C)); C = C / np.outer(d_, d_)
mu = np.sort(np.random.default_rng(21).normal(size=n)) * 1.2
# (matches run_large_n2's blocky/wide up to rng stream; recompute mu the
# same way it does)
rng2 = np.random.default_rng(21)
_ = rng2.normal(size=(n, n))   # not needed; we just re-derive mu freshly
mu = np.sort(np.random.default_rng(99).normal(size=n)) * 1.2

p1 = one_call(mu, C, n_blocks=20, log2nodes=12)
sel = []
for lo, hi in [(1e-2, 1.0), (1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6),
               (1e-12, 1e-8), (1e-18, 1e-12)]:
    band = np.where((p1 >= lo) & (p1 < hi))[0]
    if len(band):
        take = band[np.linspace(0, len(band) - 1, min(5, len(band))).astype(int)]
        sel += [int(i) for i in take]
json.dump({"mu": mu.tolist(), "C": C.tolist(), "idx": sel,
           "p_ours": [float(p1[i]) for i in sel]},
          open("referee_cases.json", "w"))
print(f"wrote {len(sel)} cases; bands down to {min(p1[sel]):.1e}")
