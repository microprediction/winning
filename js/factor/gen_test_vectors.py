"""Generate parity test vectors for the JavaScript port from the
canonical winning.factor implementation. Seeded; commit the JSON."""

import json
import sys

import numpy as np
from scipy.special import log_ndtr

sys.path.insert(0, "../..")
from winning.factor.core import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    jacobian_vector_product,
    win_probabilities_factor,
)

rng = np.random.default_rng(2026)
N, K = 12, 2
mu = rng.normal(0, 1.0, N); mu -= mu.mean()
V = rng.normal(0, 0.4, (N, K))
D = rng.uniform(0.5, 1.5, N)
F, W = hermite_nodes(K)

p, q = win_probabilities_factor(mu, V, D, F, W, return_deletions=True)

# pairwise tie densities (min-wins): w_ij = E int f_i f_j prod_{l!=i,j} S_l
sd = np.sqrt(D)
M_all = mu[None, :] + F @ V.T
lo = M_all.min() - 8 * sd.max(); hi = M_all.max() + 8 * sd.max()
x = np.linspace(lo, hi, 501); dx = x[1] - x[0]
w = np.zeros((N, N))
for c in range(len(F)):
    z = (x[None, :] - M_all[c][:, None]) / sd[:, None]
    f = np.exp(-0.5 * z**2) / (sd[:, None] * np.sqrt(2 * np.pi))
    logS = log_ndtr(-z)
    logSfield = logS.sum(0)
    for i in range(N):
        rest = np.exp(np.clip(logSfield[None, :] - logS[i] - logS, -745, 0.0))
        w[i] += W[c] * (f[i] * f * rest).sum(1) * dx
np.fill_diagonal(w, 0.0)
# cross-check against JVP columns (min-wins J off-diagonals)
Jcols = np.column_stack([jacobian_vector_product(mu, V, D, F, W, np.eye(N)[j])
                         for j in range(N)])
off = Jcols - np.diag(np.diag(Jcols))
assert np.abs(np.abs(off) - w).max() < 5e-7, np.abs(np.abs(off) - w).max()

# calibration: recover mu from its own shares (roundtrip)
mu_hat = abilities_from_probabilities_factor(p, V, D, F, W)

# normal log-CDF reference grid for the special-function port
zgrid = np.concatenate([np.linspace(-40, -2, 40), np.linspace(-2, 8, 41)])

out = {
    "problem": {"mu": mu.tolist(), "V": V.tolist(), "D": D.tolist()},
    "hermite": {"F": F.tolist(), "W": W.tolist()},
    "expected": {
        "p": p.tolist(),
        "deletions": q.tolist(),
        "w": w.tolist(),
        "mu_hat": mu_hat.tolist(),
    },
    "logndtr": {"z": zgrid.tolist(), "v": log_ndtr(zgrid).tolist()},
}
with open("test_vectors.json", "w") as fjson:
    json.dump(out, fjson)
print("wrote test_vectors.json;",
      f"p range [{p.min():.2e},{p.max():.2e}]; roundtrip",
      f"{np.abs(mu_hat - mu).max():.2e}")
