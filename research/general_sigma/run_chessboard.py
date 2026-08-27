"""Kernel Sigma by chessboard conditioning: the boundary's first answer.

Split the point cloud into a conditioning net A (alternate points in a
space-filling order) and the rest B. Sample X_A; by the screening
effect Sigma_{B|A} is nearly diagonal, so conditional on the draw the
B-race is (approximately) independent and one lattice pass prices it,
truncated at min(X_A); A-members' conditional win probabilities are
field-survival evaluations. Rao-Blackwellized, cheap draws, smooth.

Scored vs a 2^20 Sobol referee; bias predicted by the off-diagonal mass
of the conditional correlation.
"""
import time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
from winning.factor.races import race_probabilities
from randomcov.corrgens.kernelcorr import kernel_corr

rng = np.random.default_rng(11)
n = 300
# rebuild the same kernel ensemble draw with its point cloud
rngk = np.random.default_rng(11)
X = rngk.random((n, 2))
ls = 0.1 + 0.4 * rngk.random()
ls = 0.08
r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) / ls
C = (1.0 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)   # matern 3/2: full rank
np.fill_diagonal(C, 1.0)
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2

# chessboard split along a space-filling order
order = np.argsort(X[:, 0] + X[:, 1])
A = order[::2]
B = order[1::2]
Caa = C[np.ix_(A, A)] + 1e-8 * np.eye(len(A))
Cba = C[np.ix_(B, A)]
K = Cba @ np.linalg.inv(Caa)
Sbb = C[np.ix_(B, B)] - K @ Cba.T
dS = np.sqrt(np.maximum(np.diag(Sbb), 1e-10))
Rcond = Sbb / np.outer(dS, dS)
keep = np.diag(Sbb) > 1e-6
offv = np.abs(Rcond[np.ix_(keep, keep)] - np.eye(keep.sum()))
print(f"cond variances: median {np.median(np.diag(Sbb)):.3f}; "
      f"cond corr off-diag median {np.median(offv[offv>0]):.3f} "
      f"max {offv.max():.3f}", flush=True)
La = np.linalg.cholesky(Caa)

def estimate(M, seed):
    rr = np.random.default_rng(seed)
    p = np.zeros(n)
    Db = np.maximum(np.diag(Sbb), 1e-10)
    for m in range(M):
        xa = mu[A] + La @ rr.standard_normal(len(A))
        mb = mu[B] + K @ (xa - mu[A])
        amin = xa.min()
        # conditional B-race truncated at amin: append a deterministic
        # runner at amin with tiny variance; its win prob absorbs P(A wins)
        mu_ext = np.concatenate([mb, [amin]])
        D_ext = np.concatenate([Db, [1e-6]])
        pe = race_probabilities(mu_ext, D=D_ext, points=257)
        p[B] += pe[:-1]
        # distribute the A-side mass to the argmin of the draw
        p[A[np.argmin(xa)]] += pe[-1]
    return p / M

# referee
L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=5).random_base2(20),
                  1e-12, 1 - 1e-12)).T
ref = np.bincount(np.argmin(mu[:, None] + L @ z, axis=0), minlength=n) / z.shape[1]

for M in (256, 1024):
    t0 = time.time()
    reps = [estimate(M, 100 + k) for k in range(6)]
    t_est = (time.time() - t0) / 6
    Aarr = np.array(reps)
    res = ref > 1e-4
    sd = Aarr.std(axis=0)[res].mean()
    bias = np.abs(Aarr.mean(axis=0) - ref)[res].max()
    med = np.median(np.abs(Aarr.mean(axis=0) - ref)[res])
    # plain MC at equal wall time
    t0 = time.time()
    done, counts = 0, np.zeros(n)
    while time.time() - t0 < t_est:
        zz = np.random.default_rng(int(1e6 + done)).standard_normal((n, 20000))
        counts += np.bincount(np.argmin(mu[:, None] + L @ zz, axis=0), minlength=n)
        done += 20000
    pmc = counts / done
    print(f"M={M:5d}  t {t_est:5.1f}s  sd {sd:.2e}  max err {bias:.2e}  "
          f"med err {med:.2e}  | plain MC same time ({done//1000}k draws): "
          f"max err {np.abs(pmc - ref)[res].max():.2e}  "
          f"med err {np.median(np.abs(pmc - ref)[res]):.2e}", flush=True)
