"""Kernel stratification follow-up to run_ensembles4 (whose kernel cases
crashed on an rng-binding bug): RBF vs Matern-3/2 x length scale x
promoted rank, 20 seeds, same referee, same schema, appended to the same
CSV. The MC referee is shared across the m arms (same C)."""
import csv

import numpy as np

from winning.factor.core import fit_covariance
from winning.factor.races import race_probabilities

import importlib.util
spec = importlib.util.spec_from_file_location("b", "run_large_n2.py")
b = importlib.util.module_from_spec(spec)
import sys; sys.modules["b"] = b
src = open("run_large_n2.py").read()
exec(src[:src.index("rng = np.random.default_rng(21)")], b.__dict__)

n, M, SEEDS = 300, 1_000_000, 20
mu = np.sort(np.random.default_rng(5).normal(size=n)) * 1.2

def kernel_C(kind, ls, seed):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    r = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) / ls
    if kind == "rbf":
        C = np.exp(-0.5 * r * r)
    else:
        C = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    np.fill_diagonal(C, 1.0)
    w_, U_ = np.linalg.eigh(C)
    C = (U_ * np.maximum(w_, 1e-8)) @ U_.T
    dd = np.sqrt(np.diag(C))
    return C / np.outer(dd, dd)

outf = open("results_ensembles4.csv", "a", buffering=1)
out = csv.writer(outf)
for kind in ("rbf", "matern32"):
    for ls in (0.08, 0.2, 0.4):
        rows = {5: [], 12: [], "5n14": []}
        for seed in range(SEEDS):
            C = kernel_C(kind, ls, 1000 + seed)
            pmc, counts = b.big_mc(mu, C, M, np.random.default_rng(seed))
            seen = counts >= 25
            for m, nl2 in ((5, 11), (12, 11), ("5n14", 14)):
                V, D, F, W = fit_covariance(C, k=3, m=5 if m == "5n14" else m,
                                            blocks=20, nodes_log2=nl2)
                p1 = race_probabilities(mu, V=V, D=D, F=F, W=W, points=257)
                err = np.abs(p1 - pmc)
                tv = 0.5 * err.sum()
                li = err[seen].max(); md = np.median(err[seen])
                rows[m].append((tv, li, md))
                out.writerow([f"kernel-{kind}-ls{ls}-m{m}", "projected",
                              seed, tv, li, md])
        for m in (5, 12, "5n14"):
            A = np.array(rows[m])
            print(f"kernel-{kind}-ls{ls}-m{str(m):<5s} projected "
                  f"med(TV) {np.median(A[:,0]):.2e} "
                  f"q90 {np.quantile(A[:,0],0.9):.2e} "
                  f"worst {A[:,0].max():.2e} "
                  f"| med(med abs) {np.median(A[:,2]):.1e}", flush=True)
