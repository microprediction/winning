"""Quadrature error of the computed POM vector on the paper's posteriors.

For posteriors of the kind exp1_stopping and exp2_bandit pass through,
compare race_probabilities at its defaults with (a) a lattice eight
times finer, (b) 2^15 scrambled-Sobol factor nodes in place of the
default rule, and (c) four million Monte Carlo argmax draws. Also the
rank-one three-alternative example with an analytic answer, where the
default rule is known to be off by TV 0.0015.

Reports sup-norm and total-variation differences. Run from the repo
root; thread caps are set before numpy is imported.
"""
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "3")

import importlib.util                                   # noqa: E402
import numpy as np                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from winning.factor import race_probabilities           # noqa: E402
from winning.factor.core import qmc_nodes               # noqa: E402


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = load("rb", "../exp2_bandit/run_bandit.py")
rs = load("rs", "../exp1_stopping/run_stopping.py")
RNG = np.random.default_rng(20261006)
M_MC = 4_000_000


def mc_argmax(m, W, d, M=M_MC, chunk=500_000):
    tot = np.zeros(len(m))
    for _ in range(M // chunk):
        dr = (m[None, :] + RNG.normal(size=(chunk, W.shape[1])) @ W.T
              + RNG.normal(size=(chunk, len(m))) * np.sqrt(d))
        tot += np.bincount(dr.argmax(1), minlength=len(m))
    return tot / (chunk * (M // chunk))


def compare(label, m, W, d, truth=None):
    t0 = time.time()
    p = race_probabilities(-m, V=-W, D=d)
    ms = 1000 * (time.time() - t0)
    fine = race_probabilities(-m, V=-W, D=d, points=2049)
    F, Wq = qmc_nodes(W.shape[1], m=15)
    qmc = race_probabilities(-m, V=-W, D=d, F=F, W=Wq)
    ref = truth if truth is not None else mc_argmax(m, W, d)
    se = float(np.sqrt(ref.max() * (1 - ref.max()) / M_MC)) if truth is None else 0.0
    row = dict(label=label, ms=ms, max_p=float(p.max()),
               sup_lattice=float(np.abs(p - fine).max()),
               tv_lattice=float(0.5 * np.abs(p - fine).sum()),
               sup_nodes=float(np.abs(p - qmc).max()),
               tv_nodes=float(0.5 * np.abs(p - qmc).sum()),
               sup_ref=float(np.abs(p - ref).max()),
               tv_ref=float(0.5 * np.abs(p - ref).sum()),
               ref="analytic" if truth is not None else "mc", mc_se=se)
    print(f"{label:28s} ms={ms:6.1f} maxp={row['max_p']:.3f} "
          f"lattice sup={row['sup_lattice']:.1e} nodes sup={row['sup_nodes']:.1e} "
          f"tv={row['tv_nodes']:.1e} | ref({row['ref']}) sup={row['sup_ref']:.1e} "
          f"tv={row['tv_ref']:.1e} se={se:.1e}")
    return row


rows = []
# the three-alternative rank-one example with an analytic answer
V = np.array([0.0, 1.0, 3.0]); D = np.full(3, 0.1); S = np.outer(V, V) + np.diag(D)
ex = []
for i in range(3):
    j, k = [x for x in range(3) if x != i]
    r = (S[i, i] - S[i, j] - S[i, k] + S[j, k]) / np.sqrt(
        (S[i, i] + S[j, j] - 2 * S[i, j]) * (S[i, i] + S[k, k] - 2 * S[i, k]))
    ex.append(0.25 + np.arcsin(r) / (2 * np.pi))
rows.append(compare("rank-one (0,1,3), d=0.1", np.zeros(3), V.reshape(3, 1), D, np.array(ex)))

# bandit posteriors, both regimes, early and late
for kind, c in [("independent", 0.0), ("clusters", 0.75)]:
    V, d0 = rb.make_prior(kind, c)
    for t in (50, 200):
        theta = V @ RNG.normal(size=rb.RHO) + RNG.normal(size=rb.N) * np.sqrt(d0)
        counts = RNG.multinomial(t, np.full(rb.N, 1 / rb.N)).astype(float)
        sums = counts * theta + RNG.normal(size=rb.N) * np.sqrt(counts) * rb.SIGMA_N
        m, W, d = rb.posterior(V, d0, counts, sums)
        rows.append(compare(f"bandit {kind} t={t}", m, W, d))

# stopping posteriors, three regimes, early and near the exact rule's mean stop
for kind, c in [("independent", 0.0), ("aligned", 0.6), ("opposed", 0.6)]:
    V, sigma = rs.make_config(kind, c)
    theta = RNG.normal(size=rs.K) * rs.S0
    for n in (20, 70):
        Y = theta[None, :] + RNG.normal(size=(n, rs.RHO)) @ V.T + RNG.normal(size=(n, rs.K)) * sigma
        m, W, d, _ = rs.posterior_factor_form(n, Y.mean(0), V, sigma)
        rows.append(compare(f"stopping {kind} n={n}", m, W, d))

# raw results first, to an append-only log that no later run replaces
import datetime
import subprocess
try:
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                   cwd=HERE, text=True).strip()
except Exception:
    head = "unknown"
with open(os.path.join(HERE, "results.jsonl"), "a") as fh:
    fh.write(json.dumps({"when": datetime.datetime.now(datetime.timezone.utc)
                         .isoformat(timespec="seconds"),
                         "head": head, "rows": rows}) + "\n")
# then the summary the paper reads, which is the latest run
json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), indent=1)
