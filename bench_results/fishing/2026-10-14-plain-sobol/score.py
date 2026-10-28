"""Score fitted Fishing thetas: sharpness (pairwise-safe bound and the
smooth blend statistic) and the Gauss-Hermite 151^3 truth, as
../2026-10-13-benchfix/cloud_error.py. Usage:
  PYTHONPATH=<tree> python score.py py.json jl.json"""
import json
import sys

import numpy as np

import winning.likelihood as L
from winning.mnprobit import MNProbit

inp = json.load(open(sys.argv[1]))
jl = json.load(open(sys.argv[2]))
m = MNProbit(np.array(inp["X"]), np.array(inp["choice"]),
             intercepts=True, r=2)
F, W = L.nodes_for_likelihood(2, 151, 151, 0.0)
for name, theta in (("python exact", inp["python"]["theta"]),
                    ("julia exact", jl["theta_exact"])):
    beta, V = m._unpack(np.array(theta))
    mu = m.X @ beta
    orig = L.nodes_for_likelihood
    L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
    try:
        truth = L.choice_loglik_and_score(mu, V, m.choice)[0]
    finally:
        L.nodes_for_likelihood = orig
    sb = L.sharpness_bound(V, np.ones(m.J))
    ss = L._smooth_sharpness(V, np.ones(m.J))[0]
    print(f"{name:14s} sharpness_bound {sb:.2f} smooth {ss:.2f} "
          f"truth (GH 151) {truth:.2f}")
