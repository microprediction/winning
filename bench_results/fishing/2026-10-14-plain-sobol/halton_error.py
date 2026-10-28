"""Quadrature error of the sharp-branch clouds at fitted Fishing thetas:
python's plain Sobol 2^10 (seed 0), julia's plain Halton 2^10 (offset
20, as MultinomialProbit._halton_normal) and its symmetrised orbit,
against the Gauss-Hermite 151^3 truth. Usage:
  PYTHONPATH=<tree> python halton_error.py py_main.json jl_branch.json py_branch.json"""
import json
import sys

import numpy as np
from scipy.special import ndtri

import winning.likelihood as L
from winning.mnprobit import MNProbit


def halton(r, n):
    primes = [2, 3, 5, 7, 11][:r]
    F = np.zeros((n, r))
    for c, b in enumerate(primes):
        for row in range(n):
            i, f, h = row + 1 + 20, 1.0 / b, 0.0
            while i > 0:
                h += f * (i % b)
                i //= b
                f /= b
            F[row, c] = ndtri(min(max(h, 1e-12), 1 - 1e-12))
    return F, np.full(n, 1.0 / n)


def sym(F):
    out = []
    for pm in ((0, 1), (1, 0)):
        for sg in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            G = F.copy()
            G[:, :2] = F[:, list(pm)] * np.array(sg)
            out.append(G)
    G = np.vstack(out)
    return G, np.full(len(G), 1.0 / len(G))


base = json.load(open(sys.argv[1]))
m = MNProbit(np.array(base["X"]), np.array(base["choice"]),
             intercepts=True, r=2)
pts = [("main python", json.load(open(sys.argv[1]))["python"]["theta"]),
       ("branch julia", json.load(open(sys.argv[2]))["theta_exact"]),
       ("branch python", json.load(open(sys.argv[3]))["python"]["theta"])]
clouds = {"GH 151 (truth)": L.nodes_for_likelihood(2, 151, 151, 0.0),
          "plain Sobol 2^10": L.sobol_normal(2, 1, 1024, 0),
          "plain Halton 2^10": halton(3, 1024),
          "symmetrised Halton 2^10": sym(halton(3, 128)[0])}
for name, th in pts:
    beta, V = m._unpack(np.array(th))
    mu = m.X @ beta
    row = []
    for cn, (F, W) in clouds.items():
        orig = L.nodes_for_likelihood
        L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
        try:
            row.append(f"{cn} {L.choice_loglik_and_score(mu, V, m.choice)[0]:.2f}")
        finally:
            L.nodes_for_likelihood = orig
    print(f"{name}: " + "; ".join(row))
