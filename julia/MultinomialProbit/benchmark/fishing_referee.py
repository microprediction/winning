"""Shared referee: python's stabilized Sobol evaluation at every
fitted theta (python, julia exact, julia ghk)."""
import json
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import winning.likelihood as L
from winning.likelihood import choice_loglik_and_score
from winning.mnprobit import MNProbit

inp = json.load(open(sys.argv[1]))
jl = json.load(open(sys.argv[2]))
X = np.array(inp["X"])
choice = np.array(inp["choice"])
m = MNProbit(X, choice, intercepts=True, r=2)

def referee(theta):
    beta, V = m._unpack(np.array(theta))
    mu = m.X @ beta
    vals = []
    for seed in (101, 102):
        u = qmc.Sobol(3, scramble=True, seed=seed).random(2 ** 15)
        F = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
        W = np.full(len(F), 1.0 / len(F))
        orig = L.nodes_for_likelihood
        L.nodes_for_likelihood = lambda r, Qf=7, Qz=7, sharp=0.0: (F, W)
        try:
            vals.append(choice_loglik_and_score(mu, V, choice)[0])
        finally:
            L.nodes_for_likelihood = orig
    return float(np.mean(vals)), float(abs(vals[0] - vals[1]) / 2)

for name, theta in (("python exact", inp["python"]["theta"]),
                    ("julia exact", jl["theta_exact"]),
                    ("julia ghk", jl["theta_ghk"])):
    ll, se = referee(theta)
    print(f"{name:14s} referee logLik {ll:.2f} +- {se:.2f}")
