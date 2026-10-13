"""Quadrature error of the sharp-branch node clouds at the fitted Fishing
points (#717). Truth = Gauss-Hermite 151^3 (converged to 1e-3 nats).
usage: PYTHONPATH=<checkout> python cloud_error.py py.json thetas.json"""
import json
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import winning.likelihood as L
from winning.mnprobit import MNProbit

d = json.load(open(sys.argv[1]))
m = MNProbit(np.array(d["X"]), np.array(d["choice"]), intercepts=True, r=2)
thetas = json.load(open(sys.argv[2]))


def plain(n, s):
    u = qmc.Sobol(3, scramble=True, seed=s).random(n)
    return ndtri(np.clip(u, 1e-12, 1 - 1e-12)), np.full(n, 1.0 / n)


rules = {"plain Sobol 2^10 (pre-#605)": lambda s: plain(1024, s),
         "symmetrised 2^10 (#605, default)":
             lambda s: L.symmetric_sobol_normal(2, 1, 1024, s),
         "symmetrised 2^12":
             lambda s: L.symmetric_sobol_normal(2, 1, 4096, s)}
for name, th in thetas.items():
    b, V = m._unpack(np.array(th))
    mu = m.X @ b
    Vc = V - V.mean(axis=0)
    F, W = L.nodes_for_likelihood(2, 151, 151, 0.0)
    truth = L._choice_terms(mu, Vc, m.choice, np.ones(4), F, W,
                            score=False)[0].sum()
    print(f"{name}: sharpness {L.sharpness_bound(V):.2f}, "
          f"truth (GH 151) {truth:.2f}, objective "
          f"{-m._negloglik_grad(np.array(th))[0]:.2f}")
    for rn, rule in rules.items():
        e = np.array([L._choice_terms(mu, Vc, m.choice, np.ones(4),
                                      *rule(s), tilt=False,
                                      score=False)[0].sum()
                      for s in range(16)]) - truth
        print(f"  {rn:34s} bias {e.mean():+6.2f}  rms {np.sqrt((e ** 2).mean()):5.2f}"
              f"  seed 0 {e[0]:+6.2f}")
