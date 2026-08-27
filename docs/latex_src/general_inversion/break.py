"""Adversarial battery for the lattice race: push each failure axis
until something gives. Behind the paper's "Stress boundaries" paragraph.

Referee discipline learned the hard way, baked in below:
  - adaptive quadrature must run pure-relative (epsabs=0) or it
    self-terminates on the absolute criterion at tiny p and returns
    junk; entries are trusted only when the reported error is under
    1e-3 of the value;
  - both the forward race and the inversion are min-wins; no sign flip
    in the round trip;
  - (and in referee.R: TruncatedNormal masks mvtnorm::pmvnorm.)

Sections:
  A  vanishing noise            C  deep tails (breaking depth)
  B  ability spread             D  Gumbel base = softmax, exactly
  E  duplicates and continuity  F  inversion round-trip, hostile targets
  G  Jacobian vs central FD     H  factor strength (the real boundary)

Run: python break.py
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import norm, qmc
from winning.factor.core import (abilities_from_probabilities_factor,
                                 hermite_nodes, jacobian_vector_product,
                                 qmc_nodes, win_probabilities_factor)
from winning.factor.races import race_probabilities

rng = np.random.default_rng(99)


def indep_exact(mu, sd, i, depth=20):
    def f(x):
        v = norm.pdf(x, mu[i], sd[i])
        for j in range(len(mu)):
            if j != i:
                v *= norm.sf(x, mu[j], sd[j])
        return v
    return quad(f, mu[i] - depth * sd[i], mu[i] + depth * sd[i],
                limit=800, epsabs=0.0, epsrel=1e-12)


def trusted_maxrel(mu, sd, points):
    """Max relative error over entries where the referee is reliable."""
    p = race_probabilities(mu, D=sd ** 2, points=points)
    vals = [indep_exact(mu, sd, i) for i in range(len(mu))]
    tot = sum(v for v, _ in vals)
    worst, deepest, n_ok = 0.0, 1.0, 0
    for i, (pe, err) in enumerate(vals):
        if err < 1e-3 * pe:
            n_ok += 1
            deepest = min(deepest, pe / tot)
            worst = max(worst, abs(p[i] - pe / tot) / (pe / tot))
    return worst, deepest, n_ok


print("== A: vanishing noise (n=8, gaps 0.3, noise s) ==")
mu = np.arange(8) * 0.3
for s in (1.0, 0.1, 0.01):
    w, d, k = trusted_maxrel(mu, np.full(8, s), 2049)
    print(f"  s={s:5g}  trusted {k}/8, deepest p {d:.1e}, maxrel {w:.2e}")

print("== B: ability spread (n=8, sd=1) ==")
for S in (2, 20, 60):
    w, d, k = trusted_maxrel(np.linspace(0, 1, 8) * S, np.ones(8), 2049)
    print(f"  spread={S:3d}  trusted {k}/8, deepest p {d:.1e}, "
          f"maxrel {w:.2e}")

print("== C: deep tails (laggard k sd back; breaking depth) ==")
for k in (8, 12, 16, 25):
    w, d, n_ok = trusted_maxrel(
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, float(k)]), np.ones(6), 2049)
    print(f"  k={k:2d}  deepest p {d:.1e}, maxrel {w:.2e}")

print("== D: Gumbel base vs softmax ==")
for n in (10, 1000):
    m = rng.normal(size=n) * 1.5
    p = race_probabilities(m, base="gumbel", points=4001)
    c = np.pi / np.sqrt(6.0)
    soft = np.exp(-c * m); soft /= soft.sum()
    print(f"  n={n:5d}  maxrel {np.max(np.abs(p - soft) / soft):.2e}")

print("== E: duplicates and continuity ==")
mu4, D4 = np.array([0.0, 0.0, 1.0, 2.0]), np.ones(4)
p = race_probabilities(mu4, D=D4, points=1025)
print(f"  exact twins split: |p1-p2| = {abs(p[0] - p[1]):.2e}")
for eps in (1e-3, 1e-9):
    p2 = race_probabilities(mu4 + np.array([0, eps, 0, 0]), D=D4,
                            points=1025)
    print(f"  eps={eps:6.0e}: |p1-p2| = {abs(p2[0] - p2[1]):.2e}")

print("== F: inversion round-trip, hostile targets (min-wins, no flip) ==")
F2, W2 = hermite_nodes(2, 7)
for name, p_t in (
        ("one favorite", np.array([0.97, 0.01, 0.01, 0.005, 0.004, 0.001])),
        ("deep tail",    np.array([0.6, 0.3, 0.09, 0.009, 0.0009, 1e-10])),
        ("near ties",    np.array([1 / 6 + 1e-9, 1 / 6 - 1e-9, 1 / 6,
                                   1 / 6, 1 / 6, 1 / 6]))):
    p_t = p_t / p_t.sum(); n = len(p_t)
    V = rng.normal(size=(n, 2)) * 0.4
    D = 0.5 + rng.random(n)
    m = abilities_from_probabilities_factor(p_t, V, D, F2, W2, points=513,
                                            tol=1e-9, n_iter=300)
    p_b = race_probabilities(m, V=V, D=D, F=F2, W=W2, points=2049)
    print(f"  {name:12s} max|log p_hat/p| = "
          f"{np.max(np.abs(np.log(np.maximum(p_b, 1e-300)) - np.log(p_t))):.2e}")

print("== G: Jacobian vs central FD (same frozen points) ==")
n = 8
m = rng.normal(size=n); V = rng.normal(size=(n, 2)) * 0.6
D = 0.5 + rng.random(n)
h = rng.normal(size=n); h -= h.mean()
for pts in (129, 1001):
    jv = jacobian_vector_product(m, V, D, F2, W2, h, points=pts,
                                 form="grid")
    eps = 1e-5
    fd = (win_probabilities_factor(m + eps * h, V, D, F2, W2, points=pts)
          - win_probabilities_factor(m - eps * h, V, D, F2, W2,
                                     points=pts)) / (2 * eps)
    print(f"  points={pts:5d}  max|jvp-fd| = {np.max(np.abs(jv - fd)):.2e}"
          "   (coarse lattices add grid-motion terms FD sees and the"
          " frozen-grid JVP does not)" if pts == 129 else
          f"  points={pts:5d}  max|jvp-fd| = {np.max(np.abs(jv - fd)):.2e}")

print("== H: factor strength (the real boundary) ==")
n = 20
mu0 = rng.normal(size=n)
V = rng.normal(size=(n, 2)) * 4.0
D = 0.5 + rng.random(n)
L = np.linalg.cholesky(V @ V.T + np.diag(D))
z = ndtri(np.clip(qmc.Sobol(n, scramble=True, seed=1)
                  .random_base2(22), 1e-12, 1 - 1e-12)).T
ref = np.bincount(np.argmin(mu0[:, None] + L @ z, axis=0),
                  minlength=n) / z.shape[1]
for label, (F, W) in (("GH-7", hermite_nodes(2, 7)),
                      ("GH-25", hermite_nodes(2, 25)),
                      ("Sobol 2^13", qmc_nodes(2, 13))):
    p = race_probabilities(mu0, V=V, D=D, F=F, W=W, points=513)
    print(f"  loading scale 4, {label:10s} nodes={len(F):5d}  "
          f"TV {0.5 * np.abs(p - ref).sum():.2e}")
print("  verdict: heavy correlation wants QMC factor nodes, not deeper"
      " polynomial rules")
