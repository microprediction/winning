"""Skew-normal and Student-t parity vectors for the JavaScript port.
The Python side uses exact scipy survival functions through the general
engine's callable-base interface; the JS side tabulates. Seeded."""

import json
import sys

import numpy as np
from scipy.stats import norm, skewnorm, t as student_t

sys.path.insert(0, "../..")
from winning.factor.core import hermite_nodes  # noqa: E402
from winning.factor.races import race_probabilities  # noqa: E402

ALPHA, NU = 3, 4
_d = ALPHA / np.sqrt(1 + ALPHA**2)
_m = _d * np.sqrt(2 / np.pi)
_s = np.sqrt(1 - _m**2)


def skew_base(z):
    u = _m + _s * z
    S = skewnorm.sf(u, ALPHA)
    f = _s * 2 * norm.pdf(u) * norm.cdf(ALPHA * u)
    fp = _s**2 * 2 * (-u * norm.pdf(u) * norm.cdf(ALPHA * u)
                      + ALPHA * norm.pdf(u) * norm.pdf(ALPHA * u))
    return S, f, fp


def t4_base(z):
    u = np.sqrt(2.0) * z
    S = student_t.sf(u, NU)
    f = np.sqrt(2.0) * (3 / 8) * (1 + u**2 / 4) ** -2.5
    fp = 2.0 * (3 / 8) * (-2.5) * (u / 2) * (1 + u**2 / 4) ** -3.5
    return S, f, fp


# sanity: both standardized (mean 0, var 1) on a fine grid
zg = np.linspace(-40, 40, 400001)
for name, fn in [("skew", skew_base), ("t4", t4_base)]:
    _, f, _ = fn(zg)
    dz = zg[1] - zg[0]
    m0, m1 = np.trapezoid(f, dx=dz), np.trapezoid(zg * f, dx=dz)
    m2 = np.trapezoid(zg**2 * f, dx=dz)
    # t4 keeps ~1e-3 of its variance beyond |z| = 40; the mean and mass
    # checks stay tight
    assert abs(m0 - 1) < 1e-6 and abs(m1) < 1e-6 and abs(m2 - 1) < 3e-3, name

rng = np.random.default_rng(7)
N = 8
mu = rng.normal(0, 0.8, N); mu -= mu.mean()
V = rng.uniform(-0.8, 0.8, (N, 1))
D = 1.0 - V[:, 0] ** 2
F, W = hermite_nodes(1)

out = {"problem": {"mu": mu.tolist(), "V": V.tolist(), "D": D.tolist()},
       "hermite": {"F": F.tolist(), "W": W.tolist()}, "expected": {}}
for name, fn in [("skew", skew_base), ("t4", t4_base)]:
    p = race_probabilities(mu, V=V, D=D, F=F, W=W, base=fn)
    out["expected"][name] = p.tolist()
with open("test_vectors_bases.json", "w") as fjson:
    json.dump(out, fjson)
print("wrote test_vectors_bases.json")
