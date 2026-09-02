"""Cavity-Shapley for extremal portfolios: the attribution twin.

Candidates carry value Y_i = (X_i - b)_+ with X factor-Gaussian
(rank 1 here), and a coalition is worth v(A) = E max_{i in A} Y_i --
batch expected improvement when b is the incumbent. With independent
availabilities a the multilinear extension is

  V(a) = E_z int_0^inf [1 - prod_j (1 - a_j Fbar_{j|z}(y))] dy,

whose partial derivatives are cavities of the availability field
H = prod_j (1 - a_j Fbar_j). Owen's diagonal then gives every
Shapley value from ONE extra one-dimensional quadrature:

  phi_i = E_z int_y int_0^1 Fbar_i prod_{j!=i} (1 - t Fbar_j) dt dy,

all i sharing the field at each (z, y, t) by log-domain subtraction.
Deletion values and Banzhaf (a = 1/2) come from the same pass.

Checks: exhaustive coalition enumeration at n = 5 (Shapley from
2^5 coalition values vs the formula; efficiency sum phi = v(grand));
Monte Carlo for v(grand) and one deletion; duplicate symmetry.

The decisive experiment (n = 30): a cluster of near-duplicate
strong candidates, one rare-upside specialist (weak mean, fat right
tail, independent), and solid independents. Probability of
optimality, deletion value, and Shapley value rank them radically
differently -- each answers a different operational question.

The optimizer pairing: greedy selection of a size-6 group on the
same objective (whom to ENTER), then Shapley attribution within the
selected group (whom to PAY) -- near-duplicates that survive
selection split their pay.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                ".."))
from winning.factor import race_probabilities  # noqa: E402

QZ = 41
LY = 601
QT = 32
B_INCUMBENT = 0.0


def gh():
    z, w = np.polynomial.hermite_e.hermegauss(QZ)
    return z, w / w.sum()


def fields(mu, v, d, ymax=None):
    """Fbar_{i|z}(y) on a y-grid, plus node weights; Y = (X - b)_+."""
    z, wz = gh()
    sd = np.sqrt(d)
    if ymax is None:
        ymax = float((mu + np.abs(v) * 6 + 6 * sd - B_INCUMBENT).max())
        ymax = max(ymax, 1e-6)
    y = np.linspace(0.0, ymax, LY)
    # Fbar[q, i, l] = P(X_i > b + y_l | z_q)
    m = mu[None, :] + np.outer(z, v)
    tail = norm.sf((B_INCUMBENT + y[None, None, :] - m[:, :, None])
                   / sd[None, :, None])
    return y, tail, wz


def extremal_value(tail, y, wz, member=None):
    """v(A) = E max over A of Y, A given by boolean mask (default all)."""
    if member is None:
        member = np.ones(tail.shape[1], bool)
    log1m = np.log1p(-np.clip(tail[:, member, :], 0.0, 1 - 1e-15))
    miss = np.exp(log1m.sum(axis=1))          # (QZ, LY)
    return float(wz @ np.trapezoid(1.0 - miss, y, axis=1))


def deletion_values(tail, y, wz):
    """v(N) - v(N \\ i) for every i, one shared field."""
    log1m = np.log1p(-np.clip(tail, 0.0, 1 - 1e-15))
    tot = log1m.sum(axis=1)                   # (QZ, LY)
    miss_all = np.exp(tot)
    v_all = wz @ np.trapezoid(1.0 - miss_all, y, axis=1)
    # miss without i = exp(tot - log1m_i); v drops by E int (miss_wo_i
    # - miss_all)
    miss_wo = np.exp(tot[:, None, :] - log1m)  # (QZ, N, LY)
    delta = np.trapezoid(miss_wo - miss_all[:, None, :], y, axis=2)
    return float(v_all), np.asarray(wz @ delta)


def shapley_banzhaf(tail, y, wz):
    """All Shapley and Banzhaf values by Owen's diagonal, shared field
    per (z, y, t)."""
    tg, tw = np.polynomial.legendre.leggauss(QT)
    tg = 0.5 * (tg + 1.0)
    tw = 0.5 * tw
    n = tail.shape[1]
    phi = np.zeros(n)
    bz = np.zeros(n)
    for t, w_t in zip(tg, tw):
        log1m = np.log1p(-np.clip(t * tail, 0.0, 1 - 1e-15))
        tot = log1m.sum(axis=1)
        cav = np.exp(tot[:, None, :] - log1m)      # prod_{j!=i}
        integ = np.trapezoid(tail * cav, y, axis=2)  # (QZ, N)
        phi += w_t * (wz @ integ)
    log1m = np.log1p(-np.clip(0.5 * tail, 0.0, 1 - 1e-15))
    tot = log1m.sum(axis=1)
    cav = np.exp(tot[:, None, :] - log1m)
    bz = wz @ np.trapezoid(tail * cav, y, axis=2)
    return phi, np.asarray(bz)


def enumerate_shapley(tail, y, wz, n):
    """Exact Shapley from 2^n coalition values."""
    from math import factorial
    vals = {}
    for r in range(n + 1):
        for A in itertools.combinations(range(n), r):
            mask = np.zeros(n, bool)
            mask[list(A)] = True
            vals[A] = extremal_value(tail, y, wz, mask) if A else 0.0
    phi = np.zeros(n)
    for i in range(n):
        for A in vals:
            if i in A:
                continue
            Ai = tuple(sorted(A + (i,)))
            k = len(A)
            wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
            phi[i] += wgt * (vals[Ai] - vals[A])
    return phi, vals[tuple(range(n))]


if __name__ == "__main__":
    results = {}
    rng = np.random.default_rng(2)

    # --- verification at n=5 ---
    mu5 = rng.normal(0.3, 0.5, 5)
    v5 = rng.normal(0, 0.5, 5)
    d5 = 0.3 + rng.random(5)
    y, tail, wz = fields(mu5, v5, d5)
    phi_fast, bz = shapley_banzhaf(tail, y, wz)
    phi_enum, v_grand_enum = enumerate_shapley(tail, y, wz, 5)
    v_all, dele = deletion_values(tail, y, wz)
    err_shap = np.abs(phi_fast - phi_enum).max()
    err_eff = abs(phi_fast.sum() - v_all)
    # MC referee
    M = 2_000_000
    z = rng.normal(size=(M, 1))
    X = mu5 + z * v5 + rng.normal(size=(M, 5)) * np.sqrt(d5)
    Ymc = np.maximum(X - B_INCUMBENT, 0.0)
    v_mc = Ymc.max(1).mean()
    del_mc = v_mc - np.delete(Ymc, 2, axis=1).max(1).mean()
    print(f"[n=5 checks] shapley formula vs enumeration max|err| "
          f"{err_shap:.2e}; efficiency |sum phi - v| {err_eff:.2e}; "
          f"v(N) {v_all:.5f} vs MC {v_mc:.5f}; "
          f"deletion_2 {dele[2]:.5f} vs MC {del_mc:.5f}")
    results["n5"] = dict(err_shapley=float(err_shap),
                         err_efficiency=float(err_eff),
                         v=float(v_all), v_mc=float(v_mc),
                         del2=float(dele[2]), del2_mc=float(del_mc))

    # --- decisive experiment: duplicates vs specialist, n=30 ---
    n = 30
    mu = np.full(n, 0.0)
    v = np.zeros(n)
    d = np.full(n, 0.4)
    mu[:5] = 0.9          # five near-duplicate strong candidates
    v[:5] = 0.9
    d[:5] = 0.05
    mu[5] = -0.3          # the rare-upside specialist
    v[5] = 0.0
    d[5] = 6.0
    mu[6:] = rng.normal(0.1, 0.3, n - 6)   # solid independents
    v[6:] = rng.normal(0, 0.3, n - 6)
    y, tail, wz = fields(mu, v, d)
    v_all, dele = deletion_values(tail, y, wz)
    phi, bz = shapley_banzhaf(tail, y, wz)
    pom = race_probabilities(-mu, V=-v.reshape(-1, 1), D=d)
    def top3(x):
        o = np.argsort(x)[::-1][:3]
        return [[int(i), float(x[i])] for i in o]
    print(f"[n=30] v(grand) {v_all:.4f}  sum phi {phi.sum():.4f}")
    print("  top-3 by PoM      :", top3(pom))
    print("  top-3 by deletion :", top3(dele))
    print("  top-3 by Shapley  :", top3(phi))
    print(f"  duplicate block (0-4): pom {pom[:5].sum():.3f} total, "
          f"deletion {dele[:5].mean():.4f} each, "
          f"shapley {phi[:5].mean():.4f} each")
    print(f"  specialist (5): pom {pom[5]:.4f}, deletion "
          f"{dele[5]:.4f}, shapley {phi[5]:.4f}")
    results["n30"] = dict(v=float(v_all),
                          pom_top3=top3(pom), del_top3=top3(dele),
                          shap_top3=top3(phi),
                          specialist=dict(pom=float(pom[5]),
                                          deletion=float(dele[5]),
                                          shapley=float(phi[5])),
                          dup_each=dict(pom=float(pom[:5].mean()),
                                        deletion=float(dele[:5].mean()),
                                        shapley=float(phi[:5].mean())))

    # --- the optimizer pairing: enter greedily, pay by Shapley ---
    y30, tail30, wz30 = fields(np.r_[np.full(5, 0.9), [-0.3],
                                     rng.normal(0.1, 0.3, 24)],
                               np.r_[np.full(5, 0.9), [0.0],
                                     rng.normal(0, 0.3, 24)],
                               np.r_[np.full(5, 0.05), [6.0],
                                     np.full(24, 0.4)])
    chosen = []
    mask = np.zeros(30, bool)
    for _ in range(6):
        best = None
        for i in range(30):
            if mask[i]:
                continue
            mask[i] = True
            val = extremal_value(tail30, y30, wz30, mask)
            mask[i] = False
            if best is None or val > best[0]:
                best = (val, i)
        mask[best[1]] = True
        chosen.append(best[1])
    phi_in, _ = shapley_banzhaf(tail30[:, mask, :], y30, wz30)
    pay = {int(c): float(p) for c, p in
           zip(np.where(mask)[0], phi_in)}
    print(f"[pairing] greedy group of 6: {sorted(pay)}  "
          f"pay: " + " ".join(f"{k}:{pv:.3f}"
                              for k, pv in sorted(pay.items())))
    results["pairing"] = dict(group=sorted(pay), pay=pay,
                              group_value=float(best[0]))

    # --- scale timing ---
    n = 2000
    mu = rng.normal(0, 0.5, n)
    v = rng.normal(0, 0.4, n)
    d = 0.3 + rng.random(n)
    y, tail, wz = fields(mu, v, d)
    t0 = time.time()
    v_all, dele = deletion_values(tail, y, wz)
    t_del = time.time() - t0
    t0 = time.time()
    phi, bz = shapley_banzhaf(tail, y, wz)
    t_shap = time.time() - t0
    print(f"[n=2000] deletions {t_del:.1f}s, all Shapley {t_shap:.1f}s"
          f"  (efficiency err {abs(phi.sum() - v_all):.2e})")
    results["n2000"] = dict(del_seconds=t_del, shap_seconds=t_shap,
                            eff_err=float(abs(phi.sum() - v_all)))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
