"""LITE against the exact answer, on grammar-form covariance.

LITE (arXiv:2501.13535) reports accuracy as total variation to TS-MC,
an exhaustive Thompson-sampling Monte Carlo whose cost forced their
1000-dim linear-kernel reference run to 21 days. For covariances
inside the factor grammar the exact PoM is available in linear time,
so this experiment reruns the LITE comparison with the roles
reassigned: the exact vector is the ground truth, and TS-MC at a
practical budget joins F-LITE and A-LITE as an approximation to be
measured.

Ensembles (named; mu ~ N(0,1) throughout, heteroskedastic
idiosyncratic variance 0.3 + U(0,1)):
  rank1_c30 / rank1_c70   one factor, factor share 0.3 / 0.7
  rank4_c50               four factors, equal loadings scale,
                          factor share 0.5 (the linear-kernel shape
                          at modest rank)
Sizes n in {100, 1000, 10000}.

Methods: exact (pom_fast, certified against a 5e5-draw factor MC at
every cell), F-LITE and A-LITE (the numpy ports that match the JAX
originals to 1e-6; see ../qpo/README.md), and TS-MC at M = 10,000
draws, which is a generous per-cell budget at these sizes.

Metrics per cell: total variation to exact; worst relative error on
the tail (entries with exact p < 1e-3, the probabilities MC cannot
resolve: relative MC error is ~ 1/sqrt(M p)); wall-clock seconds.
"""
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "qpo"))
from pom import (pom_alite, pom_factor_mc, pom_fast, pom_flite,  # noqa
                 hermite_nodes)

ENSEMBLES = {
    "rank1_c30": (1, 0.3),
    "rank1_c70": (1, 0.7),
    "rank4_c50": (4, 0.5),
}
SIZES = (100, 1000, 10000)
MC_BUDGET = 10_000
CERT_DRAWS = 500_000


def make_instance(n, r, share, rng):
    mu = rng.normal(0.0, 1.0, n)
    u = rng.normal(size=(n, r))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    base = 0.3 + rng.random(n)
    V = u * np.sqrt(share * base)[:, None]
    d = (1.0 - share) * base
    return mu, V, d


def tail_relerr(p_hat, p_exact, cut=1e-3):
    m = (p_exact > 0) & (p_exact < cut)
    if not m.any():
        return None
    return float(np.max(np.abs(p_hat[m] - p_exact[m]) / p_exact[m]))


if __name__ == "__main__":
    results = {}
    for name, (r, share) in ENSEMBLES.items():
        nodes, weights = hermite_nodes(r, Q=15 if r <= 2 else 7)
        for n in SIZES:
            rng = np.random.default_rng(hash((name, n)) % 2 ** 31)
            mu, V, d = make_instance(n, r, share, rng)
            var = d + (V ** 2).sum(1)

            t0 = time.time()
            p_exact = pom_fast(mu, V, d, nodes=nodes, weights=weights)
            t_exact = time.time() - t0

            # certificate
            p_cert = pom_factor_mc(mu, V, d, M=CERT_DRAWS, seed=7)
            cert_tv = 0.5 * np.abs(p_exact - p_cert).sum()

            rows = {}
            t0 = time.time()
            p_f = pom_flite(mu, var)
            rows["flite"] = dict(seconds=time.time() - t0)
            t0 = time.time()
            p_a = pom_alite(mu, var)
            rows["alite"] = dict(seconds=time.time() - t0)
            t0 = time.time()
            p_mc = pom_factor_mc(mu, V, d, M=MC_BUDGET, seed=11)
            rows["tsmc_10k"] = dict(seconds=time.time() - t0)

            for key, p_hat in (("flite", p_f), ("alite", p_a),
                               ("tsmc_10k", p_mc)):
                rows[key]["tv"] = float(0.5 * np.abs(p_hat
                                                     - p_exact).sum())
                rows[key]["tail_relerr"] = tail_relerr(p_hat, p_exact)

            cell = dict(exact_seconds=t_exact,
                        certificate_tv=float(cert_tv), **{
                            k: v for k, v in rows.items()})
            results[f"{name}_n{n}"] = cell
            print(f"[{name} n={n}] exact {t_exact:.2f}s cert "
                  f"{cert_tv:.4f} | " + " | ".join(
                      f"{k} tv {rows[k]['tv']:.4f} tail "
                      f"{rows[k]['tail_relerr'] if rows[k]['tail_relerr'] is not None else float('nan'):.2f} "
                      f"({rows[k]['seconds']:.2f}s)"
                      for k in ("flite", "alite", "tsmc_10k")))

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
