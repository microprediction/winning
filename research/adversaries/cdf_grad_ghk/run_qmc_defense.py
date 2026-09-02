"""The high-rank defense: QMC factor nodes against the CDF-gradient
adversary.

Kill test A found the crossover: at rank 4 the tensor-Hermite engine
ties the factor-state GHK gradient adversary. The package already
ships the defense candidate -- winning.factor.qmc_nodes puts
scrambled-Sobol nodes into the same shared-field pass, replacing the
Q^k tensor with 2^m equal-weight points. This experiment prices it:

  ranks 4 and 8, N = 200, same ensemble as the kill test;
  truth       engine with m = 17 (131k nodes, run once);
  defense     engine with m in {10, 11, 12, 13};
  incumbent   engine with the default tensor Hermite;
  adversary   the kill test's factor-state GHK gradient (R = 512
              Sobol, L = 96), rerun here on the identical instance.

Reported per method: total variation to the m=17 truth and wall
clock. The question is whether the engine's QMC variant dominates
the adversary at rank 4 (reclaiming the tie) and at rank 8 (the
rank the adversary should win against tensor quadrature).
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                ".."))
from winning.factor import race_probabilities              # noqa: E402
from winning.factor.core import qmc_nodes                  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cdfgrad", os.path.join(os.path.dirname(__file__),
                            "run_cdf_grad.py"))
_cg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cg)

import jax.numpy as jnp                                    # noqa: E402
from scipy.stats import qmc as scipy_qmc                   # noqa: E402

N = 200
R_SOBOL = 512
L_GRID = 96


def adversary(mu, V, d):
    sd = np.sqrt(d + (V ** 2).sum(1))
    lo = (mu - 7 * sd).min()
    hi = (mu + 7 * sd).max()
    xgrid = jnp.linspace(lo, hi, L_GRID)
    sob = scipy_qmc.Sobol(d=N, scramble=True, seed=5)
    unifs = sob.random(R_SOBOL)
    f = _cg.build(mu, V, d, unifs, xgrid)
    np.asarray(f(xgrid))                    # compile
    t0 = time.time()
    p = np.asarray(f(xgrid))
    return p, time.time() - t0


if __name__ == "__main__":
    results = {}
    for k in (4, 8):
        rng = np.random.default_rng(200 + k)
        mu, V, d = _cg.make_instance(N, k, 0.5, rng)
        t0 = time.time()
        Ft, Wt = qmc_nodes(k, m=17, seed=99)
        p_truth = race_probabilities(-mu, V=-V, D=d, F=Ft, W=Wt)
        t_truth = time.time() - t0
        rows = {}
        t0 = time.time()
        p_tensor = race_probabilities(-mu, V=-V, D=d)
        rows["tensor_default"] = dict(
            seconds=time.time() - t0,
            tv=float(0.5 * np.abs(p_tensor - p_truth).sum()))
        for m in (10, 11, 12, 13):
            F, W = qmc_nodes(k, m=m, seed=1)
            t0 = time.time()
            p = race_probabilities(-mu, V=-V, D=d, F=F, W=W)
            rows[f"qmc_m{m}"] = dict(
                seconds=time.time() - t0,
                tv=float(0.5 * np.abs(p - p_truth).sum()))
        p_adv, t_adv = adversary(mu, V, d)
        rows["adversary"] = dict(
            seconds=t_adv,
            tv=float(0.5 * np.abs(p_adv - p_truth).sum()))
        results[f"k{k}"] = dict(truth_seconds=t_truth, **rows)
        print(f"[k={k}] truth m=17 in {t_truth:.1f}s")
        for name, r in rows.items():
            print(f"  {name:15s} tv {r['tv']:.5f}  {r['seconds']:.2f}s")
    with open(os.path.join(os.path.dirname(__file__),
                           "results_defense.json"), "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results_defense.json")
