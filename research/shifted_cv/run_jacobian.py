"""Section 13: does the shifted reference also reduce Jacobian noise?

At mu*, with a reference race Sigma0 calibrated to p*:
    (a) J_a = mean_m J_Sigma^(m)                       target envelope Laplacian only
    (b) J_b = J_0(nu*) + mean_m [J_Sigma^(m) - J_0^(m)] reference Jacobian + coupled correction
with J_0(nu*) computed to high precision (M0 = 20000 reference envelope draws;
the reference is a cheap race so this is a one-off cost).  The truth J_true is
the target envelope Laplacian at M = 40000.  Error is measured in random
centred matrix-vector products ||(J - J_true) v|| / ||J_true v|| and in
Frobenius norm, over R replications of M draws.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from estimators import Target
from problems import get_problem
from references import lowrank_reference, procrustes

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "jacobian.csv")
FIELDS = ["key", "n", "ref", "M", "method", "jv_rel_err_mean", "jv_rel_err_median",
          "fro_rel_err", "coupling"]


def write_row(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def lap(race, mu, eta):
    _, J = race.conditional_shares(mu, eta, want_J=True)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 250])
    ap.add_argument("--families", nargs="+", default=["dense", "factor", "clustered"])
    ap.add_argument("--regimes", nargs="+", default=["moderate"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--M", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--R", type=int, default=20)
    a = ap.parse_args()
    for n in a.n:
        for fam in a.families:
            for reg in a.regimes:
                pr = get_problem(fam, reg, n)
                target = Target(pr)
                ref = lowrank_reference(pr.Sigma_c, a.rank)
                nu = ref.invert(pr.p_star)
                mu = pr.mu_star
                rng = np.random.default_rng(0)
                _, _, J_true = target.rb.rb_shares(mu, 40000, seed=100, want_J=True)
                _, _, J0 = ref.rb.rb_shares(nu, 20000, seed=101, want_J=True)
                V = rng.standard_normal((n, 8))
                V -= V.mean(axis=0)
                JV_true = J_true @ V
                nrm = np.linalg.norm(JV_true, axis=0)
                Q = procrustes(target.rb.A, ref.rb.A)
                A0Q = ref.rb.A @ Q
                print(f"== {pr.key} rank{a.rank}: |J_true|_F={np.linalg.norm(J_true):.3f} "
                      f"|J_true-J0|_F/|J_true|_F={np.linalg.norm(J_true-J0)/np.linalg.norm(J_true):.3f}", flush=True)
                for M in a.M:
                    errs = {"target_only": [], "ref_plus_correction": [], "ref_plus_correction_indep": []}
                    fro = {k: [] for k in errs}
                    for r in range(a.R):
                        z = rng.standard_normal((M, n))
                        z_ind = rng.standard_normal((M, n))
                        Ja = lap(target.rb, mu, target.rb.eta_from_z(z)) / M
                        J0m = lap(ref.rb, nu, z @ A0Q.T) / M
                        J0i = lap(ref.rb, nu, z_ind @ A0Q.T) / M
                        Jb = J0 + Ja - J0m
                        Jc = J0 + Ja - J0i
                        for k, J in (("target_only", Ja), ("ref_plus_correction", Jb),
                                     ("ref_plus_correction_indep", Jc)):
                            e = np.linalg.norm((J - J_true) @ V, axis=0) / nrm
                            errs[k].append(e)
                            fro[k].append(np.linalg.norm(J - J_true) / np.linalg.norm(J_true))
                    e0 = np.linalg.norm((J0 - J_true) @ V, axis=0) / nrm
                    write_row({"key": pr.key, "n": n, "ref": ref.name, "M": M, "method": "ref_only",
                               "jv_rel_err_mean": float(e0.mean()), "jv_rel_err_median": float(np.median(e0)),
                               "fro_rel_err": float(np.linalg.norm(J0 - J_true) / np.linalg.norm(J_true)),
                               "coupling": "none"})
                    print(f"  M={M:4d} {'ref_only':28s} Jv rel err mean={e0.mean():.4f} median={np.median(e0):.4f}", flush=True)
                    for k in errs:
                        e = np.concatenate(errs[k])
                        write_row({"key": pr.key, "n": n, "ref": ref.name, "M": M, "method": k,
                                   "jv_rel_err_mean": float(e.mean()), "jv_rel_err_median": float(np.median(e)),
                                   "fro_rel_err": float(np.mean(fro[k])), "coupling": "procrustes"})
                        print(f"  M={M:4d} {k:28s} Jv rel err mean={e.mean():.4f} median={np.median(e):.4f} "
                              f"fro={np.mean(fro[k]):.4f}", flush=True)


if __name__ == "__main__":
    main()
