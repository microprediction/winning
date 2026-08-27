"""Plot 4: tr Cov(r_hat) versus M, measured by replication, against the
per-draw prediction tr_var / M.  A sanity check that the coupled estimators
really are plain averages (no hidden bias or long tails at small M)."""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from estimators import RB, RBCV, OneHotCV, Raw, Target, combine
from problems import get_problem
from references import LogitReference, logit_tau0, lowrank_reference

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "variance_vs_M.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--families", nargs="+", default=["dense", "factor", "clustered"])
    ap.add_argument("--regime", default="moderate")
    ap.add_argument("--M", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--R", type=int, default=200)
    a = ap.parse_args()
    rows = []
    for fam in a.families:
        pr = get_problem(fam, a.regime, a.n)
        t = Target(pr)
        p = pr.p_star
        ref = lowrank_reference(pr.Sigma_c, 4)
        nu = ref.invert(p)
        lg = LogitReference(logit_tau0(pr.Sigma_c))
        nul = lg.invert(p)
        methods = {"raw": Raw(t), "logit_shift": OneHotCV(t, lg, lambda m: nul, lambda v: p, "commonz"),
                   "lowrank4_shift": OneHotCV(t, ref, lambda m: nu, lambda v: p, "procrustes"),
                   "rb": RB(t), "rb_lowrank4_shift": RBCV(t, ref, lambda m: nu, lambda v: p, "procrustes")}
        rng = np.random.default_rng(0)
        for name, m in methods.items():
            for M in a.M:
                ests = []
                for r in range(a.R):
                    z = rng.standard_normal((M, pr.n))
                    z0 = rng.standard_normal((M, pr.n))
                    raw, ctrls = m.parts(pr.mu_star, z, z0)
                    ests.append(combine(raw, ctrls).mean(axis=0) - p)
                E = np.array(ests)
                tr_cov = float(np.sum(E.var(axis=0)))
                l1 = float(np.mean(np.abs(E).sum(axis=1)))
                rows.append({"key": pr.key, "method": name, "M": M, "tr_cov": tr_cov,
                             "mean_l1": l1, "R": a.R})
                print(f"{pr.key} {name:18s} M={M:5d} trCov={tr_cov:.3e} E|r|_1={l1:.4f}", flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
