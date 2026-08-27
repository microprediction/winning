"""Plot 7 / Sections 8-9: fixed target-share control vs moving locally matched
control as a function of distance from the solution, plus the combined
fixed+local regression control.

mu_t = mu* + t * s * d  (d a random centred direction with unit RMS, s the
ability scale).  At each mu_t the per-draw contribution variance is measured for

    raw          e_W
    fixed        e_W - (e_V(nu*) - p*)                    target-share matched
    stale        e_W - (e_V(mu_t + a*) - q0(mu_t + a*))   local control anchored at mu*
    local        e_W - (e_V(mu_t + a_t) - q0(mu_t + a_t)) local control re-anchored at mu_t
    combined     e_W - b1 C_fixed - b2 C_local             beta fitted on a pilot
and the Rao-Blackwellised analogues.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from estimators import RB, RBCV, Combined, OneHotCV, Raw, Target, per_draw_stats, pilot_beta
from problems import get_problem
from references import lowrank_reference, iid_reference

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "distance.csv")
FIELDS = ["key", "family", "regime", "n", "ref", "t", "true_resid_l1", "method", "tr_var",
          "tr_var_raw", "vrf", "agreement", "sqdiff", "beta1", "beta2", "anchor_seconds"]


def write_row(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50])
    ap.add_argument("--families", nargs="+", default=["dense", "factor", "clustered"])
    ap.add_argument("--regimes", nargs="+", default=["diffuse", "moderate"])
    ap.add_argument("--ref", default="lowrank4")
    ap.add_argument("--t", type=float, nargs="+", default=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
    ap.add_argument("--M", type=int, default=20000)
    ap.add_argument("--M_pilot", type=int, default=512)
    a = ap.parse_args()
    for n in a.n:
        for fam in a.families:
            for reg in a.regimes:
                pr = get_problem(fam, reg, n)
                print(f"== {pr.key}", flush=True)
                target = Target(pr)
                p = pr.p_star
                s = pr.scale
                ref = (lowrank_reference(pr.Sigma_c, int(a.ref[7:])) if a.ref.startswith("lowrank")
                       else iid_reference(pr.Sigma_c))
                nu_star = ref.invert(p)
                a_star = nu_star - pr.mu_star
                rng = np.random.default_rng(3)
                d = rng.standard_normal(n)
                d -= d.mean()
                d /= np.sqrt(np.mean(d ** 2))
                base = {"key": pr.key, "family": fam, "regime": reg, "n": n, "ref": ref.name}
                for t in a.t:
                    mu = pr.mu_star + t * s * d
                    # moderately accurate pilot shares at the anchor and the local inversion
                    t0 = time.time()
                    p_tilde, _ = target.rb.rb_shares(mu, a.M_pilot, seed=11)
                    p_tilde = np.maximum(p_tilde, 1e-10)
                    p_tilde /= p_tilde.sum()
                    nu_tilde = ref.invert(p_tilde)
                    a_t = nu_tilde - mu
                    anchor_sec = time.time() - t0
                    p_true, _ = target.rb.rb_shares(mu, 20000, seed=12)
                    row = dict(base, t=t, true_resid_l1=float(np.abs(p_true - p).sum()),
                               anchor_seconds=anchor_sec)
                    raw_st = per_draw_stats(Raw(target), mu, a.M, seed=1)
                    row["tr_var_raw"] = raw_st["tr_var"]
                    fixed_oh = OneHotCV(target, ref, lambda m: nu_star, lambda nu: p, "procrustes")
                    stale_oh = OneHotCV(target, ref, lambda m: m + a_star, ref.forward, "procrustes")
                    local_oh = OneHotCV(target, ref, lambda m: m + a_t, ref.forward, "procrustes")
                    fixed_rb = RBCV(target, ref, lambda m: nu_star, lambda nu: p, "procrustes")
                    stale_rb = RBCV(target, ref, lambda m: m + a_star, ref.forward, "procrustes")
                    local_rb = RBCV(target, ref, lambda m: m + a_t, ref.forward, "procrustes")
                    comb_oh = Combined([fixed_oh, local_oh])
                    comb_rb = Combined([fixed_rb, local_rb])
                    methods = [("raw", Raw(target), None), ("fixed", fixed_oh, None),
                               ("stale", stale_oh, None), ("local", local_oh, None),
                               ("rb", RB(target), None), ("rb_fixed", fixed_rb, None),
                               ("rb_stale", stale_rb, None), ("rb_local", local_rb, None)]
                    for name, m, beta in methods:
                        st = per_draw_stats(m, mu, a.M, seed=1, beta=beta)
                        write_row(dict(row, method=name, tr_var=st["tr_var"],
                                       vrf=raw_st["tr_var"] / max(st["tr_var"], 1e-300),
                                       agreement=st["agreement"], sqdiff=st["sqdiff"]))
                        print(f"  t={t:4.2f} {name:10s} VRF={raw_st['tr_var']/max(st['tr_var'],1e-300):8.2f} "
                              f"A={st['agreement']:.3f}", flush=True)
                    for name, m in (("combined", comb_oh), ("rb_combined", comb_rb)):
                        beta = pilot_beta(m, mu, 4096, seed=99)
                        st = per_draw_stats(m, mu, a.M, seed=1, beta=beta)
                        write_row(dict(row, method=name, tr_var=st["tr_var"],
                                       vrf=raw_st["tr_var"] / max(st["tr_var"], 1e-300),
                                       agreement=st["agreement"], sqdiff=st["sqdiff"],
                                       beta1=beta[0], beta2=beta[1]))
                        print(f"  t={t:4.2f} {name:10s} VRF={raw_st['tr_var']/max(st['tr_var'],1e-300):8.2f} "
                              f"beta={np.round(beta,3)}", flush=True)


if __name__ == "__main__":
    main()
