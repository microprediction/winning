"""Sections 14, 17, 19 / Plots 5 and 8: full share inversion with each
residual estimator at matched draw counts M.

All methods start from the same mu0 (the rank-r reference's exact inverse
nu*, i.e. the surrogate answer) and use the same Jacobian (target envelope
Laplacian on M_J fixed draws), so the comparison isolates the residual
estimator.  'surrogate' records the error of mu0 itself: the answer with zero
Monte Carlo.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from estimators import RB, RBCV, Combined, OneHotCV, Raw, Target, pilot_beta
from invert import newton_invert, recovery_metrics
from problems import get_problem
from references import LogitReference, diag_reference, iid_reference, logit_tau0, lowrank_reference

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "inversion.csv")
FIELDS = ["key", "family", "regime", "n", "method", "M", "seed", "rmse_mu", "rmse_mu0",
          "max_abs_mu", "corr_mu", "share_l1", "share_linf", "share_l1_floor", "final_l1",
          "iterations", "samples", "seconds", "ref_seconds", "converged", "failed", "reason", "final_se_l1", "anchors", "rmse_mu_ident", "n_ident"]


def write_row(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def existing():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(r["key"], r["method"], int(r["M"]), int(r["seed"])) for r in csv.DictReader(f)}


class LocalControl:
    """Moving locally matched control: re-anchors when mu drifts more than
    `radius` (RMS, in ability-scale units) from the anchor. Each re-anchor
    costs a pilot RB estimate and a reference inversion."""

    def __init__(self, target, ref, mu0, radius, M_pilot=512, rb=False, coupling="procrustes"):
        self.t, self.ref, self.radius, self.M_pilot, self.rb = target, ref, radius, M_pilot, rb
        self.coupling = coupling
        self.anchor = None
        self.a = None
        self.anchors = 0
        self.anchor_seconds = 0.0
        self.pilot_samples = 0
        self.reanchor(mu0)
        cls = RBCV if rb else OneHotCV
        self.inner = cls(target, ref, lambda m: m + self.a, ref.forward, coupling)
        self.smooth = self.inner.smooth
        self.cost = self.inner.cost
        self.name = ("rb_" if rb else "") + f"local_{ref.name}"

    def reanchor(self, mu):
        t0 = time.time()
        p_t, _ = self.t.rb.rb_shares(mu, self.M_pilot, seed=1234 + self.anchors)
        p_t = np.maximum(p_t, 1e-10)
        p_t /= p_t.sum()
        nu_t = self.ref.invert(p_t)
        self.a = nu_t - mu
        self.anchor = mu.copy()
        self.anchors += 1
        self.pilot_samples += self.M_pilot
        self.anchor_seconds += time.time() - t0

    def on_iterate(self, mu):
        if self.anchors < 3 and \
           np.sqrt(np.mean((mu - self.anchor) ** 2)) > self.radius * self.t.problem.scale:
            self.reanchor(mu)

    def parts(self, mu, z, z0=None):
        return self.inner.parts(mu, z, z0)


def build_methods(pr, target, rank, which):
    p = pr.p_star
    refs = {}
    t0 = time.time()
    ref_lr = lowrank_reference(pr.Sigma_c, rank)
    nu_lr = ref_lr.invert(p)
    ref_sec = time.time() - t0
    refs["lowrank"] = (ref_lr, nu_lr, ref_sec)
    ref_iid = iid_reference(pr.Sigma_c)
    nu_iid = ref_iid.invert(p)
    ref_dg = diag_reference(pr.Sigma_c)
    nu_dg = ref_dg.invert(p)
    tau0 = logit_tau0(pr.Sigma_c)
    ref_lg = LogitReference(tau0)
    nu_lg = ref_lg.invert(p)
    mu0 = nu_lr
    t0 = time.time()
    _, _, J0 = ref_lr.rb.rb_shares(nu_lr, 20000, seed=4321, want_J=True)
    ref_sec += time.time() - t0
    fixed = lambda nu: (lambda m: nu)  # noqa: E731
    pq = lambda nu: p                   # noqa: E731

    def make(name):
        if name == "raw":
            return Raw(target)
        if name == "logit_samemu":
            return OneHotCV(target, ref_lg, lambda m: m, ref_lg.forward, "commonz", name=name)
        if name == "logit_shift":
            return OneHotCV(target, ref_lg, fixed(nu_lg), pq, "commonz", name=name)
        if name == "iid_shift":
            return OneHotCV(target, ref_iid, fixed(nu_iid), pq, "procrustes", name=name)
        if name == "diag_shift":
            return OneHotCV(target, ref_dg, fixed(nu_dg), pq, "procrustes", name=name)
        if name == "lowrank_shift":
            return OneHotCV(target, ref_lr, fixed(nu_lr), pq, "procrustes", name=name)
        if name == "lowrank_local":
            return LocalControl(target, ref_lr, mu0, radius=0.5)
        if name == "lowrank_samemu":
            # shift fixed at a = nu* - mu0 = 0: reference at the CURRENT mu with its exact shares
            return OneHotCV(target, ref_lr, lambda m: m, ref_lr.forward, "procrustes", name=name)
        if name == "rb_lowrank_samemu":
            return RBCV(target, ref_lr, lambda m: m, ref_lr.forward, "procrustes", name=name)
        if name == "rb":
            return RB(target)
        if name == "rb_lowrank_shift":
            return RBCV(target, ref_lr, fixed(nu_lr), pq, "procrustes", name=name)
        if name == "rb_lowrank_local":
            return LocalControl(target, ref_lr, mu0, radius=0.5, rb=True)
        if name == "multi":
            m = Combined([OneHotCV(target, ref_lr, fixed(nu_lr), pq, "procrustes"),
                          OneHotCV(target, ref_lg, fixed(nu_lg), pq, "commonz"),
                          OneHotCV(target, ref_iid, fixed(nu_iid), pq, "procrustes")], name=name)
            m.beta = pilot_beta(m, mu0, 512, seed=77)
            return m
        if name == "rb_multi":
            m = Combined([RBCV(target, ref_lr, fixed(nu_lr), pq, "procrustes"),
                          RBCV(target, ref_iid, fixed(nu_iid), pq, "procrustes")], name=name)
            m.beta = pilot_beta(m, mu0, 512, seed=77)
            return m
        raise ValueError(name)

    return mu0, ref_sec, make, J0


ALL = ["raw", "logit_samemu", "logit_shift", "iid_shift", "diag_shift", "lowrank_shift",
       "lowrank_local", "lowrank_samemu", "rb", "rb_lowrank_shift", "rb_lowrank_local",
       "rb_lowrank_samemu", "multi", "rb_multi"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50])
    ap.add_argument("--families", nargs="+", default=["dense", "factor", "clustered"])
    ap.add_argument("--regimes", nargs="+", default=["diffuse", "moderate"])
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--M", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--max_iter", type=int, default=40)
    ap.add_argument("--tag", default="", help="suffix appended to method names (e.g. _r8)")
    a = ap.parse_args()
    done = existing()
    for n in a.n:
        for fam in a.families:
            for reg in a.regimes:
                pr = get_problem(fam, reg, n)
                target = Target(pr)
                mu0, ref_sec, make, J0 = build_methods(pr, target, a.rank, a.methods)
                m0 = recovery_metrics(mu0, pr)
                print(f"== {pr.key}: surrogate rmse={m0['rmse_mu']:.4f} rmse_id={m0['rmse_mu_ident']:.4f} share_l1={m0['share_l1']:.4f} "
                      f"(ref inversion {ref_sec:.1f}s)", flush=True)
                if (pr.key, "surrogate" + a.tag, 0, 0) not in done:
                    write_row(dict(key=pr.key, family=fam, regime=reg, n=n, method="surrogate" + a.tag, M=0, seed=0,
                                   rmse_mu0=m0["rmse_mu"], ref_seconds=ref_sec, failed=False, **m0))
                for M in a.M:
                    for seed in a.seeds:
                        for name in a.methods:
                            if (pr.key, name + a.tag, M, seed) in done:
                                continue
                            method = make(name)
                            try:
                                res = newton_invert(method, pr.p_star, mu0, M, seed=seed,
                                                    max_iter=a.max_iter, beta=getattr(method, "beta", None),
                                                    J_fixed=J0)
                                met = recovery_metrics(res["mu"], pr)
                                failed = not np.all(np.isfinite(res["mu"]))
                            except Exception as e:  # noqa: BLE001
                                print("   FAILED", name, e, flush=True)
                                res = {"final_l1": np.nan, "iterations": 0, "samples": 0, "seconds": 0, "converged": False, "reason": "error", "final_se_l1": np.nan}
                                met = {k: np.nan for k in ("rmse_mu", "rmse_mu_ident", "n_ident", "max_abs_mu", "corr_mu", "share_l1", "share_linf", "share_l1_floor")}
                                failed = True
                            extra_sec = getattr(method, "anchor_seconds", 0.0)
                            extra_samp = getattr(method, "pilot_samples", 0)
                            write_row(dict(key=pr.key, family=fam, regime=reg, n=n, method=name + a.tag, M=M, seed=seed,
                                           rmse_mu0=m0["rmse_mu"], final_l1=res["final_l1"],
                                           iterations=res["iterations"], samples=res["samples"] + extra_samp,
                                           seconds=res["seconds"] + extra_sec, ref_seconds=ref_sec,
                                           converged=res["converged"], failed=failed, reason=res["reason"],
                                           final_se_l1=res["final_se_l1"], anchors=getattr(method, "anchors", 0), **met))
                            print(f"  M={M:4d} s={seed} {name:18s} rmse={met['rmse_mu']:.4f} rmse_id={met['rmse_mu_ident']:.4f} "
                                  f"share_l1={met['share_l1']:.4f} it={res['iterations']:2d} "
                                  f"t={res['seconds']+extra_sec:6.1f}s samples={res['samples']+extra_samp} {res['reason']}", flush=True)


if __name__ == "__main__":
    main()
