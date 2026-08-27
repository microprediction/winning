"""Section 21 / Plots 1-3, 6: winner agreement and variance reduction at mu*.

For every problem and every control construction, draw M coupled samples at
the true solution and record P(W = V), the per-draw contribution variance
tr Var(c), the VRF relative to raw winner counting, and the cost-adjusted
VRF (a coupled draw costs two O(n^2) matmuls, a raw draw one).
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from estimators import (RB, RBCV, OneHotCV, Raw, Target, optimize_Q, per_draw_stats,
                        pilot_beta)
from problems import FAMILIES, REGIMES, get_problem
from references import (LogitReference, diag_reference, iid_reference, logit_tau0,
                        lowrank_reference)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "agreement.csv")
FIELDS = ["key", "family", "regime", "n", "method", "ref", "rank", "coupling",
          "tau_mult", "sqrt", "beta", "M", "tr_var", "tr_var_raw", "vrf", "vrf_cost",
          "agreement", "chance", "sqdiff", "bias_l1", "bias_se_l1", "ref_seconds",
          "ref_nodes", "ref_converged", "ref_resid", "seconds", "q_steps_accepted",
          "q_pilot_gain"]


def m_eval(n):
    return {20: 40000, 50: 40000, 100: 40000, 250: 20000, 500: 10000, 1000: 6000}.get(n, 6000)


def existing_rows():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {(r["key"], r["method"]) for r in csv.DictReader(f)}


def write_row(row):
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_problem(pr, ranks, tau_mults, do_qopt, done):
    n = pr.n
    M = m_eval(n)
    mu = pr.mu_star
    p = pr.p_star
    chance = float((p ** 2).sum())
    target = Target(pr, "sym")
    base = {"key": pr.key, "family": pr.family, "regime": pr.regime, "n": n, "M": M,
            "chance": chance, "sqrt": "sym", "beta": 1.0}
    raw_stats = per_draw_stats(Raw(target), mu, M, seed=1)
    tr_raw = raw_stats["tr_var"]
    base["tr_var_raw"] = tr_raw

    def record(method, extra, seed=1, beta=None):
        if (pr.key, method.name) in done:
            return None
        t0 = time.time()
        st = per_draw_stats(method, mu, M, seed=seed, beta=beta)
        row = dict(base)
        row.update(extra)
        row.update({"method": method.name, "tr_var": st["tr_var"],
                    "vrf": tr_raw / max(st["tr_var"], 1e-300),
                    "vrf_cost": tr_raw / max(st["tr_var"], 1e-300) / method.cost,
                    "agreement": st["agreement"], "sqdiff": st["sqdiff"],
                    "bias_l1": float(np.abs(st["mean"] - p).sum()),
                    "bias_se_l1": float(st["se"].sum()),
                    "seconds": time.time() - t0})
        write_row(row)
        print(f"  {method.name:42s} A={st['agreement']:.3f} VRF={row['vrf']:8.2f} "
              f"bias_l1={row['bias_l1']:.4f}(se {row['bias_se_l1']:.4f})", flush=True)
        return st

    record(Raw(target), {"ref": "none", "coupling": "none"})
    record(RB(target), {"ref": "none", "coupling": "none"})

    fixed = lambda nu: (lambda m: nu)   # noqa: E731
    pstar_q = lambda nu: p                # noqa: E731

    # ---- logit ---------------------------------------------------------
    # Target-share matched logit: V = argmax(tau log p* + tau g) does not depend
    # on tau, so the temperature is not a coupling parameter there; it IS one
    # for the same-mu logit control q = softmax(mu / tau).
    tau0 = logit_tau0(pr.Sigma_c)
    ref = LogitReference(tau0)
    nu = ref.invert(p)
    for cpl in ("commonz", "indep"):
        record(OneHotCV(target, ref, fixed(nu), pstar_q, cpl, name=f"logit_{cpl}"),
               {"ref": "logit", "coupling": cpl, "tau_mult": 1.0, "ref_seconds": 0.0})
    mb = OneHotCV(target, ref, fixed(nu), pstar_q, "commonz", name="logit_commonz_beta")
    beta = pilot_beta(mb, mu, 4096, seed=99)
    record(mb, {"ref": "logit", "coupling": "commonz", "tau_mult": 1.0, "beta": float(beta[0])},
           beta=beta)
    for tm in tau_mults:
        reft = LogitReference(tau0 * tm)
        record(OneHotCV(target, reft, lambda m: m, reft.forward, "commonz",
                        name=f"logit_samemu_tm{tm}"),
               {"ref": "logit_samemu", "coupling": "commonz", "tau_mult": tm})

    # ---- gaussian references --------------------------------------------
    refs = [("iid", 0, iid_reference(pr.Sigma_c)), ("diag", 0, diag_reference(pr.Sigma_c))]
    for r in ranks:
        refs.append((f"lowrank", r, lowrank_reference(pr.Sigma_c, r)))
    for fam, r, ref in refs:
        if all((pr.key, f"{ref.name}_{c}") in done for c in ("common", "procrustes", "indep")) and \
           all((pr.key, f"rb_{ref.name}_{c}") in done for c in ("common", "procrustes")):
            continue
        t0 = time.time()
        nu = ref.invert(p)
        rsec = time.time() - t0
        info = getattr(ref, "last_invert_info", {}) or {}
        q = ref.forward(nu)
        extra = {"ref": fam, "rank": r, "ref_seconds": rsec, "ref_nodes": len(ref.W),
                 "ref_converged": info.get("converged", ""), "ref_resid": info.get("residual", "")}
        print(f"  ref {ref.name}: invert {rsec:.1f}s, lattice max|q-p*|={np.abs(q - p).max():.2e}", flush=True)
        for cpl in ("common", "procrustes", "indep"):
            if cpl == "indep" and fam == "lowrank" and r not in (4,):
                continue
            record(OneHotCV(target, ref, fixed(nu), pstar_q, cpl, name=f"{ref.name}_{cpl}"),
                   dict(extra, coupling=cpl))
        for cpl in ("common", "procrustes"):
            record(RBCV(target, ref, fixed(nu), pstar_q, cpl, name=f"rb_{ref.name}_{cpl}"),
                   dict(extra, coupling=cpl))
        if fam == "lowrank" and r == 4:
            m = OneHotCV(target, ref, fixed(nu), pstar_q, "common", name=f"{ref.name}_common_beta")
            beta = pilot_beta(m, mu, 4096, seed=99)
            record(m, dict(extra, coupling="common", beta=float(beta[0])), beta=beta)
            m = RBCV(target, ref, fixed(nu), pstar_q, "common", name=f"rb_{ref.name}_common_beta")
            beta = pilot_beta(m, mu, 4096, seed=99)
            record(m, dict(extra, coupling="common", beta=float(beta[0])), beta=beta)
            # square-root comparison for the target coupling
            for kind in ("chol", "eig"):
                tk = Target(pr, kind)
                record(OneHotCV(tk, ref, fixed(nu), pstar_q, "common", name=f"{ref.name}_common_{kind}"),
                       dict(extra, coupling="common", sqrt=kind))
                record(OneHotCV(tk, ref, fixed(nu), pstar_q, "procrustes", name=f"{ref.name}_procrustes_{kind}"),
                       dict(extra, coupling="procrustes", sqrt=kind))
            if do_qopt:
                # winner-agreement hill climb from the Procrustes solution
                Q0 = OneHotCV(target, ref, fixed(nu), pstar_q, "procrustes").Q
                mk = lambda Q: OneHotCV(target, ref, fixed(nu), pstar_q, "Q", Q=Q)  # noqa: E731
                t0 = time.time()
                Q, best, hist, acc = optimize_Q(mk, Q0, mu, n, pilot_M=4096, steps=80,
                                                k=max(8, n // 10), step=0.2, seed=5)
                record(OneHotCV(target, ref, fixed(nu), pstar_q, "Q", Q=Q, name=f"{ref.name}_Qopt"),
                       dict(extra, coupling="Qopt", q_steps_accepted=acc,
                            q_pilot_gain=best - hist[0]), seed=2)
                Q0 = RBCV(target, ref, fixed(nu), pstar_q, "procrustes").Q
                mk = lambda Q: RBCV(target, ref, fixed(nu), pstar_q, "Q", Q=Q)  # noqa: E731
                Q, best, hist, acc = optimize_Q(mk, Q0, mu, n, pilot_M=2048, steps=80,
                                                k=max(8, n // 10), step=0.2, seed=5,
                                                objective="sqdiff")
                record(RBCV(target, ref, fixed(nu), pstar_q, "Q", Q=Q, name=f"rb_{ref.name}_Qopt"),
                       dict(extra, coupling="Qopt", q_steps_accepted=acc,
                            q_pilot_gain=best - hist[0]), seed=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[50, 250])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--regimes", nargs="+", default=REGIMES)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--tau", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4])
    ap.add_argument("--qopt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    done = existing_rows()
    for n in a.n:
        for fam in a.families:
            for reg in a.regimes:
                pr = get_problem(fam, reg, n, seed=a.seed)
                print(f"== {pr.key}  p_max={pr.meta['p_max']:.3f} rare(<1e-4)={pr.meta['n_below_1e-4']} "
                      f"cond={pr.meta['cond']:.1e} ref={pr.meta['ref_seconds']:.1f}s", flush=True)
                run_problem(pr, a.ranks, a.tau, a.qopt, done)


if __name__ == "__main__":
    main()
