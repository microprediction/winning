"""The factorial again, with each candidate free to choose its own scale.

The published factorial hands both factor candidates the ORACLE loadings
V_true and a unit idiosyncratic variance. That pins the loading MAGNITUDE as
well as its direction, and under a truth that is misspecified for both
candidates the oracle magnitude is not the best-fitting magnitude for a
Gumbel-base model. So the design may be crediting the Gaussian base with an
advantage that a Gumbel-base model could partly recover simply by
re-optimising one scalar.

This script tests that. Everything is as before -- same truth (t(5) factors,
skew-normal idiosyncratic, standardised), same oracle loading DIRECTION, same
calibration to identical menu shares, same deletion scoring -- except that
each candidate independently chooses a scale s on its loadings,

    V = s * V_true,   D = 1,

by minimising its own misallocation on a held-out subset of deletion blocks.
Only the ratio of factor to idiosyncratic spread matters for an argmax, and
mu is recalibrated at every s, so s is the single free parameter that a real
estimation would have supplied automatically.

Reported: the family increment at s = 1 (the published design) against the
family increment at each candidate's own best s. If the ordering survives, the
published claim is safe and now demonstrably so; if it collapses, better to
find out here.

Reuses the paper's own calibrate_base / factor_shares_base, so this is a test
of the published pipeline rather than a reimplementation of it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from run_boundaries import calibrate_base, factor_shares_base  # noqa: E402
from raceutil import hermite_nodes  # noqa: E402
from run_factorial import (A_SKEW, K, N, N_DRAWS, SEED, deletion_shares,  # noqa: E402
                           draw_top3)

SCALES = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.8, 4.0, 5.6, 8.0]


def score_blocks(base, V, D, mu_c, F, W, top3, blocks, p_menu):
    """Misallocated fraction of redistributed mass, per block."""
    out = []
    n = len(p_menu)
    for blk in blocks:
        deleted = list(blk)   # NOT a set: np.isin treats a set as one scalar
        mass = float(sum(p_menu[i] for i in blk))
        if mass <= 0:
            continue
        keep = np.array([i for i in range(n) if i not in deleted])
        q_model = np.zeros(n)
        r, _ = factor_shares_base(mu_c, V, D, F, W, base=base, keep=keep)
        q_model[keep] = r / r.sum()
        q_true = deletion_shares(top3, deleted)
        tv = 0.5 * float(np.abs(q_model - q_true).sum())
        out.append(tv / mass)
    return np.array(out)


def main():
    rng = np.random.default_rng(SEED)
    mu_true = rng.normal(0.0, 1.0, N)
    V_true = rng.normal(0.0, 0.6 / np.sqrt(K), (N, K))

    t0 = time.perf_counter()
    top3 = draw_top3(mu_true, V_true, rng, N_DRAWS)
    p_menu = np.bincount(top3[:, 0], minlength=N) / len(top3)
    print(f"truth: {N_DRAWS / 1e6:.0f}M draws in {time.perf_counter() - t0:.0f}s")

    F2, W2 = hermite_nodes(K)
    D_unit = np.ones(N)

    # split single deletions into a tuning half and a reporting half, so the
    # chosen scale is not also the scale that scores itself
    singles = [(i,) for i in range(N)]
    order = np.random.default_rng(11).permutation(N)
    tune = [singles[i] for i in order[: N // 2]]
    report = [singles[i] for i in order[N // 2:]]
    # only score blocks with resolvable deleted mass
    tune = [b for b in tune if p_menu[b[0]] > 5e-4]
    report = [b for b in report if p_menu[b[0]] > 5e-4]
    print(f"{len(tune)} tuning blocks, {len(report)} reporting blocks")

    results = {}
    for name, base in [("factor mixed logit", "gumbel"),
                       ("factor probit", "normal")]:
        best = None
        curve = []
        for s in SCALES:
            V = s * V_true
            mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
            r, _ = factor_shares_base(mu_c, V, D_unit, F2, W2, base=base)
            resid = float(np.abs(r - p_menu).max())
            sc = score_blocks(base, V, D_unit, mu_c, F2, W2, top3, tune, p_menu)
            med = float(np.median(sc))
            curve.append({"scale": s, "tune_median": med, "calib_resid": resid})
            print(f"  {name:20s} s={s:4.2f}  tuning median {med:.4f} "
                  f"(calib resid {resid:.1e})", flush=True)
            if best is None or med < best[1]:
                best = (s, med)
        s_star = best[0]
        V = s_star * V_true
        mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
        rep_star = score_blocks(base, V, D_unit, mu_c, F2, W2, top3, report, p_menu)
        mu_1 = calibrate_base(p_menu, V_true, D_unit, F2, W2, base=base)
        rep_1 = score_blocks(base, V_true, D_unit, mu_1, F2, W2, top3, report,
                             p_menu)
        results[name] = {
            "curve": curve, "s_star": s_star,
            "report_median_s1": float(np.median(rep_1)),
            "report_median_sstar": float(np.median(rep_star)),
        }
        print(f"  -> {name}: best s = {s_star}, held-out median "
              f"{np.median(rep_1):.4f} at s=1 vs {np.median(rep_star):.4f} "
              f"at s*\n", flush=True)

    g1 = results["factor mixed logit"]["report_median_s1"]
    p1 = results["factor probit"]["report_median_s1"]
    gs = results["factor mixed logit"]["report_median_sstar"]
    ps = results["factor probit"]["report_median_sstar"]
    print("=" * 72)
    print(f"published design (both at s = 1):  mixed logit {g1:.4f}  "
          f"probit {p1:.4f}   ratio {g1 / p1:.2f}x   family increment {g1 - p1:+.4f}")
    print(f"free scale per candidate:          mixed logit {gs:.4f}  "
          f"probit {ps:.4f}   ratio {gs / ps:.2f}x   family increment {gs - ps:+.4f}")
    print("=" * 72)
    results["summary"] = {
        "ratio_s1": g1 / p1, "ratio_sstar": gs / ps,
        "family_increment_s1": g1 - p1, "family_increment_sstar": gs - ps,
    }
    (HERE / "results_factorial_freescale.json").write_text(
        json.dumps(results, indent=2))
    print(f"wrote {HERE / 'results_factorial_freescale.json'}")


if __name__ == "__main__":
    main()
