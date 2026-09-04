"""A-vs-B evaluation stopping under shared prompts, replayed on real
per-item logs.

Two models, one benchmark, aligned per-item binary outcomes (the
pair file built by extract_pair.py). Items are revealed in random
order; each reveal shows BOTH models' outcomes on that item, which
is the common-random-numbers structure of shared-prompt evaluation.
The decision is which model has the higher population accuracy; the
population is the full log, so the truth is the full-data sign.

Three stopping rules, each swept over its own threshold so the
comparison is frontier against frontier (realized error vs mean
items revealed) with no constant-picking disputes:

  paired    the CRN-respecting Bayesian rule: only discordant items
            move the posterior; with Dirichlet(1,1,1,1) over pair
            categories, P(A better) = P(Beta(n10+1, n01+1) > 1/2),
            and the rule stops when max(P, 1-P) crosses the
            threshold. Pairing absorbs item difficulty exactly --
            the two-model special case of factor-conditioned
            probability-of-best stopping.
  indep     the same Bayesian stopping with the pairing ignored:
            independent Beta posteriors on each model's accuracy
            from marginal counts, P(A better) by normal
            approximation to the posterior difference. What a
            practitioner gets from two separate eval reports.
  mcnemar   repeated-testing frequentist rule: stop when
            |n10 - n01|/sqrt(n10 + n01) crosses a constant z,
            the sequential use of the McNemar statistic.

Replayed over many random orderings; forced decision by posterior
sign at the end of the log if a rule never stops.
"""
import json
import os

import numpy as np
from scipy.special import betainc, ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
R_ORDERINGS = 400
SEED = 11


def frontiers(a, b, thresholds_bayes, thresholds_z):
    n = len(a)
    truth = np.sign(a.mean() - b.mean())
    rng = np.random.default_rng(SEED)
    res = {("paired", t): [] for t in thresholds_bayes}
    res.update({("indep", t): [] for t in thresholds_bayes})
    res.update({("mcnemar", z): [] for z in thresholds_z})
    for _ in range(R_ORDERINGS):
        order = rng.permutation(n)
        aa, bb = a[order].astype(np.int64), b[order].astype(np.int64)
        n10 = np.cumsum((aa == 1) & (bb == 0))
        n01 = np.cumsum((aa == 0) & (bb == 1))
        na = np.cumsum(aa)
        nb = np.cumsum(bb)
        t_ax = np.arange(1, n + 1)
        # paired: P(A better) from discordant Beta posterior
        p_paired = 1.0 - betainc(n10 + 1, n01 + 1, 0.5)
        # independent: normal approx to difference of Beta posteriors
        ma, mb = (na + 1) / (t_ax + 2), (nb + 1) / (t_ax + 2)
        va = ma * (1 - ma) / (t_ax + 3)
        vb = mb * (1 - mb) / (t_ax + 3)
        p_indep = ndtr((ma - mb) / np.sqrt(va + vb + 1e-300))
        # mcnemar z
        disc = n10 + n01
        z = np.where(disc > 0, (n10 - n01) / np.sqrt(np.maximum(disc, 1)),
                     0.0)
        for thr in thresholds_bayes:
            for name, p in (("paired", p_paired), ("indep", p_indep)):
                hit = np.flatnonzero(np.maximum(p, 1 - p) >= thr)
                early = len(hit) > 0
                stop = hit[0] if early else n - 1
                decide = np.sign(p[stop] - 0.5)
                if decide == 0:
                    decide = np.sign(n10[stop] - n01[stop]) or 1.0
                res[(name, thr)].append((stop + 1, decide != truth,
                                         early))
        for zc in thresholds_z:
            hit = np.flatnonzero(np.abs(z) >= zc)
            early = len(hit) > 0
            stop = hit[0] if early else n - 1
            decide = np.sign(z[stop]) or 1.0
            res[("mcnemar", zc)].append((stop + 1, decide != truth,
                                         early))
    out = {}
    for key, rows in res.items():
        arr = np.array(rows, dtype=float)
        early = arr[:, 2] > 0
        out[key] = dict(
            mean_items=float(arr[:, 0].mean()),
            error=float(arr[:, 1].mean()),
            early_frac=float(early.mean()),
            error_when_early=(float(arr[early, 1].mean())
                              if early.any() else None))
    return out, truth


if __name__ == "__main__":
    import glob
    all_results = {}
    thr_b = (0.95, 0.99, 0.999, 0.9999)
    thr_z = (1.96, 2.58, 3.09, 3.72)
    for path in sorted(glob.glob(os.path.join(HERE, "pair_*.npz"))):
        z = np.load(path)
        a, b = z["a"], z["b"]
        names = [str(x) for x in z["names"]]
        tag = f"{names[0]}_vs_{names[1]}"
        print(f"[{tag}] acc {a.mean():.3f}/{b.mean():.3f}, "
              f"gap {abs(a.mean()-b.mean())*100:.1f}pt, n={len(a)}")
        out, truth = frontiers(a, b, thr_b, thr_z)
        rows = {}
        for (name, t), v in sorted(out.items()):
            rows[f"{name}@{t}"] = v
            ew = v['error_when_early']
            print(f"  {name:8s} thr={t:<7} mean items "
                  f"{v['mean_items']:7.1f}  early "
                  f"{v['early_frac']:.2f}  err(early) "
                  f"{ew if ew is not None else float('nan'):.3f}")
        all_results[tag] = dict(acc=[float(a.mean()), float(b.mean())],
                                frontier=rows)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("wrote results.json")
