"""Referee's decisive baseline: a globally calibrated power tilt.

The batteries compare two zero-parameter forecasts, Luce renormalization and
Gaussian contestant removal. Winning that comparison does not identify a
Gaussian mechanism. A one-parameter sharpening rule,

    p_i(gamma) = p_i^gamma / sum_j p_j^gamma,   i in the surviving set,

is the obvious rival: if a single gamma fitted on some question types matches
Gaussian removal on entirely held-out question types, the phenomenon is
sharpening after exclusion rather than contest geometry.

Because the stored Luce vector is the unrestricted distribution renormalized
over the survivors, and renormalizing after exponentiation is invariant to any
prior normalization, the tilt prediction is computable exactly from what is
already logged. No API calls.

Split is by question type, never within one, so gamma is never fitted and
evaluated on the same category.

Usage:  python power_tilt.py [results.json ...]
"""
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def tilt(luce, gamma):
    w = [max(x, 1e-12) ** gamma for x in luce]
    z = sum(w)
    return [x / z for x in w]


def load(files):
    rows = []
    for f in files:
        for r in json.load(open(f)):
            if "luce" in r and "actual" in r and "thurstone" in r:
                cat = r.get("category") or r.get("cell") or "?"
                rows.append({"cat": cat, "actual": r["actual"], "luce": r["luce"],
                             "thur": r["thurstone"],
                             "H": r.get("H_unq", r.get("H_slot1", 0.0)),
                             "model": r.get("model", "?")})
    return rows


def mean_kl(rows, pred):
    return sum(kl(r["actual"], pred(r)) for r in rows) / len(rows)


def fit_gamma(rows, lo=0.2, hi=6.0, iters=60):
    """Golden-section search on mean KL over the training rows."""
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        fc = mean_kl(rows, lambda r: tilt(r["luce"], c))
        fd = mean_kl(rows, lambda r: tilt(r["luce"], d))
        if fc < fd:
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return (a + b) / 2


def main():
    files = sys.argv[1:] or [str(HERE / "sweep_results.json")]
    rows = load(files)
    cats = sorted({r["cat"] for r in rows})
    print(f"{len(rows):,} cells over {len(cats)} question types from "
          f"{', '.join(Path(f).name for f in files)}")

    rng = random.Random(17)
    folds = 5
    shuffled = cats[:]
    rng.shuffle(shuffled)
    assign = {c: i % folds for i, c in enumerate(shuffled)}

    agg = {"luce": [], "thur": [], "tilt": []}
    gammas = []
    for k in range(folds):
        train = [r for r in rows if assign[r["cat"]] != k]
        test = [r for r in rows if assign[r["cat"]] == k]
        g = fit_gamma(train)
        gammas.append(g)
        for r in test:
            agg["luce"].append(kl(r["actual"], r["luce"]))
            agg["thur"].append(kl(r["actual"], r["thur"]))
            agg["tilt"].append(kl(r["actual"], tilt(r["luce"], g)))
        print(f"  fold {k}: gamma={g:.3f} fitted on {len(train):,} cells, "
              f"{len(test):,} held-out cells in {sum(1 for c in cats if assign[c]==k)} types")

    n = len(agg["luce"])
    print(f"\nheld-out mean KL over {n:,} cells (lower is better)")
    for name in ("luce", "thur", "tilt"):
        print(f"  {name:<6} {sum(agg[name])/n:.4f}")

    def cluster_boot(diffs_by_cat, B=20000, seed=5):
        """Resample question types, not cells: the referee's point 9."""
        keys = list(diffs_by_cat)
        rnd = random.Random(seed)
        means = []
        for _ in range(B):
            pick = [diffs_by_cat[keys[rnd.randrange(len(keys))]] for _ in keys]
            flat = [x for grp in pick for x in grp]
            means.append(sum(flat) / len(flat))
        means.sort()
        return means[int(.025 * B)], means[int(.975 * B)]

    # Thurstone against the tilt, clustered by question type
    per_cat = {}
    idx = 0
    for k in range(folds):
        test = [r for r in rows if assign[r["cat"]] == k]
        g = gammas[k]
        for r in test:
            d = kl(r["actual"], tilt(r["luce"], g)) - kl(r["actual"], r["thur"])
            per_cat.setdefault(r["cat"], []).append(d)
    flat = [x for v in per_cat.values() for x in v]
    m = sum(flat) / len(flat)
    lo, hi = cluster_boot(per_cat)
    print(f"\nThurstone minus calibrated tilt (positive favors Thurstone)")
    print(f"  mean {m:+.4f}, cluster bootstrap by question type [{lo:+.4f}, {hi:+.4f}]")
    wins = sum(1 for v in per_cat.values() if sum(v) / len(v) > 0)
    print(f"  question types where Thurstone beats the tilt: {wins}/{len(per_cat)}")

    # And the original headline, re-estimated with the correct clustering
    per_cat_l = {}
    for r in rows:
        per_cat_l.setdefault(r["cat"], []).append(
            kl(r["actual"], r["luce"]) - kl(r["actual"], r["thur"]))
    flat_l = [x for v in per_cat_l.values() for x in v]
    ml = sum(flat_l) / len(flat_l)
    lo_l, hi_l = cluster_boot(per_cat_l)
    print(f"\nThurstone minus Luce, same clustering")
    print(f"  mean {ml:+.4f}, cluster bootstrap by question type [{lo_l:+.4f}, {hi_l:+.4f}]")


if __name__ == "__main__":
    main()
