"""One parameter each: Thurstone noise scale against Luce power tilt.

The referee's tilt baseline carries a fitted gamma while contestant removal
carries nothing, so the comparison is unfair in Thurstone's disfavour. The
matched contest has performances N(a_i, sigma^2), which is locations a_i/sigma
under unit noise, and sigma flattens or sharpens exactly as gamma does.

Both parameters are fitted on training question types and scored on entirely
held-out question types, so the question is whether the Gaussian geometry buys
anything once both families may set their own scale.

The full unrestricted distribution is recoverable from the logs: the stored
Luce vector is normalized over survivors and the deleted item's share is
stored, so p_keep = luce * (1 - deleted_p) and p_deleted = deleted_p.

Usage:  python one_param.py [n_subsample]
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from exact_analyze import calibrate_np, win_probs_np


def kl(a, p):
    return sum(ai * math.log(ai / max(pi, 1e-12)) for ai, pi in zip(a, p) if ai > 0)


def prepare(rows, n_sub, seed=3):
    """Recover full unrestricted distributions and calibrate locations once."""
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n_sub:
            break
        dp = r.get("deleted_p")
        if dp is None or not (0.0 < dp < 1.0):
            continue
        luce = r["luce"]
        p_full = [x * (1.0 - dp) for x in luce] + [dp]   # survivors then deleted
        z = sum(p_full)
        if z <= 0:
            continue
        p_full = [x / z for x in p_full]
        try:
            a, err = calibrate_np(p_full)
        except Exception:
            continue
        if err > 0.05:
            continue
        keep_idx = list(range(len(luce)))
        out.append({"cat": r.get("category") or r.get("cell") or "?",
                    "actual": r["actual"], "luce": luce, "thur": r["thurstone"],
                    "a": np.asarray(a, float), "keep": keep_idx})
    return out


def thur_sigma(cell, sigma):
    w = win_probs_np(cell["a"][cell["keep"]] / sigma)
    s = w.sum()
    return (w / s).tolist() if s > 0 else cell["luce"]


def tilt(luce, gamma):
    w = [max(x, 1e-12) ** gamma for x in luce]
    z = sum(w)
    return [x / z for x in w]


def fit(cells, predict, lo, hi, iters=24):
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    f = lambda t: sum(kl(x["actual"], predict(x, t)) for x in cells) / len(cells)
    for _ in range(iters):
        if f(c) < f(d):
            b, d = d, c
            c = b - gr * (b - a)
        else:
            a, c = c, d
            d = a + gr * (b - a)
    return (a + b) / 2


def main():
    n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    rows = json.load(open(HERE / "sweep_results.json"))
    cells = prepare(rows, n_sub)
    cats = sorted({c["cat"] for c in cells})
    print(f"{len(cells):,} cells prepared over {len(cats)} question types")

    rng = random.Random(17)
    sh = cats[:]
    rng.shuffle(sh)
    folds = 5
    assign = {c: i % folds for i, c in enumerate(sh)}

    res = {"luce": [], "thur0": [], "thur_s": [], "tilt": []}
    per_cat = {}
    params = []
    for k in range(folds):
        tr = [c for c in cells if assign[c["cat"]] != k]
        te = [c for c in cells if assign[c["cat"]] == k]
        if not tr or not te:
            continue
        g = fit(tr, lambda c, t: tilt(c["luce"], t), 0.2, 6.0)
        s = fit(tr, lambda c, t: thur_sigma(c, t), 0.3, 8.0)
        params.append((g, s))
        print(f"  fold {k}: gamma={g:.3f} sigma={s:.3f} "
              f"({len(tr):,} train, {len(te):,} held out)")
        for c in te:
            res["luce"].append(kl(c["actual"], c["luce"]))
            res["thur0"].append(kl(c["actual"], c["thur"]))
            kt = kl(c["actual"], thur_sigma(c, s))
            kg = kl(c["actual"], tilt(c["luce"], g))
            res["thur_s"].append(kt)
            res["tilt"].append(kg)
            per_cat.setdefault(c["cat"], []).append(kg - kt)

    n = len(res["luce"])
    print(f"\nheld-out mean KL over {n:,} cells (lower is better)")
    labels = {"luce": "Luce, no parameter", "thur0": "Thurstone, no parameter",
              "tilt": "Luce tilt, one gamma", "thur_s": "Thurstone, one sigma"}
    for key in ("luce", "thur0", "tilt", "thur_s"):
        print(f"  {labels[key]:<28} {sum(res[key])/n:.4f}")

    flat = [x for v in per_cat.values() for x in v]
    m = sum(flat) / len(flat)
    keys = list(per_cat)
    rnd = random.Random(9)
    B = 20000
    means = []
    for _ in range(B):
        pick = [per_cat[keys[rnd.randrange(len(keys))]] for _ in keys]
        f2 = [x for grp in pick for x in grp]
        means.append(sum(f2) / len(f2))
    means.sort()
    print(f"\none-parameter Thurstone minus one-parameter tilt "
          f"(positive favours Thurstone)")
    print(f"  mean {m:+.4f}, cluster bootstrap by question type "
          f"[{means[int(.025*B)]:+.4f}, {means[int(.975*B)]:+.4f}]")
    wins = sum(1 for v in per_cat.values() if sum(v) / len(v) > 0)
    print(f"  question types favouring Thurstone: {wins}/{len(per_cat)}")


if __name__ == "__main__":
    main()
