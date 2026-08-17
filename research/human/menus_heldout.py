"""Predicting real restricted-menu choice from full-menu shares alone.

Every other dataset here derives subset choice from a ranking, which assumes people
would choose from a small menu as their ranking implies. This experiment does not:
Costa-Gomes, Cueva, Gerasimou and Tejiscak had each subject choose separately from
all 31 subsets of five goods, so restriction is observed rather than imputed. That
makes it the cleanest test of the question the paper actually asks, and it belongs in
the predictive table rather than beside it.

Protocol, split by subject so no subject informs its own prediction:

  1. on training subjects, first-place shares come only from the FULL five-item menu;
  2. renormalization takes those shares as worths, the race inverts them for
     locations, neither sees a single restricted-menu observation;
  3. each account predicts every menu of size two, three and four;
  4. score actual held-out choices from those menus by log loss;
  5. intervals resample subjects;
  6. the same pipeline runs on synthetic data from an exact Luce process with the
     observed shares as worths, giving the null gain at this sample size.

The forced-choice subgroup is reported separately. Those subjects cannot defer, so no
coding decision about deferral enters, which the referee asked for as the principal
analysis.

Usage:  python menus_heldout.py [n_null_reps]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

ITEMS = ["Ca", "Hi", "Pa", "Sa", "So"]
IDX = {it: i for i, it in enumerate(ITEMS)}
CSV = HERE / "data" / "costagomes_Exp1_2_PREST.csv"
FLOOR = 1e-6
ALPHA = 0.5


def load():
    """subject -> {menu: chosen index}, plus the forced-choice flag and which of the
    two experiments the subject belongs to. The archive holds two: experiment 1 ran
    all 31 subsets, experiment 2 a 26-menu design. They are separate studies and are
    reported separately rather than pooled."""
    by = defaultdict(dict)
    fc, exp = {}, {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ch = (r["choice"] or "").strip()
            menu = tuple(sorted(IDX[x.strip()] for x in r["menu"].split(",")
                                if x.strip() in IDX))
            if not ch or ch not in IDX or len(menu) < 2:
                continue
            by[r["subject"]][menu] = IDX[ch]
            fc[r["subject"]] = r["fc"].strip()
            exp[r["subject"]] = r["experiment"].strip()
    return by, fc, exp


def shares(subjects, by, full):
    """First-place shares from full-menu observations of these subjects only."""
    cts = np.zeros(len(ITEMS))
    n = 0
    for s in subjects:
        c = by[s].get(full)
        if c is not None:
            cts[c] += 1
            n += 1
    if n == 0:
        return None, 0
    return (cts + ALPHA) / (n + ALPHA * len(ITEMS)), n


def score(by, subjects, folds=5, seed=0):
    full = tuple(range(len(ITEMS)))
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(subjects))
    fold = np.array_split(rng.permutation(len(subs)), folds)
    tot_l = tot_g = 0.0
    cnt = 0
    ncal = []
    for f in range(folds):
        te = subs[fold[f]]
        tr = subs[np.concatenate([fold[g] for g in range(folds) if g != f])]
        p, nfull = shares(tr, by, full)
        if p is None:
            return None
        ncal.append(nfull)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            return None
        cache = {}
        for s in te:
            for menu, ch in by[s].items():
                if len(menu) >= len(ITEMS):
                    continue                      # the full menu is the calibration
                if menu not in cache:
                    idx = list(menu)
                    luce = np.maximum(p[idx] / p[idx].sum(), FLOOR)
                    w = win_probs_np(a[idx])
                    cache[menu] = (luce, np.maximum(w / w.sum(), FLOOR))
                luce, race = cache[menu]
                k = list(menu).index(ch)
                tot_l += -np.log(luce[k])
                tot_g += -np.log(race[k])
                cnt += 1
    if cnt == 0:
        return None
    return {"luce": tot_l / cnt, "race": tot_g / cnt,
            "gain": (tot_l - tot_g) / cnt, "n_choices": cnt,
            "n_cal": int(np.mean(ncal))}


def luce_synth(by, subjects, p, rng):
    """Same menus and same subjects, choices redrawn from an exact Luce process."""
    out = defaultdict(dict)
    for s in subjects:
        for menu in by[s]:
            idx = list(menu)
            q = p[idx] / p[idx].sum()
            out[s][menu] = idx[int(rng.choice(len(idx), p=q))]
    return out


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    by, fc, exp = load()
    groups = []
    for e in sorted(set(exp.values())):
        subs = [s for s in by if exp[s] == e]
        groups.append((f"experiment {e}, all", sorted(subs)))
        forced = [s for s in subs if fc.get(s) == "1"]
        if forced:
            groups.append((f"experiment {e}, forced choice", sorted(forced)))
    groups.append(("both experiments pooled", sorted(by)))
    full = tuple(range(len(ITEMS)))
    for label, subjects in groups:
        if len(subjects) < 40:
            print(f"{label}: only {len(subjects)} subjects, skipped")
            continue
        r = score(by, subjects)
        if r is None:
            print(f"{label}: not scorable")
            continue
        # subject bootstrap of the whole pipeline
        rng = np.random.default_rng(11)
        bs = []
        for b in range(400):
            pick = list(rng.choice(subjects, len(subjects), replace=True))
            bb = defaultdict(dict)
            for i, s in enumerate(pick):
                bb[f"{s}#{i}"] = by[s]
            rb = score(bb, sorted(bb), seed=b)
            if rb:
                bs.append(rb["gain"])
        lo, hi = np.quantile(bs, [0.025, 0.975]) if len(bs) > 50 else (np.nan, np.nan)
        # fitted-Luce null at this sample size
        p_all, _ = shares(subjects, by, full)
        rng2 = np.random.default_rng(99)
        null = []
        for b in range(reps):
            syn = luce_synth(by, subjects, p_all, rng2)
            rs = score(syn, sorted(syn), seed=1000 + b)
            if rs:
                null.append(rs["gain"])
        null = np.array(sorted(null))
        med = float(np.median(null))
        p95 = float(null[int(0.95 * (len(null) - 1))])
        pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
        print(f"\n{label}: {len(subjects)} subjects, {r['n_choices']} restricted-menu "
              f"choices scored, shares from {r['n_cal']} full-menu choices")
        print(f"  renormalization log loss   {r['luce']:.4f}")
        print(f"  Gaussian race log loss     {r['race']:.4f}")
        print(f"  gain                       {r['gain']:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]")
        print(f"  fitted-Luce null median    {med:+.4f}   95th {p95:+.4f}   "
              f"({len(null)} reps)")
        print(f"  excess over null           {r['gain'] - med:+.4f}   p = {pv:.3f}")


if __name__ == "__main__":
    main()
