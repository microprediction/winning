"""Restricted-menu prediction on lotteries, from Aguiar, Boccardi, Kashaev and Kim.

Their experiment observes choices from all thirty-one non-empty subsets of five
lotteries, with a default alternative always available, so menus run from two to six
alternatives. Subjects are a cross-section with at most two disjoint choice sets each,
which makes the data a set of population shares per menu with no within-subject
dependence to model. That is the object this paper is about, in a domain, risk, that
shares nothing with consumer goods.

The dataset is also adversarial for us on the authors' own analysis: they report that
random utility cannot explain the population behaviour while a logit attention model
survives. A Gaussian race is a random utility model, so this is a test the paper can
lose.

Encoding, recovered from the data rather than documentation: the menu column indexes
subsets of the five lotteries by size and then lexicographically, taking codes 2
through 32, and alternative 0 is the always-available default. Calibration uses the
full six-alternative menu alone; every smaller menu is then predicted and scored
against the choices actually made there.

Usage:  python lotteries.py [n_null_reps]
"""
import collections
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

DATA = HERE / "data" / "aguiar"
FLOOR = 1e-6
ALPHA = 0.5
LOTS = [1, 2, 3, 4, 5]


def code_to_menu():
    """code -> menu, as subsets of the lotteries by size then lexicographic order."""
    out, code = {}, 2
    for r in range(1, len(LOTS) + 1):
        for c in itertools.combinations(LOTS, r):
            out[code] = [0] + list(c)
            code += 1
    return out


def load(name="menu_choice_pooled.csv"):
    m = code_to_menu()
    rows = []
    with open(DATA / name) as f:
        for a, b in list(csv.reader(f))[1:]:
            code, ch = int(a), int(b)
            menu = m.get(code)
            if menu and ch in menu:
                rows.append((tuple(menu), ch))
    return rows, m


def score(rows, full, folds=5, seed=0):
    """Held-out log loss, folds over observations of the full menu and of each subset."""
    rng = np.random.default_rng(seed)
    fullobs = [r for r in rows if r[0] == full]
    subobs = [r for r in rows if r[0] != full]
    if len(fullobs) < 50 or not subobs:
        return None
    K = len(full)
    pos = {a: i for i, a in enumerate(full)}
    fa = np.array_split(rng.permutation(len(fullobs)), folds)
    sa = np.array_split(rng.permutation(len(subobs)), folds)
    tot_l = tot_g = 0.0
    n = 0
    for f in range(folds):
        tr = [fullobs[j] for g in range(folds) if g != f for j in fa[g]]
        cts = np.zeros(K)
        for _, ch in tr:
            cts[pos[ch]] += 1
        p = (cts + ALPHA) / (cts.sum() + ALPHA * K)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            return None
        a = np.asarray(a)
        cache = {}
        for j in sa[f]:
            menu, ch = subobs[j]
            if menu not in cache:
                idx = [pos[x] for x in menu]
                lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
                w = win_probs_np(a[idx])
                cache[menu] = (lu, np.maximum(w / w.sum(), FLOOR), list(menu))
            lu, ra, order = cache[menu]
            k = order.index(ch)
            tot_l += -np.log(lu[k])
            tot_g += -np.log(ra[k])
            n += 1
    return {"n_scored": n, "n_cal": len(fullobs),
            "luce": tot_l / n, "race": tot_g / n, "gain": (tot_l - tot_g) / n}


def luce_null(rows, full, reps, seed=5):
    """Redraw every choice from an exact Luce process with the observed full-menu
    shares as worths, keeping the menus and their counts exactly as observed."""
    K = len(full)
    pos = {a: i for i, a in enumerate(full)}
    cts = np.zeros(K)
    for menu, ch in rows:
        if menu == full:
            cts[pos[ch]] += 1
    u = (cts + ALPHA) / (cts.sum() + ALPHA * K)
    rng = np.random.default_rng(seed)
    out = []
    for b in range(reps):
        syn = []
        for menu, _ in rows:
            idx = [pos[x] for x in menu]
            q = u[idx] / u[idx].sum()
            syn.append((menu, menu[int(rng.choice(len(menu), p=q))]))
        r = score(syn, full, seed=1000 + b)
        if r:
            out.append(r["gain"])
    return np.array(sorted(out))


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    for name in ("menu_choice_pooled.csv", "menu_choice_low.csv",
                 "menu_choice_medium.csv", "menu_choice_high.csv"):
        if not (DATA / name).exists():
            continue
        rows, m = load(name)
        full = tuple([0] + LOTS)
        r = score(rows, full)
        if not r:
            print(f"{name}: not scorable")
            continue
        rng = np.random.default_rng(9)
        bs = sorted(score([rows[i] for i in rng.integers(0, len(rows), len(rows))],
                          full, seed=b)["gain"] for b in range(200))
        null = luce_null(rows, full, reps)
        med = float(np.median(null)) if len(null) else float("nan")
        pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1) if len(null) else float("nan")
        print(f"\n{name}: {len(rows)} usable observations, {r['n_cal']} on the full menu, "
              f"{r['n_scored']} scored on smaller menus")
        print(f"  renormalization {r['luce']:.4f}   race {r['race']:.4f}   "
              f"gain {r['gain']:+.4f}  [{bs[5]:+.4f}, {bs[194]:+.4f}]")
        print(f"  Luce null median {med:+.4f}   excess {r['gain']-med:+.4f}   "
              f"MC tail {pv:.3f}  ({len(null)} reps)")


if __name__ == "__main__":
    main()
