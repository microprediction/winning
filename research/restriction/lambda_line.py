"""Where each collection sits between the two maps, on one axis.

Both maps are points on a single line. Define the contraction slope on pairs by

    delta_ij = logit(q_ij) - log(p_i / p_j),   i the higher-share alternative,

fitted through the origin as delta = -lambda log(p_i/p_j), where p is the full-menu share
vector and q_ij the share preferring i when only {i,j} is offered. Linear renormalization
predicts lambda = 0 by construction. Case V predicts a positive lambda that depends on the
share vector, so the natural coordinate is the ratio

    r = observed lambda / Case V lambda,

which puts linear renormalization at r = 0 and Gaussian renormalization at r = 1. A collection
with r below 1 contracts less than Case V says and sits between the two maps; above 1 it
contracts more, and the Gaussian point is not far enough along the family for it.

Only collections that supply binary choices over a named pair can be placed. The ranking
collections supply them by construction. Of the observed-restriction collections, five do: Yeon
and Rahnev in both experiments, Wills, the Rouder condition that restricts twelve lines to named
pairs, and the recognition foils. Tones, Getty and Townsend and Landon restrict to sets of three
or more, so no pairwise q exists and they cannot be placed; their verdict is the held-out log loss
instead. The consumer and lottery experiments do observe pair menus and could be placed with a
loader for their subset files, which is not written yet.

Usage:  python lambda_line.py
"""
import collections
import csv
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

ALPHA = 0.5
EPS = 1e-9


def slope(pairs):
    """Fit delta = -lambda log(p_i/p_j) through the origin. pairs are (L, q) with L>0."""
    num = den = 0.0
    for L, q in pairs:
        q = min(max(q, EPS), 1 - EPS)
        d = np.log(q / (1 - q)) - L
        num += -d * L
        den += L * L
    return num / den if den > 0 else float("nan")


def both_slopes(cells):
    """cells are (p_row, i, j, q): the full-menu row for one stimulus, a named pair from
    that row with p_i >= p_j, and the observed share preferring i on the restricted menu.

    The row matters. Pooling responses across stimuli would collapse the confusion
    structure and give a marginal distribution that no restricted menu was drawn from.
    """
    obs_pairs, race_pairs = [], []
    for p, i, j, q in cells:
        L = float(np.log(p[i] / p[j]))
        if L <= 1e-6:
            continue
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        obs_pairs.append((L, q))
        w = win_probs_np(a[[i, j]])
        race_pairs.append((L, float(w[0] / w.sum())))
    if len(obs_pairs) < 3:
        return None
    lo, lr = slope(obs_pairs), slope(race_pairs)
    return {"observed": lo, "race": lr, "ratio": lo / lr if lr else float("nan"),
            "pairs": len(obs_pairs)}


# ------------------------------------------------------------------ Yeon and Rahnev
def yeonrahnev(exp, K, item):
    D = HERE / "data" / "yeonrahnev" / "tidy"
    # one full-menu row per dominant stimulus, pooled over subjects
    rows = collections.defaultdict(lambda: np.zeros(K))
    with open(D / f"exp{exp}_full_menu_counts.csv") as f:
        for r in csv.DictReader(f):
            rows[int(r[f"dominant_{item}"]) - 1][int(r["response"]) - 1] += float(r["n"])
    pair = collections.defaultdict(lambda: np.zeros(2))
    with open(D / f"exp{exp}_pair_menu_counts.csv") as f:
        for r in csv.DictReader(f):
            key = (int(r[f"dominant_{item}"]) - 1, int(r[f"alternative_{item}"]) - 1)
            pair[key] += np.array([float(r["n_correct"]), float(r["n_wrong"])])
    cells = []
    for (dom, alt), n in pair.items():
        if n.sum() < 20 or dom not in rows:
            continue
        c = rows[dom]
        p = (c + ALPHA) / (c.sum() + ALPHA * K)
        q_dom = n[0] / n.sum()
        if p[dom] >= p[alt]:
            cells.append((p, dom, alt, q_dom))
        else:
            cells.append((p, alt, dom, 1 - q_dom))
    return both_slopes(cells)


# ------------------------------------------------------------------ Rouder condition C
PAIRS_C = [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9),
           (8, 10), (9, 11), (0, 10), (1, 11)]


def rouder_c():
    K = 12
    rows = collections.defaultdict(lambda: np.zeros(K))
    pair = collections.defaultdict(lambda: np.zeros(2))
    for path in sorted(glob.glob(str(HERE / "data" / "rouder_chunk" / "c0" / "C1CS*"))):
        blocks = collections.defaultdict(list)
        for line in open(path):
            t = line.split()
            if len(t) != 7:
                continue
            blocks[int(t[1][3:])].append((int(t[4]), int(t[5])))
        for trials in blocks.values():
            seen = {x for st, rp in trials for x in (st, rp)}
            if len(seen) > 4:                       # a full-twelve block
                for st, rp in trials:
                    if 0 <= st < K and 0 <= rp < K:
                        rows[st][rp] += 1
                continue
            fits = [q for q in PAIRS_C if seen <= set(q)]
            if len(fits) != 1:
                continue
            i, j = fits[0]
            for st, rp in trials:
                if st not in (i, j) or rp not in (i, j):
                    continue
                pair[(st, i, j)][0 if rp == st else 1] += 1
    cells = []
    for (st, i, j), n in pair.items():
        if n.sum() < 20 or rows[st].sum() < 20:
            continue
        other = j if st == i else i
        c = rows[st]
        p = (c + ALPHA) / (c.sum() + ALPHA * K)
        q_st = n[0] / n.sum()
        if p[st] >= p[other]:
            cells.append((p, st, other, q_st))
        else:
            cells.append((p, other, st, 1 - q_st))
    return both_slopes(cells)


# ------------------------------------------------------------------ Wills et al., 3 -> 2
def wills():
    """Master is the three-choice condition, restricted is the two-choice condition with the
    participant's `fixed` category disallowed. Cells are (fixed, catordist)."""
    from wills_twochoice import load
    data = load()
    master, restricted = data[1], data[2]
    agg_m, agg_r = collections.defaultdict(lambda: np.zeros(3)), collections.defaultdict(lambda: np.zeros(3))
    for subj in master.values():
        for cell, c in subj.items():
            agg_m[cell] += np.asarray(c, dtype=float)
    for subj in restricted.values():
        for cell, c in subj.items():
            agg_r[cell] += np.asarray(c, dtype=float)
    cells = []
    for cell, cm in agg_m.items():
        cr = agg_r.get(cell)
        if cr is None or cm.sum() < 20 or cr.sum() < 20:
            continue
        gone = cell[0] - 1                      # the disallowed category, 1-based in the file
        keep = [k for k in range(3) if k != gone]
        p = (cm + ALPHA) / (cm.sum() + ALPHA * 3)
        i, j = (keep if p[keep[0]] >= p[keep[1]] else keep[::-1])
        tot = cr[keep].sum()
        if tot < 20:
            continue
        cells.append((p, i, j, float(cr[i] / tot)))
    return both_slopes(cells)


# ------------------------------------------------------------------ recognition foils, 4 -> 2
CODE = {"hit": 0, "fa1": 1, "fa2": 2, "fa3": 3}


def recognition():
    D = HERE / "data" / "recognition"
    four = collections.defaultdict(lambda: np.zeros(4))
    with open(D / "4afc_exp1.csv") as f:
        for r in csv.DictReader(f):
            slot = CODE.get(r["response"])
            if slot is not None:
                four[r["target"]][slot] += 1
    two = collections.defaultdict(lambda: np.zeros(2))
    with open(D / "2afc_exp1.csv") as f:
        for r in csv.DictReader(f):
            k = r.get("foiltype") or r.get("foil.type")
            if k in ("foil1", "foil2", "foil3") and r["response"] in ("hit", "fa"):
                two[(r["target"], int(k[-1]))][0 if r["response"] == "hit" else 1] += 1
    cells = []
    for (tgt, k), n in two.items():
        c = four.get(tgt)
        if c is None or c.sum() < 20 or n.sum() < 10:
            continue
        p = (c + ALPHA) / (c.sum() + ALPHA * 4)
        q_hit = n[0] / n.sum()
        if p[0] >= p[k]:
            cells.append((p, 0, k, q_hit))
        else:
            cells.append((p, k, 0, 1 - q_hit))
    return both_slopes(cells)


# ------------------------------------------------------------------ consumer goods, 5 -> 2
def consumer(experiment):
    """Costa-Gomes et al.: choices from subsets of five goods, so the two-item menus are
    observed directly alongside the full five-item menu."""
    from menus_heldout import load as load_menus, ITEMS
    by, fc, exp = load_menus()
    K = len(ITEMS)
    full = tuple(range(K))
    subjects = [s for s in by if exp.get(s) == str(experiment)]
    cts = np.zeros(K)
    for s in subjects:
        c = by[s].get(full)
        if c is not None:
            cts[c] += 1
    if cts.sum() < 20:
        return None
    p = (cts + ALPHA) / (cts.sum() + ALPHA * K)
    pair_counts = collections.defaultdict(lambda: np.zeros(2))
    for s in subjects:
        for menu, ch in by[s].items():
            if len(menu) != 2:
                continue
            i, j = menu
            pair_counts[(i, j)][0 if ch == i else 1] += 1
    cells = []
    for (i, j), n in pair_counts.items():
        if n.sum() < 20:
            continue
        if p[i] >= p[j]:
            cells.append((p, i, j, float(n[0] / n.sum())))
        else:
            cells.append((p, j, i, float(n[1] / n.sum())))
    return both_slopes(cells)


# ------------------------------------------------------------------ lotteries, 6 -> 2
def lotteries(name):
    """Aguiar et al.: pooled (menu, chosen) rows over five lotteries plus an always-present
    default, so the two-alternative menus are observed alongside the full six."""
    from lotteries import load as load_lot
    rows, _ = load_lot(name)
    full = max((r[0] for r in rows), key=len)
    K = len(full)
    pos = {a: i for i, a in enumerate(full)}
    c = np.zeros(K)
    for menu, ch in rows:
        if menu == full:
            c[pos[ch]] += 1
    if c.sum() < 20:
        return None
    p = (c + ALPHA) / (c.sum() + ALPHA * K)
    pair_counts = collections.defaultdict(lambda: np.zeros(2))
    for menu, ch in rows:
        if len(menu) != 2:
            continue
        i, j = pos[menu[0]], pos[menu[1]]
        pair_counts[(i, j)][0 if pos[ch] == i else 1] += 1
    cells = []
    for (i, j), n in pair_counts.items():
        if n.sum() < 20:
            continue
        if p[i] >= p[j]:
            cells.append((p, i, j, float(n[0] / n.sum())))
        else:
            cells.append((p, j, i, float(n[1] / n.sum())))
    return both_slopes(cells)


def main():
    rows = []
    for label, fn in [
            ("Colour naming, four lines", lambda: yeonrahnev(1, 4, "color")),
            ("Symbol naming, six symbols", lambda: yeonrahnev(2, 6, "symbol")),
            ("Line lengths, twelve to pairs", rouder_c),
            ("Wills categories, three to two", wills),
            ("Recognition foils, four to two", recognition),
            ("Consumer goods, experiment 1", lambda: consumer(1)),
            ("Consumer goods, experiment 2", lambda: consumer(2)),
            ("Gambles, high arithmetic cost", lambda: lotteries("menu_choice_pooled.csv"))]:
        try:
            out = fn()
        except Exception as e:
            print(f"  {label}: {type(e).__name__} {e}")
            continue
        if out is None:
            print(f"  {label}: too few usable pairs")
            continue
        rows.append((label, out))

    print(f"{'collection':<32}{'pairs':>6}{'observed':>10}{'Case V':>9}{'ratio':>8}")
    for label, o in rows:
        print(f"{label:<32}{o['pairs']:>6}{o['observed']:>10.3f}{o['race']:>9.3f}"
              f"{o['ratio']:>8.2f}")
    print("\nRatio 0 is linear renormalization, 1 is Gaussian renormalization. Below 1 the")
    print("collection sits between the two maps; above 1 it contracts more than Case V says.")
    print("The ranking collections are in results/lambda_table.txt on the same statistic.")


if __name__ == "__main__":
    main()
