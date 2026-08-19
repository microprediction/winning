"""Restriction on a menu the observer could not have anticipated.

Yeon and Rahnev (2020) show a display whose dominant colour (Experiment 1, four colours)
or dominant symbol (Experiment 2, six symbols) has to be named. In one condition the full
menu is available. In another the observer sees the identical display and only AFTER it
has gone is told that the answer is one of two named alternatives. The stimulus, the
viewing time and the observer are the same; the only thing that changes is which
alternatives survive. That is this paper's question asked directly, with the restricted
menu observed rather than induced from a ranking, and with no chance for the observer to
prepare for the smaller menu.

Experiment 1 also ran the same pairs ANNOUNCED IN ADVANCE. That arm is not a restriction
test -- the observer can point attention at the pair -- and it is reported here as the
control it is.

Calibration uses the full-menu row for that observer and that dominant colour alone.
The two-alternative cells on the same observer and colour are then predicted and scored.
Each cell carries its own null, drawn under the renormalisation prediction at the observed
cell total, so the comparison is against fitted Luce rather than against zero.

Usage:  python yeonrahnev.py [n_null_reps]
"""
import collections
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

DATA = HERE / "data" / "yeonrahnev" / "tidy"
FLOOR = 1e-6
ALPHA = 0.5
MIN_FULL = 20
MIN_PAIR = 5


def load(full_name, pair_name, k, item):
    """item is "color" in Experiment 1 and "symbol" in Experiment 2."""
    full = collections.defaultdict(lambda: np.zeros(k))
    with open(DATA / full_name) as f:
        for r in csv.DictReader(f):
            full[(int(r["subject"]), int(r[f"dominant_{item}"]))][int(r["response"]) - 1] += float(r["n"])
    pair = {}
    with open(DATA / pair_name) as f:
        for r in csv.DictReader(f):
            key = (int(r["subject"]), int(r[f"dominant_{item}"]), int(r[f"alternative_{item}"]))
            pair[key] = np.array([float(r["n_correct"]), float(r["n_wrong"])])
    return full, pair


def build(full, pair, k):
    """Cells grouped by the calibration row they share, so the null can resample it."""
    groups = []
    for (sub, dom), c in sorted(full.items()):
        if c.sum() < MIN_FULL:
            continue
        p = (c + ALPHA) / (c.sum() + ALPHA * k)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        cells = []
        for alt in range(1, k + 1):
            if alt == dom:
                continue
            o = pair.get((sub, dom, alt))
            if o is None or o.sum() < MIN_PAIR:
                continue
            idx = [dom - 1, alt - 1]
            u = p[idx] / p[idx].sum()
            w = win_probs_np(a[idx])
            cells.append({"lu": np.maximum(u, FLOOR),
                          "ra": np.maximum(w / w.sum(), FLOOR),
                          "u": u, "o": o, "n": int(o.sum()), "sub": sub, "idx": idx})
        if cells:
            groups.append({"p": p, "n_full": int(c.sum()), "sub": sub, "cells": cells})
    return groups


def score(cells):
    tl = tg = 0.0
    n = 0
    for c in cells:
        if c["o"].sum() <= 0:
            continue
        q = c["o"] / c["o"].sum()
        tl += float(-(q * np.log(c["lu"])).sum())
        tg += float(-(q * np.log(c["ra"])).sum())
        n += 1
    if n < 10:
        return None
    return {"cells": n, "luce": tl / n, "race": tg / n, "gain": (tl - tg) / n}


def null_rep(groups, k, rng):
    """One draw from the fitted-Luce null.

    Truth is a Luce process with the observed smoothed full-menu shares as worths. The
    full-menu row is resampled at its own count and both maps are recalibrated from it,
    so the null carries calibration noise as well as sampling noise in the pairs. The
    cheaper null that resamples only the pair counts credits the race with more than it
    should; this one does not.
    """
    out = []
    for g in groups:
        c = rng.multinomial(g["n_full"], g["p"]).astype(float)
        p = (c + ALPHA) / (c.sum() + ALPHA * k)
        a, err = calibrate_np(list(p))
        a = np.asarray(a)
        for cell in g["cells"]:
            idx = cell["idx"]
            lu = p[idx] / p[idx].sum()
            w = win_probs_np(a[idx])
            out.append({"lu": np.maximum(lu, FLOOR),
                        "ra": np.maximum(w / w.sum(), FLOOR),
                        "o": rng.multinomial(cell["n"], cell["u"]).astype(float)})
    return out


def report(label, groups, k, reps, seed=0):
    cells = [c for g in groups for c in g["cells"]]
    r = score(cells)
    if not r:
        print(f"{label}: too few cells")
        return
    rng = np.random.default_rng(seed)
    null = np.array(sorted(score(null_rep(groups, k, rng))["gain"] for _ in range(reps)))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    subs = sorted({g["sub"] for g in groups})
    bs = []
    for _ in range(400):
        pick = collections.Counter(np.array(subs)[rng.integers(0, len(subs), len(subs))])
        rep = [c for g in groups for _ in range(pick[g["sub"]]) for c in g["cells"]]
        s = score(rep)
        if s:
            bs.append(s["gain"])
    bs = sorted(bs)
    ci = f"[{bs[int(.025 * len(bs))]:+.4f}, {bs[int(.975 * len(bs))]:+.4f}]" if len(bs) > 30 else ""
    print(f"{label:<30}{r['cells']:>6}{r['luce']:>9.4f}{r['race']:>9.4f}"
          f"{r['gain']:>+9.4f}  {ci:<22}{med:>+9.4f}{r['gain'] - med:>+9.4f}{pv:>8.3f}")


def accuracies(label, groups):
    """Where each account sits relative to what the observer actually did."""
    cells = [c for g in groups for c in g["cells"]]
    w = np.array([c["n"] for c in cells], dtype=float)
    obs = np.array([c["o"][0] / c["o"].sum() for c in cells])
    lu = np.array([c["lu"][0] for c in cells])
    ra = np.array([c["ra"][0] for c in cells])
    m = lambda x: float((x * w).sum() / w.sum())
    print(f"{label:<30}{m(obs):>9.4f}{m(lu):>9.4f}{m(ra):>9.4f}"
          f"{m(lu) - m(obs):>+11.4f}{m(ra) - m(obs):>+11.4f}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    arms = []
    f1, p1 = load("exp1_full_menu_counts.csv", "exp1_pair_menu_counts.csv", 4, "color")
    arms.append(("exp 1, 4 colours, after", build(f1, p1, 4), 4))
    _, p1a = load("exp1_full_menu_counts.csv", "exp1_pair_menu_advance_counts.csv", 4, "color")
    arms.append(("exp 1, same pairs, before", build(f1, p1a, 4), 4))
    f2, p2 = load("exp2_full_menu_counts.csv", "exp2_pair_menu_counts.csv", 6, "symbol")
    arms.append(("exp 2, 6 symbols, after", build(f2, p2, 6), 6))

    print(f"{'arm':<30}{'cells':>6}{'renorm':>9}{'race':>9}{'gain':>9}"
          f"  {'subject bootstrap':<22}{'null':>9}{'excess':>9}{'tail':>8}")
    for i, (label, groups, k) in enumerate(arms):
        report(label, groups, k, reps, seed=i)

    print(f"\n{'arm':<30}{'observed':>9}{'renorm':>9}{'race':>9}"
          f"{'renorm err':>11}{'race err':>11}")
    for label, groups, _ in arms:
        accuracies(label, groups)


if __name__ == "__main__":
    main()
