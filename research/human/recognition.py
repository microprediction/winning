"""Restriction on memory foils: remove candidates, and the recogniser redistributes.

Utochkin, Azarov and Grigorev show observers a studied photograph among foils. In the
four-alternative arm the menu is {target, foil1, foil2, foil3} and the response code names
which was chosen. In the two-alternative arm the menu is {target, foil_k} for one k, named
by foil.type. Same targets, same foil images, so the smaller menu is a strict subset of the
larger by construction.

The competing alternatives are memory representations rather than goods, which is the case
this paper most wants: nobody describes recognising an object as a contest, yet the latent
structure is competing identifications.

Calibration uses the four-alternative shares alone, per target, and the two-alternative
trials on the same target are then predicted and scored. Foil1 is another exemplar of the
target's category while foils 2 and 3 are cross-category, so results are also split by that.

Two arms are different participants with different inclusion thresholds, so the fitted-Luce
null runs on the observed trial counts to absorb what that costs.

Usage:  python recognition.py [n_null_reps]
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

DATA = HERE / "data" / "recognition"
FLOOR = 1e-6
ALPHA = 0.5
CODE = {"hit": 0, "fa1": 1, "fa2": 2, "fa3": 3}


def load():
    four = collections.defaultdict(collections.Counter)     # target -> chosen slot
    two = collections.defaultdict(collections.Counter)      # (target,k) -> hit/fa
    with open(DATA / "4afc_exp1.csv") as f:
        for r in csv.DictReader(f):
            s = CODE.get(r["response"])
            if s is not None:
                four[r["target"]][s] += 1
    with open(DATA / "2afc_exp1.csv") as f:
        for r in csv.DictReader(f):
            k = r.get("foiltype") or r.get("foil.type")
            if k in ("foil1", "foil2", "foil3") and r["response"] in ("hit", "fa"):
                two[(r["target"], int(k[-1]))][r["response"]] += 1
    return four, two


def build(four, two, min_four=20, min_two=10):
    cells = []
    for tgt, cnt in four.items():
        n4 = sum(cnt.values())
        if n4 < min_four or len(cnt) < 2:
            continue
        c = np.array([cnt.get(s, 0) for s in range(4)], dtype=float)
        p = (c + ALPHA) / (c.sum() + ALPHA * 4)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        for k in (1, 2, 3):
            obs = two.get((tgt, k))
            if not obs or sum(obs.values()) < min_two:
                continue
            idx = [0, k]
            lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
            w = win_probs_np(a[idx])
            ra = np.maximum(w / w.sum(), FLOOR)
            o = np.array([obs.get("hit", 0), obs.get("fa", 0)], dtype=float)
            cells.append((lu, ra, p[idx] / p[idx].sum(), int(o.sum()), o, k, tgt))
    return cells


def score(cells, obs_list=None):
    tl = tg = 0.0
    n = 0
    for i, (lu, ra, u, tot, o0, k, tgt) in enumerate(cells):
        o = o0 if obs_list is None else obs_list[i]
        if o.sum() <= 0:
            continue
        q = o / o.sum()
        tl += float(-(q * np.log(lu)).sum())
        tg += float(-(q * np.log(ra)).sum())
        n += 1
    if n < 10:
        return None
    return {"cells": n, "luce": tl / n, "race": tg / n, "gain": (tl - tg) / n}


def report(label, cells, reps, seed=0):
    r = score(cells)
    if not r:
        print(f"{label}: too few cells")
        return
    rng = np.random.default_rng(seed)
    null = [score(cells, [rng.multinomial(t, u).astype(float)
                          for _, _, u, t, _, _, _ in cells])["gain"]
            for _ in range(reps)]
    null = np.array(sorted(null))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    tg = sorted({c[6] for c in cells})
    bs = []
    for _ in range(300):
        pick = set(np.array(tg, dtype=object)[rng.integers(0, len(tg), len(tg))])
        s = score([c for c in cells if c[6] in pick])
        if s:
            bs.append(s["gain"])
    bs = sorted(bs)
    ci = (f"[{bs[int(.025*len(bs))]:+.4f}, {bs[int(.975*len(bs))]:+.4f}]"
          if len(bs) > 30 else "")
    print(f"{label:<26}{r['cells']:>6}{r['luce']:>9.4f}{r['race']:>9.4f}"
          f"{r['gain']:>+9.4f}  {ci:<20}{med:>+9.4f}{r['gain']-med:>+9.4f}{pv:>8.3f}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    four, two = load()
    cells = build(four, two)
    print(f"{len(four)} targets in the four-alternative arm, {len(cells)} nested cells\n")
    print(f"{'split':<26}{'cells':>6}{'renorm':>9}{'race':>9}{'gain':>9}"
          f"  {'target bootstrap':<20}{'null':>9}{'excess':>9}{'tail':>8}")
    report("all foils", cells, reps)
    report("foil1, same category", [c for c in cells if c[5] == 1], reps, 1)
    report("foils 2 and 3, other", [c for c in cells if c[5] in (2, 3)], reps, 2)


if __name__ == "__main__":
    main()
