"""Restriction with the stimuli held fixed and only the responses withdrawn.

Getty, Swets, Swets and Green (1979) had three observers identify eight complex sounds.
The 8 by 8 experiment allows all eight responses. The 8 by 4 experiment presents the same
eight stimuli but allows only four responses, a different four in each of three
conditions, and the labels are the stimulus numbers, so the smaller menus nest inside the
larger by construction. Nothing about the stimulus changes; only the menu does. That is a
cleaner isolation of the response-set manipulation than any other source in the corpus.

Two splits matter. When the presented stimulus is one of the four allowed responses (a
"signal" row) the full-menu favourite survives the restriction. When it is not, the
favourite has been withdrawn and every response is an error, which is the case where
Table 5 of the paper says the race should gain most.

The authors' own conclusion is a warning about what this can show. They report that
observers retuned the weights they placed on perceptual dimensions to maximise
discriminability of whichever subset they were asked to identify, aided by the feedback
given only on that subset. If restriction retunes the representation then the surviving
alternatives are not the same alternatives, no fixed-representation map can be right, and
this is a boundary case of the quality-changing kind rather than a test between the maps.
The numbers below are reported in that light.

Data transcribed from Tables 6 and 8 of the article; every row sum reproduces the printed
total, which is the transcription check.

Usage:  python getty.py [n_null_reps]
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

DATA = HERE / "data" / "getty"
FLOOR = 1e-6
ALPHA = 0.5
K = 8


def load():
    master = {}
    with open(DATA / "master_8x8.csv") as f:
        for r in csv.DictReader(f):
            master[(r["observer"], int(r["stimulus"]))] = np.array(
                [float(r[f"r{j}"]) for j in range(1, K + 1)])
    restricted = []
    with open(DATA / "restricted_8x4.csv") as f:
        for r in csv.DictReader(f):
            restricted.append({
                "observer": r["observer"],
                "condition": int(r["condition"]),
                "signals": [int(s) for s in r["signals"].split("|")],
                "stimulus": int(r["stimulus"]),
                "obs": np.array([float(r[f"n{j}"]) for j in range(1, 5)]),
            })
    return master, restricted


def build(master, restricted):
    groups = collections.defaultdict(list)
    for row in restricted:
        c = master[(row["observer"], row["stimulus"])]
        p = (c + ALPHA) / (c.sum() + ALPHA * K)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        idx = [s - 1 for s in row["signals"]]
        u = p[idx] / p[idx].sum()
        w = win_probs_np(a[idx])
        groups[(row["observer"], row["stimulus"])].append({
            "lu": np.maximum(u, FLOOR), "ra": np.maximum(w / w.sum(), FLOOR),
            "u": u, "o": row["obs"], "n": int(row["obs"].sum()),
            "signal": row["stimulus"] in row["signals"],
            "condition": row["condition"], "observer": row["observer"],
            "p": p, "n_full": int(c.sum()), "idx": idx,
        })
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
    if n < 4:
        return None
    return {"cells": n, "luce": tl / n, "race": tg / n, "gain": (tl - tg) / n}


def null_rep(cells, rng):
    out = []
    for c in cells:
        d = rng.multinomial(c["n_full"], c["p"]).astype(float)
        p = (d + ALPHA) / (d.sum() + ALPHA * K)
        a = np.asarray(calibrate_np(list(p))[0])
        lu = p[c["idx"]] / p[c["idx"]].sum()
        w = win_probs_np(a[c["idx"]])
        out.append({"lu": np.maximum(lu, FLOOR), "ra": np.maximum(w / w.sum(), FLOOR),
                    "o": rng.multinomial(c["n"], c["u"]).astype(float)})
    return out


def report(label, cells, reps, seed=0):
    r = score(cells)
    if not r:
        print(f"{label:<34} too few cells")
        return
    rng = np.random.default_rng(seed)
    null = np.array(sorted(score(null_rep(cells, rng))["gain"] for _ in range(reps)))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    bs = []
    for _ in range(400):
        pick = rng.integers(0, len(cells), len(cells))
        s = score([cells[i] for i in pick])
        if s:
            bs.append(s["gain"])
    bs = sorted(bs)
    ci = f"[{bs[int(.025 * len(bs))]:+.4f}, {bs[int(.975 * len(bs))]:+.4f}]"
    print(f"{label:<34}{r['cells']:>5}{r['luce']:>9.4f}{r['race']:>9.4f}"
          f"{r['gain']:>+9.4f}  {ci:<22}{med:>+9.4f}{r['gain'] - med:>+9.4f}{pv:>8.3f}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    master, restricted = load()
    groups = build(master, restricted)
    cells = [c for v in groups.values() for c in v]
    print(f"{len(cells)} cells: 3 observers x 8 stimuli x 3 conditions\n")
    print(f"{'split':<34}{'cells':>5}{'renorm':>9}{'race':>9}{'gain':>9}"
          f"  {'cell bootstrap':<22}{'null':>9}{'excess':>9}{'tail':>8}")
    report("all rows", cells, reps, 0)
    report("signal rows, favourite survives", [c for c in cells if c["signal"]], reps, 1)
    report("non-signal, favourite removed", [c for c in cells if not c["signal"]], reps, 2)
    signals = {row["condition"]: row["signals"] for row in restricted}
    for cond in (1, 2, 3):
        sig = "".join(str(s) for s in signals[cond])
        report(f"condition {cond}, signals {{{sig}}}",
               [c for c in cells if c["condition"] == cond], reps, 2 + cond)

    print("\nWithin-set confusability, computed from the 8 by 8 alone and so available "
          "before any\nrestricted data is seen: of the errors a signal stimulus makes on "
          "the full menu, the\nfraction landing on another signal of that condition.\n")
    for cond in (1, 2, 3):
        S = signals[cond]
        num = den = 0.0
        for (obs, stim), c in master.items():
            if stim in S:
                num += sum(c[t - 1] for t in S if t != stim)
                den += c.sum() - c[stim - 1]
        print(f"  condition {cond}, signals {S}: {num / den:.3f}")
    print("\nThe cell bootstrap resamples cells, not observers; with three observers it "
          "understates\nuncertainty. Treat the intervals as indicative.")


if __name__ == "__main__":
    main()
