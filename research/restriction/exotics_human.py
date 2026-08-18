"""Exacta and trifecta pricing: where the discrepancy is large rather than small.

A log score averaged over every subset understates the difference between the two
accounts, for a reason that is structural rather than statistical. When shares are
equal the two predictions coincide exactly, by exchangeability, so every near-tied
pair contributes a difference of zero to the average. The discrepancy lives in
lopsided fields.

Ordered outcomes make it visible. Renormalization prices an exacta as

    P(i first, j second) = p_i * p_j / (1 - p_i),

which is Harville's formula. A contest prices it by removing i from the field and
re-running the race for second place. Both reproduce the win probabilities exactly,
so they differ only in the conditional, and the difference compounds as places are
added.

Complete rankings give the observed ordered probabilities directly, so the two
prices can be scored against the truth rather than against each other. This reports
the mispricing ratio between the two predictions, its dependence on how lopsided
the field is, and log loss on exactas and trifectas.

Usage:  python exotics_human.py
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all

FLOOR = 1e-9


def race_field(a, idx):
    w = win_probs_np(a[idx])
    return w / w.sum()


def analyse(R, name):
    n, K = R.shape
    top = R.argmin(axis=1)
    p = np.bincount(top, minlength=K) / n
    if (p <= 1e-6).any():
        return None
    a, err = calibrate_np(list(p))
    if err > 0.05:
        return None

    # observed ordered pair and triple frequencies
    order = R.argsort(axis=1)
    exa = {}
    tri = {}
    for row in order:
        exa[(row[0], row[1])] = exa.get((row[0], row[1]), 0) + 1
        tri[(row[0], row[1], row[2])] = tri.get((row[0], row[1], row[2]), 0) + 1

    rows = []
    for i in range(K):
        rest = [k for k in range(K) if k != i]
        race2 = race_field(a, rest)
        for pos, j in enumerate(rest):
            luce = p[i] * p[j] / (1.0 - p[i])
            cont = p[i] * race2[pos]
            obs = exa.get((i, j), 0) / n
            spread = abs(math.log(p[i] / p[j]))
            rows.append((obs, luce, cont, spread))

    trirows = []
    for i in range(K):
        rest = [k for k in range(K) if k != i]
        race2 = race_field(a, rest)
        for pj, j in enumerate(rest):
            rest2 = [k for k in rest if k != j]
            if not rest2:
                continue
            race3 = race_field(a, rest2)
            for pk, k in enumerate(rest2):
                luce = p[i] * (p[j] / (1 - p[i])) * (p[k] / (1 - p[i] - p[j]))
                cont = p[i] * race2[pj] * race3[pk]
                obs = tri.get((i, j, k), 0) / n
                trirows.append((obs, luce, cont))

    def logloss(rs):
        tot = 0.0
        w = 0.0
        for obs, lu, co in ((r[0], r[1], r[2]) for r in rs):
            if obs <= 0:
                continue
            tot += obs * math.log(max(co, FLOOR) / max(lu, FLOOR))
            w += obs
        return tot / w if w else float("nan")

    ratios = [r[2] / max(r[1], FLOOR) for r in rows]
    ratios.sort()
    m = len(ratios)
    # mispricing where the field is lopsided versus near-tied
    lop = [r for r in rows if r[3] > 1.0]
    tie = [r for r in rows if r[3] < 0.25]
    def relerr(rs, which):
        num = sum(abs(r[which] - r[0]) for r in rs)
        den = sum(r[0] for r in rs)
        return num / den if den else float("nan")
    return {
        "K": K, "n": n,
        "gain_exacta": logloss(rows),
        "gain_trifecta": logloss(trirows),
        "ratio_med": ratios[m // 2], "ratio_lo": ratios[m // 20],
        "ratio_hi": ratios[-max(1, m // 20)],
        "tie_luce": relerr(tie, 1) if tie else float("nan"),
        "tie_cont": relerr(tie, 2) if tie else float("nan"),
        "lop_luce": relerr(lop, 1) if lop else float("nan"),
        "lop_cont": relerr(lop, 2) if lop else float("nan"),
    }


def main():
    data = load_all()
    print("Contest price divided by Harville price for the same exacta,")
    print("and mean relative pricing error against observed frequencies.\n")
    print(f"{'dataset':<22}{'K':>3}{'exacta gain':>12}{'trifecta':>10}"
          f"{'ratio 5-50-95%':>24}")
    for name, R in sorted(data.items()):
        r = analyse(R, name)
        if not r:
            print(f"{name:<22}  not scorable")
            continue
        print(f"{name:<22}{r['K']:>3}{r['gain_exacta']:>+12.4f}"
              f"{r['gain_trifecta']:>+10.4f}"
              f"   {r['ratio_lo']:.2f} / {r['ratio_med']:.2f} / {r['ratio_hi']:.2f}")
    print(f"\n{'dataset':<22}{'near-tied pairs':>22}{'lopsided pairs':>24}")
    print(f"{'':<22}{'Harville':>11}{'contest':>11}{'Harville':>12}{'contest':>12}")
    for name, R in sorted(data.items()):
        r = analyse(R, name)
        if not r:
            continue
        print(f"{name:<22}{r['tie_luce']:>11.3f}{r['tie_cont']:>11.3f}"
              f"{r['lop_luce']:>12.3f}{r['lop_cont']:>12.3f}")
    print("\nexacta and trifecta gains are mean log-probability advantages of the")
    print("contest per observed ordered outcome; positive favours the contest.")


if __name__ == "__main__":
    main()
