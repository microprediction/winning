"""Restriction on tone-identification labels, Stewart, Brown and Chater (2005).

Listeners hear one of N pure tones and name it. The set-size-8 condition uses the
middle eight tones of the ten and set-size-6 the middle six, so the label sets nest
over physically identical tones. For a given stimulus the N=10 row of the confusion
matrix is its full-menu response distribution; restricting to the middle labels and
predicting the N=6 or N=8 row is this paper's test, on competing identifications of a
tone rather than on preferences.

Both maps are calibrated from the N=10 row alone. Loss is weighted by the observed
restricted row. Matrices are participant-averaged and between-subjects, so there is
nothing to bootstrap; the numbers below are point comparisons.

Usage:  python tones.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

D = HERE / "data" / "tones"
FLOOR = 1e-6


def load(name):
    m = np.loadtxt(D / f"{name}.csv", delimiter=",")
    return m / m.sum(axis=1, keepdims=True)


def run(spacing, small):
    big = load(f"{spacing}_N10")
    obs = load(f"{spacing}_N{small}")
    off = (10 - small) // 2                    # middle labels of the ten
    keep = list(range(off, off + small))
    tl = tg = 0.0
    n = 0
    skipped = 0
    for i in range(small):
        if (big[i + off] == 0).any():
            # an exact zero has to be floored, which sends a location far out and makes the
            # inversion ill conditioned. Two independent calibrators disagree on such rows,
            # so they are excluded rather than reported at a precision neither supports.
            skipped += 1
            continue
        p = np.maximum(big[i + off], FLOOR)
        p = p / p.sum()
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        lu = np.maximum(p[keep] / p[keep].sum(), FLOOR)
        w = win_probs_np(a[keep])
        ra = np.maximum(w / w.sum(), FLOOR)
        o = obs[i]
        tl += float(-(o * np.log(lu)).sum())
        tg += float(-(o * np.log(ra)).sum())
        n += 1
    return n, tl / n, tg / n, skipped


print(f"{'condition':<22}{'rows':>5}{'renorm':>9}{'race':>9}{'gain':>9}")
tot = []
for spacing in ("narrow", "wide"):
    for small in (6, 8):
        n, l, g, sk = run(spacing, small)
        note = f"   ({sk} row dropped for a zero cell)" if sk else ""
        print(f"{spacing+' N10->N'+str(small):<22}{n:>5}{l:>9.4f}{g:>9.4f}{l-g:>+9.4f}{note}")
        tot.append(l - g)
print(f"\nmean gain over the four conditions {np.mean(tot):+.4f}, "
      f"race ahead in {sum(1 for t in tot if t > 0)} of 4")
