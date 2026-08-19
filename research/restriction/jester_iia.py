"""Humour preference: a third human construct for the odds-invariance statistic.

The human side of the comparison rested on sushi, Netflix and two perceptual
tasks. Jester adds humour, with a larger sample than all of them together.

Jester's design is unusual and convenient: every user rates a fixed gauge set of
ten jokes before seeing anything else, on a continuous scale from -10 to +10. So
24,983 people effectively supply a complete ranking of the same ten alternatives,
and both the full-set and pair-restricted population shares follow without any
imputation.

    delta_ij = -lambda * log(p_i/p_j)

lambda = 0 is Luce's axiom, lambda = 1 abandons the prior ranking. Reported
alongside the Case V contest prediction calibrated to the observed full-set
distribution, and with equal-rating ties handled two ways, because the election
data taught us that a tie convention can move this statistic by a factor of five.

Data: http://goldberg.berkeley.edu/jester-data/jester-data-1.zip (jester-data-1.xls,
24,983 users x 100 jokes, 99 codes an unrated joke).

Usage:  python jester_iia.py path/to/jester-data-1.xls
"""
import math
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/polysemy_pilot")
from exact_analyze import calibrate_np, win_probs_np

GAUGE = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]   # 1-indexed, rated by all users


def load(path):
    d = pd.read_excel(path, header=None)
    R = d.iloc[:, 1:].to_numpy(dtype=float)
    R[R == 99] = np.nan
    G = R[:, [g - 1 for g in GAUGE]]
    keep = ~np.isnan(G).any(axis=1)
    return G[keep]


def shares(G, cols, mode):
    """Population shares of 'best among cols', ties split or dropped."""
    sub = G[:, cols]
    mx = sub.max(axis=1, keepdims=True)
    top = sub == mx
    ntop = top.sum(axis=1)
    if mode == "strict":
        ok = ntop == 1
        sub, top, ntop = sub[ok], top[ok], ntop[ok]
    w = top / ntop[:, None]
    tot = w.sum()
    return (w.sum(axis=0) / tot, len(sub)) if tot > 0 else (None, 0)


def lam(pairs, seed=4, B=20000):
    num = sum(-d * L for L, d in pairs)
    den = sum(L * L for L, _ in pairs)
    est = num / den
    random.seed(seed)
    n, bs = len(pairs), []
    for _ in range(B):
        s = [pairs[random.randrange(n)] for _ in range(n)]
        de = sum(L * L for L, _ in s)
        if de > 0:
            bs.append(sum(-d * L for L, d in s) / de)
    bs.sort()
    return est, bs[int(.025 * len(bs))], bs[int(.975 * len(bs))]


def analyse(G, mode):
    K = G.shape[1]
    full, n = shares(G, list(range(K)), mode)
    if full is None:
        return None
    live = [i for i in range(K) if full[i] > 0]
    p = [full[i] for i in live]
    z = sum(p)
    p = [x / z for x in p]
    a_loc, err = calibrate_np(p)
    obs, cv = [], []
    for x in range(len(live)):
        for y in range(x + 1, len(live)):
            i, j = live[x], live[y]
            pair, m = shares(G, [i, j], mode)
            if pair is None or pair[0] <= 0 or pair[1] <= 0:
                continue
            L = math.log(full[i] / full[j])
            if abs(L) < 1e-6:
                continue
            obs.append((L, math.log(pair[0] / pair[1]) - L))
            w = win_probs_np(a_loc[[x, y]])
            if w[0] > 0 and w[1] > 0:
                cv.append((L, math.log(w[0] / w[1]) - L))
    return obs, cv, n, err


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "jester-data-1.xls"
    G = load(path)
    print(f"{G.shape[0]:,} users with complete ratings of the {G.shape[1]} gauge jokes")
    for mode, label in (("split", "equal ratings split"),
                        ("strict", "equal-rating ties dropped")):
        r = analyse(G, mode)
        if not r:
            continue
        obs, cv, n, err = r
        e, lo, hi = lam(obs)
        c = lam(cv)
        print(f"\n  {label} ({n:,} users, {len(obs)} pairs, calibration residual {err:.4f})")
        print(f"    observed lambda      {e:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"    Case V predicts      {c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
        print(f"    overshoot            {e/c[0]:.2f}x")
    print("\n  for comparison: human preference (sushi, Netflix) 0.420, "
          "human perception 0.188,")
    print("  machines 0.690 with Case V predicting 0.120, and the axiom at 0")


if __name__ == "__main__":
    main()
