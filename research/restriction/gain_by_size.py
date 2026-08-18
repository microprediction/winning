"""Held-out gain decomposed by subset size.

The all-subsets aggregate weights each subset equally, so intermediate cardinalities
dominate simply because there are more of them. That is a defensible estimand but it
is not the one the title emphasises, which is pairwise. This reports the gain
separately for each |T|, so the pairwise number is visible on its own and any
dependence on menu size is visible too.

Usage:  python gain_by_size.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all, MAX_RESP, ALPHA, FLOOR


def by_size(R, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    fold = np.array_split(rng.permutation(n), folds)
    num = {r: 0.0 for r in range(2, K + 1)}
    cnt = {r: 0 for r in range(2, K + 1)}
    for f in range(folds):
        test = fold[f]
        train = np.concatenate([fold[g] for g in range(folds) if g != f])
        cts = np.bincount(R[train].argmin(axis=1), minlength=K).astype(float)
        p = (cts + ALPHA) / (len(train) + ALPHA * K)
        a, err = calibrate_np(list(p))
        if err > 0.05:
            return None
        Rt = R[test]
        for r in range(2, K + 1):
            for S in itertools.combinations(range(K), r):
                idx = list(S)
                luce = np.maximum(p[idx] / p[idx].sum(), FLOOR)
                w = win_probs_np(a[idx])
                race = np.maximum(w / w.sum(), FLOOR)
                win = Rt[:, idx].argmin(axis=1)
                num[r] += (-np.log(luce[win]) + np.log(race[win])).sum()
                cnt[r] += len(win)
    return {r: num[r] / cnt[r] for r in num if cnt[r]}


def main():
    data = load_all()
    sizes = range(2, 11)
    print(f"{'dataset':<22}" + "".join(f"{'|T|='+str(r):>9}" for r in sizes))
    for name, R in sorted(data.items()):
        g = by_size(R)
        if not g:
            print(f"{name:<22}  not scorable")
            continue
        cells = "".join(f"{g[r]:>+9.4f}" if r in g else f"{'':>9}" for r in sizes)
        print(f"{name:<22}{cells}", flush=True)
    print("\npositive favours the race; the |T|=2 column is the pairwise gain alone")


if __name__ == "__main__":
    main()
