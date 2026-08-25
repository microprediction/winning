"""Three things the reviews asked for, on one pass over the ranking collections.

  proper restrictions   The published score averages over every subset of size two or
                        more, which includes T=S. There the two maps coincide and the
                        gain is exactly zero by construction, so the full menu is a
                        guaranteed tie mixed into the average. It is 1 subset in 1,013
                        at K=10 and 1 in 4 at K=3, so it dilutes small-K collections
                        far more than large ones and distorts every cross-collection
                        magnitude. Here the estimand is restrictions only, 2 <= |T| < K,
                        with the old figure printed beside it.

  size vector           Three menu weightings are three summaries of one curve. The
                        curve is (g_2, ..., g_{K-1}), and within-size dispersion is not
                        reported anywhere, though Getty shows a sign reversal across
                        subsets inside one experiment.

  Brier                 Log loss is unbounded below and rewards exactly the shape the
                        advantage has: rare large gains against frequent small losses.
                        A bounded proper score is the check on whether the sign is
                        tail insurance or better transport. Multiclass Brier,
                        sum_j (q_j - y_j)^2, on the same subsets and the same folds.

Usage:  python estimand.py [datasets...]
"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all, FLOOR, MAX_RESP, ALPHA


def predictions(p):
    """Both maps on every subset of size two or more, including the full menu."""
    K = len(p)
    a, err = calibrate_np(list(p))
    preds = {}
    for r in range(2, K + 1):
        for S in itertools.combinations(range(K), r):
            idx = list(S)
            luce = np.maximum(np.array([p[i] for i in idx]), FLOOR)
            luce = luce / luce.sum()
            w = win_probs_np(np.asarray(a)[idx])
            race = np.maximum(w / w.sum(), FLOOR)
            preds[S] = (luce, race)
    return preds, err


def score(R, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    if K < 3:
        return None
    fold = np.array_split(rng.permutation(n), folds)

    # per respondent, restrictions only, for the bootstrap
    Ll = np.zeros(n); Lr = np.zeros(n)          # log loss, linear and race
    Bl = np.zeros(n); Br = np.zeros(n)          # Brier, linear and race
    c = np.zeros(n)
    # per subset size, pooled over respondents
    sz_l = np.zeros(K + 1); sz_r = np.zeros(K + 1)
    szb_l = np.zeros(K + 1); szb_r = np.zeros(K + 1)
    sz_c = np.zeros(K + 1)
    nsub = 0

    for f in range(folds):
        test = fold[f]
        train = np.concatenate([fold[g] for g in range(folds) if g != f])
        cts = np.bincount(R[train].argmin(axis=1), minlength=K).astype(float)
        p = (cts + ALPHA) / (len(train) + ALPHA * K)
        if (p <= 0).any():
            return None
        preds, err = predictions(p)
        if err > 0.05:
            return None
        nsub = sum(1 for S in preds if len(S) < K)
        for S, (luce, race) in preds.items():
            m = len(S)
            win = R[np.ix_(test, list(S))].argmin(axis=1)
            ll = -np.log(luce[win])
            lr = -np.log(race[win])
            # multiclass Brier: (1 - q_win)^2 + sum of squares off the winner
            bl = (1.0 - luce[win]) ** 2 + (luce ** 2).sum() - luce[win] ** 2
            br = (1.0 - race[win]) ** 2 + (race ** 2).sum() - race[win] ** 2
            sz_l[m] += ll.sum(); sz_r[m] += lr.sum()
            szb_l[m] += bl.sum(); szb_r[m] += br.sum()
            sz_c[m] += len(win)
            if m < K:                              # the estimand: restrictions only
                Ll[test] += ll; Lr[test] += lr
                Bl[test] += bl; Br[test] += br
                c[test] += 1

    if (c == 0).any():
        return None
    dlog = Ll / c - Lr / c
    dbri = Bl / c - Br / c

    def ci(d):
        r2 = random.Random(7)
        bs = sorted(float(np.mean(d[[r2.randrange(n) for _ in range(n)]]))
                    for _ in range(4000))
        return bs[100], bs[3900]

    sizes = [m for m in range(2, K + 1) if sz_c[m] > 0]
    gain_all = float(((sz_l[sizes] - sz_r[sizes]).sum()) / sz_c[sizes].sum())
    lo, hi = ci(dlog)
    blo, bhi = ci(dbri)
    return {
        "n": n, "K": K, "subsets": nsub,
        "luce": float((Ll / c).mean()), "race": float((Lr / c).mean()),
        "gain": float(dlog.mean()), "lo": lo, "hi": hi,
        "gain_all": gain_all,
        "brier_luce": float((Bl / c).mean()), "brier_race": float((Br / c).mean()),
        "brier_gain": float(dbri.mean()), "blo": blo, "bhi": bhi,
        "size_gain": {m: float((sz_l[m] - sz_r[m]) / sz_c[m]) for m in sizes},
    }


def main():
    wanted = sys.argv[1:]
    data = load_all()
    names = [k for k in sorted(data) if not wanted or k in wanted]
    rows = {}

    print("Restrictions only, 2 <= |T| < K. Gain is linear minus Gaussian, so positive")
    print("favours Gaussian renormalization. 'with T=S' is the published estimand.\n")
    print(f"{'dataset':<24}{'n':>6}{'K':>3}{'subs':>6}"
          f"{'gain':>9}{'95% CI':>21}{'with T=S':>10}{'ratio':>7}")
    for name in names:
        r = score(data[name])
        if not r:
            print(f"{name:<24}  not scorable")
            continue
        rows[name] = r
        ratio = r["gain"] / r["gain_all"] if r["gain_all"] else float("nan")
        print(f"{name:<24}{r['n']:>6}{r['K']:>3}{r['subsets']:>6}"
              f"{r['gain']:>+9.4f}   [{r['lo']:+.4f}, {r['hi']:+.4f}]"
              f"{r['gain_all']:>+10.4f}{ratio:>7.2f}", flush=True)

    print("\n\nBounded proper score. Multiclass Brier on the same subsets and folds,")
    print("so a positive Brier gain is Gaussian renormalization ahead there too.\n")
    print(f"{'dataset':<24}{'log gain':>10}{'Brier linear':>14}{'Brier Gaussian':>16}"
          f"{'Brier gain':>12}{'95% CI':>23}")
    agree = 0
    for name in names:
        r = rows.get(name)
        if not r:
            continue
        agree += (r["brier_gain"] > 0) == (r["gain"] > 0)
        print(f"{name:<24}{r['gain']:>+10.4f}{r['brier_luce']:>14.4f}"
              f"{r['brier_race']:>16.4f}{r['brier_gain']:>+12.5f}"
              f"   [{r['blo']:+.5f}, {r['bhi']:+.5f}]", flush=True)
    print(f"\nthe two scores agree on sign in {agree} of {len(rows)} collections")

    print("\n\nGain by size of the surviving menu. The three menu weightings are three")
    print("summaries of this curve; |T| = K is the tie the estimand now excludes.\n")
    width = max(r["K"] for r in rows.values()) if rows else 0
    head = "".join(f"{m:>9}" for m in range(2, width + 1))
    print(f"{'dataset':<24}{head}")
    for name in names:
        r = rows.get(name)
        if not r:
            continue
        cells = "".join(
            f"{r['size_gain'][m]:>+9.4f}" if m in r["size_gain"] else " " * 9
            for m in range(2, width + 1))
        print(f"{name:<24}{cells}", flush=True)


if __name__ == "__main__":
    main()
