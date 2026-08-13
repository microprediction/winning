"""How much of the available restriction signal does each account capture?

A log-loss difference of 0.002 nats sounds negligible, and taken as an effect size
it is. But it is the wrong denominator. Both accounts receive the same first-place
shares, and those shares carry almost all of the predictive content of a restricted
menu; what the restriction map contributes is a thin residual on top. The question
worth asking is how much of that residual each account captures.

Three forecasts are compared on held-out respondents, over every subset of size two
or more:

  renormalization  uses the training-fold first-place shares and assumes odds are
                   preserved. Zero extra information.
  Gaussian race    uses the same shares and assumes a contest. Zero extra
                   information.
  saturated        uses the training-fold frequency of each item being top within
                   each subset. This is a legitimate forecast, not an oracle over
                   the test data, but it has one parameter per subset per item and
                   therefore represents what a perfect restriction map could deliver
                   given the sample.

The quantity of interest is the fraction of attainable improvement,

    (LL_renorm - LL_race) / (LL_renorm - LL_saturated),

which asks how far the race travels from renormalization toward the best any
restriction map could do. A small absolute gain can be a large fraction, and that
fraction is what bears on the generating process.

Usage:  python signal_share.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all

FLOOR = 1e-6
MAX_RESP = 5000


def run(R, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    fold = np.array_split(rng.permutation(n), folds)
    tot = {"renorm": 0.0, "race": 0.0, "sat": 0.0}
    cnt = 0
    for f in range(folds):
        test = fold[f]
        train = np.concatenate([fold[g] for g in range(folds) if g != f])
        p = np.bincount(R[train].argmin(axis=1), minlength=K) / len(train)
        if (p <= 0).any():
            return None
        a, err = calibrate_np(list(p))
        if err > 0.05:
            return None
        for r in range(2, K + 1):
            for S in itertools.combinations(range(K), r):
                idx = list(S)
                lz = p[idx].sum()
                renorm = np.maximum(p[idx] / lz, FLOOR)
                w = win_probs_np(a[idx])
                race = np.maximum(w / w.sum(), FLOOR)
                # saturated: training-fold frequency of being top within S
                twin = R[np.ix_(train, idx)].argmin(axis=1)
                sat = np.bincount(twin, minlength=len(idx)) / len(train)
                sat = np.maximum(sat, FLOOR)
                sat = sat / sat.sum()
                win = R[np.ix_(test, idx)].argmin(axis=1)
                tot["renorm"] += -np.log(renorm[win]).sum()
                tot["race"] += -np.log(race[win]).sum()
                tot["sat"] += -np.log(sat[win]).sum()
                cnt += len(win)
    for k in tot:
        tot[k] /= cnt
    avail = tot["renorm"] - tot["sat"]
    got = tot["renorm"] - tot["race"]
    return {"n": n, "K": K, "renorm": tot["renorm"], "race": tot["race"],
            "sat": tot["sat"], "avail": avail, "got": got,
            "frac": got / avail if avail > 0 else float("nan")}


def main():
    data = load_all()
    print("Held-out log loss per prediction, and the share of attainable")
    print("improvement the race captures.\n")
    print(f"{'dataset':<22}{'renorm':>9}{'race':>9}{'saturated':>11}"
          f"{'available':>11}{'captured':>10}{'share':>8}")
    fracs = []
    for name, R in sorted(data.items()):
        r = run(R)
        if not r:
            print(f"{name:<22}  not scorable")
            continue
        fracs.append(r["frac"])
        print(f"{name:<22}{r['renorm']:>9.4f}{r['race']:>9.4f}{r['sat']:>11.4f}"
              f"{r['avail']:>11.4f}{r['got']:>10.4f}{100*r['frac']:>7.0f}%")
    if fracs:
        fracs.sort()
        print(f"\nmedian share of attainable improvement captured by the race: "
              f"{100*fracs[len(fracs)//2]:.0f}%")
        print("renormalization captures 0% of it by construction.")


if __name__ == "__main__":
    main()
