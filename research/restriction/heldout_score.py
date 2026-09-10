"""Held-out log loss: score the two restriction maps as probability forecasts.

The contraction slope lambda summarises a dataset in one number, which projects a
vector of pair-specific predictions onto one direction and is not a proper scoring
rule. This scores the two accounts the way forecasts should be scored.

For each dataset: split respondents into five folds; on the training folds compute
first-place shares over the full item set; calibrate both accounts to those shares
and nothing else, Luce worths being the shares themselves and race locations coming
from inverting the Gaussian winner probabilities; form each account's predicted
distribution on every subset of at least two items; then for every held-out
respondent and every subset, take that respondent's highest-ranked member of the
subset as the outcome and accumulate log loss under both accounts.

No pairwise or subset outcome enters either calibration, the evaluation covers all
subset sizes rather than pairs alone, and log loss is proper, so a correct forecast
cannot be improved by distorting it.

This module's own score()/main() are superseded and not what the paper's Table 3
reports. Their bootstrap resamples already-computed respondent losses with
calibration held fixed; estimand.py and sensitivity.py refit calibration inside
every replicate, which is what the paper describes and what its numbers are. Kept
here for the shared loaders, load_all() and preflib_sets(): every other script in
this directory imports them from this file.

Usage:  python heldout_score.py   (superseded diagnostic; see estimand.py instead)
"""
import itertools
import math
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

DATA = HERE / "data"     # committed inputs; nothing is read from a temp directory
PREFLIB = HERE / "preflib"
FLOOR = 1e-6
MAX_RESP = 5000
ALPHA = 0.5   # add-alpha, applied wherever shares are formed


def ranks_from_ratings(G):
    rng = np.random.default_rng(0)
    return (-(G + rng.random(G.shape) * 1e-9)).argsort(axis=1).argsort(axis=1) + 1.0


def load_all():
    out = {}
    try:
        import pandas as pd
        for k in (1, 2, 3):
            f = DATA / f"jester_gauge_{k}.csv.gz"
            if not f.exists():
                continue
            G = pd.read_csv(f, header=None).to_numpy(dtype=float)
            out[f"Jester file {k}"] = ranks_from_ratings(G)
    except Exception as e:
        print(f"  jester skipped: {str(e)[:70]}", file=sys.stderr)
    try:
        import pandas as pd
        f = DATA / "gss_rankcards.csv.gz"
        if f.exists():
            cols = {"GSS socialization": ["obey", "popular", "thnkself",
                                          "workhard", "helpoth"],
                    "GSS job values": ["jobinc", "jobsec", "jobhour",
                                       "jobpromo", "jobmeans"]}
            d = pd.read_csv(f)
            for name, cs in cols.items():
                M = d[cs].to_numpy(dtype=float)
                keep = np.array([(not np.isnan(r).any())
                                 and set(r) == set(range(1, len(cs) + 1))
                                 for r in M])
                out[name] = M[keep]
    except Exception as e:
        print(f"  gss skipped: {str(e)[:70]}", file=sys.stderr)
    f = HERE / "sushi3a.5000.10.order"
    if f.exists():
        rows = []
        for i, line in enumerate(f.read_text().splitlines()):
            if i == 0:
                continue
            order = [int(x) for x in line.split()[2:]]
            if len(order) == 10:
                r = [0.0] * 10
                for pos, it in enumerate(order):
                    r[it] = pos + 1
                rows.append(r)
        out["Sushi"] = np.array(rows)
    try:
        import pyreadr
        for name, rel, conv in [("Occupational prestige",
                                 "d_occup.RData", "order"),
                                ("Political goals", "German.rda", "rank"),
                                ("Sports participation", "sports.rda", "rank")]:
            p = DATA / rel
            if not p.exists():
                continue
            d = pyreadr.read_r(str(p))
            M = np.asarray(d[list(d)[0]], dtype=float)
            M = M[~np.isnan(M).any(axis=1)]
            M = M[(M >= 1).all(axis=1)]
            K = M.shape[1]
            if conv == "order":
                R = np.empty_like(M)
                for i in range(M.shape[0]):
                    for pos, it in enumerate(M[i].astype(int)):
                        R[i, it - 1] = pos + 1
                M = R
            M = M[np.array([len(set(r.astype(int))) == K for r in M])]
            out[name] = M
    except Exception as e:
        print(f"  cran skipped: {str(e)[:70]}", file=sys.stderr)
    try:
        out.update(preflib_sets())
    except Exception as e:
        print(f"  preflib skipped: {str(e)[:70]}", file=sys.stderr)
    return out


def read_soc(path):
    """Complete rankings from a PrefLib order file; returns a ranks matrix."""
    lines = [l.strip() for l in path.read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if not lines:
        return None
    rows = []
    for line in lines:
        # PrefLib order files are "count: a,b,c,d"; older files use a comma there
        head, sep, tail = line.partition(":")
        if not sep:
            head, _, tail = line.partition(",")
        try:
            cnt = int(head.strip())
            order = [int(x) for x in tail.replace("{", "").replace("}", "").split(",")
                     if x.strip()]
        except ValueError:
            continue
        if len(order) < 2 or len(set(order)) != len(order):
            continue
        rows.append((cnt, order))
    if not rows:
        return None
    K = max(max(o) for _, o in rows)
    full = [(c, o) for c, o in rows if len(o) == K]
    if not full:
        return None
    R = []
    for c, o in full:
        r = [0.0] * K
        for pos, it in enumerate(o):
            r[it - 1] = pos + 1
        R.extend([r] * min(c, 20000))
    return np.array(R, dtype=float)


def preflib_sets():
    """Each PrefLib file is a separate design: 'alternative 1' does not name the same
    stimulus in different files. Netflix alternative 1 is a different film in every file
    (Shrek in one, Spy Game in another); dots and puzzles vary similarly past the first
    slot. Stacking files with the same K, as an earlier version of this function did,
    silently pooled unrelated choice problems. Every file is its own collection.
    """
    out = {}
    for tag, label in (("dots", "Dots"), ("puzzle", "Puzzles"),
                       ("netflix", "Netflix")):
        i = 0
        for f in sorted(PREFLIB.glob(f"*{tag}*")):
            M = read_soc(f)
            if M is None or M.shape[0] < 50:
                continue
            i += 1
            out[f"{label} {i}"] = M
    return out



def predictions(p):
    K = len(p)
    a, err = calibrate_np(list(p))
    preds = {}
    for r in range(2, K + 1):
        for S in itertools.combinations(range(K), r):
            idx = list(S)
            lz = sum(p[i] for i in idx)
            luce = np.maximum(np.array([p[i] / lz for i in idx]), FLOOR)
            w = win_probs_np(a[idx])
            race = np.maximum(w / w.sum(), FLOOR)
            preds[S] = (luce, race)
    return preds, err


def score(R, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    fold = np.array_split(rng.permutation(n), folds)
    L = np.zeros(n)
    G = np.zeros(n)
    c = np.zeros(n)
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
        nsub = len(preds)
        for S, (luce, race) in preds.items():
            win = R[np.ix_(test, list(S))].argmin(axis=1)
            L[test] += -np.log(luce[win])
            G[test] += -np.log(race[win])
            c[test] += 1
    ll, lg = L / c, G / c
    diff = ll - lg
    r2 = random.Random(7)
    bs = sorted(float(np.mean(diff[[r2.randrange(n) for _ in range(n)]]))
                for _ in range(4000))
    return {"n": n, "K": K, "subsets": nsub, "luce": float(ll.mean()),
            "race": float(lg.mean()), "diff": float(diff.mean()),
            "lo": bs[100], "hi": bs[3900]}


def main():
    data = load_all()
    print(f"{'dataset':<24}{'n':>6}{'K':>3}{'subsets':>8}"
          f"{'Luce':>9}{'race':>9}{'gain':>9}{'95% CI':>20}")
    wins = tot = 0
    for name, R in sorted(data.items()):
        r = score(R)
        if not r:
            print(f"{name:<24}  not scorable")
            continue
        tot += 1
        wins += r["diff"] > 0
        print(f"{name:<24}{r['n']:>6}{r['K']:>3}{r['subsets']:>8}"
              f"{r['luce']:>9.4f}{r['race']:>9.4f}{r['diff']:>+9.4f}"
              f"   [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"\nrace has lower held-out log loss in {wins} of {tot} datasets")


if __name__ == "__main__":
    main()
