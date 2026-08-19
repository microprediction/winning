"""The two decomposition tables, which had no committed script until now.

Table "gain by rank of the item chosen" and table "probability that the favourite
finishes second" were computed ad hoc for an earlier draft and never checked in. The
figure audit in `demo/check_tables.py` found both, since no committed run accounted for
any of their twenty-two numbers. This reproduces them from the same loaders the held-out
scoring uses.

Both are decompositions of the same held-out predictions, not new experiments.

  by rank      Score every held-out prediction as usual, then bucket the contribution by
               the rank the chosen item held in the training-fold full-menu shares. The
               aggregate gain is the weighted sum of the buckets, which is printed as a
               check.

  favourite    Ask how often the full-field favourite finishes second. Both columns are
  second       sequential rules: remove the winner, then apply the map to the survivors.
               Neither is the exact ordering law of its model, which is why the paper
               reports this as an aside on heuristics rather than as a restriction test.

Usage:  python decompositions.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np
from heldout_score import load_all

FLOOR = 1e-12
ALPHA = 0.5
BY_RANK = ("Sushi", "Jester file 1", "Jester file 2", "Jester file 3")
FAV_SECOND = ("Political goals", "GSS job values", "GSS socialization", "Sushi",
              "Jester file 1")


def shares(R, K):
    """First-place shares over the full item set, with the shared add-alpha convention."""
    c = np.zeros(K)
    for row in R:
        c[int(np.argmin(row))] += 1
    return (c + ALPHA) / (c.sum() + ALPHA * K)


def race_locations(p):
    a, err = calibrate_np(list(p))
    return np.asarray(a), err


def gain_by_rank(R, K, folds=5, seed=0):
    """Held-out gain bucketed by the full-menu rank of the item actually chosen."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(R))
    buckets = np.zeros(K)
    counts = np.zeros(K)
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        p = shares(R[train], K)
        a, err = race_locations(p)
        if err > 0.05:
            return None
        order = np.argsort(-p)                      # rank 0 is the favourite
        rank_of = np.empty(K, dtype=int)
        rank_of[order] = np.arange(K)
        # pairs only: the cell where the two maps differ most and the paper's table is
        # pooled over the same subsets it scores, so use every unordered pair
        for i in range(K):
            for j in range(i + 1, K):
                sub = [i, j]
                lu = np.maximum(p[sub] / p[sub].sum(), FLOOR)
                w = win_probs_np(a[sub])
                ra = np.maximum(w / w.sum(), FLOOR)
                for row in R[test]:
                    pick = 0 if row[i] < row[j] else 1
                    r = rank_of[sub[pick]]
                    buckets[r] += -np.log(lu[pick]) + np.log(ra[pick])
                    counts[r] += 1
    total = counts.sum()
    return {"share": counts / total, "gain": np.divide(buckets, counts,
            out=np.zeros_like(buckets), where=counts > 0),
            "aggregate": buckets.sum() / total}


def favourite_second(R, K):
    """P(favourite finishes second), observed against the two sequential rules."""
    p = shares(R, K)
    a, err = race_locations(p)
    if err > 0.05:
        return None
    fav = int(np.argmax(p))
    second = 0
    for row in R:
        order = np.argsort(row)
        if order[1] == fav:
            second += 1
    observed = second / len(R)

    # sequential renormalization, which is Harville: P(w first) * P(fav first among rest)
    renorm = 0.0
    for w in range(K):
        if w == fav:
            continue
        rest = [k for k in range(K) if k != w]
        renorm += p[w] * (p[fav] / p[rest].sum())
    # sequential race: remove the winner, re-run the contest among the survivors
    race = 0.0
    for w in range(K):
        if w == fav:
            continue
        rest = [k for k in range(K) if k != w]
        q = win_probs_np(a[rest])
        q = q / q.sum()
        race += p[w] * q[rest.index(fav)]
    return {"observed": observed, "renorm": renorm, "race": race}


def main():
    data = load_all()
    print("Gain by the full-menu rank of the item chosen, pooled over "
          + ", ".join(BY_RANK) + "\n")
    pooled_share = None
    pooled_gain = None
    n_seen = 0
    agg = 0.0
    for name in BY_RANK:
        if name not in data:
            print(f"  {name}: not available")
            continue
        R = data[name]
        K = R.shape[1]
        out = gain_by_rank(R, K)
        if out is None:
            print(f"  {name}: calibration failed")
            continue
        if pooled_share is None:
            pooled_share = np.zeros(K)
            pooled_gain = np.zeros(K)
        pooled_share += out["share"]
        pooled_gain += out["gain"]
        agg += out["aggregate"]
        n_seen += 1
    if n_seen:
        pooled_share /= n_seen
        pooled_gain /= n_seen
        K = len(pooled_share)
        print("rank of chosen item  " + "".join(f"{r + 1:>8}" for r in range(K)))
        print("share of outcomes    " + "".join(f"{v:>8.2f}" for v in pooled_share))
        print("mean gain            " + "".join(f"{v:>+8.3f}" for v in pooled_gain))
        print(f"\ncontributions sum to {agg / n_seen:+.4f}")

    print("\n\nProbability that the full-field favourite finishes second\n")
    print(f"{'collection':<22}{'observed':>10}{'renorm':>10}{'race':>10}"
          f"{'renorm err':>12}{'race err':>10}{'removed':>9}")
    for name in FAV_SECOND:
        if name not in data:
            print(f"  {name}: not available")
            continue
        R = data[name]
        K = R.shape[1]
        out = favourite_second(R, K)
        if out is None:
            print(f"  {name}: calibration failed")
            continue
        er = out["renorm"] - out["observed"]
        eg = out["race"] - out["observed"]
        pct = 100 * (out["renorm"] - out["race"]) / er if er else float("nan")
        print(f"{name:<22}{out['observed']:>10.3f}{out['renorm']:>10.3f}{out['race']:>10.3f}"
              f"{er:>+12.3f}{eg:>+10.3f}{pct:>8.1f}%")
    print("\nBoth columns are sequential heuristics, not the exact ordering law of either "
          "model.")


if __name__ == "__main__":
    main()
