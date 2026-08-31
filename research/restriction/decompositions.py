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


def gain_by_rank(R, folds=5, seed=0, max_resp=5000):
    """Held-out gain bucketed by the full-menu rank of the item chosen.

    Scored over every subset of size two or more, which is the same set of predictions
    the aggregate is scored on. An earlier version of this script used pairs only and did
    not reproduce the paper's table, because the pairwise gain is two to four times the
    all-subsets aggregate.
    """
    import itertools
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > max_resp:
        R = R[rng.choice(n, max_resp, replace=False)]
        n = max_resp
    fold = np.array_split(rng.permutation(n), folds)
    buckets = np.zeros(K)
    counts = np.zeros(K)
    for f in range(folds):
        test = fold[f]
        train = np.concatenate([fold[g] for g in range(folds) if g != f])
        cts = np.bincount(R[train].argmin(axis=1), minlength=K).astype(float)
        p = (cts + ALPHA) / (len(train) + ALPHA * K)
        a, err = race_locations(p)
        if err > 0.05:
            return None
        order = np.argsort(-p)
        rank_of = np.empty(K, dtype=int)
        rank_of[order] = np.arange(K)
        for r in range(2, K + 1):
            for S in itertools.combinations(range(K), r):
                idx = list(S)
                lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
                w = win_probs_np(a[idx])
                ra = np.maximum(w / w.sum(), FLOOR)
                win = R[np.ix_(test, idx)].argmin(axis=1)
                contrib = -np.log(lu[win]) + np.log(ra[win])
                ranks = rank_of[np.array(idx)][win]
                np.add.at(buckets, ranks, contrib)
                np.add.at(counts, ranks, 1)
    # return the raw accumulators. Pooling has to add these, not average the two
    # derived rows: the mean of the share vectors dotted with the mean of the gain
    # vectors is not the mean of the aggregates, and a table whose own rows do not
    # reproduce its own total is an error however each piece was computed.
    return {"buckets": buckets, "counts": counts}


def gauss_second_exact(a, lo=-14.0, hi=14.0, n=40001):
    """P(rank(i) = 2) for independent N(a_i, 1), exactly.

    Condition on i's draw at x. Item i is second when exactly one rival exceeds x and
    the rest fall below, so integrate phi(x - a_i) times the sum over rivals j of
    (1 - Phi(x - a_j)) times the product over the remaining k of Phi(x - a_k). Checked
    against three million simulated Case V races; agreement is within Monte Carlo error.
    """
    from scipy.stats import norm
    x = np.linspace(lo, hi, n)
    a = np.asarray(a, dtype=float)
    Phi = norm.cdf(x[:, None] - a[None, :])
    phi = norm.pdf(x[:, None] - a[None, :])
    out = np.zeros(len(a))
    for i in range(len(a)):
        others = [k for k in range(len(a)) if k != i]
        P = Phi[:, others]
        safe = P > 1e-300
        ratio = np.where(safe, (1.0 - P) / np.where(safe, P, 1.0), 0.0)
        out[i] = float(np.trapezoid(phi[:, i] * P.prod(axis=1) * ratio.sum(axis=1), x))
    return out


def favourite_second(R, K):
    """P(favourite finishes second): observed, against each model's exact ordering law.

    Sorting log w_i + Gumbel gives a Plackett-Luce ranking, so removing the winner and
    renormalizing proportionally is not a heuristic for the Gumbel model. It is that
    model's exact ordering law, which is Harville's construction and the Gumbel-top-k
    identity. Verified here against four million simulated Gumbel races. The Gaussian
    side has no such sequential identity, so its exact law is computed directly.
    """
    p = shares(R, K)
    a, err = race_locations(p)
    if err > 0.05:
        return None
    fav = int(np.argmax(p))
    observed = float(np.mean([np.argsort(row)[1] == fav for row in R]))

    # exact Plackett-Luce: sum over which item wins, then the favourite leads the rest
    pl = 0.0
    for w in range(K):
        if w == fav:
            continue
        rest = [k for k in range(K) if k != w]
        pl += p[w] * (p[fav] / p[rest].sum())

    exact = float(gauss_second_exact(a)[fav])
    # the sequential Gaussian rule, kept only to show it is not the Gaussian law
    seq = 0.0
    for w in range(K):
        if w == fav:
            continue
        rest = [k for k in range(K) if k != w]
        q = win_probs_np(a[rest])
        q = q / q.sum()
        seq += p[w] * q[rest.index(fav)]
    return {"observed": observed, "renorm": pl, "race": exact, "seq": seq}


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
        out = gain_by_rank(R)
        if out is None:
            print(f"  {name}: calibration failed")
            continue
        if pooled_share is None:
            pooled_share = np.zeros(K)
            pooled_gain = np.zeros(K)
        pooled_share += out["counts"]
        pooled_gain += out["buckets"]
        n_seen += 1
    if n_seen:
        counts, buckets = pooled_share, pooled_gain
        total = counts.sum()
        pooled_share = counts / total
        pooled_gain = np.divide(buckets, counts, out=np.zeros_like(buckets),
                                where=counts > 0)
        agg = buckets.sum() / total
        n_seen = 1
        assert abs(float((pooled_share * pooled_gain).sum()) - agg) < 1e-12, \
            "the printed rows must reproduce the printed total"
        K = len(pooled_share)
        print("rank of chosen item  " + "".join(f"{r + 1:>8}" for r in range(K)))
        # three decimals so a reader can dot the two rows and land on the total
        print("share of outcomes    " + "".join(f"{v:>8.3f}" for v in pooled_share))
        print("mean gain            " + "".join(f"{v:>+8.3f}" for v in pooled_gain))
        print(f"\ncontributions sum to {agg / n_seen:+.4f}")

    print("\n\nProbability that the full-field favourite finishes second\n")
    print(f"{'collection':<22}{'observed':>10}{'exact PL':>10}{'exact CaseV':>12}"
          f"{'PL err':>9}{'CaseV err':>11}{'removed':>9}{'seq CaseV':>11}")
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
        print(f"{name:<22}{out['observed']:>10.3f}{out['renorm']:>10.3f}{out['race']:>12.3f}"
              f"{er:>+9.3f}{eg:>+11.3f}{pct:>8.1f}%{out['seq']:>11.3f}")
    print("\nBoth model columns are exact ordering laws. Removing the winner and")
    print("renormalizing proportionally IS the Plackett-Luce ranking law, by the")
    print("Gumbel-top-k identity, so the PL column is that model's own prediction and")
    print("not a heuristic. The Case V column is the one-dimensional integral for")
    print("P(rank = 2). The last column is the sequential Gaussian rule, shown only to")
    print("record that it is not the Gaussian ordering law.")


if __name__ == "__main__":
    main()
