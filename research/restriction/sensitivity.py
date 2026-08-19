"""Three sensitivities the reviews asked for, and a bootstrap that refits.

Every referee report raised the same three procedural gaps, and none of them was blocked.

  smoothing      The structural maps fit nothing, but add-alpha is an implementation
                 convention and the two maps need not respond to pseudocounts alike. One
                 collection is scorable only because the convention removes a zero cell.
                 Run at alpha in {0, 1/2, 1}.

  menu weighting Scoring every subset of size two or more weights menu sizes by how many
                 subsets have them, which for K=10 puts most of the weight near |T|=5. The
                 gain varies sharply with |T|, so the weighting is part of the estimand.
                 Report uniform over subsets, uniform over surviving size, and pairs only.

  bootstrap      The published intervals resample respondent losses with the fitted models
                 held fixed, so they omit calibration uncertainty and are too narrow. Here
                 the respondents are resampled and the whole pipeline, calibration and
                 both maps and all subsets, is rerun inside every replicate.

Usage:  python sensitivity.py [n_boot] [datasets...]
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

FLOOR = 1e-12
MAX_RESP = 5000


def predictions(p):
    K = len(p)
    a, err = calibrate_np(list(p))
    if err > 0.05:
        return None, err
    a = np.asarray(a)
    out = {}
    for r in range(2, K + 1):
        for S in itertools.combinations(range(K), r):
            idx = list(S)
            lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
            w = win_probs_np(a[idx])
            out[S] = (lu, np.maximum(w / w.sum(), FLOOR))
    return out, err


def run(R, alpha=0.5, folds=5, seed=0, weight="subsets"):
    """Held-out gain under one smoothing convention and one menu weighting."""
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    fold = np.array_split(rng.permutation(n), folds)
    # accumulate by subset size so any weighting can be applied afterwards
    L = np.zeros(K + 1)
    G = np.zeros(K + 1)
    C = np.zeros(K + 1)
    for f in range(folds):
        test = fold[f]
        train = np.concatenate([fold[g] for g in range(folds) if g != f])
        cts = np.bincount(R[train].argmin(axis=1), minlength=K).astype(float)
        p = (cts + alpha) / (len(train) + alpha * K)
        if (p <= 0).any():
            return None
        preds, err = predictions(p)
        if preds is None:
            return None
        for S, (lu, ra) in preds.items():
            win = R[np.ix_(test, list(S))].argmin(axis=1)
            L[len(S)] += float(-np.log(lu[win]).sum())
            G[len(S)] += float(-np.log(ra[win]).sum())
            C[len(S)] += len(win)
    sizes = [r for r in range(2, K + 1) if C[r] > 0]
    if not sizes:
        return None
    if weight == "subsets":                       # every subset counts once
        num = (L[sizes] - G[sizes]).sum()
        den = C[sizes].sum()
        return float(num / den)
    if weight == "size":                          # every surviving size counts once
        per = [(L[r] - G[r]) / C[r] for r in sizes]
        return float(np.mean(per))
    if weight == "pairs":
        return float((L[2] - G[2]) / C[2]) if C[2] > 0 else None
    raise ValueError(weight)


def bootstrap_refit(R, reps=200, alpha=0.5, seed=0):
    """Resample respondents and rerun calibration and scoring inside each replicate."""
    rng = np.random.default_rng(seed)
    n = len(R)
    out = []
    for b in range(reps):
        idx = rng.integers(0, n, n)
        g = run(R[idx], alpha=alpha, seed=b)
        if g is not None:
            out.append(g)
    if len(out) < 20:
        return None
    out = np.sort(np.asarray(out))
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out))]
    return float(lo), float(hi), len(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    wanted = sys.argv[2:]
    data = load_all()
    names = [k for k in sorted(data) if not wanted or k in wanted]

    print("Smoothing and menu weighting. Gain in nats per prediction.\n")
    print(f"{'dataset':<24}{'a=0':>9}{'a=1/2':>9}{'a=1':>9}"
          f"{'  |  ':>5}{'subsets':>9}{'by size':>9}{'pairs':>9}")
    for name in names:
        R = data[name]
        row = []
        for a in (0.0, 0.5, 1.0):
            g = run(R, alpha=a)
            row.append("n/a" if g is None else f"{g:+.4f}")
        wts = []
        for w in ("subsets", "size", "pairs"):
            g = run(R, alpha=0.5, weight=w)
            wts.append("n/a" if g is None else f"{g:+.4f}")
        print(f"{name:<24}{row[0]:>9}{row[1]:>9}{row[2]:>9}{'  |  ':>5}"
              f"{wts[0]:>9}{wts[1]:>9}{wts[2]:>9}")

    print(f"\n\nRespondent bootstrap with the whole pipeline refit inside each replicate, "
          f"{reps} replicates.\n")
    print(f"{'dataset':<24}{'gain':>9}{'refit 95%':>24}{'reps':>6}")
    for name in names:
        R = data[name]
        g = run(R)
        if g is None:
            print(f"{name:<24}{'n/a':>9}")
            continue
        bs = bootstrap_refit(R, reps=reps)
        if bs is None:
            print(f"{name:<24}{g:>+9.4f}{'  too few usable replicates':>24}")
            continue
        lo, hi, k = bs
        print(f"{name:<24}{g:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}{k:>6}")


if __name__ == "__main__":
    main()
