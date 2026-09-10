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
    # restrictions only; see the note in luce_null.py on why T=S is excluded
    for r in range(2, K):
        for S in itertools.combinations(range(K), r):
            idx = list(S)
            lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
            w = win_probs_np(a[idx])
            out[S] = (lu, np.maximum(w / w.sum(), FLOOR))
    return out, err


def accumulate(R, alpha=0.5, folds=5, seed=0):
    """Per-size held-out totals (L, G, C) for one already-calibrated collection.
    No weighting is applied here, so several collections' accumulators can be summed
    before weighting is chosen, which is what combining separately calibrated files
    of one source requires."""
    rng = np.random.default_rng(seed)
    n, K = R.shape
    if n > MAX_RESP:
        R = R[rng.choice(n, MAX_RESP, replace=False)]
        n = MAX_RESP
    fold = np.array_split(rng.permutation(n), folds)
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
    return L, G, C


def weighted(L, G, C, weight):
    K = len(C) - 1
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


def run(R, alpha=0.5, folds=5, seed=0, weight="subsets"):
    """Held-out gain under one smoothing convention and one menu weighting."""
    acc = accumulate(R, alpha=alpha, folds=folds, seed=seed)
    if acc is None:
        return None
    return weighted(*acc, weight)


def run_grouped(Rs, alpha=0.5, folds=5, seed=0, weight="subsets"):
    """Same as run(), but Rs is several files of one source: each is calibrated and
    scored on its own, and the per-size totals are pooled before weighting."""
    Ktot = max(R.shape[1] for R in Rs)
    L = np.zeros(Ktot + 1); G = np.zeros(Ktot + 1); C = np.zeros(Ktot + 1)
    for R in Rs:
        acc = accumulate(R, alpha=alpha, folds=folds, seed=seed)
        if acc is None:
            return None
        l, g, c = acc
        L[:len(l)] += l; G[:len(g)] += g; C[:len(c)] += c
    return weighted(L, G, C, weight)


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
    return _summarize_boot(out)


def bootstrap_refit_grouped(Rs, reps=200, alpha=0.5, seed=0):
    """Grouped refit bootstrap: every replicate resamples respondents independently
    within each file, then reruns calibration and scoring per file and pools before
    weighting, exactly as run_grouped does on the observed data."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(reps):
        resampled = [R[rng.integers(0, len(R), len(R))] for R in Rs]
        g = run_grouped(resampled, alpha=alpha, seed=b)
        if g is not None:
            out.append(g)
    return _summarize_boot(out)


def _summarize_boot(out):
    if len(out) < 20:
        return None
    out = np.sort(np.asarray(out))
    # below about eighty replicates the 2.5 and 97.5 percentiles are the extreme order
    # statistics, so what comes back is the observed range and not a 95 per cent interval.
    # Say which one it is rather than dressing a range as an interval.
    kind = "95%" if len(out) >= 80 else "range"
    lo = out[int(0.025 * len(out))] if kind == "95%" else out[0]
    hi = out[int(0.975 * len(out))] if kind == "95%" else out[-1]
    return float(lo), float(hi), len(out), kind


import re
GROUP_RE = re.compile(r"^(Netflix|Dots|Puzzles) \d+$")


def grouped_names(data, wanted):
    """One entry per source. See heldout_score.preflib_sets: a PrefLib file is a
    separate design, so Netflix/Dots/Puzzles are combined here from their per-file
    scores rather than from pooled raw data."""
    groups = {}
    out = []
    for k in sorted(data):
        m = GROUP_RE.match(k)
        if m:
            groups.setdefault(m.group(1), []).append(k)
        else:
            out.append(k)
    names = [n for n in out if not wanted or n in wanted]
    for label, keys in groups.items():
        if not wanted or label in wanted:
            names.append(label)
    return sorted(names), groups


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    wanted = sys.argv[2:]
    data = load_all()
    names, groups = grouped_names(data, wanted)

    def Rs_for(name):
        if name in groups:
            return [data[k] for k in groups[name]]
        return [data[name]]

    print("Smoothing and menu weighting. Gain in nats per prediction.\n")
    print(f"{'dataset':<24}{'a=0':>9}{'a=1/2':>9}{'a=1':>9}"
          f"{'  |  ':>5}{'subsets':>9}{'by size':>9}{'pairs':>9}")
    for name in names:
        Rs = Rs_for(name)
        row = []
        for a in (0.0, 0.5, 1.0):
            g = run_grouped(Rs, alpha=a)
            row.append("n/a" if g is None else f"{g:+.4f}")
        wts = []
        for w in ("subsets", "size", "pairs"):
            g = run_grouped(Rs, alpha=0.5, weight=w)
            wts.append("n/a" if g is None else f"{g:+.4f}")
        tag = "  (grouped)" if name in groups else ""
        print(f"{name:<24}{row[0]:>9}{row[1]:>9}{row[2]:>9}{'  |  ':>5}"
              f"{wts[0]:>9}{wts[1]:>9}{wts[2]:>9}{tag}", flush=True)

    print(f"\n\nRespondent bootstrap with the whole pipeline refit inside each replicate, "
          f"{reps} replicates.\n")
    print(f"{'dataset':<24}{'gain':>9}{'refit interval':>24}{'reps':>6}{'kind':>8}")
    for name in names:
        Rs = Rs_for(name)
        g = run_grouped(Rs)
        if g is None:
            print(f"{name:<24}{'n/a':>9}")
            continue
        # every principal interval gets the full budget. The endpoints of a 95%
        # interval are the 2.5th and 97.5th order statistics, so a few hundred
        # replicates puts them on the second or third most extreme draw.
        budget = reps
        bs = (bootstrap_refit_grouped(Rs, reps=budget) if name in groups
              else bootstrap_refit(Rs[0], reps=budget))
        if bs is None:
            print(f"{name:<24}{g:>+9.4f}{'  too few usable replicates':>24}")
            continue
        lo, hi, k, kind = bs
        print(f"{name:<24}{g:>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}{k:>6}{kind:>8}", flush=True)


if __name__ == "__main__":
    main()
