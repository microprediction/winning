"""Restriction over a twelve-line master set, within subject, with global labels.

Unpublished data from the Perception and Cognition Laboratory, University of Missouri,
public in `PerceptionCognitionLab/data0` under `1dMemory/chunk`. Subjects learn a fixed
number for each of twelve line lengths and identify them under feedback. Some blocks offer
all twelve lines; others offer a named subset, and the instructions state that "the number
assignment for each line is constant throughout the experiment", so the logged stimulus and
response are indices into the master set rather than within-block ranks.

Three sub-experiments are scored. Conditions A and D bracket the restricted blocks with
full-twelve blocks before and after, and restrict to {0,1,2,3}, {4,5,6,7}, {8,9,10,11},
{0..7} and {4..11}. Condition C restricts to twelve named pairs, so the same subject supplies
a twelve-way distribution and a binary choice over the same labels. Condition B is an
all-twelve control and is not scored.

Calibration pools the full-twelve blocks for a subject, before and after, which is the
bracket that keeps practice from loading onto the restricted blocks. Each restricted set then
supplies one cell per stimulus it contains.

Offered sets are matched to the design sets read off the experiment sources rather than
inferred from the labels observed, since a forty-trial block over twelve lines can miss a
line by chance.

Usage:  python rouder_chunk.py [n_null_reps]
"""
import collections
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

COUNTS = HERE / "data" / "rouder_chunk" / "counts.npz"
FLOOR = 1e-6
ALPHA = 0.5
K = 12
FULL = frozenset(range(K))

# design sets, from C1.C and C1_2.C in the deposit
QUARTERS = [frozenset({0, 1, 2, 3}), frozenset({4, 5, 6, 7}), frozenset({8, 9, 10, 11})]
OCTETS = [frozenset(range(8)), frozenset(range(4, 12))]
PAIRS = [frozenset(p) for p in
         [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9),
          (8, 10), (9, 11), (0, 10), (1, 11)]]
CANDIDATES = {"A": [FULL] + QUARTERS + OCTETS,
              "D": [FULL] + QUARTERS + OCTETS,
              "C": [FULL] + PAIRS}


def read(path):
    out = []
    for line in open(path):
        t = line.split()
        if len(t) != 7:
            continue
        out.append((int(t[1][3:]), int(t[4]), int(t[5])))    # block, stimulus, response
    return out


def offered(observed, candidates):
    """Smallest design set containing everything observed in the block."""
    fits = [c for c in candidates if observed <= c]
    return min(fits, key=len) if fits else None


def load_counts():
    """Per-block confusion counts, derived from the deposit by derive_rouder.py.

    The deposit itself is not redistributed: it carries no licence. What is committed is
    the count array, which is what this analysis reads.
    """
    z = np.load(COUNTS, allow_pickle=False)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for c, subj, cond, blk in zip(z["counts"], z["subject"], z["condition"], z["block"]):
        out[str(cond)][str(subj)].append((int(blk), c))
    return out


def build(cond):
    groups = []
    per_subject = load_counts().get(cond, {})
    for path in sorted(per_subject):
        blocks = {blk: c for blk, c in per_subject[path]}
        master = np.zeros((K, K))
        restricted = collections.defaultdict(lambda: np.zeros((K, K)))
        for blk, counts in blocks.items():
            seen = {i for i in range(K) if counts[i].sum() or counts[:, i].sum()}
            S = offered(seen, CANDIDATES[cond])
            if S is None:
                continue
            target = master if S == FULL else restricted[S]
            target += counts
        sub = path
        for S, counts in restricted.items():
            idx = sorted(S)
            for st in idx:
                obs = counts[st, idx]
                if obs.sum() < 5 or master[st].sum() < 20:
                    continue
                p = (master[st] + ALPHA) / (master[st].sum() + ALPHA * K)
                a, err = calibrate_np(list(p))
                if err > 0.05:
                    continue
                a = np.asarray(a)
                u = p[idx] / p[idx].sum()
                w = win_probs_np(a[idx])
                groups.append({"lu": np.maximum(u, FLOOR),
                               "ra": np.maximum(w / w.sum(), FLOOR),
                               "u": u, "o": obs, "n": int(obs.sum()),
                               "p": p, "n_full": int(master[st].sum()), "idx": idx,
                               "size": len(idx), "sub": sub, "cond": cond})
    return groups


def score(cells):
    tl = tg = 0.0
    n = 0
    for c in cells:
        if c["o"].sum() <= 0:
            continue
        q = c["o"] / c["o"].sum()
        tl += float(-(q * np.log(c["lu"])).sum())
        tg += float(-(q * np.log(c["ra"])).sum())
        n += 1
    if n < 5:
        return None
    return {"cells": n, "luce": tl / n, "race": tg / n, "gain": (tl - tg) / n}


def null_rep(cells, rng):
    out = []
    for c in cells:
        d = rng.multinomial(c["n_full"], c["p"]).astype(float)
        p = (d + ALPHA) / (d.sum() + ALPHA * K)
        a = np.asarray(calibrate_np(list(p))[0])
        lu = p[c["idx"]] / p[c["idx"]].sum()
        w = win_probs_np(a[c["idx"]])
        out.append({"lu": np.maximum(lu, FLOOR), "ra": np.maximum(w / w.sum(), FLOOR),
                    "o": rng.multinomial(c["n"], c["u"]).astype(float)})
    return out


def report(label, cells, reps, seed=0):
    r = score(cells)
    if not r:
        print(f"{label:<30} too few cells")
        return
    rng = np.random.default_rng(seed)
    null = np.array(sorted(score(null_rep(cells, rng))["gain"] for _ in range(reps)))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    subs = sorted({c["sub"] for c in cells})
    bs = []
    for _ in range(400):
        pick = collections.Counter(np.array(subs)[rng.integers(0, len(subs), len(subs))])
        rep = [c for c in cells for _ in range(pick[c["sub"]])]
        t = score(rep)
        if t:
            bs.append(t["gain"])
    bs = sorted(bs)
    ci = f"[{bs[int(.025 * len(bs))]:+.4f}, {bs[int(.975 * len(bs))]:+.4f}]" if len(bs) > 30 else ""
    print(f"{label:<30}{r['cells']:>6}{len(subs):>5}{r['luce']:>9.4f}{r['race']:>9.4f}"
          f"{r['gain']:>+9.4f}  {ci:<22}{med:>+9.4f}{r['gain'] - med:>+9.4f}{pv:>8.3f}")


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    cells = []
    for cond in ("A", "D", "C"):
        cells += build(cond)
    print(f"{len(cells)} cells over {len({c['sub'] for c in cells})} subjects\n")
    print(f"{'split':<30}{'cells':>6}{'subj':>5}{'renorm':>9}{'race':>9}{'gain':>9}"
          f"  {'subject bootstrap':<22}{'null':>9}{'excess':>9}{'tail':>8}")
    report("all restricted blocks", cells, reps, 0)
    report("twelve to two, condition C", [c for c in cells if c["size"] == 2], reps, 1)
    report("twelve to four", [c for c in cells if c["size"] == 4], reps, 2)
    report("twelve to eight", [c for c in cells if c["size"] == 8], reps, 3)
    for cond in ("A", "D", "C"):
        report(f"condition {cond}", [c for c in cells if c["cond"] == cond], reps, 4)


if __name__ == "__main__":
    main()
