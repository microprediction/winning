"""Restriction on observed news slates.

The Microsoft News dataset records, for each impression, the exact set of articles shown
and which one was clicked. So the menu is observed rather than inferred, which is what the
Wikipedia clickstream could not offer. Slates recur across users, and one slate is often a
subset of another, giving nested menus over identical articles.

For a pair of recurring slates with T a strict subset of S, click shares on S calibrate
both maps and the choices actually made on T score them. The fitted-Luce null redraws the
choices on T from the axiom applied to S's shares, keeping every slate and its number of
impressions exactly as observed.

The menus here are chosen by a recommender rather than by an experimenter, so a slate's
composition is correlated with what the system expects users to like. Both maps inherit
that identically, and the null runs on the same slates, so the comparison is fair even
though the assignment is not random.

Usage:  python mind_slates.py train.jsonl [min_reps] [n_null_reps]
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

FLOOR = 1e-6
ALPHA = 0.5


def load(path, min_reps):
    """slate -> Counter of clicked article, for slates seen at least min_reps times."""
    clicks = collections.defaultdict(collections.Counter)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            items = tuple(sorted(r["positive"] + r["negative"]))
            if len(items) < 2 or len(r["positive"]) != 1:
                continue
            clicks[items][r["positive"][0]] += 1
    return {s: c for s, c in clicks.items() if sum(c.values()) >= min_reps}


def nested(slates):
    """(big, small) with small a strict subset of big, sharing at least two items."""
    keys = sorted(slates, key=len, reverse=True)
    sets = {k: set(k) for k in keys}
    out = []
    for i, big in enumerate(keys):
        for small in keys[i + 1:]:
            if len(small) >= 2 and sets[small] < sets[big]:
                out.append((big, small))
    return out


def prepare(pairs, slates):
    prep = []
    for big, small in pairs:
        items = list(big)
        c = np.array([slates[big][it] for it in items], dtype=float)
        p = (c + ALPHA) / (c.sum() + ALPHA * len(c))
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        idx = [items.index(it) for it in small]
        lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
        w = win_probs_np(a[idx])
        ra = np.maximum(w / w.sum(), FLOOR)
        obs = np.array([slates[small][it] for it in small], dtype=float)
        prep.append((lu, ra, p[idx] / p[idx].sum(), int(obs.sum()), obs))
    return prep


def score(prep, obs_list=None):
    tl = tg = 0.0
    n = 0
    for i, (lu, ra, u, tot, obs0) in enumerate(prep):
        obs = obs0 if obs_list is None else obs_list[i]
        s = obs.sum()
        if s <= 0:
            continue
        o = obs / s
        tl += float(-(o * np.log(lu)).sum())
        tg += float(-(o * np.log(ra)).sum())
        n += 1
    if n < 20:
        return None
    return {"pairs": n, "luce": tl / n, "race": tg / n, "gain": (tl - tg) / n}


def main():
    path = sys.argv[1]
    min_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    slates = load(path, min_reps)
    print(f"{len(slates)} slates seen at least {min_reps} times", flush=True)
    pairs = nested(slates)
    print(f"{len(pairs)} nested slate pairs", flush=True)
    prep = prepare(pairs[:6000], slates)
    print(f"{len(prep)} pairs prepared", flush=True)
    r = score(prep)
    if not r:
        print("too few usable pairs")
        return
    print(f"  renormalization {r['luce']:.4f}   race {r['race']:.4f}   "
          f"gain {r['gain']:+.4f}   on {r['pairs']} pairs", flush=True)
    rng = np.random.default_rng(6)
    null = []
    for b in range(reps):
        syn = [rng.multinomial(tot, u).astype(float) for _, _, u, tot, _ in prep]
        s = score(prep, syn)
        if s:
            null.append(s["gain"])
    null = np.array(sorted(null))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    print(f"\n{r['pairs']} pairs scored")
    print(f"  renormalization {r['luce']:.4f}   race {r['race']:.4f}   gain {r['gain']:+.4f}")
    print(f"  Luce null median {med:+.4f}   excess {r['gain']-med:+.4f}   "
          f"MC tail {pv:.3f}  ({len(null)} reps)")


if __name__ == "__main__":
    main()
