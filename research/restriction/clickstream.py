"""Restriction on Wikipedia links: a page loses a destination, readers redistribute.

A Wikipedia article offers a set of outbound links and a reader picks one. The choice is
exclusive, the alternatives are the same target articles from month to month, and the
link set changes as editors add and remove links. So consecutive monthly clickstream
dumps give the same page asking the same question of a smaller menu, which is this
paper's experiment at a scale no laboratory reaches.

One limitation dominates and is stated rather than buried: the dump publishes only pairs
with at least ten clicks, so a destination absent in the later month may have been
removed by an editor or may simply have fallen below the threshold. The menu is inferred
from clicks, not read from the page's wikitext. Filters below insist the vanished
destination held a substantial share before disappearing and that the page's traffic did
not collapse, which makes a threshold artifact unlikely without excluding it. Reading
actual link sets from page revisions would settle it and is not done here.

Usage:  python clickstream.py month_a.tsv month_b.tsv [n_null_reps]
"""
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

FLOOR = 1e-6
ALPHA = 0.5
MIN_K = 4            # destinations in the earlier month
MIN_TOTAL = 500      # clicks on the page in each month
MIN_LOST_SHARE = 0.05   # the vanished destination must have mattered
MAX_TRAFFIC_DROP = 0.5  # and the page must not have lost most of its traffic


def load(path):
    per = collections.defaultdict(dict)
    with open(path) as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) < 4:
                continue
            try:
                per[a[0]][a[1]] = int(a[3])
            except ValueError:
                continue
    return per


def cells(A, B):
    """Pages whose destination set strictly shrank, with the filters applied."""
    out = []
    for page, da in A.items():
        db = B.get(page)
        if not db or len(da) < MIN_K:
            continue
        sa, sb = set(da), set(db)
        if not (sb < sa):
            continue
        ta, tb = sum(da.values()), sum(db.values())
        if ta < MIN_TOTAL or tb < MIN_TOTAL or tb < MAX_TRAFFIC_DROP * ta:
            continue
        lost = sa - sb
        if sum(da[d] for d in lost) / ta < MIN_LOST_SHARE:
            continue
        out.append((page, da, db))
    return out


def prepare(cs):
    """Both maps' predictions per page. Calibration uses only the earlier month, so it
    is identical across null replicates and is done once."""
    out = []
    for page, da, db in cs:
        dests = sorted(da)
        c = np.array([da[d] for d in dests], dtype=float)
        p = (c + ALPHA) / (c.sum() + ALPHA * len(c))
        a, err = calibrate_np(list(p))
        if err > 0.05:
            continue
        a = np.asarray(a)
        keep = sorted(db)
        idx = [dests.index(d) for d in keep]
        lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
        w = win_probs_np(a[idx])
        ra = np.maximum(w / w.sum(), FLOOR)
        u = p[idx] / p[idx].sum()
        out.append((keep, lu, ra, u, int(sum(db.values())),
                    np.array([db[d] for d in keep], dtype=float)))
    return out


def score(prep, obs_list=None):
    tot_l = tot_g = 0.0
    n = 0
    for i, (keep, lu, ra, u, tot, obs0) in enumerate(prep):
        obs = obs0 if obs_list is None else obs_list[i]
        s = obs.sum()
        if s <= 0:
            continue
        o = obs / s
        tot_l += float(-(o * np.log(lu)).sum())
        tot_g += float(-(o * np.log(ra)).sum())
        n += 1
    if n < 20:
        return None
    return {"cells": n, "luce": tot_l / n, "race": tot_g / n,
            "gain": (tot_l - tot_g) / n}


def synth(prep, rng):
    """Regenerate the later month from a Luce process on the earlier month's shares,
    keeping every page's menu and click total exactly as observed."""
    return [rng.multinomial(tot, u).astype(float) for _, _, _, u, tot, _ in prep]


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    A, B = load(a_path), load(b_path)
    cs = cells(A, B)
    print(f"{len(A)} pages in the earlier month, {len(cs)} with a shrunken menu "
          f"passing the filters", flush=True)
    prep = prepare(cs)
    r = score(prep)
    if not r:
        print("too few usable pages")
        return
    rng = np.random.default_rng(2)
    null = []
    for b in range(reps):
        s = score(prep, synth(prep, rng))
        if s:
            null.append(s["gain"])
    null = np.array(sorted(null))
    med = float(np.median(null)) if len(null) else float("nan")
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1) if len(null) else float("nan")
    print(f"\n{r['cells']} pages scored")
    print(f"  renormalization {r['luce']:.4f}   race {r['race']:.4f}   gain {r['gain']:+.4f}")
    print(f"  Luce null median {med:+.4f}   excess {r['gain']-med:+.4f}   "
          f"MC tail {pv:.3f}  ({len(null)} reps)")


if __name__ == "__main__":
    main()
