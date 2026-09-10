"""Restricted-menu prediction across state ballots in a presidential election.

In a US presidential election the alternatives are the same named candidates in every
state, and which of them appears on the ballot varies by state for legal reasons:
petition thresholds and filing deadlines, not voter preference. So the states supply
nested menus over identical alternatives, with vote shares as population choice
shares. Menus in 2024 run from three candidates to thirteen.

States are not interchangeable populations, so a raw cross-state prediction would be
dominated by politics rather than by the restriction map. Two devices deal with that.
Pairs are matched on the two-party split, so a state is only ever used to predict
another with a similar Democrat-to-Republican log odds. And the fitted-Luce null runs
on the same matched pairs, so whatever the matching fails to remove handicaps both maps
equally.

Usage:  python ballots_us.py [max_dr_gap] [n_null_reps]
"""
import collections
import csv
import sys
from pathlib import Path

import random

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

CSV = HERE / "data" / "us_president_state_2024.csv"
FLOOR = 1e-6
ALPHA = 0.5
MIN_VOTES = 500          # a candidate counts as on the ballot above this
DEM = {"DEMOCRAT", "DEMOCRATIC"}


def load():
    """state -> {candidate: votes}, dropping write-in aggregates."""
    per = collections.defaultdict(collections.Counter)
    party = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            cand = (r["candidate"] or "").strip().upper()
            if not cand or "WRITEIN" in cand or cand == "OTHER":
                continue
            if (r.get("writein") or "").strip().upper() == "TRUE":
                continue
            try:
                v = int(float(r["votes"] or 0))
            except ValueError:
                continue
            if v <= 0:
                continue
            per[r["state_po"]][cand] += v
            party[cand] = (r.get("party_simplified") or "").strip().upper()
    out = {}
    for st, c in per.items():
        keep = {k: v for k, v in c.items() if v >= MIN_VOTES}
        if len(keep) >= 3:
            out[st] = keep
    return out, party


def dr_odds(cnt, party):
    d = sum(v for k, v in cnt.items() if party.get(k) in DEM)
    r = sum(v for k, v in cnt.items() if party.get(k) == "REPUBLICAN")
    if d <= 0 or r <= 0:
        return None
    return np.log(d / r)


def predict(shares_big, menu_small):
    """Both maps, calibrated on the larger menu and restricted to the smaller."""
    cands = list(shares_big)
    p = np.array([shares_big[c] for c in cands], dtype=float)
    p = (p + ALPHA / len(p)) / (p.sum() + ALPHA)
    a, err = calibrate_np(list(p))
    if err > 0.05:
        return None
    a = np.asarray(a)
    idx = [cands.index(c) for c in menu_small]
    lu = np.maximum(p[idx] / p[idx].sum(), FLOOR)
    w = win_probs_np(a[idx])
    return lu / lu.sum(), np.maximum(w / w.sum(), FLOOR)


def score(states, party, max_gap, synth=None):
    pairs = 0
    tot_l = tot_g = tot_w = 0.0
    diffs = []
    names = sorted(states)
    for A in names:
        mA = set(states[A])
        oA = dr_odds(states[A], party)
        if oA is None:
            continue
        for B in names:
            if B == A:
                continue
            mB = set(states[B])
            if not (mB < mA) or len(mB) < 2:
                continue
            oB = dr_odds(states[B], party)
            if oB is None or abs(oA - oB) > max_gap:
                continue
            src = synth[A] if synth else states[A]
            tgt = synth[B] if synth else states[B]
            pr = predict(src, sorted(mB))
            if pr is None:
                continue
            lu, ra = pr
            w = np.array([tgt[c] for c in sorted(mB)], dtype=float)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            dl = float(-(w * np.log(lu)).sum())
            dg = float(-(w * np.log(ra)).sum())
            tot_l += dl
            tot_g += dg
            tot_w += 1.0
            pairs += 1
            diffs.append(dl - dg)
    if pairs < 5:
        return None
    return {"pairs": pairs, "luce": tot_l / tot_w, "race": tot_g / tot_w,
            "gain": (tot_l - tot_g) / tot_w, "diffs": np.array(diffs)}


def luce_synth(states, rng):
    """Redraw each state's votes from a Luce process on its own menu, using the
    national shares of those candidates as worths, so the axiom holds by construction
    while the menus and vote totals stay exactly as observed."""
    nat = collections.Counter()
    for c in states.values():
        nat.update(c)
    out = {}
    for st, cnt in states.items():
        menu = sorted(cnt)
        u = np.array([nat[c] for c in menu], dtype=float)
        u = u / u.sum()
        n = int(sum(cnt.values()))
        draw = rng.multinomial(n, u / u.sum())
        out[st] = {c: int(v) for c, v in zip(menu, draw)}
    return out


def main():
    max_gap = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    states, party = load()
    sizes = collections.Counter(len(v) for v in states.values())
    print(f"{len(states)} states, menu sizes {dict(sorted(sizes.items()))}")
    r = score(states, party, max_gap)
    if not r:
        print("no usable nested pairs at this matching tolerance")
        return
    rng = np.random.default_rng(4)
    null = []
    for b in range(reps):
        s = score(states, party, max_gap, synth=luce_synth(states, rng))
        if s:
            null.append(s["gain"])
    null = np.array(sorted(null))
    med = float(np.median(null)) if len(null) else float("nan")
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1) if len(null) else float("nan")
    print(f"\nmatched nested pairs (|D:R log-odds gap| <= {max_gap}): {r['pairs']}")
    print(f"  renormalization {r['luce']:.4f}   race {r['race']:.4f}   "
          f"gain {r['gain']:+.4f}")
    print(f"  Luce null median {med:+.4f}   excess {r['gain']-med:+.4f}   "
          f"MC tail {pv:.3f}  ({len(null)} reps)")

    # pair-level bootstrap. The scoreboard calls a row a draw when this covers zero,
    # so every row needs one and not only the designed experiments.
    rb = random.Random(5)
    d = r["diffs"]
    bs = sorted(float(d[[rb.randrange(len(d)) for _ in range(len(d))]].mean())
                for _ in range(4000))
    print(f"  pair bootstrap 95% [{bs[100]:+.4f}, {bs[3900]:+.4f}]  "
          f"from {len(d)} pairs")


if __name__ == "__main__":
    main()
