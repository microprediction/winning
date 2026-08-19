"""Wills, Reimers, Stewart, Suret and McLaren (2000, Experiment 2), two-choice condition.

This is the experiment whose published conclusion the present paper reaches independently:
"the ratio rule is an inappropriate theory of categorical decision and should be replaced
by a system based on the principles of Thurstonian choice." The authors did not, however,
score a parameter-free restriction map out of sample, which is what this does.

Design, from the archived codebook. Participants learned three categories A, B, C from
twelve-icon stimuli, then classified thirteen test stimulus types. Each participant has a
`fixed` category, the one for which training items contained four elements. Conditions:

  1  three-choice    test responses A, B, C all allowed        <- the master menu
  2  two-choice      the participant's `fixed` response is disallowed   <- the restriction
  3  novel-elements  a different manipulation, not a restriction, unused here

The restriction is between groups, twelve participants each, but it is balanced by
construction: three `fixed` values by thirteen stimulus types by forty trials, in both
conditions. Verified against the data -- the disallowed response occurs zero times in the
two-choice condition and 510 times in the three-choice condition, so the removed category
is a real competitor on the master menu.

A cell is (fixed, catordist). `catordist` means different things for different `fixed`
values, which is why cells must be matched on the pair rather than pooled over it.

  renormalization  keeps the master cell's odds between the two survivors
  Gaussian race    inverts the master cell's three shares for locations, drops the
                   disallowed category, re-runs the contest between the survivors

Forty trials per cell is thin, so a contraction map gains from shrinkage alone; the
fitted-Luce null is what separates that from structure and it is not a formality here.

Data: research/restriction/data/wills/, mirrored from
https://www.andywills.info/willslab-dau/cam1/  (Wills 2014, Data and Analysis Unit CAM1).

Usage:  python wills_twochoice.py [n_null_reps]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

DATA = HERE / "data" / "wills" / "cam1data.txt"
CATS = [1, 2, 3]
MASTER_COND, RESTRICTED_COND = 1, 2
TEST_PHASE = 2
ALPHA = 0.5
FLOOR = 1e-6


def load():
    """-> {cond: {subject: {cell: counts over A,B,C}}}, cell = (fixed, catordist)."""
    out = {MASTER_COND: defaultdict(lambda: defaultdict(lambda: np.zeros(3))),
           RESTRICTED_COND: defaultdict(lambda: defaultdict(lambda: np.zeros(3)))}
    with open(DATA) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cond, phase = int(r["cond"]), int(r["phase"])
            if phase != TEST_PHASE or cond not in out:
                continue
            cell = (int(r["fixed"]), int(r["catordist"]))
            out[cond][r["subj"]][cell][int(r["resp"]) - 1] += 1
    return out


def cell_counts(bysubj, subjects):
    """Pool the given participants into per-cell response counts."""
    agg = defaultdict(lambda: np.zeros(3))
    for s in subjects:
        for cell, c in bysubj[s].items():
            agg[cell] = agg[cell] + c
    return agg


def predictions(master, fixed):
    """Both maps from one master cell. Neither sees a two-choice observation."""
    p = (master + ALPHA) / (master.sum() + ALPHA * 3)
    keep = [i for i in range(3) if i != fixed - 1]
    luce = p[keep] / p[keep].sum()
    a, err = calibrate_np(list(p))
    w = win_probs_np(np.asarray(a)[keep])
    return keep, np.maximum(luce, FLOOR), np.maximum(w / w.sum(), FLOOR), err


def score(master_cells, restricted_cells):
    tot_l = tot_g = 0.0
    n = 0
    gains, maxerr = [], 0.0
    for cell, obs in sorted(restricted_cells.items()):
        if cell not in master_cells:
            continue
        fixed = cell[0]
        keep, luce, race, err = predictions(master_cells[cell], fixed)
        maxerr = max(maxerr, err)
        o = obs[keep]
        if o.sum() == 0:
            continue
        ll = -float((o * np.log(luce)).sum())
        lg = -float((o * np.log(race)).sum())
        tot_l += ll
        tot_g += lg
        n += int(o.sum())
        gains.append((ll - lg) / o.sum())
    if n == 0:
        return None
    return {"luce": tot_l / n, "race": tot_g / n, "gain": (tot_l - tot_g) / n,
            "n": n, "cells": len(gains), "gains": np.array(gains), "cal_err": maxerr}


def null_rep(master_cells, restricted_cells, rng):
    """Luce is true: master and restricted counts drawn from the same worths."""
    m_syn, r_syn = {}, {}
    for cell, obs in restricted_cells.items():
        if cell not in master_cells:
            continue
        mc = master_cells[cell]
        p_true = mc / mc.sum()
        keep = [i for i in range(3) if i != cell[0] - 1]
        q = p_true[keep] / p_true[keep].sum()
        m_syn[cell] = rng.multinomial(int(mc.sum()), p_true).astype(float)
        draw = rng.multinomial(int(obs[keep].sum()), q).astype(float)
        full = np.zeros(3)
        for j, i in enumerate(keep):
            full[i] = draw[j]
        r_syn[cell] = full
    return score(m_syn, r_syn)["gain"]


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    d = load()
    msubs, rsubs = sorted(d[MASTER_COND]), sorted(d[RESTRICTED_COND])
    mc = cell_counts(d[MASTER_COND], msubs)
    rc = cell_counts(d[RESTRICTED_COND], rsubs)
    r = score(mc, rc)

    # participant bootstrap on both groups at once: shares and targets both move
    rng = np.random.default_rng(3)
    boot = []
    for _ in range(2000):
        mb = cell_counts(d[MASTER_COND], list(rng.choice(msubs, len(msubs))))
        rb = cell_counts(d[RESTRICTED_COND], list(rng.choice(rsubs, len(rsubs))))
        rr = score(mb, rb)
        if rr:
            boot.append(rr["gain"])
    lo, hi = np.quantile(boot, [0.025, 0.975])

    rng2 = np.random.default_rng(202)
    null = np.array(sorted(null_rep(mc, rc, rng2) for _ in range(reps)))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)

    print(f"Wills et al. (2000) Exp 2: three-choice -> two-choice, "
          f"{len(msubs)} vs {len(rsubs)} participants")
    print(f"  {r['cells']} cells, {r['n']} held-out two-choice trials")
    print(f"  renormalization log loss   {r['luce']:.4f}")
    print(f"  Gaussian race log loss     {r['race']:.4f}")
    print(f"  gain                       {r['gain']:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fitted-Luce null median    {med:+.4f}  ({len(null)} reps)")
    print(f"  excess over null           {r['gain'] - med:+.4f}   p = {pv:.3f}")
    print(f"  worst calibration residual {r['cal_err']:.2e}")

    print("\nby disallowed category")
    for fx in CATS:
        sub_r = {c: v for c, v in rc.items() if c[0] == fx}
        rr = score(mc, sub_r)
        print(f"  fixed = {fx}: {rr['cells']:2d} cells, {rr['n']:4d} trials, "
              f"gain {rr['gain']:+.4f}")

    print("\ndummy stimuli (catordist 10-13) versus the graded series (1-9)")
    for label, sel in (("graded 1-9", range(1, 10)), ("dummy 10-13", range(10, 14))):
        sub_r = {c: v for c, v in rc.items() if c[1] in sel}
        rr = score(mc, sub_r)
        print(f"  {label:12}: {rr['cells']:2d} cells, {rr['n']:4d} trials, "
              f"gain {rr['gain']:+.4f}")


if __name__ == "__main__":
    main()
