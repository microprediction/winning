"""Townsend and Landon (1982): the constant-ratio rule on its own ground.

This is the origin of the empirical literature the paper joins. Four subjects identified
tachistoscopic letters from a master set {A,E,F,H,X} and from three subsets, {A,E,F,H},
{A,E,X} and {F,H,X}, each run as a separate block, each letter presented 240 times per
block. The authors compared observed subset confusions against the constant-ratio rule,
which is proportional renormalization of the master row, and reported that it holds "to a
reasonable first approximation" but fails systematically: confusions concentrate onto the
surviving near-substitute rather than spreading proportionally.

Each row of a confusion matrix is a restriction problem in its own right. Given stimulus
A, the master row is a distribution over five responses; the subset row is a distribution
over the survivors. Nothing in the subset blocks informs either prediction, so this is out
of sample without needing a fold structure, and because master and subset come from the
same subject there is no population-heterogeneity confound at all.

  renormalization  keeps the master row's odds among survivors, which is the CRR exactly
  Gaussian race    inverts the master row for locations, drops the removed responses,
                   re-runs the contest among survivors

Both are parameter free and neither sees a subset observation. Scored by log loss against
the held-out subset counts, against a fitted-Luce null at this sample size, since a
contraction map gains on noisy shares even when Luce is true.

Data: research/restriction/notes/crr/digitized/, OCR of Tables 1-4 verified two ways -- every
row sums to 240, and the paper's own printed CRR entries recompute from the recovered
master counts. Two cells are unresolvable errors in the published table and their rows are
dropped; --repair runs the sensitivity check with the arithmetically forced values.

Usage:  python townsend_rows.py [n_null_reps] [--repair]
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "polysemy_pilot"))
from exact_analyze import calibrate_np, win_probs_np

DIG = HERE / "notes" / "crr" / "digitized"
MASTER = "AEFHX"
SUBSETS = ["AEFH", "AEX", "FHX"]
SUBJECTS = ["s1", "s2", "s3", "s4"]
NPER = 240
ALPHA = 0.5
FLOOR = 1e-6

# Cells the published table prints inconsistently: the row does not sum to 240 and the
# image matches the text, so these are the authors' errors, not OCR. The arithmetic
# repair is unique in both cases but remains an inference, so it is opt-in.
REPAIRS = {("s1", "AEFH", "H", "A"): 40, ("s4", "FHX", "F", "H"): 56}


def read(subject, name, repair=False):
    """-> (response letters, {stimulus: counts}), dropping rows with unresolved cells."""
    path = DIG / f"townsend1982_{subject}_{name}.csv"
    rows = {}
    with open(path) as f:
        rdr = csv.reader(f)
        head = next(rdr)[1:]
        for rec in rdr:
            stim, vals = rec[0], rec[1:]
            out = []
            for letter, v in zip(head, vals):
                if v.strip() == "NA":
                    fix = REPAIRS.get((subject, name, stim, letter))
                    if repair and fix is not None:
                        out.append(float(fix))
                    else:
                        out = None
                        break
                else:
                    out.append(float(v))
            if out is not None:
                rows[stim] = np.array(out)
    return head, rows


def predictions(master_counts, master_letters, subset_letters):
    """Both maps, from one master row only. Neither sees a subset observation."""
    p = (master_counts + ALPHA) / (master_counts.sum() + ALPHA * len(master_counts))
    idx = [master_letters.index(c) for c in subset_letters]
    luce = p[idx] / p[idx].sum()
    a, err = calibrate_np(list(p))
    w = win_probs_np(np.asarray(a)[idx])
    race = w / w.sum()
    return np.maximum(luce, FLOOR), np.maximum(race, FLOOR), err


def rows_for(repair=False):
    """Every (subject, subset, stimulus) restriction with both matrices present."""
    out = []
    for s in SUBJECTS:
        mletters, mrows = read(s, MASTER, repair)
        for name in SUBSETS:
            sletters, srows = read(s, name, repair)
            for stim, obs in srows.items():
                if stim in mrows:
                    out.append((s, name, stim, mletters, mrows[stim], sletters, obs))
    return out


def score(rows):
    """Log loss per held-out trial, plus the per-row gains for bootstrapping."""
    tot_l = tot_g = 0.0
    n = 0
    gains, maxerr = [], 0.0
    for _, _, _, ml, mc, sl, obs in rows:
        luce, race, err = predictions(mc, ml, sl)
        maxerr = max(maxerr, err)
        ll = -float((obs * np.log(luce)).sum())
        lg = -float((obs * np.log(race)).sum())
        tot_l += ll
        tot_g += lg
        n += int(obs.sum())
        gains.append((ll - lg) / obs.sum())
    return {"luce": tot_l / n, "race": tot_g / n, "gain": (tot_l - tot_g) / n,
            "n": n, "rows": len(rows), "gains": np.array(gains), "cal_err": maxerr}


def null_rep(rows, rng):
    """Luce is true: master counts and subset counts both drawn from the same worths."""
    syn = []
    for s, name, stim, ml, mc, sl, obs in rows:
        p_true = mc / mc.sum()
        idx = [ml.index(c) for c in sl]
        q_true = p_true[idx] / p_true[idx].sum()
        syn.append((s, name, stim, ml,
                    rng.multinomial(int(mc.sum()), p_true).astype(float),
                    sl, rng.multinomial(int(obs.sum()), q_true).astype(float)))
    return score(syn)["gain"]


def report(label, rows, reps, rng_seed=7):
    r = score(rows)
    rng = np.random.default_rng(rng_seed)
    boot = [float(np.average(r["gains"][rng.integers(0, len(rows), len(rows))]))
            for _ in range(2000)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    rng2 = np.random.default_rng(101)
    null = np.array(sorted(null_rep(rows, rng2) for _ in range(reps)))
    med = float(np.median(null))
    pv = (float((null >= r["gain"]).sum()) + 1) / (len(null) + 1)
    print(f"\n{label}: {r['rows']} restricted rows, {r['n']} held-out trials")
    print(f"  renormalization (CRR) log loss  {r['luce']:.4f}")
    print(f"  Gaussian race log loss          {r['race']:.4f}")
    print(f"  gain                            {r['gain']:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print(f"  fitted-Luce null median         {med:+.4f}  ({len(null)} reps)")
    print(f"  excess over null                {r['gain'] - med:+.4f}   p = {pv:.3f}")
    print(f"  worst calibration residual      {r['cal_err']:.2e}")
    return r


def main():
    reps = 200
    repair = "--repair" in sys.argv
    for arg in sys.argv[1:]:
        if arg.isdigit():
            reps = int(arg)
    rows = rows_for(repair)
    report("all subsets pooled" + (" (repaired cells)" if repair else ""), rows, reps)
    for name in SUBSETS:
        sub = [r for r in rows if r[1] == name]
        if sub:
            report(f"subset {name}", sub, reps)
    print("\nper-row detail (positive gain favours the Gaussian race)")
    print(f"  {'subj':4} {'subset':6} {'stim':4} {'gain':>9}   observed vs CRR vs race")
    for (s, name, stim, ml, mc, sl, obs) in rows:
        luce, race, _ = predictions(mc, ml, sl)
        g = (-(obs * np.log(luce)).sum() + (obs * np.log(race)).sum()) / obs.sum()
        o = obs / obs.sum()
        cells = "  ".join(f"{c}:{o[i]:.3f}/{luce[i]:.3f}/{race[i]:.3f}"
                          for i, c in enumerate(sl))
        print(f"  {s:4} {name:6} {stim:4} {g:+9.4f}   {cells}")


if __name__ == "__main__":
    main()
