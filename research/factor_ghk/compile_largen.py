"""Compile runs/largen_n*.csv into markdown tables for RESULTS.md part 2.

    python research/factor_ghk/compile_largen.py

The CSVs are append-only: a resumed run appends again, possibly against a
larger truth (n=100,000 went from a 2^15 to a 2^17 truth).  A table must be
scored against ONE truth, so (#169):

  * the truth revision is the latest one in the file (largest truth_m, ties
    to the last appended); the header metadata is taken from its rows;
  * each (method, target) keeps its last row, and its errors are
    RECOMPUTED from the saved estimate `p` against that truth -- the stored
    abs_err / rel_err are never displayed, since a method that was not
    rerun after the resume carries errors against the old truth;
  * a method whose rows predate the latest truth is marked with a dagger;
  * a target the latest truth did not score is dropped, and said so.
"""
import csv
import glob
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))


def revision(r):
    return (int(r["truth_m"]), r["truth_vs_scramble"], r["truth_vs_mc"])


def compile_rows(rows):
    """Markdown for one field's rows (dicts as read from the CSV)."""
    if not rows:
        return ""
    # the latest truth: largest truth_m, and among equals the last appended
    latest_m = max(int(r["truth_m"]) for r in rows)
    rev = [revision(r) for r in rows if int(r["truth_m"]) == latest_m][-1]
    cur = [r for r in rows if revision(r) == rev]
    h = cur[0]
    truth = OrderedDict()
    for r in cur:
        truth[r["target"]] = float(r["truth_p"])
    # last row per (method, target), over the whole file
    last = OrderedDict()
    for r in rows:
        last[(r["method"], r["target"])] = r
    methods = list(OrderedDict.fromkeys(m for m, _ in last))
    targets = sorted(truth, key=lambda t: -truth[t])
    dropped = sorted({t for _, t in last if t not in truth})

    def err(m, t):
        p = float(last[(m, t)]["p"]); a = abs(p - truth[t])
        return a, a / truth[t]

    stale = {m for m in methods if any((m, t) in last and revision(last[(m, t)]) != rev for t in targets)}
    out = [f"\n### n = {int(h['n']):,}, rank {h['rank']}, D = {h['D']}, sharpness {float(h['sharp']):.1f}\n",
           f"Truth: Sobol 2^{rev[0]} (seed 101), vs second scramble {float(rev[1]):.1e}, vs MC {float(rev[2]):.1e}. "
           f"Read nothing below {2 * float(rev[1]):.0e}.\n"]
    if stale:
        out.append(f"† priced before the truth was extended to 2^{rev[0]}; its saved estimates are rescored "
                   f"against that truth here, its wall time is from the earlier run.\n")
    if dropped:
        out.append(f"Targets without a 2^{rev[0]} truth, not scored: {', '.join(dropped)}.\n")
    out.append("| method | passes | per target | max abs err | max rel err | wall s |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        ts = [t for t in targets if (m, t) in last]
        if not ts:
            continue
        r0 = last[(m, ts[0])]
        passes = int(r0["passes"]); budget = int(r0["budget"])
        per = budget if "Hybrid" in m else ("-" if "ghk" in m else passes)
        e = [err(m, t) for t in ts]
        name = m + (" †" if m in stale else "")
        out.append(f"| {name} | {passes if passes else '-'} | {per} | {max(a for a, _ in e):.1e} | {max(q for _, q in e):.1e} | {float(r0['seconds']):.1f} |")
    out.append("\nPer-target relative error, favourites left, longshots right:\n")
    out.append("| method | " + " | ".join(f"{truth[t]:.1e}" for t in targets) + " |")
    out.append("|---|" + "---:|" * len(targets))
    for m in methods:
        if not any((m, t) in last for t in targets):
            continue
        name = m + (" †" if m in stale else "")
        out.append(f"| {name} | " + " | ".join(f"{err(m, t)[1]:.1e}" if (m, t) in last else "-" for t in targets) + " |")
    return "\n".join(out)


def main(paths=None):
    paths = paths or sorted(glob.glob(os.path.join(HERE, "runs", "largen_n*.csv")), key=lambda p: int(os.path.basename(p)[8:-4]))
    for f in paths:
        with open(f) as fh:
            s = compile_rows(list(csv.DictReader(fh)))
        if s:
            print(s)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
