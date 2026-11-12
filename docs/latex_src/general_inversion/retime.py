"""Regenerate every wall-clock number in the paper, on a machine quiet
enough for the answer to mean something.

Why this exists: accuracy claims survive a busy machine, timings do not.
The saturated table's RMSE column reproduced exactly after the sixth
review's Jacobian repair, but its seconds column could not be checked --
load average was 10.7 and the two MSL arms, which never touch the
changed code, slowed by 2.6x and 1.7x. A number that moves when a
neighbouring process wakes up is not a measurement.

Usage:

    python retime.py                # every task, refusing a busy machine
    python retime.py --list         # what would run, and its paper value
    python retime.py --only saturated,grammar_inversion
    python retime.py --skip tenmillion,invert_million   # the hour-long ones
    python retime.py --force        # measure anyway, and say so in the output
    python retime.py --json out.json

Each task names the paper value it is checking and where that value sits,
so a discrepancy points at a line to edit rather than at a mystery. Tasks
are ordered cheapest first, so a broken environment fails in seconds
rather than after the ten-million-contestant run.

Total runtime on an idle machine is roughly one hour, dominated by
tenmillion (~4 min plus ~5 GB of memory) and invert_million (~25 min).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BENCH = os.path.join(HERE, "bench.py")
MLE2 = os.path.join(ROOT, "research", "mnp_estimation", "run_mle2.py")


class Task:
    def __init__(self, key, paper, where, argv, minutes, note=""):
        self.key = key
        self.paper = paper          # the value(s) printed in the paper
        self.where = where          # section, so a mismatch is actionable
        self.argv = argv
        self.minutes = minutes      # rough cost on an idle machine
        self.note = note


TASKS = [
    Task("per_alternative", "17.5 s against 14 ms at n=200 rank 2",
         "Benchmarks, per-alternative comparator",
         [sys.executable, BENCH, "bm"], 1),
    Task("grammar_inversion",
         "independent 0.05 s, blocks 0.17 s, tree 0.20 s, factor rank-2 "
         "0.57 s, nested 1.18 s",
         "Inversion, per-grammar round trips (n=400)",
         [sys.executable, BENCH, "invert"], 2),
    Task("ghk", "race 2.9 / 4.8 / 15 ms; GHK R=1e3 0.9 / 11 / 170 ms, "
         "R=1e4 5.9 / 92 / 1840 ms at n=10 / 50 / 200",
         "Benchmarks, against GHK (table)",
         [sys.executable, BENCH, "ghk"], 1),
    Task("law", "GHK R=1e3 0.18, 1.69, 19.4 s at n=200, 500, 1000; "
         "exponents 1.9 / 2.9",
         "Introduction footnote; Appendix A anchor t(1000) at R=1e4",
         [sys.executable, BENCH, "law"], 2),
    Task("genz200", "one n=200 probability 0.57 s at maxpts=2.5e5; "
         "full vector 114 s against 14 ms",
         "Benchmarks, against the field (n=200 paragraph)",
         [sys.executable, BENCH, "genz200"], 1,
         "needs R with mvtnorm and jsonlite"),
    Task("tilting", "n=50: 0.3 s (R=1e3) vs lattice 0.07 s; n=200: 31 s, "
         "50 s (R=1e4) vs 0.28 s",
         "Related work, minimax tilting paragraph",
         [sys.executable, os.path.join(ROOT, "research", "experiments",
                                       "exp22_minimax_tilting",
                                       "run_tilting.py")], 10,
         "rewrites exp22's results.csv"),
    Task("saturated", "exact 3.0 s, MSL-100 2.6 s, MSL-1000 18.4 s",
         "Estimation, saturated table (RMSE column verified separately "
         "and unchanged: 0.0148 / 0.0275 / 0.0161)",
         [sys.executable, MLE2], 10),
    Task("covariates", "exact 77.5 s, MSL-100 11.1 s, MSL-1000 27.9 s",
         "Estimation with covariates",
         [sys.executable, os.path.join(ROOT, "research", "mnp_estimation",
                                       "run_mle3.py")], 15),
    Task("large_n", "fit 1.0 s, one call 3.6 s at n=2000; 1e6 draws 18.5 s",
         "Large-n benchmark (general Sigma)",
         [sys.executable, os.path.join(ROOT, "research", "general_sigma",
                                       "run_large_n.py")], 1),
    Task("scale", "n=1e4, 1e5, 1e6 forward in 0.12, 1.2, 18 s",
         "Scale paragraph, forward pricing",
         [sys.executable, BENCH, "scale"], 2,
         "rank-one factor; needs the Rust kernels for the paper's figure"),
    Task("invert_million", "n=1e6 inversion in 105 s independent, 12 min "
         "correlated",
         "Scale paragraph, inversion",
         [sys.executable, BENCH, "invertmillion"], 25),
    Task("tenmillion", "ten-million block field in 102 s",
         "Scale paragraph, block grammar",
         [sys.executable, BENCH, "tenmillion"], 4,
         "needs roughly 5 GB of memory"),
]


def machine_is_quiet(per_core_limit=0.25):
    """(quiet, description). Load average per core, plus the loudest
    processes, because 'the machine was busy' is the single most common
    reason a benchmark disagrees with a published one.

    A quarter of the cores busy is the line: the tasks here are
    single-threaded by construction (the runner pins the BLAS thread
    counts), so they need one free core each, not an idle machine, but
    they do need the memory bandwidth and the turbo headroom that a
    saturated machine does not have."""
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return True, "load average unavailable on this platform"
    cores = os.cpu_count() or 1
    per_core = one / cores
    desc = (f"load {one:.1f} / {five:.1f} / {fifteen:.1f} over {cores} "
            f"cores = {per_core:.2f} per core (limit {per_core_limit})")
    return per_core < per_core_limit, desc


def loud_processes(limit=5):
    try:
        out = subprocess.run(["ps", "axo", "pcpu,comm"], capture_output=True,
                             text=True, timeout=20).stdout.splitlines()[1:]
    except Exception:
        return []
    rows = []
    for line in out:
        parts = line.split(None, 1)
        if len(parts) == 2:
            try:
                pct = float(parts[0])
            except ValueError:
                continue
            if pct > 20.0:
                rows.append((pct, parts[1].strip()))
    rows.sort(reverse=True)
    return rows[:limit]


def run(task, env):
    t0 = time.perf_counter()
    proc = subprocess.run(task.argv, capture_output=True, text=True, env=env,
                          cwd=os.path.dirname(task.argv[1]))
    wall = time.perf_counter() - t0
    return {
        "key": task.key,
        "paper": task.paper,
        "where": task.where,
        "wall_seconds": round(wall, 2),
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip()[-2000:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--force", action="store_true",
                    help="measure on a busy machine and label the result")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    chosen = TASKS
    if args.only:
        keep = set(args.only.split(","))
        chosen = [t for t in chosen if t.key in keep]
    if args.skip:
        drop = set(args.skip.split(","))
        chosen = [t for t in chosen if t.key not in drop]

    if args.list:
        total = sum(t.minutes for t in chosen)
        for t in chosen:
            print(f"{t.key:20s} ~{t.minutes:3d} min   paper: {t.paper}")
            print(f"{'':20s}              {t.where}")
            if t.note:
                print(f"{'':20s}              note: {t.note}")
        print(f"\ntotal ~{total} min")
        return 0

    quiet, desc = machine_is_quiet()
    loud = loud_processes()
    print(f"machine: {platform.platform()}")
    print(f"         {desc}")
    if loud:
        print("         busiest: " +
              ", ".join(f"{p:.0f}% {c}" for p, c in loud))
    if not quiet and not args.force:
        print("\nREFUSING to measure on a busy machine. Timings taken here "
              "would not be comparable with the paper's, which is the whole "
              "point of this script. Wait for the machine to go quiet, or "
              "pass --force and treat the output as indicative only.")
        return 2
    if not quiet:
        print("\nWARNING: --force on a busy machine. These numbers are "
              "indicative only and must not go into the paper.")

    env = dict(os.environ)
    # keep the numeric stack single-threaded so the comparison is of the
    # algorithm rather than of how many cores happened to be free
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(var, "1")

    results = []
    for t in chosen:
        print(f"\n=== {t.key} (~{t.minutes} min) ===")
        print(f"paper: {t.paper}")
        print(f"       {t.where}")
        r = run(t, env)
        r["machine_quiet"] = bool(quiet)
        results.append(r)
        if r["returncode"] != 0:
            print(f"FAILED (rc {r['returncode']})")
            if r["stderr"]:
                print(r["stderr"][-800:])
        else:
            print(r["stdout"])
        print(f"[wall {r['wall_seconds']:.1f} s]")

    print("\n" + "=" * 70)
    print("SUMMARY -- compare each measured line against the paper value "
          "and edit main.tex where they disagree")
    for r in results:
        status = "ok" if r["returncode"] == 0 else "FAILED"
        print(f"  {r['key']:20s} {status:6s} {r['wall_seconds']:8.1f} s wall"
              f"   paper: {r['paper']}")
    if not quiet:
        print("\n(taken on a busy machine with --force: indicative only)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"machine": platform.platform(), "load": desc,
                       "quiet": bool(quiet), "results": results}, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
