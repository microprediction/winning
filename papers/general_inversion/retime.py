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
    Task("per_alternative", "22 s against 36 ms at n=200 rank 2",
         "Benchmarks, per-alternative comparator",
         [sys.executable, BENCH, "bm"], 1),
    Task("grammar_inversion",
         "tree 0.37 s, blocks 0.67 s, factor rank-2 1.7 s, nested 3.7 s",
         "Inversion, per-grammar round trips (n=400)",
         [sys.executable, BENCH, "invert"], 2),
    Task("ghk", "GHK 5.5, 16.6, 51.3 s at n=200, 500, 1000, R=1e4",
         "Benchmarks, GHK comparator",
         [sys.executable, BENCH, "ghk"], 5),
    Task("saturated", "exact 2.4 s, MSL-100 3.5 s, MSL-1000 39.9 s",
         "Estimation, saturated table (RMSE column verified separately "
         "and unchanged: 0.0148 / 0.0275 / 0.0161)",
         [sys.executable, MLE2], 10),
    Task("scale", "n=1e4, 1e5, 1e6 forward in 0.18, 2.7, 29 s",
         "Scale paragraph, forward pricing",
         [sys.executable, BENCH, "scale"], 2,
         "rank-one factor; needs the Rust kernels for the paper's figure"),
    Task("invert_million", "n=1e6 inversion in 80 s independent, 22 min "
         "correlated",
         "Scale paragraph, inversion",
         [sys.executable, BENCH, "invertmillion"], 25),
    Task("tenmillion", "ten-million block field in 245 s",
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
    proc = subprocess.run(task.argv, capture_output=True, text=True, env=env)
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
