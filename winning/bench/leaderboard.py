"""Render the leaderboard from bench_results/records.jsonl.

    python -m winning.bench.leaderboard

Per problem: methods ranked by wall time within accuracy bands, plus the
best time to reach each band. Written to bench_results/LEADERBOARD.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "bench_results"
BANDS = [(1e-3, "1e-3"), (5e-4, "5e-4")]


def main():
    recs = [json.loads(line) for line
            in (RESULTS / "records.jsonl").read_text().splitlines()]
    # keep the best (fastest) record per (problem, method, band)
    by_problem = defaultdict(list)
    for r in recs:
        by_problem[r["problem"]].append(r)
    lines = ["# Arena leaderboard", "",
             "Best wall time to reach each accuracy band "
             "(max-coordinate error vs the cached reference; the reference "
             "carries its own ~7e-4 noise, so sub-noise entries are "
             "reference-limited).", ""]
    for pid in sorted(by_problem):
        rows = by_problem[pid]
        n = rows[0]["n"]; k = rows[0]["k"]
        lines += [f"## {pid} (N={n}, k={k})", "",
                  "| band | winner | seconds | runner-up | seconds |",
                  "|---|---|---|---|---|"]
        for tol, label in BANDS:
            qual = defaultdict(lambda: float("inf"))
            for r in rows:
                if r["max_err"] <= tol:
                    qual[r["method"]] = min(qual[r["method"]], r["seconds"])
            ranked = sorted(qual.items(), key=lambda kv: kv[1])
            if not ranked:
                lines.append(f"| {label} | (none qualified) | | | |")
                continue
            w = ranked[0]
            ru = ranked[1] if len(ranked) > 1 else ("-", float("nan"))
            lines.append(f"| {label} | **{w[0]}** | {w[1]:.3f} "
                         f"| {ru[0]} | {ru[1]:.3f} |")
        lines.append("")
    (RESULTS / "LEADERBOARD.md").write_text("\n".join(lines) + "\n")
    print("wrote", RESULTS / "LEADERBOARD.md")


if __name__ == "__main__":
    main()
