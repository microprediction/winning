"""Scaling plot and table from the log of invert_structures.py.

Reads log_invert_structures.txt (the output of invert_structures.py at
n = 10,000, 100,000 and 1,000,000) and writes

  inversion_scaling.png   log-log wall-clock time of the inversion
                          against n, one line per covariance class,
                          with the fitted slope from the smallest to
                          the largest n in the label
  inversion_scaling.csv   one row per class and n: forward seconds,
                          inversion seconds, converged, sweeps, largest
                          log residual, largest ability error

No random numbers. Times are whatever the log holds; the log's first
lines name the machine.

    python plot_inversion_scaling.py [log] [outdir]
"""
import csv
import math
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "log_invert_structures.txt"
out = sys.argv[2] if len(sys.argv) > 2 else "."

ROW = re.compile(
    r"^(?P<kind>independent|factor r1|blocks|tree|nested)\s+"
    r"forward\s+(?P<fwd>[\d.]+) s\s+inversion\s+(?P<inv>[\d.]+) s\s+"
    r"converged (?P<conv>True|False)\s+sweeps\s+(?P<sweeps>\d+)\s+"
    r"max\|log p resid\| (?P<resid>\S+)\s+max\|mu err\| (?P<err>\S+)")

rows, n = [], None
for line in open(log):
    m = re.match(r"^n=([\d,]+)", line)
    if m:
        n = int(m.group(1).replace(",", ""))
        continue
    m = ROW.match(line)
    if m and n:
        rows.append(dict(kind=m["kind"], n=n, forward_s=float(m["fwd"]),
                         inversion_s=float(m["inv"]), converged=m["conv"],
                         sweeps=int(m["sweeps"]), max_log_resid=m["resid"],
                         max_mu_err=m["err"]))

with open(f"{out}/inversion_scaling.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# name, colour, marker: fixed order, one entry per class
SERIES = [("independent", "independent", "#2a78d6", "o"),
          ("factor r1", "factor, rank one", "#eb6834", "s"),
          ("blocks", "blocks", "#1baf7a", "^"),
          ("tree", "tree", "#eda100", "D"),
          ("nested", "nested", "#e87ba4", "v")]
INK, MUTED, GRID = "#1a1a19", "#5f5e58", "#e4e3de"

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
ends = []
for key, name, colour, marker in SERIES:
    pts = sorted((r["n"], r["inversion_s"]) for r in rows if r["kind"] == key)
    if not pts:
        continue
    xs, ys = zip(*pts)
    slope = (math.log(ys[-1] / ys[0]) / math.log(xs[-1] / xs[0])
             if len(pts) > 1 else float("nan"))
    ax.plot(xs, ys, color=colour, marker=marker, markersize=7, linewidth=2,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label=f"{name} (slope {slope:.2f})")
    ends.append((ys[-1], xs[-1], name))

# direct labels at the right-hand ends, nudged apart where they collide
ends.sort()
placed = []
for y, x, name in ends:
    ly = y
    if placed and ly < placed[-1] * 1.35:
        ly = placed[-1] * 1.35
    placed.append(ly)
    ax.annotate(name, (x, y), xytext=(x * 1.25, ly), color=INK, fontsize=9,
                va="center")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(7e3, 9e6)
ax.set_xlabel("contestants, n", color=INK)
ax.set_ylabel("inversion wall-clock time, seconds", color=INK)
ax.set_title("Share inversion to a log residual below 1e-8, by covariance class",
             color=INK, fontsize=11, loc="left")
ax.grid(True, which="major", color=GRID, linewidth=0.8)
ax.grid(False, which="minor")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
fig.tight_layout()
fig.savefig(f"{out}/inversion_scaling.png", facecolor=fig.get_facecolor())
print(f"{len(rows)} rows -> {out}/inversion_scaling.csv, {out}/inversion_scaling.png")
for r in rows:
    print(f"{r['kind']:12s} n={r['n']:>9,d}  forward {r['forward_s']:7.1f} s  "
          f"inversion {r['inversion_s']:8.1f} s  sweeps {r['sweeps']:3d}  "
          f"resid {r['max_log_resid']}  converged {r['converged']}")
