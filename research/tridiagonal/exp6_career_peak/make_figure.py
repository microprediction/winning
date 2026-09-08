"""Figure for the win-nodes paper: the posterior ability path a rating
system reports, and underneath it the query that path cannot answer.

Two panels, one shared time axis. Two measures on different scales get
two panels rather than two y-scales. Single series per panel, so no
legend and no categorical palette; greyscale only, so it survives
print and any form of colour vision.

  julia dump_figure.jl tennis_paths.json fig.json
  python make_figure.py fig.json ../../../papers/win_nodes/nadal.pdf
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

d = json.load(open(sys.argv[1]))
years = np.array(d["years"])
mean = np.array(d["mean"])
sd = np.array(d["sd"])
P = np.array(d["argmax"])

fig, (ax0, ax1) = plt.subplots(
    2, 1, figsize=(5.6, 3.9), sharex=True,
    gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.12})

ax0.fill_between(years, mean - 2 * sd, mean + 2 * sd,
                 color="0.85", linewidth=0)
ax0.plot(years, mean, color="0.15", linewidth=1.4)
ax0.set_ylabel("posterior ability")
ax0.text(0.015, 0.9, "posterior path, $\\pm 2$ s.d.", transform=ax0.transAxes,
         fontsize=8.5, color="0.3")
for s in ("top", "right"):
    ax0.spines[s].set_visible(False)
ax0.tick_params(labelsize=8)

top = np.argsort(P)[-2:]
colors = ["0.75"] * len(P)
for t in top:
    colors[t] = "0.2"
ax1.bar(years, P, width=0.72, color=colors, linewidth=0)
ax1.set_ylabel("$\\Pr$(peak year)")
ax1.set_xlabel("season")
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.tick_params(labelsize=8)
ax1.set_ylim(0, max(P) * 1.32)
for t in sorted(top):
    ax1.annotate(f"{years[t]}\n{P[t]:.2f}", (years[t], P[t]),
                 textcoords="offset points", xytext=(0, 4),
                 ha="center", fontsize=8, color="0.15")

fig.savefig(sys.argv[2], bbox_inches="tight", pad_inches=0.02)
print(f"wrote {sys.argv[2]}")
