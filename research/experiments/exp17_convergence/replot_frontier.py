"""Regenerate the accuracy-time frontier figure from committed results.csv.

One labeled point per method (its highest-effort tested setting), scored on
one fixed problem against one common reference; the shaded band marks the
reference's own noise, below which errors are unresolvable (the lattice
point sits at that floor; its true error is ~1e-8 by self-convergence).

Run:  python experiments/exp17_convergence/replot_frontier.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent

rows = [r.split(",") for r in (HERE / "results.csv").read_text().splitlines()]
noise = float([r[2] for r in rows if r[1] == "truth_noise"][0])
pts = {}
for r in rows:
    if r[0] == "D" and " " in r[-1]:
        meth = r[1].rsplit("_", 1)[0].replace("_", " ")
        t_s, err_s = r[-1].split(" ")
        pts.setdefault(meth, []).append((float(t_s.rstrip("s")), float(err_s)))

series = {
    "lattice": ("o", "#c2410c", "lattice\n(L=375, GH5)"),
    "direct MC": ("s", "#5b7c99", "direct MC\n($R=10^7$)"),
    "QMC direct": ("d", "#2a9d8f", "Sobol direct\n($2^{20}$ pts)"),
    "GHK": ("^", "#9a9a9a", "GHK\n($R=10^4$)"),
    "QMC-GHK": ("v", "#6a5acd", "QMC-GHK\n($R=2^{13}$)"),
}

fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.axhspan(1e-6, noise, color="#dddddd", alpha=0.6, zorder=0)
ax.text(0.045, noise * 0.7, "below the reference's own noise: unresolvable\n"
        "(lattice true error $\\sim 10^{-8}$ by self-convergence)",
        fontsize=7.5, color="#555555", va="top")
for key, (mk, c, lab) in series.items():
    data = sorted(sum([v for k, v in pts.items()
                       if k.lower().startswith(key.lower())], []))
    if not data:
        continue
    # lattice: all settings sit at the reference floor, so show the
    # cheapest; stochastic methods: show the highest-effort (most accurate)
    t, e = data[0] if key == "lattice" else data[-1]
    ax.loglog([t], [e], mk, color=c, ms=9, zorder=3)
    ax.annotate(lab.replace("\\n", "\n"), (t, e), textcoords="offset points",
                xytext=(8, 4), fontsize=8, color=c)
ax.set_xlabel("wall time (s)")
ax.set_ylabel("max abs share error vs common reference")
ax.set_title("Same problem, same reference: each method at its most\n"
             "accurate tested setting (N=200, k=2, all 200 shares)",
             fontsize=10)
ax.set_xlim(0.02, 60)
ax.set_ylim(2e-5, 2e-2)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(HERE / "figures" / "frontier_full.png", dpi=150)
print("rewrote figures/frontier_full.png")
