"""The six figures.

1  rank vs qPO efficiency          -- does the posterior have low effective rank
2  rank vs top-100 recall          -- does the batch itself survive
3  N vs seconds, log-log           -- runtime scaling of each method
4  wall time vs efficiency         -- the accuracy/runtime frontier
5  batch Tanimoto distributions    -- does correlation still buy diversity
6  closed-loop discovery           -- acquired molecules vs top-1% recovered
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
RES = HERE / "results"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})
C = {"fast": "#c2410c", "contrast": "#ea580c", "factorMC": "#1d4ed8",
     "dense": "#111827", "lite": "#059669", "ucb": "#7c3aed",
     "greedy": "#9ca3af", "alite": "#34d399"}


def _sweeps():
    out = []
    for p in sorted(RES.glob("*/sweep.csv")):
        out.append(pd.read_csv(p))
    return pd.concat(out, ignore_index=True) if out else None


# --------------------------------------------------------------------------

LABEL = {"qm9_gap_seed7": "QM9 seed 7 (UCB pool)",
         "qm9_gap_seed11": "QM9 seed 11 (UCB pool)",
         "qm9_gap_seed7_random": "QM9 random pool",
         "qm9_gap_bo0_iter10": "QM9 posterior at round 10",
         "wong_antibiotics_Mean_50uM_seed7_min": "antibiotic screen"}
ORDER = list(LABEL)


def _snapshots(df, N=1000):
    """One curve per posterior, at a fixed universe size."""
    out = []
    for tag in ORDER:
        g = df[(df.snapshot == tag) & (df.N == N)]
        if len(g):
            out.append((tag, g))
    return out


def fig1_rank_vs_efficiency(df, N=1000):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, metric, label in (
            (axes[0], "qpo_efficiency", "qPO efficiency  $\\eta_r$"),
            (axes[1], "top100_recall", "top-100 recall against reference")):
        for tag, g in _snapshots(df, N):
            h = g[g.method == "qPO-fast-eig"].sort_values("rank")
            line, = ax.plot(np.maximum(h["rank"], 0.7), h[metric], "o-", ms=3.5,
                            lw=1.4, label=LABEL[tag])
            # same models, scored by factor Monte Carlo, which has no quadrature
            k = g[g.method == "qPO-factorMC"].sort_values("rank")
            if len(k):
                ax.plot(np.maximum(k["rank"], 0.7), k[metric], "--", lw=1.1,
                        alpha=0.55, dashes=(4, 2), color=line.get_color())
        ref = df[(df.method == "reference-2nd-seed") & (df.N == N)][metric]
        if len(ref):
            ax.axhline(float(ref.mean()), color="k", ls=":", lw=1)
            ax.text(0.75, float(ref.mean()), " reference vs itself", fontsize=7,
                    va="bottom")
        mc = df[(df.method.astype(str).str.startswith("qPO-MC-")) & (df.N == N)][metric]
        if len(mc):
            ax.axhline(float(mc.mean()), color="#111827", ls="--", lw=1)
            ax.text(0.75, float(mc.mean()), " shipped qPO default", fontsize=7,
                    va="top")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("factor rank $r$   (0 plotted at 0.7)")
        ax.set_ylabel(label)
    axes[0].plot([], [], "k-", lw=1.4, label="fast probit (fixed nodes)")
    axes[0].plot([], [], "k--", lw=1.1, dashes=(4, 2), label="same model, factor MC")
    axes[0].legend(fontsize=6.2, loc="lower center", ncol=2)
    axes[0].set_title(f"Does qPO have low effective rank?  (N = {N})", fontsize=10)
    axes[1].set_title("Does the batch itself survive?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_rank_vs_efficiency.png")
    plt.close(fig)


def fig2_rank_vs_recall(df, N=1000):
    """Batch agreement across universe sizes, one panel per posterior family."""
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for tag in ORDER:
        g = df[df.snapshot == tag]
        if not len(g):
            continue
        h = g[g.method == "qPO-fast-eig"].groupby("rank")["top100_recall"]
        m, lo, hi = h.mean(), h.min(), h.max()
        line, = ax.plot(np.maximum(m.index, 0.7), m.values, "o-", ms=3.5, lw=1.4,
                        label=LABEL[tag])
        ax.fill_between(np.maximum(m.index, 0.7), lo.values, hi.values,
                        alpha=0.12, lw=0, color=line.get_color())
    ref = df[df.method == "reference-2nd-seed"]["top100_recall"]
    if len(ref):
        ax.axhline(float(ref.mean()), color="k", ls=":", lw=1)
    mc = df[df.method.astype(str).str.startswith("qPO-MC-")]["top100_recall"]
    if len(mc):
        ax.axhline(float(mc.mean()), color="#111827", ls="--", lw=1)
        ax.text(0.75, float(mc.mean()), " shipped qPO default", fontsize=7,
                va="bottom")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("factor rank $r$   (0 plotted at 0.7)")
    ax.set_ylabel("top-100 recall against reference")
    ax.set_title("Batch agreement, band over N = 500 to 2000", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rank_vs_recall.png")
    plt.close(fig)


def fig3_runtime_scaling():
    f = RES / "phase6_scaling.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    ranks = sorted(d["rank"].unique())
    r0 = ranks[min(2, len(ranks) - 1)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    g = d[d["rank"] == r0].sort_values("N")
    axes[0].loglog(g.N, g.fast_pom_seconds, "o-", color=C["fast"], ms=4,
                   label="fast factor probit")
    axes[0].loglog(g.N, g.factormc_pom_seconds, "s-", color=C["factorMC"], ms=4,
                   label=f"factor MC ({int(g.mc_samples.iloc[0]):,})")
    axes[0].loglog(g.N, g.flite_seconds, "^-", color=C["lite"], ms=4,
                   label="F-LITE")
    dm = g.dropna(subset=["dense_mc10k_seconds"])
    axes[0].loglog(dm.N, dm.dense_mc10k_seconds, "d-", color=C["dense"], ms=4,
                   label="dense MC-qPO (10k)")
    if len(dm) and len(g) > len(dm):
        axes[0].axvline(dm.N.max() * 1.4, color="k", ls=":", lw=1)
        axes[0].text(dm.N.max() * 1.5, g.fast_pom_seconds.min(),
                     " dense $\\Sigma$ no longer fits", fontsize=7, rotation=90)
    axes[0].set_xlabel("candidates $N$")
    axes[0].set_ylabel("seconds")
    axes[0].set_title(f"scaling in $N$ (rank {r0})", fontsize=10)
    axes[0].legend(fontsize=7)

    N0 = sorted(d.N.unique())[min(2, len(d.N.unique()) - 1)]
    g = d[d.N == N0].sort_values("rank")
    axes[1].loglog(g["rank"], g.fast_pom_seconds, "o-", color=C["fast"], ms=4,
                   label="fast factor probit")
    axes[1].loglog(g["rank"], g.factormc_pom_seconds, "s-", color=C["factorMC"],
                   ms=4, label="factor MC")
    axes[1].set_xlabel("factor rank $r$")
    axes[1].set_ylabel("seconds")
    axes[1].set_title(f"scaling in $r$ (N = {N0:,})", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_runtime_scaling.png")
    plt.close(fig)


def fig4_frontier():
    files = sorted(RES.glob("budget_*.csv"))
    if not files:
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, metric, lab in ((axes[0], "qpo_efficiency", "qPO efficiency $\\eta$"),
                            (axes[1], "top100_recall", "top-100 recall")):
        g = d[d.method == "qPO-dense-MC"].groupby("budget").agg(
            {"seconds": "mean", metric: "mean"}).sort_values("seconds")
        ax.semilogx(g.seconds, g[metric], "d-", color=C["dense"], ms=4,
                    label="dense MC-qPO (shipped)")
        for rank, gg in d[d.method == "qPO-fast"].groupby("rank"):
            gg = gg.sort_values("seconds")
            ax.semilogx(gg.seconds, gg[metric], "o-", ms=3.5,
                        label=f"fast probit r={rank}", alpha=0.9)
        for rank, gg in d[d.method == "qPO-factorMC"].groupby("rank"):
            gg = gg.sort_values("seconds")
            ax.semilogx(gg.seconds, gg[metric], "s--", ms=3, alpha=0.55,
                        label=f"factor MC r={rank}")
        for name, key in (("F-LITE", "lite"), ("qPO-independent-exact", "greedy")):
            gg = d[d.method == name]
            if len(gg):
                ax.semilogx(gg.seconds, gg[metric], "*", ms=10,
                            color=C[key], label=name)
        ceil = d[d.method == "reference-2nd-seed"][metric]
        if len(ceil):
            ax.axhline(float(ceil.min()), color="k", ls=":", lw=1)
        ax.set_xlabel("wall-clock seconds")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=6.2, loc="lower right", ncol=2)
    axes[0].set_title("Accuracy against runtime", fontsize=10)
    axes[1].set_title("Batch agreement against runtime", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_frontier.png")
    plt.close(fig)


def fig5_diversity(df):
    want = ["reference-2nd-seed", "qPO-fast-eig", "qPO-fast-eig-r0",
            "F-LITE", "A-LITE", "UCB", "Greedy"]
    sub = df[df.N == df.N.min()]
    rows = []
    ref = sub[sub.method == "reference-2nd-seed"]
    if len(ref):
        rows.append(("full qPO\n(reference)", ref.iloc[0]))
    f = sub[(sub.method == "qPO-fast-eig") & (sub["rank"] == 4)]
    if len(f):
        rows.append(("fast correlated\nqPO (r=4)", f.iloc[0]))
    f0 = sub[(sub.method == "qPO-fast-eig") & (sub["rank"] == 0)]
    if len(f0):
        rows.append(("independent\nqPO (r=0)", f0.iloc[0]))
    for m, lab in (("F-LITE", "F-LITE"), ("UCB", "UCB"), ("Greedy", "Greedy")):
        g = sub[sub.method == m]
        if len(g):
            rows.append((lab, g.iloc[0]))
    if not rows:
        return
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    x = np.arange(len(rows))
    mean = [r[1]["mean_tanimoto"] for r in rows]
    p90 = [r[1]["p90_tanimoto"] for r in rows]
    frac = [r[1]["frac_pairs_gt_0.4"] for r in rows]
    axes[0].bar(x - 0.2, mean, 0.4, label="mean", color=C["fast"])
    axes[0].bar(x + 0.2, p90, 0.4, label="90th percentile", color=C["factorMC"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    axes[0].set_ylabel("pairwise Tanimoto in the batch")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Batch similarity", fontsize=10)
    axes[1].bar(x, frac, 0.6, color=C["lite"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    axes[1].set_ylabel("fraction of pairs with Tanimoto > 0.4")
    axes[1].set_title("Redundant pairs", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_diversity.png")
    plt.close(fig)


OPT = {"qm9": {"top10_ave": 0.4655, "top100_ave": 0.3969},
       "wong_antibiotics": {}}


def fig6_closed_loop():
    files = [RES / "closed_loop.csv", RES / "closed_loop_more.csv"]
    files = [f for f in files if f.exists()]
    if not files:
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # top-k averages are the paper's metrics and they have a ceiling; the
    # fraction of the true top 1% is a coverage metric and behaves differently,
    # so both are shown rather than one standing for the other
    metrics = [("top10_ave", "average of the 10 best acquired"),
               ("frac_top1pct", "fraction of true top 1% acquired")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for ax, (m, lab) in zip(axes, metrics):
        if m not in d.columns:
            continue
        opt = OPT.get(str(d.dataset.iloc[0]), {}).get(m)
        if opt:
            ax.axhline(opt, color="k", ls=":", lw=1)
            ax.text(d.n_acquired.min(), opt, " library optimum", fontsize=7,
                    va="bottom")
        for method, g in d.groupby("method"):
            s = g.groupby("n_acquired")[m].agg(["mean", "std", "count"])
            ax.plot(s.index, s["mean"], "-o", ms=3, lw=1.3, label=method)
            if s["count"].max() > 1:
                se = s["std"] / np.sqrt(s["count"])
                ax.fill_between(s.index, s["mean"] - se, s["mean"] + se,
                                alpha=0.15, lw=0)
        ax.set_xlabel("molecules acquired")
        ax.set_ylabel(lab)
    axes[0].legend(fontsize=7)
    axes[0].set_title("Closed-loop discovery", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_closed_loop.png")
    plt.close(fig)


def main():
    df = _sweeps()
    if df is not None:
        fig1_rank_vs_efficiency(df)
        fig2_rank_vs_recall(df)
        fig5_diversity(df)
    fig3_runtime_scaling()
    fig4_frontier()
    fig6_closed_loop()
    print("figures in", FIG)
    for p in sorted(FIG.glob("*.png")):
        print("  ", p.name)


if __name__ == "__main__":
    main()
