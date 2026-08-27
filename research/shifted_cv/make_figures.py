"""Figures for the shifted control variate experiment (Section 20)."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def onehot_rows(df):
    return df[df["agreement"].notna() & (df["method"] != "raw")].copy()


def plot1(df):
    d = onehot_rows(df)
    d = d[d["coupling"] != "indep"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for fam, g in d.groupby("ref"):
        ax.scatter(g["agreement"], g["vrf"], s=12, alpha=0.6, label=fam)
    A = np.linspace(0.02, 0.995, 200)
    # idealised: raw per-draw variance 1 - sum p^2 ~ 0.8, coupled variance 2 (1 - A)
    for s2 in (0.3, 0.9):
        ax.plot(A, (1 - s2) / (2 * (1 - A)), "k--", lw=0.8, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("winner agreement P(W = V)")
    ax.set_ylabel("variance reduction factor")
    ax.set_title("Plot 1: VRF vs winner agreement (dashed: (1 - sum p^2) / 2(1 - A))")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot1_vrf_vs_agreement.png"), dpi=150)


def plot2(df):
    d = onehot_rows(df)
    d = d[(d["coupling"].isin(["procrustes", "commonz"])) & (d["beta"] == 1.0) & (d["sqrt"] == "sym")]
    d = d[~d["method"].str.contains("samemu")]
    order = ["logit", "iid", "diag"] + [f"lowrank{r}" for r in (1, 2, 4, 8, 16)]
    d["fam"] = d.apply(lambda r: r["ref"] if r["ref"] != "lowrank" else f"lowrank{int(r['rank'])}", axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, col, ttl in ((axes[0], "agreement", "P(W = V)"), (axes[1], "vrf", "VRF")):
        data = [d[d["fam"] == f][col].values for f in order]
        ax.boxplot([x for x in data], labels=order, showfliers=False)
        for i, x in enumerate(data):
            ax.scatter(np.full(len(x), i + 1) + np.random.uniform(-0.15, 0.15, len(x)), x, s=6, alpha=0.5)
        ax.set_ylabel(ttl)
        ax.tick_params(axis="x", rotation=45)
    axes[1].set_yscale("log")
    axes[0].plot([1, len(order)], [d["chance"].median()] * 2, "r:", label="median chance level")
    axes[0].legend(fontsize=7)
    fig.suptitle("Plot 2: winner agreement and VRF by control-race family (target-share matched, best coupling)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot2_agreement_by_family.png"), dpi=150)


def plot3(df):
    d = df[df["ref"] == "logit_samemu"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, g in d.groupby("key"):
        g = g.sort_values("tau_mult")
        axes[0].plot(g["tau_mult"], g["agreement"], "-o", ms=3, alpha=0.6, lw=0.8)
        axes[1].plot(g["tau_mult"], g["vrf"], "-o", ms=3, alpha=0.6, lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("tau / tau0")
    axes[0].set_ylabel("P(W = V), same-mu logit")
    axes[1].set_ylabel("VRF")
    axes[1].axhline(1.0, color="k", lw=0.8)
    fig.suptitle("Plot 3: same-mu logit control vs temperature (target-share matched logit is tau-invariant)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot3_logit_temperature.png"), dpi=150)


def plot4(dv, da):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for (key, m), g in dv.groupby(["key", "method"]):
        g = g.sort_values("M")
        ax.plot(g["M"], g["tr_cov"], "-o", ms=3, lw=0.9, label=f"{key.split('_')[0]}:{m}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("M (coupled draws)")
    ax.set_ylabel("tr Cov(r_hat) by replication")
    ax.set_title("Plot 4: residual variance vs M")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot4_variance_vs_M.png"), dpi=150)


def plot5(di):
    d = di[di["method"] != "surrogate"]
    keys = sorted(d["key"].unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        g = d[d["key"] == key]
        sur = di[(di["key"] == key) & (di["method"] == "surrogate")]["rmse_mu"].values
        for m, gg in g.groupby("method"):
            agg = gg.groupby("M").agg(rmse=("rmse_mu", "median"), sec=("seconds", "median")).reset_index()
            ax.plot(agg["sec"] + 1e-3, agg["rmse"], "-o", ms=3, lw=0.9, label=m)
        if len(sur):
            ax.axhline(sur[0], color="k", ls=":", label="surrogate (no MC)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(key, fontsize=8)
        ax.set_xlabel("seconds")
        ax.set_ylabel("RMSE(mu)")
    axes[0][0].legend(fontsize=5)
    fig.suptitle("Plot 5: ability recovery RMSE vs runtime (median over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot5_rmse_vs_runtime.png"), dpi=150)


def plot6(df):
    d = df[(df["ref"] == "lowrank") & (df["coupling"] == "procrustes") & (df["beta"] == 1.0) & (df["sqrt"] == "sym")].copy()
    d["rb"] = d["method"].str.startswith("rb_")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, isrb, ttl in ((axes[0], False, "one-hot shifted control"), (axes[1], True, "Rao-Blackwell + shifted control")):
        g = d[d["rb"] == isrb]
        for key, gg in g.groupby("key"):
            gg = gg.sort_values("rank")
            ax.plot(gg["rank"], gg["vrf"], "-o", ms=3, lw=0.7, alpha=0.5)
        med = g.groupby("rank")["vrf"].median()
        ax.plot(med.index, med.values, "k-o", lw=2, label="median")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.axhline(1, color="r", lw=0.8)
        ax.axhline(2, color="r", lw=0.8, ls=":")
        ax.set_xlabel("rank of reference covariance")
        ax.set_ylabel("VRF")
        ax.set_title(ttl)
        ax.legend()
    fig.suptitle("Plot 6: variance reduction vs reference rank (dotted red: break-even at 2x draw cost)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot6_vrf_vs_rank.png"), dpi=150)


def plot7(dd):
    keys = sorted(dd["key"].unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        g = dd[dd["key"] == key]
        for m, gg in g.groupby("method"):
            if m == "raw":
                continue
            gg = gg.sort_values("t")
            ax.plot(gg["t"] + 0.01, gg["vrf"], "-o", ms=3, lw=0.9, label=m,
                    ls="--" if m.startswith("rb") else "-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.6)
        ax.set_title(key, fontsize=8)
        ax.set_xlabel("RMS distance from mu* (ability-scale units, +0.01)")
        ax.set_ylabel("VRF")
    axes[0][0].legend(fontsize=5)
    fig.suptitle("Plot 7: fixed target-share control vs moving local control vs distance from the solution")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot7_fixed_vs_moving.png"), dpi=150)


def plot8(df, di):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sel = ["raw", "rb", "lowrank4_procrustes", "rb_lowrank4_procrustes"]
    d = df[df["method"].isin(sel)].pivot_table(index="key", columns="method", values="tr_var")
    d = d.dropna()
    for m, c in (("rb", "tab:orange"), ("lowrank4_procrustes", "tab:green"),
                 ("rb_lowrank4_procrustes", "tab:red")):
        axes[0].scatter(d["raw"], d[m], s=14, alpha=0.7, label=m, color=c)
    lim = [d.values.min() * 0.5, d.values.max() * 2]
    axes[0].plot(lim, lim, "k-", lw=0.8)
    for f in (10, 100, 1000):
        axes[0].plot(lim, [v / f for v in lim], "k:", lw=0.6)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("per-draw tr Var, raw winner counting")
    axes[0].set_ylabel("per-draw tr Var, estimator")
    axes[0].set_title("dotted: 10x / 100x / 1000x reduction", fontsize=8)
    axes[0].legend(fontsize=6)
    if di is not None:
        g = di[di["method"].isin(["raw", "rb", "lowrank_shift", "rb_lowrank_shift"])]
        agg = g.groupby(["method", "M"])["rmse_mu"].median().reset_index()
        for m, gg in agg.groupby("method"):
            axes[1].plot(gg["M"], gg["rmse_mu"], "-o", ms=3, label=m)
        sur = di[di["method"] == "surrogate"]["rmse_mu"].median()
        axes[1].axhline(sur, color="k", ls=":", label="surrogate")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("M")
        axes[1].set_ylabel("median RMSE(mu) over problems")
        axes[1].legend(fontsize=6)
    fig.suptitle("Plot 8: raw vs Rao-Blackwell vs Rao-Blackwell + shifted control")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "plot8_raw_rb_rbcv.png"), dpi=150)


def main():
    da = load("agreement.csv")
    if da is not None:
        da["beta"] = pd.to_numeric(da["beta"], errors="coerce").fillna(1.0)
        plot1(da); plot2(da); plot3(da); plot6(da)
    dv = load("variance_vs_M.csv")
    if dv is not None:
        plot4(dv, da)
    di = load("inversion.csv")
    if di is not None:
        plot5(di)
    dd = load("distance.csv")
    if dd is not None:
        plot7(dd)
    if da is not None:
        plot8(da, di)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
