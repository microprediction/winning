"""Figures for the T-Rex race study from results_*.json. Usage:
python plot_trex_race.py results_L1p_MC200.json"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1]
R = json.load(open(path))
here = os.path.dirname(os.path.abspath(path))
g = R["grid"]; T = np.array(g["T"]); v = np.array(g["v"])
fdp, fh, fm, tpp = (np.array(g[k]) for k in ("fdp", "fdp_hat", "fdp_model", "tpp"))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for vi, lab in ((0, "v = 0.5"), (5, "v = 0.75"), (9, "v = 0.95")):
    ax[0].plot(T, fdp[:, vi], "k-", alpha=0.3 + 0.3 * (vi // 5))
    ax[0].plot(T, fh[:, vi], "C1--", alpha=0.3 + 0.3 * (vi // 5))
    ax[0].plot(T, fm[:, vi], "C0:", alpha=0.3 + 0.3 * (vi // 5))
ax[0].plot([], [], "k-", label="FDP (truth)"); ax[0].plot([], [], "C1--", label="T-Rex FDP-hat (deflated)")
ax[0].plot([], [], "C0:", label="race-model FDP"); ax[0].axhline(R["setup"]["alpha"], color="grey", lw=0.8)
ax[0].set_xlabel("T (dummies at termination)"); ax[0].set_ylabel("average FDP"); ax[0].legend(fontsize=8)
ax[0].set_title("v = 0.5 (light), 0.75, 0.95 (dark)")

d = R["detail_one_dataset"]["10"] if "10" in R["detail_one_dataset"] else R["detail_one_dataset"][list(R["detail_one_dataset"])[-1]]
ax[1].scatter(d["marginal_t_actives"], d["strength_actives"], c="C3", label="actives")
gr = R["graded_betas"].get("10") or R["graded_betas"][list(R["graded_betas"])[-1]]
ax[1].scatter(gr["marginal_t"], gr["strength"], c="C2", marker="s", label="graded betas")
lim = [0, max(max(d["marginal_t_actives"]), max(gr["marginal_t"])) * 1.05]
ax[1].plot(lim, lim, "k:", lw=0.8); ax[1].set_xlabel(r"marginal $\sqrt{n}\,|corr(x_j, y)|$")
ax[1].set_ylabel("inverted strength (dummy sd)"); ax[1].legend(fontsize=8); ax[1].set_title("strength from Phi, T = 10")

ax[2].plot(T, tpp[:, 0], "C0-", label="v = 0.5"); ax[2].plot(T, tpp[:, 5], "C1-", label="v = 0.75"); ax[2].plot(T, tpp[:, 9], "C2-", label="v = 0.95")
ax[2].set_xlabel("T"); ax[2].set_ylabel("average TPP"); ax[2].legend(fontsize=8); ax[2].set_title("power")
fig.tight_layout(); out = os.path.join(here, "trex_race.png"); fig.savefig(out, dpi=130); print("->", out)
