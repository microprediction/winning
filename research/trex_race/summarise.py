"""Markdown tables for the README from results_L1p_MC*.json and
results_extensions_MC*.json. Usage: python summarise.py base.json ext.json"""
import json, sys
import numpy as np

B = json.load(open(sys.argv[1])); E = json.load(open(sys.argv[2]))
f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
pm = lambda ms, d=3: "-" if ms is None or ms[0] is None else (f"{ms[0]:.{d}f}" if ms[1] is None else f"{ms[0]:.{d}f} ± {ms[1]:.{d}f}")

print("### Engine against the closed form (noiseless runner, identical field)\n")
print("| T | s | D_j | points | engine | closed form |\n|---|---|---|---|---|---|")
for r in B["engine_check"]:
    print(f"| {r['T']} | {r['s']} | {r['D_j']} | {r['points']} | {r['engine']:.4f} | {r['closed_form']:.4f} |")

print("\n### One data set: the race law for the nulls (predicted / observed number of nulls with Phi_T > v)\n")
print("| T | v = 0 | v = 0.25 | v = 0.5 | v = 0.75 | null strength mean, sd | Phi = 0 nulls (floored) |\n|---|---|---|---|---|---|---|")
for T, d in B["detail_one_dataset"].items():
    p_, o_ = d["nulls_with_phi_above_v"]["predicted"], d["nulls_with_phi_above_v"]["observed"]
    cells = " | ".join(f"{p_[v]:.1f} / {o_[v]}" for v in ("0.0", "0.25", "0.5", "0.75"))
    m, s = d["null_strength_mean_sd"]
    print(f"| {T} | {cells} | {m:.2f}, {s:.2f} | {d['null_strength_clipped_at_floor']} |")

print("\n### One data set: actives (T = 1)\n")
d = B["detail_one_dataset"]["1"]
print("| active | Phi_1 | strength on the dummy scale | marginal sqrt(n)|corr| |\n|---|---|---|---|")
for i, (a, b, c) in enumerate(zip(d["phi_actives"], d["strength_actives"], d["marginal_t_actives"])):
    print(f"| {i + 1} | {a:.2f} | {b:.2f} | {c:.2f} |")
print(f"\nSpearman(strength, |beta|) over all p at T = 1: {d['spearman_strength_vs_beta_all']:.3f}; "
      f"Spearman(strength, marginal t) among nulls: {d['spearman_strength_vs_marginal_t_nulls']:.3f}.")
g = B["graded_betas"]
for T, x in g.items():
    print(f"\nGraded betas, T = {T}: beta {x['beta']}; Phi {x['phi']}; strength {x['strength']}; "
          f"Spearman(strength, beta) = {x['spearman_strength_vs_beta_actives']:.2f}, "
          f"Spearman(marginal t, beta) = {x['spearman_marginal_t_vs_beta_actives']:.2f}.")

print("\n### Fig. 5 grid, L = p, MC = %d: average FDP / T-Rex FDP-hat / race-model FDP, and TPP\n" % B["setup"]["MC"])
gr = B["grid"]; T = gr["T"]; v = gr["v"]
cols = [(0, "v = 0.5"), (5, "v = 0.75"), (9, "v = 0.95")]
print("| T | " + " | ".join(f"{c} FDP / hat / model / TPP" for _, c in cols) + " |")
print("|---|" + "---|" * len(cols))
for ti in (0, 1, 4, 9, 19):
    if ti >= len(T):
        continue
    row = [f"{gr['fdp'][ti][vi]:.3f} / {gr['fdp_hat'][ti][vi]:.3f} / {gr['fdp_model'][ti][vi]:.3f} / {gr['tpp'][ti][vi]:.2f}" for vi, _ in cols]
    print(f"| {T[ti]} | " + " | ".join(row) + " |")

print("\n### Calibrated at alpha = 0.1 (Algorithm 1 at L = p), MC = %d\n" % B["setup"]["MC"])
print("| estimator in the calibration | FDR | TPR | mean selected | median T* | median v* |\n|---|---|---|---|---|---|")
for k, x in B["calibrated"].items():
    print(f"| {k} | {x['fdr']:.3f} ± {x['fdr_se']:.3f} | {x['tpr']:.3f} ± {x['tpr_se']:.3f} | {x['mean_selected']:.1f} | {f(x['T_star_median'], 0)} | {f(x['v_star_median'], 2)} |")

print("\n## A. Position-aware statistic: AUC actives vs nulls (MC = %d)\n" % list(E.values())[0]["MC"])
print("| design | T | AUC(Phi) | AUC(position posterior) | paired difference |\n|---|---|---|---|---|")
for kind, R in E.items():
    for T, a in R["A_auc"].items():
        print(f"| {kind} | {T} | {pm(a['phi'])} | {pm(a['position'])} | {pm(a['paired_diff'], 4)} |")

print("\n## B. Realised FDR and TPR at target 0.1\n")
print("| design | method | FDR | TPR | mean selected | share empty |\n|---|---|---|---|---|---|")
names = {"trex": "(i) T-Rex as published (Alg. 1, L = p)", "position": "(ii) ranked by position posterior, sized by V-hat",
         "position_model": "(ii-m) same ranking, sized by the race-model estimate",
         "block": "(iii) block statistic, block level, sized by block V-hat", "block_model": "(iii-m) block statistic, sized by the race-model estimate",
         "trex_blocklevel": "(i) mapped to block level"}
for kind, R in E.items():
    for k in ("trex", "position", "position_model", "trex_blocklevel", "block", "block_model"):
        x = R["B"][k]
        print(f"| {kind} | {names[k]} | {pm(x['fdr'])} | {pm(x['tpr'])} | {x['mean_R']:.1f} | {x['share_empty']:.2f} |")

print("\n## C. Calibration of the estimators on the sets (ii) and (iii) select\n")
print("| design | selection | n | realised FDP | V-hat / R | race-model | FDP - V-hat | FDP - model |\n|---|---|---|---|---|---|---|---|")
for kind, R in E.items():
    for k in ("position", "position_model", "block", "block_model"):
        x = R["C"][k]
        print(f"| {kind} | {k} | {x['n']} | {pm(x['fdp'])} | {pm(x['vhat'])} | {pm(x['model'])} | {pm(x['fdp_minus_vhat'])} | {pm(x['fdp_minus_model'])} |")
