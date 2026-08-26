"""Do trained classifiers break IIA? Linear-probe tier.

A frozen net's masked softmax renormalises by identity, and the Bayes
target under menu restriction is renormalised truth, so neither can
violate IIA. The chooser that CAN is the training procedure: retrain
the same recipe on the restricted menu and ask which restriction map of
the full-menu model predicts what the specialist actually does --
proportional renormalisation (the constant-ratio rule) or deletion from
a fitted Gaussian race. Cross-menu stability is the claim of interest;
a distillation null (specialists trained on labels sampled from the
renormalised generalist, so IIA holds in the label process by
construction) separates procedure from pipeline artifact.

Tier-1 design (decision layer only, everything convex):
  features   penultimate resnet56 activations for the 10k CIFAR test
             images (extracted here with a forward hook)
  split      5k probe-train / 5k probe-eval, stratified
  generalist multinomial logistic probe on all 100 classes
  race       tied (V,D) recalibration fitted to the generalist probe's
             train-split logits (winners = true labels), params SAVED
  menus      20 superclasses; 20 random 5-subsets; 20 confusable
             5-subsets (top pairs of the generalist's confusion matrix,
             greedily grown)
  specialist per menu: same probe recipe on train images with y in S
  score      mean KL(specialist || map) over eval images with y in S,
             for map in {renormalised generalist, race-diag deletion,
             race-r2 deletion}; secondary: true-label NLL of each
  null       per menu: specialist retrained on labels drawn from the
             renormalised generalist; same scoring

Stages checkpoint to data/specialist_stage*.npz so the script resumes.
Run: python research/recalibration/run_specialist_menus.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from winning.factor.core import win_probabilities_factor, hermite_nodes  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RANK = 2
N_MENUS = 20
MENU_SIZE = 5
SEED = 21


# ---------------------------------------------------------------- stage 1
def stage_features():
    out = DATA / "specialist_stage1_features.npz"
    if out.exists():
        return np.load(out)
    import torch, torchvision, pickle
    SP = Path("/private/tmp/claude-501/-Users-petercotton-github-kinetics/"
              "d3b37c99-3f1b-45cf-8229-efe313e9c80f/scratchpad")
    ds = torchvision.datasets.CIFAR100(root=str(SP / "cifar"), train=False,
                                       download=False)
    model = torch.hub.load("chenyaofo/pytorch-cifar-models",
                           "cifar100_resnet56", pretrained=True,
                           trust_repo=True)
    model.eval()
    feats = []
    hook = model.fc.register_forward_pre_hook(
        lambda m, inp: feats.append(inp[0].detach().numpy()))
    mean = np.array([0.5071, 0.4865, 0.4409])
    std = np.array([0.2673, 0.2564, 0.2762])
    X = (ds.data.astype(np.float32) / 255.0 - mean) / std
    X = torch.tensor(X.transpose(0, 3, 1, 2), dtype=torch.float32)
    with torch.no_grad():
        for a in range(0, len(X), 500):
            model(X[a:a + 500])
    hook.remove()
    F_ = np.concatenate(feats)
    np.savez_compressed(out, features=F_, labels=np.array(ds.targets))
    return np.load(out)


# ---------------------------------------------------------------- stage 2
def stage_probe(F_, y):
    out = DATA / "specialist_stage2_probe.npz"
    if out.exists():
        return np.load(out)
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    tr, ev = np.sort(idx[:5000]), np.sort(idx[5000:])
    clf = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8)
    clf.fit(F_[tr], y[tr])
    logits = F_ @ clf.coef_.T + clf.intercept_
    np.savez_compressed(out, train_idx=tr, eval_idx=ev, logits=logits,
                        coef=clf.coef_, intercept=clf.intercept_)
    return np.load(out)


# ---------------------------------------------------------------- stage 3
def stage_race(logits_tr, y_tr, E):
    out = DATA / "specialist_stage3_race.npz"
    if out.exists():
        return np.load(out)
    sys.path.insert(0, str(HERE))
    from run_cifar_recalibration import fit_race
    t0 = time.perf_counter()
    V2, D2, s2 = fit_race(logits_tr, y_tr, E, RANK)
    Vd, Dd, sdg = fit_race(logits_tr, y_tr, E, RANK, diag_only=True)
    print(f"race fits {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez_compressed(out, V2=V2, D2=D2, s2=np.array(s2),
                        Vd=Vd, Dd=Dd, sd=np.array(sdg))
    return np.load(out)


# ---------------------------------------------------------------- menus
def build_menus(P_gen_eval, y_eval, K, rng):
    coarse = np.load(DATA / "cifar100_logits.npz")["coarse_of_fine"]
    menus = [("super", np.where(coarse == c)[0]) for c in range(20)]
    for m in range(N_MENUS):
        menus.append(("random", np.sort(rng.choice(K, MENU_SIZE, replace=False))))
    # confusable: greedy growth from the most-confused pairs of the probe
    pred = P_gen_eval.argmax(axis=1)
    C = np.zeros((K, K))
    for t, p in zip(y_eval, pred):
        if t != p:
            C[t, p] += 1
    C = C + C.T
    used_pairs = set()
    for m in range(N_MENUS):
        i, j = np.unravel_index(np.argmax(C), C.shape)
        S = [int(i), int(j)]
        C[i, j] = C[j, i] = -1
        while len(S) < MENU_SIZE:
            gains = C[S].sum(axis=0)
            gains[S] = -np.inf
            S.append(int(np.argmax(gains)))
        menus.append(("confus", np.sort(np.array(S))))
        for a in S:
            for b in S:
                if a != b:
                    C[a, b] = -1
    return menus


# ---------------------------------------------------------------- scoring
def softmax(lg):
    z = lg - lg.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def kl(p, q):
    p = np.maximum(p, 1e-12); q = np.maximum(q, 1e-12)
    return float(np.mean((p * (np.log(p) - np.log(q))).sum(axis=1)))


def main():
    d1 = stage_features()
    F_, y = d1["features"].astype(np.float64), d1["labels"]
    d2 = stage_probe(F_, y)
    tr, ev = d2["train_idx"], d2["eval_idx"]
    logits = d2["logits"]
    K = logits.shape[1]
    E = d2["coef"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    d3 = stage_race(logits[tr], y[tr], E)
    V2, D2, s2 = d3["V2"], d3["D2"], float(d3["s2"])
    Vd, Dd, sdg = d3["Vd"], d3["Dd"], float(d3["sd"])
    Fq, Wq = hermite_nodes(RANK)

    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(SEED + 1)
    P_gen = softmax(logits)
    menus = build_menus(P_gen[ev], y[ev], K, rng)

    rows = ["menu_type,menu_id,n_eval,kl_renorm,kl_diag,kl_r2,"
            "null_kl_renorm,null_kl_diag,"
            "nll_spec,nll_renorm,nll_diag,nll_r2"]
    agg = {}
    for mi, (mtype, S) in enumerate(menus):
        tr_m = tr[np.isin(y[tr], S)]
        ev_m = ev[np.isin(y[ev], S)]
        if len(tr_m) < 50 or len(ev_m) < 50:
            continue
        pos = {c: i for i, c in enumerate(S)}
        # specialist on real labels
        spec = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8)
        spec.fit(F_[tr_m], np.array([pos[c] for c in y[tr_m]]))
        P_spec = spec.predict_proba(F_[ev_m])
        # map 1: renormalised generalist
        P_ren = P_gen[ev_m][:, S]
        P_ren = P_ren / P_ren.sum(axis=1, keepdims=True)
        # maps 2-3: race deletions
        P_dg = np.stack([win_probabilities_factor(-logits[i] / sdg, Vd, Dd,
                                                  Fq, Wq, keep=S)
                         for i in ev_m])
        P_dg = P_dg / P_dg.sum(axis=1, keepdims=True)
        P_r2 = np.stack([win_probabilities_factor(-logits[i] / s2, V2, D2,
                                                  Fq, Wq, keep=S)
                         for i in ev_m])
        P_r2 = P_r2 / P_r2.sum(axis=1, keepdims=True)
        # distillation null specialist
        y_null = np.array([rng.choice(len(S), p=(lambda q: q / q.sum())(
            np.maximum(P_gen[i][S], 1e-12))) for i in tr_m])
        specn = LogisticRegression(C=1.0, max_iter=2000, tol=1e-8)
        specn.fit(F_[tr_m], y_null) if len(np.unique(y_null)) == len(S) else None
        P_specn = (specn.predict_proba(F_[ev_m])
                   if len(np.unique(y_null)) == len(S) else None)

        y_ev_pos = np.array([pos[c] for c in y[ev_m]])
        def nll(P):
            return float(-np.mean(np.log(np.maximum(
                P[np.arange(len(y_ev_pos)), y_ev_pos], 1e-12))))
        r = dict(kl_renorm=kl(P_spec, P_ren), kl_diag=kl(P_spec, P_dg),
                 kl_r2=kl(P_spec, P_r2),
                 null_kl_renorm=(kl(P_specn, P_ren) if P_specn is not None
                                 else np.nan),
                 null_kl_diag=(kl(P_specn, P_dg) if P_specn is not None
                               else np.nan),
                 nll_spec=nll(P_spec), nll_renorm=nll(P_ren),
                 nll_diag=nll(P_dg), nll_r2=nll(P_r2))
        rows.append(f"{mtype},{mi},{len(ev_m)}," + ",".join(
            f"{r[k]:.6f}" for k in
            ["kl_renorm", "kl_diag", "kl_r2", "null_kl_renorm",
             "null_kl_diag", "nll_spec", "nll_renorm", "nll_diag",
             "nll_r2"]))
        agg.setdefault(mtype, []).append(r)
        print(f"{mtype:7s} {mi:3d} n={len(ev_m):4d} "
              f"KL(spec||renorm)={r['kl_renorm']:.4f} "
              f"KL(spec||diag)={r['kl_diag']:.4f} "
              f"KL(spec||r2)={r['kl_r2']:.4f} "
              f"null(renorm)={r['null_kl_renorm']:.4f}", flush=True)

    print("\n=== medians by menu type (KL to specialist; lower = better map) ===")
    for mtype, rs in agg.items():
        med = {k: float(np.median([x[k] for x in rs])) for k in rs[0]}
        print(f"{mtype:7s} renorm {med['kl_renorm']:.4f}  diag "
              f"{med['kl_diag']:.4f}  r2 {med['kl_r2']:.4f}  "
              f"null-renorm {med['null_kl_renorm']:.4f}  "
              f"(nll: spec {med['nll_spec']:.3f} renorm "
              f"{med['nll_renorm']:.3f} diag {med['nll_diag']:.3f} "
              f"r2 {med['nll_r2']:.3f})")
    (HERE / "results_specialist_menus.csv").write_text("\n".join(rows) + "\n")
    print(f"wrote {HERE / 'results_specialist_menus.csv'}")


if __name__ == "__main__":
    main()
