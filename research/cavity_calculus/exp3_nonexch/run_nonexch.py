"""Non-exchangeable pass@k on Pass8-Rollouts: do duplicates and length
carry information the exchangeable hierarchical model cannot use?

Models (success iff z_i > 0; theta ~ N(mu, tau^2) is the prompt factor):
  E   z_i = theta + eps_i                          exchangeable (exp2)
  L   z_i = theta + beta*ell_i + eps_i             + standardized log length
  C   z_i = theta + eta_{c(i)} + eps_i             + answer-cluster effect,
                                                     eta_c ~ N(0, sig^2)
  LC  both.
Clusters: samples of one prompt with the same extracted final answer.
Samples without an extracted answer are singletons.
Given theta the likelihood factorizes over clusters, so it is a
nested one-dimensional integral (theta outside, eta inside each
cluster), computed by numerical Gauss-Hermite quadrature with Q theta
nodes and R eta nodes. Not exact: the node-convergence sweep is in
results.json["quadrature_convergence"]. The capped sig = 3 makes the
eta integrand steep, so R matters: R = 11 is off by ~1e-3 nat/sample
for LC. Default Q = R = 61 (within ~3e-6 of Q = 201, R = 151).

Usage: run_nonexch.py [--refit] [--Q 61] [--R 61]
  --refit  refit hyperparameters (warm-started from results.json);
           without it the fitted values in results.json are reused
           and all likelihoods are re-evaluated at (Q, R).
Every run appends its full output as one JSON line to runs.jsonl
(append-only, committed) and overwrites results.json with the latest.

Tests:
 0. Descriptives: truncation, length, duplicate-answer pairs, and
    text similarity among different-answer pairs (prompt-level
    bootstrap; within-prompt linear probability model controlling
    for truncation and length, prompt-clustered standard errors).
 1. Held-out per-sample log-likelihood, hyperparameters fitted on even
    prompts, scored on odd prompts (all 8 samples). LR for sig > 0.
 2. Model-free: held-out (samples 4-7) success rate for prompts with a
    given training count (samples 0-3), split by the number of
    distinct training answers and by the number of truncated samples.
 3. Scenario B (before generating): posterior over theta from samples
    0-3, predict fresh samples 4-7: per-prompt pass@4 Bernoulli and
    per-sample log loss; E vs C.
 4. Scenario A (which sample to trust): predict each held-out sample
    given its own length and its answer-cluster relation to the
    verified samples; E, L, C, LC.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
G = 8
SIG_CAP = 3.0   # unbounded MLE runs sig -> inf (same answer == same fate); cap it
CONV_RULES = [(41, 11), (61, 21), (81, 25), (101, 31), (41, 61), (61, 61),
              (101, 61), (161, 101), (201, 151)]


def _arg(name, default):
    return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


Q, R = _arg("--Q", 61), _arg("--R", 61)
zq = lwq = zr = lwr = None


def set_nodes(q, r):
    """Probabilists' Gauss-Hermite rules for theta (q nodes) and eta (r)."""
    global zq, lwq, zr, lwr
    zq, wq = np.polynomial.hermite_e.hermegauss(q); lwq = np.log(wq / wq.sum())
    zr, wr = np.polynomial.hermite_e.hermegauss(r); lwr = np.log(wr / wr.sum())


set_nodes(Q, R)


def load():
    z = np.load(os.path.join(HERE, "features.npz"))
    n = len(z["prompt_index"])
    order = np.lexsort((z["sample_id"], z["index"]))
    ok = (z["reward"][order] >= 1 - 1e-6).reshape(n, G).astype(float)
    ell = np.log(z["nchar"][order].astype(float))
    ell = ((ell - ell.mean()) / ell.std()).reshape(n, G)
    ah = z["ans_hash"][order].reshape(n, G)
    trunc = ~z["think_closed"][order].reshape(n, G)
    cl = np.zeros((n, G), dtype=int)
    for p in range(n):
        seen = {}
        for i in range(G):
            h = ah[p, i]
            if h == 0:
                cl[p, i] = len(seen); seen[("single", i)] = cl[p, i]
            else:
                cl[p, i] = seen.setdefault(h, len(seen))
    return ok, ell, cl, trunc, ah


def loglik(Y, M, ELL, CL, mu, tau, sig, beta, chunk=None):
    """log P(observed samples) per prompt. M masks samples in play."""
    if chunk is None:   # keep the (P, n, Q, R) block near 400*8*41*11 doubles
        chunk = max(8, int(400 * 41 * 11 / (len(zq) * (len(zr) if sig > 0 else 1))))
    out = np.empty(len(Y))
    theta = mu + tau * zq
    eta = sig * zr if sig > 0 else np.zeros(1)
    lw_r = lwr if sig > 0 else np.zeros(1)
    n = Y.shape[1]
    onehot = None
    for a in range(0, len(Y), chunk):
        y, m, el, cl = (v[a:a + chunk] for v in (Y, M, ELL, CL))
        s = 2 * y - 1
        arg = s[..., None, None] * (theta[None, None, :, None]
                                    + eta[None, None, None, :]
                                    + beta * el[..., None, None])
        lp = log_ndtr(arg) * m[..., None, None]                 # (P,n,Q,R)
        oh = (cl[..., None] == np.arange(G)).astype(float)       # (P,n,G) cluster ids run 0..G-1
        sumc = np.einsum("pic,piqr->pcqr", oh, lp)              # (P,c,Q,R)
        termc = logsumexp(sumc + lw_r, axis=-1)                 # (P,c,Q)
        out[a:a + chunk] = logsumexp(termc.sum(1) + lwq, axis=-1)
    return out


def fit(Y, M, ELL, CL, use_sig, use_beta, x0=None):
    def unpack(x):
        mu, ltau = x[0], x[1]
        sig = SIG_CAP / (1 + np.exp(-x[2])) if use_sig else 0.0
        beta = x[3] if use_beta else 0.0
        return mu, np.exp(ltau), sig, beta

    def nll(x):
        return -loglik(Y, M, ELL, CL, *unpack(x)).sum()
    x0 = np.array([-0.6, 0.3, 0.0, 0.0]) if x0 is None else x0
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 0.05, "maxiter": 600})
    return unpack(res.x), -res.fun, res.x


def _encode(f):
    """Optimizer coordinates for a stored fit (inverse of fit.unpack)."""
    r = min(max(f["sig"] / SIG_CAP, 1e-6), 1 - 1e-6)
    return np.array([f["mu"], np.log(max(f["tau"], 1e-12)),
                     np.log(r / (1 - r)), f["beta"]])


def descriptives(ok, ell, trunc, ah, jac, prefix_len, n_boot=2000, seed=0):
    """Test 0: what is observable before grading, and its raw relation
    to fate. All counts are emitted so the prose can be checked."""
    n = len(ok)
    okb = ok.astype(bool)
    d = {}
    ts = np.argwhere(okb & trunc)
    d["truncation"] = {
        "n_samples": int(ok.size), "n_truncated": int(trunc.sum()),
        "trunc_rate": float(trunc.mean()),
        "truncated_successes": int(len(ts)),
        "truncated_success_rate": float(len(ts) / trunc.sum()),
        "truncated_success_positions_prompt_sample": ts.tolist(),
        "truncated_successes_with_no_extracted_answer": int((ah[okb & trunc] == 0).sum()),
        "closed_success_rate": float(ok[~trunc].mean())}
    qs = np.quantile(ell, [.2, .4, .6, .8]); b = np.searchsorted(qs, ell, side="right")
    d["success_by_length_quintile"] = [float(ok[b == k].mean()) for k in range(5)]
    d["prefix_len"] = {"median": float(np.median(prefix_len)),
                       "frac_eq_19": float((prefix_len == 19).mean()),
                       "max": int(prefix_len.max())}
    iu = np.triu_indices(G, 1)
    A, B = ah[:, iu[0]], ah[:, iu[1]]
    J = jac[:, iu[0], iu[1]].astype(float)
    cof = (ok[:, iu[0]] == ok[:, iu[1]])
    same = (A == B) & (A != 0)
    diff = A != B
    d["pairs_same_answer"] = {
        "n_pairs": int(A.size), "n_same_nonzero_answer": int(same.sum()),
        "frac_same_nonzero_answer": float(same.mean()),
        "cofate_same_answer": float(cof[same].mean()),
        "n_different_hash": int(diff.sum()),
        "cofate_different_hash": float(cof[diff].mean()),
        "note": "same = equal nonzero answer hash; different = unequal hash "
                "(a pair with one missing answer counts as different; two "
                "missing answers are neither)"}
    # co-fate by global Jaccard quartile among different-answer pairs
    P = np.broadcast_to(np.arange(n)[:, None], A.shape)
    rng = np.random.default_rng(seed)
    sim = {}
    for label, m in (("different_hash", diff),
                     ("different_hash_both_answered", diff & (A != 0) & (B != 0))):
        j, c, p = J[m], cof[m].astype(float), P[m]
        cut = np.quantile(j, [.25, .5, .75]); qb = np.searchsorted(cut, j, side="right")
        num = np.zeros((n, 4)); den = np.zeros((n, 4))
        np.add.at(num, (p, qb), c); np.add.at(den, (p, qb), 1.0)
        rate = num.sum(0) / den.sum(0)
        boots = np.empty(n_boot)
        for t in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            r_ = (w @ num) / (w @ den); boots[t] = r_[3] - r_[0]
        sim[label] = {"n_pairs": int(m.sum()), "jaccard_quartile_cuts": cut.tolist(),
                      "cofate_by_jaccard_quartile": rate.tolist(),
                      "Q4_minus_Q1": float(rate[3] - rate[0]),
                      "Q4_minus_Q1_prompt_bootstrap_95": np.quantile(boots, [.025, .975]).tolist(),
                      "within_prompt_lpm": _fe_lpm(m, J, cof, trunc, ell, A, B, P, n, iu)}
    d["text_similarity_different_answers"] = sim
    return d


def _fe_lpm(m, J, cof, trunc, ell, A, B, P, n, iu):
    """co-fate ~ Jaccard (per SD) + #truncated in pair (dummies) + mean
    and |diff| of standardized log length + both-answered, with prompt
    fixed effects (within transform) and prompt-clustered sandwich SEs."""
    nt = (trunc[:, iu[0]].astype(int) + trunc[:, iu[1]])[m]
    j = J[m]; js = (j - j.mean()) / j.std()
    cols = {"jaccard_per_sd": js, "one_truncated": (nt == 1).astype(float),
            "both_truncated": (nt == 2).astype(float),
            "mean_log_len": ((ell[:, iu[0]] + ell[:, iu[1]]) / 2)[m],
            "absdiff_log_len": np.abs(ell[:, iu[0]] - ell[:, iu[1]])[m],
            "both_answered": ((A != 0) & (B != 0))[m].astype(float)}
    p = P[m]; y = cof[m].astype(float); cnt = np.bincount(p, minlength=n)

    def dm(v):
        return v - (np.bincount(p, weights=v, minlength=n) / np.maximum(cnt, 1))[p]
    names = [k for k, v in cols.items() if dm(v).std() > 1e-9]
    X = np.column_stack([dm(cols[k]) for k in names]); Y = dm(y)
    XtX = X.T @ X; beta = np.linalg.solve(XtX, X.T @ Y); e = Y - X @ beta
    g = np.column_stack([np.bincount(p, weights=X[:, k] * e, minlength=n) for k in range(X.shape[1])])
    Vi = np.linalg.inv(XtX); V = Vi @ (g.T @ g) @ Vi; se = np.sqrt(np.diag(V))
    b0 = float((X[:, 0] @ Y) / (X[:, 0] @ X[:, 0]))
    return {"coef": dict(zip(names, beta.tolist())), "se": dict(zip(names, se.tolist())),
            "jaccard_per_sd_prompt_FE_only": b0, "mean_cofate": float(y.mean())}


def heldout_ll(name, f, ok, Mall, ell, cl, test):
    return float(loglik(ok[test], Mall[test], ell[test], cl[test],
                        f["mu"], f["tau"], f["sig"], f["beta"]).sum() / (test.sum() * G))


def main():
    t0 = time.time()
    ok, ell, cl, trunc, ah = load()
    z = np.load(os.path.join(HERE, "features.npz"))
    n = len(ok)
    train = np.arange(n) % 2 == 0
    test = ~train
    Mall = np.ones_like(ok)
    out = {"n_prompts": int(n), "success_rate": float(ok.mean()),
           "trunc_rate": float(trunc.mean()), "quadrature": {"Q": Q, "R": R}}
    out["test0_descriptives"] = descriptives(ok, ell, trunc, ah, z["jac"], z["prefix_len"])
    out["pairs_same_answer"] = out["test0_descriptives"]["pairs_same_answer"]
    print("T0 descriptives done", f"[{time.time()-t0:.0f}s]", file=sys.stderr, flush=True)
    specs = {"E": (False, False), "L": (False, True),
             "C": (True, False), "LC": (True, True)}
    fits = {}
    prev = os.path.join(HERE, "results.json")
    old_fits = json.load(open(prev))["test1_heldout"] if os.path.exists(prev) else None
    reuse = old_fits is not None and "--refit" not in sys.argv
    if reuse:
        print("reusing fitted hyperparameters from results.json", file=sys.stderr)
    for name, (us, ub) in specs.items():
        if reuse:
            f = {k: old_fits[name][k] for k in ("mu", "tau", "sig", "beta", "fit_rule")
                 if k in old_fits[name]}
            f.setdefault("fit_rule", [41, 11])
            f["train_ll_per_sample"] = heldout_ll(name, f, ok, Mall, ell, cl, train)
            f["test_ll_per_sample"] = heldout_ll(name, f, ok, Mall, ell, cl, test)
            f["_x"] = _encode(f)
            fits[name] = f
            continue
        if old_fits is not None:
            x0 = _encode(old_fits[name])
        else:
            warm = {"C": "E", "LC": "L"}.get(name)
            x0 = None if warm is None else np.array([fits[warm]["_x"][0], fits[warm]["_x"][1], 0.0, fits[warm]["_x"][3]])
        params, ll_tr, x = fit(ok[train], Mall[train], ell[train], cl[train], us, ub, x0=x0)
        ll_te = loglik(ok[test], Mall[test], ell[test], cl[test], *params).sum()
        fits[name] = {"mu": params[0], "tau": params[1], "sig": params[2],
                      "beta": params[3], "fit_rule": [Q, R],
                      "train_ll_per_sample": ll_tr / (train.sum() * G),
                      "test_ll_per_sample": ll_te / (test.sum() * G), "_x": x}
        print(f"{name:3s} mu={params[0]:+.3f} tau={params[1]:.3f} "
              f"sig={params[2]:.3f} beta={params[3]:+.3f}  "
              f"ll/sample train {fits[name]['train_ll_per_sample']:.4f} "
              f"test {fits[name]['test_ll_per_sample']:.4f}  [{time.time()-t0:.0f}s]",
              file=sys.stderr, flush=True)
    ntr = train.sum() * G
    out["test1_heldout"] = {k: {kk: (vv if kk == "fit_rule" else float(vv))
                                for kk, vv in v.items() if kk != "_x"}
                            for k, v in fits.items()}
    # node convergence: same fitted parameters, held-out ll at each rule
    conv = {}
    if "--no-conv" not in sys.argv:
        for q, r in CONV_RULES:
            set_nodes(q, r)
            conv[f"Q={q},R={r}"] = {k: heldout_ll(k, f, ok, Mall, ell, cl, test)
                                    for k, f in fits.items()}
            print(f"conv Q={q} R={r} {conv[f'Q={q},R={r}']}", file=sys.stderr, flush=True)
        set_nodes(Q, R)
    out["quadrature_convergence"] = {
        "what": "odd-prompt held-out log likelihood per sample, fitted parameters fixed",
        "by_rule": conv}
    out["sig_cap"] = SIG_CAP
    out["test1_LR"] = {
        "C_vs_E_2dLL_train": float(2 * ntr * (fits["C"]["train_ll_per_sample"] - fits["E"]["train_ll_per_sample"])),
        "LC_vs_L_2dLL_train": float(2 * ntr * (fits["LC"]["train_ll_per_sample"] - fits["L"]["train_ll_per_sample"])),
        "L_vs_E_2dLL_train": float(2 * ntr * (fits["L"]["train_ll_per_sample"] - fits["E"]["train_ll_per_sample"])),
    }

    # ---- Test 2: model-free evidence weighting -------------------------
    tr_idx, te_idx = np.arange(4), np.arange(4, 8)
    s_tr = ok[:, tr_idx].sum(1).astype(int)
    heldout_rate = ok[:, te_idx].mean(1)
    heldout_any = (ok[:, te_idx].sum(1) > 0).astype(float)
    ndist = np.array([len(set(cl[p, tr_idx])) for p in range(n)])
    ntrunc = trunc[:, tr_idx].sum(1)
    t2 = {}
    for s in range(4):
        row = {}
        for d in range(1, 5):
            sel = (s_tr == s) & (ndist == d)
            if sel.sum() >= 30:
                row[f"distinct={d}"] = {"n": int(sel.sum()),
                                       "heldout_rate": float(heldout_rate[sel].mean()),
                                       "heldout_pass@4": float(heldout_any[sel].mean())}
        rowt = {}
        for k in range(5):
            sel = (s_tr == s) & (ntrunc == k)
            if sel.sum() >= 30:
                rowt[f"truncated={k}"] = {"n": int(sel.sum()),
                                         "heldout_rate": float(heldout_rate[sel].mean()),
                                         "heldout_pass@4": float(heldout_any[sel].mean())}
        t2[f"train_successes={s}"] = {"by_distinct_answers": row, "by_truncated": rowt}
    # cross-tab: distinct answers given NO truncated training sample
    xt = {}
    for s in range(4):
        row = {}
        for d in range(1, 5):
            sel = (s_tr == s) & (ndist == d) & (ntrunc == 0)
            if sel.sum() >= 30:
                row[f"distinct={d}"] = {"n": int(sel.sum()),
                                       "heldout_rate": float(heldout_rate[sel].mean()),
                                       "heldout_pass@4": float(heldout_any[sel].mean())}
        xt[f"train_successes={s},truncated=0"] = row
    t2["crosstab_no_truncation"] = xt
    out["test2_modelfree"] = t2

    # ---- Test 5: how much do the non-exchangeable statistics predict? --
    # Logistic regression of each held-out sample's success on the
    # composition of the four training samples (count, distinct answers,
    # truncated), fitted on even prompts, scored on odd. Not a generative
    # model; a measure of the information the count alone throws away.
    def logistic_fit(X, y, C=10.0):
        """sklearn LogisticRegression(C=C) objective without sklearn:
        C * sum logloss + 0.5 ||w||^2, unpenalized separate intercept."""
        def f(v):
            w, b0 = v[:-1], v[-1]
            t = X @ w + b0
            ll = np.logaddexp(0, t) - y * t
            g = 1 / (1 + np.exp(-t)) - y
            return C * ll.sum() + 0.5 * w @ w, np.append(C * X.T @ g + w, C * g.sum())
        v = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B",
                     options={"maxiter": 5000, "gtol": 1e-8}).x
        return lambda Z: 1 / (1 + np.exp(-(Z @ v[:-1] + v[-1])))
    def feats(cols):
        return np.column_stack([np.ones(n)] + cols)
    Xa = feats([s_tr, ndist, ntrunc, s_tr * ndist, s_tr * ntrunc])
    Xb = feats([s_tr])
    yy = ok[:, te_idx]
    t5 = {}
    for nm, X in (("count_only", Xb), ("count+distinct+truncated", Xa)):
        Xr = np.repeat(X, 4, axis=0); yr = yy.ravel()
        trm = np.repeat(train, 4)
        pred = logistic_fit(Xr[trm], yr[trm])
        p = np.clip(pred(Xr[~trm]), 1e-9, 1 - 1e-9)
        y = yr[~trm]
        t5[nm] = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    out["test5_feature_logistic_heldout_logloss"] = t5
    print(f"T5 logistic held-out logloss: {t5}", file=sys.stderr, flush=True)

    # ---- Test 3: scenario B, predict fresh samples ----------------------
    # posterior predictive for a fresh sample: P(y_new=1 | train) via
    # likelihood ratio with an appended sample in a NEW cluster. Length of
    # the fresh sample is unknown, so L-models are excluded here.
    def append(Y, M, ELL, CL, ynew, cnew, elnew):
        return (np.concatenate([Y, ynew[:, None]], 1),
                np.concatenate([M, np.ones((len(Y), 1))], 1),
                np.concatenate([ELL, elnew[:, None]], 1),
                np.concatenate([CL, cnew[:, None]], 1))
    Ytr, Mtr = ok[:, tr_idx], np.ones((n, 4))
    ELtr, CLtr = ell[:, tr_idx], cl[:, tr_idx]
    t3 = {}
    for name in ("E", "C"):
        f = fits[name]; par = (f["mu"], f["tau"], f["sig"], 0.0)
        base = loglik(Ytr, Mtr, ELtr, CLtr, *par)
        cnew = CLtr.max(1) + 1
        p1 = np.exp(loglik(*append(Ytr, Mtr, ELtr, CLtr, np.ones(n), cnew, np.zeros(n)), *par) - base)
        # pass@4 for four fresh samples in four fresh clusters: independent given theta
        # -> P(all fail) needs joint; compute via appending four failures in distinct new clusters
        Y4, M4, E4, C4 = Ytr, Mtr, ELtr, CLtr
        for j in range(4):
            Y4, M4, E4, C4 = append(Y4, M4, E4, C4, np.zeros(n), cnew + j, np.zeros(n))
        pfail4 = np.exp(loglik(Y4, M4, E4, C4, *par) - base)
        ppass4 = 1 - pfail4
        te = test
        eps = 1e-12
        ll_sample = (ok[te][:, te_idx] * np.log(p1[te, None] + eps)
                     + (1 - ok[te][:, te_idx]) * np.log(1 - p1[te, None] + eps)).mean()
        ll_pass4 = (heldout_any[te] * np.log(ppass4[te] + eps)
                    + (1 - heldout_any[te]) * np.log(1 - ppass4[te] + eps)).mean()
        brier_pass4 = ((ppass4[te] - heldout_any[te]) ** 2).mean()
        # calibration by (train count, distinct answers) cell
        cells = {}
        for s in range(4):
            for d in range(1, 5):
                sel = te & (s_tr == s) & (ndist == d)
                if sel.sum() >= 30:
                    cells[f"s={s},distinct={d}"] = {"n": int(sel.sum()),
                                                    "pred": float(p1[sel].mean()),
                                                    "obs": float(heldout_rate[sel].mean())}
        t3[name] = {"per_sample_logloss": float(-ll_sample),
                    "pass4_logloss": float(-ll_pass4), "pass4_brier": float(brier_pass4),
                    "aggregate_pass4_pred": float(ppass4[te].mean()),
                    "aggregate_pass4_obs": float(heldout_any[te].mean()),
                    "cells": cells}
        print(f"B {name}: per-sample logloss {-ll_sample:.4f}, pass@4 logloss {-ll_pass4:.4f}", file=sys.stderr, flush=True)
    out["test3_scenarioB"] = t3

    # ---- Test 4: scenario A, which held-out sample to trust -------------
    t4 = {}
    for name in ("E", "L", "C", "LC"):
        f = fits[name]; par = (f["mu"], f["tau"], f["sig"], f["beta"])
        base = loglik(Ytr, Mtr, ELtr, CLtr, *par)
        lls = []
        for j in te_idx:
            cj = cl[:, j]  # cluster id already consistent with training ids per prompt
            p1 = np.exp(loglik(*append(Ytr, Mtr, ELtr, CLtr, np.ones(n), cj, ell[:, j]), *par) - base)
            p1 = np.clip(p1, 1e-9, 1 - 1e-9)
            y = ok[:, j]
            lls.append(-(y * np.log(p1) + (1 - y) * np.log(1 - p1))[test])
        lls = np.stack(lls, 1)
        # split by whether the held-out sample shares an answer with a verified training sample
        shares = np.stack([np.isin(cl[:, j], cl[:, tr_idx].T.tolist() and cl[:, tr_idx]) if False else
                           np.array([cl[p, j] in set(cl[p, tr_idx]) and ah[p, j] != 0 for p in range(n)])
                           for j in te_idx], 1)[test]
        t4[name] = {"logloss_all": float(lls.mean()),
                    "logloss_shares_answer_with_verified": float(lls[shares].mean()),
                    "logloss_novel_answer": float(lls[~shares].mean()),
                    "frac_shares": float(shares.mean())}
        print(f"A {name}: logloss {lls.mean():.4f} (shares {lls[shares].mean():.4f}, novel {lls[~shares].mean():.4f})", file=sys.stderr, flush=True)
    out["test4_scenarioA"] = t4
    out["runtime_s"] = time.time() - t0
    out["argv"] = sys.argv[1:]
    out["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        import subprocess
        out["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                                         capture_output=True, text=True).stdout.strip()
    except Exception:
        out["git_head"] = None
    # append-only raw log first, then the latest-run convenience copy
    with open(os.path.join(HERE, "runs.jsonl"), "a") as fh:
        fh.write(json.dumps(out) + "\n")
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("appended runs.jsonl, wrote results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
