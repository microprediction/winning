"""Extensions of the T-Rex race study (#1030), after run_trex_race.py has
saved the race table race_table_L1p.npy (P_T(s), T = 1..T_cap, on S_GRID).

A. Position-aware statistic. T-Rex keeps "entered before the T-th dummy or
   not". The race model also knows WHERE: at termination T, variable j
   entered ahead of exactly d_j of the L dummies (d_j < T), or is censored
   (d_j >= T). The dummies are i.i.d. half-normal and the real variable is
   a noiseless runner of strength s (see run_trex_race), so the likelihood
   of d_j given s is the place probability in the identical field,
   P(exactly d ahead | s) = P_{d+1}(s) - P_d(s) = Binomial(L, P(|Z| > s))
   at d, read off the race table; the censored event has 1 - P_T(s). The K
   experiments are independent given the data, so the posterior over s on
   a grid under a half-normal(s0) prior is an exact product; the statistic
   is its mean. If the positions carry nothing beyond inclusion, this is a
   monotone transform of Phi and the AUCs coincide.
B. Dependent designs: AR(1) rho = 0.5 (their Fig. 7) and a clustered design
   (blocks of 5, within-block correlation 0.8, actives one per block).
   (i) T-Rex as published; (ii) selection ranked by the position-aware
   statistic, sized by the T-Rex FDP estimator; (iii) a block statistic,
   P(the best of a block is ahead of the T-th dummy) under the race model,
   integrated over the members' posteriors, selection at block level. The
   runners are noiseless, so the block's design correlation has no
   performance noise to act on; it enters only the law of a null block's
   strengths (the synthetic reference blocks of C).
C. Calibration: realised FDP against the T-Rex estimate V-hat / R on the
   sets (ii) and (iii) select, which have no proof behind them, and
   against the race-model estimate (synthetic reference runners drawn from
   the model, counted above the same threshold).

Usage: OMP_NUM_THREADS=1 python run_trex_extensions.py [MC] [designs]
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import lars_path

from run_trex_race import (N, P, P1, SNR, K, L, ALPHA, V_GRID, S_GRID, MAX_ITER, deflated, race_table, HERE)

MC = int(sys.argv[1]) if len(sys.argv) > 1 else 50
DESIGNS = sys.argv[2:] or ["independent", "ar1", "clustered"]
T_CAP = int(os.environ.get("T_CAP", 30))
TAB = race_table(L, T_CAP, S_GRID)                                # (T_cap, len(S_GRID)), closed form
T_EVAL = [t for t in (5, 10, 20) if t <= T_CAP]
S0 = 2.0                        # half-normal prior sd on the strength (the grid is s >= 0)
N_SYNTH = 4000
BLOCK = 5
rng = np.random.default_rng(20251109)


# ---------------------------------------------------------------- designs
def design_cov(kind):
    if kind == "independent":
        return None
    if kind == "ar1":
        i = np.arange(P)
        return 0.5 ** np.abs(i[:, None] - i[None, :])
    if kind == "clustered":
        C = np.eye(P)
        for b in range(P // BLOCK):
            sl = slice(b * BLOCK, (b + 1) * BLOCK)
            C[sl, sl] = 0.8
            C[sl, sl][np.diag_indices(BLOCK)] = 1.0
        np.fill_diagonal(C, 1.0)
        return C
    raise ValueError(kind)


def simulate(kind, chol):
    Z = rng.standard_normal((N, P))
    X = Z if chol is None else Z @ chol.T
    beta = np.zeros(P)
    if kind == "clustered":
        blocks = rng.choice(P // BLOCK, P1, replace=False)
        act = blocks * BLOCK + rng.integers(0, BLOCK, P1)
    else:
        act = rng.choice(P, P1, replace=False)
    beta[act] = 1.0
    signal = X @ beta
    y = signal + np.sqrt(signal.var() / SNR) * rng.standard_normal(N)
    return X, y, beta, np.sort(act)


def experiments_positions(X, y):
    """Per experiment: each candidate's entry index (inf if never) and the
    dummies' entry indices, sorted."""
    cand = np.full((K, P), np.inf); dums = []
    for k in range(K):
        D = rng.standard_normal((N, L))
        _, active, _ = lars_path(np.hstack([X, D]), y, method="lar", max_iter=MAX_ITER)
        active = np.asarray(active)
        isd = active >= P
        cand[k, active[~isd]] = np.nonzero(~isd)[0]
        dums.append(np.nonzero(isd)[0])
    return cand, dums


def inclusion_and_rank(cand, dums, T):
    """At termination T: Phi_T (P,), and d (K, P) = dummies ahead of each
    candidate, T meaning censored (not included)."""
    d = np.full((K, P), T, dtype=int)
    for k in range(K):
        if len(dums[k]) >= T:
            cut = dums[k][T - 1]
            inc = cand[k] < cut
            # dummies ahead of j = number of dummy positions below cand[k, j]
            d[k, inc] = np.searchsorted(dums[k], cand[k, inc])
        else:
            inc = cand[k] < np.inf
            d[k, inc] = np.searchsorted(dums[k], cand[k, inc])
    return (d < T).mean(axis=0), d


# ---------------------------------------------------------------- the race-model statistic
def loglik_table(T):
    """(T + 1, S) log-likelihoods: rows 0..T-1 exactly d dummies ahead,
    row T censored (>= T ahead)."""
    P_ = np.vstack([np.zeros(TAB.shape[1]), TAB[:T]])           # P_0 = 0, P_1..P_T
    exact = np.diff(P_, axis=0)                                 # P_{d+1} - P_d, d = 0..T-1
    cens = 1.0 - TAB[T - 1]
    return np.log(np.clip(np.vstack([exact, cens]), 1e-300, None))


def posterior(d, LL):
    """d: (K, m) outcomes in 0..T; LL: (T+1, S). Posterior weights (m, S)
    of s on S_GRID under a half-normal(S0) prior."""
    lp = norm.logpdf(S_GRID, 0, S0)[None, :] + LL[d].sum(axis=0)    # (m, S)
    lp -= lp.max(axis=1, keepdims=True)
    w = np.exp(lp); w /= w.sum(axis=1, keepdims=True)
    return w


def posterior_mean(d, LL):
    return posterior(d, LL) @ S_GRID


def posterior_draws(w, n_draws):
    """(m, n_draws) strengths drawn from grid posteriors w (m, S)."""
    c = np.cumsum(w, axis=1)
    u = rng.random((w.shape[0], n_draws))
    idx = np.array([np.searchsorted(c[i], u[i]) for i in range(w.shape[0])]).clip(0, len(S_GRID) - 1)
    return S_GRID[idx]


def auc(stat, is_act):
    """Mann-Whitney AUC with ties at half."""
    a, b = stat[is_act], stat[~is_act]
    return float((np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :])))


def synthetic_null_stats(T, LL, n=N_SYNTH, R=None):
    """Outcomes of reference runners drawn from the race model: strength
    s = |z|, z ~ N(0, 1) (or a correlated block of them, R given), K
    outcomes drawn from the place law. Returns the outcomes d (K, n) and s."""
    if R is None:
        s = np.abs(rng.standard_normal(n))
    else:
        s = np.abs((rng.standard_normal((n // R.shape[0], R.shape[0])) @ np.linalg.cholesky(R).T).ravel())
        n = len(s)
    probs = np.exp(LL).T                                          # (S, T+1)
    pi = np.stack([np.interp(s, S_GRID, probs[:, c]) for c in range(T + 1)], axis=1)
    pi /= pi.sum(axis=1, keepdims=True)
    cum = np.cumsum(pi, axis=1)
    u = rng.random((K, n))
    d = (u[:, :, None] > cum[None, :, :]).sum(axis=2).clip(0, T)
    return d, s


def block_statistic(w, T, draws=400):
    """P(the best of the block is ahead of the T-th dummy), integrated over
    the members' posteriors: w (nb * B, S) posterior weights; the block's
    best strength is the max over members, and P_T of it is read off the
    race table."""
    nb = w.shape[0] // BLOCK
    s = posterior_draws(w, draws).reshape(nb, BLOCK, draws).max(axis=1)      # (nb, draws)
    return np.interp(s, S_GRID, TAB[T - 1]).mean(axis=1)


def size_by_model(order, stat, synth, n0, alpha):
    """Largest prefix of `order` whose race-model estimate of selected
    nulls, n0 * P(synthetic null statistic >= the prefix's last statistic),
    divided by R is <= alpha."""
    ss = np.sort(synth)
    frac = 1.0 - np.searchsorted(ss, stat[order], side="left") / len(ss)
    R = np.arange(1, len(order) + 1)
    ok = np.nonzero(n0 * frac / R <= alpha)[0]
    return int(ok[-1] + 1) if len(ok) else 0


T_MODEL = (1, 2, 3, 4, 5, 7, 10, 15, 20, 30)      # the model-sized arms search these T (cost)


def size_by_vhat(order, one_minus_phi, alpha):
    """Largest prefix of `order` whose V-hat / R <= alpha (V-hat = sum of
    1 - Phi' over the prefix)."""
    v = np.cumsum(one_minus_phi[order])
    R = np.arange(1, len(order) + 1)
    ok = np.nonzero(v / R <= alpha)[0]
    return int(ok[-1] + 1) if len(ok) else 0


def trex_published(phi, phi_def, T_max):
    """Algorithm 1 on Phi (the calibration of the paper)."""
    best = (0, None, None)
    T = 1
    while T <= T_max:
        sel_hi = phi[T - 1] > V_GRID[-1]
        if sel_hi.sum() and (1 - phi_def[T - 1][sel_hi]).sum() / sel_hi.sum() > ALPHA:
            break
        for v in V_GRID:
            sel = phi[T - 1] > v; R = sel.sum()
            if R and (1 - phi_def[T - 1][sel]).sum() / R <= ALPHA and R > best[0]:
                best = (int(R), float(v), T)
        T += 1
    return best


# ---------------------------------------------------------------- main
def run_design(kind):
    t0 = time.time()
    C = design_cov(kind); chol = None if C is None else np.linalg.cholesky(C)
    blocks = np.arange(P) // BLOCK
    nb = P // BLOCK
    Rb = np.eye(BLOCK) if C is None else C[:BLOCK, :BLOCK]
    LLs = {T: loglik_table(T) for T in range(1, T_CAP + 1)}
    A_auc = {T: {"phi": [], "position": []} for T in T_EVAL}
    B = {"trex": [], "position": [], "block": [], "trex_blocklevel": [], "position_model": [], "block_model": []}
    Cc = {"position": [], "block": [], "position_model": [], "block_model": []}
    for rep in range(MC):
        X, y, beta_, act = simulate(kind, chol)
        is_act = np.zeros(P, bool); is_act[act] = True
        blk_act = np.zeros(nb, bool); blk_act[np.unique(blocks[act])] = True
        cand, dums = experiments_positions(X, y)
        T_max = min(T_CAP, min(len(x) for x in dums))
        phi = np.zeros((T_CAP, P)); dd = {}
        for T in range(1, T_CAP + 1):
            phi[T - 1], dd[T] = inclusion_and_rank(cand, dums, T)
        phi_def = deflated(phi, L)
        # A
        for T in T_EVAL:
            m = posterior_mean(dd[T], LLs[T])
            A_auc[T]["phi"].append(auc(phi[T - 1], is_act)); A_auc[T]["position"].append(auc(m, is_act))
        # B (i) as published
        R, v, T = trex_published(phi, phi_def, T_max)
        sel = (phi[T - 1] > v) if R else np.zeros(P, bool)
        B["trex"].append({"R": int(R), "fdp": float((sel & ~is_act).sum() / max(R, 1)), "tpp": float((sel & is_act).sum() / P1), "T": T})
        bs = np.zeros(nb, bool); bs[np.unique(blocks[sel])] = True
        B["trex_blocklevel"].append({"R": int(bs.sum()), "fdp": float((bs & ~blk_act).sum() / max(bs.sum(), 1)), "tpp": float((bs & blk_act).sum() / P1)})
        # B (ii) ranked by the position-aware statistic, sized by V-hat; best over T
        best = (0, None, None, None)
        for T in range(1, T_max + 1):
            m = posterior_mean(dd[T], LLs[T])
            order = np.argsort(-m)
            Rn = size_by_vhat(order, 1 - phi_def[T - 1], ALPHA)
            if Rn > best[0]:
                best = (Rn, T, order[:Rn], m)
        Rn, T, chosen, m = best
        sel = np.zeros(P, bool)
        if Rn:
            sel[chosen] = True
        fdp = float((sel & ~is_act).sum() / max(Rn, 1))
        B["position"].append({"R": int(Rn), "fdp": fdp, "tpp": float((sel & is_act).sum() / P1), "T": T})
        if Rn:
            vhat = float((1 - phi_def[T - 1][sel]).sum() / Rn)
            thr = m[chosen].min()
            d_syn, _ = synthetic_null_stats(T, LLs[T])
            synth = posterior_mean(d_syn, LLs[T])
            model = float((P - P1) * (synth >= thr).mean() / Rn)
            Cc["position"].append({"fdp": fdp, "vhat": vhat, "model": model})
        # (ii-m) the same ranking, sized by the race-model estimate (synthetic reference runners)
        best = (0, None, None, None, None)
        for T in [t for t in T_MODEL if t <= T_max]:
            m = posterior_mean(dd[T], LLs[T]); order = np.argsort(-m)
            d_syn, _ = synthetic_null_stats(T, LLs[T]); synth = posterior_mean(d_syn, LLs[T])
            Rm = size_by_model(order, m, synth, P - P1, ALPHA)
            if Rm > best[0]:
                best = (Rm, T, order[:Rm], m, synth)
        Rm, T, chosen, m, synth = best
        sel = np.zeros(P, bool)
        if Rm:
            sel[chosen] = True
        fdp = float((sel & ~is_act).sum() / max(Rm, 1))
        B["position_model"].append({"R": int(Rm), "fdp": fdp, "tpp": float((sel & is_act).sum() / P1), "T": T})
        if Rm:
            Cc["position_model"].append({"fdp": fdp, "vhat": float((1 - phi_def[T - 1][sel]).sum() / Rm),
                                         "model": float((P - P1) * (synth >= m[chosen].min()).mean() / Rm)})
        # B (iii) block statistic, selection at block level
        best = (0, None, None, None, None)
        for T in range(1, T_max + 1):
            st = block_statistic(posterior(dd[T], LLs[T]), T)
            phiB = (dd[T].reshape(K, nb, BLOCK).min(axis=2) < T).mean(axis=0)      # any member included
            order = np.argsort(-st)
            Rb_ = size_by_vhat(order, 1 - phiB, ALPHA)
            if Rb_ > best[0]:
                best = (Rb_, T, order[:Rb_], st, phiB)
        Rb_, T, chosen, st, phiB = best
        bs = np.zeros(nb, bool)
        if Rb_:
            bs[chosen] = True
        fdpB = float((bs & ~blk_act).sum() / max(Rb_, 1))
        B["block"].append({"R": int(Rb_), "fdp": fdpB, "tpp": float((bs & blk_act).sum() / P1), "T": T})
        if Rb_:
            vhat = float((1 - phiB[bs]).sum() / Rb_)
            thr = st[chosen].min()
            d_syn, _ = synthetic_null_stats(T, LLs[T], n=BLOCK * 800, R=Rb)
            stB = block_statistic(posterior(d_syn, LLs[T]), T)
            model = float((nb - P1) * (stB >= thr).mean() / Rb_)
            Cc["block"].append({"fdp": fdpB, "vhat": vhat, "model": model})
        # (iii-m) the block statistic, sized by the race-model estimate (synthetic null blocks)
        best = (0, None, None, None, None, None)
        for T in [t for t in T_MODEL if t <= T_max]:
            st = block_statistic(posterior(dd[T], LLs[T]), T); order = np.argsort(-st)
            d_syn, _ = synthetic_null_stats(T, LLs[T], n=BLOCK * 400, R=Rb)
            stB = block_statistic(posterior(d_syn, LLs[T]), T, draws=200)
            Rm = size_by_model(order, st, stB, nb - P1, ALPHA)
            if Rm > best[0]:
                phiB = (dd[T].reshape(K, nb, BLOCK).min(axis=2) < T).mean(axis=0)
                best = (Rm, T, order[:Rm], st, stB, phiB)
        Rm, T, chosen, st, stB, phiB = best
        bs = np.zeros(nb, bool)
        if Rm:
            bs[chosen] = True
        fdpB = float((bs & ~blk_act).sum() / max(Rm, 1))
        B["block_model"].append({"R": int(Rm), "fdp": fdpB, "tpp": float((bs & blk_act).sum() / P1), "T": T})
        if Rm:
            Cc["block_model"].append({"fdp": fdpB, "vhat": float((1 - phiB[bs]).sum() / Rm),
                                      "model": float((nb - P1) * (stB >= st[chosen].min()).mean() / Rm)})
        if (rep + 1) % 10 == 0:
            print(f"  {kind} rep {rep + 1}/{MC} {time.time() - t0:.0f}s", flush=True)

    def ms(x):
        x = np.asarray(x, float)
        return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))] if len(x) > 1 else [float(x.mean()) if len(x) else None, None]
    out = {"design": kind, "MC": MC, "seconds": time.time() - t0,
           "A_auc": {T: {"phi": ms(A_auc[T]["phi"]), "position": ms(A_auc[T]["position"]),
                         "paired_diff": ms(np.array(A_auc[T]["position"]) - np.array(A_auc[T]["phi"]))} for T in T_EVAL},
           "B": {k_: {"fdr": ms([r["fdp"] for r in v_]), "tpr": ms([r["tpp"] for r in v_]), "mean_R": float(np.mean([r["R"] for r in v_])),
                      "share_empty": float(np.mean([r["R"] == 0 for r in v_]))} for k_, v_ in B.items()},
           "C": {k_: {"n": len(v_), "fdp": ms([r["fdp"] for r in v_]), "vhat": ms([r["vhat"] for r in v_]), "model": ms([r["model"] for r in v_]),
                      "fdp_minus_vhat": ms([r["fdp"] - r["vhat"] for r in v_]), "fdp_minus_model": ms([r["fdp"] - r["model"] for r in v_])}
                 for k_, v_ in Cc.items()}}
    return out


def main():
    res = {}
    for kind in DESIGNS:
        print(kind, flush=True)
        res[kind] = run_design(kind)
        print(json.dumps({k: res[kind][k] for k in ("A_auc", "B", "C")}, indent=1), flush=True)
    path = os.path.join(HERE, f"results_extensions_MC{MC}.json")
    json.dump(res, open(path, "w"), indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
