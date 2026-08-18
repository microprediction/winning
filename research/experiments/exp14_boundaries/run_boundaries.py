"""Experiment 14: the two boundary studies for the algorithm paper.

A. FULL-COVARIANCE BOUNDARY. The lattice transform requires a factor
   approximation Sigma ~= V V^T + D; GHK handles arbitrary Sigma. Where is the
   honest crossover? We generate correlation matrices with tunable spectral decay
   (eigenvalues ~ m^-gamma), fit factors at ranks k, and compare share errors
   against 8*10^6-draw MC truth, with GHK at R in {10^3, 10^4} as the reference.
   We EXPECTED a clear GHK-wins regime for dense slowly-decaying Sigma. The
   finding is different and reported as it fell: at N=50 and k<=8 the lattice
   matches or beats GHK at R=10^4 for every tested decay rate; the factor floor
   decays slowest at INTERMEDIATE decay (gamma=1.5), which is where a GHK
   advantage would first appear if accuracy demands exceeded the affordable
   factor floor.

B. SUBSTITUTION FIDELITY. Why want probit-style models at scale at all? Ground
   truth here is misspecified FOR EVERY candidate: utilities mu* + V* f + e with
   t(5)-distributed factors and skew-normal idiosyncratic noise. Candidates are
   calibrated to the same full-menu shares (structure V* supplied, as a
   geometry-style prior): plain logit (IIA), factor MIXED LOGIT (Gumbel
   idiosyncratic + Gaussian factors), factor PROBIT (Gaussian + Gaussian). They
   are scored on deletion counterfactuals against fresh MC truths. This isolates
   two questions: how much substitution error IIA costs, and whether the
   idiosyncratic noise law matters once the factor structure is right.

The generic-base factor forward implemented here (normal or Gumbel-min
idiosyncratic) has its own anchor: a Gumbel base with zero loadings must
reproduce softmax exactly.

Run:  python experiments/exp14_boundaries/run_boundaries.py
Outputs: results.csv, figures/full_sigma.png, figures/substitution.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp13_ghk_benchmark"))
from raceutil import factor_model_contrast, hermite_nodes, qmc_nodes  # noqa: E402
from run_ghk_benchmark import ghk_all_shares, lattice_shares, mc_shares  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 33


# ---------------------------------------------------------------------------
# Generic-base factor race (normal or gumbel-min idiosyncratic), min-wins
# ---------------------------------------------------------------------------


def _normal(z):
    S = np.maximum(1.0 - ndtr(z), 1e-300)
    f = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    fp = -z * f                                        # f'(z)
    return S, f, fp


def _gumbel_min(z):
    """STANDARDIZED min-Gumbel (mean 0, variance 1): e = (g + gamma_E) sqrt(6)/pi
    with g standard min-Gumbel. Standardization matters: without it the Gumbel
    candidate carries pi^2/6 times the idiosyncratic variance of the normal
    candidate, confounding noise FAMILY with noise SCALE (referee catch)."""
    c = np.pi / np.sqrt(6.0)
    u = np.minimum(z * c - 0.5772156649015329, 30.0)
    eu = np.exp(u)
    S = np.maximum(np.exp(-eu), 1e-300)
    f = c * eu * S
    fp = c * c * eu * S * (1.0 - eu)
    return S, f, fp


BASES = {"normal": _normal, "gumbel": _gumbel_min}


def factor_shares_base(mu, V, D, F, W, base="normal", keep=None, points=1501):
    """Min-wins win probabilities, arbitrary idiosyncratic base; also slopes."""
    mu = np.asarray(mu, dtype=float)
    V = np.atleast_2d(V)
    sd = np.sqrt(np.asarray(D, dtype=float))
    if keep is not None:
        mu, V, sd = mu[keep], V[keep], sd[keep]
    n = len(mu)
    fn = BASES[base]
    M_all = mu[None, :] + F @ V.T
    span = 22.0 if base == "gumbel" else 8.0           # std gumbel-min left tail
    x = np.linspace(M_all.min() - span * sd.max(), M_all.max() + 8.0 * sd.max(), points)
    dx = x[1] - x[0]
    p = np.zeros(n)
    slope = np.zeros(n)
    chunk = max(1, int(5e6 / (n * points)))
    for a in range(0, len(F), chunk):
        M = M_all[a:a + chunk]
        Wc = W[a:a + chunk]
        z = (x[None, None, :] - M[:, :, None]) / sd[None, :, None]
        S, f, fp = fn(z)
        f = f / sd[None, :, None]
        logS = np.log(S)
        rest = np.exp(np.clip(logS.sum(axis=1)[:, None, :] - logS, -745.0, 0.0))
        p += Wc @ (np.sum(f * rest, axis=2) * dx)
        slope += Wc @ (np.sum(-fp / sd[None, :, None] ** 2 * rest, axis=2) * dx)
    total = p.sum()
    return p / total, slope / total


def calibrate_base(target, V, D, F, W, base="normal", n_iter=60, tol=1e-8):
    """Coordinate-Newton inversion for the generic-base factor race."""
    target = np.asarray(target, dtype=float)
    target = target / target.sum()
    logt = np.log(np.maximum(target, 1e-300))
    mu = -(logt - logt.mean()) / 2.0 * (1.0 if base == "normal" else 1.0)
    for _ in range(n_iter):
        phat, sl = factor_shares_base(mu, V, D, F, W, base=base)
        resid = np.log(np.maximum(phat, 1e-300)) - logt
        if np.abs(resid).max() < tol:
            break
        dlogp = np.minimum(sl / np.maximum(phat, 1e-300), -1e-6)
        mu = mu - np.clip(resid / dlogp, -2.0, 2.0)
        mu -= mu.mean()
    return mu


def spectral_corr(n, gamma, basis):
    """Correlation matrix with pre-standardization eigenvalues ~ m^-gamma.

    Fourth-review fixes: (1) the orthogonal basis is passed in and SHARED
    across gamma values, so decay-rate comparisons are not confounded with
    basis realization; (2) the power law holds BEFORE diagonal
    standardization -- rescaling to unit diagonal perturbs the spectrum, so
    the actual post-standardization eigenvalues are returned for disclosure.
    """
    lam = np.arange(1, n + 1, dtype=float) ** (-gamma)
    C = (basis * lam) @ basis.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    return C, np.sort(np.linalg.eigvalsh(C))[::-1]


def main():
    rng = np.random.default_rng(SEED)
    rows = ["part,quantity,value"]
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)

    # ---- anchor: gumbel base at V=0 reproduces softmax exactly -----------------
    mu = rng.normal(0, 0.8, 12)
    F1, W1 = hermite_nodes(1)
    p, _ = factor_shares_base(mu, np.zeros((12, 1)), np.ones(12), F1, W1, base="gumbel")
    soft = np.exp(-mu * np.pi / np.sqrt(6.0)); soft /= soft.sum()   # unit-variance Gumbel scale
    e_anchor = np.abs(p - soft).max()
    print(f"anchor: gumbel base, V=0 vs exact softmax: {e_anchor:.2e}")
    rows.append(f"anchor,gumbel_softmax,{e_anchor:.3e}")
    assert e_anchor < 1e-4

    # ================= Part A: full-covariance boundary ==========================
    print("\nPart A: full-covariance boundary (N=50, truth = 8e6-draw MC)")
    N = 50
    KS = [1, 2, 3, 5, 8]
    gammas = [0.5, 1.5, 3.0]
    mu = rng.normal(0.0, 1.0, N)
    curves = {}
    ghk_refs = {}
    basis, _ = np.linalg.qr(rng.standard_normal((N, N)))   # SHARED across gamma
    for gamma in gammas:
        C, eig_actual = spectral_corr(N, gamma, basis)
        print(f"  gamma={gamma}: actual top-5 eigenvalues after diagonal "
              f"standardization: {[f'{e:.2f}' for e in eig_actual[:5]]}")
        L = np.linalg.cholesky(C + 1e-10 * np.eye(N))
        # MC truth under EXACT Sigma
        counts = np.zeros(N)
        r2 = np.random.default_rng(9)
        for _ in range(40):
            X = mu[:, None] + L @ r2.standard_normal((N, 200_000))
            counts += np.bincount(np.argmin(X, axis=0), minlength=N)
        truth = counts / counts.sum()
        errs, times = [], []
        Pn = np.eye(N) - np.ones((N, N)) / N
        for k in KS:
            V, D = factor_model_contrast(C, k)   # choice-relevant quotient fit
            qres = (np.linalg.norm(Pn @ (C - V @ V.T - np.diag(D)) @ Pn)
                    / np.linalg.norm(Pn @ C @ Pn))
            rows.append(f"A,gamma{gamma}_k{k}_quotient_residual,{qres:.3e}")
            Fk, Wk = hermite_nodes(k) if k <= 4 else qmc_nodes(k)
            t0 = time.perf_counter()
            pk, _ = factor_shares_base(mu, V, D, Fk, Wk)   # min-wins, as the MC truth
            times.append(time.perf_counter() - t0)
            errs.append(np.abs(pk - truth).max())
            if k in (1, 8):
                # decomposition: covariance-fit error vs integration error,
                # via an MC truth under the FITTED factor model
                r3 = np.random.default_rng(17)
                cnt = np.zeros(N)
                for _ in range(40):
                    ff = r3.standard_normal((200_000, k))
                    X = (mu[None, :] + ff @ V.T
                         + np.sqrt(D)[None, :] * r3.standard_normal((200_000, N)))
                    cnt += np.bincount(np.argmin(X, axis=1), minlength=N)
                t_fit = cnt / cnt.sum()
                e_cov = np.abs(t_fit - truth).max()
                e_int = np.abs(pk - t_fit).max()
                print(f"    k={k} decomposition: covariance-fit error "
                      f"{e_cov:.1e}, integration error {e_int:.1e}")
                rows += [f"A,gamma{gamma}_k{k}_coverr,{e_cov:.3e}",
                         f"A,gamma{gamma}_k{k}_interr,{e_int:.3e}"]
        curves[gamma] = errs
        top4 = np.sort(np.linalg.eigvalsh(C))[::-1][:4].sum() / N
        # GHK reference errors at the EXACT Sigma (its structural advantage).
        # ghk_prob computes P(max utility); our truth is min-wins, so pass -mu.
        from run_ghk_benchmark import ghk_prob
        ghk_times = {}
        for R in (1000, 10_000):
            t0 = time.perf_counter()
            pg = np.array([ghk_prob(-mu, C, i, R=R, seed=100 + i) for i in range(N)])
            ghk_times[R] = time.perf_counter() - t0
            pg = pg / pg.sum()
            ghk_refs[(gamma, R)] = np.abs(pg - truth).max()
        print(f"    wall times: lattice per k {[f'{t:.1f}s' for t in times]}; "
              f"GHK R=1e3 {ghk_times[1000]:.1f}s, R=1e4 {ghk_times[10_000]:.1f}s "
              f"(NOT wall-time matched; reported for context)")
        for k, t_ in zip(KS, times):
            rows.append(f"A,gamma{gamma}_k{k}_seconds,{t_:.2f}")
        rows += [f"A,gamma{gamma}_ghk1e3_seconds,{ghk_times[1000]:.2f}",
                 f"A,gamma{gamma}_ghk1e4_seconds,{ghk_times[10_000]:.2f}]".rstrip(']')]
        print(f"  gamma={gamma}: top-4 eig share {100*top4:.0f}%  "
              f"lattice err by k: {[f'{e:.1e}' for e in errs]}  "
              f"GHK: R=1e3 {ghk_refs[(gamma,1000)]:.1e}, R=1e4 {ghk_refs[(gamma,10000)]:.1e}")
        for k, e in zip(KS, errs):
            rows.append(f"A,gamma{gamma}_k{k},{e:.3e}")
        rows += [f"A,gamma{gamma}_ghk1e3,{ghk_refs[(gamma,1000)]:.3e}",
                 f"A,gamma{gamma}_ghk1e4,{ghk_refs[(gamma,10000)]:.3e}"]

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    cols = {0.5: "#2a1a12", 1.5: "#8a6a52", 3.0: "#c2410c"}
    for gamma in gammas:
        ax.semilogy(KS, curves[gamma], "o-", color=cols[gamma],
                    label=f"lattice, spectral decay γ={gamma}")
        ax.axhline(ghk_refs[(gamma, 10_000)], color=cols[gamma], ls=":", lw=1)
    ax.axhline(np.nan, color="#9a9a9a", ls=":", label="GHK R=10⁴ (per γ, dotted)")
    ax.set_xlabel("factor rank k")
    ax.set_ylabel("max abs share error")
    ax.set_title("Rank-k fitted-factor approximation vs GHK", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "full_sigma.png", dpi=150)

    # ================= Part B: substitution fidelity ==============================
    print("\nPart B: substitution fidelity (truth misspecified for every candidate)")
    N, k = 50, 2
    mu_true = rng.normal(0.0, 1.0, N)
    V_true = rng.normal(0.0, 0.6 / np.sqrt(k), (N, k))
    sd_true = np.sqrt(rng.uniform(0.5, 1.5, N))
    A_SKEW = 3.0

    def truth_mc(keep, n_draws, seed):
        r = np.random.default_rng(seed)
        counts = np.zeros(len(keep))
        idx = np.asarray(keep)
        done = 0
        while done < n_draws:
            m = min(200_000, n_draws - done)
            f = r.standard_t(5, size=(m, k)) / np.sqrt(5.0 / 3.0)   # unit-variance t5
            d0 = np.abs(r.standard_normal((m, len(idx))))
            d1 = r.standard_normal((m, len(idx)))
            delta = A_SKEW / np.sqrt(1 + A_SKEW**2)
            e = delta * d0 + np.sqrt(1 - delta**2) * d1              # skew-normal(a=3)
            e = (e - delta * np.sqrt(2 / np.pi)) / np.sqrt(1 - 2 * delta**2 / np.pi)
            U = mu_true[idx][None, :] + f @ V_true[idx].T + sd_true[idx][None, :] * e
            counts += np.bincount(np.argmin(U, axis=1), minlength=len(idx))
            done += m
        return counts / counts.sum()

    all_idx = np.arange(N)
    p_menu = truth_mc(all_idx, 4_000_000, 7)

    F2, W2 = hermite_nodes(2)
    D_assumed = sd_true**2
    cands = {}
    cands["plain logit (IIA)"] = ("logit", None)
    t0 = time.perf_counter()
    mu_ml = calibrate_base(p_menu, V_true, D_assumed, F2, W2, base="gumbel")
    mu_pr = calibrate_base(p_menu, V_true, D_assumed, F2, W2, base="normal")
    r_ml, _ = factor_shares_base(mu_ml, V_true, D_assumed, F2, W2, base="gumbel")
    r_pr, _ = factor_shares_base(mu_pr, V_true, D_assumed, F2, W2, base="normal")
    res_ml = float(np.abs(r_ml - p_menu).max())
    res_pr = float(np.abs(r_pr - p_menu).max())
    print(f"  calibrations done in {time.perf_counter()-t0:.0f}s; menu-share match: "
          f"mixed logit {res_ml:.1e}, probit {res_pr:.1e}")
    rows += [f"B,calib_residual_mixedlogit,{res_ml:.3e}",
             f"B,calib_residual_probit,{res_pr:.3e}"]

    def predict(name, keep):
        if name == "plain logit (IIA)":
            q = p_menu[keep]
            return q / q.sum()
        if name == "factor mixed logit":
            q, _ = factor_shares_base(mu_ml, V_true, D_assumed, F2, W2,
                                      base="gumbel", keep=keep)
        else:
            q, _ = factor_shares_base(mu_pr, V_true, D_assumed, F2, W2,
                                      base="normal", keep=keep)
        return q

    names = ["plain logit (IIA)", "factor mixed logit", "factor probit"]
    per_block = []                                     # (name, size, mass, tv)
    brng = np.random.default_rng(2)
    for size in (1, 2):
        for t in range(12):
            B = np.sort(brng.choice(N, size=size, replace=False))
            keep = np.setdiff1d(all_idx, B)
            q_true = truth_mc(keep, 2_000_000, 100 + 10 * size + t)
            mass = float(p_menu[B].sum())
            for nm in names:
                q = predict(nm, keep)
                per_block.append((nm, size, mass,
                                  0.5 * float(np.abs(q - q_true).sum())))
    # mass-stratified reporting: blocks whose deleted mass is at the MC noise
    # floor are uninformative; report large / mid strata and raw means
    print(f"  deletion blocks, stratified by deleted share mass:")
    print(f"{'model':>22} {'mass>10%':>10} {'2-10%':>8} {'raw singles':>12} {'raw pairs':>10}")
    n_big = len({(s_, m) for _, s_, m, _ in per_block if m > 0.10})
    n_mid = len({(s_, m) for _, s_, m, _ in per_block if 0.02 < m <= 0.10})
    print(f"  strata sizes: {n_big} blocks with mass>10%, {n_mid} with 2-10% "
          f"(of 24 total; small strata -> single-design caveat)")
    rows += [f"B,n_blocks_big,{n_big}", f"B,n_blocks_mid,{n_mid}"]
    for nm in names:
        big = [tv / m for n_, s_, m, tv in per_block if n_ == nm and m > 0.10]
        mid = [tv / m for n_, s_, m, tv in per_block if n_ == nm and 0.02 < m <= 0.10]
        s1 = np.mean([tv for n_, s_, m, tv in per_block if n_ == nm and s_ == 1])
        s2 = np.mean([tv for n_, s_, m, tv in per_block if n_ == nm and s_ == 2])
        big_s = f"{np.mean(big):.3f}" if big else "--"
        mid_s = f"{np.mean(mid):.3f}" if mid else "--"
        print(f"{nm:>22} {big_s:>10} {mid_s:>8} {s1:>12.4f} {s2:>10.4f}")
        rows += [f"B,{nm}_massfrac_big,{big_s}", f"B,{nm}_massfrac_mid,{mid_s}",
                 f"B,{nm}_singles,{s1:.5f}", f"B,{nm}_pairs,{s2:.5f}"]

    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
    xs = np.arange(2); wd = 0.24
    for j, (nm, c) in enumerate(zip(names, ("#9a9a9a", "#e8a87c", "#c2410c"))):
        vals = []
        for lo, hi in ((0.10, 10.0), (0.02, 0.10)):
            sel = [tv / m for n_, s_, m, tv in per_block if n_ == nm and lo < m <= hi]
            vals.append(np.mean(sel) if sel else 0.0)
        ax2.bar(xs + (j - 1) * wd, vals, wd, label=nm, color=c)
    ax2.set_xticks(xs, ["deleted mass > 10%", "deleted mass 2–10%"])
    ax2.set_ylabel("TV / deleted mass (misallocated fraction)")
    ax2.set_title("Substitution fidelity under misspecification\n"
                  "(t(5) factors, skew-normal idiosyncratic truth)", fontsize=10)
    ax2.legend(fontsize=8.5)
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout(); fig2.savefig(fig_dir / "substitution.png", dpi=150)

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print("\nwrote results.csv, figures/full_sigma.png, figures/substitution.png")


if __name__ == "__main__":
    main()
