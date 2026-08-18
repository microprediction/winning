"""Experiment 13: the GHK benchmark -- deterministic factor-probit choice
probabilities and share inversion vs the field's incumbent.

The claim under test (paper/algorithm-target.md): the lattice fast ability
transform with factor conditioning computes multinomial-probit choice
probabilities for ALL N alternatives deterministically and smoothly, at scales
where the GHK simulator -- the standard tool of discrete-choice econometrics --
is unusable; and it inverts market shares to utilities (the probit analogue of
the BLP contraction), which GHK-based practice essentially never attempts.

Model: utilities U_i = mu_i + v_i . f + sqrt(D_i) e_i, f ~ N(0, I_k), e iid
N(0,1); alternative with the largest utility is chosen. Mapped to the package's
min-wins convention by negating utilities.

Parts (every method is anchored to ground truth before any comparison):
  A. Correctness anchors: N=2 closed form Phi(dmu / sqrt(var of difference));
     N=5 all methods vs 10^7-draw Monte Carlo; package parity (thurstone
     FactorRace vs the lean raceutil implementation of the same algorithm).
  B. Accuracy/time frontier for the FULL share vector, N in {5,20,50,200}:
     GHK (R=1000, per-alternative) vs lattice vs 2*10^6-draw MC truth.
  C. Large N in {1000, 5000}: lattice vs MC truth; GHK cost extrapolated from
     its measured O(N^3 + R N^2)-per-alternative scaling (it is not run --
     reported honestly as infeasible).
  D. Derivative smoothness at N=50: second-difference noise of P_1(mu + t e_2)
     along a line -- deterministic lattice vs GHK with common random numbers
     vs GHK with fresh draws. This is the estimation-relevant metric.
  E. Share inversion at N=1000 (k=2): targets are MC shares (5*10^6 draws, so
     no inverse crime); report forward-match, utility recovery, iterations.
  F. Assortment ensemble at N=200: all single-removal share vectors from one
     conditional field pass vs per-removal recomputation.

Run:  python experiments/exp13_ghk_benchmark/run_ghk_benchmark.py
Outputs: results.csv, figures/frontier.png, figures/smoothness.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import (  # noqa: E402
    abilities_from_probabilities_factor,
    hermite_nodes,
    win_probabilities_factor,
)

HERE = Path(__file__).resolve().parent
SEED = 21


# ---------------------------------------------------------------------------
# The lattice method (argmax-utility convention wrapped around min-wins)
# ---------------------------------------------------------------------------


def lattice_shares(mu, V, D, nodes=None, weights=None):
    if nodes is None:
        nodes, weights = hermite_nodes(V.shape[1])
    return win_probabilities_factor(-np.asarray(mu), np.asarray(V), np.asarray(D),
                                    nodes, weights)


# ---------------------------------------------------------------------------
# Monte Carlo truth (plain frequency simulation; noise quantified)
# ---------------------------------------------------------------------------


def mc_shares(mu, V, D, n_draws, seed=9, mem_budget_bytes=1.5e9):
    # chunk size from an explicit memory budget: the utility matrix (m, n) plus
    # its idiosyncratic draw are the peak (2 float64 arrays + temporaries)
    n, k = V.shape
    chunk = max(10_000, int(mem_budget_bytes / (n * 8 * 4)))
    rng = np.random.default_rng(seed)
    counts = np.zeros(n)
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        f = rng.standard_normal((m, k))
        U = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((m, n))
        counts += np.bincount(np.argmax(U, axis=1), minlength=n)
        done += m
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# GHK simulator (standard; validated in part A before use)
# ---------------------------------------------------------------------------


def ghk_prob(mu, Sigma, i, R=1000, seed=9, u=None):
    """GHK estimate of P(alternative i has the max utility).

    Differences d_j = U_j - U_i <= 0 for all j != i; sequential sampling of
    truncated normals along the Cholesky of the difference covariance.
    Pass `u` (R x N-1 uniforms) for common random numbers across calls.
    """
    n = len(mu)
    others = [j for j in range(n) if j != i]
    a = mu[others] - mu[i]
    M = np.zeros((n - 1, n))
    M[np.arange(n - 1), others] = 1.0
    M[:, i] -= 1.0
    C = M @ Sigma @ M.T
    L = np.linalg.cholesky(C + 1e-12 * np.eye(n - 1))
    if u is None:
        u = np.random.default_rng(seed).random((R, n - 1))
    R = u.shape[0]
    z = np.zeros((R, n - 1))
    prob = np.ones(R)
    for kk in range(n - 1):
        b = (-a[kk] - z[:, :kk] @ L[kk, :kk]) / L[kk, kk]
        Fb = ndtr(b)
        prob *= Fb
        z[:, kk] = ndtri(np.clip(u[:, kk] * Fb, 1e-15, 1 - 1e-15))
    return float(prob.mean())


def ghk_all_shares(mu, V, D, R=1000, seed=9):
    Sigma = V @ V.T + np.diag(D)
    p = np.array([ghk_prob(mu, Sigma, i, R=R, seed=seed + i) for i in range(len(mu))])
    return p / p.sum()


# ---------------------------------------------------------------------------


def make_problem(n, k, rng, spread=1.0):
    mu = rng.normal(0.0, spread, n)
    V = rng.normal(0.0, 0.5 / np.sqrt(k), (n, k))
    D = rng.uniform(0.5, 1.5, n)
    return mu, V, D


def main():
    rng = np.random.default_rng(SEED)
    rows = ["part,quantity,value"]

    # ---- Part A: correctness anchors ---------------------------------------
    print("Part A: correctness anchors")
    mu2 = np.array([0.3, -0.2])
    V2 = np.array([[0.6], [-0.1]])
    D2 = np.array([0.8, 1.2])
    var_diff = (V2[0, 0] - V2[1, 0]) ** 2 + D2[0] + D2[1]
    exact = ndtr((mu2[0] - mu2[1]) / np.sqrt(var_diff))
    p_lat = lattice_shares(mu2, V2, D2)
    Sig2 = V2 @ V2.T + np.diag(D2)
    p_ghk = ghk_prob(mu2, Sig2, 0, R=200_000)
    print(f"  N=2 closed form {exact:.6f}: lattice err {abs(p_lat[0]-exact):.2e}, "
          f"GHK err {abs(p_ghk-exact):.2e}")
    rows += [f"A,n2_lattice_err,{abs(p_lat[0]-exact):.3e}",
             f"A,n2_ghk_err,{abs(p_ghk-exact):.3e}"]
    assert abs(p_lat[0] - exact) < 5e-4 and abs(p_ghk - exact) < 5e-3

    mu5, V5, D5 = make_problem(5, 2, rng)
    truth5 = mc_shares(mu5, V5, D5, 10_000_000)
    e_lat = np.abs(lattice_shares(mu5, V5, D5) - truth5).max()
    e_ghk = np.abs(ghk_all_shares(mu5, V5, D5, R=100_000) - truth5).max()
    print(f"  N=5 vs 1e7-draw MC (noise ~2e-4): lattice {e_lat:.2e}, GHK {e_ghk:.2e}")
    rows += [f"A,n5_lattice_err,{e_lat:.3e}", f"A,n5_ghk_err,{e_ghk:.3e}"]
    assert e_lat < 1e-3 and e_ghk < 2e-3

    # package parity: the thurstone FactorRace implements the same algorithm
    try:
        from thurstone import Density, FactorRace, UniformLattice
        lat = UniformLattice(L=500, unit=0.02)
        bases = [Density.skew_normal(lat, 0.0, float(np.sqrt(d)), 0.0) for d in D5]
        p_pkg = FactorRace(bases, -mu5, V5).state_prices()
        par = np.abs(p_pkg - lattice_shares(mu5, V5, D5)).max()
        print(f"  package parity (thurstone.FactorRace): max diff {par:.2e}")
        rows.append(f"A,package_parity,{par:.3e}")
    except ImportError:
        print("  thurstone not importable; parity check skipped")

    # ---- Part B: accuracy/time frontier, full share vector ------------------
    print("\nPart B: full share vector, GHK vs lattice vs MC truth")
    Ns = [5, 20, 50, 200]
    t_ghk_list, t_lat_list, e_ghk_list, e_lat_list = [], [], [], []
    def timed3(fn):
        fn()                                    # warm-up discarded
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
        return out, float(np.median(ts))

    for n in Ns:
        mu, V, D = make_problem(n, 2, rng)
        truth = mc_shares(mu, V, D, 2_000_000)
        p_l, t_l = timed3(lambda: lattice_shares(mu, V, D))
        p_g, t_g = timed3(lambda: ghk_all_shares(mu, V, D, R=1000))
        rel_l = np.abs(p_l - truth).max()
        rel_g = np.abs(p_g - truth).max()
        t_lat_list.append(t_l); t_ghk_list.append(t_g)
        e_lat_list.append(rel_l); e_ghk_list.append(rel_g)
        print(f"  N={n:>4}: lattice {t_l*1e3:7.0f} ms err {rel_l:.1e}   "
              f"GHK(R=1000) {t_g*1e3:8.0f} ms err {rel_g:.1e}")
        rows.append(f"B,{n},lat_ms={t_l*1e3:.1f};lat_err={rel_l:.2e};"
                    f"ghk_ms={t_g*1e3:.1f};ghk_err={rel_g:.2e}")

    # ---- Part C: large N (GHK infeasible; cost extrapolated) ----------------
    print("\nPart C: large N")
    big_Ns, big_t = [], []
    for n in (1000, 5000):
        mu, V, D = make_problem(n, 2, rng, spread=1.5)
        t0 = time.perf_counter(); p_l = lattice_shares(mu, V, D)
        t_l = time.perf_counter() - t0
        truth = mc_shares(mu, V, D, 500_000)
        err = np.abs(p_l - truth).max()
        noise = np.sqrt(truth.max() / 500_000)
        # GHK cost extrapolated by empirical power law fitted on part B timings
        alpha, logc = np.polyfit(np.log(Ns[1:]), np.log(t_ghk_list[1:]), 1)
        t_ghk_extrap = np.exp(logc) * n ** alpha
        big_Ns.append(n); big_t.append(t_l)
        print(f"  N={n}: lattice {t_l:.1f}s, err {err:.1e} (MC noise ~{noise:.0e}); "
              f"GHK extrapolated ~{t_ghk_extrap/3600:.1f} h")
        rows.append(f"C,{n},lat_s={t_l:.2f};err={err:.2e};ghk_extrap_h={t_ghk_extrap/3600:.2f}")

    # ---- Part D: derivative smoothness ---------------------------------------
    print("\nPart D: derivative smoothness (second-difference noise along a line)")
    n = 50
    mu, V, D = make_problem(n, 2, rng)
    Sigma = V @ V.T + np.diag(D)
    ts = np.linspace(-0.5, 0.5, 41)
    F, W = hermite_nodes(2)
    u_crn = np.random.default_rng(4).random((1000, n - 1))
    curves = {"lattice": [], "GHK-CRN": [], "GHK-fresh": []}
    for j, t in enumerate(ts):
        m = mu.copy(); m[2] += t
        curves["lattice"].append(lattice_shares(m, V, D, F, W)[1])
        curves["GHK-CRN"].append(ghk_prob(m, Sigma, 1, u=u_crn))
        curves["GHK-fresh"].append(ghk_prob(m, Sigma, 1, R=1000, seed=100 + j))
    noise = {kk: float(np.abs(np.diff(v, 2)).max() / (ts[1] - ts[0]) ** 2)
             for kk, v in curves.items()}
    print("  max |second difference| / dt^2 (curvature noise):")
    for kk, v in noise.items():
        print(f"    {kk:>10}: {v:9.2f}")
        rows.append(f"D,{kk},{v:.3f}")

    # ---- Part E: share inversion at N=1000 ------------------------------------
    print("\nPart E: share inversion (probit BLP step), N=1000, k=2")
    n = 1000
    mu_true, V, D = make_problem(n, 2, rng, spread=1.2)
    mu_true -= mu_true.mean()
    target = mc_shares(mu_true, V, D, 5_000_000)      # MC target: no inverse crime
    target = np.maximum(target, 1e-7); target /= target.sum()
    t0 = time.perf_counter()
    mu_hat = abilities_from_probabilities_factor(target, V, D, *hermite_nodes(2))
    t_inv = time.perf_counter() - t0
    back = win_probabilities_factor(mu_hat, V, D, *hermite_nodes(2))
    fwd_err = np.abs(back - target).max()
    # recovery on well-identified alternatives (shares above ~3x MC noise);
    # mu_hat is a min-wins ability = -utility, identified up to location
    good = target > 3e-4
    util_hat = -mu_hat
    err_util = np.abs((util_hat - util_hat[good].mean())
                      - (mu_true - mu_true[good].mean()))[good].max()
    print(f"  inverted in {t_inv:.0f}s; forward-match {fwd_err:.2e} "
          f"(target noise ~{np.sqrt(target.max()/5e6):.0e}); "
          f"utility recovery (identified alts) max err {err_util:.3f}")
    rows += [f"E,invert_s,{t_inv:.1f}", f"E,forward_match,{fwd_err:.3e}",
             f"E,utility_recovery,{err_util:.4f}"]

    # ---- Part F: assortment ensemble ------------------------------------------
    print("\nPart F: assortment (single-removal) ensemble, N=200")
    n = 200
    mu, V, D = make_problem(n, 2, rng)
    F2, W2 = hermite_nodes(2)
    t0 = time.perf_counter()
    _, q = win_probabilities_factor(-mu, V, D, F2, W2, return_deletions=True)
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    worst = 0.0
    for i in range(0, n, 20):                          # sample of direct recomputes
        keep = np.setdiff1d(np.arange(n), [i])
        direct = win_probabilities_factor(-mu[keep], V[keep], D[keep], F2, W2)
        worst = max(worst, np.abs(direct - q[i][keep]).max())
    t_each = (time.perf_counter() - t0) / (n // 20)
    print(f"  one pass: {t_one:.1f}s for all {n} removals vs {t_each*n:.1f}s "
          f"recomputing ({t_each*n/t_one:.1f}x); max diff {worst:.1e}")
    rows += [f"F,onepass_s,{t_one:.2f}", f"F,recompute_s,{t_each*n:.2f}",
             f"F,max_diff,{worst:.2e}"]

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")

    # ---- figures ------------------------------------------------------------------
    fig_dir = HERE / "figures"; fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    alpha_fit, logc_fit = np.polyfit(np.log(Ns[1:]), np.log(t_ghk_list[1:]), 1)
    Nx = np.array([200, 500, 1000, 2000, 5000], dtype=float)
    # matched-error GHK: scale R by (err_ghk/err_lat)^2 per R^{-1/2}; cost ~ R
    t_matched = [t * (eg / el) ** 2 for t, eg, el
                 in zip(t_ghk_list, e_ghk_list, e_lat_list)]
    am, lm = np.polyfit(np.log(Ns[1:]), np.log(t_matched[1:]), 1)
    ax.loglog(Nx, np.exp(lm) * Nx**am, "--", color="#8a6a52", lw=1.2)
    ax.loglog(Ns, t_matched, "^-", color="#8a6a52",
              label="GHK at lattice-matched error ($R^{-1/2}$ scaled)")
    ax.loglog(Nx, np.exp(logc_fit) * Nx**alpha_fit, "--", color="#2a1a12", lw=1)
    ax.loglog(Ns, np.array(t_ghk_list), "o-", color="#2a1a12",
              label="GHK at R=1000 (err $\\sim\\!10^{-3}$--$10^{-2}$)")
    ax.loglog(Ns + big_Ns, np.array(t_lat_list + big_t), "s-", color="#c2410c",
              label="lattice transform (err $\\sim\\!3\\times10^{-4}$)")
    ax.set_xlabel("number of alternatives N")
    ax.set_ylabel("wall time for the full share vector (s)")
    ax.set_title("All N factor-probit shares: wall time vs N\n(dashed: power-law extrapolation)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "frontier.png", dpi=150)

    fig2, ax2 = plt.subplots(figsize=(6, 4.4))
    for kk, c in zip(curves, ("#c2410c", "#2a1a12", "#9a9a9a")):
        ax2.plot(ts, curves[kk], "-", color=c, lw=1.4, label=kk)
    ax2.set_xlabel(r"perturbation $t$ of a rival's utility")
    ax2.set_ylabel(r"$P_2(\mu + t e_3)$")
    ax2.set_title("Fixed-design smoothness along a utility path:\n"
                  "choice probability along a line", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)
    fig2.tight_layout(); fig2.savefig(fig_dir / "smoothness.png", dpi=150)
    print("\nwrote results.csv, figures/frontier.png, figures/smoothness.png")


if __name__ == "__main__":
    main()
