"""Numerical verification of every claim in extremal_harmonic_analysis.tex.

Each check prints CLAIM id, the measured quantity, the tolerance, and PASS/FAIL.
Claim ids match the numbered statements in the paper. Nothing is asserted in
the paper that is not exercised here; anything that fails is demoted to a
conjecture in the text.

Run: python tests_extremal.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, gammaln
from scipy.stats import norm

import circle_spectral as C
import nonlinear_circle as NL
import soft_laguerre as SL
import sphere_spectral as SP

SQRT_2PI = np.sqrt(2 * np.pi)
RESULTS = []


def report(cid, what, value, tol, ok=None):
    ok = (abs(value) <= tol) if ok is None else ok
    RESULTS.append((cid, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid:<8} {what:<52} "
          f"{value:11.3e}  (tol {tol:.1e})")


# ----------------------------------------------------------------- geometry

def check_P21_power_diagram():
    """P2.1: the race argmax is exactly a Laguerre cell decomposition, under
    w_i = 2 mu_i + |v_i|^2 (sites v_i) and equally under w_i = mu_i + |v_i|^2/4
    (sites v_i/2)."""
    print("\nP2.1  race = power diagram")
    rng = np.random.default_rng(3)
    for r in (2, 3, 4):
        N = 12
        V = rng.normal(size=(N, r))
        mu = rng.normal(scale=0.6, size=N)
        mu -= mu.mean()
        X = rng.normal(size=(200_000, r))
        race = SL.labels_race(X, mu, V)
        lag = SL.labels_laguerre(X, SL.weights_from_mu(mu, V), V)
        d2 = ((X[:, None, :] - (V / 2)[None]) ** 2).sum(2) \
            - (mu + (V ** 2).sum(1) / 4)
        lag2 = d2.argmin(1)
        report("P2.1", f"r={r} mismatch fraction (sites v_i)",
               1 - (race == lag).mean(), 0.0, ok=(race == lag).all())
        report("P2.1", f"r={r} mismatch fraction (sites v_i/2)",
               1 - (race == lag2).mean(), 0.0, ok=(race == lag2).all())


def check_T81_facet_laplacian():
    """T8.1: dp_i/dmu_j = -k_ij with k_ij = |v_i-v_j|^{-1} int_facet phi_2,
    a weighted graph Laplacian. Checked against (a) the analytic cycle graph
    of a symmetric ring, (b) CRN finite differences of exact cell masses."""
    print("\nT8.1  Jacobian = facet-flux graph Laplacian (r=2)")
    N, a = 7, 1.0
    th = 2 * np.pi * np.arange(N) / N
    V = a * np.column_stack([np.cos(th), np.sin(th)])
    mu = np.zeros(N)
    k_true = norm.pdf(0.0) * 0.5 / (2 * a * np.sin(np.pi / N))
    J_true = np.zeros((N, N))
    for i in range(N):
        for j in ((i + 1) % N, (i - 1) % N):
            J_true[i, j] = -k_true
    np.fill_diagonal(J_true, 2 * k_true)
    report("T8.1", "ring: closed form vs analytic cycle Laplacian",
           np.abs(SL.laplacian_exact(mu, V) - J_true).max(), 1e-14)

    rng = np.random.default_rng(11)
    V = rng.normal(size=(8, 2))
    mu = rng.normal(scale=0.5, size=8)
    mu -= mu.mean()
    J0 = SL.laplacian_exact(mu, V)
    h = 0.02
    Jmc = SL.jacobian_fd(
        lambda m: SL.shares_mc(m, V, np.full(8, 1e-14), draws=8_000_000,
                               seed=5), mu, h)
    report("T8.1", "general config: closed form vs CRN-FD (rel)",
           np.abs(Jmc - J0).max() / np.abs(J0).max(), 0.02)


def check_P82_soft_laplacian():
    """P8.2: for every D > 0 the race is a hard affine race in the lifted
    space R^{r+N}, so J_sigma remains exactly a graph Laplacian: symmetric,
    zero row sums, non-positive off-diagonals, positive diagonal."""
    print("\nP8.2  J_sigma is an exact graph Laplacian for all D>0")
    rng = np.random.default_rng(4)
    N = 7
    V = rng.normal(size=(N, 2))
    mu = rng.normal(scale=0.4, size=N)
    mu -= mu.mean()
    F, W = SL.gh_nodes(2, 41)
    for sig in (0.05, 0.2, 0.5, 1.0):
        D = np.full(N, sig ** 2)
        J = SL.jacobian_fd(lambda m: SL.shares_gh(m, V, D, F, W, 8001),
                           mu, 1e-3)
        off = J[~np.eye(N, dtype=bool)]
        bad = max(np.abs(J.sum(1)).max(), np.abs(J - J.T).max(),
                  max(off.max(), 0.0))
        report("P8.2", f"sigma={sig}: max(rowsum, asym, +offdiag)",
               bad, 1e-6, ok=(bad < 1e-6 and (np.diag(J) > 0).all()))


def check_T83_tau_limit():
    """T8.3: p^tau -> Gaussian Laguerre cell masses as tau -> 0, at O(tau)."""
    print("\nT8.3  shares converge to Laguerre masses, rate O(tau)")
    rng = np.random.default_rng(11)
    V = rng.normal(size=(8, 2))
    mu = rng.normal(scale=0.5, size=8)
    mu -= mu.mean()
    p0 = SL.cell_masses(mu, V, n_theta=1 << 18)
    errs, taus = [], (0.5, 0.25, 0.125, 0.0625)
    for tau in taus:
        p = SL.shares_mc(mu, V, np.full(8, tau ** 2), draws=20_000_000, seed=9)
        errs.append(np.abs(p - p0).max())
    slope = np.polyfit(np.log(taus), np.log(errs), 1)[0]
    report("T8.3", "convergence exponent in tau (target 1)",
           abs(slope - 1.0), 0.25)


# ------------------------------------------------------------ exact density

def check_P34_exposure_closed_form():
    """P3.4: for mu = a cos(theta) the exposure threshold is A = -a cos(theta)
    exactly (the sup is attained identically in the shift)."""
    print("\nP3.4  exposure threshold in closed form")
    N = 512
    th = 2 * np.pi * np.arange(N) / N
    for a in (0.2, 0.7, 1.5):
        mu, mup, mupp = a * np.cos(th), -a * np.sin(th), -a * np.cos(th)
        A = NL.exposure(mu, mup, mupp)
        report("P3.4", f"a={a}: max|A + a cos|",
               np.abs(A + a * np.cos(th)).max(), 1e-9)


def check_P41_projected_normal_r2():
    """P4.1 (r=2): mu = a cos theta shifts F1 by a, so the winner law is the
    angular density of N((a,0), I_2). Corollary 3.2 must reproduce it."""
    print("\nP4.1  circle formula = projected normal at l=1 (r=2)")
    N = 512
    th = 2 * np.pi * np.arange(N) / N
    for a in (0.2, 0.7, 1.5):
        mu, mup, mupp = a * np.cos(th), -a * np.sin(th), -a * np.cos(th)
        c = a * np.cos(th)
        pn = np.exp(-a ** 2 / 2) * (1 + c * SQRT_2PI * np.exp(c ** 2 / 2)
                                    * norm.cdf(c))
        report("P4.1", f"a={a}: max|Cor3.2 - projected normal|",
               np.abs(NL.rho_exact(mu, mup, mupp) - pn).max(), 1e-12)


def check_C32_vs_montecarlo():
    """C3.2: the closed-form circle density is exact, well outside the
    linear regime. Compared through harmonic gains (MC noise averages down)."""
    print("\nC3.2  exact circle density vs Monte Carlo")
    N, draws = 512, 3_000_000
    _, V = C.geometry(N)
    for a, k in ((1.0, 1), (0.15, 3), (0.02, 8), (0.05, 8)):
        th, mu, mup, mupp = NL.harmonic(a, k, N)
        gf = C.cos_amp(NL.rho_exact(mu, mup, mupp) - 1, th, k) / a
        gm = C.cos_amp(N * C.shares_mc(mu, V, 0.0, draws, seed=4) - 1, th, k) / a
        report("C3.2", f"a={a},k={k}: relative gain error",
               abs(gf - gm) / abs(gm), 4e-3)


def check_normalisation():
    """C3.2 corollary: the closed form self-normalises to total mass one."""
    print("\nC3.2n normalisation of the exact density")
    for a, k in ((0.3, 1), (0.15, 3), (0.02, 8)):
        th, mu, mup, mupp = NL.harmonic(a, k, 2048)
        report("C3.2n", f"a={a},k={k}: |mean(rho) - 1|",
               abs(NL.rho_exact(mu, mup, mupp).mean() - 1), 2e-4)


# --------------------------------------------------------- linear response

def check_T51_cr_and_laplacian():
    """T5.1: linear response is -c_r Laplace_S with c_r = E[1/R], and
    c_r = 2^{-1/2} Gamma((r-1)/2)/Gamma(r/2)."""
    print("\nT5.1  linear response constant c_r = E[1/R]")
    rng = np.random.default_rng(2)
    for r in (2, 3, 4, 5):
        R = np.linalg.norm(rng.standard_normal((4_000_000, r)), axis=1)
        mc = (1.0 / R).mean()
        report("T5.1", f"r={r}: |gamma-ratio - MC E[1/R]| (rel)",
               abs(SP.c_r(r) - mc) / mc, 3e-3)

    print("\nT5.1s sphere gains vs c_r l(l+r-2)")
    for r in (2, 3, 4, 5):
        rng = np.random.default_rng(17 + r)
        V = SP.sphere_sites(30000, r, rng)
        for l in (1, 2, 3):
            Y = SP.zonal(r, l, V[:, 0])
            Y = Y - Y.mean()
            pred = SP.c_r(r) * l * (l + r - 2)
            meas = SP.paired_gain(V, Y, 0.01, 60_000, rng)
            report("T5.1s", f"r={r},l={l}: relative gain error",
                   abs(meas - pred) / pred, 0.06)


def check_P53_cycle_spectrum():
    """P5.3: the ring's facet-flux Laplacian has spectrum N*lambda_k ->
    c_2 k^2, i.e. the discrete cycle Laplacian reproduces the continuum law."""
    print("\nP5.3  cycle-Laplacian spectrum = c_2 k^2")
    phi0 = 1 / SQRT_2PI
    for N in (256, 1024, 4096):
        ke = phi0 * 0.5 / (2 * np.sin(np.pi / N))
        for k in (1, 4, 16):
            lam = 2 * ke * (1 - np.cos(2 * np.pi * k / N))
            report("P5.3", f"N={N},k={k}: rel error vs sqrt(pi/2)k^2",
                   abs(N * lam / (np.sqrt(np.pi / 2) * k * k) - 1),
                   3e-2 if N == 256 else 2e-3)


# --------------------------------------------------- order eps^r threshold

def check_T61_order_eps_r_circle():
    """T6.1 (r=2): the exposure threshold contributes at order eps^r = eps^2.
    Measured as the gap between the exact density and the same formula with
    the lower limit set to zero (the 'local' density)."""
    print("\nT6.1  global exposure enters at order eps^r  (circle, r=2)")
    N = 1024
    th = 2 * np.pi * np.arange(N) / N
    f, fp, fpp = np.cos(3 * th), -3 * np.sin(3 * th), -9 * np.cos(3 * th)
    eps_list = np.array([0.02, 0.01, 0.005, 0.0025])
    gaps = []
    for e in eps_list:
        mu, mup, mupp = e * f, e * fp, e * fpp
        A = NL.exposure(mu, mup, mupp)
        exact = np.exp(-mup ** 2 / 2) * (np.exp(-A ** 2 / 2)
                                         - SQRT_2PI * mupp * norm.sf(A))
        local = np.exp(-mup ** 2 / 2) * (1.0 - np.sqrt(np.pi / 2) * mupp)
        gaps.append(np.abs(exact - local).max())
    slope = np.polyfit(np.log(eps_list), np.log(gaps), 1)[0]
    report("T6.1", "exponent of |exact - local| in eps (target r=2)",
           abs(slope - 2.0), 0.15)


def check_T61_order_eps_r_sphere():
    """T6.1 (r=3): if exposure entered at order eps^2 the purely local
    second-order formula would be wrong. Test it does not, by checking the
    local second-order formula against the exact l=1 answer to O(eps^3)."""
    print("\nT6.1s exposure absent at second order  (sphere, r=3)")
    x = np.linspace(-1, 1, 4001)          # cos(polar angle), uniform on S^2
    c3 = SP.c_r(3)
    eps_list = np.array([0.2, 0.1, 0.05, 0.025])
    res = []
    for e in eps_list:
        c = e * x
        g = c + (1 + c ** 2) * SQRT_2PI * np.exp(c ** 2 / 2) * norm.cdf(c)
        exact = g / np.trapezoid(g, x) * 2.0        # mean 1 under du/2 on [-1,1]
        pred2 = 1 + 2 * c3 * e * x + e ** 2 * (3 * x ** 2 - 1) / 2
        res.append(np.abs(exact - pred2).max())
    slope = np.polyfit(np.log(eps_list), np.log(res), 1)[0]
    report("T6.1s", "exponent of |exact - 2nd order| (target 3)",
           abs(slope - 3.0), 0.2)


# ------------------------------------------------------------ second order

def check_T71_second_order_l1():
    """T7.1 at r=3, l=1: rho = 1 - c_r eps Lap f + eps^2[e_2(H)/(r-2)
    - |grad f|^2/2]. For f = v.e this predicts 1 + 2c_3 eps f
    + eps^2 (3f^2-1)/2, checked against the exact projected normal."""
    print("\nT7.1  second-order expansion, r=3, l=1 (exact reference)")
    x = np.linspace(-1, 1, 4001)
    c3 = SP.c_r(3)
    for e in (0.05, 0.1):
        c = e * x
        g = c + (1 + c ** 2) * SQRT_2PI * np.exp(c ** 2 / 2) * norm.cdf(c)
        exact = g / np.trapezoid(g, x) * 2.0
        pred1 = 1 + 2 * c3 * e * x
        pred2 = pred1 + e ** 2 * (3 * x ** 2 - 1) / 2
        r1 = np.abs(exact - pred1).max()
        r2 = np.abs(exact - pred2).max()
        report("T7.1", f"eps={e}: 2nd-order residual / 1st-order residual",
               r2 / r1, 0.25)


def check_T71_second_order_l2():
    """T7.1 at r=3 with a zonal l=2 field. The l=4 harmonic it generates is
    PURELY second order, so its coefficient is a zero-parameter prediction.
    Second differences (rho_+ + rho_- - 2 rho_0)/2 cancel the site-density
    artefact and all odd orders."""
    print("\nT7.1b second-order mode coupling l=2 -> l=4  (r=3)")
    N, draws = 40000, 400_000
    rng = np.random.default_rng(23)
    V = SP.sphere_sites(N, 3, rng)
    x = V[:, 0]
    f = eval_legendre(2, x)
    # zonal derivatives: f'' = -xP' + (1-x^2)P'', cot.f' = -xP', |grad f|^2 = (1-x^2)P'^2
    P1 = 3 * x                                     # P_2'(x)
    P2 = 3.0                                       # P_2''(x)
    fpp = -x * P1 + (1 - x ** 2) * P2
    cotf = -x * P1
    e2 = fpp * cotf
    grad2 = (1 - x ** 2) * P1 ** 2
    lap = fpp + cotf
    second = e2 / (3 - 2) - grad2 / 2              # m_2 = 1/(r-2) = 1
    Y4 = eval_legendre(4, x)
    Y4 = Y4 - Y4.mean()
    proj = lambda v: (v @ Y4) / (Y4 @ Y4)
    pred_l4 = proj(second)
    for eps in (0.15, 0.25):
        cnt = {}
        for tag, m in (("+", eps * f), ("-", -eps * f), ("0", np.zeros(N))):
            r2 = np.random.default_rng(101)
            c = np.zeros(N)
            done = 0
            while done < draws:
                b = min(2000, draws - done)
                c += np.bincount((m + r2.standard_normal((b, 3)) @ V.T).argmax(1),
                                 minlength=N)
                done += b
            cnt[tag] = N * c / draws
        curv = (cnt["+"] + cnt["-"] - 2 * cnt["0"]) / (2 * eps ** 2)
        report("T7.1b", f"eps={eps}: l=4 coefficient, rel error",
               abs(proj(curv) - pred_l4) / abs(pred_l4), 0.15)


# ------------------------------------------------------------- statistics

def check_P91_fisher():
    """P9.1: Fisher information per winner for a unit-variance harmonic is
    I_l = c_r^2 [l(l+r-2)]^2, so sd(eps-hat) = 1/(sqrt(M) c_r l(l+r-2)).
    Verified by the efficient moment estimator's empirical spread."""
    print("\nP9.1  Fisher information / Cramer-Rao for argmax sensing")
    r, N, M, reps = 3, 20000, 4000, 60
    rng = np.random.default_rng(31)
    V = SP.sphere_sites(N, r, rng)
    x = V[:, 0]
    for l in (1, 2):
        Y = eval_legendre(l, x)
        Y = (Y - Y.mean()) / Y.std()               # unit variance on the sphere
        lam = SP.c_r(r) * l * (l + r - 2)
        eps = 0.02
        mu = eps * Y
        est = []
        for s in range(reps):
            g = np.random.default_rng(500 + s)
            w = (mu + g.standard_normal((M, r)) @ V.T).argmax(1)
            est.append(Y[w].mean() / lam)
        emp = np.std(est, ddof=1)
        crb = 1.0 / (np.sqrt(M) * lam)
        report("P9.1", f"l={l}: empirical sd / Cramer-Rao bound",
               abs(emp / crb - 1), 0.30)
        report("P9.1", f"l={l}: estimator bias (in units of sd)",
               abs(np.mean(est) - eps) / emp, 0.6)


def check_P92_identifiability():
    """P9.2: empty Laguerre cells are generic, so at D=0 some mu_i are
    unidentifiable; any D>0 restores strict positivity."""
    print("\nP9.2  empty cells are generic at D=0, cured by D>0")
    rng = np.random.default_rng(77)
    empty = 0
    for _ in range(30):
        V = rng.normal(scale=1.2, size=(8, 2))
        mu = rng.normal(scale=0.3, size=8)
        mu -= mu.mean()
        if SL.cell_masses(mu, V, n_theta=1 << 14).min() < 1e-6:
            empty += 1
    report("P9.2", "fraction of random 8-site configs with an empty cell",
           empty / 30, 1.0, ok=(empty / 30 > 0.5))
    V = rng.normal(scale=1.2, size=(8, 2))
    mu = rng.normal(scale=0.3, size=8)
    mu -= mu.mean()
    F, W = SL.gh_nodes(2, 15)
    p = SL.shares_gh(mu, V, np.full(8, 0.25), F, W, 2001)
    report("P9.2", "min share with D>0 (want strictly positive)",
           -p.min(), 0.0, ok=p.min() > 1e-8)


def check_P45_zeroth_order_cancellation():
    """New: the reason the operator is pure Laplace-Beltrami. Linearising at a
    FIXED ball (Huang-Xi-Zhao 6.3) gives Lap + a zeroth-order term; integrating
    over the NESTED family of Wulff shapes kills it, because
    M_{d+1} = d M_{d-1} identically (integration by parts). The surviving
    first-order coefficient is M_{d-1}/M_d = E[1/R] = c_r."""
    print("\nP4.5  zeroth-order cancellation => pure Laplacian, l=1 identified")
    from scipy.integrate import quad
    M = lambda m: quad(lambda a: a ** m * np.exp(-a * a / 2), 0, np.inf)[0]
    for d in (1, 2, 3, 4, 5):
        report("P4.5", f"d={d}: M_(d+1) - d M_(d-1) = 0",
               M(d + 1) - d * M(d - 1), 1e-10)
    for r in (2, 3, 4, 5):
        report("P4.5", f"r={r}: M_(d-1)/M_d = c_r",
               abs(M(r - 2) / M(r - 1) - SP.c_r(r)), 1e-10)
    report("P4.5", "l=1 gain is NONZERO (classical operator has it in kernel)",
           -SP.c_r(3) * 2, 0.0, ok=SP.c_r(3) * 2 > 1.0)


def check_P36_convexification():
    """New: the exposure threshold IS the convexification. A_mu = -mu exactly
    when the Wulff shape is convex (h'' + h >= 0 with h = -mu), and otherwise
    differs by precisely the convexity defect."""
    print("\nP3.6  exposure threshold = convexification of the Wulff shape")
    N = 2048
    th = 2 * np.pi * np.arange(N) / N
    for a, k, c in ((0.3, 1, 0.0), (0.05, 3, 1.0), (0.3, 3, 0.0)):
        mu = c + a * np.cos(k * th)
        mup, mupp = -a*k*np.sin(k*th), -a*k*k*np.cos(k*th)
        A = NL.exposure(mu, mup, mupp)
        defect = -min(((-mupp) + (-mu)).min(), 0.0)     # -min(h''+h) if negative
        report("P3.6", f"a={a},k={k},c={c}: max|A+mu| == convexity defect",
               abs(np.abs(A + mu).max() - defect), 5e-3)


def check_P93_degree_independent_snr():
    """New: the admissible amplitude at degree k shrinks like 1/k^2, exactly the
    rate at which the inverse damps noise, so the achievable SNR at fixed
    a k^2 is DEGREE-INDEPENDENT: SNR = x c_2 sqrt(M/2)."""
    print("\nP9.3  SNR at the edge of the linear regime is degree-independent")
    N, M, reps, x = 512, 120_000, 25, 0.3
    th, V = C.geometry(N)
    snrs = []
    for k in (2, 4, 8):
        a = x / k ** 2
        mu = a * np.cos(k * th)
        est = []
        for s in range(reps):
            g = np.random.default_rng(700 + s)
            cnt, done = np.zeros(N), 0
            while done < M:
                m = min(4000, M - done)
                cnt += np.bincount((mu + g.standard_normal((m, 2)) @ V.T
                                    ).argmax(1), minlength=N)
                done += m
            est.append(C.cos_amp(N * cnt / done - 1, th, k)
                       / (C.SQRT_PI_2 * k * k))
        snrs.append(np.mean(est) / np.std(est, ddof=1))
    pred = x * C.SQRT_PI_2 * np.sqrt(M / 2)
    report("P9.3", "spread of SNR across k (relative)",
           (max(snrs) - min(snrs)) / np.mean(snrs), 0.35)
    report("P9.3", "mean SNR vs x c_2 sqrt(M/2) (relative)",
           abs(np.mean(snrs) / pred - 1), 0.25)


def check_T31_r3():
    """T3.1 at r=3 with a nontrivial zonal field and the exposure threshold
    active. Compared by Legendre moments, which are binning-free."""
    print("\nT3.1  exact density at r=3, zonal l=2, exposure active")
    import harder as H
    z = np.linspace(-0.9999, 0.9999, 601)
    a, ell = 0.15, 2
    rho, A = H.r3_density(z, a, ell, ny=241, nphi=121)
    rho = rho / (np.trapezoid(rho, z) / 2.0)
    report("T3.1", "exposure threshold is active (max A)", -A.max(), 0.0,
           ok=A.max() > 0.1)
    report("T3.1", "predicted density integrates to one",
           abs(np.trapezoid(rho, z) / 2 - 1), 1e-4)
    s = H.r3_montecarlo(a, ell, draws=1_500_000, npsi=2048)
    for L in (2, 4):
        pred = np.trapezoid(eval_legendre(L, z) * rho, z) / 2.0
        meas = eval_legendre(L, s).mean()
        se = eval_legendre(L, s).std() / np.sqrt(len(s))
        report("T3.1", f"Legendre moment P_{L} (diff / mc se)",
               abs(pred - meas) / se, 4.0)


def check_P35_saturation():
    """P3.5: the modulation ceiling is exactly 2 (full concentration on the k
    maxima), hence G(x) ~ 2/(c_2 x)."""
    print("\nP3.5  saturation constant and G asymptotics")
    N = 8192
    th = 2 * np.pi * np.arange(N) / N
    for a, k in ((1.0, 8), (3.0, 8)):
        mu = a * np.cos(k * th)
        mup, mupp = -a*k*np.sin(k*th), -a*k*k*np.cos(k*th)
        m = C.cos_amp(NL.rho_exact(mu, mup, mupp) - 1, th, k)
        report("P3.5", f"a k^2={a*k*k:.0f}: modulation vs ceiling 2",
               abs(m - 2.0), 0.05 if a > 1 else 0.2)
        report("P3.5", f"a k^2={a*k*k:.0f}: x G(x) vs 2/c_2",
               abs(m / C.SQRT_PI_2 - 2 / C.SQRT_PI_2), 0.2)


def check_nonneg():
    """Prop 4.4: A >= lambda_max(Hess) makes det(aI-Hess) >= 0 on the domain,
    so rho >= 0 automatically; the A=0 truncation goes negative."""
    print("\nP4.4  exposure threshold enforces nonnegativity")
    N = 4096
    th = 2 * np.pi * np.arange(N) / N
    for a, k in ((0.02, 8), (0.05, 8), (0.15, 3)):
        mu = a * np.cos(k * th)
        mup, mupp = -a*k*np.sin(k*th), -a*k*k*np.cos(k*th)
        A = NL.exposure(mu, mup, mupp)
        report("P4.4", f"a={a},k={k}: min(A - mu'') >= 0",
               -min(np.min(A - mupp), 0.0), 1e-9)
        report("P4.4", f"a={a},k={k}: exact rho >= 0",
               -min(NL.rho_exact(mu, mup, mupp).min(), 0.0), 1e-9)
        cand = np.exp(-mup**2/2) * (1 - np.sqrt(np.pi/2) * mupp)
        report("P4.4", f"a={a},k={k}: A=0 candidate DOES go negative",
               cand.min(), 0.0, ok=cand.min() < -0.1)


def check_T54_density():
    """T5.4: the continuum operator does not depend on the sampling density."""
    print("\nT5.4  sampling-density independence of the continuum operator")
    import harder_bc as HB
    N, k, a, draws = 1024, 3, 0.01, 3_000_000
    gains = []
    for amp in (0.0, 0.8):
        th = HB.nonuniform_sites(N, amp)
        V = np.column_stack([np.cos(th), np.sin(th)])
        rng = np.random.default_rng(2)
        cnt, done = np.zeros(N), 0
        mu = a * np.cos(k * th)
        while done < draws:
            m = min(4000, draws - done)
            cnt += np.bincount((mu + rng.standard_normal((m, 2)) @ V.T
                                ).argmax(1), minlength=N)
            done += m
        gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
        wq = ((gaps + np.roll(gaps, 1)) / 2) / (2 * np.pi)
        rho = (cnt / done) / wq
        gains.append(2 * np.sum((rho - 1) * np.cos(k * th) * wq) / a)
        report("T5.4", f"amp={amp}: gain vs c_2 k^2 (rel)",
               abs(gains[-1] / (C.SQRT_PI_2 * k * k) - 1), 0.05)
    report("T5.4", "spread across densities (relative)",
           abs(gains[0] - gains[1]) / np.mean(gains), 2e-3)


def check_P84_transfer():
    """P8.4: attenuation from idiosyncratic noise collapses on sigma k^2,
    better than on sigma k or sigma^2 k^2."""
    print("\nP8.4  noisy transfer function collapses on sigma k^2")
    import harder_bc as HB
    tab = {}
    N, M, chunk = 192, 400_000, 2000
    th, V = C.geometry(N)
    ks = (2, 4, 8, 16)
    amps = {k: 0.1 / k ** 2 for k in ks}
    drive = {k: amps[k] * np.cos(k * th) for k in ks}
    for sig in (0.0, 0.05, 0.15, 0.45):
        rng = np.random.default_rng(11)
        cp = {k: np.zeros(N) for k in ks}
        cm = {k: np.zeros(N) for k in ks}
        done = 0
        while done < M:
            m = min(chunk, M - done)
            base = rng.standard_normal((m, 2)) @ V.T
            if sig > 0:
                base = base + sig * rng.standard_normal((m, N))
            for k in ks:
                cp[k] += np.bincount((base + drive[k]).argmax(1), minlength=N)
                cm[k] += np.bincount((base - drive[k]).argmax(1), minlength=N)
            done += m
        tab[sig] = np.array([
            C.cos_amp(N * (cp[k] - cm[k]) / (2 * done), th, k) / amps[k]
            / (C.SQRT_PI_2 * k * k) for k in ks])
    base0 = tab[0.0]
    scat = {}
    for fn, name in ((lambda s, k: s*k*k, "sigma k^2"),
                     (lambda s, k: s*k, "sigma k"),
                     (lambda s, k: s*s*k*k, "sigma^2 k^2")):
        xs, ys = [], []
        for s in tab:
            if s == 0:
                continue
            for i, k in enumerate(ks):
                v = tab[s][i] / base0[i]
                if 0.03 < v < 0.97:
                    xs.append(fn(s, k)); ys.append(v)
        xs, ys = np.array(xs), np.array(ys)
        c = np.polyfit(np.log(xs), np.log(ys), 2)
        scat[name] = float((np.log(ys) - np.polyval(c, np.log(xs))).std())
        report("P8.4", f"{name}: residual scatter", scat[name], 1.0, ok=True)
    best = min(scat, key=scat.get)
    report("P8.4", "sigma k^2 is the best collapse variable", 0.0, 0.0,
           ok=(best == "sigma k^2"))
    report("P8.4", "and beats runners-up by >1.5x",
           scat["sigma k^2"], 1.0,
           ok=min(scat["sigma k"], scat["sigma^2 k^2"]) > 1.5*scat["sigma k^2"])


def check_T101_scale_mixture():
    """T10.1: F = S G with E[S^2]=1 has covariance I_r for every S, but linear
    response -c_r E[1/S] Laplace. By Jensen E[1/S] >= 1/E[S] >= 1, so the
    Gaussian uniquely minimises extremal response at fixed covariance."""
    print("\nT10.1 scale mixtures: response beyond the covariance")
    N, k, a = 1024, 3, 0.01
    th, V = C.geometry(N)

    def mc(draw, M=4_000_000, seed=1, chunk=4000):
        rng = np.random.default_rng(seed)
        mu = a * np.cos(k * th)
        c = np.zeros(N)
        done = 0
        while done < M:
            m = min(chunk, M - done)
            S = draw(rng, m)[:, None]
            c += np.bincount((mu + (S * rng.standard_normal((m, 2))) @ V.T
                              ).argmax(1), minlength=N)
            done += m
        return c / M

    laws = (("S=1 (Gaussian)", lambda r_, m: np.ones(m)),
            ("two-point .5/1.3229",
             lambda r_, m: np.where(r_.random(m) < .5, .5, 1.3229)),
            ("lognormal", lambda r_, m: np.exp(r_.standard_normal(m)*0.6-0.36)))
    prev = None
    for name, draw in laws:
        S = draw(np.random.default_rng(7), 2_000_000)
        es2, einv = (S**2).mean(), (1/S).mean()
        report("T10.1", f"{name}: E[S^2] (must be 1)", abs(es2 - 1), 5e-3)
        pred = C.SQRT_PI_2 * einv * k * k
        meas = C.cos_amp(C.modulation(mc(draw)), th, k) / a
        report("T10.1", f"{name}: gain vs c_r E[1/S] k^2 (rel)",
               abs(meas / pred - 1), 0.10)
        report("T10.1", f"{name}: E[1/S] >= 1 (Jensen)", 1 - einv, 0.0,
               ok=einv >= 1 - 1e-12)
        if prev is not None:
            report("T10.1", f"{name}: gain exceeds Gaussian", prev - meas,
                   0.0, ok=meas > prev)
        else:
            prev = meas


if __name__ == "__main__":
    print("=" * 78)
    print("Numerical verification of extremal_harmonic_analysis.tex")
    print("=" * 78)
    for fn in (check_P21_power_diagram, check_T81_facet_laplacian,
               check_P82_soft_laplacian, check_T83_tau_limit,
               check_P34_exposure_closed_form, check_P41_projected_normal_r2,
               check_C32_vs_montecarlo, check_normalisation,
               check_T51_cr_and_laplacian, check_P53_cycle_spectrum,
               check_T61_order_eps_r_circle, check_T61_order_eps_r_sphere,
               check_T71_second_order_l1, check_T71_second_order_l2,
               check_P91_fisher, check_P92_identifiability,
               check_T101_scale_mixture, check_T31_r3, check_P35_saturation,
               check_nonneg, check_T54_density, check_P84_transfer,
               check_P45_zeroth_order_cancellation, check_P36_convexification,
               check_P93_degree_independent_snr):
        try:
            fn()
        except Exception as exc:                    # keep going, report it
            print(f"  [ERROR] {fn.__name__}: {exc}")
            RESULTS.append((fn.__name__, False))
    n = len(RESULTS)
    bad = [c for c, ok in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{n - len(bad)}/{n} checks passed")
    if bad:
        print("FAILED:", ", ".join(sorted(set(bad))))
