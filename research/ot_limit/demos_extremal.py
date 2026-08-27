"""Demonstration figures for extremal_harmonic_analysis.tex.

Produces figures/demo_*.png. Each panel illustrates one numbered claim.

Run: python demos_extremal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import circle_spectral as C
import nonlinear_circle as NL
import soft_laguerre as SL
import sphere_spectral as SP

FIG = Path(__file__).resolve().parents[2] / "experiments" / "figures"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
SQRT_2PI = np.sqrt(2 * np.pi)


def demo_cells():
    """P2.1 + P8.2: hard power cells, and their softening by D > 0."""
    rng = np.random.default_rng(8)
    N = 9
    V = rng.normal(size=(N, 2)) * 1.1
    mu = rng.normal(scale=0.4, size=N)
    mu -= mu.mean()
    g = np.linspace(-3.2, 3.2, 420)
    X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    fig, ax = plt.subplots(1, 4, figsize=(15.5, 4.1))
    lab = SL.labels_race(X, mu, V).reshape(len(g), len(g))
    ax[0].imshow(lab.T, origin="lower", extent=[-3.2, 3.2] * 2, cmap="tab10",
                 interpolation="nearest")
    ax[0].set_title("hard race, $\\arg\\max_i\\ \\mu_i+v_i^\\top F$")
    lab2 = SL.labels_laguerre(X, SL.weights_from_mu(mu, V), V).reshape(
        len(g), len(g))
    ax[1].imshow(lab2.T, origin="lower", extent=[-3.2, 3.2] * 2, cmap="tab10",
                 interpolation="nearest")
    ax[1].set_title(f"power diagram (identical: "
                    f"{(lab == lab2).mean()*100:.0f}% match)")
    for k, sig in enumerate((0.25, 0.8)):
        pi = SL.conditional_pi(X, mu, V, np.full(N, sig**2))
        H = -(pi * np.log(np.maximum(pi, 1e-300))).sum(1).reshape(len(g), len(g))
        im = ax[2 + k].imshow(H.T, origin="lower", extent=[-3.2, 3.2] * 2,
                              cmap="magma")
        ax[2 + k].set_title(f"assignment entropy, $\\sigma={sig}$")
        fig.colorbar(im, ax=ax[2 + k], fraction=0.046)
    for a in ax:
        a.scatter(V[:, 0], V[:, 1], c="w", s=22, edgecolors="k", zorder=3)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIG / "demo_cells.png", dpi=140)
    print("  wrote demo_cells.png")


def demo_circle():
    """C3.2 + T6.1: the exact density, and where the exposure term enters."""
    N = 1024
    th, V = C.geometry(N)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))

    for a, k, c in ((0.02, 3, "C0"), (0.15, 3, "C1"), (0.05, 8, "C2")):
        _, mu, mup, mupp = NL.harmonic(a, k, N)
        ax[0].plot(th, NL.rho_exact(mu, mup, mupp), c, lw=2,
                   label=f"exact  a={a}, k={k}")
        ax[0].plot(th, N * C.shares_mc(mu, V, 0.0, 1_500_000, seed=4), c + "--",
                   lw=0.8, alpha=.7)
        ax[0].plot(th, NL.rho_linear(mupp), c, ls=":", lw=1.2, alpha=.8)
    ax[0].set(xlabel=r"$\theta$", ylabel=r"$\rho(\theta)$",
              title="exact (solid) vs MC (dashed) vs linear (dotted)")
    ax[0].legend(fontsize=8)

    xs, Gs = [], []
    for a in (0.005, 0.02, 0.08):
        for k in (2, 4, 8, 16):
            _, mu, mup, mupp = NL.harmonic(a, k, N)
            g = C.cos_amp(NL.rho_exact(mu, mup, mupp) - 1, th, k) / a
            xs.append(a * k * k)
            Gs.append(g / (C.SQRT_PI_2 * k * k))
    ax[1].semilogx(xs, Gs, "o")
    ax[1].set(xlabel=r"$a k^2 = \|\mu''\|$", ylabel=r"$G$",
              title="gain $/\\,c_2k^2$ collapses on $\\|\\mu''\\|$")
    ax[1].grid(alpha=.3)

    eps = np.array([0.02, 0.01, 0.005, 0.0025])
    f, fp, fpp = np.cos(3*th), -3*np.sin(3*th), -9*np.cos(3*th)
    gaps = []
    for e in eps:
        mu, mup, mupp = e*f, e*fp, e*fpp
        A = NL.exposure(mu, mup, mupp)
        ex = np.exp(-mup**2/2)*(np.exp(-A**2/2) - SQRT_2PI*mupp*norm.sf(A))
        gaps.append(np.abs(ex - np.exp(-mup**2/2)*(1 - np.sqrt(np.pi/2)*mupp)).max())
    s = np.polyfit(np.log(eps), np.log(gaps), 1)[0]
    ax[2].loglog(eps, gaps, "o-", label=f"fitted slope {s:.2f}")
    ax[2].loglog(eps, gaps[0]*(eps/eps[0])**2, "k--", label=r"$\varepsilon^{r}=\varepsilon^2$")
    ax[2].set(xlabel=r"$\varepsilon$", ylabel=r"$|\rho-\rho^{\rm loc}|_\infty$",
              title="exposure term enters at order $\\varepsilon^r$")
    ax[2].legend(); ax[2].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "demo_circle.png", dpi=140)
    print("  wrote demo_circle.png")


def demo_spectrum():
    """T5.1 + P9.1: the Laplace-Beltrami gain law, and reverse conditioning."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.1))
    for r in (2, 3, 4, 5):
        rng = np.random.default_rng(17 + r)
        V = SP.sphere_sites(30000, r, rng)
        ls = (1, 2, 3) if r == 5 else (1, 2, 3, 4)
        meas = []
        for l in ls:
            Y = SP.zonal(r, l, V[:, 0])
            meas.append(SP.paired_gain(V, Y - Y.mean(), 0.01, 60_000, rng))
        pred = [SP.c_r(r) * l * (l + r - 2) for l in ls]
        ax[0].plot(pred, meas, "o", label=f"r={r}")
    lim = [0, 21]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set(xlabel=r"predicted $c_r\,\ell(\ell+r-2)$", ylabel="measured gain",
              title="Laplace--Beltrami gain law", xlim=lim, ylim=lim)
    ax[0].legend(); ax[0].grid(alpha=.3)

    ell = np.arange(1, 13)
    for r, c in ((2, "C0"), (3, "C1"), (5, "C2")):
        ax[1].semilogy(ell, SP.c_r(r) * ell * (ell + r - 2), c + "o-",
                       label=f"forward gain, r={r}")
        ax[1].semilogy(ell, 1 / (SP.c_r(r) * ell * (ell + r - 2)), c + "s--",
                       label=f"inverse noise gain, r={r}")
    ax[1].set(xlabel=r"harmonic degree $\ell$", ylabel="gain",
              title="forward amplifies, inverse damps")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, which="both")

    laws = (("Gaussian", lambda g, m: np.ones(m)),
            ("two-point", lambda g, m: np.where(g.random(m) < .5, .5, 1.3229)),
            ("lognormal", lambda g, m: np.exp(g.standard_normal(m)*.6 - .36)))
    N, k, a = 1024, 3, 0.01
    th, V = C.geometry(N)
    names, meas, pred = [], [], []
    for name, draw in laws:
        S = draw(np.random.default_rng(7), 1_000_000)
        rng = np.random.default_rng(1)
        cnt, done = np.zeros(N), 0
        mu = a * np.cos(k * th)
        while done < 3_000_000:
            m = min(4000, 3_000_000 - done)
            s = draw(rng, m)[:, None]
            cnt += np.bincount((mu + (s*rng.standard_normal((m, 2))) @ V.T
                                ).argmax(1), minlength=N)
            done += m
        names.append(name)
        meas.append(C.cos_amp(C.modulation(cnt/done), th, k) / a)
        pred.append(C.SQRT_PI_2 * (1/S).mean() * k * k)
    xx = np.arange(3)
    ax[2].bar(xx - .18, pred, .36, label=r"$c_r\mathbb{E}[S^{-1}]k^2$")
    ax[2].bar(xx + .18, meas, .36, label="measured")
    ax[2].set_xticks(xx, names)
    ax[2].set(ylabel="gain", title="equal covariance, different response")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "demo_spectrum.png", dpi=140)
    print("  wrote demo_spectrum.png")


def demo_recovery():
    """P9.1: recover harmonic coefficients from winner locations alone."""
    N, M = 2048, 2_000_000
    th, V = C.geometry(N)
    truth = {1: 0.030, 2: 0.020, 5: 0.008}
    mu = sum(a * np.cos(k * th) for k, a in truth.items())
    p = C.shares_mc(mu, V, 0.0, M, seed=3)
    m = C.modulation(p)
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ks = np.arange(1, 9)
    est = [C.cos_amp(m, th, k) / (C.SQRT_PI_2 * k * k) for k in ks]
    ax.bar(ks - .18, [truth.get(k, 0.0) for k in ks], .36, label="true $a_k$")
    ax.bar(ks + .18, est, .36, label=r"recovered $\hat p_k/(c_2k^2)$")
    ax.set(xlabel="harmonic $k$", ylabel="coefficient",
           title=f"recovery from {M:,} winner locations only")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "demo_recovery.png", dpi=140)
    for k in ks[:6]:
        print(f"     k={k}: true {truth.get(k,0.0):+.4f}  "
              f"recovered {est[k-1]:+.4f}")
    print("  wrote demo_recovery.png")


if __name__ == "__main__":
    print("demonstrations ->", FIG)
    demo_cells()
    demo_circle()
    demo_spectrum()
    demo_recovery()
