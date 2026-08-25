"""Does a real Bayesian-optimization GP posterior have low argmax rank?

This is the make-or-break numeric for the Entropy Search application. BO's
default kernel is Matern-5/2, whose Mercer eigenvalues decay only algebraically,
roughly k^-(5+d)/d: fast in one or two dimensions, slow by ten. If the
"flat spectrum buys nothing" failure mode reappears at moderate d, then the
factor probit does not help Entropy Search on the problems people actually run,
and that has to be known before anything is built on it.

Two quantities per dimension, and the second is the one that decides:

  covariance error   off-diagonal Frobenius residual of the rank-r fit
  argmax fidelity    total variation and top-k agreement of the resulting
                     probability of maximality against a dense Monte Carlo
                     reference on the same posterior

The lesson from the molecular case was that these two come apart -- a residual
worth 35% of the covariance was worth 1.2% of the decision -- so reporting the
first alone would answer the wrong question.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from factorize import eig_factor, top_eigen  # noqa: E402
from metrics import batch_agreement, qpo_efficiency, tv_error  # noqa: E402
from pom import pom_fast, pom_full_mc, pom_independent, sobol_nodes  # noqa: E402


def matern52(X1, X2, lengthscale):
    """Matern 5/2 with a single (isotropic) lengthscale."""
    d2 = np.maximum(((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1), 0.0)
    r = np.sqrt(d2) / lengthscale
    s5 = np.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 / 3.0 * r ** 2) * np.exp(-s5 * r)


def rbf(X1, X2, lengthscale):
    d2 = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


KERNELS = {"matern52": matern52, "rbf": rbf}


def posterior(kern, n_rep, d, n_train, lengthscale, noise, seed):
    """GP posterior over n_rep representer points after n_train observations."""
    rng = np.random.default_rng(seed)
    Xtr = rng.random((n_train, d))
    Xs = rng.random((n_rep, d))
    k = KERNELS[kern]
    # a draw from the prior as the "true" function, so the mean is realistic
    Kall = k(np.vstack([Xtr, Xs]), np.vstack([Xtr, Xs]), lengthscale)
    Kall += 1e-8 * np.eye(len(Kall))
    L = np.linalg.cholesky(Kall)
    f = L @ rng.standard_normal(len(Kall))
    y = f[:n_train] + np.sqrt(noise) * rng.standard_normal(n_train)

    Ktt = k(Xtr, Xtr, lengthscale) + noise * np.eye(n_train)
    Kst = k(Xs, Xtr, lengthscale)
    Kss = k(Xs, Xs, lengthscale)
    A = np.linalg.solve(Ktt, Kst.T)
    mu = Kst @ np.linalg.solve(Ktt, y)
    Sigma = Kss - Kst @ A + noise * np.eye(n_rep)
    return mu, 0.5 * (Sigma + Sigma.T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="matern52", choices=list(KERNELS))
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 20])
    ap.add_argument("--n-rep", type=int, default=1000)
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--lengthscale", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--ref-samples", type=int, default=4_000_000)
    ap.add_argument("--sobol-m", type=int, default=9)
    ap.add_argument("--points", type=int, default=129)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    rows = []
    for d in args.dims:
        for seed in args.seeds:
            mu, Sigma = posterior(args.kernel, args.n_rep, d, args.n_train,
                                  args.lengthscale, args.noise, seed)
            n = len(mu)
            off = ~np.eye(n, dtype=bool)
            fro = float(np.sqrt(np.sum(Sigma[off] ** 2)))
            w = np.linalg.eigvalsh(Sigma)[::-1]
            w = np.maximum(w, 0)
            frac = np.cumsum(w) / w.sum()

            t0 = time.time()
            p_ref = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=999,
                                chunk=max(1, int(4e7 // n)))
            p_ref2 = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=998,
                                 chunk=max(1, int(4e7 // n)))
            ceiling = batch_agreement(p_ref, p_ref2, (args.batch,), mu=mu)
            ref_tv = tv_error(p_ref, p_ref2)
            eig = top_eigen(Sigma, max(args.ranks) if max(args.ranks) else 1)

            for r in args.ranks:
                V, dd = eig_factor(Sigma, r, eig=eig if r else None)
                resid = Sigma - (V @ V.T + np.diag(dd))
                if r == 0:
                    p = pom_independent(mu, dd, points=args.points)
                    nodes = 0
                else:
                    F, W = sobol_nodes(r, m=args.sobol_m, seed=0)
                    p = pom_fast(mu, V, dd, F, W, points=args.points)
                    nodes = len(F)
                ba = batch_agreement(p_ref, p, (args.batch,), mu=mu)
                rows.append({
                    "kernel": args.kernel, "d": d, "seed": seed, "rank": r,
                    "n": n, "n_train": args.n_train,
                    "lengthscale": args.lengthscale,
                    "cov_offdiag_rel": float(np.sqrt(np.sum(resid[off] ** 2))) / fro,
                    "trace_frac_r": float(frac[max(r - 1, 0)]) if r else 0.0,
                    "tv_vs_ref": tv_error(p_ref, p),
                    "ref_self_tv": ref_tv,
                    f"top{args.batch}_recall": ba[f"top{args.batch}_recall"],
                    f"ref_top{args.batch}_recall": ceiling[f"top{args.batch}_recall"],
                    "qpo_efficiency": qpo_efficiency(p_ref, p, args.batch, mu),
                    "max_p": float(p_ref.max()), "sobol_nodes": nodes,
                })
            g = pd.DataFrame(rows)
            g = g[(g.d == d) & (g.seed == seed)]
            print(f"d={d:3d} seed={seed}  ref self-TV {ref_tv:.4f}  "
                  f"({time.time() - t0:.0f}s)  " +
                  "  ".join(f"r{int(t.rank)}:TV{t.tv_vs_ref:.3f}/rec{getattr(t, f'top{args.batch}_recall'):.2f}"
                            for t in g.itertuples()), flush=True)

    df = pd.DataFrame(rows)
    dest = HERE / "results" / f"matern_{args.kernel}.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    piv = df.pivot_table(index="d", columns="rank", values="tv_vs_ref", aggfunc="median")
    print("\nmedian total variation against the reference, by dimension and rank:")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nreference self-TV by dimension (the floor):")
    print(df.groupby("d").ref_self_tv.median().to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
