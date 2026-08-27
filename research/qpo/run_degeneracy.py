"""What happens as the idiosyncratic variance goes to zero.

A pure linear-bandit posterior over arm means is X C X', which is rank d
exactly and has NO idiosyncratic part. That is not this method's form -- it is
the degenerate boundary of it. The whole construction rests on conditional
independence given the factors, with D supplying the conditional spread that
the lattice integrates. With D = 0 the conditional argmax given the factor is a
deterministic indicator, p_i becomes the probability of a polyhedral cell in
factor space, and the lattice has nothing left to do.

This measures how fast that happens, because the answer decides whether linear
bandits are a headline application or a footnote. Sigma = V V' + eps * D0 with
eps swept down, everything else fixed, scored against dense Monte Carlo on the
same Sigma.

Reported per eps:
  tv            total variation against the reference
  rel_tail      median relative error on the smallest decile of p
  n_effective   how many candidates still carry appreciable probability
The headline property being tested is that relative error does not depend on
p_i; if it survives to small eps, the boundary is benign.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from metrics import tv_error  # noqa: E402
from pom import pom_fast, pom_full_mc, sobol_nodes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 1e-3, 1e-4])
    ap.add_argument("--sobol-ms", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--points", type=int, default=257)
    ap.add_argument("--ref-samples", type=int, default=8_000_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, r = args.n, args.rank
    V = rng.standard_normal((n, r)) / np.sqrt(r)
    D0 = rng.uniform(0.5, 1.5, n)
    mu = rng.standard_normal(n) * 0.3
    mu -= mu.mean()

    rows = []
    for eps in args.eps:
        d = eps * D0
        Sigma = V @ V.T + np.diag(d)
        p_ref, se = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=7,
                                chunk=max(1, int(4e7 // n)), return_se=True)
        p_ref2 = pom_full_mc(mu, Sigma, M=args.ref_samples, seed=8,
                             chunk=max(1, int(4e7 // n)))
        floor = tv_error(p_ref, p_ref2)
        nz = p_ref > 0
        n_eff = int((p_ref > 1e-4).sum())
        for m in args.sobol_ms:
            F, W = sobol_nodes(r, m=m, seed=0)
            p = pom_fast(mu, V, d, F, W, points=args.points)
            # relative error where the reference can resolve it
            ok = p_ref > 10 * se
            rel = np.abs(p[ok] - p_ref[ok]) / p_ref[ok]
            lo = p_ref[ok] <= np.quantile(p_ref[ok], 0.25)
            hi = p_ref[ok] >= np.quantile(p_ref[ok], 0.75)
            rows.append({
                "eps": eps, "nodes": len(F), "rank": r, "n": n,
                "tv": tv_error(p_ref, p), "ref_self_tv": floor,
                "rel_median": float(np.median(rel)),
                "rel_bottom_quartile": float(np.median(rel[lo])),
                "rel_top_quartile": float(np.median(rel[hi])),
                "n_effective": n_eff, "max_p": float(p_ref.max()),
            })
            print(f"  eps={eps:8.1e} Q={len(F):5d}  TV={rows[-1]['tv']:.4f} "
                  f"(floor {floor:.4f})  rel err: bottom-q "
                  f"{rows[-1]['rel_bottom_quartile']:.3f} top-q "
                  f"{rows[-1]['rel_top_quartile']:.3f}  "
                  f"n_eff={n_eff}", flush=True)

    df = pd.DataFrame(rows)
    dest = HERE / "results" / "degeneracy.csv"
    df.to_csv(dest, index=False)
    print(f"\nwrote {dest}")
    print("\nTV against the reference, by eps and node count:")
    print(df.pivot_table(index="eps", columns="nodes", values="tv").to_string(
        float_format=lambda x: f"{x:.4f}"))
    print("\nratio of bottom-quartile to top-quartile relative error "
          "(1.0 = error independent of p, the headline property):")
    d2 = df.copy()
    d2["ratio"] = d2.rel_bottom_quartile / np.maximum(d2.rel_top_quartile, 1e-12)
    print(d2.pivot_table(index="eps", columns="nodes", values="ratio").to_string(
        float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
