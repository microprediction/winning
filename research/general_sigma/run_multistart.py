"""Multistart dispersion for the dense-covariance fit.

Section 6 of the paper says the identified objective is nonconvex in
(V, D), claims no global certificate, and offers multistart dispersion
as the cheap diagnostic. It then reports no dispersion, which invites
exactly the question it was meant to pre-empt. This measures it.

For each ensemble: fit from the default start and from several random
diagonal starts, and report

  * the spread of the identified objective ||P(C - VV' - diag D)P||_F^2
    across starts, relative to the best one found;
  * how often a random start beats the default, which is the question
    that matters -- dispersion is only a problem if the shipped start is
    the bad one;
  * the choice-level consequence, as total variation between the race
    priced at each start's fit and at the best fit. An objective gap
    nobody can see in probabilities is not a defect worth reporting.

    python run_multistart.py --starts 8 --n 300
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from winning.factor.core import factor_model_projected, _projected_sq
from winning.factor.races import race_probabilities


def ensembles(n, rng):
    """A few covariance shapes with different conditioning, including the
    one the ensemble study found hardest (block equicorrelation, where
    the raw eigenfit spends its rank on a near-common component)."""
    out = {}

    B = rng.standard_normal((n, 3))
    out["factor rank-3"] = B @ B.T + np.diag(0.5 + rng.random(n))

    k, size = 6, n // 6
    C = np.eye(n)
    lab = np.repeat(np.arange(k), size)
    lab = np.concatenate([lab, np.full(n - len(lab), k - 1)])
    for c in range(k):
        idx = np.where(lab == c)[0]
        C[np.ix_(idx, idx)] = 0.7
    np.fill_diagonal(C, 1.0)
    out["block equicorrelation"] = C

    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    out["exponential decay"] = 0.9 ** d + 1e-8 * np.eye(n)

    A = rng.standard_normal((n, n)) / np.sqrt(n)
    S = A @ A.T + np.eye(n) * 0.4
    s = np.sqrt(np.diag(S))
    out["wishart-ish"] = S / np.outer(s, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    mu = rng.standard_normal(args.n) * 0.5
    mu -= mu.mean()
    results = {}

    print(f"n = {args.n}, rank k = {args.k}, {args.starts} random starts "
          f"plus the default\n")
    header = (f"{'ensemble':24s} {'default obj':>12s} {'best obj':>11s} "
              f"{'worst/best':>11s} {'beat default':>13s} {'max TV':>9s}")
    print(header)
    print("-" * len(header))

    for name, C in ensembles(args.n, rng).items():
        scale = float(np.mean(np.diag(C)))
        objs, fits = [], []
        V0, D0_ = factor_model_projected(C, args.k)
        obj_default = _projected_sq(C, V0, D0_)
        objs.append(obj_default)
        fits.append((V0, D0_))
        r2 = np.random.default_rng(11)
        for s in range(args.starts):
            # starts spanning two decades of assumed idiosyncratic share
            frac = 10.0 ** r2.uniform(-1.5, 0.0)
            D0 = np.full(len(C), frac * scale) * (0.5 + r2.random(len(C)))
            V, D = factor_model_projected(C, args.k, D0=D0)
            objs.append(_projected_sq(C, V, D))
            fits.append((V, D))
        objs = np.array(objs)
        best = int(np.argmin(objs))
        beat = int((objs[1:] < objs[0] * (1 - 1e-9)).sum())

        # choice-level consequence against the best fit found
        p_best = race_probabilities(mu, V=fits[best][0], D=fits[best][1])
        tvs = []
        for V, D in fits:
            p = race_probabilities(mu, V=V, D=D)
            tvs.append(0.5 * float(np.abs(p - p_best).sum()))
        print(f"{name:24s} {objs[0]:12.6g} {objs[best]:11.6g} "
              f"{objs.max()/max(objs[best],1e-300):11.4f} "
              f"{beat:5d}/{args.starts:<7d} {max(tvs):9.2e}")
        results[name] = {"obj_default": float(objs[0]),
                         "obj_best": float(objs[best]),
                         "obj_all": [float(o) for o in objs],
                         "best_is_default": best == 0,
                         "n_beating_default": beat,
                         "max_tv_vs_best": float(max(tvs))}

    print("\nReading: worst/best near 1.0 means the starts agree. "
          "'beat default' counts random starts strictly better than the "
          "shipped start. max TV is the largest choice-level disagreement "
          "between any start's fit and the best one.")

    # The block row is the interesting one and the reason this script
    # reports TV at all. Every start reaches the same objective to 4e-16
    # AND the same D to every digit, yet the races disagree by TV 0.25.
    # Cause: centering leaves the six-block matrix with a FIVE-fold
    # degenerate eigenvalue, so a rank-3 fit picks an arbitrary
    # three-dimensional subspace of a five-dimensional one. Every choice
    # is equally optimal and they imply different races. Objective
    # dispersion cannot see this; choice dispersion can. The fix is rank,
    # not optimization, which the sweep below demonstrates.
    C = ensembles(args.n, np.random.default_rng(0))["block equicorrelation"]
    P = np.eye(len(C)) - np.ones((len(C), len(C))) / len(C)
    w = np.linalg.eigvalsh(P @ C @ P)[::-1]
    mult = int((w > 0.5 * w[0]).sum())
    print(f"\nRank sweep on the degenerate ensemble (centered spectrum has "
          f"{mult} tied leading eigenvalues at {w[0]:.1f}):")
    print(f"{'rank':>6s} {'objective':>14s} {'max TV across starts':>22s}")
    for k in (3, mult - 1, mult, mult + 2):
        r3 = np.random.default_rng(11)
        fs = [factor_model_projected(C, k)]
        for _ in range(4):
            frac = 10.0 ** r3.uniform(-1.5, 0.0)
            fs.append(factor_model_projected(
                C, k, D0=np.full(len(C), frac) * (0.5 + r3.random(len(C)))))
        pp = [race_probabilities(mu, V=V, D=D) for V, D in fs]
        tv = max(0.5 * float(np.abs(q - pp[0]).sum()) for q in pp)
        print(f"{k:6d} {_projected_sq(C, *fs[0]):14.6g} {tv:22.2e}")
    print("\nAt the multiplicity the fit is exact and the ambiguity is "
          "gone. Below it, equal objective does not mean equal race: "
          "compare races across starts, not objectives.")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"n": args.n, "k": args.k, "starts": args.starts,
                       "results": results}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
