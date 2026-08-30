"""Sample splitting for the zero-win tail at n = 2000.

The paper reports that at n = 2000 the lattice agrees with a 1e6-path
simulation at its noise floor on the resolved bulk, and then says the
honest thing about the rest: 82 percent of alternatives have zero
simulation wins, that set is chosen by the same simulation that would
have to certify it, and no fixed-set binomial bound applies to it. It
names sample splitting as the test that would apply. This runs it.

    half A  ->  defines the zero-win set Z (no wins in A)
    half B  ->  independent draws; count how many land anywhere in Z

Because Z is fixed before B is looked at, B's count is binomial in the
true mass of Z, and a Clopper-Pearson interval is valid. That interval
is what the model's predicted mass on Z has to sit inside. The bulk
comparison is also reported on half B alone, so no statistic in this
script is evaluated on the draws that selected its own target.

The referee simulates the FITTED factor race (mu + V f + sqrt(D) eps),
which is the object the lattice claims to price. The dense matrix enters
through the fit, whose own residual is reported separately in the
ensemble study and is not what this script is testing.

    python run_split_referee.py --n 2000 --draws 2000000
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import beta

from winning.factor.core import fit_covariance
from winning.factor.races import race_probabilities


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.isf(alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def simulate(mu, V, D, draws, seed, chunk=20000):
    """Win counts for the factor race, argmin wins, in chunks."""
    n, k = V.shape
    rng = np.random.default_rng(seed)
    sd = np.sqrt(D)
    counts = np.zeros(n, dtype=np.int64)
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        f = rng.standard_normal((m, k))
        X = mu[None, :] + f @ V.T + rng.standard_normal((m, n)) * sd[None, :]
        idx = np.argmin(X, axis=1)
        counts += np.bincount(idx, minlength=n)
        done += m
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=2_000_000,
                    help="TOTAL draws; split into two independent halves")
    ap.add_argument("--spread", type=float, default=1.2,
                    help="ability spread; the zero-win fraction is driven "
                         "by this, not by the covariance. 1.2 reproduces "
                         "the paper's wide-field regime where most of the "
                         "field is hopeless")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    n = args.n
    rng = np.random.default_rng(7)
    # a dense correlation with real structure: three global factors, a
    # block layer, and idiosyncratic noise -- the shape the grammar is
    # meant to serve, so the fit residual does not dominate the test
    B = rng.standard_normal((n, 3)) * 0.5
    lab = rng.integers(0, 40, size=n)
    Bl = np.zeros((n, 40))
    Bl[np.arange(n), lab] = 0.45
    S = B @ B.T + Bl @ Bl.T + np.diag(0.5 + rng.random(n))
    s = np.sqrt(np.diag(S))
    C = S / np.outer(s, s)
    mu = np.sort(rng.standard_normal(n)) * args.spread
    mu -= mu.mean()

    t0 = time.time()
    V, D, F, W = fit_covariance(C)
    t_fit = time.time() - t0
    t0 = time.time()
    p = race_probabilities(mu, V=V, D=D, F=F, W=W)
    t_price = time.time() - t0
    print(f"n={n}  fit {t_fit:.2f}s  price {t_price:.2f}s  "
          f"rank {V.shape[1]}  sum(p)={p.sum():.9f}", flush=True)

    half = args.draws // 2
    t0 = time.time()
    a = simulate(mu, V, D, half, seed=11)
    b = simulate(mu, V, D, half, seed=12)
    print(f"simulation {args.draws:,d} draws in two halves: "
          f"{time.time()-t0:.1f}s", flush=True)

    # --- the selected set, chosen on A alone -------------------------
    Z = a == 0
    print(f"\nzero-win set from half A: {Z.sum()}/{n} alternatives "
          f"({100*Z.sum()/n:.1f}%)")
    model_mass = float(p[Z].sum())
    kB = int(b[Z].sum())
    lo, hi = clopper_pearson(kB, half)
    inside = lo <= model_mass <= hi
    print(f"  model mass on Z            {model_mass:.3e}")
    print(f"  half B wins in Z           {kB} of {half:,d}")
    print(f"  95% Clopper-Pearson for Z  [{lo:.3e}, {hi:.3e}]")
    print(f"  model inside the interval  {'YES' if inside else 'NO'}")

    # --- the bulk, scored on B alone ---------------------------------
    resolved = b >= 25
    fB = b / half
    tv_all = 0.5 * float(np.abs(p - fB).sum())
    per_entry = float(np.abs(p[resolved] - fB[resolved]).max())
    med_entry = float(np.median(np.abs(p[resolved] - fB[resolved])))
    print(f"\nscored on half B only ({resolved.sum()} alternatives with "
          f">=25 wins):")
    print(f"  full-vector TV (zeros included)  {tv_all:.3e}")
    print(f"  max  |p - freq| on resolved      {per_entry:.3e}")
    print(f"  median                           {med_entry:.3e}")
    # the noise floor B alone can achieve, for comparison
    noise = 0.5 * float(np.abs(a / half - fB).sum())
    print(f"  half A vs half B TV (noise floor) {noise:.3e}")

    if args.out:
        json.dump({"n": n, "draws": args.draws, "rank": int(V.shape[1]),
                   "zero_win_set": int(Z.sum()),
                   "model_mass_on_Z": model_mass, "half_B_wins_in_Z": kB,
                   "ci": [lo, hi], "model_inside_ci": bool(inside),
                   "tv_all": tv_all, "max_resolved": per_entry,
                   "median_resolved": med_entry, "noise_floor_tv": noise},
                  open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
