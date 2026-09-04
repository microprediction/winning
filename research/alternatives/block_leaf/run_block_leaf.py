"""Kill test B: dense block leaves under the shared block cavity.

Blocks are independent; covariance WITHIN a block is arbitrary dense
-- outside the private-factor block grammar. The hybrid of
ALTERNATIVES.md: a Monte Carlo leaf per block (dense Cholesky is
cheap at leaf size m), and the exact cross-block cavity

    p_i = E[ 1{i = argmax of block c} * prod_{d != c} H_d(M_c) ],

with H_d the block-max CDF and M_c the block's max. Each block's
samples price all its members at once, and the product over other
blocks is one shared field: the total log-CDF T(x) = sum_d log H_d(x)
is accumulated on a single global grid (each block contributes one
searchsorted pass), then every sample reads T at its block max and
subtracts its own block's term -- O(blocks) work, not O(blocks^2).

Compared against, at matched wall-clock where sensible:
  global_mc   plain Monte Carlo on the full field (the honest
              unstructured baseline);
  grammar     the engine after fitting the dense block-diagonal
              covariance to its grammar (the fit residual is the
              price of density, measured here);
  truth       a 2e7-draw Monte Carlo, block-exploiting, separate seed.

Measured: total variation to truth, wall-clock, and scaling of the
hybrid across block counts at fixed leaf size.
"""
import json
import os
import sys
import time

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                ".."))
from winning.factor import race_probabilities   # noqa: E402

L_GRID = 2001


def make_blocks(n, m, rng):
    """Means and dense within-block covariances (Wishart-ish)."""
    covs, mus, chols = [], [], []
    for _ in range(n // m):
        A = rng.normal(size=(m, m + 2)) / np.sqrt(m + 2)
        C = A @ A.T + 0.05 * np.eye(m)
        covs.append(C)
        mus.append(rng.normal(0, 0.7, m))
        chols.append(np.linalg.cholesky(C))
    return mus, covs, chols


def hybrid(mus, chols, R, rng):
    """Block-leaf MC + shared cavity; returns p (n,) and seconds.

    Fully batched: one Gaussian tensor and one einsum sample every
    leaf at once, sorts and searchsorted run per block on contiguous
    rows, and the member accumulation is a single bincount over
    (block, argmax) pairs weighted by the cross-block cavity."""
    t0 = time.time()
    nb = len(mus)
    m = len(mus[0])
    MU = np.stack(mus)                       # (nb, m)
    LC = np.stack(chols)                     # (nb, m, m)
    Z = rng.normal(size=(nb, R, m))
    X = MU[:, None, :] + np.einsum("brk,bmk->brm", Z, LC)
    maxes = X.max(2)                         # (nb, R)
    amaxes = X.argmax(2)                     # (nb, R)
    lo = float(maxes.min()) - 1e-9
    hi = float(maxes.max()) + 1e-9
    grid = np.linspace(lo, hi, L_GRID)
    sorted_m = np.sort(maxes, axis=1)        # (nb, R)
    counts = np.empty((nb, L_GRID))
    own = np.empty((nb, R))
    for c in range(nb):                      # searchsorted per block
        counts[c] = np.searchsorted(sorted_m[c], grid, side="right")
        own[c] = np.searchsorted(sorted_m[c], maxes[c], side="right")
    with np.errstate(divide="ignore"):
        T = np.log(counts / R).sum(0)        # (L_GRID,)
        own = np.log(own / R)
    rest = np.interp(maxes.ravel(), grid, T).reshape(nb, R) - own
    wgt = np.exp(np.clip(rest, -745.0, 0.0))
    flat_idx = (np.arange(nb)[:, None] * m + amaxes).ravel()
    p = np.bincount(flat_idx, weights=wgt.ravel(),
                    minlength=nb * m) / R
    return p, time.time() - t0


def global_mc(mus, chols, draws, rng):
    t0 = time.time()
    nb, m = len(mus), len(mus[0])
    counts = np.zeros(nb * m)
    block = max(1, min(draws, 20_000_000 // (nb * m)))
    done = 0
    while done < draws:
        b = min(block, draws - done)
        vals = np.empty((b, nb))
        args = np.empty((b, nb), dtype=int)
        for c, (mu, Lc) in enumerate(zip(mus, chols)):
            Xb = mu + rng.normal(size=(b, m)) @ Lc.T
            vals[:, c] = Xb.max(1)
            args[:, c] = Xb.argmax(1)
        wc = vals.argmax(1)
        wi = args[np.arange(b), wc] + wc * m
        counts += np.bincount(wi, minlength=nb * m)
        done += b
    return counts / draws, time.time() - t0


def grammar_fit(mus, covs):
    t0 = time.time()
    n = len(mus) * len(mus[0])
    m = len(mus[0])
    Sigma = np.zeros((n, n))
    for c, C in enumerate(covs):
        Sigma[c * m:(c + 1) * m, c * m:(c + 1) * m] = C
    mu = np.concatenate(mus)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = race_probabilities(-mu, cov=Sigma)
    return p, time.time() - t0


if __name__ == "__main__":
    results = {}
    rng = np.random.default_rng(4)

    for n, m in ((64, 8), (512, 8), (512, 16), (4096, 8)):
        mus, covs, chols = make_blocks(n, m, rng)
        p_truth, t_truth = global_mc(mus, chols, 20_000_000,
                                     np.random.default_rng(99))
        p_h, t_h = hybrid(mus, chols, R=8192,
                          rng=np.random.default_rng(7))
        # matched-time global MC: draws chosen to spend ~t_h
        rate_probe, t_probe = global_mc(mus, chols, 200_000,
                                        np.random.default_rng(13))
        draws_matched = max(10_000, int(200_000 * t_h / max(t_probe,
                                                            1e-9)))
        p_g, t_g = global_mc(mus, chols, draws_matched,
                             np.random.default_rng(17))
        row = dict(
            hybrid=dict(tv=float(0.5 * np.abs(p_h - p_truth).sum()),
                        seconds=t_h),
            global_mc=dict(tv=float(0.5 * np.abs(p_g - p_truth).sum()),
                           seconds=t_g, draws=draws_matched),
            truth_seconds=t_truth)
        if n <= 512:
            p_f, t_f = grammar_fit(mus, covs)
            row["grammar_fit"] = dict(
                tv=float(0.5 * np.abs(p_f - p_truth).sum()),
                seconds=t_f)
        results[f"n{n}_m{m}"] = row
        msg = (f"[n={n} m={m}] hybrid tv {row['hybrid']['tv']:.4f} "
               f"({t_h:.2f}s) | global MC tv "
               f"{row['global_mc']['tv']:.4f} ({t_g:.2f}s, "
               f"{draws_matched} draws)")
        if "grammar_fit" in row:
            msg += (f" | grammar tv {row['grammar_fit']['tv']:.4f} "
                    f"({row['grammar_fit']['seconds']:.2f}s)")
        print(msg)

    with open(os.path.join(os.path.dirname(__file__), "results.json"),
              "w") as fj:
        json.dump(results, fj, indent=2)
    print("wrote results.json")
