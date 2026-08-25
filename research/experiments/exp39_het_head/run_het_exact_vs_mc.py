"""Experiment 39: the heteroskedastic classifier head, exact versus MC-softmax.

Collier et al. (CVPR 2021) and HET-XL (ICLR 2023) place a factor Gaussian
race on the classifier output layer: logits u = mu + V f + sqrt(D) eps,
prediction p_c = P(c = argmax), approximated by S Monte Carlo samples of a
temperature softmax, mean_s softmax(u_s / tau). Their tau explicitly trades
bias (tau > 0 smooths the argmax) against variance (tau -> 0 explodes the
MC noise at fixed S).

By the Gumbel-argmax identity, E[softmax(u/tau)] is itself a hard race with
tau-Gumbel-convolved idiosyncratic noise, which the shared field computes
exactly (winning.factor.races, temperature argument). So the exact map
evaluates their estimator's own expectation at every tau, including tau = 0.
This script measures what the two approximations cost:

  (A) MC noise: sd across resamples of the S-sample estimator, per class,
      against the exact value, at their production S and tau.
  (B) tau bias: exact tempered probabilities against exact tau = 0 argmax
      probabilities -- the bias their tau introduces even at S = infinity.
  (C) anchors: exact tau = 0 map against 20M-draw common-random-number MC.
  (D) wall time for the exact map at K = 1000.

Head geometry mimics a trained ImageNet-scale head: K classes, low-rank
factor covariance (rank 6 by pruned Gauss-Hermite; HET's larger ranks need
the Sobol rule and are timed separately), heteroskedastic D, logit margins
drawn so top-1 confidence spans the realistic 0.3-0.95 range.

Run:  python experiments/exp39_het_head/run_het_exact_vs_mc.py
Output: results.csv, printed summary.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from winning.factor.core import hermite_nodes, qmc_nodes  # noqa: E402
from winning.factor.races import race_probabilities  # noqa: E402

HERE = Path(__file__).resolve().parent
K = 1000
RANK = 3   # pruned product Gauss-Hermite regime; HET-scale rank 15 is timed via Sobol below
S_PROD = 50          # HET production sample count
TAUS = [1.0, 0.5, 0.25, 0.1]
N_RESAMPLE = 200     # resamples of the S-sample estimator for noise sd
N_ANCHOR = 20_000_000
SEED = 39


def simulate_head(rng, k=K, rank=RANK):
    """Logits and covariance shaped like a trained heteroskedastic head."""
    mu = rng.normal(0.0, 2.0, k)
    lead = rng.integers(0, k)
    mu[lead] += rng.uniform(1.0, 4.0)          # a realistic top-1 margin
    V = rng.normal(0.0, 0.4 / np.sqrt(rank), (k, rank))
    hot = rng.choice(k, k // 20, replace=False)
    V[hot] *= 3.0                              # a few strongly loaded classes
    D = rng.uniform(0.3, 1.0, k)
    D[hot] *= rng.uniform(1.5, 4.0, len(hot))  # heteroskedastic tail
    return mu, V, D


def mc_softmax(mu, V, D, tau, S, rng):
    """The HET estimator: mean over S samples of softmax(u/tau) (max-wins)."""
    f = rng.standard_normal((S, V.shape[1]))
    eps = rng.standard_normal((S, len(mu)))
    u = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * eps
    z = u / tau
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).mean(axis=0)


def mc_argmax_anchor(mu, V, D, n_draws, rng, chunk=20_000):
    counts = np.zeros(len(mu))
    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        f = rng.standard_normal((m, V.shape[1]))
        eps = rng.standard_normal((m, len(mu)))
        u = mu[None, :] + f @ V.T + np.sqrt(D)[None, :] * eps
        counts += np.bincount(u.argmax(axis=1), minlength=len(mu))
        done += m
    return counts / n_draws


def exact_race(mu, V, D, tau, F, W):
    """Exact max-wins probabilities via the min-wins engine at -mu."""
    return race_probabilities(-np.asarray(mu, float), V=V, D=D, F=F, W=W,
                              base="normal", temperature=tau, points=2001)


def main():
    rng = np.random.default_rng(SEED)
    mu, V, D = simulate_head(rng)
    F, W = hermite_nodes(RANK)
    print(f"K={K}, rank={RANK}, {len(W)} pruned factor nodes")

    rows = ["quantity,tau,value"]

    # (C) anchor the tau=0 exact map against big MC
    t0 = time.perf_counter()
    p_exact0 = exact_race(mu, V, D, 0.0, F, W)
    t_exact = time.perf_counter() - t0
    p_exact0 = p_exact0 / p_exact0.sum()
    t0 = time.perf_counter()
    p_mc = mc_argmax_anchor(mu, V, D, N_ANCHOR, np.random.default_rng(1))
    t_anchor = time.perf_counter() - t0
    big = p_mc > 200 / N_ANCHOR
    sd_c = np.sqrt(p_mc[big] * (1 - p_mc[big]) / N_ANCHOR)
    z = np.abs(p_exact0[big] - p_mc[big]) / sd_c
    print(f"(C) anchor vs {N_ANCHOR/1e6:.0f}M-draw MC over {big.sum()} classes:"
          f" max z {z.max():.2f}, mean z {z.mean():.2f}, z>4 count {(z>4).sum()}"
          f"; exact pass {t_exact:.2f}s, anchor {t_anchor:.0f}s")
    rows += [f"anchor_max_z,0,{z.max():.4f}",
             f"anchor_mean_z,0,{z.mean():.4f}",
             f"anchor_z_gt4,0,{(z>4).sum()}",
             f"exact_pass_seconds,0,{t_exact:.3f}"]

    # (A) + (B): per-tau exact value, MC noise around it, and tau bias
    print(f"\n{'tau':>6} {'MC-mean check':>14} {'sd/p top':>11} {'sd/p mid':>11} "
          f"{'all-zero':>10} {'bias top':>12} {'bias mid':>12}")
    print("   (sd/p over 200 resamples of S=50; all-zero = classes never "
          "receiving any sample mass; bias = |dlog| tempered vs argmax)")
    for tau in TAUS:
        p_tau = exact_race(mu, V, D, tau, F, W)
        p_tau = p_tau / p_tau.sum()
        ests = np.stack([mc_softmax(mu, V, D, tau, S_PROD,
                                    np.random.default_rng(1000 + r))
                         for r in range(N_RESAMPLE)])
        bias_check = np.abs(ests.mean(0) - p_tau).max()
        rel_sd = ests.std(0) / np.maximum(p_tau, 1e-300)
        zero_frac = float((ests == 0).all(axis=0).mean())
        top = p_tau > 1e-2
        mid = (p_tau > 1e-4) & ~top
        sd_top = np.median(rel_sd[top]) if top.any() else np.nan
        sd_mid = np.median(rel_sd[mid]) if mid.any() else np.nan
        dlog = np.abs(np.log(np.maximum(p_tau, 1e-300)) -
                      np.log(np.maximum(p_exact0, 1e-300)))
        bias_top = np.median(dlog[p_exact0 > 1e-2]) if (p_exact0 > 1e-2).any() else np.nan
        bias_mid = np.median(dlog[(p_exact0 > 1e-4) & (p_exact0 <= 1e-2)])
        print(f"{tau:6.2f} {bias_check:14.2e} {sd_top:11.3f} {sd_mid:11.3f} "
              f"{zero_frac:10.2%} {bias_top:12.3f} {bias_mid:12.3f}")
        rows += [f"mc_estimator_bias_check,{tau},{bias_check:.6e}",
                 f"mc_rel_sd_median_p_gt_1e2,{tau},{sd_top:.6f}",
                 f"mc_rel_sd_median_mid,{tau},{sd_mid:.6f}",
                 f"mc_all_zero_class_fraction,{tau},{zero_frac:.6f}",
                 f"tau_bias_median_dlogp_top,{tau},{bias_top:.6f}",
                 f"tau_bias_median_dlogp_mid,{tau},{bias_mid:.6f}"]

    # (D) timing at HET-XL-like rank via Sobol
    Fq, Wq = qmc_nodes(15, m=11)
    t0 = time.perf_counter()
    _ = exact_race(mu, rng.normal(0.0, 0.15, (K, 15)), D, 0.0, Fq, Wq)
    t15 = time.perf_counter() - t0
    print(f"\n(D) exact pass at rank 15 (Sobol 2^11): {t15:.1f}s (NumPy)")
    rows.append(f"exact_pass_seconds_rank15,0,{t15:.3f}")

    (HERE / "results.csv").write_text("\n".join(rows) + "\n")
    print(f"\nwrote {HERE / 'results.csv'}")


if __name__ == "__main__":
    main()
