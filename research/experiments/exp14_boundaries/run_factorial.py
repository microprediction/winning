"""Experiment 14, Part B replacement: the 2x2 factorial substitution study
(third referee round).

The earlier Part B could not attribute its increments: the "factor mixed
logit" candidate received both factor loadings AND alternative-specific
Gumbel scales, so the move from plain logit combined two changes (and with
alternative-specific scales the model is a heteroskedastic factor-Gumbel
race, not standard mixed logit). Here the design isolates the two axes:

  Truth: U = mu* + V* f + e, t(5) factors (unit variance), skew-normal(a=3)
  idiosyncratic noise standardized to mean 0 variance 1, HOMOSKEDASTIC
  (common unit scale). Misspecified for every candidate. Note the sign
  convention: the race is min-wins, so a=+3 skew in race coordinates is
  LEFT-skew in utility coordinates.

  Candidates (all common unit idiosyncratic scale, all calibrated to the
  same full-menu shares, Gaussian factors with the true loadings supplied):

                      Gumbel base          Gaussian base
      V = 0           independent Luce     independent probit
      V = V*          factor mixed logit   factor probit

  Independent Luce = plain multinomial logit = IIA renormalization, exactly.

  Deletion truths: utilities are drawn ONCE (2*10^7 draws); the top-3
  alternatives per draw are retained, from which the full-menu shares and
  every single- and pair-deletion truth follow with common random numbers --
  all 50 single deletions and 100 random pairs, tightly coupled, populating
  the mass strata by design rather than by 24 random blocks.

Reported: misallocated fraction of redistributed mass (TV / deleted mass) per
model per deletion; the factor increment within each family, the family
increment at each factor setting, and their interaction; individual
observations plotted (no bars).

Run:  python experiments/exp14_boundaries/run_factorial.py
Outputs: results_factorial.csv, figures/factorial.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_boundaries import calibrate_base, factor_shares_base  # noqa: E402
from raceutil import hermite_nodes  # noqa: E402

HERE = Path(__file__).resolve().parent
SEED = 33
N, K = 50, 2
A_SKEW = 3.0
N_DRAWS = 20_000_000


def draw_top3(mu, V, rng, n_draws):
    """Draw min-wins utilities from the truth once; return (m, 3) int16 of the
    three smallest alternatives per draw, in order."""
    top3 = np.empty((n_draws, 3), dtype=np.int16)
    done = 0
    delta = A_SKEW / np.sqrt(1 + A_SKEW**2)
    while done < n_draws:
        m = min(400_000, n_draws - done)
        f = rng.standard_t(5, size=(m, K)) / np.sqrt(5.0 / 3.0)
        d0 = np.abs(rng.standard_normal((m, N)))
        d1 = rng.standard_normal((m, N))
        e = delta * d0 + np.sqrt(1 - delta**2) * d1
        e = (e - delta * np.sqrt(2 / np.pi)) / np.sqrt(1 - 2 * delta**2 / np.pi)
        U = mu[None, :] + f @ V.T + e
        part = np.argpartition(U, 2, axis=1)[:, :3]
        order = np.argsort(np.take_along_axis(U, part, axis=1), axis=1)
        top3[done:done + m] = np.take_along_axis(part, order, axis=1)
        done += m
    return top3


def deletion_shares(top3, deleted):
    """Post-deletion share vector from common rankings. `deleted` is a set of
    at most two alternatives; the winner is the first-ranked survivor."""
    w = top3[:, 0].copy()
    for lvl in (1, 2):
        mask = np.isin(w, deleted)
        if not mask.any():
            break
        w[mask] = top3[mask, lvl]
    counts = np.bincount(w, minlength=N).astype(float)
    for d in deleted:
        counts[d] = 0.0
    return counts / counts.sum()


def main():
    rng = np.random.default_rng(SEED)
    rows = ["quantity,value"]
    mu_true = rng.normal(0.0, 1.0, N)
    V_true = rng.normal(0.0, 0.6 / np.sqrt(K), (N, K))

    t0 = time.perf_counter()
    top3 = draw_top3(mu_true, V_true, rng, N_DRAWS)
    p_menu = np.bincount(top3[:, 0], minlength=N) / len(top3)
    print(f"truth: {N_DRAWS/1e6:.0f}M common draws in {time.perf_counter()-t0:.0f}s; "
          f"menu share range [{p_menu.min():.4f}, {p_menu.max():.4f}]")

    F2, W2 = hermite_nodes(2)
    D_unit = np.ones(N)
    Vz = np.zeros((N, K))
    models = {}
    t0 = time.perf_counter()
    for name, base, V in [("independent Luce", "gumbel", Vz),
                          ("independent probit", "normal", Vz),
                          ("factor mixed logit", "gumbel", V_true),
                          ("factor probit", "normal", V_true)]:
        mu_c = calibrate_base(p_menu, V, D_unit, F2, W2, base=base)
        r, _ = factor_shares_base(mu_c, V, D_unit, F2, W2, base=base)
        resid = float(np.abs(r - p_menu).max())
        models[name] = (base, V, mu_c)
        rows.append(f"calib_residual_{name.replace(' ', '_')},{resid:.3e}")
        print(f"  calibrated {name}: menu residual {resid:.1e}")
    print(f"calibrations in {time.perf_counter()-t0:.0f}s")

    # every single deletion + 100 random pairs, truths from the common draws
    blocks = [(i,) for i in range(N)]
    prng = np.random.default_rng(4)
    seen = set()
    while len(seen) < 100:
        i, j = sorted(prng.choice(N, 2, replace=False))
        seen.add((int(i), int(j)))
    blocks += sorted(seen)

    per_obs = []          # (model, block_size, deleted_mass, tv_over_mass)
    t0 = time.perf_counter()
    skipped = 0
    for B in blocks:
        mass = float(p_menu[list(B)].sum())
        if mass < 5e-4:       # below identification/noise floor: uninformative
            skipped += 1
            continue
        q_true = deletion_shares(top3, list(B))
        keep = np.setdiff1d(np.arange(N), B)
        for name, (base, V, mu_c) in models.items():
            q, _ = factor_shares_base(mu_c, V, D_unit, F2, W2, base=base, keep=keep)
            full = np.zeros(N); full[keep] = q
            tv = 0.5 * float(np.abs(full - q_true).sum())
            per_obs.append((name, len(B), mass, tv / mass))
    print(f"{len(blocks)-skipped} deletion blocks scored in "
          f"{time.perf_counter()-t0:.0f}s ({skipped} skipped: deleted mass < 5e-4)")
    rows.append(f"n_blocks_skipped,{skipped}")

    # mass strata (populated by design now)
    strata = [(">10%", 0.10, 10.0), ("2-10%", 0.02, 0.10),
              ("0.5-2%", 0.005, 0.02), ("0.05-0.5%", 0.0005, 0.005)]
    names = list(models)
    print(f"\n{'model':>22}", *[f"{s[0]:>10}" for s in strata], "   (mean TV/mass; n in header)")
    hdr_counts = {}
    for lab, lo, hi in strata:
        nblocks = len({(sz, m) for _, sz, m, _ in per_obs if lo < m <= hi})
        hdr_counts[lab] = nblocks
        rows.append(f"n_blocks_{lab},{nblocks}")
    print(f"{'n blocks':>22}", *[f"{hdr_counts[s[0]]:>10}" for s in strata])
    summ = {}
    for nm in names:
        vals = []
        for lab, lo, hi in strata:
            sel = [r_ for n_, sz, m, r_ in per_obs if n_ == nm and lo < m <= hi]
            v = np.mean(sel) if sel else np.nan
            vals.append(v)
            rows.append(f"{nm.replace(' ', '_')}_{lab},{v:.5f}")
            summ[(nm, lab)] = v
        print(f"{nm:>22}", *[f"{v:>10.3f}" for v in vals])

    # singles vs pairs, reported separately (sixth-review request)
    print()
    for nm in names:
        s1 = [r_ for n_, sz, m, r_ in per_obs if n_ == nm and sz == 1]
        s2 = [r_ for n_, sz, m, r_ in per_obs if n_ == nm and sz == 2]
        print(f"{nm:>22}: singles mean TV/mass {np.mean(s1):.3f} "
              f"(n={len(s1)}), pairs {np.mean(s2):.3f} (n={len(s2)})")
        rows += [f"{nm.replace(' ', '_')}_singles_mean,{np.mean(s1):.5f}",
                 f"{nm.replace(' ', '_')}_pairs_mean,{np.mean(s2):.5f}"]

    # factorial decomposition on the mid stratum (largest informative stratum)
    for lab, _, _ in strata[:2]:
        iL, iP = summ[("independent Luce", lab)], summ[("independent probit", lab)]
        fL, fP = summ[("factor mixed logit", lab)], summ[("factor probit", lab)]
        print(f"\nstratum {lab}: factor increment | Gumbel {iL-fL:+.3f}, "
              f"Gaussian {iP-fP:+.3f}; family increment | V=0 {iL-iP:+.3f}, "
              f"V=V* {fL-fP:+.3f}; interaction {(iL-fL)-(iP-fP):+.3f}")
        rows += [f"factor_incr_gumbel_{lab},{iL-fL:.5f}",
                 f"factor_incr_gaussian_{lab},{iP-fP:.5f}",
                 f"family_incr_indep_{lab},{iL-iP:.5f}",
                 f"family_incr_factor_{lab},{fL-fP:.5f}",
                 f"interaction_{lab},{(iL-fL)-(iP-fP):.5f}"]

    (HERE / "results_factorial.csv").write_text("\n".join(rows) + "\n")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    lx = [r_ for n_, sz, m, r_ in per_obs
          if n_ == "independent Luce" and sz == 1]
    py = [r_ for n_, sz, m, r_ in per_obs
          if n_ == "factor probit" and sz == 1]
    ms = [m for n_, sz, m, r_ in per_obs
          if n_ == "independent Luce" and sz == 1]
    sc = ax.scatter(lx, py, c=np.log10(ms), cmap="copper_r", s=42,
                    zorder=3, edgecolors="white", linewidths=0.5)
    lim = [min(min(lx), min(py)) * 0.7, max(max(lx), max(py)) * 1.4]
    ax.plot(lim, lim, "--", color="#999999", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    wins = sum(1 for a_, b_ in zip(lx, py) if b_ < a_)
    med = float(np.median(np.array(lx) / np.array(py)))
    ax.text(0.04, 0.93, f"below the line: factor probit better\n"
            f"({wins}/{len(lx)} deletions, median {med:.1f}x less "
            f"misallocation)", transform=ax.transAxes, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("log10 deleted share mass", fontsize=8)
    ax.set_xlabel("plain logit: misallocated fraction of deleted mass")
    ax.set_ylabel("factor probit: misallocated fraction")
    ax.set_title("Same deletion, two models (all scored single deletions)",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "factorial.png", dpi=150)
    print("\nwrote results_factorial.csv, figures/factorial.png")


if __name__ == "__main__":
    main()
