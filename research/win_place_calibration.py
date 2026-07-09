"""Joint win+place calibration: two market prices identify (loc, scale).

The ability transform inverts WIN probabilities into locations. HK also
quotes PLACE odds (top-3 for 7+ runner fields, top-2 below): a second price
per horse, identifying a second parameter. This fits a full Thurstonian
field spec (loc_i, scale_i) per race by coordinate sweeps — locations chase
market win probabilities (monotone log-ratio steps), scales chase market
place probabilities (secant steps; the sign of d place/d scale flips between
favorites and longshots) — with common-random-number simulation making the
objective deterministic. Gauges: locations centered, place probabilities
normalized to the number of paid places (also strips the overround).

Judged prequentially at positions 1-6 from IDENTICAL race-t prices:
  A. Luce peeling of win odds            (the industry shortcut)
  B. Thurstone simulation, loc only      (previous best market pipeline)
  C. Thurstone simulation, loc + scale   (this file: win+place jointly)
Plus the trait test: do a horse's market-implied scales persist across its
races? Persistence = the market believes in per-horse consistency — the
scale-learning premise validated with market money rather than outcomes.

Run:  .venv/bin/python research/win_place_calibration.py

Measured (July 2026, 6,047 races with full place odds, log loss by position):
    luce (win only)      2.0394 2.2560 2.3927 2.4570 2.4950 2.5141 | mean 2.3590
    thurstone loc        2.0384 2.2559 2.3792 2.4351 2.4704 2.4858 | mean 2.3441
    thurstone loc+scale  2.0411 2.2512 2.3715 2.4280 2.4662 2.4837 | mean 2.3403
    fitted scales mean 1.03 sd 0.24; per-horse split-half persistence 0.772
Verdicts: (1) the place market carries rank information beyond win odds, and
joint (loc, scale) calibration extracts it — best at every position 2-6 for
a rounding-error cost at P1; (2) market-implied volatility is a strongly
persistent horse trait (0.77 split-half over 2,693 horses) — the market
prices "this horse is erratic" as a durable attribute, externally validating
the scale-learning premise; (3) caveat under test: persistence could partly
proxy class/rank persistence — the implied-vs-REALIZED-vol analysis and the
Jacobian conditioning-by-rank readout (save_implied_params.py) discriminate.
The exact top-k lattice machinery built here is thurstone issue 01 realized.
"""

from __future__ import annotations

import numpy as np
from thurstone import AbilityCalibrator, Density, UniformLattice

from place_probabilities import logloss_at

TARGETS = (1, 2, 3, 4, 5, 6)
S_FIT = 2048
S_SCORE = 4096
SWEEPS = 6
CLIP = 1e-9


def _normal_pdfs(locs, scales, grid):
    z = (grid[None, :] - locs[:, None]) / scales[:, None]
    p = np.exp(-0.5 * z * z)
    return p / p.sum(axis=1, keepdims=True)


def exact_win_topk(locs, scales, grid, k):
    """Exact P(win) and P(top-k) per contestant on the lattice: for each i,
    a Poisson-binomial DP over opponents' CDFs tracks P(0),P(1),...,P(k-1)
    opponents ahead, integrated against i's density. O(N^2 * L * k)."""
    pdfs = _normal_pdfs(locs, scales, grid)
    cdfs = np.cumsum(pdfs, axis=1)
    n = len(locs)
    pw = np.empty(n)
    pk = np.empty(n)
    for i in range(n):
        c = np.zeros((k, len(grid)))
        c[0] = 1.0
        for j in range(n):
            if j == i:
                continue
            q = cdfs[j]
            for m in range(k - 1, 0, -1):
                c[m] = c[m] * (1 - q) + c[m - 1] * q
            c[0] = c[0] * (1 - q)
        pw[i] = float(np.dot(pdfs[i], c[0]))
        pk[i] = float(np.dot(pdfs[i], c.sum(axis=0)))
    return np.maximum(pw, CLIP), np.maximum(pk, CLIP)


GRID = np.linspace(-8.0, 8.0, 321)
RIDGE = 0.08


def fit_loc_scale(win_probs, place_probs, n_places, cal, rng):
    """Ridge-regularized Gauss-Newton on the exact lattice map."""
    del rng  # deterministic now
    win = np.maximum(np.asarray(win_probs, float), CLIP)
    win = win / win.sum()
    place = np.clip(np.asarray(place_probs, float), CLIP, 0.98)
    n = len(win)
    locs = np.asarray(cal.solve_from_prices(list(win)), float)
    locs = np.clip(locs - locs.mean(), -3.5, 3.5)
    log_sc = np.zeros(n)

    def residuals(th):
        lo = th[:n] - th[:n].mean()
        sc = np.exp(np.clip(th[n:], -0.9, 0.9))
        pw, pk = exact_win_topk(lo, sc, GRID, n_places)
        r = np.concatenate([
            np.log(pw) - np.log(win),
            np.log(pk) - np.log(place),
            np.sqrt(RIDGE) * th[n:],
        ])
        return r

    th = np.concatenate([locs, log_sc])
    lam = 1.0
    r = residuals(th)
    cost = float(r @ r)
    for _ in range(10):
        J = np.empty((len(r), 2 * n))
        h = 1e-4
        for c in range(2 * n):
            tp = th.copy()
            tp[c] += h
            J[:, c] = (residuals(tp) - r) / h
        A = J.T @ J + lam * np.eye(2 * n)
        step = np.linalg.solve(A, -(J.T @ r))
        th_new = th + step
        r_new = residuals(th_new)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            th, r, cost = th_new, r_new, cost_new
            lam = max(lam * 0.5, 1e-3)
        else:
            lam *= 4.0
        if cost < 1e-8:
            break
    locs = th[:n] - th[:n].mean()
    scales = np.exp(np.clip(th[n:], -0.9, 0.9))
    return locs, scales


def rank_matrix(locs, scales, rng, size=S_SCORE):
    perf = locs[None, :] + scales[None, :] * rng.standard_normal((size, len(locs)))
    sim = perf.argsort(axis=1).argsort(axis=1) + 1
    n = len(locs)
    out = np.zeros((n, n))
    for k in range(1, n + 1):
        out[k - 1] = (sim == k).mean(axis=0)
    return out


def main():
    from collections import defaultdict

    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = [e for e in hkracing_events() if e.context and "place_market" in e.context]
    n_warm = int(len(events) * 0.2)
    rng = np.random.default_rng(23)
    lattice = UniformLattice(L=150, unit=0.1)
    cal = AbilityCalibrator(Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0))
    print(f"HK races with full place odds: {len(events)}; scoring after {n_warm}\n")

    acc = {k: {t: [] for t in TARGETS} for k in ("luce", "thurstone_loc", "thurstone_loc_scale")}
    fitted = defaultdict(list)  # horse -> [(race_idx, scale)]

    for idx, ev in enumerate(events):
        if idx < n_warm:
            continue
        pm = ev.context["place_market"]
        win = np.maximum(np.asarray(ev.market, float), CLIP)
        win = win / win.sum()
        n = len(ev.names)
        pos_idx = {}
        for t in TARGETS:
            holders = [i for i, r in enumerate(ev.ranks) if r == t]
            pos_idx[t] = holders[0] if len(holders) == 1 and t <= n else None

        # A: Luce peeling from win odds
        g = rng.gumbel(size=(S_SCORE, n))
        sim = (-(np.log(win) + g)).argsort(axis=1).argsort(axis=1) + 1
        lp = np.zeros((n, n))
        for k in range(1, n + 1):
            lp[k - 1] = (sim == k).mean(axis=0)
        # B: Thurstone, loc only
        locs0 = np.asarray(cal.solve_from_prices(list(win)), float)
        tp0 = rank_matrix(locs0, np.ones(n), rng)
        # C: Thurstone, loc + scale from win+place
        locs, scales = fit_loc_scale(win, pm["probs"], pm["n_places"], cal, rng)
        tp1 = rank_matrix(locs, scales, rng)
        for nm, sc in zip(ev.names, scales):
            fitted[nm].append(float(sc))

        for t in TARGETS:
            if pos_idx[t] is None:
                continue
            acc["luce"][t].append(logloss_at(lp[t - 1], pos_idx[t]))
            acc["thurstone_loc"][t].append(logloss_at(tp0[t - 1], pos_idx[t]))
            acc["thurstone_loc_scale"][t].append(logloss_at(tp1[t - 1], pos_idx[t]))

    print(f"{'pipeline':24s}" + "".join(f"{('P' + str(t)):>9s}" for t in TARGETS) + f"{'mean':>9s}")
    for k in acc:
        cells = [np.mean(acc[k][t]) for t in TARGETS]
        print(f"{k:24s}" + "".join(f"{c:9.4f}" for c in cells) + f"{np.mean(cells):9.4f}")

    # persistence of market-implied scale as a horse trait
    halves = [(np.mean(v[0::2]), np.mean(v[1::2])) for v in fitted.values() if len(v) >= 6]
    a = np.array([x for x, _ in halves]); b = np.array([y for _, y in halves])
    corr = np.corrcoef(a, b)[0, 1]
    allsc = np.concatenate([np.asarray(v) for v in fitted.values()])
    print(f"\nfitted scales: mean {allsc.mean():.3f}, sd {allsc.std():.3f}")
    print(f"per-horse persistence (split-half corr, {len(halves)} horses with 6+ runs): {corr:.3f}")


if __name__ == "__main__":
    main()
