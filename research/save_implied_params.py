"""Refit win+place (loc, scale) per race and SAVE everything the follow-up
analyses need: fitted params, market rank, and per-horse Jacobian diagnostics
(scale-column sensitivity and loc/scale collinearity — the direct answer to
'is scale identified for favorites?').

Output: ~/.cache/winning/implied_params.npz
Run:  .venv/bin/python research/save_implied_params.py
"""

from __future__ import annotations

import os

import numpy as np
from thurstone import AbilityCalibrator, Density, UniformLattice

from win_place_calibration import CLIP, GRID, exact_win_topk, fit_loc_scale


def jacobian_diag(locs, scales, win, place, n_places):
    """Per-horse: scale-column norm and |cos| angle to its own loc column."""
    n = len(locs)
    th = np.concatenate([locs, np.log(scales)])

    def resid(t):
        lo = t[:n] - t[:n].mean()
        sc = np.exp(np.clip(t[n:], -0.9, 0.9))
        pw, pk = exact_win_topk(lo, sc, GRID, n_places)
        return np.concatenate([np.log(pw) - np.log(win), np.log(pk) - np.log(place)])

    r0 = resid(th)
    J = np.empty((len(r0), 2 * n))
    h = 1e-4
    for c in range(2 * n):
        tp = th.copy()
        tp[c] += h
        J[:, c] = (resid(tp) - r0) / h
    sens = np.linalg.norm(J[:, n:], axis=0)
    cosang = np.empty(n)
    for i in range(n):
        a, b = J[:, i], J[:, n + i]
        cosang[i] = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return sens, cosang


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = [e for e in hkracing_events() if e.context and "place_market" in e.context]
    lattice = UniformLattice(L=150, unit=0.1)
    cal = AbilityCalibrator(Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0))
    rng = np.random.default_rng(23)

    rows = []  # race_idx, horse, loc, scale, market_win, market_rank, sens, cos, finish_rank, field
    for idx, ev in enumerate(events):
        pm = ev.context["place_market"]
        win = np.maximum(np.asarray(ev.market, float), CLIP)
        win = win / win.sum()
        place = np.clip(np.asarray(pm["probs"], float), CLIP, 0.98)
        locs, scales = fit_loc_scale(win, place, pm["n_places"], cal, rng)
        sens, cosang = jacobian_diag(locs, scales, win, place, pm["n_places"])
        mrank = (-win).argsort().argsort() + 1
        for i, nm in enumerate(ev.names):
            rows.append((idx, nm, locs[i], scales[i], win[i], mrank[i],
                         sens[i], cosang[i], ev.ranks[i], len(ev.names)))
        if idx % 500 == 0:
            print(f"{idx}/{len(events)}", flush=True)

    out = os.path.expanduser("~/.cache/winning/implied_params.npz")
    np.savez_compressed(
        out,
        race=np.array([r[0] for r in rows]),
        horse=np.array([r[1] for r in rows]),
        loc=np.array([r[2] for r in rows]),
        scale=np.array([r[3] for r in rows]),
        market_win=np.array([r[4] for r in rows]),
        market_rank=np.array([r[5] for r in rows]),
        scale_sens=np.array([r[6] for r in rows]),
        loc_scale_cos=np.array([r[7] for r in rows]),
        finish_rank=np.array([r[8] for r in rows]),
        field=np.array([r[9] for r in rows]),
    )
    print("saved", out, len(rows), "runner-rows")


if __name__ == "__main__":
    main()
