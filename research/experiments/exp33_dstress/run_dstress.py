"""Experiment 33: D-heterogeneity stress test for the common grid.

The common envelope is set by max_i sqrt(D_i) while a fixed L sets the
spacing. If one D_i is orders of magnitude smaller than another, its
conditional density can be narrower than the grid spacing even though
the envelope is adequate. This experiment measures forward accuracy at
production settings under variance ratios 1e2 and 1e3, against (a)
higher-resolution internal references (L = 2001, 8001) and (b) an
independent 2e7-draw Monte Carlo reference.

Run:  python experiments/exp33_dstress/run_dstress.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raceutil import hermite_nodes, win_probabilities_factor  # noqa: E402

HERE = Path(__file__).resolve().parent
N, K = 20, 2
R_MC = 20_000_000


def mc_reference(a, V, D, rng):
    counts = np.zeros(N)
    chunk = 500_000
    done = 0
    while done < R_MC:
        m = min(chunk, R_MC - done)
        f = rng.standard_normal((m, K))
        x = a[None, :] + f @ V.T + np.sqrt(D)[None, :] * rng.standard_normal((m, N))
        counts += np.bincount(np.argmin(x, axis=1), minlength=N)
        done += m
    return counts / R_MC


def main():
    rng = np.random.default_rng(33)
    F, W = hermite_nodes(K)
    lines = []
    for ratio in (1e2, 1e3):
        a = rng.normal(0, 1.0, N)
        a -= a.mean()
        V = rng.normal(0, np.sqrt(0.36 / K), (N, K))
        # D log-spaced from 1/ratio to 1, shuffled
        D = np.exp(np.linspace(np.log(1.0 / ratio), 0.0, N))
        rng.shuffle(D)

        ps = {L: win_probabilities_factor(a, V, D, F, W, points=L)
              for L in (501, 2001, 8001)}
        pmc = mc_reference(a, V, D, rng)

        ref = ps[8001]
        for L in (501, 2001):
            mask = ref > 1e-9
            logerr = np.abs(np.log(ps[L][mask]) - np.log(ref[mask])).max()
            abserr = np.abs(ps[L] - ref).max()
            lines.append(f"ratio {ratio:.0e} L={L}: max|dlog|={logerr:.3e} "
                         f"max|dp|={abserr:.3e} vs L=8001")
        # vs MC on resolvable shares (> ~30/R_MC)
        mask = pmc > 30 / R_MC
        logerr_mc = np.abs(np.log(ps[501][mask]) - np.log(pmc[mask])).max()
        lines.append(f"ratio {ratio:.0e} L=501 vs MC({R_MC:.0e} draws): "
                     f"max|dlog|={logerr_mc:.3e} on {mask.sum()} resolvable "
                     f"(MC rel noise ~{1/np.sqrt(pmc[mask].min()*R_MC):.1e})")
        lines.append(f"ratio {ratio:.0e} share range: [{ref.min():.2e}, "
                     f"{ref.max():.2e}], D range [{D.min():.1e}, {D.max():.1e}]")
    out = "\n".join(lines) + "\n"
    print(out)
    (HERE / "results.txt").write_text(out)


if __name__ == "__main__":
    main()
