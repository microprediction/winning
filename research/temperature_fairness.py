"""Fair tempering: every system gets the same prequential temperature layer.

The chess mismatch study fixed the site ratings with a fitted temperature —
but a fitted post-hoc parameter must be offered to every competitor or to
none. Here EVERY system gets an identical online recalibration layer:
q_i ∝ p_i^s renormalized, with s refit every 250 scored events by grid
search on PAST scored events only (no leakage). Report raw and tempered log
loss plus the final fitted s per system.

The fitted s doubles as a SELF-HONESTY diagnostic: systems whose
probabilities already embed honest uncertainty should fit s ~ 1 and gain
nothing (the Bayesian-shrinkage story, falsifiable); point-estimate systems
should fit s > 1 and gain.

Run:  .venv/bin/python research/temperature_fairness.py
"""

from __future__ import annotations

import math

import numpy as np

from winning import EloRating, Glicko2Rating, ThurstoneRating
from winning.shims import TrueSkillRating

REFIT_EVERY = 250
S_GRID = np.linspace(0.5, 2.0, 31)
CLIP = 1e-12


class OnlineTemperature:
    def __init__(self):
        self.s = 1.0
        self._logp, self._starts, self._winners = [], [0], []
        self._since = 0

    def apply(self, probs):
        z = self.s * np.log(np.maximum(probs, CLIP))
        z -= z.max()
        q = np.exp(z)
        return q / q.sum()

    def record(self, probs, winner_idx):
        self._logp.extend(np.log(np.maximum(probs, CLIP)))
        self._winners.append(self._starts[-1] + winner_idx)
        self._starts.append(len(self._logp))
        self._since += 1
        if self._since >= REFIT_EVERY:
            self._refit()
            self._since = 0

    def _refit(self):
        logp = np.asarray(self._logp)
        starts = np.asarray(self._starts[:-1])
        winners = np.asarray(self._winners)
        race_of = np.repeat(np.arange(len(starts)), np.diff(self._starts))
        best = (math.inf, self.s)
        for s in S_GRID:
            z = s * logp
            zmax = np.maximum.reduceat(z, starts)
            sums = np.add.reduceat(np.exp(z - zmax[race_of]), starts)
            loss = float(-np.mean(z[winners] - zmax - np.log(sums)))
            if loss < best[0]:
                best = (loss, float(s))
        self.s = best[1]


def run(events, label):
    systems = {
        "Elo": EloRating(),
        "Glicko-2": Glicko2Rating(),
        "TrueSkill": TrueSkillRating(),
        "Thurstone lattice": ThurstoneRating(),
    }
    acc = {nm: {"raw": [], "temp": [], "layer": OnlineTemperature()} for nm in systems}
    n_warm = int(len(events) * 0.2)
    for idx, ev in enumerate(events):
        for sy in systems.values():
            sy.elapse(ev.dt)
        if idx >= n_warm:
            holders = [i for i, r in enumerate(ev.ranks) if r == min(ev.ranks)]
            winner = holders[0] if len(holders) == 1 else None
            if winner is not None:
                for nm, sy in systems.items():
                    p = np.asarray(sy.win_probabilities(ev.names), dtype=float)
                    q = acc[nm]["layer"].apply(p)
                    acc[nm]["raw"].append(-math.log(max(p[winner] / p.sum(), CLIP)))
                    acc[nm]["temp"].append(-math.log(max(q[winner], CLIP)))
                    acc[nm]["layer"].record(p, winner)
        for sy in systems.values():
            sy.observe(ev.names, ev.ranks, dt=0.0)

    print(label)
    print(f"  {'system':<20s} {'raw ll':>9s} {'tempered':>9s} {'gain':>8s} {'fitted s':>9s}")
    for nm, d in acc.items():
        raw, temp = np.mean(d["raw"]), np.mean(d["temp"])
        print(f"  {nm:<20s} {raw:9.4f} {temp:9.4f} {raw-temp:+8.4f} {d['layer'].s:9.2f}")
    print()


def main():
    from winning.benchmarks.chess import chess_events
    from winning.benchmarks.kaggle_datasets import hkracing_events

    run(hkracing_events(), "HK racing (6,348 races)")
    run(chess_events(), "chess, Lichess 2013-01 (121k games)")


if __name__ == "__main__":
    main()
