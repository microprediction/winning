"""Research: can model + CURRENT race odds beat the raw market? (Benter's move)

The market row is called a ceiling because its inputs strictly contain every
rating system's inputs. But the pari-mutuel carries known biases (favorite-
longshot), so a log-linear pool q_i ∝ market_i^a · model_i^b can undercut the
raw market's log loss: `a` acts as a temperature fixing miscalibration, `b`
adds whatever residual signal the model carries. Bill Benter's Hong Kong
syndicate (this dataset's market, same era) monetized exactly this structure.

Leakage discipline: the exponents (a, b) are refit every 250 scored races by
grid search on ALL PAST scored races only; race t is always priced with
parameters that never saw it. The model probabilities come from the
market-hybrid rater (history odds + outcomes, never race t's odds); the
market probabilities are race t's own normalized inverse odds.

Rows reported:
    market raw            (a=1, b=0)  — the usual ceiling
    market recalibrated   (fit a, b=0) — pure longshot-bias correction
    pool (fit a, b)       — Benter-style combination

Run:  .venv/bin/python research/beat_the_market.py   (needs the hkracing cache)

Measured (July 2026, HK racing, 5,079 scored races):
    market raw (ceiling)          log_loss 2.0407   ece 0.0053
    market recalibrated (a=1.05)  log_loss 2.0398   ece 0.0045
    pool market^a * model^b       log_loss 2.0397   (b=0.05)
Verdict: the 1997-2005 HK pari-mutuel is essentially efficient against this
information set. Past results and past odds are public information the market
already absorbs, so the model exponent fits at b=0.05 — nothing material.
The mild temperature a=1.05 confirms a small favorite-longshot bias worth
~0.001 log loss and a visible calibration gain. Benter-scale edges came from
fundamental context (weights, draw, pace, sectionals) this rater does not yet
consume — which is the condition-aware agenda, and the reason the fitted
a=1.05 now defines the oracle of the rating lab (planning/rating_lab.md)
rather than a betting strategy.
"""

from __future__ import annotations

import math

import numpy as np

from market_hybrid import MarketHybridThurstoneRating

REFIT_EVERY = 250
A_GRID = np.linspace(0.6, 1.8, 25)
B_GRID = np.linspace(0.0, 0.8, 17)
CLIP = 1e-12


class OnlinePool:
    """Log-linear pool with exponents grid-fit on past races only."""

    def __init__(self, fit_b: bool = True):
        self.fit_b = fit_b
        self.a, self.b = 1.0, 0.0
        self._logm, self._logp, self._starts, self._winners = [], [], [0], []
        self._since_fit = 0

    def predict(self, market, model):
        z = self.a * np.log(np.maximum(market, CLIP)) + self.b * np.log(
            np.maximum(model, CLIP)
        )
        z = z - z.max()
        q = np.exp(z)
        return q / q.sum()

    def record(self, market, model, winner_idx: int) -> None:
        self._logm.extend(np.log(np.maximum(market, CLIP)))
        self._logp.extend(np.log(np.maximum(model, CLIP)))
        self._winners.append(self._starts[-1] + winner_idx)
        self._starts.append(len(self._logm))
        self._since_fit += 1
        if self._since_fit >= REFIT_EVERY:
            self._refit()
            self._since_fit = 0

    def _refit(self) -> None:
        logm = np.asarray(self._logm)
        logp = np.asarray(self._logp)
        starts = np.asarray(self._starts[:-1])
        winners = np.asarray(self._winners)
        b_grid = B_GRID if self.fit_b else np.array([0.0])
        best = (math.inf, self.a, self.b)
        for a in A_GRID:
            base = a * logm
            for b in b_grid:
                z = base + b * logp
                zmax = np.maximum.reduceat(z, starts)
                # subtract each race's max before exponentiating
                race_of = np.repeat(np.arange(len(starts)), np.diff(self._starts))
                ez = np.exp(z - zmax[race_of])
                sums = np.add.reduceat(ez, starts)
                loss = float(-np.mean(z[winners] - zmax - np.log(sums)))
                if loss < best[0]:
                    best = (loss, float(a), float(b))
        _, self.a, self.b = best


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events
    from winning.benchmarks.metrics import Metrics

    events = hkracing_events()
    n_warm = int(len(events) * 0.2)
    print(f"HK racing: {len(events)} races; scoring after {n_warm}-race warmup\n")

    hy = MarketHybridThurstoneRating(market_obs_sd=0.7)
    rows = {
        "market raw (ceiling)": (Metrics(), OnlinePool(fit_b=True)),  # a=1,b=0 fixed
        "market recalibrated (fit a)": (Metrics(), OnlinePool(fit_b=False)),
        "pool: market^a * model^b": (Metrics(), OnlinePool(fit_b=True)),
    }

    for idx, ev in enumerate(events):
        hy.elapse(ev.dt)
        market = np.asarray(ev.market, dtype=float)
        if idx >= n_warm:
            model = np.asarray(hy.win_probabilities(ev.names), dtype=float)
            winner = min(range(len(ev.ranks)), key=lambda i: ev.ranks[i])
            for label, (metrics, pool) in rows.items():
                if label.startswith("market raw"):
                    q = market / market.sum()
                else:
                    q = pool.predict(market, model)
                metrics.score_event(list(q), ev.ranks, list(q))
            for label, (metrics, pool) in rows.items():
                if not label.startswith("market raw"):
                    pool.record(market, model, winner)
        hy.observe(ev.names, ev.ranks, dt=0.0)
        if ev.market is not None:
            hy.observe_market(ev.names, ev.market)

    for label, (metrics, pool) in rows.items():
        s = metrics.summary()
        extra = "" if label.startswith("market raw") else f"  (a={pool.a:.2f}, b={pool.b:.2f})"
        print(
            f"{label:30s} log_loss={s['log_loss']:.4f} acc={s['accuracy']:.4f} "
            f"ece={s['ece']:.4f}{extra}"
        )


if __name__ == "__main__":
    main()
