"""Research prototype: odds as an INPUT, not just a ceiling.

thurstone's ability transform inverts a market's win probabilities into
relative abilities — a high-precision observation of the same latent quantity
that finish orders reveal one noisy draw at a time. This hybrid feeds both to
the lattice rater: each race updates beliefs with (a) the exact
order-statistic likelihood of the outcome, as usual, and (b) a Gaussian
observation at the market-implied ability of that race (inverted via
thurstone.AbilityCalibrator, median-centered to remove the per-race gauge).

Forecast protocol is honest: race t is predicted from beliefs built on races
1..t-1's outcomes AND odds — never race t's own odds — so the comparison
against the outcome-only rater isolates the value of market history, and the
Market row (which does use race t's odds) remains the ceiling.

Run:  .venv/bin/python research/market_hybrid.py   (needs the hkracing cache)
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from thurstone import AbilityCalibrator, Density

from winning.thurstonerating import ThurstoneRating


class MarketHybridThurstoneRating(ThurstoneRating):
    """ThurstoneRating that also learns from past races' market odds."""

    def __init__(self, market_obs_sd: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.market_obs_sd = float(market_obs_sd)
        base = Density.skew_normal(self.lattice, loc=0.0, scale=self.beta, a=0.0)
        self._calibrator = AbilityCalibrator(base)

    def observe_market(self, names: Sequence[str], probs: Sequence[float]) -> None:
        """Update each contestant's belief with the market-implied ability."""
        implied = np.asarray(self._calibrator.solve_from_prices(list(probs)), dtype=float)
        implied = implied - float(np.median(implied))  # remove per-race gauge
        r = self.market_obs_sd
        for nm, m in zip(names, implied):
            self._register(nm)
            like = np.exp(-0.5 * ((self._grid - m) / r) ** 2)
            post = self._beliefs[nm] * like
            total = post.sum()
            if total > 0:
                self._beliefs[nm] = post / total


def main():
    from winning import ThurstoneRating as Fixed
    from winning.benchmarks.forward_chain import evaluate
    from winning.benchmarks.kaggle_datasets import hkracing_events
    from winning.benchmarks.metrics import Metrics

    events = hkracing_events()
    print(f"HK racing: {len(events)} races, market attached to all\n")

    # baseline rows through the standard loop
    for label, system, fixed in [
        ("outcome-only ThurstoneRating", Fixed(), None),
        ("market ceiling (race t odds)", None, "market"),
    ]:
        m = evaluate(system, events, fixed=fixed)
        s = m.summary()
        print(
            f"{label:34s} log_loss={s['log_loss']:.4f} acc={s['accuracy']:.4f} "
            f"tau={s['kendall_tau']:.4f} ece={s['ece']:.4f}"
        )

    # hybrid: bespoke loop so past-race odds enter AFTER race t is scored
    for obs_sd in (0.7, 0.5, 0.35):
        hy = MarketHybridThurstoneRating(market_obs_sd=obs_sd)
        m = Metrics()
        n_warm = int(len(events) * 0.2)
        for idx, ev in enumerate(events):
            hy.elapse(ev.dt)
            if idx >= n_warm:
                probs = hy.win_probabilities(ev.names)
                mus = [hy.rating(nm).mu for nm in ev.names]
                m.score_event(probs, ev.ranks, mus, truth=ev.truth)
            hy.observe(ev.names, ev.ranks, dt=0.0)
            if ev.market is not None:
                hy.observe_market(ev.names, ev.market)
        s = m.summary()
        print(
            f"{'hybrid, market_obs_sd=' + str(obs_sd):34s} log_loss={s['log_loss']:.4f} "
            f"acc={s['accuracy']:.4f} tau={s['kendall_tau']:.4f} ece={s['ece']:.4f}"
        )


if __name__ == "__main__":
    main()
