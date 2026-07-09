"""Rating-lab experiment: 2-D (sprint/staying) ratings and skew calibration.

Uses the HK racing lab (oracle = market^1.05, see planning/rating_lab.md) to
judge ThurstoneRating variants by TV to market-implied truth — the high-power
metric — alongside log loss and ECE:

- skew grid: skewed base densities (time-like: a>0 puts the heavy tail on the
  slow side, the right shape if bad days are worse than good days are good)
- 2-D distance ratings: effective ability = base + lambda(d) * tilt, with
  lambda running from -1/2 at 1000m to +1/2 at 2400m. The tilt makes a
  horse's rating a line over distance. Updates reuse the exact stage
  likelihood over EFFECTIVE ability, then split it into components by
  correlating against the other component's scaled marginal (the same
  EP-with-marginals move as the scale-learning prototype); mid-distance races
  leave the tilt untouched, which is the right identification structure.

Run:  .venv/bin/python research/lab_variants.py   (needs the hkracing cache)

Measured (July 2026, TV to oracle / log loss / ECE):
    Glicko-2 (reference)      0.3059 / 2.3575 / 0.0057
    1-D gaussian (baseline)   0.3207 / 2.3934 / 0.0108
    1-D skew a=1              0.3192 / 2.4040 / 0.0139
    1-D skew a=2              0.3205 / 2.4355 / 0.0183
    1-D skew a=-1             0.3199 / 2.3992 / 0.0129
    2-D distance, gaussian    0.3207 / 2.3962 / 0.0114
    2-D distance, skew a=2    0.3209 / 2.4497 / 0.0187
Verdict: both variants are null on HK racing. The 2-D machinery identifies
specialists sharply on synthetic alternating outcomes (see the smoke test in
git history / tests), but trainers already place horses at suitable trips, so
the counterfactual distance variation the tilt needs rarely appears in real
races — selection masks the axis. Skew buys ~nothing here. Cheap decisive
negatives are exactly what the lab is for; pace (race-level variation that
placement cannot mask) is the better-posed axis — see pace_axis.py.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from thurstone import Density, Race

from winning import Glicko2Rating, ThurstoneRating
from winning.benchmarks.metrics import Metrics
from winning.thurstonerating import _conv_same, _gauss_kernel


def skew_kernel(a: float, unit: float = 0.1, scale: float = 1.0, half: int = 90) -> np.ndarray:
    x = np.arange(-half, half + 1) * unit
    t = x / scale
    p = np.exp(-0.5 * t * t) * (
        1.0 + np.array([math.erf(a * ti / math.sqrt(2.0)) for ti in t])
    )
    return p / p.sum()


def _scaled_density(p: np.ndarray, factor: float, L: int) -> np.ndarray:
    """Density of factor*X given the lattice density of X (|factor| <= 1)."""
    n = 2 * L + 1
    out = np.zeros(n)
    if abs(factor) < 1e-9:
        out[L] = 1.0
        return out
    pos = np.arange(-L, L + 1) * factor + L
    lo = np.floor(pos).astype(int)
    frac = pos - lo
    m = (lo >= 0) & (lo < n)
    np.add.at(out, lo[m], p[m] * (1.0 - frac[m]))
    m2 = (lo + 1 >= 0) & (lo + 1 < n)
    np.add.at(out, lo[m2] + 1, p[m2] * frac[m2])
    s = out.sum()
    return out / s if s > 0 else out


class DistanceThurstoneRating(ThurstoneRating):
    """2-D ratings: mid-distance ability plus a sprint/staying tilt."""

    def __init__(self, tilt_prior_sigma: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.tilt_prior_sigma = float(tilt_prior_sigma)
        self._tilts: Dict[str, np.ndarray] = {}

    @staticmethod
    def lam(distance: float) -> float:
        return min(max((float(distance) - 1000.0) / 1400.0, 0.0), 1.0) - 0.5

    def _tilt(self, name: str) -> np.ndarray:
        t = self._tilts.get(name)
        if t is None:
            t = np.exp(-0.5 * (self._grid / self.tilt_prior_sigma) ** 2)
            t = t / t.sum()
            self._tilts[name] = t
        return t

    def _effective(self, name: str, lam: float) -> np.ndarray:
        base = self._current(name)
        scaled = _scaled_density(self._tilt(name), lam, self.lattice.L)
        eff = _conv_same(base, scaled)
        s = eff.sum()
        return eff / s if s > 0 else base

    def win_probabilities(self, names: Sequence[str], distance: float = 1400.0) -> List[float]:
        lam = self.lam(distance)
        densities = [
            Density(self.lattice, self._perf_pdf(self._effective(nm, lam))) for nm in names
        ]
        return [float(p) for p in Race(densities).state_prices()]

    def observe(
        self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0, distance: float = 1400.0
    ) -> None:
        from winning.ratingsystem import validate_event

        validate_event(names, ranks)
        self.elapse(dt)
        for nm in names:
            self._register(nm)
            self._tilt(nm)
        lam = self.lam(distance)

        by_rank: Dict[int, List[str]] = {}
        for nm, rk in zip(names, ranks):
            by_rank.setdefault(rk, []).append(nm)
        groups = [sorted(by_rank[rk]) for rk in sorted(by_rank)]

        base_prior = {nm: self._beliefs[nm] for nm in names}
        tilt_prior = {nm: self._tilts[nm] for nm in names}
        base_post = dict(base_prior)
        tilt_post = dict(tilt_prior)
        for _ in range(self.iterations):
            eff = {}
            for nm in names:
                scaled = _scaled_density(tilt_post[nm], lam, self.lattice.L)
                e = _conv_same(base_post[nm], scaled)
                s = e.sum()
                eff[nm] = e / s if s > 0 else base_post[nm]
            loglik = self._event_loglik(groups, eff)
            nb, nt = {}, {}
            for nm in names:
                leff = np.exp(loglik[nm] - loglik[nm].max())
                # base component: correlate out the scaled tilt marginal
                scaled = _scaled_density(tilt_post[nm], lam, self.lattice.L)
                l_base = _conv_same(leff, scaled[::-1])
                post = base_prior[nm] * np.maximum(l_base, 1e-300)
                s = post.sum()
                nb[nm] = post / s if s > 0 else base_prior[nm]
                # tilt component: correlate out the base marginal, then read
                # the likelihood at lam * grid (flat when lam == 0)
                g = _conv_same(leff, base_post[nm][::-1])
                l_tilt = np.interp(lam * self._grid, self._grid, g)
                post = tilt_prior[nm] * np.maximum(l_tilt, 1e-300)
                s = post.sum()
                nt[nm] = post / s if s > 0 else tilt_prior[nm]
            base_post, tilt_post = nb, nt
        self._beliefs.update(base_post)
        self._tilts.update(tilt_post)

    def tilt_mean(self, name: str) -> float:
        """Positive = stays better than sprints (time-like sign flipped)."""
        return -float(np.dot(self._tilt(name), self._grid))


def run_lab(system, events, uses_distance: bool = False) -> dict:
    m = Metrics()
    n_warm = int(len(events) * 0.2)
    for idx, ev in enumerate(events):
        system.elapse(ev.dt)
        d = float(ev.context["distance"]) if ev.context else 1400.0
        kw = {"distance": d} if uses_distance else {}
        if idx >= n_warm:
            probs = system.win_probabilities(ev.names, **kw)
            mus = [system.rating(nm).mu for nm in ev.names]
            m.score_event(probs, ev.ranks, mus, truth=ev.truth)
        system.observe(ev.names, ev.ranks, dt=0.0, **kw)
    return m.summary()


def main():
    from winning.benchmarks.forward_chain import evaluate
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    print(f"HK racing lab: {len(events)} races, oracle = market^1.05\n")

    s = evaluate(Glicko2Rating(), events).summary()
    print(f"{'Glicko-2 (reference)':34s} tv={s['tv_vs_oracle']:.4f} "
          f"log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}")

    configs = [
        ("1-D gaussian (baseline)", ThurstoneRating(), False),
        ("1-D skew a=1", ThurstoneRating(base_kernel=skew_kernel(1.0)), False),
        ("1-D skew a=2", ThurstoneRating(base_kernel=skew_kernel(2.0)), False),
        ("1-D skew a=-1", ThurstoneRating(base_kernel=skew_kernel(-1.0)), False),
        ("2-D distance, gaussian", DistanceThurstoneRating(), True),
        ("2-D distance, skew a=2", DistanceThurstoneRating(base_kernel=skew_kernel(2.0)), True),
    ]
    for label, system, uses_distance in configs:
        s = run_lab(system, events, uses_distance=uses_distance)
        print(f"{label:34s} tv={s['tv_vs_oracle']:.4f} "
              f"log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}")


if __name__ == "__main__":
    main()
