"""Rating-lab experiment: pace as a LATENT third axis.

Distance is known before a race; pace is not. The resolution: predict with a
MIXTURE of pace-conditional probability estimates (mixing over the empirical
pace distribution seen so far at that distance), then update conditional on
the REALIZED pace, which HK sectional times reveal ex post (sec_time1,
standardized online within distance bucket — running statistics use past
races only, no leakage).

Effective ability = base + lam_dist * tilt_dist + lam_pace * tilt_pace.
Each component updates by correlating the exact stage likelihood (over
effective ability) against the other components' scaled marginals — the same
EP-with-marginals move as the distance tilt, one component deeper. A mixture
of pace-conditional lattice densities is just a sum of arrays: no Gaussian
family closure needed, which is where the lattice representation quietly
pays again.

Run:  .venv/bin/python research/pace_axis.py   (needs the hkracing cache)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from thurstone import Density, Race

from lab_variants import DistanceThurstoneRating, _scaled_density, run_lab
from winning import ThurstoneRating
from winning.benchmarks.metrics import Metrics
from winning.thurstonerating import _conv_same

PACE_QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


class _RunningStats:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.values: List[float] = []

    def zscore_then_update(self, x: float) -> float:
        """z against PAST observations only, then absorb x."""
        if self.n >= 20 and self.m2 > 0:
            sd = (self.m2 / self.n) ** 0.5
            z = (x - self.mean) / sd
        else:
            z = 0.0
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        return z


class PaceDistanceThurstoneRating(DistanceThurstoneRating):
    """3 components per horse: base, distance tilt, pace tilt (latent axis)."""

    def __init__(self, pace_prior_sigma: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.pace_prior_sigma = float(pace_prior_sigma)
        self._pace_tilts: Dict[str, np.ndarray] = {}
        self._sec1_stats: Dict[str, _RunningStats] = {}
        self._pace_history: Dict[str, List[float]] = {}

    def _pace_tilt(self, name: str) -> np.ndarray:
        t = self._pace_tilts.get(name)
        if t is None:
            t = np.exp(-0.5 * (self._grid / self.pace_prior_sigma) ** 2)
            t = t / t.sum()
            self._pace_tilts[name] = t
        return t

    def realized_pace(self, distance: float, sec1: float) -> float:
        """Standardize sec_time1 online within distance bucket; hot early
        pace (small sec1) maps to positive lam_pace."""
        key = str(int(distance))
        stats = self._sec1_stats.setdefault(key, _RunningStats())
        z = stats.zscore_then_update(sec1)
        lam = min(max(-z / 2.0, -0.5), 0.5)
        self._pace_history.setdefault(key, []).append(lam)
        return lam

    def _pace_mixture(self, distance: float) -> List[float]:
        hist = self._pace_history.get(str(int(distance)), [])
        if len(hist) < 20:
            return [-0.25, 0.0, 0.25]
        arr = np.sort(np.asarray(hist))
        return [float(arr[int(q * (len(arr) - 1))]) for q in PACE_QUANTILES]

    def _effective3(self, name: str, lam_d: float, lam_p: float) -> np.ndarray:
        base = self._current(name)
        e = _conv_same(base, _scaled_density(self._tilt(name), lam_d, self.lattice.L))
        e = _conv_same(e, _scaled_density(self._pace_tilt(name), lam_p, self.lattice.L))
        s = e.sum()
        return e / s if s > 0 else base

    def win_probabilities(self, names: Sequence[str], distance: float = 1400.0) -> List[float]:
        """Mixture of pace-conditional estimates over the predictive pace grid."""
        lam_d = self.lam(distance)
        acc = np.zeros(len(names))
        grid = self._pace_mixture(distance)
        for lam_p in grid:
            densities = [
                Density(self.lattice, self._perf_pdf(self._effective3(nm, lam_d, lam_p)))
                for nm in names
            ]
            acc += np.asarray(Race(densities).state_prices(), dtype=float)
        acc /= len(grid)
        return [float(p) for p in acc / acc.sum()]

    def observe(
        self,
        names: Sequence[str],
        ranks: Sequence[int],
        dt: float = 1.0,
        distance: float = 1400.0,
        sec1: float = None,
    ) -> None:
        from winning.ratingsystem import validate_event

        validate_event(names, ranks)
        self.elapse(dt)
        for nm in names:
            self._register(nm)
            self._tilt(nm)
            self._pace_tilt(nm)
        lam_d = self.lam(distance)
        lam_p = self.realized_pace(distance, sec1) if sec1 is not None else 0.0

        by_rank: Dict[int, List[str]] = {}
        for nm, rk in zip(names, ranks):
            by_rank.setdefault(rk, []).append(nm)
        groups = [sorted(by_rank[rk]) for rk in sorted(by_rank)]

        priors = {
            nm: (self._beliefs[nm], self._tilts[nm], self._pace_tilts[nm]) for nm in names
        }
        post = dict(priors)
        L = self.lattice.L
        for _ in range(self.iterations):
            eff = {}
            for nm in names:
                b, td, tp = post[nm]
                e = _conv_same(b, _scaled_density(td, lam_d, L))
                e = _conv_same(e, _scaled_density(tp, lam_p, L))
                s = e.sum()
                eff[nm] = e / s if s > 0 else b
            loglik = self._event_loglik(groups, eff)
            new_post = {}
            for nm in names:
                b, td, tp = post[nm]
                pb, ptd, ptp = priors[nm]
                leff = np.exp(loglik[nm] - loglik[nm].max())
                # each component: correlate out the OTHER two components
                rest_d = _scaled_density(td, lam_d, L)
                rest_p = _scaled_density(tp, lam_p, L)
                others_b = _conv_same(rest_d, rest_p)  # density of sum of tilts
                l_base = _conv_same(leff, others_b[::-1])
                nb = pb * np.maximum(l_base, 1e-300)
                nb = nb / nb.sum() if nb.sum() > 0 else pb

                others_d = _conv_same(b, rest_p)
                g = _conv_same(leff, others_d[::-1])
                l_td = np.interp(lam_d * self._grid, self._grid, g)
                ntd = ptd * np.maximum(l_td, 1e-300)
                ntd = ntd / ntd.sum() if ntd.sum() > 0 else ptd

                others_p = _conv_same(b, rest_d)
                g = _conv_same(leff, others_p[::-1])
                l_tp = np.interp(lam_p * self._grid, self._grid, g)
                ntp = ptp * np.maximum(l_tp, 1e-300)
                ntp = ntp / ntp.sum() if ntp.sum() > 0 else ptp

                new_post[nm] = (nb, ntd, ntp)
            post = new_post
        for nm, (nb, ntd, ntp) in post.items():
            self._beliefs[nm] = nb
            self._tilts[nm] = ntd
            self._pace_tilts[nm] = ntp


def main():
    from winning.benchmarks.kaggle_datasets import hkracing_events

    events = hkracing_events(oracle_temperature=1.05)
    print(f"HK racing lab: {len(events)} races, oracle = market^1.05\n")

    # 3-D rater with pace: bespoke loop passing distance + sec1
    system = PaceDistanceThurstoneRating()
    m = Metrics()
    n_warm = int(len(events) * 0.2)
    for idx, ev in enumerate(events):
        system.elapse(ev.dt)
        d = float(ev.context["distance"])
        try:
            sec1 = float(ev.context["sec_time1"])
        except (TypeError, ValueError):
            sec1 = None
        if idx >= n_warm:
            probs = system.win_probabilities(ev.names, distance=d)
            mus = [system.rating(nm).mu for nm in ev.names]
            m.score_event(probs, ev.ranks, mus, truth=ev.truth)
        system.observe(ev.names, ev.ranks, dt=0.0, distance=d, sec1=sec1)
    s = m.summary()
    print(f"{'3-D base+distance+pace':34s} tv={s['tv_vs_oracle']:.4f} "
          f"log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}")

    s = run_lab(DistanceThurstoneRating(), events, uses_distance=True)
    print(f"{'2-D base+distance (reference)':34s} tv={s['tv_vs_oracle']:.4f} "
          f"log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}")
    s = run_lab(ThurstoneRating(), events, uses_distance=False)
    print(f"{'1-D baseline (reference)':34s} tv={s['tv_vs_oracle']:.4f} "
          f"log_loss={s['log_loss']:.4f} ece={s['ece']:.4f}")


if __name__ == "__main__":
    main()
