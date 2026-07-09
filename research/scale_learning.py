"""Research prototype: learning per-contestant performance VARIATION.

Every benchmarked rating system (TrueSkill, OpenSkill, Glicko-2, Elo, and this
package's ThurstoneRating) assumes a common performance noise. Real fields mix
steady and erratic contestants, and the heteroskedastic synthetic benchmark
shows what that assumption costs. This prototype extends the lattice rater
with a joint belief over (ability, noise scale): each contestant carries one
ability density per point of a small scale grid, plus mixture weights over the
grid. Events update both by exact Bayes on the Plackett-peeled stage
likelihoods — the ability densities slice-by-slice, the scale weights via each
slice's marginal likelihood.

Run:  .venv/bin/python research/scale_learning.py

Outputs a comparison on the heteroskedastic world (log loss, TV vs oracle,
ECE) between the fixed-scale rater, this scale-learner, and the oracle, plus
the correlation between learned scales and the true per-contestant noise.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from thurstone import Density, Race, UniformLattice

from winning.ratingsystem import Rating, RatingSystem, validate_event
from winning.thurstonerating import _conv_same, _gauss_kernel

SCALE_GRID = (0.55, 0.8, 1.0, 1.3, 1.7)


class ScaleLearningThurstoneRating(RatingSystem):
    """Joint (ability, scale) beliefs on the lattice; scale learned per contestant."""

    def __init__(
        self,
        prior_sigma: float = 1.0,
        scales: Sequence[float] = SCALE_GRID,
        tau: float = 0.02,
        unit: float = 0.1,
        L: int = 150,
        iterations: int = 2,
    ):
        self.prior_sigma = float(prior_sigma)
        self.scales = tuple(float(s) for s in scales)
        self.tau = float(tau)
        self.iterations = max(1, int(iterations))
        self.lattice = UniformLattice(L=int(L), unit=float(unit))
        self._grid = self.lattice.grid
        self._kernels = [_gauss_kernel(unit, s) for s in self.scales]
        # name -> (weights over scales, [ability density per scale])
        self._beliefs: Dict[str, tuple] = {}
        self._last_time: Dict[str, float] = {}
        self._clock: float = 0.0

    # ---------------- belief plumbing ----------------

    def _prior(self):
        p = np.exp(-0.5 * (self._grid / self.prior_sigma) ** 2)
        p = p / p.sum()
        k = len(self.scales)
        return (np.full(k, 1.0 / k), [p.copy() for _ in range(k)])

    def _current(self, name: str):
        got = self._beliefs.get(name)
        if got is None:
            return self._prior()
        w, slices = got
        dt = self._clock - self._last_time[name]
        if dt <= 0 or self.tau <= 0:
            return (w.copy(), [s.copy() for s in slices])
        k = _gauss_kernel(self.lattice.unit, self.tau * math.sqrt(dt))
        out = []
        for s in slices:
            b = _conv_same(s, k)
            t = b.sum()
            out.append(b / t if t > 0 else self._prior()[1][0])
        return (w.copy(), out)

    def _register(self, name: str) -> None:
        self._beliefs[name] = self._current(name)
        self._last_time[name] = self._clock

    def _perf_mixture(self, belief) -> np.ndarray:
        w, slices = belief
        p = np.zeros_like(self._grid)
        for wi, b, kern in zip(w, slices, self._kernels):
            p = p + wi * _conv_same(b, kern)
        t = p.sum()
        return p / t if t > 0 else p

    # ---------------- interface ----------------

    def elapse(self, dt: float) -> None:
        self._clock += float(dt)

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        densities = [
            Density(self.lattice, self._perf_mixture(self._current(nm))) for nm in names
        ]
        prices = Race(densities).state_prices()
        return [float(p) for p in prices]

    def rating(self, name: str) -> Rating:
        w, slices = self._current(name)
        mean = 0.0
        second = 0.0
        for wi, b in zip(w, slices):
            m = float(np.dot(b, self._grid))
            v = float(np.dot(b, (self._grid - m) ** 2))
            mean += wi * m
            second += wi * (v + m * m)
        var = max(second - mean * mean, 0.0)
        return Rating(mu=-mean, sigma=math.sqrt(var))

    def learned_scale(self, name: str) -> float:
        """Posterior mean of the performance-noise scale."""
        w, _ = self._current(name)
        return float(np.dot(w, self.scales))

    def known(self) -> List[str]:
        return list(self._beliefs)

    # ---------------- update ----------------

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        validate_event(names, ranks)
        self.elapse(dt)
        for nm in names:
            self._register(nm)

        by_rank: Dict[int, List[str]] = {}
        for nm, rk in zip(names, ranks):
            by_rank.setdefault(rk, []).append(nm)
        groups = [sorted(by_rank[rk]) for rk in sorted(by_rank)]

        priors = {nm: self._beliefs[nm] for nm in names}
        working = {nm: (w.copy(), [s.copy() for s in slices]) for nm, (w, slices) in priors.items()}
        for sweep in range(self.iterations):
            stage = self._stage_functions(groups, working)
            new_working = {}
            for nm in names:
                pw, pslices = priors[nm]
                weights = np.zeros(len(self.scales))
                slices = []
                for si, (kern, prior_slice) in enumerate(zip(self._kernels, pslices)):
                    like = stage[nm](kern)
                    post = prior_slice * like
                    total = post.sum()
                    # marginal likelihood of this scale slice drives the
                    # weights — but only on the first sweep: later EP sweeps
                    # revisit the SAME event and would double-count its
                    # evidence, driving all weights toward the widest scale
                    weights[si] = (pw[si] * max(total, 1e-300)) if sweep == 0 else total
                    slices.append(post / total if total > 0 else prior_slice)
                if sweep == 0:
                    weights = weights / weights.sum()
                else:
                    weights = working[nm][0]
                new_working[nm] = (weights, slices)
            working = new_working
        self._beliefs.update(working)

    def _stage_functions(self, groups, beliefs):
        """For each contestant, a callable kernel -> event likelihood over
        ability, so each scale slice can be evaluated with its own kernel.
        Opponent marginals are their full (ability x scale) mixtures."""
        flat = [nm for g in groups for nm in g]
        perf = {nm: self._perf_mixture(beliefs[nm]) for nm in flat}
        cdf = {nm: np.cumsum(p) for nm, p in perf.items()}
        surv = {nm: 1.0 - c for nm, c in cdf.items()}

        parts: Dict[str, list] = {nm: [] for nm in flat}  # (kind, array) factors
        for s in range(len(groups) - 1):
            group = groups[s]
            rest = [nm for g in groups[s + 1 :] for nm in g]
            m = len(rest)
            pre = [np.ones_like(self._grid)]
            for t in range(m):
                pre.append(pre[t] * surv[rest[t]])
            suf = [np.ones_like(self._grid)] * (m + 1)
            suf[m] = np.ones_like(self._grid)
            for t in range(m - 1, -1, -1):
                suf[t] = suf[t + 1] * surv[rest[t]]

            for nm in group:
                parts[nm].append(("win", pre[m]))

            cdf_max = np.ones_like(self._grid)
            for nm in group:
                cdf_max = cdf_max * cdf[nm]
            pdf_max = np.diff(np.insert(cdf_max, 0, 0.0))
            for t, nm in enumerate(rest):
                parts[nm].append(("lose", np.cumsum(pdf_max * pre[t] * suf[t + 1])))

        def make(nm):
            factors = parts[nm]

            def event_like(kernel):
                loglik = np.zeros_like(self._grid)
                for kind, arr in factors:
                    like = _conv_same(arr, kernel)
                    loglik += np.log(np.maximum(like, 1e-300))
                out = np.exp(loglik - loglik.max())
                return out

            return event_like

        return {nm: make(nm) for nm in flat}


def main():
    import statistics

    from winning import ThurstoneRating
    from winning.benchmarks.events import synthetic_world
    from winning.benchmarks.forward_chain import evaluate

    print("Heteroskedastic world: 200 contestants, 3000 events, noise sd 0.6 or 1.6")
    events = synthetic_world(
        num_contestants=200, num_events=3000, noise_sigmas=(0.6, 1.6), seed=17
    )

    rows = []
    scale_learner = ScaleLearningThurstoneRating()
    for label, system in [
        ("fixed-scale ThurstoneRating", ThurstoneRating()),
        ("scale-learning (this file)", scale_learner),
    ]:
        m = evaluate(system, events)
        s = m.summary()
        rows.append((label, s))
        print(
            f"{label:30s} log_loss={s['log_loss']:.4f} tv={s['tv_vs_oracle']:.4f} "
            f"ece={s['ece']:.4f} sec={s['seconds']:.1f}"
        )
    m = evaluate(None, events, fixed="truth")
    s = m.summary()
    print(f"{'oracle (true ability+noise)':30s} log_loss={s['log_loss']:.4f} tv=0.0000")

    # did it identify who is erratic? reconstruct truth from the generator
    import random

    rng = random.Random(17)
    abilities = {f"c{i}": rng.gauss(0.0, 1.0) for i in range(200)}
    noise = {nm: rng.choice([0.6, 1.6]) for nm in abilities}
    seen = [nm for nm in scale_learner.known() if nm in noise]
    learned = [scale_learner.learned_scale(nm) for nm in seen]
    true = [noise[nm] for nm in seen]
    lo = [x for x, t in zip(learned, true) if t == 0.6]
    hi = [x for x, t in zip(learned, true) if t == 1.6]
    n = len(learned)
    mean_l, mean_t = sum(learned) / n, sum(true) / n
    cov = sum((x - mean_l) * (t - mean_t) for x, t in zip(learned, true)) / n
    corr = cov / (statistics.pstdev(learned) * statistics.pstdev(true))
    print()
    print(f"learned scale, steady group (true 0.6): mean {statistics.mean(lo):.3f}")
    print(f"learned scale, erratic group (true 1.6): mean {statistics.mean(hi):.3f}")
    print(f"correlation(learned, true) = {corr:.3f} over {n} contestants")


if __name__ == "__main__":
    main()
