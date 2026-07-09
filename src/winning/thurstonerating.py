"""ThurstoneRating: an outcome-driven Thurstonian rating system on the lattice.

Each contestant's belief over latent ability is a full density on a shared
lattice (no Gaussian restriction). Performance in an event is ability plus
standard performance noise (the thurstone base density, N(0,1) by default,
time-like: lowest performance wins).

Update. An observed finish order is decomposed by Plackett peeling over rank
groups: P(order) = prod_s P(group s ahead of everyone slower). Each stage
factor, viewed as a likelihood in one contestant's ability with opponents at
their current predictive performance marginals, is computed exactly on the
lattice (group members: base correlated with the product of the slower field's
survivals; the slower field via a prefix-sum identity against the slowest
group member's density, which keeps kernels compact). Contestants sharing a
rank are a dead-heat group: they update against the slower field only, never
against each other, so updates are permutation-invariant. Beliefs multiply by
their accumulated stage likelihoods and renormalize. This is an iterated
approximation in the same spirit as TrueSkill's EP, but each factor is an
exact joint order statistic over the whole remaining field rather than a
pairwise/adjacent Gaussian approximation, and posteriors keep their (skewed,
non-Gaussian) shapes. A single two-contestant update matches exact Bayes to
lattice precision (see tests).

Dynamics. Between a contestant's events, ability diffuses: the belief is
convolved with N(0, tau^2 * dt). Time advances via elapse(dt) (or the dt
argument of observe); win_probabilities and rating are read-only and price
beliefs diffused to the current clock without registering unknown names.

Prediction. Field win probabilities come from thurstone's Race.state_prices()
on the predictive performance densities (belief convolved with base): exact,
dead-heat aware, O(N) in field size.

Ratings are exposed as "higher is better" (the internal time-like sign is
flipped at the boundary).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from thurstone import Density, Race, UniformLattice

from .ratingsystem import Rating, RatingSystem, validate_event

_SUPPORT_Z = 6.0


def _gauss_kernel(unit: float, sd: float) -> np.ndarray:
    half = max(1, int(math.ceil(_SUPPORT_Z * sd / unit)))
    x = np.arange(-half, half + 1) * unit
    k = np.exp(-0.5 * (x / sd) ** 2)
    return k / k.sum()


def _conv_same(
    p: np.ndarray, kernel: np.ndarray, left: float = 0.0, right: float = 0.0
) -> np.ndarray:
    """Center-cropped convolution that always returns len(p), even when the
    kernel is longer than the grid. left/right give the constant continuation
    of p beyond the lattice: 0 for densities, but survival products continue
    as 1 on the left and prefix sums saturate at their last value on the
    right — zero-padding those would crush likelihoods near the edges."""
    half = (len(kernel) - 1) // 2
    padded = np.concatenate([np.full(half, left), p, np.full(half, right)])
    full = np.convolve(padded, kernel)
    start = 2 * half
    return full[start : start + len(p)]


def _stable_seed(names, salt: int) -> int:
    import zlib

    return zlib.crc32("|".join(names).encode()) ^ (salt & 0xFFFFFFFF)


class ThurstoneRating(RatingSystem):
    def __init__(
        self,
        prior_sigma: float = 1.0,
        beta: float = 1.0,
        tau: float = 0.02,
        unit: float = 0.1,
        L: int = 150,
        iterations: int = 2,
        base_kernel: "np.ndarray" = None,
    ):
        """prior_sigma: sd of the ability prior; beta: performance noise sd;
        tau: ability diffusion sd per unit time; unit/L: lattice geometry
        (span +/- L*unit must comfortably cover ability spread plus drift);
        iterations: EP-style sweeps per event (opponent marginals recomputed
        from partially-updated beliefs on each sweep).

        base_kernel: optional odd-length array replacing the N(0, beta)
        performance noise — ANY distribution on the lattice step size works
        (skewed, heavy-tailed, a DNF mixture...). Time-like convention: mass
        at positive offsets means slower/worse performance."""
        self.prior_sigma = float(prior_sigma)
        self.beta = float(beta)
        self.tau = float(tau)
        self.iterations = max(1, int(iterations))
        self.lattice = UniformLattice(L=int(L), unit=float(unit))
        self._grid = self.lattice.grid
        if base_kernel is not None:
            k = np.asarray(base_kernel, dtype=float)
            if len(k) % 2 == 0 or k.min() < 0 or k.sum() <= 0:
                raise ValueError("base_kernel must be odd-length, non-negative, with mass")
            self._base_kernel = k / k.sum()
        else:
            self._base_kernel = _gauss_kernel(unit, self.beta)
        self._beliefs: Dict[str, np.ndarray] = {}
        self._last_time: Dict[str, float] = {}
        self._clock: float = 0.0

    # ---------------- belief plumbing ----------------

    def _prior(self) -> np.ndarray:
        p = np.exp(-0.5 * (self._grid / self.prior_sigma) ** 2)
        return p / p.sum()

    def _diffused(self, belief: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0 or self.tau <= 0:
            return belief
        k = _gauss_kernel(self.lattice.unit, self.tau * math.sqrt(dt))
        b = _conv_same(belief, k)
        s = b.sum()
        return b / s if s > 0 else self._prior()

    def _current(self, name: str) -> np.ndarray:
        """Belief diffused to the current clock; read-only, unknown -> prior."""
        b = self._beliefs.get(name)
        if b is None:
            return self._prior()
        return self._diffused(b, self._clock - self._last_time[name])

    def _register(self, name: str) -> None:
        """Store the up-to-date belief for a participant (mutating)."""
        self._beliefs[name] = self._current(name)
        self._last_time[name] = self._clock

    def _perf_pdf(self, belief: np.ndarray) -> np.ndarray:
        p = _conv_same(belief, self._base_kernel)
        s = p.sum()
        return p / s if s > 0 else p

    # ---------------- prediction ----------------

    def elapse(self, dt: float) -> None:
        self._clock += float(dt)

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        densities = [Density(self.lattice, self._perf_pdf(self._current(nm))) for nm in names]
        prices = Race(densities).state_prices()
        return [float(p) for p in prices]

    def performance_samples(self, names: Sequence[str], size: int = 32) -> np.ndarray:
        self._sample_calls = getattr(self, "_sample_calls", 0) + 1
        rng = np.random.default_rng(_stable_seed(names, self._sample_calls))
        cols = []
        unit = self.lattice.unit
        for nm in names:
            p = self._perf_pdf(self._current(nm))
            total = p.sum()
            p = p / total if total > 0 else np.full_like(p, 1.0 / len(p))
            x = rng.choice(self._grid, size=size, p=p)
            x = x + rng.uniform(-unit / 2, unit / 2, size=size)
            cols.append(-x)  # internal is time-like; expose higher-is-better
        return np.column_stack(cols)

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

        participants = list(names)
        priors = {nm: self._beliefs[nm] for nm in participants}
        working = dict(priors)
        for _ in range(self.iterations):
            loglik = self._event_loglik(groups, working)
            working = {}
            for nm in participants:
                like = np.exp(loglik[nm] - loglik[nm].max())
                post = priors[nm] * like
                total = post.sum()
                working[nm] = post / total if total > 0 else priors[nm]
        self._beliefs.update(working)

    def _event_loglik(self, groups: List[List[str]], beliefs) -> Dict[str, np.ndarray]:
        flat = [nm for g in groups for nm in g]
        perf = {nm: self._perf_pdf(beliefs[nm]) for nm in flat}
        cdf = {nm: np.cumsum(p) for nm, p in perf.items()}
        surv = {nm: 1.0 - c for nm, c in cdf.items()}

        loglik = {nm: np.zeros_like(self._grid) for nm in flat}
        # the stage likelihoods are CORRELATIONS with the base density; for a
        # symmetric (Gaussian) base that equals convolution, but any skewed or
        # heavy-tailed base_kernel must be flipped here
        kernel = self._base_kernel[::-1]
        for s in range(len(groups) - 1):
            group = groups[s]
            rest = [nm for g in groups[s + 1 :] for nm in g]
            # prefix/suffix survival products over the rest, for leave-one-out
            m = len(rest)
            pre = [None] * (m + 1)
            pre[0] = np.ones_like(self._grid)
            for t in range(m):
                pre[t + 1] = pre[t] * surv[rest[t]]
            suf = [None] * (m + 1)
            suf[m] = np.ones_like(self._grid)
            for t in range(m - 1, -1, -1):
                suf[t] = suf[t + 1] * surv[rest[t]]

            # group member g: L(a) = sum_x base(x-a) * prod_rest surv(x),
            # a correlation, computed as convolution with the flipped kernel.
            # Tied members carry no information about each other.
            lw = _conv_same(pre[m], kernel, left=1.0, right=0.0)
            for nm in group:
                self._accumulate(loglik, nm, lw)

            # slower contestant j: the whole group finished ahead, i.e. the
            # slowest group member beat j. With f_max the density of the
            # group's max (time-like) and h_j = f_max * prod_{k in rest, k!=j}
            # surv_k, writing P(eps > x - a) = sum_u base(u) 1[x <= a + u]
            # turns L(a) = sum_x h_j(x) P(X_j > x | a) into a correlation of
            # the prefix sum of h_j with the base kernel.
            # (a one-lattice-cell tie asymmetry remains between the strict
            # rest-survival above and the inclusive prefix sum below; it is
            # O(unit/beta) and negligible at the default resolution)
            cdf_max = np.ones_like(self._grid)
            for nm in group:
                cdf_max = cdf_max * cdf[nm]
            pdf_max = np.diff(np.insert(cdf_max, 0, 0.0))
            for t, nm in enumerate(rest):
                h = pdf_max * pre[t] * suf[t + 1]
                ch = np.cumsum(h)
                lj = _conv_same(ch, kernel, left=0.0, right=float(ch[-1]))
                self._accumulate(loglik, nm, lj)
        return loglik

    def _accumulate(self, loglik: Dict[str, np.ndarray], name: str, like: np.ndarray) -> None:
        like = np.maximum(like, 1e-300)
        loglik[name] += np.log(like)

    # ---------------- reporting ----------------

    def rating(self, name: str) -> Rating:
        b = self._current(name)
        mean = float(np.dot(b, self._grid))
        var = float(np.dot(b, (self._grid - mean) ** 2))
        return Rating(mu=-mean, sigma=math.sqrt(max(var, 0.0)))

    def known(self) -> List[str]:
        return list(self._beliefs)
