"""Multi-entrant Elo.

The classic system, extended to N-entrant events the standard way: an event
counts as N-1 pairwise games for each contestant with the K-factor scaled by
1/(N-1), and win probabilities for a field follow the Luce extension of the
Elo pairwise curve (softmax of rating/400 in base 10).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .ratingsystem import Rating, RatingSystem, pairwise_scores, validate_event


class EloRating(RatingSystem):
    def __init__(self, k: float = 32.0, scale: float = 400.0, initial: float = 1500.0):
        self.k = float(k)
        self.scale = float(scale)
        self.initial = float(initial)
        self._r: Dict[str, float] = {}

    def _get(self, name: str) -> float:
        return self._r.get(name, self.initial)

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        validate_event(names, ranks)
        n = len(names)
        r = [self._get(nm) for nm in names]
        scores = pairwise_scores(ranks)
        k_eff = self.k / (n - 1)
        deltas = [0.0] * n
        for i in range(n):
            for j, s in scores[i]:
                expected = 1.0 / (1.0 + 10.0 ** ((r[j] - r[i]) / self.scale))
                deltas[i] += k_eff * (s - expected)
        for nm, ri, di in zip(names, r, deltas):
            self._r[nm] = ri + di

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        strengths = [10.0 ** (self._get(nm) / self.scale) for nm in names]
        total = sum(strengths)
        return [s / total for s in strengths]

    def rating(self, name: str) -> Rating:
        return Rating(mu=self._get(name), sigma=None)

    def performance_samples(self, names: Sequence[str], size: int = 32):
        """Elo's pairwise curve is Bradley-Terry, i.e. Gumbel performances
        with scale 400/ln(10)."""
        import numpy as np

        from .thurstonerating import _stable_seed

        self._sample_calls = getattr(self, "_sample_calls", 0) + 1
        rng = np.random.default_rng(_stable_seed(names, self._sample_calls))
        scale = self.scale / math.log(10.0)
        return np.column_stack(
            [rng.gumbel(self._get(nm), scale, size=size) for nm in names]
        )

    def known(self) -> List[str]:
        return list(self._r)
