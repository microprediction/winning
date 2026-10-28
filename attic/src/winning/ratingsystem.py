"""The shared interface every rating system in this package satisfies.

An event is a contest between named contestants with an observed finish
order. Systems are used prequentially: ask for predictions first, then
show the system the outcome.

Conventions:
    - ranks are 1-based finish positions; 1 is the winner; equal ranks are ties
    - Rating.mu is "higher is better" for every system in this package
    - dt is elapsed time since the previous event on whatever clock the
      caller prefers (days, race indices, ...); systems with dynamics use it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass
class Rating:
    mu: float
    sigma: Optional[float] = None  # None for systems with no uncertainty (e.g. Elo)

    def ordinal(self, z: float = 3.0) -> float:
        """Conservative estimate mu - z*sigma, TrueSkill-style."""
        return self.mu if self.sigma is None else self.mu - z * self.sigma


class RatingSystem:
    """Observe finish orders; predict win probabilities."""

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        """Update ratings from one event. ranks[i] is the finish position of names[i]."""
        raise NotImplementedError

    def elapse(self, dt: float) -> None:
        """Advance the clock without an event, so predictions reflect elapsed
        time (uncertainty growth). No-op for systems without time dynamics."""

    def performance_samples(self, names: Sequence[str], size: int = 32):
        """Optional: a (size x len(names)) array of sampled performances
        (higher is better) from the predictive distribution, for
        distribution-level evaluation (rank-PIT). None if unsupported."""
        return None

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        """Probability each contestant wins an event among exactly these contestants."""
        raise NotImplementedError

    def rating(self, name: str) -> Rating:
        raise NotImplementedError

    def leaderboard(self) -> List[tuple]:
        """(name, Rating) pairs sorted best-first by conservative estimate."""
        names = self.known()
        pairs = [(n, self.rating(n)) for n in names]
        return sorted(pairs, key=lambda nr: -nr[1].ordinal())

    def known(self) -> List[str]:
        raise NotImplementedError


def validate_event(names: Sequence[str], ranks: Sequence[int]) -> None:
    if len(names) != len(ranks):
        raise ValueError("names and ranks must have equal length")
    if len(names) < 2:
        raise ValueError("an event needs at least two contestants")
    if len(set(names)) != len(names):
        raise ValueError("duplicate contestant names in one event")
    if min(ranks) < 1:
        raise ValueError("ranks are 1-based finish positions")


def pairwise_scores(ranks: Sequence[int]) -> Dict[int, List[tuple]]:
    """For each contestant index, list of (opponent_index, score) with score in {0, 0.5, 1}."""
    n = len(ranks)
    out: Dict[int, List[tuple]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ranks[i] < ranks[j]:
                s = 1.0
            elif ranks[i] > ranks[j]:
                s = 0.0
            else:
                s = 0.5
            out[i].append((j, s))
    return out
