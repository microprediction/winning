"""Finishing-order conventions and converters.

Every ranked update in ``winning.ratings`` takes ``order``: the entrant
indices listed from first to last finisher, best first (max-wins). A
rank array, positions[i] = finishing position of entrant i, is the
inverse permutation. Both are valid permutations of 0..K-1, so no type,
length, range or uniqueness check can tell them apart. Passed as
``order``, a rank array is silently wrong at K >= 3 and produces a
plausible walk that learns much less; at K = 2 the two conventions
coincide, which is why the error survives pairwise validation.

    perf       [ 0.2  0.9 -0.4  0.5]     entrant 1 is best
    order      [1 3 0 2]                 first to last finisher: pass this
    positions  [2 0 3 1]                 position of entrant i: convert first

``np.argsort`` is the right conversion in every case; which sign to
sort by is the part callers get wrong, so the converters here name the
input instead.

Self-test for any ranked update: feed the same ``order`` thirty times
from a flat prior and the posterior means come out decreasing along
it. With ``update_ranking_exact`` and ``order = [0, 1, 2, 3, 4, 5]``
the means are about [3.52, 1.74, 0.54, -0.54, -1.74, -3.52]; a rank
array passed by mistake gives a non-monotone result at K >= 3.

Retired or censored entrants are not ties: omit them and pass the
partial order over the finishers (``update_ranking_exact`` marginalises
the rest). Dead heats are not a total order and every converter here
raises on them.
"""
from __future__ import annotations

import numpy as np

__all__ = ["order_from_positions", "order_from_performance",
           "order_from_times", "positions_from_order"]


def _distinct_finite(x, name):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a non-empty array of finite values")
    if np.unique(x).size != x.size:
        raise ValueError(f"{name} contain a tie; a finishing order needs distinct "
                         "values (omit non-finishers rather than tying them)")
    return x


def order_from_positions(positions):
    """Best-first ``order`` from finishing positions, positions[i] the
    position of entrant i with the winner smallest. Any strictly
    increasing code works: 0-based, 1-based, or points-style gaps."""
    p = _distinct_finite(positions, "positions")
    return np.argsort(p, kind="stable")


def order_from_performance(perf):
    """Best-first ``order`` from performances, higher is better
    (max-wins, the ratings convention)."""
    x = _distinct_finite(perf, "performances")
    return np.argsort(-x, kind="stable")


def order_from_times(times):
    """Best-first ``order`` from finishing times or any smaller-is-better
    score (min-wins, the race engine's convention)."""
    t = _distinct_finite(times, "times")
    return np.argsort(t, kind="stable")


def positions_from_order(order):
    """The inverse permutation: positions[order[k]] = k, so the winner
    has position 0."""
    o = np.asarray(order).ravel()
    if o.size == 0 or not np.issubdtype(o.dtype, np.integer):
        raise ValueError("order must be a non-empty array of integer entrant indices")
    if not np.array_equal(np.sort(o), np.arange(o.size)):
        raise ValueError("order must be a permutation of 0..K-1 (a partial order "
                         "has no positions for the omitted entrants)")
    pos = np.empty(o.size, dtype=int)
    pos[o] = np.arange(o.size)
    return pos
