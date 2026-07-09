"""The prequential loop: elapse time, predict, score, then observe."""

from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

from ..ratingsystem import RatingSystem
from .events import Event
from .metrics import Metrics


def evaluate(
    system: Optional[RatingSystem],
    events: Sequence[Event],
    warmup: float = 0.2,
    fixed: Optional[str] = None,
    rank_pit: bool = False,
    progress: Callable = None,
) -> Metrics:
    """Run system over events in order. Each event's dt is elapsed BEFORE
    prediction so beliefs are diffused to the event's own time; the first
    `warmup` fraction of events updates ratings without being scored.

    fixed="truth" or fixed="market" scores the events' attached probabilities
    (the oracle floor / the market ceiling) instead of a system's predictions
    (system may be None). Events lacking that attribute are not scored, so a
    fixed row is only directly comparable to system rows when the attribute is
    attached to (almost) every event — check the `events` count in summary()
    before comparing. All shipped datasets attach market/truth to every event
    or none.

    Events with a shared first place (a dead-heat winner) still update ratings
    but are not scored: the winner metrics need a unique winner.
    """
    m = Metrics()
    n_warm = int(len(events) * warmup)
    t0 = time.perf_counter()
    for idx, ev in enumerate(events):
        if system is not None:
            system.elapse(ev.dt)
        if idx >= n_warm and not _tied_first(ev.ranks):
            probs = mus = None
            if fixed is not None:
                attached = getattr(ev, fixed)
                if attached is not None:
                    probs = list(attached)
                    mus = list(attached)
            else:
                probs = system.win_probabilities(ev.names)
                mus = [system.rating(nm).mu for nm in ev.names]
            if probs is not None:
                m.score_event(probs, ev.ranks, mus, truth=ev.truth)
                if rank_pit and system is not None:
                    _accumulate_pit(m, system, ev)
        if system is not None:
            system.observe(ev.names, ev.ranks, dt=0.0)
        if progress and idx % 500 == 0:
            progress(idx, len(events))
    m.seconds = time.perf_counter() - t0
    return m


def _tied_first(ranks: Sequence[int]) -> bool:
    best = min(ranks)
    return sum(1 for r in ranks if r == best) > 1


def _accumulate_pit(m: Metrics, system: RatingSystem, ev: Event, size: int = 32) -> None:
    """Randomized rank-PIT: sample fields from the system's predictive
    performance distributions, and place each observed finish position within
    its simulated distribution. Uniform PITs = variation modeled correctly."""
    import random

    import numpy as np

    samples = system.performance_samples(ev.names, size=size)
    if samples is None:
        return
    samples = np.asarray(samples, dtype=float)
    sim_ranks = (-samples).argsort(axis=1).argsort(axis=1) + 1  # per-sample positions
    rng = random.Random(len(m.pits) * 7919 + len(ev.names))
    for i, observed in enumerate(ev.ranks):
        lo = float(np.mean(sim_ranks[:, i] <= observed - 1))
        hi = float(np.mean(sim_ranks[:, i] <= observed))
        m.pits.append(rng.uniform(lo, hi) if hi > lo else hi)
