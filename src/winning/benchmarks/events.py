"""Event containers and synthetic worlds with known ground truth."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class Event:
    names: List[str]
    ranks: List[int]  # 1-based finish positions, aligned with names
    dt: float = 1.0  # time since previous event on the shared clock
    truth: Optional[List[float]] = None  # oracle win probabilities, if known
    market: Optional[List[float]] = None  # market-implied win probabilities, if known


def synthetic_world(
    num_contestants: int = 200,
    num_events: int = 4000,
    field_size: Sequence[int] = (6, 12),
    ability_sigma: float = 1.0,
    drift_tau: float = 0.0,
    noise_sigmas: Optional[Sequence[float]] = None,
    seed: int = 17,
) -> List[Event]:
    """Races between contestants with latent Gaussian abilities.

    Performance = ability + noise; higher performance wins. Noise sd is 1 for
    everyone unless noise_sigmas gives choices to draw per contestant — the
    heteroskedastic world where steady and erratic contestants coexist, which
    every benchmarked system's common-noise assumption gets wrong to some
    degree. Abilities take a Gaussian random-walk step of sd drift_tau*sqrt(dt)
    between events. Oracle win probabilities (true abilities AND true noise)
    are attached to each event so benchmarks have an achievable floor.
    """
    from ..exact import gaussian_win_probabilities

    rng = random.Random(seed)
    abilities = {f"c{i}": rng.gauss(0.0, ability_sigma) for i in range(num_contestants)}
    noise = {
        nm: (rng.choice(list(noise_sigmas)) if noise_sigmas else 1.0) for nm in abilities
    }
    names_all = list(abilities)
    events: List[Event] = []
    lo, hi = min(field_size), max(field_size)
    for _ in range(num_events):
        n = rng.randint(lo, hi)
        field_names = rng.sample(names_all, n)
        if drift_tau > 0:
            for nm in names_all:
                abilities[nm] += rng.gauss(0.0, drift_tau)
        perf = {nm: abilities[nm] + rng.gauss(0.0, noise[nm]) for nm in field_names}
        order = sorted(field_names, key=lambda nm: -perf[nm])
        ranks = {nm: k + 1 for k, nm in enumerate(order)}
        truth = gaussian_win_probabilities(
            [abilities[nm] for nm in field_names],
            [noise[nm] for nm in field_names],
            beta=0.0,
        )
        events.append(
            Event(
                names=list(field_names),
                ranks=[ranks[nm] for nm in field_names],
                dt=1.0,
                truth=truth,
            )
        )
    return events
