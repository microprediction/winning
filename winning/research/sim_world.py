from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .density import Density
from .dynamic import RaceObservation
from .inference import AbilityCalibrator
from .lattice import UniformLattice


def sigma_true(dt: float, alpha: float = 0.04) -> float:
    """Ground‑truth stickiness: random‑walk std scales like sqrt(alpha * dt)."""
    return math.sqrt(max(alpha * max(dt, 0.0), 1e-12))


def simulate_schedule(
    rng: np.random.Generator,
    n_horses: int,
    n_races: int,
    race_size_range: Tuple[int, int],
    horizon_days: float,
) -> Tuple[List[float], List[List[int]]]:
    """Return sorted race times and a list of participant index sets per race."""
    times = np.sort(rng.uniform(0.0, horizon_days, size=n_races)).tolist()
    fields: List[List[int]] = []
    lo, hi = race_size_range
    for _ in range(n_races):
        m = int(rng.integers(low=lo, high=hi + 1))
        m = max(2, min(m, n_horses))
        field = rng.choice(n_horses, size=m, replace=False).tolist()
        fields.append(field)
    return times, fields


def simulate_world(
    rng: np.random.Generator,
    n_horses: int = 60,
    n_races: int = 90,
    race_size_range: Tuple[int, int] = (8, 14),
    horizon_days: float = 240.0,
    alpha: float = 0.04,  # RW diffusion scale in σ_true
    sigma0: float = 0.8,  # initial ability std
    bookmaker_rel_tau: float = 0.5,  # per-horse ability noise std (relative skill)
    bookmaker_bias_tau: float = 0.0,  # race-level bias std (relative race strength)
) -> Tuple[
    List[RaceObservation],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    callable,
]:
    """
    Simulate a dataset:
      - returns races with observed (bookmaker) prices,
      - true per‑horse abilities and times at their races,
      - the bookmaker's noisy ability samples (μ̂) and times,
      - reference to sigma_true(dt).
    """
    times, fields = simulate_schedule(
        rng,
        n_horses=n_horses,
        n_races=n_races,
        race_size_range=race_size_range,
        horizon_days=horizon_days,
    )
    horse_ids = [f"H{i}" for i in range(n_horses)]

    # Build per‑horse race index lists
    idx_per_horse: Dict[int, List[int]] = {h: [] for h in range(n_horses)}
    for r_i, fld in enumerate(fields):
        for h in fld:
            idx_per_horse[h].append(r_i)
    for h in range(n_horses):
        idx_per_horse[h].sort(key=lambda r_i: times[r_i])

    # Simulate true abilities μ at race times
    mu_at_race: Dict[Tuple[int, int], float] = {}
    true_theta: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    true_times: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    book_theta: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    book_times: Dict[str, List[float]] = {hid: [] for hid in horse_ids}
    for h in range(n_horses):
        mu = float(rng.normal(0.0, sigma0))
        prev_t: Optional[float] = None
        for r_i in idx_per_horse[h]:
            t = times[r_i]
            if prev_t is not None:
                dt = t - prev_t
                mu += float(rng.normal(0.0, sigma_true(dt, alpha=alpha)))
            prev_t = t
            mu_at_race[(h, r_i)] = mu
            hid = horse_ids[h]
            true_theta[hid].append(mu)
            true_times[hid].append(t)

    # Forward model (also used by bookmaker)
    lattice = UniformLattice(L=400, unit=0.1)
    base = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)
    forward = AbilityCalibrator(base, n_iter=3)

    # Build observed bookmaker prices from noisy abilities:
    races: List[RaceObservation] = []
    for r_i, fld in enumerate(fields):
        ids = [horse_ids[h] for h in fld]
        mu_true = np.array([mu_at_race[(h, r_i)] for h in fld], dtype=float)
        # true probabilities (fair)
        p_true = np.asarray(forward.state_prices_from_ability(mu_true.tolist()), dtype=float)
        # bookmaker noisy ability view:
        #   - relative per-horse noise
        #   - plus a race-level bias (same shift added to all entrants)
        race_bias = float(rng.normal(0.0, bookmaker_bias_tau))
        mu_hat = mu_true + rng.normal(0.0, bookmaker_rel_tau, size=len(fld)) + race_bias
        p_obs = np.asarray(forward.state_prices_from_ability(mu_hat.tolist()), dtype=float)
        winner_idx = int(rng.choice(len(fld), p=p_true / max(np.sum(p_true), 1e-12)))
        winner_id = ids[winner_idx]
        # store bookmaker noisy ability samples at each race time
        for hid, mhat in zip(ids, mu_hat.tolist()):
            book_theta[hid].append(float(mhat))
            book_times[hid].append(times[r_i])
        races.append(
            RaceObservation(
                race_id=f"R{r_i}",
                time=times[r_i],
                horse_ids=ids,
                prices=p_obs,
                winner=winner_id,
            )
        )

    true_theta_np = {h: np.asarray(v, dtype=float) for h, v in true_theta.items()}
    true_times_np = {h: np.asarray(v, dtype=float) for h, v in true_times.items()}
    book_theta_np = {h: np.asarray(v, dtype=float) for h, v in book_theta.items()}
    book_times_np = {h: np.asarray(v, dtype=float) for h, v in book_times.items()}
    return (
        races,
        true_theta_np,
        true_times_np,
        book_theta_np,
        book_times_np,
        (lambda dt: sigma_true(dt, alpha=alpha)),
    )
