"""Exact field win probabilities for Gaussian beliefs, via the thurstone lattice.

Given each contestant's belief N(mu_i, sigma_i^2) over ability (higher is
better) and common performance noise sd beta, the probability that contestant i
records the best performance in the field is a joint order statistic of N
non-identical Gaussians. TrueSkill and friends approximate related quantities
pairwise; the thurstone lattice computes the winner probabilities directly
(dead-heat aware). Any Gaussian-belief rating system can therefore share this
one prediction routine, which separates "quality of the belief update" from
"quality of the prediction formula" in benchmarks.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
from thurstone import Density, Race, UniformLattice

_GRID_POINTS_PER_BETA = 10  # lattice unit = beta / 10
_SUPPORT_Z = 6.0


def _normal_density(lattice: UniformLattice, loc: float, scale: float) -> Density:
    x = lattice.grid
    z = (x - loc) / scale
    p = np.exp(-0.5 * z * z)
    return Density(lattice, p)


def gaussian_win_probabilities(
    mus: Sequence[float], sigmas: Sequence[float], beta: float = 1.0
) -> List[float]:
    """P(win) for each contestant with performance ~ N(mu_i, sigma_i^2 + beta^2).

    mus are "higher is better"; sign is flipped internally onto the thurstone
    (time-like, lowest wins) convention.
    """
    if len(mus) != len(sigmas):
        raise ValueError("mus and sigmas must have equal length")
    if beta < 0 or any(s < 0 for s in sigmas):
        raise ValueError("beta and sigmas must be non-negative")
    if len(mus) == 1:
        return [1.0]
    if len(mus) == 2:
        # closed form: P(A) = Phi((muA - muB)/sqrt(sA^2 + sB^2 + 2 beta^2));
        # ~500x faster than the lattice and free of discretization error
        denom = math.sqrt(sigmas[0] ** 2 + sigmas[1] ** 2 + 2.0 * beta * beta)
        if denom == 0.0:
            p = 0.5 if mus[0] == mus[1] else (1.0 if mus[0] > mus[1] else 0.0)
        else:
            p = 0.5 * (1.0 + math.erf((mus[0] - mus[1]) / (denom * math.sqrt(2.0))))
        return [p, 1.0 - p]
    center = float(np.mean(mus))
    locs = [-(m - center) for m in mus]  # thurstone convention: lower is better
    scales = [math.sqrt(s * s + beta * beta) for s in sigmas]
    if min(scales) <= 0:
        raise ValueError(
            "every contestant needs positive total sd (sigma^2 + beta^2) in fields of 3+"
        )
    # resolution must resolve the SHARPEST density: one high-variance
    # contestant must not coarsen everyone else's grid
    reach = max(abs(l) + _SUPPORT_Z * s for l, s in zip(locs, scales))
    unit = min(scales) / _GRID_POINTS_PER_BETA
    unit = max(unit, reach / 500_000)  # cost cap: at most ~a million points
    L = int(math.ceil(reach / unit)) + 2
    if L > 2_000_000:
        raise ValueError(
            f"lattice would need {L} points; the mu spread ({reach:.3g}) is "
            "enormous relative to the noise scale — have the ratings diverged?"
        )
    lattice = UniformLattice(L=L, unit=unit)
    densities = [_normal_density(lattice, loc, s) for loc, s in zip(locs, scales)]
    prices = Race(densities).state_prices()
    return [float(p) for p in prices]
