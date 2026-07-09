"""Thin shims wrapping third-party rating packages into the RatingSystem
interface, for benchmarking. Requires the `benchmarks` extra:

    pip install winning[benchmarks]

TrueSkill note: the algorithm is patented by Microsoft and the trademarked
system is licensed for non-commercial use; it appears here strictly as a
research comparator.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .exact import gaussian_win_probabilities
from .ratingsystem import Rating, RatingSystem, validate_event


class TrueSkillRating(RatingSystem):
    """Wraps the reference `trueskill` package (one-player teams).

    Win probabilities are computed from the (mu, sigma) beliefs by exact
    lattice order statistics with the environment's beta, since the package
    itself offers no field win probability.
    """

    def __init__(self, draw_probability: float = 0.0, **env_kwargs):
        import trueskill

        self._env = trueskill.TrueSkill(draw_probability=draw_probability, **env_kwargs)
        self._r: Dict[str, object] = {}

    def _get(self, name: str):
        if name not in self._r:
            self._r[name] = self._env.create_rating()
        return self._r[name]

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        validate_event(names, ranks)
        groups = [(self._get(nm),) for nm in names]
        new_groups = self._env.rate(groups, ranks=[r - 1 for r in ranks])
        for nm, grp in zip(names, new_groups):
            self._r[nm] = grp[0]

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        mus = [self._get(nm).mu for nm in names]
        sigmas = [self._get(nm).sigma for nm in names]
        return gaussian_win_probabilities(mus, sigmas, beta=self._env.beta)

    def rating(self, name: str) -> Rating:
        r = self._get(name)
        return Rating(mu=r.mu, sigma=r.sigma)

    def performance_samples(self, names: Sequence[str], size: int = 32):
        self._sample_calls = getattr(self, "_sample_calls", 0) + 1
        return _gaussian_samples(
            [(self._get(nm).mu, self._get(nm).sigma) for nm in names],
            self._env.beta,
            size,
            (tuple(names), self._sample_calls),
        )

    def known(self) -> List[str]:
        return list(self._r)


class OpenSkillRating(RatingSystem):
    """Wraps `openskill` (Weng-Lin) models: PlackettLuce (default),
    ThurstoneMostellerFull, BradleyTerryFull, etc. Uses the model's native
    predict_win; pass exact_predict=True to use lattice order statistics on
    the (mu, sigma) beliefs instead.
    """

    def __init__(self, model: str = "PlackettLuce", exact_predict: bool = False, **model_kwargs):
        import openskill.models as osm

        self._model = getattr(osm, model)(**model_kwargs)
        self._exact = bool(exact_predict)
        self._name = model
        self._r: Dict[str, object] = {}

    def _get(self, name: str):
        if name not in self._r:
            self._r[name] = self._model.rating(name=name)
        return self._r[name]

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        validate_event(names, ranks)
        teams = [[self._get(nm)] for nm in names]
        new_teams = self._model.rate(teams, ranks=list(ranks))
        for nm, team in zip(names, new_teams):
            self._r[nm] = team[0]

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        if self._exact:
            mus = [self._get(nm).mu for nm in names]
            sigmas = [self._get(nm).sigma for nm in names]
            return gaussian_win_probabilities(mus, sigmas, beta=self._model.beta)
        teams = [[self._get(nm)] for nm in names]
        return [float(p) for p in self._model.predict_win(teams)]

    def rating(self, name: str) -> Rating:
        r = self._get(name)
        return Rating(mu=r.mu, sigma=r.sigma)

    def performance_samples(self, names: Sequence[str], size: int = 32):
        self._sample_calls = getattr(self, "_sample_calls", 0) + 1
        return _gaussian_samples(
            [(self._get(nm).mu, self._get(nm).sigma) for nm in names],
            self._model.beta,
            size,
            (tuple(names), self._sample_calls),
        )

    def known(self) -> List[str]:
        return list(self._r)


def _gaussian_samples(mu_sigma, beta: float, size: int, seed_parts=None):
    import math

    import numpy as np

    from .thurstonerating import _stable_seed

    names, salt = seed_parts if seed_parts else ((), 0)
    rng = np.random.default_rng(_stable_seed([str(n) for n in names], salt))
    return np.column_stack(
        [
            rng.normal(mu, math.sqrt(sig * sig + beta * beta), size=size)
            for mu, sig in mu_sigma
        ]
    )
