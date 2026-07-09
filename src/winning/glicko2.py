"""Glicko-2, implemented from Glickman's specification (glicko.net/glicko/glicko2.pdf).

Each event a player appears in is treated as one rating period containing the
N-1 pairwise results implied by the finish order. Idle time inflates RD per the
spec: a player who has been away for `elapsed / period` rating periods has
phi* = sqrt(phi^2 + sigma^2 * elapsed/period), capped at the initial RD; time
advances via elapse(dt) or the dt argument of observe. Field win probabilities
use the exact lattice order statistics over each player's Gaussian belief, with
the logistic performance noise replaced by its moment-matched normal
(sd pi/sqrt(3) on the Glicko-2 internal scale).
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

from .exact import gaussian_win_probabilities
from .ratingsystem import Rating, RatingSystem, pairwise_scores, validate_event

GLICKO_SCALE = 173.7178
LOGISTIC_SD = math.pi / math.sqrt(3.0)  # normal moment-match of the standard logistic


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / (math.pi * math.pi))


def _expected(mu: float, mu_j: float, phi_j: float) -> float:
    x = -_g(phi_j) * (mu - mu_j)
    x = max(-40.0, min(40.0, x))  # extreme rating gaps would overflow exp
    return 1.0 / (1.0 + math.exp(x))


class Glicko2Rating(RatingSystem):
    def __init__(
        self,
        initial: float = 1500.0,
        initial_rd: float = 350.0,
        initial_vol: float = 0.06,
        tau: float = 0.5,
        period: float = 30.0,
    ):
        """period: time units (of dt) per Glicko-2 rating period, for idle
        RD inflation; with day-valued dt the default is monthly periods."""
        self.initial = float(initial)
        self.initial_rd = float(initial_rd)
        self.initial_vol = float(initial_vol)
        self.tau = float(tau)
        self.period = float(period)
        self._state: Dict[str, tuple] = {}  # name -> (rating, rd, vol)
        self._last_time: Dict[str, float] = {}
        self._clock: float = 0.0

    def elapse(self, dt: float) -> None:
        self._clock += float(dt)

    def _get(self, name: str) -> tuple:
        """State with RD inflated for time idle since last appearance."""
        if name not in self._state:
            return (self.initial, self.initial_rd, self.initial_vol)
        r, rd, vol = self._state[name]
        idle = (self._clock - self._last_time[name]) / self.period
        if idle > 0:
            phi = rd / GLICKO_SCALE
            phi = math.sqrt(phi * phi + vol * vol * idle)
            rd = min(phi * GLICKO_SCALE, self.initial_rd)
        return (r, rd, vol)

    def observe(self, names: Sequence[str], ranks: Sequence[int], dt: float = 1.0) -> None:
        validate_event(names, ranks)
        self.elapse(dt)
        pre = [self._get(nm) for nm in names]
        mus = [(r - self.initial) / GLICKO_SCALE for r, _, _ in pre]
        phis = [rd / GLICKO_SCALE for _, rd, _ in pre]
        vols = [v for _, _, v in pre]
        scores = pairwise_scores(ranks)

        for i, nm in enumerate(names):
            v_inv = 0.0
            delta_sum = 0.0
            for j, s in scores[i]:
                gj = _g(phis[j])
                e = _expected(mus[i], mus[j], phis[j])
                v_inv += gj * gj * e * (1.0 - e)
                delta_sum += gj * (s - e)
            if v_inv <= 0.0:
                self._state[nm] = pre[i]
                self._last_time[nm] = self._clock
                continue
            v = 1.0 / v_inv
            delta = v * delta_sum
            sigma_new = self._new_volatility(phis[i], v, delta, vols[i])
            # Guards beyond the spec: decomposing a large field floods one
            # rating period with N-1 results, which can run the volatility
            # and rating to numerical extremes. Cap volatility, keep RD at or
            # below its initial value, and keep ratings in a sane band.
            sigma_new = min(sigma_new, 0.5)
            phi_star = math.sqrt(phis[i] ** 2 + sigma_new**2)
            phi_new = 1.0 / math.sqrt(1.0 / (phi_star**2) + 1.0 / v)
            phi_new = min(phi_new, self.initial_rd / GLICKO_SCALE)
            mu_new = mus[i] + phi_new**2 * delta_sum
            mu_new = max(-10.0, min(10.0, mu_new))
            self._state[nm] = (
                mu_new * GLICKO_SCALE + self.initial,
                phi_new * GLICKO_SCALE,
                sigma_new,
            )
            self._last_time[nm] = self._clock

    def _new_volatility(self, phi: float, v: float, delta: float, sigma: float) -> float:
        """Illinois-algorithm iteration from the Glicko-2 spec (step 5)."""
        a = math.log(sigma * sigma)
        tau = self.tau
        eps = 1e-6

        def f(x: float) -> float:
            ex = math.exp(x)
            num = ex * (delta * delta - phi * phi - v - ex)
            den = 2.0 * (phi * phi + v + ex) ** 2
            return num / den - (x - a) / (tau * tau)

        A = a
        if delta * delta > phi * phi + v:
            B = math.log(delta * delta - phi * phi - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        fA, fB = f(A), f(B)
        while abs(B - A) > eps:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)
            if fC * fB <= 0:
                A, fA = B, fB
            else:
                fA = fA / 2.0
            B, fB = C, fC
        return math.exp(A / 2.0)

    def win_probabilities(self, names: Sequence[str]) -> List[float]:
        pre = [self._get(nm) for nm in names]
        mus = [(r - self.initial) / GLICKO_SCALE for r, _, _ in pre]
        sigmas = [rd / GLICKO_SCALE for _, rd, _ in pre]
        return gaussian_win_probabilities(mus, sigmas, beta=LOGISTIC_SD)

    def rating(self, name: str) -> Rating:
        r, rd, _ = self._get(name)
        return Rating(mu=r, sigma=rd)

    def performance_samples(self, names: Sequence[str], size: int = 32):
        import numpy as np

        from .thurstonerating import _stable_seed

        self._sample_calls = getattr(self, "_sample_calls", 0) + 1
        rng = np.random.default_rng(_stable_seed(names, self._sample_calls))
        cols = []
        for nm in names:
            r, rd, _ = self._get(nm)
            sd = math.sqrt(rd * rd + (LOGISTIC_SD * GLICKO_SCALE) ** 2)
            cols.append(rng.normal(r, sd, size=size))
        return np.column_stack(cols)

    def known(self) -> List[str]:
        return list(self._state)
