from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .density import Density, _pdf_from_cdf
from .order_stats import expected_payoff_with_multiplicity
from .order_stats import winner_of_many as os_winner_of_many

# ---- Tie handling strategies ----


class TieModel:
    def draw_payoff(self, multiplicity_rest_at_k: float) -> float:
        raise NotImplementedError


class HalfPointTie(TieModel):
    def draw_payoff(self, multiplicity_rest_at_k: float) -> float:
        # split evenly (1/2) by default
        return 0.5


DEFAULT_TIE = HalfPointTie()

# ---- Core pricing primitives ----


def cdf_min(densities: Sequence[Density]) -> np.ndarray:
    """CDF of the minimum of independent contestants."""
    cdfs = [d.cdf() for d in densities]
    S = [1.0 - c for c in cdfs]
    prod_S = np.ones_like(cdfs[0])
    for s in S:
        prod_S *= s
    return 1.0 - prod_S


def winner_of_many(densities: Sequence[Density]) -> Density:
    """Return density of the minimum (winner) over the group."""
    lattice = densities[0].lattice
    c = cdf_min(densities)
    p = _pdf_from_cdf(c)
    return Density(lattice, p)


def rest_min_cdf(all_min_cdf: np.ndarray, self_cdf: np.ndarray) -> np.ndarray:
    """CDF of min of 'rest' contestants, computed from all vs self."""
    S_all = 1.0 - all_min_cdf
    S_self = 1.0 - self_cdf
    S_rest = (S_all + 1e-18) / (S_self + 1e-12)
    c_rest = 1.0 - S_rest
    return np.maximum.accumulate(np.minimum(c_rest, 1.0))


def conditional_win_draw_loss(
    self_d: Density, rest_cdf: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-lattice probabilities of win/draw/loss against 'rest min'."""
    cdfA = self_d.cdf()
    pdfA = self_d.p
    pdfRest = _pdf_from_cdf(rest_cdf)
    win = pdfA * (1.0 - rest_cdf)  # X < Y
    draw = pdfA * pdfRest  # X == Y
    loss = pdfRest * (1.0 - cdfA)  # Y < X
    return win, draw, loss


def expected_payoff_vs_rest(
    self_d: Density, all_min_cdf: np.ndarray, tie_model: TieModel = DEFAULT_TIE
) -> np.ndarray:
    rest_c = rest_min_cdf(all_min_cdf, self_d.cdf())
    win, draw, _ = conditional_win_draw_loss(self_d, rest_c)
    draw_payoff = tie_model.draw_payoff(multiplicity_rest_at_k=1.0)
    return win + draw * draw_payoff


@dataclass
class Race:
    densities: List[Density]
    tie_model: TieModel = DEFAULT_TIE

    def __post_init__(self):
        if len(self.densities) == 0:
            raise ValueError("Race requires at least one density.")
        L = self.densities[0].lattice.L
        unit = self.densities[0].lattice.unit
        for d in self.densities:
            if d.lattice.L != L or d.lattice.unit != unit:
                raise ValueError("All densities must share the same lattice.")

    def winner_density(self) -> Density:
        return winner_of_many(self.densities)

    def state_prices(self) -> np.ndarray:
        """Risk-neutral winning probabilities for each contestant (multiplicity-aware)."""
        densityAll, multAll = os_winner_of_many(self.densities)
        cdfAll = densityAll.cdf()
        prices = []
        for d in self.densities:
            ep = expected_payoff_with_multiplicity(d, densityAll, multAll, cdf=None, cdfAll=cdfAll)
            prices.append(float(np.sum(ep)))
        p = np.array(prices, dtype=float)
        S = p.sum()
        return p / S if S > 0 else p


class StatePricer:
    @staticmethod
    def prices_from_dividends(dividends: Sequence[float], nan_value: float = 2000.0) -> np.ndarray:
        inv = []
        for x in dividends:
            v = nan_value if (x is None or (isinstance(x, float) and np.isnan(x))) else x
            inv.append(0.0 if v <= 0 else 1.0 / float(v))
        p = np.array(inv, dtype=float)
        S = p.sum()
        return p / S if S > 0 else p

    @staticmethod
    def dividends_from_prices(prices: Sequence[float], multiplicity: float = 1.0) -> np.ndarray:
        p = np.array(prices, dtype=float)
        S = p.sum()
        p = p / S if S > 0 else p
        out = np.full_like(p, np.nan, dtype=float)
        mask = p > 0
        out[mask] = 1.0 / (multiplicity * p[mask])
        return out
