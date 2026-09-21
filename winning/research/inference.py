from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clustering import ClusterSplitter
from .density import Density
from .order_stats import expected_payoff_with_multiplicity, winner_of_many
from .pricing import Race, StatePricer

# ---- Densities from offsets ----


def densities_from_offsets(base: Density, offsets: Sequence[float]) -> List[Density]:
    out = []
    for o in offsets:
        out.append(base.shift_fractional(o))
    return out


def state_prices_from_densities(densities: Sequence[Density]) -> List[float]:
    race = Race(list(densities))
    return list(race.state_prices())


# ---- Implicit map: offset -> price ----


def implicit_state_prices(base: Density, offsets: Sequence[float]) -> List[float]:
    dens = densities_from_offsets(base, offsets)
    return state_prices_from_densities(dens)


# ---- Calibration (inverse) ----


@dataclass
class AbilityCalibrator:
    base: Density
    offset_grid: Optional[Sequence[float]] = None
    n_iter: int = 3
    # Optional per-runner scale handling for 2D calibration (loc, scale)
    scales: Optional[np.ndarray] = field(default=None, repr=False)
    scale_span: float = 0.5  # +/- around each scale value (physical units of scale)
    scale_steps: int = 3  # number of scale samples per runner
    loc_span: float = 5.0  # +/- location range (physical units) for 2D path
    loc_step: float = 0.25  # step for location grid (physical units)
    skew_a: float = 0.0  # skew-normal 'a' used in density_for
    # Cached lookup curves for reuse (global calibration, multi-race)
    # 1D: single-field curve (loc -> price) and inverse (price -> offset)
    lookup_curve_1d_prices: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    lookup_curve_1d_inverse: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    # 2D: per-scale curves; keys are scale (float)
    lookup_curves_2d_prices: Dict[float, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    lookup_curves_2d_inverse: Dict[float, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self):
        if self.offset_grid is None:
            L = self.base.lattice.L
            self.offset_grid = list(range(int(-L / 2), int(L / 2)))[::-1]

    def set_scales(self, scales: Sequence[float]) -> None:
        self.scales = np.asarray(scales, dtype=float)

    def density_for(self, loc: float, scale: float) -> Density:
        lat = self.base.lattice
        return Density.skew_normal(lat, loc=loc, scale=scale, a=self.skew_a)

    # ---- Curve rebuilders (no inversion) for global fitting ----
    def rebuild_curves_from_field_1d(self, locs: Sequence[float]) -> None:
        """Rebuild 1D lookup curves (loc -> price, and price -> offset) given current field locs."""
        unit = self.base.lattice.unit
        # Convert physical locs to lattice steps for density shifting
        offsets = np.asarray(locs, dtype=float) / unit
        grid = list(self.offset_grid)
        # Field distribution with multiplicity
        current_densities = densities_from_offsets(self.base, offsets)
        densityAll, multAll = winner_of_many(current_densities)
        cdfAll = densityAll.cdf()
        # Build global implicit curve p(g)
        implied_prices = []
        for g in grid:
            d_g = self.base.shift_fractional(float(g))
            payoff_vec = expected_payoff_with_multiplicity(
                d_g, densityAll, multAll, cdf=None, cdfAll=cdfAll
            )
            implied_prices.append(float(np.sum(payoff_vec)))
        implied_prices = np.array(implied_prices, dtype=float)
        # Cache price curve in physical units (ascending by loc)
        locs_phys = unit * np.array(grid, dtype=float)
        order_loc = np.argsort(locs_phys)
        self.lookup_curve_1d_prices = {
            "locs": locs_phys[order_loc],
            "prices": implied_prices[order_loc],
        }
        # Cache inverse curve (price -> offset steps), monotone in price
        pairs = sorted(zip(implied_prices, grid), key=lambda t: t[0])
        xp = np.array([t[0] for t in pairs], dtype=float)
        fp = np.array([t[1] for t in pairs], dtype=float)
        xp_unique, idx = np.unique(xp, return_index=True)
        fp_unique = fp[idx]
        self.lookup_curve_1d_inverse = {
            "prices": xp_unique,
            "offsets": fp_unique,
        }

    def rebuild_curves_from_field_2d(self, locs: Sequence[float], scales: Sequence[float]) -> None:
        """Rebuild 2D lookup curves (per-scale loc -> price and price -> loc) given current field (locs, scales)."""
        locs_arr = np.asarray(locs, dtype=float)
        scales_arr = np.asarray(scales, dtype=float)
        n = len(locs_arr)
        # Build current field densities
        current_densities = [
            self.density_for(float(locs_arr[j]), float(scales_arr[j])) for j in range(n)
        ]
        densityAll, multAll = winner_of_many(current_densities)
        cdfAll = densityAll.cdf()
        # Location grid (physical units)
        loc_grid = np.arange(
            -self.loc_span, self.loc_span + self.loc_step, self.loc_step, dtype=float
        )
        # Use unique scales present
        unique_s_values = sorted(set(float(s) for s in scales_arr.tolist()))
        self.lookup_curves_2d_prices.clear()
        self.lookup_curves_2d_inverse.clear()
        for s in unique_s_values:
            p_curve = []
            for loc_candidate in loc_grid:
                d_i_candidate = self.density_for(float(loc_candidate), float(s))
                payoff_vec = expected_payoff_with_multiplicity(
                    d_i_candidate, densityAll, multAll, cdf=None, cdfAll=cdfAll
                )
                p_curve.append(float(np.sum(payoff_vec)))
            p_curve = np.array(p_curve, dtype=float)
            # Cache price curve per scale (loc -> price), sorted by loc
            order_loc = np.argsort(loc_grid)
            locs_sorted = loc_grid[order_loc]
            prices_sorted = p_curve[order_loc]
            self.lookup_curves_2d_prices[float(s)] = (locs_sorted, prices_sorted)
            # Cache inverse (price -> loc)
            pairs = sorted(zip(p_curve, loc_grid), key=lambda t: t[0])
            xp = np.array([pp for pp, _ in pairs], dtype=float)
            fp = np.array([ll for _, ll in pairs], dtype=float)
            xp_unique, idx = np.unique(xp, return_index=True)
            fp_unique = fp[idx]
            self.lookup_curves_2d_inverse[float(s)] = (xp_unique, fp_unique)

    def solve_from_prices(
        self,
        prices: Sequence[float],
        *,
        initial_offsets: Optional[Sequence[float]] = None,
    ) -> List[float]:
        prices_arr = np.asarray(prices, dtype=float)
        n = len(prices_arr)
        # 2D path if scales provided and length matches; else 1D fallback
        if self.scales is None or len(self.scales) != n:
            # ---- 1D global-curve inversion against current field (winning-style) ----
            if initial_offsets is None:
                initial_offsets = [0.0 for _ in prices_arr]
            unit = self.base.lattice.unit
            offsets = np.array(initial_offsets, dtype=float) / unit
            grid = list(self.offset_grid)
            # Ensure ascending xp for np.interp later (we sort by price anyway)
            for _ in range(self.n_iter):
                # Build densities from current offsets, compute field w/ multiplicity
                current_densities = densities_from_offsets(self.base, offsets)
                densityAll, multAll = winner_of_many(current_densities)
                # Build a single implicit curve p(g) for a representative runner vs field
                implied_prices = []
                for g in grid:
                    d_g = self.base.shift_fractional(float(g))
                    payoff_vec = expected_payoff_with_multiplicity(
                        d_g, densityAll, multAll, cdf=None, cdfAll=densityAll.cdf()
                    )
                    implied_prices.append(float(np.sum(payoff_vec)))
                implied_prices = np.array(implied_prices, dtype=float)
                # Cache price curve in physical units (loc -> price), sorted by loc ascending
                unit = self.base.lattice.unit
                locs_phys = unit * np.array(grid, dtype=float)
                order_loc = np.argsort(locs_phys)
                self.lookup_curve_1d_prices = {
                    "locs": locs_phys[order_loc],
                    "prices": implied_prices[order_loc],
                }
                # Sort by price to enforce monotonic xp, unique keys
                pairs = sorted(zip(implied_prices, grid), key=lambda t: t[0])
                xp = np.array([t[0] for t in pairs], dtype=float)
                fp = np.array([t[1] for t in pairs], dtype=float)
                xp_unique, idx = np.unique(xp, return_index=True)
                fp_unique = fp[idx]
                # Cache inverse curve (price -> offset in steps)
                self.lookup_curve_1d_inverse = {
                    "prices": xp_unique,
                    "offsets": fp_unique,
                }
                # Invert all prices at once against the single curve
                p_clamped = np.clip(prices_arr, xp_unique.min(), xp_unique.max())
                offsets = np.interp(p_clamped, xp_unique, fp_unique)
            # Return abilities in physical units
            return list(np.array(offsets, dtype=float) * unit)

        # ---- 2D calibration in (loc, scale) per runner ----
        scales = np.asarray(self.scales, dtype=float)
        unit = self.base.lattice.unit
        # Initialize locs (abilities) in physical units
        if initial_offsets is None:
            locs = np.zeros(n, dtype=float)
        else:
            locs = np.asarray(initial_offsets, dtype=float) * unit

        # Location grid (physical units) around 0
        loc_grid = np.arange(
            -self.loc_span, self.loc_span + self.loc_step, self.loc_step, dtype=float
        )

        def scale_grid_for(si: float) -> np.ndarray:
            if self.scale_steps <= 1:
                return np.array([max(1e-6, si)], dtype=float)
            offsets_scale = np.linspace(
                -self.scale_span, self.scale_span, self.scale_steps, dtype=float
            )
            sg = si + offsets_scale
            return np.maximum(1e-6, sg.astype(float))

        for _ in range(self.n_iter):
            # Precompute field distribution once per iteration (current locs/scales), multiplicity-aware
            current_densities = [
                self.density_for(float(locs[j]), float(scales[j])) for j in range(n)
            ]
            densityAll, multAll = winner_of_many(current_densities)
            cdfAll = densityAll.cdf()
            # Precompute a single interpolation table per scale value encountered
            scale_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}
            # Build the union of all scale samples needed this iteration
            all_s_values = []
            per_runner_sg = []
            for i in range(n):
                sg = scale_grid_for(float(scales[i]))
                per_runner_sg.append(sg)
                all_s_values.extend(list(sg))
            unique_s_values = sorted(set(float(s) for s in all_s_values))
            # Build curve for each unique scale once: xp(price), fp(loc)
            for s in unique_s_values:
                p_curve = []
                for loc_candidate in loc_grid:
                    d_i_candidate = self.density_for(float(loc_candidate), float(s))
                    payoff_vec = expected_payoff_with_multiplicity(
                        d_i_candidate, densityAll, multAll, cdf=None, cdfAll=cdfAll
                    )
                    p_curve.append(float(np.sum(payoff_vec)))
                p_curve = np.array(p_curve, dtype=float)
                # Cache price curve in physical units (loc -> price), sorted by loc
                locs_phys = np.array(loc_grid, dtype=float)
                order_loc = np.argsort(locs_phys)
                self.lookup_curves_2d_prices[float(s)] = (
                    locs_phys[order_loc],
                    p_curve[order_loc],
                )
                pairs = sorted(zip(p_curve, loc_grid), key=lambda t: t[0])
                xp = np.array([pp for pp, _ in pairs], dtype=float)
                fp = np.array([ll for _, ll in pairs], dtype=float)
                # Unique price keys
                xp_unique, idx = np.unique(xp, return_index=True)
                fp_unique = fp[idx]
                scale_cache[s] = (xp_unique, fp_unique)
                # Cache inverse (price -> loc) at this scale
                self.lookup_curves_2d_inverse[float(s)] = (xp_unique, fp_unique)
            # Now invert per runner using cached per-scale curves, then interpolate across scale
            for i in range(n):
                si = float(scales[i])
                pi_target = float(prices_arr[i])
                sg = per_runner_sg[i]
                loc_estimates = []
                for s in sg:
                    xp_unique, fp_unique = scale_cache[float(s)]
                    p_clamped = float(np.clip(pi_target, xp_unique.min(), xp_unique.max()))
                    loc_s = float(np.interp(p_clamped, xp_unique, fp_unique))
                    loc_estimates.append(loc_s)
                loc_estimates = np.array(loc_estimates, dtype=float)
                if len(sg) == 1:
                    locs[i] = loc_estimates[0]
                else:
                    locs[i] = float(np.interp(si, sg, loc_estimates))

        # Remove translation (identifiability) by median-centering
        locs = locs - np.median(locs)
        return list(locs)

    def solve_from_dividends(
        self, dividends: Sequence[float], nan_value: float = 2000.0
    ) -> List[float]:
        prices = StatePricer.prices_from_dividends(dividends, nan_value=nan_value)
        return self.solve_from_prices(prices)

    # forward direction (ability -> state prices/dividends), with clustering for extended offsets
    def state_prices_from_ability(self, ability: Sequence[float]) -> List[float]:
        offsets = [a / self.base.lattice.unit for a in ability]
        splitter = ClusterSplitter()
        return splitter.extended_state_prices(self.base, offsets)

    def dividends_from_ability(
        self, ability: Sequence[float], multiplicity: float = 1.0
    ) -> List[float]:
        prices = self.state_prices_from_ability(ability)
        return list(1.0 / (multiplicity * np.array(prices)))
