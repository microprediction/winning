from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .inference import AbilityCalibrator


@dataclass
class RaceLS:
    calibrator: AbilityCalibrator
    horse_ids: List[str]
    prices: np.ndarray
    scales: Optional[np.ndarray] = None
    # Precomputed raw local locs from per-race inversion (physical units)
    local_locs: Optional[np.ndarray] = None


@dataclass
class GlobalLSCalibrator:
    """
    Relative-then-global least squares:
      1) Per race: invert prices -> local locs (once), center per race to remove translation
      2) Stitch globally via (slope-weighted) LS across overlapping runners
    """

    horse_ids: List[str]
    races: List[RaceLS] = field(default_factory=list)
    theta: Dict[str, float] = field(default_factory=dict)  # global abilities

    def __post_init__(self):
        if not self.theta:
            self.theta = {hid: 0.0 for hid in self.horse_ids}

    def add_race(
        self,
        calibrator: AbilityCalibrator,
        horse_ids: Sequence[str],
        prices: Sequence[float],
        scales: Optional[Sequence[float]] = None,
    ) -> None:
        prices_arr = np.asarray(prices, dtype=float)
        scales_arr = None if scales is None else np.asarray(scales, dtype=float)
        # Ensure lookup curves exist at least once
        if (calibrator.lookup_curve_1d_prices is None) and (not calibrator.lookup_curves_2d_prices):
            calibrator.solve_from_prices(prices_arr)
        # Precompute local raw locs once for this race
        local_locs = np.array(calibrator.solve_from_prices(prices_arr), dtype=float)
        self.races.append(
            RaceLS(
                calibrator=calibrator,
                horse_ids=list(horse_ids),
                prices=prices_arr,
                scales=scales_arr,
                local_locs=local_locs,
            )
        )

    def _invert_and_center(self, race: RaceLS) -> np.ndarray:
        """Return median-centered per-race locs (translation-free)."""
        if race.local_locs is None:
            est = np.array(race.calibrator.solve_from_prices(race.prices), dtype=float)
            race.local_locs = est
        return race.local_locs - np.median(race.local_locs)

    def _slope_weight(self, cal: AbilityCalibrator, loc: float, scale: Optional[float]) -> float:
        """Approximate |dp/dmu| using cached curves; fall back to 1.0."""
        try:
            if scale is not None and len(cal.lookup_curves_2d_prices) > 0:
                # nearest scale curve
                scales = sorted(cal.lookup_curves_2d_prices.keys())
                s_arr = np.array(scales, dtype=float)
                idx = int(np.searchsorted(s_arr, scale))
                if idx <= 0:
                    s_sel = s_arr[0]
                elif idx >= len(s_arr):
                    s_sel = s_arr[-1]
                else:
                    s_sel = (
                        s_arr[idx - 1]
                        if (scale - s_arr[idx - 1]) <= (s_arr[idx] - scale)
                        else s_arr[idx]
                    )
                locs, prices = cal.lookup_curves_2d_prices[float(s_sel)]
                dprices = np.gradient(prices, locs)
                mu_c = float(np.clip(loc, locs.min(), locs.max()))
                dp = float(np.interp(mu_c, locs, dprices))
                return abs(dp) + 1e-12
            if cal.lookup_curve_1d_prices:
                locs = cal.lookup_curve_1d_prices["locs"]
                prices = cal.lookup_curve_1d_prices["prices"]
                dprices = np.gradient(prices, locs)
                mu_c = float(np.clip(loc, locs.min(), locs.max()))
                dp = float(np.interp(mu_c, locs, dprices))
                return abs(dp) + 1e-12
        except Exception:
            pass
        return 1.0

    def fit(
        self,
        use_slope_weights: bool = True,
        ridge: float = 0.0,
        weight_cap: Optional[float] = None,
    ) -> None:
        """
        Solve for global theta via one-pass (slope-weighted) LS on centered per-race inversions.
        This is a "relative-then-average" stitch, fast and robust.
        """
        sum_w_y: Dict[str, float] = {hid: 0.0 for hid in self.horse_ids}
        sum_w: Dict[str, float] = {hid: ridge for hid in self.horse_ids}  # ridge on diagonal

        for race in self.races:
            centered = self._invert_and_center(race)
            # accumulate per horse
            for j, hid in enumerate(race.horse_ids):
                if use_slope_weights:
                    sc = None if race.scales is None else float(race.scales[j])
                    # Use raw local loc for slope weight (not centered)
                    loc_for_slope = (
                        float(race.local_locs[j])
                        if race.local_locs is not None
                        else float(centered[j])
                    )
                    w = self._slope_weight(race.calibrator, loc_for_slope, sc)
                    if weight_cap is not None:
                        w = min(w, float(weight_cap))
                else:
                    w = 1.0
                sum_w_y[hid] += w * float(centered[j])
                sum_w[hid] += w

        # Closed-form per-horse since design is diagonal by construction
        for hid in self.horse_ids:
            denom = sum_w[hid]
            self.theta[hid] = (sum_w_y[hid] / denom) if denom > 0 else 0.0

        # Fix gauge: center global theta to zero median
        med = np.median([self.theta[hid] for hid in self.horse_ids])
        for hid in self.horse_ids:
            self.theta[hid] -= float(med)
