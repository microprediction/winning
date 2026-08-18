from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .density import Density
from .inference import AbilityCalibrator

# --- Public containers --------------------------------------------------------


@dataclass
class RaceObservation:
    """
    Minimal container for one race observation.
    - race_id: opaque identifier for the race
    - time: numeric time (e.g., days since epoch). Must be comparable across races
    - horse_ids: list of horse identifiers in post order (only used for alignment)
    - prices: risk‑neutral winning probabilities for the same order of horse_ids
    - winner: optional winner id (future extensions may use ranks/margins)
    """

    race_id: str
    time: float
    horse_ids: List[str]
    prices: Sequence[float]
    winner: Optional[str] = None


SigmaFn = Callable[[float], float]  # maps Δt -> σ(Δt)


@dataclass
class DynamicThurstoneCalibrator:
    """
    Dynamic Thurstone‑style calibrator on top of the static AbilityCalibrator.

    Pipeline:
      1) For each race, run AbilityCalibrator.solve_from_prices(prices) to get
         "raw" per‑race abilities for the entrants.
      2) Assemble per‑horse trajectories (times and abilities) across races.
      3) Optionally smooth each trajectory with a random‑walk prior whose
         increment std is σ(Δt).
      4) Optionally estimate σ(Δt) from observed ability increments.

    If bookmaker_sigma > 0, all price→ability inversions are performed using a
    predictive base density obtained by convolving the base density with a
    zero‑mean Gaussian over ability offsets (in ability units). This models a
    bookmaker adding their own uncertainty before pricing.
    """

    base_density: Density
    races: List[RaceObservation]
    ability_calibrator_kwargs: dict = field(default_factory=dict)

    # Bookmaker's ability‑uncertainty std (in ability units). If zero, the
    # original behaviour is recovered.
    bookmaker_sigma: float = 0.0

    # learned / produced attributes
    theta_: Dict[str, np.ndarray] = field(init=False, default_factory=dict)
    times_: Dict[str, np.ndarray] = field(init=False, default_factory=dict)

    # piecewise sigma(Δt) if learned via fit_sigma
    sigma_edges_: Optional[np.ndarray] = field(init=False, default=None)
    sigma_vals_: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # ensure races are sorted by time
        self.races = sorted(self.races, key=lambda r: r.time)

    # --- bookmaker predictive density -----------------------------------------
    def _predictive_base_density(self) -> Density:
        """
        Density used when inverting prices. If bookmaker_sigma == 0, returns the
        original base density. Otherwise returns a mixture of shifted base
        densities with Gaussian weights over ability offsets.
        """
        if self.bookmaker_sigma <= 0.0:
            return self.base_density

        base = self.base_density
        unit = float(base.lattice.unit)
        # express bookmaker std in lattice steps
        sigma_steps = float(self.bookmaker_sigma) / max(unit, 1e-12)
        max_steps = int(math.ceil(4.0 * sigma_steps))
        if max_steps <= 0:
            return base

        offsets_steps = np.arange(-max_steps, max_steps + 1, dtype=float)
        # Gaussian over ability offsets (ability units)
        offsets_ability = offsets_steps * unit
        w = np.exp(-0.5 * (offsets_ability / float(self.bookmaker_sigma)) ** 2)
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            return base
        w /= w_sum

        pdf_pred = np.zeros_like(np.asarray(base.p, dtype=float), dtype=float)
        for o_steps, weight in zip(offsets_steps, w):
            if weight == 0.0:
                continue
            shifted = base.shift_fractional(float(o_steps))
            pdf_pred += weight * np.asarray(shifted.p, dtype=float)

        return Density(base.lattice, pdf_pred)

    def _new_calibrator(self) -> AbilityCalibrator:
        """Factory for AbilityCalibrator using the predictive base density."""
        pred_base = self._predictive_base_density()
        return AbilityCalibrator(pred_base, **self.ability_calibrator_kwargs)

    # --- indexing -------------------------------------------------------------
    def _build_horse_index(self) -> Dict[str, List[int]]:
        """Map horse_id -> sorted list of race indices it appears in."""
        idx: Dict[str, List[int]] = {}
        for r_i, race in enumerate(self.races):
            for h in race.horse_ids:
                idx.setdefault(h, []).append(r_i)
        for h, ndxs in idx.items():
            ndxs.sort(key=lambda i: self.races[i].time)
        return idx

    # --- initial per‑race abilities ------------------------------------------
    def _initial_abilities_from_prices(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Use AbilityCalibrator race‑by‑race to get raw abilities (no smoothing).

        Returns
        -------
        theta_raw : dict[horse_id] -> np.ndarray (time‑ordered abilities)
        times     : dict[horse_id] -> np.ndarray (aligned times)
        """
        horse_index = self._build_horse_index()

        # per‑race ability vectors (aligned to race.horse_ids)
        race_abilities: List[Dict[str, float]] = []
        for race in self.races:
            cal = self._new_calibrator()
            # race.prices should be risk‑neutral winning probabilities
            ability_vec = np.asarray(cal.solve_from_prices(race.prices), dtype=float)
            # Center per-race to remove translation ambiguity
            ability_vec = ability_vec - float(np.median(ability_vec))
            per_horse = {h: float(a) for h, a in zip(race.horse_ids, ability_vec)}
            race_abilities.append(per_horse)

        theta_raw: Dict[str, List[float]] = {h: [] for h in horse_index.keys()}
        times: Dict[str, List[float]] = {h: [] for h in horse_index.keys()}

        for h, ndxs in horse_index.items():
            for i in ndxs:
                theta_raw[h].append(race_abilities[i][h])
                times[h].append(self.races[i].time)

        theta_np = {h: np.asarray(theta_raw[h], dtype=float) for h in theta_raw}
        times_np = {h: np.asarray(times[h], dtype=float) for h in times}
        return theta_np, times_np

    # --- trajectory smoother --------------------------------------------------
    @staticmethod
    def _smooth_trajectory(
        times: np.ndarray,
        m: np.ndarray,
        sigma_fn: SigmaFn,
        obs_var: float = 1.0,
    ) -> np.ndarray:
        """
        Smooth raw abilities m_j at times t_j with a random‑walk prior:

            sum_j (θ_j - m_j)^2 / obs_var
          + sum_{j>1} (θ_j - θ_{j-1})^2 / σ(Δt_j)^2

        This yields a symmetric tridiagonal linear system A θ = b.
        """
        J = len(m)
        if J <= 1:
            return m.copy()

        dt = np.diff(times)
        lam_obs = 1.0 / float(obs_var)
        # process precision per gap
        lam_proc = np.array(
            [1.0 / max(float(sigma_fn(float(d))), 1e-6) ** 2 for d in dt], dtype=float
        )

        A = np.zeros((J, J), dtype=float)
        b = lam_obs * m.astype(float).copy()

        # first row
        A[0, 0] = lam_obs + lam_proc[0]
        A[0, 1] = -lam_proc[0]
        # interior rows
        for j in range(1, J - 1):
            lp = lam_proc[j - 1]
            ln = lam_proc[j]
            A[j, j - 1] = -lp
            A[j, j] = lam_obs + lp + ln
            A[j, j + 1] = -ln
        # last row
        A[J - 1, J - 2] = -lam_proc[J - 2]
        A[J - 1, J - 1] = lam_obs + lam_proc[J - 2]

        theta = np.linalg.solve(A, b)
        return theta

    # --- public API -----------------------------------------------------------
    def fit_abilities(
        self,
        sigma_function: Optional[SigmaFn] = None,
        obs_var: float = 1.0,
    ) -> None:
        """
        Fit (or just stage) dynamic abilities θ_{h,j}.
        If sigma_function is None, store per‑race static abilities.
        Else smooth each horse trajectory with random‑walk prior using σ(Δt).
        """
        theta_raw, times = self._initial_abilities_from_prices()

        if sigma_function is None:
            self.theta_ = theta_raw
            self.times_ = times
            return

        theta_smooth: Dict[str, np.ndarray] = {}
        for h, m in theta_raw.items():
            t = times[h]
            if len(m) <= 1:
                theta_smooth[h] = m.copy()
            else:
                theta_smooth[h] = self._smooth_trajectory(t, m, sigma_function, obs_var=obs_var)

        self.theta_ = theta_smooth
        self.times_ = times

    def fit_sigma(
        self,
        n_bins: int = 5,
        min_points: int = 20,
        monotone: bool = True,
        meas_var: float = 0.0,
    ) -> SigmaFn:
        """
        Learn a piecewise‑constant σ(Δt) from current θ trajectories.
        Requires self.theta_ and self.times_ (e.g., after fit_abilities(None)).

        Parameters
        ----------
        n_bins : int
            Number of Δt quantile bins.
        min_points : int
            Soft floor on number of points per bin for variance estimate.
        monotone : bool
            If True, enforce non‑decreasing σ with Δt.
        meas_var : float
            Known measurement variance in ability space for each per‑race estimate.
            Since increments include two independent measurements, the increment
            variance is inflated by ~2*meas_var; we subtract that baseline before
            taking the square root.
        """
        dts: List[float] = []
        dtheta2: List[float] = []
        for h, theta in self.theta_.items():
            t = self.times_[h]
            if len(theta) <= 1:
                continue
            dt_h = np.diff(t)
            dθ = np.diff(theta)
            dts.extend(dt_h.tolist())
            dtheta2.extend((dθ**2).tolist())

        if not dts:
            raise ValueError("No ability increments to fit sigma(Δt).")

        dts_arr = np.asarray(dts, dtype=float)
        dtheta2_arr = np.asarray(dtheta2, dtype=float)

        # bin edges from quantiles
        quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
        edges = np.quantile(dts_arr, quantiles)
        # expand edges slightly to include endpoints
        edges[0] -= 1e-9
        edges[-1] += 1e-9

        sigma_vals: List[float] = []
        for k in range(n_bins):
            lo, hi = edges[k], edges[k + 1]
            mask = (dts_arr >= lo) & (dts_arr < hi)
            if np.sum(mask) < max(1, int(min_points) // max(1, n_bins)):
                var_k = float(np.mean(dtheta2_arr))  # fallback to global variance
            else:
                var_k = float(np.mean(dtheta2_arr[mask]))
            var_k = max(var_k - 2.0 * float(meas_var), 1e-12)
            sigma_vals.append(math.sqrt(var_k))

        sigma_vals_arr = np.asarray(sigma_vals, dtype=float)
        if monotone and len(sigma_vals_arr) > 1:
            # enforce non‑decreasing σ with Δt (simple pooled adjacent violators)
            for k in range(1, len(sigma_vals_arr)):
                if sigma_vals_arr[k] < sigma_vals_arr[k - 1]:
                    sigma_vals_arr[k] = sigma_vals_arr[k - 1]

        self.sigma_edges_ = edges
        self.sigma_vals_ = sigma_vals_arr

        def sigma_fn(dt: float) -> float:
            d = float(dt)
            k = int(np.searchsorted(edges, d, side="right") - 1)
            k = max(0, min(k, len(sigma_vals_arr) - 1))
            return float(sigma_vals_arr[k])

        return sigma_fn

    # --- helpers for measurement-noise calibration ----------------------------
    def _collect_increments(
        self,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect (Δt, (Δθ)^2) from current stored per-horse trajectories.
        If dt_min/dt_max are provided, restrict to that window.
        """
        dts: List[float] = []
        dtheta2: List[float] = []
        for h, theta in self.theta_.items():
            t = self.times_[h]
            if len(theta) <= 1:
                continue
            dt_h = np.diff(t)
            dth = np.diff(theta)
            if dt_min is not None or dt_max is not None:
                m = np.ones_like(dt_h, dtype=bool)
                if dt_min is not None:
                    m &= dt_h >= dt_min
                if dt_max is not None:
                    m &= dt_h <= dt_max
                dt_h = dt_h[m]
                dth = dth[m]
            if dt_h.size == 0:
                continue
            dts.extend(dt_h.tolist())
            dtheta2.extend((dth**2).tolist())
        return np.asarray(dts, dtype=float), np.asarray(dtheta2, dtype=float)

    def fit_sigma_autocalibrate(
        self,
        n_bins: int = 5,
        min_points: int = 20,
        monotone: bool = True,
        small_quantile: float = 0.2,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
    ) -> Tuple[SigmaFn, float]:
        """
        Estimate measurement variance τ^2 from smallest-Δt increments, then fit
        σ(Δt) via the binned estimator with meas_var=τ̂^2.

        Returns (sigma_fn, meas_var_hat).
        """
        dts, dtheta2 = self._collect_increments(dt_min=dt_min, dt_max=dt_max)
        if dts.size == 0:
            raise ValueError("No ability increments to autocalibrate meas_var.")
        q = float(np.clip(small_quantile, 0.01, 0.9))
        cutoff = float(np.quantile(dts, q))
        mask_small = dts <= cutoff
        if not np.any(mask_small):
            meas_var_hat = 0.0
        else:
            meas_var_hat = 0.5 * float(np.mean(dtheta2[mask_small]))
            meas_var_hat = max(meas_var_hat, 0.0)
        sigma_fn = self.fit_sigma(
            n_bins=n_bins,
            min_points=min_points,
            monotone=monotone,
            meas_var=meas_var_hat,
        )
        return sigma_fn, meas_var_hat

    # --- simple per-race refinement using observed winner ---------------------
    def _refine_with_result_once(
        self,
        ability: np.ndarray,
        horse_ids: Sequence[str],
        winner_id: Optional[str],
        step: float = 0.5,
        eps: float = 0.05,
    ) -> np.ndarray:
        """
        One finite-difference gradient step to increase log probability of winner.
        If winner_id is None or not in horse_ids, return ability unchanged.
        """
        if winner_id is None or winner_id not in horse_ids:
            return ability
        winner_idx = int(horse_ids.index(winner_id))
        cal = self._new_calibrator()
        base_probs = np.asarray(cal.state_prices_from_ability(ability.tolist()), dtype=float)
        # loss = -log q_w
        loss0 = -float(np.log(max(base_probs[winner_idx], 1e-15)))
        grad = np.zeros_like(ability, dtype=float)
        for i in range(len(ability)):
            a_pert = ability.copy()
            a_pert[i] -= eps  # negative shift = better ability
            p = np.asarray(cal.state_prices_from_ability(a_pert.tolist()), dtype=float)
            li = -float(np.log(max(p[winner_idx], 1e-15)))
            grad[i] = (li - loss0) / (-eps)
        # update and re-center to preserve translation invariance
        ability_new = ability - step * grad
        ability_new = ability_new - float(np.median(ability_new))
        return ability_new

    def fit_abilities_with_results(
        self,
        refine_steps: int = 1,
        refine_step_size: float = 0.5,
        refine_eps: float = 0.05,
    ) -> None:
        """
        Build per-race raw abilities from prices, then nudge each race's vector
        to better explain the observed winner (if provided).
        Stores the refined, time-aligned per-horse trajectories.
        """
        horse_index = self._build_horse_index()
        race_abilities: List[Dict[str, float]] = []
        for race in self.races:
            cal = self._new_calibrator()
            ability_vec = np.asarray(cal.solve_from_prices(race.prices), dtype=float)
            # Center per-race to remove translation ambiguity before refinement
            ability_vec = ability_vec - float(np.median(ability_vec))
            a = ability_vec.copy()
            for _ in range(max(0, int(refine_steps))):
                a = self._refine_with_result_once(
                    a,
                    race.horse_ids,
                    race.winner,
                    step=refine_step_size,
                    eps=refine_eps,
                )
            per_horse = {h: float(x) for h, x in zip(race.horse_ids, a.tolist())}
            race_abilities.append(per_horse)

        theta_raw: Dict[str, List[float]] = {h: [] for h in horse_index.keys()}
        times: Dict[str, List[float]] = {h: [] for h in horse_index.keys()}
        for h, ndxs in horse_index.items():
            for i in ndxs:
                theta_raw[h].append(race_abilities[i][h])
                times[h].append(self.races[i].time)
        self.theta_ = {h: np.asarray(theta_raw[h], dtype=float) for h in theta_raw}
        self.times_ = {h: np.asarray(times[h], dtype=float) for h in times}

    # --- parametric σ(Δt) ≈ sqrt(α Δt), optionally estimating meas_var ----------
    def fit_sigma_parametric(
        self,
        meas_var: Optional[float] = None,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
    ) -> Tuple[SigmaFn, float, float]:
        """
        Fit α in Var(Δθ) ≈ 2*meas_var + α*Δt by least squares.
        If meas_var is None, also fit intercept and set meas_var_hat = max(intercept/2, 0).
        Returns (sigma_fn, alpha_hat, meas_var_hat) where sigma_fn(dt)=sqrt(max(α dt, ε)).
        """
        dts: List[float] = []
        dtheta2: List[float] = []
        for h, theta in self.theta_.items():
            t = self.times_[h]
            if len(theta) <= 1:
                continue
            dt_h = np.diff(t)
            dth = np.diff(theta)
            if dt_min is not None or dt_max is not None:
                m = np.ones_like(dt_h, dtype=bool)
                if dt_min is not None:
                    m &= dt_h >= dt_min
                if dt_max is not None:
                    m &= dt_h <= dt_max
                dt_h = dt_h[m]
                dth = dth[m]
            if dt_h.size == 0:
                continue
            dts.extend(dt_h.tolist())
            dtheta2.extend((dth**2).tolist())

        if not dts:
            raise ValueError("No ability increments to fit parametric σ(Δt).")

        x = np.asarray(dts, dtype=float)
        y = np.asarray(dtheta2, dtype=float)

        if meas_var is None:
            X = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a = max(float(beta[0]), 0.0)
            b = max(float(beta[1]), 0.0)
            meas_var_hat = max(a * 0.5, 0.0)
            alpha_hat = b
        else:
            y_adj = np.maximum(y - 2.0 * float(meas_var), 0.0)
            denom = float(np.dot(x, x))
            alpha_hat = 0.0 if denom <= 0 else max(float(np.dot(x, y_adj) / denom), 0.0)
            meas_var_hat = float(meas_var)

        def sigma_fn(dt: float) -> float:
            return math.sqrt(max(alpha_hat * float(dt), 1e-12))

        return sigma_fn, alpha_hat, meas_var_hat
