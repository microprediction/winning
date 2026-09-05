from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .density import Density
from .dynamic import RaceObservation  # reuse existing container to keep DRY
from .inference import AbilityCalibrator


@dataclass
class HorseFilterState:
    """Filter state for a single horse."""

    mean: float
    var: float
    last_time: float


@dataclass
class KalmanAbilityTracker:
    """
    Per-horse 1-D Kalman tracker over latent abilities using price-implied abilities as observations.

    Model (for each horse h):
        x_{j} = x_{j-1} + w_j,     w_j ~ N(0, q * Δt_j)
        y_{j} = x_{j} + v_j,       v_j ~ N(0, r)

    where:
      - x_j is the latent ability at the j-th race for this horse,
      - y_j is the centered, price-implied ability from the race inversion,
      - Δt_j is the time between successive races for this horse,
      - q is process variance per unit time,
      - r is observation variance.
    """

    base_density: Density
    ability_calibrator_kwargs: dict = field(default_factory=dict)

    # Kalman parameters
    process_var_per_time: float = 0.1
    obs_var: float = 0.5
    init_var: float = 10.0
    min_dt: float = 1e-3
    min_var: float = 1e-9

    # Internal state
    _state: Dict[str, HorseFilterState] = field(init=False, default_factory=dict)
    _history: Dict[str, List[Tuple[float, float]]] = field(init=False, default_factory=dict)
    _centers: Dict[str, List[Tuple[float, float]]] = field(init=False, default_factory=dict)

    def update_race(self, race: RaceObservation) -> None:
        """
        Invert prices for a race, center the ability vector to remove translation gauge,
        and feed observations into per-horse Kalman updates. Races should be processed
        in chronological order for best results.
        """
        cal = AbilityCalibrator(self.base_density, **self.ability_calibrator_kwargs)
        ability_vec = np.asarray(cal.solve_from_prices(race.prices), dtype=float)
        center = float(np.median(ability_vec))
        ability_centered = ability_vec - center

        for horse_id, ability_obs in zip(race.horse_ids, ability_centered):
            t = float(race.time)
            self._kf_update(horse_id=horse_id, time=t, y=float(ability_obs))
            # record the removed race center for reconstruction
            self._centers.setdefault(horse_id, []).append((t, center))

    def get_horse_state(self, horse_id: str) -> Optional[HorseFilterState]:
        """Return current filter state for a horse, or None if unseen."""
        return self._state.get(horse_id)

    def _kf_update(self, horse_id: str, time: float, y: float) -> None:
        """One 1-D Kalman predict+update step for a single horse."""
        state = self._state.get(horse_id)

        if state is None:
            mean_prev = 0.0
            var_prev = float(self.init_var)
            dt = 0.0
        else:
            mean_prev = float(state.mean)
            var_prev = float(state.var)
            dt = float(time - state.last_time)
            if dt < 0.0:
                dt = 0.0

        dt_eff = max(dt, self.min_dt)
        Q = float(self.process_var_per_time) * dt_eff

        # Predict
        mean_pred = mean_prev
        var_pred = var_prev + Q

        # Update
        R = float(self.obs_var)
        S = var_pred + R
        K = 0.0 if S <= 0.0 else var_pred / S

        mean_new = mean_pred + K * (y - mean_pred)
        var_new = (1.0 - K) * var_pred
        if var_new < self.min_var:
            var_new = self.min_var

        self._state[horse_id] = HorseFilterState(mean=mean_new, var=var_new, last_time=float(time))
        self._history.setdefault(horse_id, []).append((float(time), float(y)))
        # _centers updated in update_race

    # -------------------------- EM for q and r --------------------------
    def fit_em(self, num_iters: int = 10, fix_obs_var: Optional[float] = None) -> None:
        """
        Batch EM across horses to estimate process_var_per_time (q) and obs_var (r).
        Uses filter + RTS smoother per horse to accumulate sufficient statistics.
        """
        if not self._history:
            raise ValueError("No observations stored; call update_race first.")

        q = float(self.process_var_per_time)
        r = float(self.obs_var if fix_obs_var is None else fix_obs_var)

        for _ in range(max(1, int(num_iters))):
            r_num = 0.0
            r_den = 0.0
            q_num = 0.0
            q_den = 0.0

            for _, obs_list in self._history.items():
                if len(obs_list) == 0:
                    continue
                times = np.array([t for (t, _) in obs_list], dtype=float)
                ys = np.array([y for (_, y) in obs_list], dtype=float)
                rn, rd, qn, qd = self._em_stats_single_horse(times, ys, q=q, r=r)
                if fix_obs_var is None:
                    r_num += rn
                    r_den += rd
                q_num += qn
                q_den += qd

            if fix_obs_var is None and r_den > 0.0:
                r = r_num / r_den
            if q_den > 0.0:
                q = q_num / q_den

        self.process_var_per_time = float(q)
        self.obs_var = float(r)

    # -------------------------- smoothing output --------------------------
    def smooth_horse(self, horse_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return (times, smoothed_means, smoothed_vars) for a horse using the current
        process_var_per_time and obs_var. None if horse has no observations.
        """
        obs_list = self._history.get(horse_id)
        if not obs_list:
            return None
        times = np.array([t for (t, _) in obs_list], dtype=float)
        ys = np.array([y for (_, y) in obs_list], dtype=float)

        q = float(self.process_var_per_time)
        r = float(self.obs_var)
        J = len(ys)

        m_f = np.zeros(J, dtype=float)
        P_f = np.zeros(J, dtype=float)
        m_pred = np.zeros(J, dtype=float)
        P_pred = np.zeros(J, dtype=float)

        # Prior
        m_prev = 0.0
        P_prev = float(self.init_var)
        m_pred[0] = m_prev
        P_pred[0] = P_prev
        S0 = P_pred[0] + r
        K0 = 0.0 if S0 <= 0.0 else P_pred[0] / S0
        m_f[0] = m_pred[0] + K0 * (ys[0] - m_pred[0])
        P_f[0] = (1.0 - K0) * P_pred[0]

        for j in range(1, J):
            dt = float(times[j] - times[j - 1])
            if dt < 0.0:
                dt = 0.0
            dt_eff = max(dt, self.min_dt)
            Qj = q * dt_eff
            m_pred[j] = m_f[j - 1]
            P_pred[j] = P_f[j - 1] + Qj
            Sj = P_pred[j] + r
            Kj = 0.0 if Sj <= 0.0 else P_pred[j] / Sj
            m_f[j] = m_pred[j] + Kj * (ys[j] - m_pred[j])
            P_f[j] = (1.0 - Kj) * P_pred[j]

        # RTS
        m_s = m_f.copy()
        P_s = P_f.copy()
        for j in range(J - 1, 0, -1):
            dt = float(times[j] - times[j - 1])
            if dt < 0.0:
                dt = 0.0
            dt_eff = max(dt, self.min_dt)
            Qj = q * dt_eff
            P_pred_j = P_f[j - 1] + Qj
            A = 0.0 if P_pred_j <= 0.0 else P_f[j - 1] / P_pred_j
            m_s[j - 1] = m_f[j - 1] + A * (m_s[j] - m_f[j - 1])
            P_s[j - 1] = P_f[j - 1] + A * A * (P_s[j] - P_pred_j)

        return times, m_s, P_s

    def smooth_horse_abs(
        self, horse_id: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return absolute (times, smoothed_means_plus_center, smoothed_vars), by adding
        back the per-race centers that were removed during observation construction.
        """
        base = self.smooth_horse(horse_id)
        if base is None:
            return None
        times, m_s, P_s = base
        centers = self._centers.get(horse_id)
        if not centers or len(centers) != len(times):
            # cannot reconstruct; return relative
            return times, m_s, P_s
        c_arr = np.array([c for (_, c) in centers], dtype=float)
        return times, m_s + c_arr, P_s

    def smooth_horses_zero_mean(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Produce zero-mean anchored trajectories on race times:
        - run smooth_horse() for each horse to get relative means on its observation times
        - for each race time t, subtract the cross-horse average of means at that exact time
        Returns dict: horse_id -> (times, anchored_means). Variances omitted for brevity.
        """
        # First pass: collect relative smooths
        rel: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for h, obs in self._history.items():
            if not obs:
                continue
            base = self.smooth_horse(h)
            if base is None:
                continue
            times, means, _ = base
            rel[h] = (times, means)

        # Aggregate per-time means
        time_accum_sum: Dict[float, float] = {}
        time_accum_cnt: Dict[float, int] = {}
        for times, means in rel.values():
            for t, m in zip(times, means):
                time_accum_sum[t] = time_accum_sum.get(t, 0.0) + float(m)
                time_accum_cnt[t] = time_accum_cnt.get(t, 0) + 1
        time_mean: Dict[float, float] = {
            t: (time_accum_sum[t] / max(1, time_accum_cnt[t])) for t in time_accum_sum
        }

        # Subtract the per-time mean
        anchored: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for h, (times, means) in rel.items():
            adj = np.array(
                [float(m) - float(time_mean.get(float(t), 0.0)) for t, m in zip(times, means)],
                dtype=float,
            )
            anchored[h] = (times, adj)
        return anchored

    def _em_stats_single_horse(
        self,
        times: np.ndarray,
        ys: np.ndarray,
        q: float,
        r: float,
    ) -> Tuple[float, float, float, float]:
        """
        Sufficient statistics for EM for a single horse sequence.
        Returns (r_num, r_den, q_num, q_den).
        """
        J = int(len(ys))
        if J == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Forward filter
        m_f = np.zeros(J, dtype=float)
        P_f = np.zeros(J, dtype=float)
        m_pred = np.zeros(J, dtype=float)
        P_pred = np.zeros(J, dtype=float)

        # Prior
        m_prev = 0.0
        P_prev = float(self.init_var)

        # First observation j=0
        m_pred[0] = m_prev
        P_pred[0] = P_prev  # Δt=0
        S0 = P_pred[0] + r
        K0 = 0.0 if S0 <= 0.0 else P_pred[0] / S0
        m_f[0] = m_pred[0] + K0 * (ys[0] - m_pred[0])
        P_f[0] = (1.0 - K0) * P_pred[0]

        # Forward pass
        for j in range(1, J):
            dt = float(times[j] - times[j - 1])
            if dt < 0.0:
                dt = 0.0
            dt_eff = max(dt, self.min_dt)
            Qj = q * dt_eff

            m_pred[j] = m_f[j - 1]
            P_pred[j] = P_f[j - 1] + Qj
            Sj = P_pred[j] + r
            Kj = 0.0 if Sj <= 0.0 else P_pred[j] / Sj
            m_f[j] = m_pred[j] + Kj * (ys[j] - m_pred[j])
            P_f[j] = (1.0 - Kj) * P_pred[j]

        # RTS smoother
        m_s = m_f.copy()
        P_s = P_f.copy()
        # Cross-covariance term Cov(x_j, x_{j-1})
        C = np.zeros(J, dtype=float)

        for j in range(J - 1, 0, -1):
            dt = float(times[j] - times[j - 1])
            if dt < 0.0:
                dt = 0.0
            dt_eff = max(dt, self.min_dt)
            Qj = q * dt_eff
            P_pred_j = P_f[j - 1] + Qj
            A = 0.0 if P_pred_j <= 0.0 else P_f[j - 1] / P_pred_j
            m_s[j - 1] = m_f[j - 1] + A * (m_s[j] - m_f[j - 1])
            P_s[j - 1] = P_f[j - 1] + A * A * (P_s[j] - P_pred_j)
            C[j] = A * P_s[j]

        # Stats for r
        r_num = 0.0
        r_den = 0.0
        for j in range(J):
            err = ys[j] - m_s[j]
            r_num += err * err + P_s[j]
            r_den += 1.0

        # Stats for q
        q_num = 0.0
        q_den = 0.0
        for j in range(1, J):
            dt = float(times[j] - times[j - 1])
            if dt < 0.0:
                dt = 0.0
            dt_eff = max(dt, self.min_dt)
            Ex2 = m_s[j] * m_s[j] + P_s[j]
            Ex_prev2 = m_s[j - 1] * m_s[j - 1] + P_s[j - 1]
            Exx_prev = m_s[j] * m_s[j - 1] + C[j]
            E_delta2 = Ex2 + Ex_prev2 - 2.0 * Exx_prev
            q_num += E_delta2 / dt_eff
            q_den += 1.0

        return r_num, r_den, q_num, q_den
