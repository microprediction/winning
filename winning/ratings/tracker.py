"""Scalable message-passing dynamic ratings: per-entity marginal beliefs.

The finished form of the per-entity Bayesian filter (cf. the
``research.kalman_tracker`` scaffold), wired to winning's FACTOR calibration
(``race_probabilities`` / ``abilities_from_race``) throughout -- one model for
predict and for every observation, so the update is internally consistent.
State is one (mean, var) per entity: no joint covariance, so a contest touches
only its entrants -- O(entrants), no eigendecomposition. The scalable
alternative to the dense ``rate_history`` / ``walk_forward``.

Temporal model (per entity; there is no natural ability level):
  - the MEAN is pegged to the last value -- ability does not revert;
  - the VARIANCE grows between contests sub-diffusively,
        v <- v + drift * dt**alpha,   alpha < 1,
    so the std grows slower than sqrt(dt) -- no OU ceiling / stationary level.

Observations -- BOTH sources are CONTRAST evidence (no absolute level; only
within-contest contrasts are identified) and fold through the SAME conjugate
contrast-space update (``market.update_market``), differing only in what is
observed and its noise:
  - MARKET observer: prices are a noisy read of ability contrasts;
    y = abilities_from_race(prices), noise tau2.
  - RESULTS observer: transformed finishing performances (min-wins) are a noisy
    read of ability contrasts; y = -(transformed scores), noise beta2. Any
    raw-result -> performance transform is the CALLER's, on the factor ability
    unit -- winning stays domain-agnostic.
If only a finishing order is known (no magnitudes), the exact N-way Thurstone
factor (``nway``) is used for that observer instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from .nway import update_winner, update_ranking, update_ranking_exact
from .market import update_market
from ..factor.races import race_probabilities


def _neg(a):
    return -np.asarray(a, dtype=float)


def _truncnorm(mean, sd, lo, hi, rng):
    """One draw from N(mean, sd^2) truncated to (lo, hi) via inverse CDF."""
    a = (lo - mean) / sd if np.isfinite(lo) else -40.0
    b = (hi - mean) / sd if np.isfinite(hi) else 40.0
    Fa = ndtr(a); Fb = ndtr(b)
    return mean + sd * ndtri(np.clip(Fa + rng.random() * (Fb - Fa), 1e-12, 1 - 1e-12))


def order_augmented(m, v, order, beta2, rng, n_aug=60, burn=20):
    """Consistent censored-order update by DATA AUGMENTATION.

    Observing only the finishing ORDER censors the performance magnitudes; a
    single Gaussian moment-match of the (non-Gaussian) rank posterior --
    ``nway.update_ranking`` -- is only approximately consistent. This reinstates
    the censored magnitudes and Gibbs-samples the joint (ability a, latent
    performance pi):

        pi | a  ~ N(-a, beta2)  truncated to the observed finishing order,
        a  | pi ~ conjugate posterior from prior N(m, v) and observation -pi
                  with noise beta2,

    returning the posterior (mean, var) of a. This is the principled treatment
    of the censored observation (proper Bayes, no Gaussian projection of the
    rank posterior). It is SLOW -- a Gibbs sweep per contest -- and exists only
    for order-only sports; whenever performance magnitudes are observed, feed
    ``scores`` instead -- the exact linear-Gaussian update.
    """
    m = np.asarray(m, float); v = np.asarray(v, float); n = len(m); sb = np.sqrt(beta2)
    pv = 1.0 / (1.0 / v + 1.0 / beta2)                    # var of a | pi (conjugate)
    a = m.copy(); pi = np.empty(n); pi[list(order)] = np.sort(-a)   # init respecting order
    sA = np.zeros(n); sA2 = np.zeros(n); cnt = 0
    for it in range(burn + n_aug):
        for r in range(n):                                # pi | a, order (truncate to neighbours)
            i = order[r]
            lo = pi[order[r - 1]] if r > 0 else -np.inf
            hi = pi[order[r + 1]] if r < n - 1 else np.inf
            pi[i] = _truncnorm(-a[i], sb, lo, hi, rng)
        a = pv * (m / v + (-pi) / beta2) + np.sqrt(pv) * rng.standard_normal(n)   # a | pi
        if it >= burn:
            sA += a; sA2 += a * a; cnt += 1
    E = sA / cnt
    return E, np.maximum(sA2 / cnt - E * E, 1e-9)


@dataclass
class AbilityState:
    """Marginal belief for one entity: mean, variance, last-seen time."""
    mean: float
    var: float
    last_time: float


class AbilityTracker:
    """Per-entity marginal ability filter over a history of contests.

    Parameters
    ----------
    drift, drift_exp : between-contest variance term structure,
                       v += drift * dt**drift_exp; drift_exp < 1 keeps std growth
                       below sqrt(dt) (no level).
    init_var         : variance on an entity's first appearance.
    beta2            : intrinsic performance (Thurstone) variance -- also the
                       results observer's noise.
    tau2             : market observer's noise.
    base             : lattice base density name.
    """

    def __init__(self, drift: float = 0.02, drift_exp: float = 0.5,
                 init_var: float = 4.0, beta2: float = 1.0, tau2: float = 0.25,
                 base: str = "normal", order_method: str = "exact",
                 n_aug: int = 60, burn: int = 20, seed: int = 0):
        self.drift = float(drift); self.drift_exp = float(drift_exp)
        self.init_var = float(init_var)
        self.beta2 = float(beta2); self.tau2 = float(tau2)
        self.base = base
        # order-only observations (no margins) -- 'exact' = winning's exact
        # win-node ranking factor (nway.update_ranking_exact; validated against the
        # order_augmented Gibbs in tests); 'moment' = the fast biased moment-match;
        # 'augmented' = the slow Gibbs reference. Irrelevant whenever magnitudes
        # are observed, in which case `scores` is the path.
        self.order_method = order_method; self.n_aug = int(n_aug); self.burn = int(burn)
        self.rng = np.random.default_rng(seed)
        self.state: Dict[str, AbilityState] = {}

    # -- belief access -------------------------------------------------------
    def _gather(self, ids: Sequence[str], t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Marginals for `ids`, variance grown forward to t (mean pegged)."""
        m = np.empty(len(ids)); v = np.empty(len(ids))
        for k, i in enumerate(ids):
            s = self.state.get(i)
            if s is None:
                m[k], v[k] = 0.0, self.init_var
            else:
                dt = max(float(t) - s.last_time, 0.0)
                m[k] = s.mean
                v[k] = s.var + self.drift * dt ** self.drift_exp
        return m, v

    def _store(self, ids, m, v, t):
        # NO per-contest recentering. The global level is unidentified, but the
        # gauge is pinned once by the zero-mean prior on debuts; contrast
        # evidence cannot move a field's common level, so subtracting the field
        # mean here would actively zero each race's average ability and erase
        # the class differences that propagate between races through shared
        # runners (the varying-field washout bug: model sat at uniform).
        for k, i in enumerate(ids):
            self.state[i] = AbilityState(float(m[k]), float(max(v[k], 1e-9)), float(t))

    # -- prediction ----------------------------------------------------------
    def predict(self, ids: Sequence[str], t: float, points: int = 257) -> np.ndarray:
        """Predictive win probabilities among `ids` at t, belief uncertainty
        marginalised into the performance variance (same factor forward map)."""
        m, v = self._gather(ids, t)
        return race_probabilities(-m, D=self.beta2 + v, base=self.base, points=points)

    # -- observation ---------------------------------------------------------
    def observe(self, ids: Sequence[str], t: float, scores: Optional[Sequence[float]] = None,
                order: Optional[Sequence[int]] = None, winner: Optional[int] = None,
                prices: Optional[Sequence[float]] = None) -> None:
        """Fold one contest's evidence. BOTH observers (market prices, results)
        are contrast evidence through the same conjugate update."""
        m, v = self._gather(ids, t)
        if prices is not None:                                     # market observer
            m, v, _ = update_market(m, v, np.asarray(prices, float), tau2=self.tau2)
        if scores is not None:                                     # results observer, MAGNITUDES
            # The consistent path whenever performance magnitudes are observed.
            # Transformed performances are a linear-Gaussian observation
            # of ability, so update_market's conjugate contrast update is exact --
            # verified unbiased (single race, performance-noise-only) and calibrated
            # under drift on synthetic data.
            m, v, _ = update_market(m, v, np.asarray(scores, float), tau2=self.beta2, invert=_neg)
        elif order is not None:                                    # ORDER-ONLY fallback
            # Ranks censor the magnitudes. 'exact' uses winning's exact win-node
            # ranking factor (matches the Gibbs reference); 'moment' is the fast
            # biased moment-match; 'augmented' is the slow Gibbs reference itself.
            # With magnitudes available, `scores` is the right path, not this.
            if self.order_method == "exact":
                m, v = update_ranking_exact(m, v, list(order), beta2=self.beta2, base=self.base)
            elif self.order_method == "augmented":
                m, v = order_augmented(m, v, list(order), self.beta2, self.rng,
                                       n_aug=self.n_aug, burn=self.burn)
            else:
                m, v = update_ranking(m, v, list(order), beta2=self.beta2, base=self.base)
        elif winner is not None:                                   # winner-only fallback (see order note)
            m, v, _ = update_winner(m, v, int(winner), beta2=self.beta2, base=self.base)
        self._store(ids, m, v, t)

    def rating(self, i: str) -> Optional[Tuple[float, float]]:
        s = self.state.get(i)
        return None if s is None else (s.mean, float(np.sqrt(s.var)))


def _winner_of(contest) -> Optional[int]:
    if contest.get("scores") is not None:
        return int(np.argmin(np.asarray(contest["scores"], float)))   # min-wins
    if contest.get("order") is not None:
        return int(contest["order"][0])
    if contest.get("winner") is not None:
        return int(contest["winner"])
    return None


def walk_forward(contests: Sequence[dict], warmup: int = 20, market_arm: bool = True,
                 **params) -> dict:
    """Walk-forward: predict each contest from the history strictly before it
    (log-loss), then fold it in. Same return shape as ``history.walk_forward``,
    O(entrants) per contest. Each contest: 'runners', optional 't', a result as
    'scores'(transformed performances)/'order'/'winner', optional 'p_market'.
    """
    trk = AbilityTracker(**params)
    ll_model = ll_market = 0.0; n_scored = 0
    recs: List[dict] = []
    for i, c in enumerate(contests):
        ids = c["runners"]; t = float(c.get("t", 0.0)); w = _winner_of(c)
        if i >= warmup and w is not None:
            p = trk.predict(ids, t)
            ll_model += float(np.log(max(p[w], 1e-12)))
            rec = {"i": i, "p_model": p, "winner": w}
            if c.get("p_market") is not None:
                pm = np.asarray(c["p_market"], float); pm = pm / pm.sum()
                ll_market += float(np.log(max(pm[w], 1e-12))); rec["p_market"] = pm
            recs.append(rec); n_scored += 1
        trk.observe(ids, t, scores=c.get("scores"), order=c.get("order"),
                    winner=c.get("winner"),
                    prices=(c.get("p_market") if market_arm else None))
    return {"log_loss_model": ll_model, "log_loss_market": ll_market,
            "n_scored": n_scored, "records": recs, "tracker": trk}
