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

BLOCK CORRELATION (same-group entrants). Pass ``groups`` with a contest and
set ``rho`` and every observer prices the blocked model, variance-
preserving: x_j = s_j + sqrt(rho beta2) z_g(j) + sqrt((1 - rho) beta2) eps_j,
so each marginal keeps variance beta2 and only the joint changes. Fit and
prediction use the same V. rho is selected by the filter's own evidence,
``tune_block_rho`` -- the sum of log P(observation) every update already
returns -- which never sees a target event.

WHAT THE CODE DOES, and what is measured. The shared component z_g is
redrawn at every contest, a mean-zero draw common to the group's members
in that contest; a persistent group advantage is a different object and
belongs in the means (a group column in the design of fit_design_ratings,
or a shared offset), whatever this term is set to. The Formula 1
measurement that motivated the feature (bandits exp33 / exp34,
2026-09-12) is the caution: teammates share a car and the data carry the
correlation (training evidence prefers rho = 0.4 to independence by 28.7
nats over 158 races; same-team 1-2 finishes 0.274 observed against 0.123
under independence), yet fitted and priced consistently under the blocked
likelihood the winner log-loss on 106 held-out races was WORSE by +0.053
[+0.019, +0.087], and the joint same-team advantage fell to -0.019
[-0.052, +0.011] and is withdrawn. The signature is confident
mis-ranking rather than blurring: the mean probability on the actual
winner barely moves (0.319 to 0.311) while the spread of log
probabilities widens, so the loss sits in a minority of confident misses.
Two mechanisms were tested and rejected on that output -- damage does not
concentrate in races won by persistently strong teams, and predictions
do not smear toward uniform -- so no mechanism is claimed. No performance
claim is made for this feature in either direction; the open question is
a persistent group term in the mean fitted alongside rho.

Cost: one factor dimension per group of two or more in the contest; two or
fewer ride a 7^r Gauss-Hermite tensor (Qf), more ride 2**nodes_log2 Sobol
nodes (about 15 s per K = 8 field with four groups on one core; 15-29 s
for a 20-entrant field with ten pairs). Singletons cost nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from .nway import (update_winner, update_ranking, update_ranking_exact,
                   update_winner_correlated, update_order_correlated,
                   predictive_win_probabilities, _order_pass, _predictive_curves)
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


def block_loadings(groups, rho, beta2=1.0):
    """Variance-preserving block loadings for a field with group labels.

    Returns (V, beta2_vec): V is (n, G) with sqrt(rho * beta2) on the members
    of each group of size >= 2 (one column per such group; None when there
    are none or rho == 0), and beta2_vec (n,) is the idiosyncratic variance,
    (1 - rho) * beta2 for grouped entrants and beta2 for singletons, so every
    marginal keeps variance beta2 whatever rho is. A label of None means "no
    group". Adding shared noise ON TOP of unit noise instead (lambda z + eps)
    inflates the total to 1 + lambda^2 while the means were fitted at unit
    variance, and the marginals flatten mechanically -- measured at +0.077
    on a winner control before this parameterisation cut it to +0.017.
    """
    groups = list(groups)
    n = len(groups)
    b2 = np.full(n, float(beta2))
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    members: Dict[object, List[int]] = {}
    for j, g in enumerate(groups):
        if g is None:
            continue
        members.setdefault(g, []).append(j)
    multi = [ix for ix in members.values() if len(ix) >= 2]
    if rho <= 0.0 or not multi:
        return None, b2
    V = np.zeros((n, len(multi)))
    for k, ix in enumerate(multi):
        V[ix, k] = np.sqrt(rho * beta2)
        b2[ix] = (1.0 - rho) * beta2
    return V, b2


def _order_evidence(m, v, order, beta2, base):
    """log P(order) under the prior predictive N(m, v) + base noise: the
    evidence the independent order update does not return itself."""
    m = np.asarray(m, dtype=float); v = np.asarray(v, dtype=float)
    sd = np.sqrt(v + beta2)
    curves = None if base == "normal" else _predictive_curves(v, beta2, base)
    lp, _ = _order_pass(m, sd, np.asarray(order, dtype=int), base=base,
                        curves=curves)
    return float(lp)


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
    rho              : share of performance variance same-group entrants have
                       in common (block_loadings); active only for contests
                       observed or predicted with ``groups``.
    Qf               : Gauss-Hermite nodes per factor dimension for the
                       block-correlated updates at rank r <= 2 (two groups
                       of two or more in a contest).
    nodes_log2       : log2 of the scrambled-Sobol node count at r >= 3,
                       where Qf is inert. Cost per blocked update goes as
                       nodes x K^2 and is the real ceiling on block
                       correlation at double-digit group counts: an F1-shaped
                       field (20 entrants, 10 two-car teams, r = 10) costs
                       15-29 s per update at the default 1024 nodes, and
                       neither update_order_full with a block-diagonal
                       covariance nor the winner update is cheaper (bandits
                       measurement, 2026-09-11).

    ``evidence`` accumulates log P(observation) over everything observed --
    market prices, scores, orders and winners, independent or blocked -- the
    filter's marginal likelihood, for tuning rho (tune_block_rho) or any
    other knob without a held-out target.
    """

    def __init__(self, drift: float = 0.02, drift_exp: float = 0.5,
                 init_var: float = 4.0, beta2: float = 1.0, tau2: float = 0.25,
                 base: str = "normal", order_method: str = "exact",
                 n_aug: int = 60, burn: int = 20, seed: int = 0,
                 rho: float = 0.0, Qf: int = 7, nodes_log2: int = 10):
        self.drift = float(drift); self.drift_exp = float(drift_exp)
        self.init_var = float(init_var)
        self.beta2 = float(beta2); self.tau2 = float(tau2)
        self.base = base
        if not 0.0 <= float(rho) < 1.0:
            raise ValueError("rho must lie in [0, 1)")
        self.rho = float(rho); self.Qf = int(Qf); self.nodes_log2 = int(nodes_log2)
        self.evidence = 0.0
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

    def _blocks(self, groups, n):
        if groups is None:
            return None, self.beta2
        groups = list(groups)
        if len(groups) != n:
            raise ValueError("groups must have one label per entrant")
        V, b2 = block_loadings(groups, self.rho, self.beta2)
        return V, (self.beta2 if V is None else b2)

    # -- prediction ----------------------------------------------------------
    def predict(self, ids: Sequence[str], t: float, points: int = 257,
                groups: Optional[Sequence] = None) -> np.ndarray:
        """Predictive win probabilities among `ids` at t, belief uncertainty
        marginalised into the performance variance (same factor forward map).
        With ``groups`` and rho > 0 the field is priced under the blocked
        model that observe() fits -- the same V, never a different one."""
        m, v = self._gather(ids, t)
        V, b2 = self._blocks(groups, len(m))
        if self.base != "normal":
            # the belief convolved with the base noise, on the updates'
            # own node rule: predict and evidence agree on every base
            return predictive_win_probabilities(m, v, beta2=b2, base=self.base, V=V,
                                                Qf=self.Qf, nodes_log2=self.nodes_log2,
                                                points=points)
        return race_probabilities(-m, V=V, D=b2 + v, base=self.base, points=points)

    # -- observation ---------------------------------------------------------
    def observe(self, ids: Sequence[str], t: float, scores: Optional[Sequence[float]] = None,
                order: Optional[Sequence[int]] = None, winner: Optional[int] = None,
                prices: Optional[Sequence[float]] = None,
                groups: Optional[Sequence] = None) -> None:
        """Fold one contest's evidence. BOTH observers (market prices, results)
        are contrast evidence through the same conjugate update. ``groups``
        (one label per entrant, None for ungrouped) switches every observer to
        the block-correlated model when rho > 0; log P(observation) is added
        to ``self.evidence`` on every path."""
        m, v = self._gather(ids, t)
        V, b2 = self._blocks(groups, len(m))
        if prices is not None:                                     # market observer
            # prices are inverted under the same (blocked or independent)
            # performance model the outcome is priced with
            model = {} if V is None else {"V": V, "D": b2}
            m, v, lz = update_market(m, v, np.asarray(prices, float), tau2=self.tau2, **model)
            self.evidence += lz
        if scores is not None:                                     # results observer, MAGNITUDES
            # The consistent path whenever performance magnitudes are observed.
            # Transformed performances are a linear-Gaussian observation
            # of ability, so update_market's conjugate contrast update is exact --
            # verified unbiased (single race, performance-noise-only) and calibrated
            # under drift on synthetic data.
            if V is None:
                m, v, lz = update_market(m, v, np.asarray(scores, float), tau2=self.beta2, invert=_neg)
            else:
                # correlated contrast noise P (V V' + diag b2) P: the full-covariance
                # conjugate node on a diagonal prior, marginals kept (same ADF
                # projection the independent path makes)
                from .history import update_margins_full
                m, S, lz = update_margins_full(m, np.diag(v), scores=-np.asarray(scores, float),
                                               V=V, beta2=b2)
                v = np.maximum(np.diag(S).copy(), 1e-6)
            self.evidence += lz
        elif order is not None:                                    # ORDER-ONLY fallback
            # Ranks censor the magnitudes. 'exact' uses winning's exact win-node
            # ranking factor (matches the Gibbs reference); 'moment' is the fast
            # biased moment-match; 'augmented' is the slow Gibbs reference itself.
            # With magnitudes available, `scores` is the right path, not this.
            order = list(order)
            if V is not None:
                m, v, lz = update_order_correlated(m, v, order, V, beta2=b2, Qf=self.Qf,
                                                   base=self.base, nodes_log2=self.nodes_log2)
            else:
                lz = _order_evidence(m, v, order, self.beta2, self.base)
                if self.order_method == "exact":
                    m, v = update_ranking_exact(m, v, order, beta2=self.beta2, base=self.base)
                elif self.order_method == "augmented":
                    m, v = order_augmented(m, v, order, self.beta2, self.rng,
                                           n_aug=self.n_aug, burn=self.burn)
                else:
                    m, v = update_ranking(m, v, order, beta2=self.beta2, base=self.base)
            self.evidence += lz
        elif winner is not None:                                   # winner-only fallback (see order note)
            if V is not None:
                m, v, lz = update_winner_correlated(m, v, int(winner), V, beta2=b2, Qf=self.Qf,
                                                    base=self.base, nodes_log2=self.nodes_log2)
            else:
                m, v, p = update_winner(m, v, int(winner), beta2=self.beta2, base=self.base)
                lz = float(np.log(max(float(p), 1e-300)))
            self.evidence += lz
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
    'scores'(transformed performances)/'order'/'winner', optional 'p_market',
    optional 'groups' (block correlation, see AbilityTracker). The tracker's
    accumulated ``evidence`` is returned alongside.
    """
    trk = AbilityTracker(**params)
    ll_model = ll_market = 0.0; n_scored = 0
    recs: List[dict] = []
    for i, c in enumerate(contests):
        ids = c["runners"]; t = float(c.get("t", 0.0)); w = _winner_of(c)
        groups = c.get("groups")
        if i >= warmup and w is not None:
            p = trk.predict(ids, t, groups=groups)
            ll_model += float(np.log(max(p[w], 1e-12)))
            rec = {"i": i, "p_model": p, "winner": w}
            if c.get("p_market") is not None:
                pm = np.asarray(c["p_market"], float); pm = pm / pm.sum()
                ll_market += float(np.log(max(pm[w], 1e-12))); rec["p_market"] = pm
            recs.append(rec); n_scored += 1
        trk.observe(ids, t, scores=c.get("scores"), order=c.get("order"),
                    winner=c.get("winner"),
                    prices=(c.get("p_market") if market_arm else None),
                    groups=groups)
    return {"log_loss_model": ll_model, "log_loss_market": ll_market,
            "n_scored": n_scored, "records": recs, "tracker": trk,
            "evidence": trk.evidence}


def tune_block_rho(contests: Sequence[dict], rho_grid=(0.0, 0.1, 0.25, 0.4, 0.6),
                   market_arm: bool = True, **params) -> dict:
    """Select the block correlation by the filter's marginal likelihood: one
    walk per rho over contests carrying 'groups', summing log P(observation)
    from every update (nothing held out, no target event consulted). Returns
    rho_grid, evidence (per rho), rho (the argmax) and at_edge -- a rho at
    either end of the grid is a capped rival, not a tuned one; widen the
    grid. ``params`` go to AbilityTracker (drift, beta2, base, Qf, ...).

    On Formula 1 this evidence preferred rho = 0.4 to independence by
    28.7 nats -- the correlation is in the data -- and the blocked model
    still predicted held-out winners worse (module docstring). A
    likelihood ratio says the structure is present, not that pricing it
    this way predicts better. Cost: one blocked walk per grid point
    (module docstring)."""
    evidence = []
    for rho in rho_grid:
        trk = AbilityTracker(rho=float(rho), **params)
        for c in contests:
            trk.observe(c["runners"], float(c.get("t", 0.0)),
                        scores=c.get("scores"), order=c.get("order"),
                        winner=c.get("winner"),
                        prices=(c.get("p_market") if market_arm else None),
                        groups=c.get("groups"))
        evidence.append(trk.evidence)
    best = int(np.argmax(evidence))
    return {"rho_grid": tuple(float(r) for r in rho_grid),
            "evidence": np.asarray(evidence), "rho": float(rho_grid[best]),
            "at_edge": best in (0, len(rho_grid) - 1)}
