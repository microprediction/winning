"""Point-in-time ability estimation from a racing history.

Peter's spec: an entire history of results, market prices, finish
positions and lengths should yield relative abilities point in time.
The pieces: the per-race two-source update (winning.ratings.market),
plus the two additions this module supplies --

TIME DIFFUSION between races: skills follow an OU-style drift,
s_{t+dt} = lam s_t + noise with lam = exp(-dt/timescale), so beliefs
decay toward the population prior and uncertainty reinflates. Diagonal
and full-covariance members.

LENGTHS (margins): the nicest observation of all. Margins are
performance CONTRASTS observed directly: with X = s + V f + eps and
margins converted to performance units by lengths_scale, the
observation is y = P X = P s + P(V f + eps) -- linear-Gaussian with
contrast noise covariance P (V V' + beta2 I) P. Conjugate, exact, no
quadrature. Modeling rule: full margins SUBSUME the finishing order
(the order is their sign pattern), so a race consumes prices + margins
when lengths exist and prices + order/winner when they do not --
never both, which would double-count.

Every update returns logZ, so the history's total evidence is available
for tuning tau2 (market noise), beta2, drift timescale and
lengths_scale by marginal likelihood.
"""

from __future__ import annotations

import numpy as np

from .full import _psd_repair


def diffuse(m, v, dt=1.0, timescale=200.0, prior_mean=0.0, prior_var=1.0):
    """OU drift toward the population prior (diagonal belief):
    lam = exp(-dt/timescale); m <- prior + lam (m - prior),
    v <- lam^2 v + (1 - lam^2) prior_var."""
    lam = float(np.exp(-dt / timescale))
    m = prior_mean + lam * (np.asarray(m, dtype=float) - prior_mean)
    v = lam * lam * np.asarray(v, dtype=float) + (1 - lam * lam) * prior_var
    return m, v


def diffuse_full(m, S, dt=1.0, timescale=200.0, prior_mean=0.0,
                 prior_var=1.0):
    """OU drift for a full-covariance belief: cross-correlations decay
    with the same factor, uncertainty reinflates toward the independent
    population prior."""
    lam = float(np.exp(-dt / timescale))
    m = prior_mean + lam * (np.asarray(m, dtype=float) - prior_mean)
    S = lam * lam * np.asarray(S, dtype=float) \
        + (1 - lam * lam) * prior_var * np.eye(len(m))
    return m, S


def update_margins_full(m, S, margins=None, V=None, beta2=1.0,
                        lengths_scale=1.0, meas_var=0.0, scores=None,
                        transform=None):
    """Conjugate cardinal-performance update, full covariance:
    y = P s + w with w ~ N(0, P (V V' + beta2 I + meas_var I) P) on the
    contrast space. Two input conventions: margins= (lengths behind the
    winner, LOWER is better, negated internally) or scores= (points /
    goals / times-negated, HIGHER is better, used as-is after scaling).
    transform= applies a sub-linear margin transform (a compression
    scale c for c*asinh(L/c), or a callable) with its Jacobian added to
    logZ, so evidence-based tuning of c stays honest: raw lengths are
    superlinear in performance deficit (eased finishers, collapsing
    pace), and staying Gaussian on TRANSFORMED margins is the
    long-right-tail model placed in the measurement map. Returns
    (m_post, S_post, logZ). The team form with an assignment matrix is
    teams.update_team_margins_full; this is A = I."""
    from .teams import _cardinal_observation, _lift_contrast_update, _noise_cov
    m = np.asarray(m, dtype=float)
    n = len(m)
    y, log_jac = _cardinal_observation(margins, scores, lengths_scale, transform)
    Cn = _noise_cov(n, beta2, meas_var, V)
    return _lift_contrast_update(m, S, np.eye(n), y, Cn, log_jac)


def rate_history(races, ids=None, prior_mean=0.0, prior_var=1.0,
                 timescale=200.0, tau2=0.25, beta2=1.0, lengths_scale=0.2,
                 meas_var=0.0, transform=None, state=None,
                 return_state=False, base="normal"):
    """Forward filter over a racing history (full-covariance belief),
    every observation lifted through the full state (a proper
    full-state filter: exact for conjugate observations, including the
    cross-covariances between a race's entrants and everyone else).

    races: iterable of dicts with keys
      't'        -- time (any unit consistent with timescale)
      'runners'  -- list of runner identifiers
      and any of:
      'p_market' -- market win probabilities (pre-race)
      'margins'  -- lengths behind the winner, aligned with runners
                    (subsumes the order; preferred when present)
      'scores'   -- cardinal performance, higher better (goals, points,
                    negated times); same conjugate node, opposite sign
      'order'    -- finishing order, indices into runners, best first
      'winner'   -- index into runners
      optional 'V' -- (k, r) loadings for the race's k runners

    Returns (ratings, history_logZ): ratings maps runner id ->
    (mean, sd) at the final time; history_logZ is the total evidence,
    the objective for tuning the hyperparameters.
    """
    if state is not None:
        index = dict(state["index"])
        m = np.asarray(state["m"], dtype=float).copy()
        S = np.asarray(state["S"], dtype=float).copy()
        t_last = state["t"]
    else:
        all_ids = ids
        if all_ids is None:
            seen = []
            for race in races:
                for rid in race["runners"]:
                    if rid not in seen:
                        seen.append(rid)
            all_ids = seen
        index = {rid: i for i, rid in enumerate(all_ids)}
        n = len(index)
        m = np.full(n, float(prior_mean))
        S = np.eye(n) * float(prior_var)
        t_last = None
    from .teams import (update_team_margins_full, update_team_market_full,
                        update_team_order_full, update_team_winner_full)
    total_logZ = 0.0
    n_all = len(index)
    for race in races:
        t = float(race.get("t", 0.0))
        if t_last is not None and t > t_last:
            m, S = diffuse_full(m, S, dt=t - t_last, timescale=timescale,
                                prior_mean=prior_mean, prior_var=prior_var)
        t_last = t
        idx = np.array([index[r] for r in race["runners"]])
        # every observation goes through the FULL state via the race's
        # selection matrix: the entrants' block is what the observation
        # sees, but the Kalman lift S A' moves every entity's mean and
        # every cross-covariance. Updating the sub-block alone and
        # writing it back left entrant/non-entrant covariances stale --
        # exact on disjoint fields, wrong on overlapping ones even for
        # conjugate scores (means off by 1.2 after 60 races of 5 among
        # 12 on a static world; contrast coverage 0.37), and the
        # evidence order-dependent (verifier, 2026-09-11).
        A = np.zeros((len(idx), n_all)); A[np.arange(len(idx)), idx] = 1.0
        V = race.get("V")
        if race.get("p_market") is not None:
            m, S, lz = update_team_market_full(
                m, S, A, race["p_market"], tau2=tau2,
                **({} if V is None else
                   {"V": np.atleast_2d(np.asarray(V, float)),
                    "D": np.full(len(idx), beta2)}))
            total_logZ += lz
        if race.get("margins") is not None or race.get("scores") is not None:
            m, S, lz = update_team_margins_full(
                m, S, A, margins=race.get("margins"),
                scores=race.get("scores"), V=V, beta2=beta2,
                lengths_scale=lengths_scale, meas_var=meas_var,
                transform=transform)
            total_logZ += lz
        elif race.get("order") is not None:
            m, S, lz = update_team_order_full(m, S, A, race["order"], V=V,
                                              beta2=beta2, base=base)
            total_logZ += lz
        elif race.get("winner") is not None:
            if base != "normal":
                raise NotImplementedError(
                    "winner-only observations are Gaussian only in the "
                    "full-covariance filter (the shared-field winner pass is "
                    "analytic in the Gaussian log domain); pass the finishing "
                    "order, where a non-normal base carries information")
            m, S, lz = update_team_winner_full(m, S, A, race["winner"], V=V,
                                               beta2=beta2)
            total_logZ += lz
        S = _psd_repair(S)
    sd = np.sqrt(np.maximum(np.diag(S), 0.0))
    ratings = {rid: (float(m[i]), float(sd[i])) for rid, i in index.items()}
    if return_state:
        return ratings, float(total_logZ), \
            {"m": m, "S": S, "t": t_last, "index": index,
             "prior_mean": prior_mean, "prior_var": prior_var,
             "timescale": timescale}
    return ratings, float(total_logZ)


def predict_race(state, runners, t=None, V=None, beta2=1.0, points=257,
                 base="normal"):
    """Predictive win probabilities (and fair odds) for the NEXT race.

    state: from rate_history(..., return_state=True). runners: ids (new
    ids get the population prior). t: race time (diffuses the belief
    forward; None prices at the state's time). The predictive
    performance covariance is S_field + V V' + beta2 I -- belief
    uncertainty PLUS race noise, which is what makes these genuine
    predictive odds rather than point estimates raced against each
    other. Returns (p, odds) with odds = 1/p (no overround).
    """
    from ..factor.races import race_probabilities

    m = np.asarray(state["m"], dtype=float)
    S = np.asarray(state["S"], dtype=float)
    if t is not None and state["t"] is not None and t > state["t"]:
        m, S = diffuse_full(m, S, dt=t - state["t"],
                            timescale=state.get("timescale", 200.0),
                            prior_mean=state.get("prior_mean", 0.0),
                            prior_var=state.get("prior_var", 1.0))
    k = len(runners)
    mu = np.full(k, float(state.get("prior_mean", 0.0)))
    Sf = np.eye(k) * float(state.get("prior_var", 1.0))
    known = [(a, state["index"][r]) for a, r in enumerate(runners)
             if r in state["index"]]
    for a, i in known:
        mu[a] = m[i]
        for b, j in known:
            Sf[a, b] = S[i, j]
    B = np.broadcast_to(np.asarray(beta2, dtype=float), (k,)).astype(float)
    if base != "normal":
        # non-normal noise: the belief split into a lattice-borne
        # diagonal part convolved with the base noise plus quadratured
        # loadings, as the updates price it (nway.predictive_win_probabilities)
        from .nway import predictive_win_probabilities
        p = predictive_win_probabilities(mu, S=Sf, beta2=B, base=base, V=V, points=points)
        return p, 1.0 / np.maximum(p, 1e-12)
    C = Sf + np.diag(B)
    if V is not None:
        Vm = np.atleast_2d(np.asarray(V, dtype=float))
        if Vm.shape[0] != k:
            Vm = Vm.T
        C = C + Vm @ Vm.T
    p = race_probabilities(-mu, cov=C, points=points, base=base)
    return p, 1.0 / np.maximum(p, 1e-12)


def tune_history(races, tune=("tau2", "beta2", "timescale",
                              "lengths_scale"), maxiter=60, **fixed):
    """Fit hyperparameters by maximizing the filter's total evidence
    (Nelder-Mead over log-parameters; the objective is the logZ stream
    every update already returns). Returns (best_params, best_logZ)."""
    from scipy.optimize import minimize

    defaults = dict(tau2=0.25, beta2=1.0, timescale=200.0,
                    lengths_scale=0.2)
    defaults.update(fixed)
    x0 = np.log([defaults[k] for k in tune])

    def neg_evidence(x):
        params = dict(defaults)
        params.update({k: float(np.exp(v)) for k, v in zip(tune, x)})
        try:
            _, lz = rate_history(races, **params)
        except Exception:
            return 1e12
        return -lz

    res = minimize(neg_evidence, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3,
                            "fatol": 1e-3})
    best = dict(defaults)
    best.update({k: float(np.exp(v)) for k, v in zip(tune, res.x)})
    return best, -float(res.fun)


def walk_forward(races, warmup=20, market_arm=True, V_key="V",
                 **params):
    """Walk-forward evaluation: predict each race's winner from the
    history strictly before it, score by log-loss. Reports the
    PURE-FORM arm (no market input to the prediction; the filter still
    consumes prices historically unless market_arm=False strips them),
    and the market's own log-loss as the benchmark when prices are
    present. Returns a dict of totals and per-race records."""
    races = list(races)
    state = None
    recs = []
    ll_model = ll_market = 0.0
    n_scored = 0
    for i, race in enumerate(races):
        if i >= warmup and (race.get("winner") is not None
                            or race.get("order") is not None
                            or race.get("margins") is not None):
            p, _ = predict_race(state, race["runners"],
                                t=float(race.get("t", 0.0)),
                                V=race.get(V_key),
                                beta2=params.get("beta2", 1.0))
            if race.get("order") is not None:
                w = int(race["order"][0])
            elif race.get("winner") is not None:
                w = int(race["winner"])
            else:
                w = int(np.argmin(np.asarray(race["margins"])))
            ll_model += float(np.log(max(p[w], 1e-12)))
            rec = {"i": i, "p_model": p, "winner": w}
            if race.get("p_market") is not None:
                pm = np.asarray(race["p_market"], dtype=float)
                pm = pm / pm.sum()
                ll_market += float(np.log(max(pm[w], 1e-12)))
                rec["p_market"] = pm
            recs.append(rec)
            n_scored += 1
        feed = dict(race)
        if not market_arm:
            feed.pop("p_market", None)
        _, _, state = rate_history([feed], state=state, return_state=True,
                                   **params) if state is not None else             rate_history([feed], return_state=True,
                         ids=sorted({r for rc in races
                                     for r in rc["runners"]}),
                         **params)
    return {"log_loss_model": ll_model, "log_loss_market": ll_market,
            "n_scored": n_scored, "records": recs}
