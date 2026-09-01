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

from .full import _psd_repair, update_market_full, update_order_full, \
    update_winner_full
from .market import update_race


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


def _margin_obs(margins, ref, n, lengths_scale):
    """Performance contrasts from finishing margins. margins[i] is
    runner i's distance behind the reference (winner: 0), in lengths;
    lengths_scale converts lengths to performance units (max-wins:
    further behind = lower performance)."""
    y = -np.asarray(margins, dtype=float) * float(lengths_scale)
    return y - y.mean()


def update_margins_full(m, S, margins=None, V=None, beta2=1.0,
                        lengths_scale=1.0, meas_var=0.0, scores=None,
                        transform=None):
    """Conjugate cardinal-performance update, full covariance:
    y = P s + w with w ~ N(0, P (V V' + beta2 I + meas_var I) P) on the
    contrast space. Two input conventions: margins= (lengths behind the
    winner, LOWER is better, negated internally) or scores= (points /
    goals / times-negated, HIGHER is better, used as-is after scaling).
    Returns (m_post, S_post, logZ)."""
    m = np.asarray(m, dtype=float)
    S = _psd_repair(np.asarray(S, dtype=float))
    n = len(m)
    if (margins is None) == (scores is None):
        raise ValueError("pass exactly one of margins= or scores=")
    log_jac = 0.0
    if scores is not None:
        y = np.asarray(scores, dtype=float) * float(lengths_scale)
        y = y - y.mean()
    else:
        Lm = np.asarray(margins, dtype=float)
        if transform is not None:
            # sub-linear margin transform (Peter): raw lengths are
            # superlinear in performance deficit (eased horses,
            # collapsing pace), so staying Gaussian on TRANSFORMED
            # margins IS the long-right-tail model, placed in the
            # measurement map. asinh(L/c)*c has derivative 1 at zero
            # (close finishes stay linear-Gaussian, lengths_scale keeps
            # its meaning) and is logarithmic in the tail (blowouts
            # discounted). Evidence-based tuning of c REQUIRES the
            # change-of-variables Jacobian added to logZ -- without it,
            # more compressive transforms win spuriously by shrinking
            # the data -- so it is included here.
            if callable(transform):
                c = None
                Lt = np.asarray(transform(Lm), dtype=float)
                dL = 1e-6 * (1.0 + np.abs(Lm))
                deriv = (np.asarray(transform(Lm + dL), dtype=float)
                         - Lt) / dL
            else:
                c = float(transform)
                Lt = c * np.arcsinh(Lm / c)
                deriv = 1.0 / np.sqrt(1.0 + (Lm / c) ** 2)
            log_jac = float(np.sum(np.log(np.maximum(
                deriv * float(lengths_scale), 1e-300))))
            Lm = Lt
        y = _margin_obs(Lm, 0, n, lengths_scale)
    P = np.eye(n) - np.ones((n, n)) / n
    B = np.broadcast_to(np.asarray(beta2, dtype=float), (n,)).astype(float)
    Cn = np.diag(B + float(meas_var))
    if V is not None:
        Vm = np.atleast_2d(np.asarray(V, dtype=float))
        if Vm.shape[0] != n:
            Vm = Vm.T
        Cn = Cn + Vm @ Vm.T
    N = P @ Cn @ P                       # contrast noise covariance
    M = P @ S @ P + N                    # innovation covariance (contrasts)
    lam, U = np.linalg.eigh(M)
    keep = lam > 1e-10 * max(lam.max(), 1e-300)
    Uk = U[:, keep]
    r = y - P @ m
    z = Uk.T @ r
    # Kalman gain restricted to the observed subspace
    K = S @ P @ (Uk * (1.0 / lam[keep])) @ Uk.T
    m_new = m + K @ r
    S_new = _psd_repair(S - K @ (P @ S))
    logZ = float(-0.5 * (np.sum(z * z / lam[keep])
                         + np.sum(np.log(lam[keep]))
                         + keep.sum() * np.log(2.0 * np.pi))) + log_jac
    return m_new, S_new, logZ


def rate_history(races, ids=None, prior_mean=0.0, prior_var=1.0,
                 timescale=200.0, tau2=0.25, beta2=1.0, lengths_scale=0.2,
                 meas_var=0.0, transform=None, state=None,
                 return_state=False, base="normal"):
    """Forward filter over a racing history (full-covariance belief).

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
    total_logZ = 0.0
    for race in races:
        t = float(race.get("t", 0.0))
        if t_last is not None and t > t_last:
            m, S = diffuse_full(m, S, dt=t - t_last, timescale=timescale,
                                prior_mean=prior_mean, prior_var=prior_var)
        t_last = t
        idx = np.array([index[r] for r in race["runners"]])
        mk, Sk = m[idx], S[np.ix_(idx, idx)]
        V = race.get("V")
        if race.get("p_market") is not None:
            mk, Sk, lz = update_market_full(
                mk, Sk, race["p_market"], tau2=tau2,
                **({} if V is None else
                   {"V": np.atleast_2d(np.asarray(V, float)),
                    "D": np.full(len(idx), beta2)}))
            total_logZ += lz
        if race.get("margins") is not None or race.get("scores") is not None:
            mk, Sk, lz = update_margins_full(
                mk, Sk, margins=race.get("margins"),
                scores=race.get("scores"), V=V, beta2=beta2,
                lengths_scale=lengths_scale, meas_var=meas_var,
                transform=transform)
            total_logZ += lz
        elif race.get("order") is not None:
            mk, Sk, lz = update_order_full(mk, Sk, race["order"], V=V,
                                           beta2=beta2, base=base)
            total_logZ += lz
        elif race.get("winner") is not None:
            mk, Sk, lz = update_winner_full(mk, Sk, race["winner"], V=V,
                                            beta2=beta2, base=base)
            total_logZ += lz
        m[idx] = mk
        S[np.ix_(idx, idx)] = Sk
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
        hist = [race] if state is not None else races[:i + 1]
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
