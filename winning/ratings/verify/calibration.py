"""Simulation-based calibration of the FILTERS (I9 of the plan).

Ability paths are drawn from the filter's own temporal model, contests
are simulated from the observation model, the filter is run, and the
posterior is scored on what it is meant to know. The score is taken on
pairwise CONTRASTS of entities that met: only contrasts are identified
by contest outcomes, and the diagonal tracker's marginal is overconfident
by exactly the unidentified common level (a rank-one, perfectly
correlated component a diagonal state cannot hold), while its contrasts
are calibrated or slightly conservative (F3). z_ij = ((s_i - s_j) -
(m_i - m_j)) / sqrt(v_i + v_j - 2 c_ij), with c_ij = 0 for the tracker
and S_ij for rate_history. Statistics per replicate world (mean z^2,
coverage at 1.96), aggregated over R worlds with a cluster standard
error, against pre-registered bands; UNDERPOWERED when three standard
errors do not fit inside the band.

At drift 0 both filters share one static generative model whose exact
joint posterior is closed form for score observations, so any
miscalibration there is the filter's own (the diagonal projection for
the tracker; F1 -- the entrant sub-block update -- for rate_history,
which is therefore MEASURED until it is fixed). Drift cells simulate
each filter's own increments and are never compared across filters.
"""

from __future__ import annotations

import numpy as np

from .core import Result, check


def _world(rng, E, T, K, init_var, beta2, drift=0.0, drift_exp=0.5, kind="scores"):
    """Abilities s_e ~ N(0, init_var); at each appearance an entity's
    ability moves by N(0, drift * gap^drift_exp), the tracker's own
    variance growth made literal; performance x = s + sqrt(beta2) eps
    (max-wins). Returns (contests, truth) with truth the abilities at
    each entity's last appearance."""
    s = rng.normal(0.0, np.sqrt(init_var), E)
    last = np.zeros(E)
    contests, truth = [], s.copy()
    for t in range(1, T + 1):
        ids = rng.choice(E, K, replace=False)
        if drift > 0:
            gap = t - last[ids]
            s[ids] = s[ids] + rng.normal(0.0, 1.0, K) * np.sqrt(drift * gap ** drift_exp)
        last[ids] = t
        x = s[ids] + np.sqrt(beta2) * rng.normal(size=K)
        c = {"runners": [str(i) for i in ids], "t": float(t)}
        if kind == "scores":
            c["scores"] = (-x).tolist()          # tracker scores are min-wins
        elif kind == "orders":
            c["order"] = list(np.argsort(-x))
        else:
            c["winner"] = int(np.argmax(x))
        contests.append(c)
        truth[ids] = s[ids]
    return contests, truth


def _contrast_z(truth, m, v, S=None, pairs=None):
    i, j = pairs
    d = (truth[i] - truth[j]) - (m[i] - m[j])
    var = v[i] + v[j] - (0.0 if S is None else 2.0 * S[i, j])
    return d / np.sqrt(np.maximum(var, 1e-12))


def _aggregate(zs):
    z2 = np.array([float(np.mean(z * z)) for z in zs])
    cov = np.array([float(np.mean(np.abs(z) < 1.96)) for z in zs])
    R = len(zs)
    return {"z2": float(z2.mean()), "z2_se": float(z2.std(ddof=1) / np.sqrt(R)),
            "coverage": float(cov.mean()), "cov_se": float(cov.std(ddof=1) / np.sqrt(R)),
            "R": R, "n_z": int(sum(len(z) for z in zs))}


def _band_result(ctx, name, agg, key, regime, measured=False):
    m = ctx.mark(key)
    if m is None or measured:
        return [Result(f"{name}.z2", "calibration", "MEASURED", statistic=agg["z2"],
                       se=agg["z2_se"], regime=regime, n=agg["n_z"],
                       detail=f"mean contrast z^2 (coverage {agg['coverage']:.3f}); "
                              f"{'no mark yet' if m is None else 'measured pending fix'}",
                       extras=agg)]
    lo, hi = m["z2_band"]; clo, chi = m["coverage_band"]
    inside = lo <= agg["z2"] <= hi and clo <= agg["coverage"] <= chi
    thin = 3 * agg["z2_se"] > (hi - lo) / 2 or 3 * agg["cov_se"] > (chi - clo) / 2
    if not inside:
        verdict, detail = "FAIL", (f"z^2 {agg['z2']:.3f} outside [{lo}, {hi}] or coverage "
                                   f"{agg['coverage']:.3f} outside [{clo}, {chi}]")
    elif thin:
        verdict, detail = "UNDERPOWERED", f"3 SE ({3 * agg['z2_se']:.3f}) exceeds the band half-width"
    else:
        verdict, detail = "ok", ""
    return [Result(f"{name}.z2", "calibration", verdict, statistic=agg["z2"], se=agg["z2_se"],
                   tolerance=hi, regime=regime, n=agg["n_z"], detail=detail, extras=agg)]


def _run_tracker(ctx, kind, R, drift=0.0, drift_exp=0.5, E=12, T=60, K=5,
                 init_var=1.0, beta2=1.0):
    from ..tracker import AbilityTracker
    zs, zm = [], []
    for r in range(R):
        rng = ctx.rng(f"{kind}:{drift}:{drift_exp}:{r}")
        contests, truth = _world(rng, E, T, K, init_var, beta2, drift, drift_exp, kind)
        trk = AbilityTracker(drift=drift, drift_exp=drift_exp, init_var=init_var, beta2=beta2)
        met = np.zeros((E, E), dtype=bool)
        for c in contests:
            ids = [int(i) for i in c["runners"]]
            met[np.ix_(ids, ids)] = True
            trk.observe(c["runners"], c["t"], scores=c.get("scores"), order=c.get("order"),
                        winner=c.get("winner"))
        m = np.array([trk.state[str(e)].mean if str(e) in trk.state else 0.0 for e in range(E)])
        v = np.array([trk.state[str(e)].var if str(e) in trk.state else init_var for e in range(E)])
        i, j = np.triu_indices(E, 1)
        keep = met[i, j]
        zs.append(_contrast_z(truth, m, v, pairs=(i[keep], j[keep])))
        zm.append((truth - m) / np.sqrt(v))
    return _aggregate(zs), _aggregate(zm)


@check("calibration.tracker", profiles=("fast",), group="calibration", cost_s=20.0)
def tracker(ctx):
    """AbilityTracker on its own generative model: scores and orders at
    drift 0 (fast), plus drift cells (full). Contrast z^2 and coverage
    against the pre-registered band; marginals reported as MEASURED."""
    out = []
    cells = [("scores", 0.0, 0.5), ("orders", 0.0, 0.5)]
    if ctx.profile in ("full", "exhaustive"):
        cells += [("winners", 0.0, 0.5), ("scores", 0.02, 1.0), ("scores", 0.02, 0.5),
                  ("orders", 0.02, 0.5)]
    R = int(ctx.param("sbc_worlds"))
    for kind, drift, dexp in cells:
        Rk = R if kind in ("scores", "orders") else max(10, R // 2)
        agg, aggm = _run_tracker(ctx, kind, Rk, drift, dexp)
        reg = {"kind": kind, "drift": drift, "drift_exp": dexp}
        out += _band_result(ctx, f"{ctx.name}.{kind}.drift{drift}.exp{dexp}", agg,
                            "calibration.tracker.contrast", reg)
        out.append(Result(f"{ctx.name}.{kind}.drift{drift}.exp{dexp}.marginal", "calibration",
                          "MEASURED", statistic=aggm["z2"], se=aggm["z2_se"], regime=reg,
                          n=aggm["n_z"],
                          detail="marginal z^2: overconfident by the unidentified level (F3)",
                          extras=aggm))
    return out


def _run_history(ctx, kind, R, E=12, T=60, K=5, prior_var=1.0, beta2=1.0):
    from ..history import rate_history
    zs = []
    for r in range(R):
        rng = ctx.rng(f"history:{kind}:{r}")
        contests, truth = _world(rng, E, T, K, prior_var, beta2, 0.0, 0.5, kind)
        races = []
        met = np.zeros((E, E), dtype=bool)
        for c in contests:
            ids = [int(i) for i in c["runners"]]
            met[np.ix_(ids, ids)] = True
            race = {"t": c["t"], "runners": ids}
            if kind == "scores":
                race["scores"] = (-np.asarray(c["scores"])).tolist()   # higher is better here
            else:
                race["order"] = c["order"]
            races.append(race)
        # timescale huge: no drift (both models static)
        _, _, st = rate_history(races, ids=list(range(E)), timescale=1e12, prior_var=prior_var,
                                beta2=beta2, lengths_scale=1.0, return_state=True)
        m, S = st["m"], st["S"]
        i, j = np.triu_indices(E, 1)
        keep = met[i, j]
        zs.append(_contrast_z(truth, m, np.diag(S), S=S, pairs=(i[keep], j[keep])))
    return _aggregate(zs)


@check("calibration.history", profiles=("fast",), group="calibration", cost_s=10.0)
def history(ctx):
    """rate_history on the static conjugate model (scores; fast) and on
    orders (full, small fields per F5). A full-state conjugate filter is
    exactly calibrated here; the sub-block update (F1) is not, so the
    scores cell is MEASURED until F1 is fixed and then gated at the
    exact band."""
    out = []
    R = int(ctx.param("sbc_worlds"))
    agg = _run_history(ctx, "scores", R)
    out += _band_result(ctx, f"{ctx.name}.scores.drift0", agg, "calibration.history.contrast",
                        {"kind": "scores"}, measured=ctx.marks.MARKS.get("calibration.history.contrast", {}).get("measured", True))
    if ctx.profile in ("full", "exhaustive"):
        agg = _run_history(ctx, "orders", max(10, R // 4), E=8, T=30, K=3)
        out += _band_result(ctx, f"{ctx.name}.orders.drift0", agg, "calibration.history.contrast",
                            {"kind": "orders"}, measured=True)
    return out
