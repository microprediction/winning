"""The MAP factor fitter (I10 of the plan): derivative identities, the
Laplace posterior's calibration on its own prior, recovery scaling with
data, and held-out scoring against tuned rivals with a leakage guard.
"""

from __future__ import annotations

import numpy as np

from .core import Result, check


@check("factor.derivatives", profiles=("smoke",), group="factor", cost_s=2.0)
def derivatives(ctx):
    """Analytic gradient and Hessian of the penalised objective against
    finite differences, every event shape, normal and logistic bases;
    the Hessian symmetric and positive definite."""
    from ..factor_ratings import (_Stacked, _design_triplets, _loglik_terms,
                                  _as_ridge, _penalised_hessian)
    from ..simulate import mixed_events
    out = []
    n_feat = 7
    for bname in ("normal", "logistic"):
        rng = ctx.rng(bname)
        evs = mixed_events(rng, n_feat)
        st = _Stacked(_design_triplets(evs, n_feat), n_feat)
        theta = rng.normal(size=n_feat) * 0.5
        lam = _as_ridge(1.0, n_feat); w = rng.random(st.n_events) + 0.5

        def f(th):
            lp, dmu = _loglik_terms(th, st, bname)
            return (0.5 * float(th @ (lam * th)) - float(w @ lp),
                    lam * th - np.asarray(st.ZT @ (dmu * w[st.row_event])).ravel())
        _, g = f(theta)
        gfd = np.empty(n_feat); Hfd = np.empty((n_feat, n_feat))
        for j in range(n_feat):
            e = np.zeros(n_feat); e[j] = 1e-5
            gfd[j] = (f(theta + e)[0] - f(theta - e)[0]) / 2e-5
            e[j] = 1e-4
            Hfd[:, j] = (f(theta + e)[1] - f(theta - e)[1]) / 2e-4
        H = _penalised_hessian(theta, st, lam, w, bname).toarray()
        reg = {"base": bname}
        dg = float(np.abs(g - gfd).max() / max(1.0, np.abs(gfd).max()))
        v, tol = ctx.verdict(dg, key="factor.gradient_fd")
        out.append(Result(f"{ctx.name}.gradient.{bname}", "factor", v, statistic=dg,
                          tolerance=tol, regime=reg))
        dh = float(np.abs(H - Hfd).max() / np.abs(Hfd).max())
        v, tol = ctx.verdict(dh, key="factor.hessian_fd")
        out.append(Result(f"{ctx.name}.hessian.{bname}", "factor", v, statistic=dh,
                          tolerance=tol, regime=reg))
        spd = float(max(np.abs(H - H.T).max(), max(0.0, -np.linalg.eigvalsh(H).min())))
        v, tol = ctx.verdict(spd, key="factor.hessian_spd")
        out.append(Result(f"{ctx.name}.hessian_spd.{bname}", "factor", v, statistic=spd,
                          tolerance=tol, regime=reg))
    return out


def _prior_world(rng, n_ent, K, n_events, lam_level, lam_offset):
    """A world drawn from the ridge prior itself: coefficients with
    precision lam per column, binary conditions, unit noise."""
    B = np.column_stack([rng.normal(0, 1 / np.sqrt(lam_level), n_ent),
                         rng.normal(0, 1 / np.sqrt(lam_offset), n_ent)])
    evs = []
    for _ in range(n_events):
        sub = rng.choice(n_ent, K, replace=False)
        x = np.array([1.0, float(rng.random() < 0.5)])
        perf = B[sub] @ x + rng.normal(size=K)
        evs.append((sub, np.argsort(-perf), x))
    return B, evs


@check("factor.se_calibration", profiles=("full",), group="factor", cost_s=120.0)
def se_calibration(ctx):
    """Simulation-based calibration of the Laplace posterior: worlds
    drawn from the ridge prior the fitter assumes, fitted with the same
    ridge; z = (truth - estimate) / se over entities and columns, mean
    z^2 and coverage across R worlds with a cluster standard error.
    MEASURED until the baseline sets the band."""
    from ..factor_ratings import fit_factor_ratings
    R = int(ctx.param("factor_worlds"))
    lam = np.array([1.0, 4.0])
    zs = []
    for r in range(R):
        rng = ctx.rng(f"world:{r}")
        B, evs = _prior_world(rng, 20, 5, 300, lam[0], lam[1])
        Bh, se = fit_factor_ratings(evs, 20, 2, ridge=lam, return_se=True)
        zs.append(((B - Bh) / se).ravel())
    z2 = np.array([float(np.mean(z * z)) for z in zs])
    cov = np.array([float(np.mean(np.abs(z) < 1.96)) for z in zs])
    agg = {"z2": float(z2.mean()), "z2_se": float(z2.std(ddof=1) / np.sqrt(R)),
           "coverage": float(cov.mean()), "R": R}
    m = ctx.mark("factor.se_calibration")
    if m is None:
        return [Result(f"{ctx.name}.z2", "factor", "MEASURED", statistic=agg["z2"],
                       se=agg["z2_se"], n=R * 40, detail="Laplace SBC on the ridge prior; no mark yet",
                       extras=agg)]
    lo, hi = m["z2_band"]
    inside = lo <= agg["z2"] <= hi
    thin = 3 * agg["z2_se"] > (hi - lo) / 2
    verdict = "FAIL" if not inside else ("UNDERPOWERED" if thin else "ok")
    return [Result(f"{ctx.name}.z2", "factor", verdict, statistic=agg["z2"], se=agg["z2_se"],
                   tolerance=hi, n=R * 40, extras=agg,
                   detail="" if verdict == "ok" else f"z^2 {agg['z2']:.3f}, band [{lo}, {hi}]")]


@check("factor.recovery_slope", profiles=("full",), group="factor", cost_s=90.0)
def recovery_slope(ctx):
    """RMSE of the recovered contrasts against the truth falls as one
    over root n: the slope of log RMSE on log n over n in {200, 800,
    3200}, R replicates; and RMSE at the largest n against the mean
    Laplace SE. MEASURED until the baseline sets the band (expected
    slope in [-0.65, -0.35])."""
    from ..factor_ratings import fit_factor_ratings
    from ..simulate import factor_world
    ns = (200, 800, 3200)
    R = 3
    rmse = np.zeros((R, len(ns))); ratio = np.zeros(R)
    for r in range(R):
        for k, n in enumerate(ns):
            rng = ctx.rng(f"{r}:{n}")
            B, evs = factor_world(rng, 20, 5, n, off_sd=0.5, n_cov=2)
            Bh, se = fit_factor_ratings(evs, 20, 2, ridge=np.array([1.0, 3.0]), return_se=True)
            d = (Bh - B)[:, 0]; d = d - d.mean()              # contrasts of levels
            rmse[r, k] = float(np.sqrt(np.mean(d * d)))
            if n == ns[-1]:
                ratio[r] = rmse[r, k] / float(se[:, 0].mean())
    lr = np.log(rmse).mean(axis=0)
    slope = float(np.polyfit(np.log(ns), lr, 1)[0])
    extras = {"rmse": rmse.mean(axis=0).tolist(), "n": list(ns),
              "rmse_over_se_at_max_n": float(ratio.mean())}
    m = ctx.mark("factor.recovery_slope")
    if m is None:
        return [Result(f"{ctx.name}.slope", "factor", "MEASURED", statistic=slope, n=R,
                       detail="log RMSE vs log n slope; no mark yet", extras=extras)]
    lo, hi = m["band"]
    v = "ok" if lo <= slope <= hi else "FAIL"
    return [Result(f"{ctx.name}.slope", "factor", v, statistic=slope, tolerance=hi, n=R,
                   extras=extras, detail="" if v == "ok" else f"slope {slope:.2f} outside [{lo}, {hi}]")]


@check("factor.heldout", profiles=("full",), group="factor", cost_s=60.0)
def heldout(ctx):
    """Held-out proper scoring on a synthetic world: events strictly
    after the training window, normal-base MAP against a Gumbel-base
    MAP (Plackett-Luce) and uniform, each with its ridge tuned on a
    shared validation window; plus the leakage guard's positive
    control (validation leaked into training must improve the held-out
    score by more than three paired standard errors, otherwise the
    guard is untested). MEASURED margins; the falsification (normal
    trailing Plackett-Luce on normal-generated data by more than two
    paired SE) is a FAIL."""
    from ..factor_ratings import fit_factor_ratings, factor_loglik
    rng = ctx.rng("world")
    n_ent, K = 20, 5
    B = np.column_stack([rng.normal(size=n_ent), rng.normal(size=n_ent) * 0.5])

    def draw(n):
        evs = []
        for _ in range(n):
            sub = rng.choice(n_ent, K, replace=False)
            x = np.array([1.0, float(rng.random() < 0.5)])
            perf = B[sub] @ x + rng.normal(size=K)
            evs.append((sub, np.argsort(-perf), x))
        return evs
    train, val, test = draw(400), draw(200), draw(400)
    out = []
    scores = {}
    for bname in ("normal", "gumbel"):
        best = None
        for lam in (0.3, 1.0, 3.0, 10.0):
            Bh = fit_factor_ratings(train, n_ent, 2, ridge=np.array([1.0, lam]), base=bname)
            vl = -factor_loglik(Bh, val, base=bname).mean()
            if best is None or vl < best[0]:
                best = (vl, lam, Bh)
        scores[bname] = -factor_loglik(best[2], test, base=bname)
        out.append(Result(f"{ctx.name}.ridge.{bname}", "factor", "MEASURED", statistic=best[1],
                          detail="tuned offset ridge (grid edge if 0.3 or 10)"))
    uniform = np.full(len(test), np.log(K))
    d = scores["normal"] - scores["gumbel"]
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    stat = float(d.mean())
    v = "FAIL" if stat > 2 * se else "MEASURED"
    out.append(Result(f"{ctx.name}.normal_vs_plackett_luce", "factor", v, statistic=stat, se=se,
                      n=len(d), detail="paired held-out loss difference (normal - PL), falsification "
                                       "if normal trails by > 2 SE on normal-generated data"))
    du = scores["normal"] - uniform
    out.append(Result(f"{ctx.name}.normal_vs_uniform", "factor", "MEASURED", statistic=float(du.mean()),
                      se=float(du.std(ddof=1) / np.sqrt(len(du))), n=len(du),
                      detail="paired held-out loss difference (normal - uniform)"))
    # leakage guard: training on train + test must improve the test score clearly
    Bleak = fit_factor_ratings(train + test, n_ent, 2, ridge=np.array([1.0, 3.0]))
    dl = scores["normal"] - (-factor_loglik(Bleak, test))
    sel = float(dl.std(ddof=1) / np.sqrt(len(dl)))
    v = "ok" if dl.mean() > 3 * sel else "FAIL"
    out.append(Result(f"{ctx.name}.leakage_guard_positive_control", "factor", v,
                      statistic=float(dl.mean()), se=sel, tolerance=3 * sel, n=len(dl),
                      detail="" if v == "ok" else "leaking the test set did not improve the score: guard untested"))
    return out
