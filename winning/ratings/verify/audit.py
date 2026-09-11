"""The property audit (P1-P8), ported from the bandits measurement
programme's tests/audit_ratings_bulletproof.py with fixed seeds.

Motivated there by a bug found by accident: base="laplace" made
posterior variance GROW, because a kinked density leaves quadrature
noise on the lattice gradient and differencing it produced wrong-signed
curvature -- expressed as variance EXPLOSION on one harness and as
COLLAPSE to the clip floor on another (the quiet face: a falsely
confident rating looks healthy and prices badly). So every
base-accepting update is swept against every base:

  P1  outputs are finite
  P2  variance never INCREASES under evidence on log-concave bases
      (Prekopa makes this a theorem, so a violation is always a bug)
  P3  variance does not COLLAPSE to a clip floor -- BOTH known floors
      (1e-6 in update_winner, 1e-4 in the clip paths), and the minimum
      reached is always reported so a new floor cannot hide
  P4  the mean moves toward the truth
  P5  a long soak stays sane
  P6  the posterior is calibrated: z ~ N(0, 1), 95 percent coverage
  P7  independent code paths agree on identical evidence
  P8  a Monte Carlo referee agrees with the analytic winner update

What changed in the port. Seeds are FIXED per driver (marks.SEEDS: 14
for the three correlated / full drivers, 25 elsewhere) and there is no
time box at all: a budget that could cut below the floor made a cell's
verdict depend on machine load, cutting the slowest cells first, which
are the correlated drivers, which are also the noisiest -- under CPU
contention it read a clean tree's update_order_correlated x logistic
(robust sd 0.557 at 7 seeds) and x laplace (0.649) as failures that 25
seeds clear (0.806 / 0.985 and 0.952 / 0.945; the first 7 seeds
reproduce the bad numbers exactly). A gate that fails differently on a
loaded machine is not a gate. UNDERPOWERED stays a verdict distinct
from ok: a thin cell reading as a pass is how that hid.

The world draws its noise from the base under test (the model's own
generative assumption); the bandits world drew GAUSSIAN noise under
every base, and that world is kept as the `audit.legacy` group with the
original seeds 5000+s so the historical digits reproduce, as a labelled
robustness regime. Regimes beyond the bandits baseline (beta2 in
{0.25, 4}, prior variance in {0.25, 9}, one-factor loadings 0.6)
exercise what the original never did.

P6 gates on the pairwise-CONTRAST z (only contrasts are identified --
F3 of the plan -- and 28 pairs per seed against the marginal's 8 arms
give the tail test its power at 14 seeds) and reports the bandits
statistic (truth and posterior means each centred on the field, scaled
by the marginal sd) alongside as MEASURED; the legacy group gates on
the bandits statistic with the bandits power rule so its digits and
verdicts reproduce.
"""

from __future__ import annotations

import numpy as np

from .core import Result, check
from . import marks

M, K = 8, 5
CLIP_FLOORS = (1e-6, 1e-4)

BASES = [("normal", "normal", True), ("gumbel", "gumbel", True),
         ("logistic", "logistic", True), ("laplace", "laplace", True),
         ("student4", None, False), ("failure", None, False)]


def _base(name, obj):
    if obj is not None:
        return obj
    from ...factor.races import failure_base, student_base
    return {"student4": student_base(4.0), "failure": failure_base(0.15)}.get(name, name)


# --------------------------------------------------------------- drivers
# each applies ONE update to the subset and returns (m_new, v_new);
# V is the subset's loadings or None; the diagonal drivers cannot take V

def drive_winner(m, v, S, order, base, beta2, V):
    from ..nway import update_winner
    if V is not None:
        raise NotImplementedError("diagonal path: V is not applicable")
    return update_winner(m[S], v[S], int(order[0]), beta2=beta2, base=base)[:2]


def drive_ranking(m, v, S, order, base, beta2, V):
    from ..nway import update_ranking
    if V is not None:
        raise NotImplementedError("diagonal path: V is not applicable")
    return update_ranking(m[S], v[S], order, beta2=beta2, base=base)[:2]


def drive_ranking_exact(m, v, S, order, base, beta2, V):
    from ..nway import update_ranking_exact
    if V is not None:
        raise NotImplementedError("diagonal path: V is not applicable")
    return update_ranking_exact(m[S], v[S], order, beta2=beta2, base=base)[:2]


def drive_winner_corr(m, v, S, order, base, beta2, V):
    from ..nway import update_winner_correlated
    Vs = np.zeros((len(S), 1)) if V is None else V
    return update_winner_correlated(m[S], v[S], int(order[0]), Vs, beta2=beta2, base=base)[:2]


def drive_order_corr(m, v, S, order, base, beta2, V):
    from ..nway import update_order_correlated
    Vs = np.zeros((len(S), 1)) if V is None else V
    return update_order_correlated(m[S], v[S], order, Vs, beta2=beta2, base=base)[:2]


def drive_winner_full(m, v, S, order, base, beta2, V):
    from ..full import update_winner_full
    mm, SS = update_winner_full(m[S], np.diag(v[S]), int(order[0]), V=V, beta2=beta2,
                                base=base)[:2]
    return mm, np.diag(SS).copy()


def drive_order_full(m, v, S, order, base, beta2, V):
    from ..full import update_order_full
    mm, SS = update_order_full(m[S], np.diag(v[S]), order, V=V, beta2=beta2, base=base)[:2]
    return mm, np.diag(SS).copy()


DRIVERS = [("update_winner", drive_winner), ("update_ranking", drive_ranking),
           ("update_ranking_exact", drive_ranking_exact),
           ("update_winner_correlated", drive_winner_corr),
           ("update_order_correlated", drive_order_corr),
           ("update_winner_full", drive_winner_full),
           ("update_order_full", drive_order_full)]

BASELINE = {"beta2": 1.0, "prior_var": 1.0, "V": 0.0}
REGIMES = [BASELINE, {"beta2": 0.25, "prior_var": 1.0, "V": 0.0},
           {"beta2": 4.0, "prior_var": 1.0, "V": 0.0},
           {"beta2": 1.0, "prior_var": 0.25, "V": 0.0},
           {"beta2": 1.0, "prior_var": 9.0, "V": 0.0},
           {"beta2": 1.0, "prior_var": 1.0, "V": 0.6}]


def _regime_tag(r):
    return f"b{r['beta2']}.pv{r['prior_var']}.V{r['V']}"


def _loadings(regime, rng):
    if regime["V"] == 0.0:
        return None
    return (regime["V"] * np.where(rng.random(M) < 0.5, 1.0, -1.0))[:, None]


def soak(driver, base, rng, n=40, regime=BASELINE, noise="base"):
    """Apply n updates from a fresh world, tracking the worst per-step
    variance increase, floor hits (both floors) and the minimum variance."""
    from ..simulate import independent_world
    V = _loadings(regime, rng)
    a, events = independent_world(rng, M, K, n, prior_var=regime["prior_var"],
                                  beta2=regime["beta2"], V=V, base=base, noise=noise)
    m, v = np.zeros(M), np.full(M, float(regime["prior_var"]))
    worst_increase, floored = 0.0, 0
    for S, order in events:
        mm, vv = driver(m, v, S, order, base, regime["beta2"], None if V is None else V[S])
        if not (np.all(np.isfinite(mm)) and np.all(np.isfinite(vv))):
            return dict(ok=False, why="non-finite", a=a, m=m, v=v)
        worst_increase = max(worst_increase, float((vv - v[S]).max()))
        floored += int(sum((vv <= f * 1.001).sum() for f in CLIP_FLOORS))
        m[S], v[S] = mm, vv
    return dict(ok=True, a=a, m=m, v=v, worst_increase=worst_increase,
                floored=floored, min_var=float(v.min()))


def calibration(driver, base, rngs, n=60, regime=BASELINE, noise="base"):
    """P6 over a FIXED list of generators (one soak each): the bandits
    z (field-centred truth minus field-centred mean, over the marginal
    sd) and the pairwise contrast z. Returns None if any soak fails."""
    zs, zc = [], []
    for rng in rngs:
        r = soak(driver, base, rng, n=n, regime=regime, noise=noise)
        if not r["ok"]:
            return None
        mu = r["m"] - r["m"].mean(); at = r["a"] - r["a"].mean()
        zs.append((at - mu) / np.sqrt(r["v"]))
        i, j = np.triu_indices(M, 1)
        zc.append(((r["a"][i] - r["a"][j]) - (r["m"][i] - r["m"][j]))
                  / np.sqrt(r["v"][i] + r["v"][j]))

    def stats(z):
        z = np.concatenate(z)
        robust = float((np.percentile(z, 75) - np.percentile(z, 25)) / 1.349)
        return dict(sd=float(z.std()), coverage=float(np.mean(np.abs(z) < 1.96)),
                    robust=robust, tail=float(np.mean(np.abs(z) > 10)), n_z=int(z.size))
    return stats(zs), stats(zc)


def _p6_verdict(ctx, dname, bname, s, thin=None):
    th = ctx.mark("audit.P6")["thresholds"]
    bad = (abs(s["robust"] - 1.0) > th["robust"] or s["tail"] >= th["tail"]
           or abs(s["coverage"] - 0.95) > th["coverage"])
    if thin is None:
        # power: three standard errors must fit inside the threshold, and
        # a 1 percent tail needs 300 z-values to be seen at all
        se_sd = 1.0 / np.sqrt(2.0 * s["n_z"]); se_cov = np.sqrt(0.0475 / s["n_z"])
        thin = (3 * se_sd > th["robust"]) or (3 * se_cov > th["coverage"]) or (s["n_z"] < 300)
    env = marks.EXPECTED_APPROX.get(f"audit.P6.{dname}")
    if bad and env is not None:
        e = env["envelope"]
        inside = (e["robust"][0] <= s["robust"] <= e["robust"][1]
                  and e["coverage"][0] <= s["coverage"] <= e["coverage"][1])
        if inside:
            return "EXPECTED_APPROX", env["why"]
        return "FAIL", "outside the documented envelope"
    if bad:
        return "FAIL", f"robust sd {s['robust']:.2f} cover {s['coverage']:.2f} tail {s['tail']:.3f}"
    if thin:
        return "UNDERPOWERED", f"only {s['n_z']} z-values"
    return "ok", ""


# --------------------------------------------------------------- checks

def _p1_p5(ctx, regimes, n, noise="base", legacy_seed=None, name=None):
    out = []
    p2_tol = float(ctx.mark("audit.P2")["tolerance"])
    p4_tol = float(ctx.mark("audit.P4")["tolerance"])
    name = name or ctx.name
    for regime in regimes:
        rt = _regime_tag(regime)
        for dname, driver in DRIVERS:
            for bname, bobj, concave in BASES:
                base = _base(bname, bobj)
                cell = f"{name}.{dname}.{bname}.{rt}"
                reg = {"driver": dname, "base": bname, **regime}
                rng = (np.random.default_rng(legacy_seed) if legacy_seed is not None
                       else ctx.rng(f"{dname}:{bname}:{rt}"))
                try:
                    r = soak(driver, base, rng, n=n, regime=regime, noise=noise)
                except NotImplementedError as e:
                    out.append(Result(cell, "audit", "SKIP", regime=reg,
                                      detail=f"not applicable: {str(e)[:60]}"))
                    continue
                if not r["ok"]:
                    out.append(Result(cell, "audit", "FAIL", regime=reg, detail="P1 " + r["why"]))
                    continue
                corr = float(np.corrcoef(r["m"], r["a"])[0, 1])
                bad = []
                if concave and r["worst_increase"] > p2_tol:
                    bad.append(f"P2 var grew by {r['worst_increase']:.2e}")
                if r["floored"] > 0:
                    bad.append(f"P3 {r['floored']} floored")
                if corr < p4_tol:
                    bad.append(f"P4 corr {corr:.2f}")
                if r["v"].mean() > regime["prior_var"]:
                    bad.append(f"P5 mean var {r['v'].mean():.2f} >= prior")
                out.append(Result(cell, "audit", "ok" if not bad else "FAIL",
                                  statistic=corr, tolerance=p4_tol, regime=reg, n=n,
                                  detail="; ".join(bad),
                                  extras={"worst_increase": r["worst_increase"],
                                          "floored": r["floored"], "min_var": r["min_var"],
                                          "mean_var": float(r["v"].mean())}))
    return out


@check("audit.P1_P5", profiles=("fast",), group="audit", cost_s=25.0)
def p1_p5(ctx):
    """P1-P5 at the baseline regime, every driver x base, a short soak
    in fast (10 updates) and the bandits length (40) in full."""
    n = 10 if ctx.profile == "fast" else 40
    return _p1_p5(ctx, [BASELINE], n)


@check("audit.regime.P1_P5", profiles=("full",), group="audit", cost_s=110.0)
def regime_p1_p5(ctx):
    """P1-P5 one-factor-at-a-time off the baseline: beta2, prior
    variance, one-factor loadings."""
    return _p1_p5(ctx, REGIMES[1:], 40)


def _p6_cells(ctx, dname, driver, regimes, noise="base", legacy=False, name=None):
    out = []
    name = name or ctx.name
    n_seeds = marks.SEEDS.get(dname, marks.DEFAULT_SEEDS)
    for regime in regimes:
        rt = _regime_tag(regime)
        for bname, bobj, _c in BASES:
            cell = f"{name}.{dname}.{bname}.{rt}"
            reg = {"driver": dname, "base": bname, **regime}
            if (dname, bname) in REDUNDANT and regime["V"] == 0.0:
                out.append(Result(cell, "audit", "SKIP", regime=reg,
                                  detail="identical to the diagonal path at V=0 "
                                         "(identity.reductions pins it); 7x the cost"))
                continue
            base = _base(bname, bobj)
            if legacy:
                rngs = [np.random.default_rng(5000 + s) for s in range(n_seeds)]
                seeds = [5000 + s for s in range(n_seeds)]
            else:
                seeds = [ctx.seed(f"{dname}:{bname}:{rt}:{s}") for s in range(n_seeds)]
                rngs = [np.random.default_rng(s) for s in seeds]
            try:
                c = calibration(driver, base, rngs, regime=regime, noise=noise)
            except NotImplementedError as e:
                out.append(Result(cell, "audit", "SKIP", regime=reg,
                                  detail=f"not applicable: {str(e)[:60]}"))
                continue
            if c is None:
                out.append(Result(cell, "audit", "FAIL", regime=reg, detail="P6 soak non-finite"))
                continue
            s, sc = c
            th = ctx.mark("audit.P6")["thresholds"]
            if legacy:
                # the bandits statistic and the bandits power rule (fewer
                # than 6 seeds is thin), so the historical digits and
                # verdicts reproduce
                verdict, detail = _p6_verdict(ctx, dname, bname, s, thin=n_seeds < 6)
                primary, other, olabel = s, sc, "contrast"
            else:
                # pairwise contrasts are the identified quantities (F3) and
                # carry 28 z-values per seed against the marginal's 8
                verdict, detail = _p6_verdict(ctx, dname, bname, sc)
                primary, other, olabel = sc, s, "marginal"
            out.append(Result(cell, "audit", verdict, statistic=primary["robust"],
                              tolerance=th["robust"], regime=reg, seeds=seeds,
                              n=primary["n_z"], detail=detail,
                              extras={"z_sd": primary["sd"], "coverage": primary["coverage"],
                                      "tail": primary["tail"], "seeds_used": n_seeds}))
            out.append(Result(cell + "." + olabel, "audit", "MEASURED",
                              statistic=other["robust"], regime=reg, seeds=seeds, n=other["n_z"],
                              detail=f"{olabel} z: robust sd, coverage in extras",
                              extras={"z_sd": other["sd"], "coverage": other["coverage"],
                                      "tail": other["tail"]}))
    return out


REDUNDANT = {("update_winner_correlated", "student4"), ("update_order_correlated", "student4")}


def _register_p6():
    for dname, driver in DRIVERS:
        def _mk(dname=dname, driver=driver):
            @check(f"audit.P6.{dname}", profiles=("full",), group="audit", cost_s=300.0)
            def p6(ctx):
                return _p6_cells(ctx, dname, driver, [BASELINE])

            @check(f"audit.legacy.{dname}", profiles=("full",), group="audit", cost_s=300.0)
            def legacy(ctx):
                """The bandits world: Gaussian noise under every base, seeds
                5000 + s -- the port's acceptance test and a labelled
                robustness regime (misspecified noise)."""
                return _p6_cells(ctx, dname, driver, [BASELINE], noise="gaussian", legacy=True)

            @check(f"audit.regime.P6.{dname}", profiles=("exhaustive",), group="audit",
                   cost_s=1500.0)
            def regime_p6(ctx):
                return _p6_cells(ctx, dname, driver, REGIMES[1:])
        _mk()


_register_p6()


@check("audit.P7", profiles=("fast",), group="audit", cost_s=3.0)
def p7(ctx):
    """Independent paths agree on identical evidence: the winner paths
    and the exact order paths, means (the bandits 0.05 outer bound) and
    variances (measured; the tighter per-path identities live in
    identity.reductions)."""
    out = []
    rng = ctx.rng("world")
    a = rng.normal(0, 1, M); a -= a.mean()
    S = np.arange(K); order = np.array([2, 0, 4, 1, 3])
    m0, v0 = np.zeros(M), np.ones(M)
    tol = float(ctx.mark("audit.P7")["tolerance"])
    for bname, bobj, _c in BASES:
        base = _base(bname, bobj)
        vals = {}
        for dname, driver in DRIVERS:
            try:
                vals[dname] = driver(m0, v0, S, order, base, 1.0, None)
            except NotImplementedError:
                pass
        groups = [("winner", [k for k in vals if "winner" in k]),
                  ("order_exact", [k for k in vals if "winner" not in k and k != "update_ranking"])]
        for label, keys in groups:
            if len(keys) < 2:
                continue
            ms = np.array([vals[k][0] for k in keys]); vs = np.array([vals[k][1] for k in keys])
            spread_m = float(np.abs(ms - ms.mean(0)).max())
            spread_v = float(np.abs(vs - vs.mean(0)).max())
            out.append(Result(f"{ctx.name}.{label}.{bname}.means", "audit",
                              "ok" if spread_m <= tol else "FAIL", statistic=spread_m,
                              tolerance=tol, regime={"base": bname}, n=len(keys),
                              detail="" if spread_m <= tol else "paths disagree"))
            out.append(Result(f"{ctx.name}.{label}.{bname}.variances", "audit", "MEASURED",
                              statistic=spread_v, regime={"base": bname}, n=len(keys),
                              detail="variance spread across paths (curvature methods differ)"))
    return out


@check("audit.P8", profiles=("fast",), group="audit", cost_s=4.0)
def p8(ctx):
    """Monte Carlo referee on the winner update at the bandits
    three-arm configuration, every base (samplers now exist for
    student-t and the failure lump): rejection-sampled posterior mean
    and variance against the analytic update."""
    from ..nway import update_winner
    from ..simulate import sample_noise
    out = []
    N = int(ctx.param("p8_draws"))
    th = ctx.mark("audit.P8")
    m0 = np.array([0.4, 0.0, -0.3]); v0 = np.array([0.6, 0.4, 0.5])
    for bname, bobj, _c in BASES:
        base = _base(bname, bobj)
        mm, vv = update_winner(m0, v0, 0, base=base)[:2]
        rng = ctx.rng(bname)
        A = m0[None, :] + rng.normal(0, np.sqrt(v0), (N, 3))
        noise = sample_noise(rng, base, (N, 3))
        win = (A + noise).argmax(1) == 0
        mc_m, mc_v = A[win].mean(0), A[win].var(0)
        dm = float(np.abs(mm - mc_m).max()); dv = float(np.abs(vv - mc_v).max())
        se = float(np.sqrt(mc_v.max() / max(win.sum(), 1)))
        bad = dm > th["dm"] + 4 * se or dv > th["dv"]
        out.append(Result(f"{ctx.name}.{bname}", "audit", "ok" if not bad else "FAIL",
                          statistic=max(dm, dv), se=se, tolerance=th["dm"],
                          regime={"base": bname}, n=int(win.sum()),
                          detail="" if not bad else f"dm {dm:.4f} dv {dv:.4f}",
                          extras={"dm": dm, "dv": dv, "p_hat": float(win.mean())}))
    return out
