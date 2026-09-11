"""Deterministic identity and metamorphic checks (I4, I5, I7, I8 of the
plan): reductions between code paths that must agree, closed forms the
lattice must reproduce, coarsening (a partial order is the mixture of
the full orders it censors), invariances (gauge, permutation, scale,
team assignment), and predict-versus-evidence (fit and price under one
model). Exact checks fail on a bug; the few that measure an
approximation say so in their mark's basis.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.special import log_ndtr, logsumexp

from .core import Result, check
from .moments import _base_obj, _order_evidence


def _bases(which):
    from ...factor.races import (exponential_power_base, failure_base,
                                 skew_logistic_base, skew_normal_base,
                                 student_base)
    named = [("normal", "normal"), ("gumbel", "gumbel"),
             ("logistic", "logistic"), ("laplace", "laplace")]
    if which == "named":
        return named
    return named + [("student4", student_base(4.0)),
                    ("failure0.15", failure_base(0.15)),
                    ("failure_std", failure_base(0.15, standardize=True)),
                    ("expo1.5", exponential_power_base(1.5)),
                    ("expo6", exponential_power_base(6.0)),
                    ("skewlog0.5", skew_logistic_base(0.5)),
                    ("skewnorm2", skew_normal_base(2.0))]


def _res(ctx, name, stat, key, regime=None, detail_fail="", **kw):
    verdict, tol = ctx.verdict(stat, key=key)
    return Result(name, "identity", verdict, statistic=float(stat), tolerance=tol,
                  regime=regime or {}, detail=detail_fail if verdict == "FAIL" else "", **kw)


def _rel(a, b, scale):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale)


@check("simulate.sampler_matches_density", profiles=("smoke",),
       group="simulate", cost_s=1.0)
def sampler_matches_density(ctx):
    """Every base's sampler against the base's own survival function:
    the Kolmogorov distance of n min-wins draws (winning.ratings.simulate
    .check_sampler). Exact samplers sit at the DKW floor; a sign or
    scale slip is two orders of magnitude above the mark."""
    from ..simulate import check_sampler
    n = int(ctx.param("ks_n"))
    m = ctx.mark()
    tol = m["tolerance_by_n"][n]
    out = []
    for name, base in _bases(ctx.param("ks_bases")):
        d = check_sampler(base, ctx.rng(name), n=n)
        out.append(Result(f"{ctx.name}.{name}", "simulate",
                          "ok" if d <= tol else "FAIL", statistic=d,
                          tolerance=tol, n=n, regime={"base": name},
                          detail="" if d <= tol else "sampler disagrees with its density"))
    return out


# ---------------------------------------------------------------------------
# I8: reductions between paths that must agree
# ---------------------------------------------------------------------------

@check("identity.reductions", profiles=("smoke",), group="identity", cost_s=3.0)
def reductions(ctx):
    """V = 0 correlated updates equal the independent members (means,
    variances and evidence, the order evidence against the exact
    ordered-statistics pass); a diagonal belief through the full-
    covariance path equals the diagonal path; the conjugate market
    node's full and diagonal forms agree exactly."""
    from ..nway import (update_winner, update_ranking_exact,
                        update_winner_correlated, update_order_correlated)
    from ..full import update_winner_full, update_order_full, update_market_full
    from ..market import update_market
    bases = ["normal", "laplace"] if ctx.profile == "smoke" else \
        ["normal", "gumbel", "logistic", "laplace", "student4", "failure"]
    out = []
    K = 5
    for bname in bases:
        base = _base_obj(bname)
        rng = ctx.rng(bname)
        m = rng.normal(0, 1, K); v = rng.uniform(0.5, 1.5, K)
        order = list(rng.permutation(K)); w = int(order[0])
        regime = {"base": bname}
        V0 = np.zeros((K, 1))
        # winner: correlated at V=0 vs independent
        cls = "normal" if bname == "normal" else "curve"
        m1, v1, p1 = update_winner(m, v, w, base=base)
        m2, v2, lz2 = update_winner_correlated(m, v, w, V0, base=base)
        out.append(_res(ctx, f"{ctx.name}.winner_v0_m_evidence.{bname}",
                        max(_rel(m1, m2, 1.0), abs(np.log(p1) - lz2)),
                        "identity.reductions.exact", regime))
        # variances: analytic curvature (curve path) or eps 1e-4 differences
        # (normal) against the mixture's eps 1e-3 differences
        out.append(_res(ctx, f"{ctx.name}.winner_v0_v.{bname}", _rel(v1, v2, 1.0),
                        f"identity.reductions.v0_v.{cls}", regime))
        # order: correlated at V=0 vs exact ranking; evidence vs the pass
        m3, v3 = update_ranking_exact(m, v, order, base=base)
        m4, v4, lz4 = update_order_correlated(m, v, order, V0, base=base)
        lz3 = _order_evidence(m, v, order, 1.0, base)
        out.append(_res(ctx, f"{ctx.name}.order_v0.{bname}",
                        max(_rel(m3, m4, 1.0), _rel(v3, v4, 1.0)),
                        "identity.reductions.exact", regime))
        out.append(_res(ctx, f"{ctx.name}.order_v0_evidence.{bname}", abs(lz3 - lz4),
                        "identity.reductions.exact", regime))
        # full-covariance with diagonal S vs the diagonal paths (different
        # curvature steps: a tolerance identity on v, exact on m)
        try:
            m5, S5, lz5 = update_winner_full(m, np.diag(v), w, base=base)
            out.append(_res(ctx, f"{ctx.name}.winner_full_diag_m.{bname}", _rel(m1, m5, 1.0),
                            f"identity.reductions.full_m.{cls}", regime))
            out.append(_res(ctx, f"{ctx.name}.winner_full_diag_v.{bname}",
                            _rel(v1, np.diag(S5), 1.0), f"identity.reductions.full_v.{cls}", regime))
            out.append(_res(ctx, f"{ctx.name}.winner_full_diag_evidence.{bname}",
                            abs(np.log(p1) - lz5), f"identity.reductions.full_evidence.{cls}", regime))
        except NotImplementedError as e:
            out.append(Result(f"{ctx.name}.winner_full_diag.{bname}", "identity", "SKIP",
                              regime=regime, detail=f"not supported: {str(e)[:60]}"))
        m6, S6, lz6 = update_order_full(m, np.diag(v), order, base=base)
        out.append(_res(ctx, f"{ctx.name}.order_full_diag_m.{bname}", _rel(m3, m6, 1.0),
                        f"identity.reductions.full_m.{cls}", regime))
        out.append(_res(ctx, f"{ctx.name}.order_full_diag_v.{bname}", _rel(v3, np.diag(S6), 1.0),
                        f"identity.reductions.full_v.{cls}", regime))
        out.append(_res(ctx, f"{ctx.name}.order_full_diag_evidence.{bname}", abs(lz3 - lz6),
                        f"identity.reductions.full_evidence.{cls}", regime))
    # market: full vs diagonal, exact
    rng = ctx.rng("market")
    m = rng.normal(0, 1, K); v = rng.uniform(0.5, 1.5, K)
    p = rng.dirichlet(np.ones(K) * 3)
    a1, b1, z1 = update_market(m, v, p, tau2=0.3)
    a2, S2, z2 = update_market_full(m, np.diag(v), p, tau2=0.3)
    out.append(_res(ctx, f"{ctx.name}.market_full_diag",
                    max(_rel(a1, a2, 1.0), _rel(b1, np.diag(S2), 1.0), abs(z1 - z2)),
                    "identity.reductions.exact"))
    # correlated with V through both paths: quadrature rules differ
    # (Gauss-Hermite 7 vs 9 nodes), so this is measured, not exact
    V = (0.6 * (-1.0) ** np.arange(K))[:, None]
    m7, v7, lz7 = update_winner_correlated(m, v, 2, V)
    m8, S8, lz8 = update_winner_full(m, np.diag(v), 2, V=V)
    out.append(_res(ctx, f"{ctx.name}.winner_full_vs_correlated_V",
                    max(_rel(m7, m8, 1.0), _rel(v7, np.diag(S8), 1.0), abs(lz7 - lz8)),
                    "identity.reductions.full_vs_correlated_V"))
    m9, v9, lz9 = update_order_correlated(m, v, [2, 0, 4, 1, 3], V)
    m10, S10, lz10 = update_order_full(m, np.diag(v), [2, 0, 4, 1, 3], V=V)
    out.append(_res(ctx, f"{ctx.name}.order_full_vs_correlated_V",
                    max(_rel(m9, m10, 1.0), _rel(v9, np.diag(S10), 1.0), abs(lz9 - lz10)),
                    "identity.reductions.full_vs_correlated_V"))
    return out


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------

def two_player_moments(m, v, beta2):
    """Exact posterior of (s1, s2) given x1 > x2 with x_i = s_i + e_i,
    s_i ~ N(m_i, v_i), e_i ~ N(0, beta2_i): the truncation moments of the
    Gaussian difference. Returns (m_post, v_post, logP)."""
    m = np.asarray(m, float); v = np.asarray(v, float)
    b = np.broadcast_to(np.asarray(beta2, float), (2,))
    c = np.sqrt(v[0] + v[1] + b[0] + b[1])
    t = (m[0] - m[1]) / c
    lam = np.exp(-0.5 * t * t - 0.5 * np.log(2 * np.pi) - log_ndtr(t))
    kappa = lam * (lam + t)
    sign = np.array([1.0, -1.0])
    mp = m + sign * v / c * lam
    vp = v - v * v / (c * c) * kappa
    return mp, vp, float(log_ndtr(t))


@check("identity.closed_form_k2", profiles=("smoke",), group="identity", cost_s=2.0)
def closed_form_k2(ctx):
    """Two-player contests under the normal base against the exact
    truncation moments, on every path (winner, order, correlated at
    V = 0, full covariance with a diagonal belief), including a diffuse-
    prior ladder for the full path (the belief-split regression guard).
    Means are scaled by the prior sd, variances relative."""
    from ..nway import (update_winner, update_ranking_exact,
                        update_winner_correlated, update_order_correlated)
    from ..full import update_winner_full, update_order_full
    out = []
    for prior_sd in (1.0, 3.0, 10.0, 30.0):
        for beta2 in (0.25, 1.0, 4.0):
            rng = ctx.rng(f"{prior_sd}:{beta2}")
            m = rng.normal(0, prior_sd, 2)
            v = prior_sd ** 2 * rng.uniform(0.5, 1.5, 2)
            mp, vp, lp = two_player_moments(m, v, beta2)
            regime = {"prior_sd": prior_sd, "beta2": beta2}
            V0 = np.zeros((2, 1))
            paths = {
                "update_winner": (lambda: update_winner(m, v, 0, beta2=beta2)),
                "update_ranking_exact": (lambda: update_ranking_exact(m, v, [0, 1], beta2=beta2) + (None,)),
                "update_winner_correlated": (lambda: update_winner_correlated(m, v, 0, V0, beta2=beta2)),
                "update_order_correlated": (lambda: update_order_correlated(m, v, [0, 1], V0, beta2=beta2)),
                "update_winner_full": (lambda: update_winner_full(m, np.diag(v), 0, beta2=beta2)),
                "update_order_full": (lambda: update_order_full(m, np.diag(v), [0, 1], beta2=beta2)),
            }
            for pname, fn in paths.items():
                r = fn()
                mm = np.asarray(r[0]); vv = np.asarray(r[1])
                if vv.ndim == 2:
                    vv = np.diag(vv)
                ev = r[2]
                if pname == "update_winner" and ev is not None:
                    ev = float(np.log(ev))
                cls = "fd_rel" if pname.endswith("_full") else (
                    "analytic" if pname == "update_winner" else "fd")
                out.append(_res(ctx, f"{ctx.name}.{pname}.m.sd{prior_sd}.b{beta2}",
                                _rel(mm, mp, prior_sd), "identity.closed_form_k2.m", regime))
                out.append(_res(ctx, f"{ctx.name}.{pname}.v.sd{prior_sd}.b{beta2}",
                                float(np.abs(vv - vp).max() / vp.max()),
                                f"identity.closed_form_k2.v.{cls}", regime))
                if ev is not None:
                    out.append(_res(ctx, f"{ctx.name}.{pname}.logp.sd{prior_sd}.b{beta2}",
                                    abs(ev - lp), "identity.closed_form_k2.logp", regime))
    return out


# ---------------------------------------------------------------------------
# I5: coarsening -- a partial order is the mixture of the full orders it censors
# ---------------------------------------------------------------------------

@check("identity.coarsening", profiles=("fast",), group="identity", cost_s=4.0)
def coarsening(ctx):
    """Partial-order likelihood equals the log-sum over every insertion
    of the omitted entrant; the partial-order update equals the mixture
    of the full-order updates weighted by those insertion probabilities
    (first two moments); a winner-only update equals the mixture over
    every order with that winner; the omitted entrant is untouched
    under a diagonal prior and the ranked entrants match the sub-field
    computation."""
    from ..nway import update_winner, update_ranking_exact
    out = []
    K = 5
    for bname in (["normal", "logistic"] if ctx.profile != "smoke" else ["normal"]):
        base = _base_obj(bname)
        rng = ctx.rng(bname)
        m = rng.normal(0, 1, K); v = rng.uniform(0.5, 1.5, K)
        regime = {"base": bname}
        partial = [3, 0, 4, 1]                 # entrant 2 omitted
        full_orders = [partial[:i] + [2] + partial[i:] for i in range(K)]
        lps = np.array([_order_evidence(m, v, o, 1.0, base) for o in full_orders])
        lp_partial = _order_evidence(m, v, partial, 1.0, base)
        out.append(_res(ctx, f"{ctx.name}.partial_loglik.{bname}",
                        abs(logsumexp(lps) - lp_partial), "identity.coarsening.loglik", regime))
        w = np.exp(lps - logsumexp(lps))
        posts = [update_ranking_exact(m, v, o, base=base) for o in full_orders]
        mix_m = sum(wi * p[0] for wi, p in zip(w, posts))
        mix_v = sum(wi * (p[1] + p[0] ** 2) for wi, p in zip(w, posts)) - mix_m ** 2
        mp, vp = update_ranking_exact(m, v, partial, base=base)
        out.append(_res(ctx, f"{ctx.name}.partial_mixture_m.{bname}", _rel(mp, mix_m, 1.0),
                        "identity.coarsening.m", regime))
        out.append(_res(ctx, f"{ctx.name}.partial_mixture_v.{bname}", _rel(vp, mix_v, 1.0),
                        "identity.coarsening.v", regime))
        # the omitted entrant is untouched, exactly, under a diagonal prior
        out.append(_res(ctx, f"{ctx.name}.omitted_untouched.{bname}",
                        abs(mp[2] - m[2]) + abs(vp[2] - v[2]), "identity.coarsening.exact", regime))
        # ranked entrants equal the sub-field computation
        sub = np.array(partial)
        ms, vs = update_ranking_exact(m[sub], v[sub], [0, 1, 2, 3], base=base)
        out.append(_res(ctx, f"{ctx.name}.subfield.{bname}",
                        max(_rel(mp[sub], ms, 1.0), _rel(vp[sub], vs, 1.0)),
                        "identity.coarsening.subfield", regime))
        # winner-only equals the mixture over the (K-1)! orders with that winner
        Kw = 4
        mw = m[:Kw]; vw = v[:Kw]
        orders = [[1] + list(p) for p in itertools.permutations([0, 2, 3])]
        lps = np.array([_order_evidence(mw, vw, o, 1.0, base) for o in orders])
        w = np.exp(lps - logsumexp(lps))
        posts = [update_ranking_exact(mw, vw, o, base=base) for o in orders]
        mix_m = sum(wi * p[0] for wi, p in zip(w, posts))
        mix_v = sum(wi * (p[1] + p[0] ** 2) for wi, p in zip(w, posts)) - mix_m ** 2
        m1, v1, p1 = update_winner(mw, vw, 1, base=base)
        out.append(_res(ctx, f"{ctx.name}.winner_is_order_mixture_p.{bname}",
                        abs(np.log(p1) - logsumexp(lps)), "identity.coarsening.loglik", regime))
        out.append(_res(ctx, f"{ctx.name}.winner_is_order_mixture_m.{bname}", _rel(m1, mix_m, 1.0),
                        "identity.coarsening.m", regime))
        out.append(_res(ctx, f"{ctx.name}.winner_is_order_mixture_v.{bname}", _rel(v1, mix_v, 1.0),
                        "identity.coarsening.v", regime))
    return out


# ---------------------------------------------------------------------------
# I7: invariances
# ---------------------------------------------------------------------------

def _all_drivers(base, beta2=1.0):
    """(name, fn) with fn(m, v, event) -> (m', v', logZ) over every
    update type at V = 0 / diagonal S; event is (winner, order)."""
    from ..nway import (update_winner, update_ranking_exact,
                        update_winner_correlated, update_order_correlated)
    from ..full import update_winner_full, update_order_full

    def _w(m, v, ev):
        mm, vv, p = update_winner(m, v, ev[0], beta2=beta2, base=base)
        return mm, vv, float(np.log(p))

    def _o(m, v, ev):
        mm, vv = update_ranking_exact(m, v, ev[1], beta2=beta2, base=base)
        return mm, vv, _order_evidence(m, v, ev[1], beta2, base)

    def _wc(m, v, ev):
        mm, vv, lz = update_winner_correlated(m, v, ev[0], np.zeros((len(m), 1)),
                                              beta2=beta2, base=base)
        return mm, vv, float(lz)

    def _oc(m, v, ev):
        mm, vv, lz = update_order_correlated(m, v, ev[1], np.zeros((len(m), 1)),
                                             beta2=beta2, base=base)
        return mm, vv, float(lz)

    def _wf(m, v, ev):
        mm, SS, lz = update_winner_full(m, np.diag(v), ev[0], beta2=beta2, base=base)
        return mm, np.diag(SS).copy(), float(lz)

    def _of(m, v, ev):
        mm, SS, lz = update_order_full(m, np.diag(v), ev[1], beta2=beta2, base=base)
        return mm, np.diag(SS).copy(), float(lz)
    return [("update_winner", _w), ("update_ranking_exact", _o),
            ("update_winner_correlated", _wc), ("update_order_correlated", _oc),
            ("update_winner_full", _wf), ("update_order_full", _of)]


@check("identity.invariances", profiles=("fast",), group="identity", cost_s=6.0)
def invariances(ctx):
    """Gauge shift (a common constant added to every mean shifts every
    posterior mean by it and changes nothing else), permutation
    equivariance (relabelling permutes the answer), scale covariance
    (means, prior sds and noise sds scaled together scale the
    posterior; the evidence is unchanged), team assignment A = I equals
    the individual full update."""
    from ..market import update_market
    from ..teams import update_team_winner_full, update_team_order_full
    from ..full import update_winner_full, update_order_full
    out = []
    K = 5
    bases = ["normal", "logistic"] if ctx.profile != "smoke" else ["normal"]
    for bname in bases:
        base = _base_obj(bname)
        rng = ctx.rng(bname)
        m = rng.normal(0, 1, K); v = rng.uniform(0.5, 1.5, K)
        order = list(rng.permutation(K)); ev = (int(order[0]), order)
        regime = {"base": bname}
        perm = rng.permutation(K); inv = np.argsort(perm)
        ev_p = (int(inv[ev[0]]), [int(inv[j]) for j in order])
        for dname, fn in _all_drivers(base):
            try:
                m1, v1, z1 = fn(m, v, ev)
            except NotImplementedError as e:
                out.append(Result(f"{ctx.name}.{dname}.{bname}", "identity", "SKIP",
                                  regime=regime, detail=f"not supported: {str(e)[:60]}"))
                continue
            cls = "full" if dname.endswith("_full") else "diag"
            # gauge
            c = 3.7
            m2, v2, z2 = fn(m + c, v, ev)
            out.append(_res(ctx, f"{ctx.name}.gauge.{dname}.{bname}",
                            max(_rel(m2 - c, m1, 1.0), _rel(v2, v1, 1.0), abs(z2 - z1)),
                            f"identity.invariances.gauge.{cls}", regime))
            # permutation
            m3, v3, z3 = fn(m[perm], v[perm], ev_p)
            out.append(_res(ctx, f"{ctx.name}.permutation.{dname}.{bname}",
                            max(_rel(m3, m1[perm], 1.0), _rel(v3, v1[perm], 1.0), abs(z3 - z1)),
                            f"identity.invariances.permutation.{cls}", regime))
            # scale (m, sd, noise sd all by c): evidence unchanged, m by c, v by c^2
            for sc in (0.1, 10.0):
                fn_s = dict(_all_drivers(base, beta2=sc * sc))[dname]
                m4, v4, z4 = fn_s(sc * m, sc * sc * v, ev)
                out.append(_res(ctx, f"{ctx.name}.scale{sc}.{dname}.{bname}.m",
                                max(_rel(m4 / sc, m1, 1.0), abs(z4 - z1)),
                                "identity.invariances.scale_m", {**regime, "c": sc}))
                out.append(_res(ctx, f"{ctx.name}.scale{sc}.{dname}.{bname}.v",
                                _rel(v4 / (sc * sc), v1, 1.0),
                                f"identity.invariances.scale_v.{cls}", {**regime, "c": sc}))
    # market gauge: prices carry no level, so the level is untouched
    rng = ctx.rng("market")
    m = rng.normal(0, 1, K); v = np.full(K, 0.8); p = rng.dirichlet(np.ones(K) * 3)
    a1, b1, z1 = update_market(m, v, p, tau2=0.3)
    a2, b2, z2 = update_market(m + 2.5, v, p, tau2=0.3)
    out.append(_res(ctx, f"{ctx.name}.gauge.update_market",
                    max(_rel(a2 - 2.5, a1, 1.0), _rel(b2, b1, 1.0), abs(z2 - z1)),
                    "identity.invariances.gauge.diag"))
    # teams with the identity assignment
    S = np.diag(v) + 0.1
    m1, S1, z1 = update_winner_full(m, S, 2)
    m2, S2, z2 = update_team_winner_full(m, S, np.eye(K), 2)
    out.append(_res(ctx, f"{ctx.name}.team_identity_winner",
                    max(_rel(m1, m2, 1.0), _rel(S1, S2, 1.0), abs(z1 - z2)),
                    "identity.invariances.exact"))
    m1, S1, z1 = update_order_full(m, S, [2, 0, 4, 1, 3])
    m2, S2, z2 = update_team_order_full(m, S, np.eye(K), [2, 0, 4, 1, 3])
    out.append(_res(ctx, f"{ctx.name}.team_identity_order",
                    max(_rel(m1, m2, 1.0), _rel(S1, S2, 1.0), abs(z1 - z2)),
                    "identity.invariances.exact"))
    return out


# ---------------------------------------------------------------------------
# I4: predict versus evidence, and evidence bookkeeping
# ---------------------------------------------------------------------------

@check("identity.predict_evidence", profiles=("fast",), group="identity", cost_s=2.0)
def predict_evidence(ctx):
    """The probability the tracker's predict() assigns to the observed
    winner equals the evidence increment its winner update returns:
    fit and price under one model. Exact under the normal base
    (independent and blocked); non-normal bases are measured until the
    predictor prices the belief-noise convolution the update does (F2)."""
    from ..tracker import AbilityTracker
    out = []
    ids = list("abcdef")
    groups = ["x", "x", "y", "y", None, "z"]
    for bname in ("normal", "gumbel", "logistic", "laplace"):
        for rho in (0.0, 0.4):
            trk = AbilityTracker(base=bname, rho=rho, drift=0.0, init_var=1.0, Qf=5)
            rng = ctx.rng(f"{bname}:{rho}")
            for i, mm in zip(ids, rng.normal(0, 0.7, 6)):
                trk.state[i] = trk.state.get(i) or type(trk).__mro__[0]  # placeholder
            from ..tracker import AbilityState
            for i, mm in zip(ids, rng.normal(0, 0.7, 6)):
                trk.state[i] = AbilityState(float(mm), float(rng.uniform(0.5, 1.5)), 0.0)
            g = groups if rho > 0 else None
            p = trk.predict(ids, 0.0, groups=g)
            e0 = trk.evidence
            trk.observe(ids, 0.0, winner=2, groups=g)
            stat = abs(np.log(p[2]) - (trk.evidence - e0))
            key = ("identity.predict_evidence.normal" if bname == "normal"
                   else "identity.predict_evidence.other")
            if rho > 0 and bname == "normal":
                key = "identity.predict_evidence.blocked"
            out.append(_res(ctx, f"{ctx.name}.{bname}.rho{rho}", stat, key,
                            {"base": bname, "rho": rho}))
    return out


@check("identity.evidence_bookkeeping", profiles=("smoke",), group="identity", cost_s=2.0)
def evidence_bookkeeping(ctx):
    """rate_history resumed from a saved state equals the single pass
    (state and total evidence additive); the tracker's evidence equals
    the sum of its per-observation increments; predict is invariant to
    a common shift of the stored means."""
    from ..history import rate_history
    from ..simulate import history_world
    from ..tracker import AbilityTracker, AbilityState
    out = []
    races, _ = history_world(ctx.rng("history"), 8, 20, 50.0)
    ids = sorted({r for rc in races for r in rc["runners"]})
    r1, z1, st1 = rate_history(races, ids=ids, timescale=50.0, return_state=True)
    _, za, sta = rate_history(races[:10], ids=ids, timescale=50.0, return_state=True)
    _, zb, st2 = rate_history(races[10:], state=sta, timescale=50.0, return_state=True)
    out.append(_res(ctx, f"{ctx.name}.resume_state",
                    max(_rel(st1["m"], st2["m"], 1.0), _rel(st1["S"], st2["S"], 1.0)),
                    "identity.evidence.exact"))
    out.append(_res(ctx, f"{ctx.name}.resume_evidence", abs(z1 - (za + zb)),
                    "identity.evidence.exact"))
    trk = AbilityTracker(drift=0.0)
    rng = ctx.rng("tracker")
    ids = list("abcde")
    total = 0.0
    for t in range(4):
        before = trk.evidence
        trk.observe(ids, float(t), order=list(rng.permutation(5)))
        total += trk.evidence - before
    out.append(_res(ctx, f"{ctx.name}.tracker_increments", abs(total - trk.evidence),
                    "identity.evidence.exact"))
    p1 = trk.predict(ids, 4.0)
    for i in ids:
        s = trk.state[i]
        trk.state[i] = AbilityState(s.mean + 7.0, s.var, s.last_time)
    p2 = trk.predict(ids, 4.0)
    out.append(_res(ctx, f"{ctx.name}.predict_gauge", float(np.abs(p1 - p2).max()),
                    "identity.evidence.exact"))
    return out
