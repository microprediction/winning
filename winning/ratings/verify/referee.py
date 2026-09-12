"""Monte Carlo referees (P8 extended): the posterior of the generative
model the updates claim to serve, sampled by rejection, against the
analytic moments -- for the independent, correlated, full-covariance
and team updates, at fixed configurations (research/adjudications/
predictive_referee.py's rows, chosen in 2026-09-04 to include ones the
original audit did not) and at configurations drawn from marks.
CONFIG_SEED so the implementer neither picks nor sees them first.

Pass marks are the referee's own: |dm| <= 0.005 + 4 SE on every mean,
|dv| / v <= 0.006 + 4 SE on every variance (SE from the accepted
sample), cross terms of a full covariance 0.02 + 4 SE. The normal base
is the regression row: its path is exact and must stay at its
historical accuracy.
"""

from __future__ import annotations

import numpy as np

from .core import Result, check
from . import marks
from ..simulate import correlated_draws


def _base_obj(name):
    from ...factor.races import failure_base, student_base
    return {"student4": student_base(4.0), "failure": failure_base(0.15)}.get(name, name)


def mc_posterior(rng, m, v, beta2, event, base="normal", V=None, N=1_000_000,
                 chunk=250_000):
    """Rejection-sampled posterior of the skills given the event
    ("winner", i) or ("order", list): mean, covariance, acceptance
    rate and the count accepted. Draws in chunks so N can be large."""
    m = np.asarray(m, float); v = np.asarray(v, float)
    n = len(m)
    kept = []
    total = 0
    done = 0
    while done < N:
        M = min(chunk, N - done)
        s, x = correlated_draws(rng, M, m, v, V=V, beta2=beta2, base=base)
        if event[0] == "winner":
            keep = x.argmax(axis=1) == int(event[1])
        else:
            order = np.asarray(event[1], int)
            keep = (np.argsort(-x, axis=1) == order[None, :]).all(axis=1)
        kept.append(s[keep])
        total += int(keep.sum())
        done += M
    sk = np.concatenate(kept) if kept else np.zeros((0, n))
    if total < 50:
        return None
    return {"mean": sk.mean(axis=0), "cov": np.cov(sk.T, ddof=1),
            "p": total / N, "n": total}


MIN_ACCEPTED = 2000


def _compare(ctx, name, mm, vv, ref, regime, SS=None):
    """Results for means, variances and (optionally) cross terms. Fewer
    than MIN_ACCEPTED accepted draws cannot decide a 0.005 mark (the
    4 SE term would swamp it), so the cell is UNDERPOWERED rather than
    a pass with a meaningless mark."""
    out = []
    n = ref["n"]
    if n < MIN_ACCEPTED:
        return [Result(name, "referee", "UNDERPOWERED", n=n, regime=regime,
                       detail=f"only {n} accepted draws (acceptance {ref['p']:.2e}); "
                              "condition on a likelier event or raise the draw count")]
    mc_m, mc_v = ref["mean"], np.diag(ref["cov"])
    se_m = float(np.sqrt(mc_v / n).max())
    dm = float(np.abs(np.asarray(mm) - mc_m).max())
    verdict, tol = ctx.verdict(dm, key="referee.mean", se=se_m, k=4)
    out.append(Result(f"{name}.mean", "referee", verdict, statistic=dm, se=se_m,
                      tolerance=tol, regime=regime, n=n,
                      detail="" if verdict != "FAIL" else "posterior mean off"))
    se_v = float(np.sqrt(2.0 / n))
    dv = float((np.abs(np.asarray(vv) - mc_v) / mc_v).max())
    verdict, tol = ctx.verdict(dv, key="referee.var", se=se_v, k=4)
    out.append(Result(f"{name}.var", "referee", verdict, statistic=dv, se=se_v,
                      tolerance=tol, regime=regime, n=n,
                      detail="" if verdict != "FAIL" else "posterior variance off"))
    if SS is not None:
        C = ref["cov"]
        off = ~np.eye(len(mm), dtype=bool)
        dc = float(np.abs(np.asarray(SS)[off] - C[off]).max())
        se_c = float(np.sqrt(np.abs(C[off]).max() ** 2 / n + (mc_v.max() ** 2) / n))
        verdict, tol = ctx.verdict(dc, key="referee.cross", se=se_c, k=4)
        out.append(Result(f"{name}.cross", "referee", verdict, statistic=dc, se=se_c,
                          tolerance=tol, regime=regime, n=n,
                          detail="" if verdict != "FAIL" else "posterior cross terms off"))
    return out


WINNER_CONFIGS = [
    ("A_bandits", np.array([0.4, 0.0, -0.3]), np.array([0.6, 0.4, 0.5]), 1.0, 0),
    ("B_n4_longshot", np.array([0.0, -0.5, 0.3, 0.8]), np.array([1.2, 0.3, 0.7, 0.5]), 0.6, 2),
    ("C_diffuse", np.array([1.0, -1.0]), np.array([4.0, 0.25]), 1.5, 1),
]
ORDER_CONFIGS = [
    ("A_order021", np.array([0.4, 0.0, -0.3]), np.array([0.6, 0.4, 0.5]), 1.0, [0, 2, 1]),
    ("B_order3021", np.array([0.0, -0.5, 0.3, 0.8]), np.array([1.2, 0.3, 0.7, 0.5]), 0.6, [3, 0, 2, 1]),
]
REFEREE_BASES = ["normal", "logistic", "laplace", "gumbel", "student4"]


def fresh_configs(rng, n, kinds=("winner", "order")):
    """Configurations the implementer did not choose: field size 3-6
    (orders 3-4), spread means, unequal variances up to 3, noise
    variance 0.5-2, near-ties with probability one half."""
    out = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        K = int(rng.integers(3, 7)) if kind == "winner" else int(rng.integers(3, 5))
        m = rng.normal(0, 0.8, K)
        if rng.random() < 0.5:
            m[:2] = m[0] + rng.normal(0, 0.05, 2)          # a near tie
        v = np.exp(rng.uniform(np.log(0.05), np.log(3.0), K))
        beta2 = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        if kind == "winner":
            ev = ("winner", int(rng.integers(K)))
        else:
            ev = ("order", list(rng.permutation(K)))
        out.append((f"fresh{i}_{kind}_K{K}", m, v, beta2, ev))
    return out


@check("referee.winner_fixed", profiles=("fast",), group="referee", cost_s=8.0)
def winner_fixed(ctx):
    """update_winner against the rejection-sampled posterior at the
    three fixed configurations, five bases."""
    from ..nway import update_winner
    N = int(ctx.param("referee_draws"))
    out = []
    for cname, m, v, beta2, w in WINNER_CONFIGS:
        for bname in REFEREE_BASES:
            base = _base_obj(bname)
            ref = mc_posterior(ctx.rng(f"{cname}:{bname}"), m, v, beta2, ("winner", w),
                               base=base, N=N)
            mm, vv, p = update_winner(m, v, w, beta2=beta2, base=base)
            reg = {"config": cname, "base": bname}
            out += _compare(ctx, f"{ctx.name}.{cname}.{bname}", mm, vv, ref, reg)
    return out


@check("referee.order_fixed", profiles=("fast",), group="referee", cost_s=8.0)
def order_fixed(ctx):
    """update_ranking_exact against the rejection-sampled posterior of
    a full finishing order at the two fixed configurations."""
    from ..nway import update_ranking_exact
    N = int(ctx.param("referee_draws"))
    out = []
    for cname, m, v, beta2, order in ORDER_CONFIGS:
        for bname in REFEREE_BASES:
            base = _base_obj(bname)
            ref = mc_posterior(ctx.rng(f"{cname}:{bname}"), m, v, beta2, ("order", order),
                               base=base, N=N)
            if ref is None:
                out.append(Result(f"{ctx.name}.{cname}.{bname}", "referee", "UNDERPOWERED",
                                  detail="fewer than 50 accepted draws"))
                continue
            mm, vv = update_ranking_exact(m, v, order, beta2=beta2, base=base)
            reg = {"config": cname, "base": bname}
            out += _compare(ctx, f"{ctx.name}.{cname}.{bname}", mm, vv, ref, reg)
    return out


@check("referee.correlated", profiles=("full",), group="referee", cost_s=60.0)
def correlated(ctx):
    """The correlated updates with one-factor loadings against the
    rejection-sampled posterior (winner and order), normal and
    logistic bases."""
    from ..nway import update_winner_correlated, update_order_correlated
    N = int(ctx.param("referee_draws_full"))
    out = []
    m = np.array([0.3, 0.0, -0.2, 0.1, -0.4]); v = np.array([0.8, 1.0, 0.6, 1.2, 0.9])
    V = np.array([[1.0], [0.9], [0.1], [-0.8], [0.2]])
    for bname in ("normal", "logistic"):
        base = _base_obj(bname)
        ref = mc_posterior(ctx.rng(f"w:{bname}"), m, v, 1.0, ("winner", 3), base=base, V=V, N=N)
        mm, vv, _ = update_winner_correlated(m, v, 3, V, base=base)
        out += _compare(ctx, f"{ctx.name}.winner.{bname}", mm, vv, ref, {"base": bname})
        m4 = np.array([0.4, 0.0, -0.3, 0.0]); v4 = np.full(4, 0.9)
        V4 = np.array([[0.9], [0.8], [-0.7], [0.1]]); order = [1, 0, 3, 2]
        ref = mc_posterior(ctx.rng(f"o:{bname}"), m4, v4, 1.0, ("order", order), base=base,
                           V=V4, N=2 * N)
        if ref is None:
            out.append(Result(f"{ctx.name}.order.{bname}", "referee", "UNDERPOWERED",
                              detail="fewer than 50 accepted draws"))
            continue
        mm, vv, _ = update_order_correlated(m4, v4, order, V4, base=base)
        out += _compare(ctx, f"{ctx.name}.order.{bname}", mm, vv, ref, {"base": bname})
    return out


@check("referee.full_covariance", profiles=("full",), group="referee", cost_s=90.0)
def full_covariance(ctx):
    """update_winner_full / update_order_full with a dense rank-2
    belief and one-factor loadings against the rejection-sampled joint
    posterior, cross terms included; the diffuse-prior ladder that the
    belief split must survive."""
    from ..full import update_winner_full, update_order_full
    N = int(ctx.param("referee_draws_full"))
    out = []
    rng0 = ctx.rng("belief")
    K = 4
    for prior_sd in (1.0, 3.0, 10.0):
        B = rng0.normal(0, 0.5, (K, 2)) * prior_sd
        S = B @ B.T + np.diag(prior_sd ** 2 * rng0.uniform(0.3, 1.0, K))
        m = rng0.normal(0, prior_sd, K)
        V = (0.6 * (-1.0) ** np.arange(K))[:, None]
        # joint draws with a dense belief: s = m + L z. The event is a
        # PRIOR-PREDICTIVE draw (the typical event at this belief), not a
        # fixed order: under a spread of ten standard deviations a fixed
        # order can be astronomically improbable, leaving both the sampler
        # and the update in the degrade-gracefully regime (the baseline
        # run's sd 10 cell accepted almost nothing and read as a FAIL)
        L = np.linalg.cholesky(S)
        rng_ev = ctx.rng(f"{prior_sd}:event")
        x0 = m + rng_ev.normal(size=K) @ L.T + rng_ev.normal(size=1) @ V.T + rng_ev.normal(size=K)
        events = (("winner", int(np.argmax(x0))), ("order", list(map(int, np.argsort(-x0)))))
        for kind, ev in events:
            rng = ctx.rng(f"{prior_sd}:{kind}")
            kept, total, done = [], 0, 0
            NN = N if kind == "winner" else 2 * N
            while done < NN:
                Mc = min(250_000, NN - done)
                z = rng.normal(size=(Mc, K))
                s = m + z @ L.T
                f = rng.normal(size=(Mc, 1))
                x = s + f @ V.T + rng.normal(size=(Mc, K))
                keep = (x.argmax(axis=1) == ev if kind == "winner"
                        else (np.argsort(-x, axis=1) == np.asarray(ev)[None, :]).all(axis=1))
                kept.append(s[keep]); total += int(keep.sum()); done += Mc
            sk = np.concatenate(kept)
            if total < 50:
                out.append(Result(f"{ctx.name}.{kind}.sd{prior_sd}", "referee", "UNDERPOWERED",
                                  detail="fewer than 50 accepted draws"))
                continue
            ref = {"mean": sk.mean(0), "cov": np.cov(sk.T, ddof=1), "n": total, "p": total / NN}
            reg = {"prior_sd": prior_sd, "kind": kind, "event": ev}
            if kind == "winner":
                mm, SS, _ = update_winner_full(m, S, ev, V=V)
                out += _compare(ctx, f"{ctx.name}.{kind}.sd{prior_sd}", mm, np.diag(SS), ref,
                                reg, SS=SS)
                continue
            # the order update under a diffuse dense belief: on prior-centred
            # nodes the 2^10 Sobol cloud carried 13-35 percent effective
            # weight and the baseline measured relative variance errors of
            # 0.13 at prior sd 10 and 0.05 at 3 (2026-09-11). The mixture
            # engines now recentre the cloud on the factor posterior
            # (nway._recentre_nodes): 0.0079 and 0.024 at the same nodes,
            # so every cell gates; the 2^12 row is kept as the remedy ladder.
            mm, SS, _ = update_order_full(m, S, ev, V=V)
            out += _compare(ctx, f"{ctx.name}.{kind}.sd{prior_sd}", mm, np.diag(SS), ref,
                            reg, SS=SS)
            if prior_sd > 1.0:
                mc_v = np.diag(ref["cov"])
                mm, SS, _ = update_order_full(m, S, ev, V=V, nodes_log2=12)
                out.append(Result(f"{ctx.name}.{kind}.sd{prior_sd}.nodes12", "referee",
                                  "MEASURED", regime={**reg, "nodes_log2": 12}, n=total,
                                  statistic=float((np.abs(np.diag(SS) - mc_v) / mc_v).max()),
                                  detail="the same cell at 2^12 nodes (remedy ladder)",
                                  extras={"dm_over_sd": float(np.abs(mm - ref["mean"]).max() / prior_sd),
                                          "cross": float(np.abs((SS - ref["cov"])[~np.eye(K, dtype=bool)]).max())}))
    return out


@check("referee.teams", profiles=("full",), group="referee", cost_s=30.0)
def teams(ctx):
    """update_team_winner_full lifts a team result to the members:
    against the rejection-sampled player posterior, cross terms
    included (three pairs, one winner)."""
    from ..teams import update_team_winner_full
    N = int(ctx.param("referee_draws_full"))
    rng = ctx.rng("teams")
    n, k = 6, 3
    A = np.zeros((k, n))
    for team in range(k):
        A[team, 2 * team] = 1.0; A[team, 2 * team + 1] = 1.0
    m = rng.normal(size=n) * 0.3
    S = np.diag(0.5 + rng.random(n))
    kept, total, done = [], 0, 0
    while done < N:
        Mc = min(250_000, N - done)
        s = m + rng.standard_normal((Mc, n)) * np.sqrt(np.diag(S))
        X = s @ A.T + rng.standard_normal((Mc, k))
        keep = X.argmax(axis=1) == 1
        kept.append(s[keep]); total += int(keep.sum()); done += Mc
    sk = np.concatenate(kept)
    ref = {"mean": sk.mean(0), "cov": np.cov(sk.T, ddof=1), "n": total, "p": total / N}
    mm, SS, _ = update_team_winner_full(m, S, A, 1, beta2=1.0)
    return _compare(ctx, f"{ctx.name}.winner", mm, np.diag(SS), ref, {}, SS=SS)


@check("referee.fresh", profiles=("full",), group="referee", cost_s=120.0)
def fresh(ctx):
    """Configurations drawn from marks.CONFIG_SEED (not chosen by the
    implementer): the independent winner and order updates against the
    rejection-sampled posterior on every referee base."""
    from ..nway import update_winner, update_ranking_exact
    N = int(ctx.param("referee_draws_full"))
    out = []
    cfg = fresh_configs(np.random.default_rng(marks.CONFIG_SEED), 8)
    for cname, m, v, beta2, ev in cfg:
        for bname in REFEREE_BASES:
            base = _base_obj(bname)
            ref = mc_posterior(ctx.rng(f"{cname}:{bname}"), m, v, beta2, ev, base=base,
                               N=N if ev[0] == "winner" else 2 * N)
            if ref is None:
                out.append(Result(f"{ctx.name}.{cname}.{bname}", "referee", "UNDERPOWERED",
                                  detail="fewer than 50 accepted draws"))
                continue
            if ev[0] == "winner":
                mm, vv, _ = update_winner(m, v, ev[1], beta2=beta2, base=base)
            else:
                mm, vv = update_ranking_exact(m, v, ev[1], beta2=beta2, base=base)
            reg = {"config": cname, "base": bname, "config_seed": marks.CONFIG_SEED,
                   "m": np.round(m, 3).tolist(), "v": np.round(v, 3).tolist(), "beta2": beta2,
                   "event": [ev[0], ev[1] if ev[0] == "winner" else list(map(int, ev[1]))]}
            out += _compare(ctx, f"{ctx.name}.{cname}.{bname}", mm, vv, ref, reg)
    return out
