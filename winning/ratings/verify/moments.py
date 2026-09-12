"""Enumerated moment identities (I1-I3 of the plan), the sharpest test a
Gaussian-belief moment update can be given.

For an update m' = m + v g, v' = v + v^2 h with g and h the gradient and
diagonal curvature of log P(y | m) under the prior predictive, three
identities hold EXACTLY when g and h are the true derivatives of the
true marginal likelihood, whatever the base:

  I3  sum_y P(y) = 1                                (normalisation)
  I1  sum_y P(y) (m'_y - m) = 0                     (score identity)
  I2  sum_y P(y) (v'_y + (m'_y - m)^2) = v          (Bartlett identity)

with the sum over every outcome y the observation can take: the K
winners, or the K! finishing orders. The engine's own P(y) is used, so
the sums are deterministic and MC-free (tier E); deviations measure
lattice mass, adjoint or JVP gradients, the finite-difference curvature
step, the log-concave clamp and the variance floor -- and nothing else,
because any node set whose weights sum to one is itself a valid model
(the identities are quadrature-blind; the generative tier and the MC
referee see quadrature). I2 is reported twice: as shipped, and with the
curvature clamp bypassed, so a clamp that is doing work shows up as the
difference. Full-covariance updates get the matrix form
sum_y P(y) (S'_y + d_y d_y') = S.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager

import numpy as np

from .core import Result, check
from .. import nway as _nway


# ---------------------------------------------------------------------------
# drivers: one signature for every update type
# ---------------------------------------------------------------------------

def _order_evidence(m, v, order, beta2, base):
    sd = np.sqrt(np.asarray(v, float) + np.asarray(beta2, float))
    curves = None if base == "normal" else _nway._predictive_curves(v, beta2, base)
    lp, _ = _nway._order_pass(np.asarray(m, float), sd, np.asarray(order, int),
                              base=base, curves=curves)
    return float(lp)


def winner_drivers(base, V, beta2):
    """(name, fn) with fn(m, v, w) -> (m', v', logp[, S'])."""
    from ..nway import update_winner, update_winner_correlated
    from ..full import update_winner_full
    drivers = []
    if V is None:
        def _w(m, v, w):
            mm, vv, p = update_winner(m, v, w, beta2=beta2, base=base)
            return mm, vv, float(np.log(max(p, 1e-300)))
        drivers.append(("update_winner", _w))
    Vz = V

    def _wc(m, v, w):
        Vv = np.zeros((len(m), 1)) if Vz is None else Vz
        mm, vv, lz = update_winner_correlated(m, v, w, Vv, beta2=beta2, base=base)
        return mm, vv, float(lz)
    drivers.append(("update_winner_correlated", _wc))

    def _wf(m, v, w):
        mm, SS, lz = update_winner_full(m, np.diag(v), w, V=Vz, beta2=beta2, base=base)
        return mm, np.diag(SS).copy(), float(lz), SS
    drivers.append(("update_winner_full", _wf))
    return drivers


def order_drivers(base, V, beta2):
    from ..nway import update_ranking_exact, update_order_correlated
    from ..full import update_order_full
    drivers = []
    if V is None:
        def _o(m, v, order):
            mm, vv = update_ranking_exact(m, v, order, beta2=beta2, base=base)
            return mm, vv, _order_evidence(m, v, order, beta2, base)
        drivers.append(("update_ranking_exact", _o))
    Vz = V

    def _oc(m, v, order):
        Vv = np.zeros((len(m), 1)) if Vz is None else Vz
        mm, vv, lz = update_order_correlated(m, v, order, Vv, beta2=beta2, base=base)
        return mm, vv, float(lz)
    drivers.append(("update_order_correlated", _oc))

    def _of(m, v, order):
        mm, SS, lz = update_order_full(m, np.diag(v), order, V=Vz, beta2=beta2, base=base)
        return mm, np.diag(SS).copy(), float(lz), SS
    drivers.append(("update_order_full", _of))
    return drivers


@contextmanager
def unclamped_curvature():
    """Bypass nway._clamp_d2 (the log-concave curvature clamp) so I2 can
    be measured on the raw finite-difference curvature."""
    orig = _nway._clamp_d2
    _nway._clamp_d2 = lambda d2, base: d2
    try:
        yield
    finally:
        _nway._clamp_d2 = orig


# ---------------------------------------------------------------------------
# the sums
# ---------------------------------------------------------------------------

def enumerate_identities(fn, m, v, outcomes, matrix=False):
    """Run fn over every outcome, weight by the engine's own P(y), and
    return dict(I1, I2, I3[, I2_matrix]) as max-abs residuals (I2
    relative to v; the matrix form relative to the Frobenius norm of
    diag(v))."""
    m = np.asarray(m, float); v = np.asarray(v, float)
    ps, dm, tv, SS = [], [], [], []
    for y in outcomes:
        out = fn(m, v, y)
        mm, vv, lp = out[0], out[1], out[2]
        p = float(np.exp(lp))
        ps.append(p)
        d = np.asarray(mm, float) - m
        dm.append(p * d)
        tv.append(p * (np.asarray(vv, float) + d * d))
        if matrix and len(out) > 3:
            SS.append(p * (np.asarray(out[3], float) + np.outer(d, d)))
    ps = np.asarray(ps)
    res = {"I3": float(abs(ps.sum() - 1.0)),
           "I1": float(np.abs(np.sum(dm, axis=0)).max()),
           "I2": float((np.abs(np.sum(tv, axis=0) - v) / v).max())}
    if matrix and SS:
        S0 = np.diag(v)
        res["I2_matrix"] = float(np.linalg.norm(np.sum(SS, axis=0) - S0)
                                 / np.linalg.norm(S0))
    return res


def _cell_prior(rng, K, prior_var):
    m = rng.normal(0.0, np.sqrt(prior_var), K)
    v = prior_var * rng.uniform(0.5, 1.5, K)
    return m, v


def _loadings(kind, K, rng=None):
    if kind == 0 or kind == "0":
        return None
    if kind == "1f":
        return (0.6 * (-1.0) ** np.arange(K))[:, None]
    if kind == "2f":
        j = np.arange(K)
        return np.column_stack([0.5 * (-1.0) ** j, 0.5 * (-1.0) ** (j // 2)])
    raise ValueError(kind)


def _base_obj(name):
    from ...factor.races import failure_base, student_base
    return {"student4": student_base(4.0), "failure": failure_base(0.15)}.get(name, name)


def _base_class(name):
    return "normal" if name == "normal" else "curve"


# regime grid per profile: (base, beta2, prior_var, V, K, K_orders) with
# K_orders the size of the sub-field whose K_orders! orders are
# enumerated (0: winners only). Costs (single core): K! orders on the
# correlated path ~2.4 s at K=5 normal, ~7 s on curve bases; two-factor
# loadings put 49 (correlated) and 81 (full) nodes under every order
# and cost 17-25 s per K=5 cell, so fast enumerates their winners only.
_FAST = [("normal", 1.0, 1.0, "0", 5, 5),
         ("gumbel", 1.0, 1.0, "0", 5, 3), ("logistic", 1.0, 1.0, "0", 5, 3),
         ("laplace", 1.0, 1.0, "0", 5, 3), ("student4", 1.0, 1.0, "0", 5, 3),
         ("failure", 1.0, 1.0, "0", 5, 3),
         ("normal", 0.25, 1.0, "0", 4, 4), ("normal", 4.0, 1.0, "0", 4, 4),
         ("normal", 1.0, 0.25, "0", 4, 4), ("normal", 1.0, 9.0, "0", 4, 4),
         ("normal", 1.0, 1.0, "1f", 4, 4), ("normal", 1.0, 1.0, "2f", 5, 0),
         ("normal", 1.0, 1.0, "0", 2, 2), ("normal", 1.0, 1.0, "0", 8, 0),
         ("logistic", 1.0, 1.0, "1f", 2, 2)]
_SMOKE = [("normal", 1.0, 1.0, "0", 3, 3), ("laplace", 1.0, 1.0, "0", 3, 3)]


def _grid(profile):
    if profile == "smoke":
        return _SMOKE
    if profile == "fast":
        return _FAST
    cells = []
    for b2 in (0.25, 1.0, 4.0):
        for pv in (0.25, 1.0, 9.0):
            for Vk in ("0", "1f", "2f"):
                for K in (2, 5, 8):
                    ko = {2: 2, 5: 5, 8: 0}[K]
                    if K == 5 and Vk == "2f":
                        ko = 4
                    cells.append(("normal", b2, pv, Vk, K, ko))
    cells.append(("normal", 1.0, 1.0, "0", 6, 6))
    for base in ("gumbel", "logistic", "laplace", "student4", "failure"):
        for b2 in (0.25, 1.0, 4.0):
            for pv in (0.25, 1.0, 9.0):
                for Vk in ("0", "1f"):
                    for K in (2, 5):
                        cells.append((base, b2, pv, Vk, K, K if K == 2 else 4))
    return cells


# how each driver obtains its curvature: analytic (the independent
# winner update's curve path), an absolute central difference of the
# mixture gradient at eps = 1e-3 (nway._mixture_update, update_ranking_
# exact), or a relative step of 0.15 sd (full._mixture_update_full,
# which has no log-concave clamp)
_CURVATURE = {"update_winner": "analytic",
              "update_winner_correlated": "fd", "update_ranking_exact": "fd",
              "update_order_correlated": "fd",
              "update_winner_full": "fd_rel", "update_order_full": "fd_rel"}


def _mark_for(ctx, ident, base_name, event, driver):
    m = ctx.mark("identity.moments")
    cls = _base_class(base_name)
    if ident == "I2":
        key = f"I2.{cls}.{_CURVATURE[driver]}.{event}"
    else:
        key = f"{ident}.{cls}.{event}"
    return float(m["floors"][key])


@check("identity.moments", profiles=("smoke",), group="identity", cost_s=25.0)
def moments(ctx):
    """I1/I2/I3 enumerated over the K winners and the K_orders! finishing
    orders of a sub-field, for every update type at the profile's regime
    grid; I2 as shipped and, where a clamp exists, with it bypassed."""
    out = []
    for (bname, beta2, prior_var, Vk, K, K_orders) in _grid(ctx.profile):
        base = _base_obj(bname)
        regime = {"base": bname, "beta2": beta2, "prior_var": prior_var, "V": Vk, "K": K}
        tag = f"{bname}.b{beta2}.pv{prior_var}.V{Vk}.K{K}"
        rng = ctx.rng(tag)
        m, v = _cell_prior(rng, K, prior_var)
        V = _loadings(Vk, K)
        events = [("winner", list(range(K)), winner_drivers(base, V, beta2), m, v)]
        if K_orders >= 2:
            ko = int(K_orders)
            Vo = None if V is None else V[:ko]
            events.append(("order", [list(p) for p in itertools.permutations(range(ko))],
                           order_drivers(base, Vo, beta2), m[:ko], v[:ko]))
        for event, outcomes, drivers, mm, vv in events:
            for dname, fn in drivers:
                sub = "" if event == "winner" else f".Ko{len(mm)}"
                name = f"{ctx.name}.{dname}.{event}.{tag}{sub}"
                has_clamp = _CURVATURE[dname] != "fd_rel"
                try:
                    res = enumerate_identities(fn, mm, vv, outcomes, matrix=dname.endswith("_full"))
                    raw = None
                    if has_clamp:
                        with unclamped_curvature():
                            raw = enumerate_identities(fn, mm, vv, outcomes)
                except NotImplementedError as e:
                    out.append(Result(name, "identity", "SKIP", regime=regime,
                                      detail=f"not supported: {str(e)[:80]}"))
                    continue
                for ident in ("I1", "I2", "I3"):
                    tol = _mark_for(ctx, ident, bname, event, dname)
                    stat = res[ident]
                    out.append(Result(f"{name}.{ident}", "identity",
                                      "ok" if stat <= tol else "FAIL",
                                      statistic=stat, tolerance=tol, regime=regime,
                                      n=len(outcomes)))
                if raw is not None:
                    tol = _mark_for(ctx, "I2", bname, event, dname)
                    out.append(Result(f"{name}.I2_raw", "identity",
                                      "ok" if raw["I2"] <= tol else "FAIL",
                                      statistic=raw["I2"], tolerance=tol, regime=regime,
                                      n=len(outcomes),
                                      detail="" if raw["I2"] <= tol else
                                      "unclamped curvature breaks total variance: the clamp is doing work"))
                if "I2_matrix" in res:
                    tol = _mark_for(ctx, "I2", bname, event, dname)
                    out.append(Result(f"{name}.I2_matrix", "identity",
                                      "ok" if res["I2_matrix"] <= tol else "FAIL",
                                      statistic=res["I2_matrix"], tolerance=tol,
                                      regime=regime, n=len(outcomes)))
    return out
