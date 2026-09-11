"""Factor ratings: an entity's ability as a small vector over observed
conditions, fitted by MAP through the engine's ranking likelihood.

Model. Entrant j of a contest has ability mu_j = z_j . theta, with z_j
an observed feature row and theta shared across the data set -- the
DESIGN form (fit_design_ratings). The factor rating is the special
case in which entity i owns a coefficient row B_i over a covariate
vector x observed on the contest: a level plus offsets, mu = B_i . x
(fit_factor_ratings). Conditional on the features the performances are
independent with unit base noise, so an observed finishing order has
likelihood order_loglik and the gradient is the engine's ranking
gradient pushed through the design, dL/dtheta = Z' dL/dmu.

Measured in the bandits programme (Chatbot Arena, 65,178 battles over
53 models; Lichess, 59,399 games over 639 players; the spec is
bandits/specs/factor_ratings_api.md and every number below has a
per-observation CSV behind it):

- Pooling, not conditioning, is the active ingredient. Within one
  estimator a separate rating per category does not significantly beat
  one scalar rating on either data set (P 0.81-0.88 on Arena; +0.0006
  n.s. on chess), while the factor form, one level estimated from every
  contest with offsets shrunk toward it, beats both: -0.0051 [-0.0071,
  -0.0030] nats per game against the stratified arm on chess, monotone
  in cell sparsity. Against the production system the decomposition
  matters (bandits exp35, three independent splits of the 59,399 games;
  bandits/results/exp35_glicko2_seed{0,11,22}.csv and
  results/exp35_output.txt): Glicko-2 pooled 0.6323 / 0.6304 / 0.6365,
  Glicko-2 per time control (Lichess's architecture) 0.6234 / 0.6241 /
  0.6294, this estimator's scalar 0.6117 / 0.6131 / 0.6145, factor
  0.6073 / 0.6104 / 0.6112. Factor beats Lichess's architecture by
  0.014-0.018 nats, every interval excluding zero; roughly four fifths
  of that is the estimator (0.017-0.022) and one fifth the factor
  structure (0.003-0.004, intervals excluding zero on every split).
  Stratification does help Glicko-2 (+0.006 to +0.009), so the finding
  is that partial pooling beats it within a good estimator, not that
  conditioning is useless -- the spec's section 2.2 said otherwise and
  is formally corrected.
- The per-feature ridge is the mechanism, not a nicety. One shared
  ridge made the factor arm LOSE on chess time control (+0.0037);
  levels free and offsets shrunk turned the same data into a decisive
  win (-0.0045 [-0.0057, -0.0032]). ridge= therefore takes a vector,
  and sweep_offset_ridge tunes the offset penalty on a validation
  slice, reporting when the optimum sits at the scalar limit (the
  axis is then unidentifiable in that data).
- Dimension is selectable from data the user may see: on Arena the
  test loss troughs at a level plus two offsets and validation
  troughs at the same place; d = 4 and 5 are significantly worse.
- The gain is real, bounded and channel-limited: ~3% of what a rating
  extracts over uniform on binary comparisons (eight times what a
  saturated per-matchup model achieves), sixteen percent on cardinal
  benchmark scores.

Two transfer conditions, both checkable before fitting. Abilities must
be static relative to the covariate: on Formula 1, constructor
identity looked worth -0.25 nats against a driver-only baseline until
the baseline was given a tuned recency half-life, after which it was
worth +0.04 -- an omitted time dimension flatters any richer
parameterisation, so tune weights= for every arm before comparing.
And the covariate must be exogenously varied: chess opening family
gives a null result because players ARE their openings (per-player
tactical share sd 0.415 against a no-choice binomial null of 0.058),
not because the axis is absent (it is real at ~4 sigma). A null on a
self-selected covariate means unidentifiable, not absent;
covariate_contrast_report is the one-line check.

Max-wins throughout: order lists rows best first, higher theta is
better, and predictions negate into the min-wins race engine.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import log_ndtr

from .nway import _base_rows, _order_pass_batch

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_WORK_BUDGET = 4_000_000        # floats per (K, chunk, L) lattice work array
_LP_FLOOR = -700.0              # a numerically impossible order, kept finite


# ---------------------------------------------------------------------------
# design assembly
# ---------------------------------------------------------------------------

def _finish_perm(order, K):
    """Row permutation putting the ordered rows first, best first, then
    the unranked rows (a partial order marginalises them exactly)."""
    order = np.asarray(order, dtype=int).ravel()
    if order.size < 1 or order.size > K:
        raise ValueError("order must list between 1 and K distinct rows")
    if order.min() < 0 or order.max() >= K or len(set(order.tolist())) != order.size:
        raise ValueError("order must be distinct row indices in 0..K-1")
    rest = np.setdiff1d(np.arange(K), order)
    return np.concatenate([order, rest]), order.size


class _Stacked:
    """Every event's design rows in one sparse matrix, each event's rows
    in finishing order (ordered rows first). Events are grouped by
    (field size, number of ranked rows) so the lattice passes run
    batched across events of a kind."""

    def __init__(self, triplets, n_feat):
        rows, cols, vals = [], [], []
        starts, sizes, n_ord = [], [], []
        r0 = 0
        for r, c, v, K, n in triplets:
            rows.append(np.asarray(r, dtype=int) + r0)
            cols.append(np.asarray(c, dtype=int))
            vals.append(np.asarray(v, dtype=float))
            starts.append(r0)
            sizes.append(int(K))
            n_ord.append(int(n))
            r0 += int(K)
        if not starts:
            raise ValueError("no events")
        self.n_feat = int(n_feat)
        self.n_rows = r0
        self.n_events = len(starts)
        self.starts = np.asarray(starts)
        self.sizes = np.asarray(sizes)
        self.n_ordered = np.asarray(n_ord)
        rows = np.concatenate(rows) if rows else np.zeros(0, dtype=int)
        cols = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
        vals = np.concatenate(vals) if vals else np.zeros(0)
        if cols.size and (cols.min() < 0 or cols.max() >= n_feat):
            raise ValueError("feature index out of range for n_feat")
        self.Z = sparse.csr_matrix((vals, (rows, cols)),
                                   shape=(self.n_rows, self.n_feat))
        self.ZT = self.Z.T.tocsr()
        self.groups = {}
        for e, key in enumerate(zip(self.sizes.tolist(), self.n_ordered.tolist())):
            self.groups.setdefault(key, []).append(e)
        self.groups = {k: np.asarray(v) for k, v in self.groups.items()}
        # per-row event index, for expanding event weights to rows
        self.row_event = np.repeat(np.arange(self.n_events), self.sizes)


def _design_triplets(events, n_feat):
    for ev in events:
        if len(ev) != 2:
            raise ValueError("design events are (Z, order)")
        Z, order = ev
        if sparse.issparse(Z):
            Zc = sparse.csr_matrix(Z)
        else:
            Zc = sparse.csr_matrix(np.atleast_2d(np.asarray(Z, dtype=float)))
        K, nf = Zc.shape
        if nf != n_feat:
            raise ValueError(f"Z has {nf} columns, expected n_feat={n_feat}")
        perm, n = _finish_perm(order, K)
        Zp = Zc[perm].tocoo()
        yield Zp.row, Zp.col, Zp.data, K, n


def _factor_rows(subset, x, n_entities, n_cov):
    subset = np.asarray(subset, dtype=int).ravel()
    K = subset.size
    if K == 0:
        raise ValueError("empty subset")
    if subset.min() < 0 or subset.max() >= n_entities:
        raise ValueError("entity index out of range for n_entities")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        if x.shape != (n_cov,):
            raise ValueError(f"x must have length n_cov={n_cov}")
        X = np.broadcast_to(x, (K, n_cov))
    elif x.shape == (K, n_cov):
        X = x
    else:
        raise ValueError("x must be (n_cov,) shared by the field or "
                         "(K, n_cov) per entrant")
    return subset, X


def _factor_triplets(events, n_entities, n_cov):
    for ev in events:
        if len(ev) != 3:
            raise ValueError("factor events are (subset, order, x)")
        subset, order, x = ev
        subset, X = _factor_rows(subset, x, n_entities, n_cov)
        K = subset.size
        perm, n = _finish_perm(order, K)
        Xp = np.asarray(X)[perm]
        jj, cc = np.nonzero(Xp)
        yield jj, subset[perm][jj] * n_cov + cc, Xp[jj, cc], K, n


def factor_design(subset, x, n_entities, n_cov):
    """The (K, n_entities * n_cov) sparse design of one contest in the
    factor form: row j carries the covariate vector in entity
    subset[j]'s block of columns. Use it to mix factor ratings with
    extra shared columns (a home advantage, a colour term) via
    scipy.sparse.hstack and fit_design_ratings."""
    subset, X = _factor_rows(subset, x, n_entities, n_cov)
    K = subset.size
    jj, cc = np.nonzero(np.asarray(X))
    return sparse.csr_matrix((np.asarray(X)[jj, cc],
                              (jj, subset[jj] * n_cov + cc)),
                             shape=(K, n_entities * n_cov))


# ---------------------------------------------------------------------------
# likelihood, batched by field size
# ---------------------------------------------------------------------------

def _winner_batch(Ms, base="normal"):
    """log P(row 0 is the maximum) and its gradient for a batch of mean
    vectors under unit base noise: one lattice pass over the shared
    grid, every event at once. Same integrand as the moment updates'
    winner pass (density of the winner against the product of the
    others' CDFs), without belief convolution because MAP conditions
    on the means."""
    Ms = np.atleast_2d(np.asarray(Ms, dtype=float))
    E, K = Ms.shape
    pad = 8.0
    if base != "normal":
        span = getattr(base, "span", None)
        pad = max(pad, float(max(span))) if span is not None else max(pad, 12.0)
    lo = float(Ms.min()) - pad
    hi = float(Ms.max()) + pad
    L = int(np.clip(np.ceil((hi - lo) / (16.0 / 2001.0)), 2001, 8001))
    x = np.linspace(lo, hi, L)
    dx = x[1] - x[0]
    g = np.empty((K, E, L))
    dg = np.empty((K, E, L))
    logF = np.empty((K, E, L))
    for j in range(K):
        gj, dgj, Fj = _base_rows(x[None, :], Ms[:, j][:, None], 1.0, base)
        g[j] = gj
        dg[j] = dgj
        logF[j] = np.log(np.maximum(Fj, 1e-300))
    A = logF[1:].sum(axis=0)                     # (E, L): prod_{j != 0} F_j
    R = np.exp(np.clip(A, -745.0, 0.0))
    p = np.maximum((g[0] * R).sum(axis=1) * dx, 1e-300)
    grad = np.empty((E, K))
    grad[:, 0] = (dg[0] * R).sum(axis=1) * dx / p
    for j in range(1, K):
        rest = np.exp(np.clip(A - logF[j], -745.0, 0.0))
        grad[:, j] = -(g[0] * g[j] * rest).sum(axis=1) * dx / p
    return np.log(p), grad


def _loglik_terms(theta, st, base="normal"):
    """Per-event log-likelihood (n_events,) and d loglik / d mu per row
    (n_rows,), for the stacked design at coefficients theta."""
    mu = np.asarray(st.Z @ theta, dtype=float).ravel()
    lp = np.empty(st.n_events)
    dmu = np.zeros(st.n_rows)
    for (K, n_ord), idx in st.groups.items():
        ri = st.starts[idx][:, None] + np.arange(K)[None, :]      # (E, K)
        Ms = mu[ri]
        if base == "normal" and K == 2:
            # closed form: P(x_0 > x_1) = Phi((mu_0 - mu_1) / sqrt 2)
            t = (Ms[:, 0] - Ms[:, 1]) / np.sqrt(2.0)
            l = log_ndtr(t)
            h = np.exp(-0.5 * t * t - _LOG_SQRT_2PI - l) / np.sqrt(2.0)
            lp[idx] = l
            dmu[ri[:, 0]] = h
            dmu[ri[:, 1]] = -h
            continue
        chunk = max(1, _WORK_BUDGET // (K * 4001))
        order = np.arange(n_ord)
        for c0 in range(0, len(idx), chunk):
            sl = slice(c0, c0 + chunk)
            if n_ord >= 2:
                l, gr = _order_pass_batch(Ms[sl], np.ones(K), order, base=base)
            else:
                l, gr = _winner_batch(Ms[sl], base=base)
            lp[idx[sl]] = np.where(np.isfinite(l), l, _LP_FLOOR)
            dmu[ri[sl]] = gr
    return lp, dmu


def _as_ridge(ridge, n_feat):
    if np.isscalar(ridge):
        lam = np.full(n_feat, float(ridge))
    else:
        lam = np.asarray(ridge, dtype=float).ravel()
    if lam.shape != (n_feat,):
        raise ValueError(f"ridge must be a scalar or one value per feature "
                         f"({n_feat})")
    if np.any(lam < 0.0):
        raise ValueError("ridge penalties must be non-negative")
    return lam


def _fit(st, ridge, weights, base, n_iter, theta0):
    lam = _as_ridge(ridge, st.n_feat)
    w = (np.ones(st.n_events) if weights is None
         else np.asarray(weights, dtype=float).ravel())
    if w.shape != (st.n_events,):
        raise ValueError("weights must have one entry per event")
    if np.any(w < 0.0):
        raise ValueError("weights must be non-negative")
    row_w = w[st.row_event]

    def nlp(theta):
        lp, dmu = _loglik_terms(theta, st, base)
        total = 0.5 * float(theta @ (lam * theta)) - float(w @ lp)
        grad = lam * theta - np.asarray(st.ZT @ (dmu * row_w)).ravel()
        return total, grad

    x0 = np.zeros(st.n_feat) if theta0 is None else np.asarray(theta0, float)
    res = minimize(nlp, x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": int(n_iter)})
    lp, _ = _loglik_terms(res.x, st, base)
    info = {"objective": float(res.fun), "n_iter": int(res.nit),
            "success": bool(res.success), "message": str(res.message),
            "loglik": float(w @ lp), "ridge": lam}
    return res.x, info


# ---------------------------------------------------------------------------
# public: design form
# ---------------------------------------------------------------------------

def fit_design_ratings(events, n_feat, ridge=1.0, weights=None, base="normal",
                       n_iter=300, theta0=None, return_info=False):
    """MAP coefficients theta (n_feat,) of ability mu = Z theta from
    ranked contests.

    events: iterable of (Z, order). Z is (K, n_feat), dense or
    scipy.sparse, one row per entrant; order lists row indices best
    first -- the full finishing order, a partial order (the unlisted
    rows are marginalised exactly), or a single row for winner-only
    feedback. K = 2 under the normal base is closed form; every other
    shape runs the engine's ordered-statistics lattice pass, batched
    across events of the same size.

    ridge: scalar, or one value per feature. The vector is the point:
    a weak penalty on level columns and a strong one on offset or
    per-member columns is what makes an entity with little data fall
    back on its group instead of on zero, and a single shared penalty
    measurably reverses the sign of a factor-vs-scalar comparison
    (module docstring).

    weights: per-event multipliers on the log-likelihood. In a drifting
    sport a batch fit that weights a decade-old contest like last
    week's is an omission that flatters any richer design, since extra
    structure absorbs staleness the base model cannot express;
    exponential recency weights are the batch analogue of an online
    drift and must be tuned per arm for the same reason.

    base: the standardised noise density (winning.factor.races.BASES or
    a callable), unit scale. Returns theta, or (theta, info) with
    return_info=True."""
    st = _Stacked(_design_triplets(events, n_feat), n_feat)
    theta, info = _fit(st, ridge, weights, base, n_iter, theta0)
    return (theta, info) if return_info else theta


def design_loglik(theta, events, base="normal"):
    """Per-event log-likelihood (n_events,) of ranked contests
    (Z, order) at coefficients theta -- the validation objective, on
    the same likelihood the fit used."""
    theta = np.asarray(theta, dtype=float).ravel()
    st = _Stacked(_design_triplets(events, theta.size), theta.size)
    lp, _ = _loglik_terms(theta, st, base)
    return lp


# ---------------------------------------------------------------------------
# public: factor form
# ---------------------------------------------------------------------------

def _factor_ridge(ridge, n_entities, n_cov):
    if np.isscalar(ridge):
        return float(ridge)
    lam = np.asarray(ridge, dtype=float).ravel()
    if lam.shape == (n_cov,):
        return np.tile(lam, n_entities)
    return lam


def fit_factor_ratings(events, n_entities, n_cov, ridge=1.0, weights=None,
                       base="normal", n_iter=300, B0=None, return_info=False):
    """Ability as a vector over observed conditions: entity i's ability
    in a contest with covariate x is B[i] . x. Returns B, shape
    (n_entities, n_cov).

    events: iterable of (subset, order, x). subset lists entity indices
    in 0..n_entities-1; order lists positions into subset, best first
    (full, partial, or winner-only, as in fit_design_ratings); x is the
    contest's covariate vector (n_cov,), or (K, n_cov) when it varies
    per entrant. Convention: x[0] = 1 is the level, the remaining
    entries are conditions (flags or measurements) whose coefficients
    are offsets from it.

    ridge: scalar; or (n_cov,), one penalty per covariate applied to
    every entity -- the measured default is a weak level penalty and a
    stronger offset penalty, tuned by sweep_offset_ridge; or the full
    (n_entities * n_cov,) vector. A scalar cannot express "levels free,
    offsets shrunk" and is the confound the module docstring records.

    weights, base, n_iter: as fit_design_ratings. B0 warm-starts."""
    n_entities, n_cov = int(n_entities), int(n_cov)
    st = _Stacked(_factor_triplets(events, n_entities, n_cov),
                  n_entities * n_cov)
    theta0 = None if B0 is None else np.asarray(B0, float).reshape(-1)
    theta, info = _fit(st, _factor_ridge(ridge, n_entities, n_cov), weights,
                       base, n_iter, theta0)
    B = theta.reshape(n_entities, n_cov)
    return (B, info) if return_info else B


def factor_loglik(B, events, base="normal"):
    """Per-event log-likelihood (n_events,) of factor-form contests
    (subset, order, x) under ratings B."""
    B = np.asarray(B, dtype=float)
    n_entities, n_cov = B.shape
    st = _Stacked(_factor_triplets(events, n_entities, n_cov),
                  n_entities * n_cov)
    lp, _ = _loglik_terms(B.reshape(-1), st, base)
    return lp


def factor_abilities(B, subset, x):
    """mu (K,) = B[subset] . x for one contest; x as in fit_factor_ratings."""
    B = np.asarray(B, dtype=float)
    subset, X = _factor_rows(subset, x, *B.shape)
    return np.einsum("kc,kc->k", B[subset], np.asarray(X))


def predict_factor(B, subset, x, base="normal", points=257):
    """Win probabilities of one contest under factor ratings B: the
    race engine at the fitted abilities with unit base noise -- the
    same model the likelihood conditioned on."""
    from ..factor.races import race_probabilities
    mu = factor_abilities(B, subset, x)
    return race_probabilities(-mu, D=np.ones(len(mu)), base=base,
                              points=points)


def sweep_offset_ridge(train_events, val_events, n_entities, n_cov,
                       level_ridge=1.0, offset_grid=(1.0, 3.0, 10.0, 30.0,
                                                     100.0, 300.0),
                       level_col=0, weights=None, base="normal", n_iter=300):
    """Tune the offset penalty on a validation slice, level penalty
    fixed. Fits fit_factor_ratings once per grid value, scores the
    mean negative log-likelihood of val_events, and returns a dict:
    offset_grid, val_loss (per grid value), offset_ridge (the argmin),
    B (its ratings), at_scalar_limit (the argmin is the top of the
    grid: the offsets are being driven to zero, so the condition axis
    is unidentifiable in this data and the factor form collapses onto
    the scalar rating -- the chess-openings outcome), and at_edge
    (either end: widen the grid before trusting the number).

    The protocol this encodes: every arm's own knob tuned on the same
    validation horizon as the test, grid edges chased, and the scalar
    limit read as a finding rather than a fit."""
    grid = [float(g) for g in offset_grid]
    losses, fits = [], []
    for lam in grid:
        ridge = np.full(int(n_cov), lam)
        ridge[level_col] = float(level_ridge)
        B = fit_factor_ratings(train_events, n_entities, n_cov, ridge=ridge,
                               weights=weights, base=base, n_iter=n_iter)
        losses.append(-float(np.mean(factor_loglik(B, val_events, base))))
        fits.append(B)
    best = int(np.argmin(losses))
    return {"offset_grid": grid, "val_loss": np.asarray(losses),
            "offset_ridge": grid[best], "B": fits[best],
            "at_scalar_limit": best == len(grid) - 1,
            "at_edge": best in (0, len(grid) - 1)}


def covariate_contrast_report(events, n_entities, n_cov):
    """Exogeneity check for the factor form: does each entity's
    covariate vary across its own contests, or is the covariate a
    property of the entity?

    For each covariate column: share (n_entities, n_cov) is the mean
    covariate value over each entity's appearances (NaN when unseen);
    sd_observed is the spread of that share across entities; sd_null
    is the spread the same appearance counts would produce if every
    appearance drew its covariate independently from the pooled
    distribution (the no-choice null: Var_pop(x) / n_i per entity,
    binomial for a flag); ratio = sd_observed / sd_null. A ratio near
    one means the condition is assigned, and offsets are identified
    from within-entity contrast. A ratio far above one means entities
    choose their conditions -- the chess-openings case, 0.415 against
    0.058 -- and a null factor result there is a statement about
    identification, not about the axis. The level column (constant)
    has zero pooled variance and reports NaN."""
    n_entities, n_cov = int(n_entities), int(n_cov)
    counts = np.zeros(n_entities)
    sums = np.zeros((n_entities, n_cov))
    tot = np.zeros(n_cov)
    tot2 = np.zeros(n_cov)
    N = 0
    for ev in events:
        subset, _, x = ev
        subset, X = _factor_rows(subset, x, n_entities, n_cov)
        X = np.asarray(X)
        np.add.at(counts, subset, 1.0)
        np.add.at(sums, subset, X)
        tot += X.sum(axis=0)
        tot2 += (X * X).sum(axis=0)
        N += subset.size
    if N == 0:
        raise ValueError("no events")
    seen = counts > 0
    share = np.full((n_entities, n_cov), np.nan)
    share[seen] = sums[seen] / counts[seen, None]
    pop_mean = tot / N
    pop_var = np.maximum(tot2 / N - pop_mean ** 2, 0.0)
    sd_obs = np.std(share[seen], axis=0) if seen.sum() > 1 else np.zeros(n_cov)
    sd_null = np.sqrt(pop_var * float(np.mean(1.0 / counts[seen])))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sd_null > 0, sd_obs / sd_null, np.nan)
    return {"share": share, "counts": counts, "pop_mean": pop_mean,
            "sd_observed": sd_obs, "sd_null": sd_null, "ratio": ratio,
            "n_entities_seen": int(seen.sum())}
