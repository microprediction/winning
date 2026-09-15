# winning — instructions for coding agents

Read this before touching `winning.ratings`. Most of it exists because a
session spent hours reinventing things the package already does.

## Pick the right filter FIRST

There are two ratings filters. Choosing the wrong one cost a machine reboot.

| | `AbilityTracker` (`ratings/tracker.py`) | `rate_history` / `walk_forward` (`ratings/history.py`) |
|---|---|---|
| state | one (mean, var) per entity, **no joint covariance** | dense N×N covariance over *every* entity |
| cost per contest | O(entrants) | O(N²) — plus eigendecompositions of the full N×N |
| use when | many entities, small fields (leagues, pairwise sports, ladders) | small closed fields where cross-covariances matter |
| drift | mean pegged, `v += drift·dt^drift_exp` (random walk, no level) | OU toward `prior_mean` with `timescale` — one rate for both reversion and uncertainty growth |

**Default to `AbilityTracker` for anything with hundreds of entities.**
367 NCAA teams on the dense filter was 5 full-matrix `eigh` calls per game
and saturated all cores; the tracker never builds that matrix.

**Name trap:** `from winning.ratings import walk_forward` is the *dense*
`history.walk_forward`. The tracker's is only at
`winning.ratings.tracker.walk_forward`. Same return shape, plus `tracker`
and `evidence`.

## The caller owns the results → performance transform

`winning` is domain-agnostic. It never sees points, goals or seconds.

- Tracker: pass `scores` = performances already on the ability unit,
  **min-wins** (lower is better). `observe(..., scores=)` negates internally.
- Dense filter: pass `margins` (lengths behind the winner, lower is better)
  and `lengths_scale`, or `scores`.

Calibrate this before anything else, or prices and results land on two
different ability scales and the filter cannot be coherent. Recipe for
pairwise contests with a margin: find an unbiased margin forecast
(a point spread, or a regression), take the residual sd σ, then

    lengths_scale = sqrt(2·beta2) / σ

On NCAAB the closing spread is unbiased (slope 1.001) with σ = 11.10 points,
giving 0.127. A guessed 0.08 silently down-weighted results 37% against
prices and produced under-dispersed predictions that looked like a
modelling problem. Check it with the regression in the diagnostics section.

## Prices are inverted under the outcome model — on purpose

Market prices go through `abilities_from_race` with race noise only. A
filter that has seen one price with noise `tau2` still has belief variance,
so its prediction sits closer to evens than the price it was given. That
shrinkage is correct Bayesian behaviour and vanishes as the belief sharpens.
It is **not** a bug; `tests/test_market_update.py::
test_update_race_market_leg_inverts_under_the_outcome_model` guards the
semantics. Do not "fix" it by inverting through the predictive covariance.

## Tuning: use the library's verdicts, not your eyes

Any sweep goes through `winning.ratings.tuning`:

    from winning.ratings.tuning import select, select_grid, require_live
    best, report = select({0.06: .0200, 0.13: .0125, 0.20: .0085}, "lengths_scale")
    print(report)   # -> lengths_scale=0.2  AT GRID EDGE: extend the grid

`select` reports INERT (a tie-break, not tuning), AT GRID EDGE (extend it)
and SATURATED (benign edge). `require_live` raises on an inert parameter.
`tune_block_rho` shows the pattern. A value chosen at a grid edge has not
been tuned; say so.

- **Do not tune a drift/`timescale` on a window with no season gap.** The
  parameter is unidentifiable there and Nelder-Mead will run it to
  decades.
- Evidence (`tune_history`, `tracker.evidence`) and held-out log-loss can
  disagree. Evidence on one season preferred "trust the price, ignore the
  results"; predictive log-loss wanted the opposite. Report which objective
  you used.
- Always validate on data the tuning never saw. The model/market gap on
  NCAAB drifted wider by season, so in-sample gains halved out of sample.

## Verify after any change to `winning/ratings`

    python -m winning.ratings.verify --profile fast     # ~1 min, gates CI

775 checks: martingale of the posterior mean, law of total variance,
normalisation, gauge/permutation/scale/team invariances, predict-versus-
evidence, calibration, referees. Run this instead of writing ad-hoc
invariant tests — it already encodes the identities you would be checking.
`--profile full` is ~30 min; adjudicated runs are recorded in
`research/adjudications/ratings_verify.md`. Marks live in
`verify/marks.py` only and are set *before* the fix they judge.

## Diagnostics that actually locate a problem

When a filter underperforms a market, run these before theorising:

1. **Scale check.** Regress actual margin on the filter's pre-game expected
   margin (mean contrast ÷ `lengths_scale`). Slope ≈ 1 means the ability
   unit is right. Slope 3.4 meant `lengths_scale` was 3.5× off.
2. **Martingale check.** For each entity, regress the next contest's
   surprise (actual − expected) on the previous one. A significant positive
   slope is under-reaction: information the update did not take. On NCAAB
   the slope was +0.056 (t=7.7) — real, small, and worth ~1% CLV against a
   4.2% overround.
3. **Dispersion.** `sd(Φ⁻¹(p_model)) / sd(Φ⁻¹(p_market))`. Below 1 the
   ratings are compressed. Do not cure this by inflating an observation
   precision; find the mis-scaling.
4. **Calibration by bucket**, and **by season phase** if there is a
   season structure — an early-season gap 10× the late-season gap is a
   drift problem, not a modelling one.

## Resource discipline

Numerics here are BLAS-heavy and `eigh` fans out over every core. Before
running anything on a history:

- set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
  `VECLIB_MAXIMUM_THREADS` **in-process before importing numpy** (the repo
  allowlists `OMP_NUM_THREADS=3 python …`);
- one job at a time, foreground; bound the problem size by env var and
  measure ms/contest on a subset before launching a full history;
- macOS has no `timeout`, and Darwin refuses `setrlimit(RLIMIT_AS)` — the
  bounded N *is* the cage;
- never pipe a long run through `tail`; it buffers until exit and you
  lose all progress output.

Profile before optimising. Two confident diagnoses of a hot spot were both
wrong; `cProfile` was right. Two-runner normal races have a closed form
(`races.py`), so a pairwise sport should never reach the lattice.

## Known gaps

- `AbilityTracker` has no line-up / weighted-membership support: `runners`
  are flat ids. A shared effect such as home advantage (a `__HOME__`
  pseudo-entity in the host's row) is only expressible on the dense filter.
  Adding an assignment matrix to `observe`/`predict` is the natural
  extension; `teams.py` already speaks that language.
- Dense-filter `update_team_market_full` ignores `beta2` when no factor
  loadings are passed, so the price arm and results arm silently disagree
  about the noise scale the moment `beta2 ≠ 1`. Keep `beta2 = 1` on that
  path, or pass `V`.
