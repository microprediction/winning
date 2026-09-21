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
A few hundred entities on the dense filter meant five full-matrix `eigh`
calls per contest and saturated every core; the tracker never builds that
matrix.

**Name trap:** `from winning.ratings import walk_forward` is the *dense*
`history.walk_forward`. The tracker's is only at
`winning.ratings.tracker.walk_forward`. Same return shape, plus `tracker`
and `evidence`.

## The caller owns the results → performance transform

`winning` is domain-agnostic. It never sees points, goals or seconds.

- Tracker: pass `scores` = performances already on the ability unit,
  **min-wins** (lower is better). `observe(..., scores=)` negates internally.
- Dense filter: pass `margins` (the gap to the winner in whatever unit
  the caller measures, lower is better) and `lengths_scale`, or `scores`.
  `lengths_scale` is the constant that converts those margins into
  performance units; the name is historical and carries no assumption
  about what the margin is.

Calibrate this before anything else, or prices and results land on two
different ability scales and the filter cannot be coherent. Recipe for
pairwise contests with a margin: find an unbiased forecast of the margin
(a published line, or a regression), take the residual sd σ, then

    lengths_scale = sqrt(2·beta2) / σ

A guessed value a third too small silently down-weights results against
prices and produces under-dispersed predictions that look like a modelling
problem. Check it with the regression in the diagnostics section.

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
- Always validate on data the tuning never saw. A model/market gap can
  drift by season, and in-sample gains can halve out of sample.

## Verify after any change to `winning/ratings`

    python -m winning.ratings.verify --profile fast     # ~1 min, gates CI

775 checks: martingale of the posterior mean, law of total variance,
normalisation, gauge/permutation/scale/team invariances, predict-versus-
evidence, calibration, referees. Run it in addition to targeted regression
tests for any new interface — it encodes the engine's identities, not the
contract of code you just added, and a change can pass all 775 checks while
destroying a Gaussian tail (a review of the first cut of the pairwise closed
form found exactly that — that code is not in this branch, see the resource
section, but the lesson about the verifier's reach stands).
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
   slope is under-reaction: information the update did not take.
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
wrong; `cProfile` was right. A two-runner normal race is a single Gaussian
contrast and does NOT need the lattice — but the closed form for it is not
in this branch. It was written on `pair-closed-form`, whose PR (#64) was
closed; the commits survive only as the local tag
`lost/pair-closed-form-pr64`. Until it lands, a pairwise sport does reach
the lattice, and that window search is the largest cost on a head-to-head
filter. Measure before assuming otherwise.

## Parameter shapes go through `winning.shapes`

`V` means factor loadings in every public verb and is an `(n, rank)`
matrix, one ROW per contestant. `as_loadings` is the ONE place that rule
is decided — it accepts a scalar, an `(n,)` vector, `(n, rank)` and
`(rank, n)` as the same race and raises on anything else. `as_idio` and
`as_variance` do the same for `D` and for the belief variance `v`, and
both require non-negative entries to a relative 1e-12.

Never normalise loadings with `np.atleast_2d`: it reads a length-n vector
as `(1, n)` — one contestant with n factors — and the gauge-fix then
zeroes the loadings, so the race silently degenerates to the independent
one. That was issue #66, and a static check in
`tests/test_shape_contract.py` now fails if the pattern returns.

That file is also the template for any new contract here: a discovery
test forces every public verb taking the parameter into the sweep, the
fixtures are NON-CONSTANT (a constant loading column is gauge-fixed to
zero and cannot tell a broken factor race from a correct independent
one), and each verb must be shown to MOVE the answer before any equality
is believed.

## Known gaps

- Neither online filter has a mechanism for a persistent shared offset
  such as home advantage. The package's home for that is a design column
  in `fit_design_ratings` with its own ridge (tracker.py docstring: "a
  persistent group advantage ... belongs in the means"). Do NOT express it
  as a pseudo-entity member of a line-up — that gives an offset a team's
  prior and a team's diffusion, and overloads `runners`. If the online
  filters need it, add an explicit offsets/design argument to both.
- Dense-filter `update_team_market_full` ignores `beta2` when no factor
  loadings are passed, so the price arm and results arm silently disagree
  about the noise scale the moment `beta2 ≠ 1`. Keep `beta2 = 1` on that
  path, or pass `V`.
