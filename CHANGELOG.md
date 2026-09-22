# Changelog

## Unreleased

- Pinned: `abilities_from_race` does not converge when TWO runners hold
  nearly all the mass, at any field size. With two dominant runners and
  the rest at 1e-4 the residual is 7.5e-3 at n=10 and 1.1e-2 at n=30;
  with three or more substantive runners the same negligible tail
  converges in 15-28 iterations. More iterations make it worse
  (oscillation), the boundary is smooth (12 -> 41 iterations as the rest
  shrink from 0.20 to 0.02 each), and it is scale-independent. n=2 has a
  closed form; an EFFECTIVE pair goes to the lattice, where the
  coordinate-Newton moves one favourite's probability nearly one-for-one
  with the other's and oscillates. Under the skew-normal base it
  returned the two favourites swapped and only warned. This is a
  late-contest field with two contenders left, not a toy.
  `tests/test_inverse_effective_pair.py` carries the defect as strict
  xfails (n=3, 10, 30) and pins the mechanism. Likely fix: couple the
  dominant pair's update (2x2 Newton block) or damp when the effective
  field is two. No code change here.

- `winning.thurstone` is gone. It was a deprecation alias for
  `winning.research`, itself the retired research engine, so an import
  of `thurstone` crossed two deprecation layers to reach code that is
  ~38x slower than `calibrate_abilities` on the inverse problem
  (1.01 s vs 0.026 s at n=40) and round-trips to 3.8e-5 against
  1.3e-9 -- the allocation session's measurement on its inputs, not
  reproduced in this repo -- and has no Rust dispatch. The external `thurstone` shim package
  imports this alias and will now fail; that is the point. Nothing
  inside `winning` imported it. `winning.research.AbilityCalibrator`
  stays but now warns on construction, naming the front door and the
  measured gap, so nobody lands on it by accident.
- Pinned: `race_probabilities(cov=C)` does not reproduce an exactly
  structured C at small n. An exact r-factor correlation has an exact
  V/D, yet `cov=` is 5.3e-3 / 6.6e-3 / 8.4e-3 off the V/D answer for
  r = 1 / 3 / 5 at n = 8, invariant to `points`. The mechanism is not a
  one-factor collapse: `fit_covariance` stacks k=3 factors + block
  loadings + m=5 eigendirections + floored D, validated at large n; at
  n = 8 the stages sum to n and it degenerates into a FULL-rank fit with
  D on its 1e-6 floor. The residual is then ~1e-6, so no warning fires
  -- they judge fit quality -- but with D ~ 0 the factor-node quadrature
  cannot resolve the near-step conditional race. `tests/
  test_cov_exact_structure.py` carries the defect as a strict xfail and
  pins the mechanism (full rank, floor bound, residual small) so the fix
  goes to the fit's rank selection, not to the race. Reported by the
  allocation session against a 3M-path simulation; no code change here.
- The compiled kernels now reach the ratings hot path. `rust_active()`
  returned True while `update_winner_full` ran pure numpy for 97% of its
  time: the two `factor.core` kernels that ARE that 97%
  (`win_probabilities_factor`, 24-31%; `jacobian_vector_product`,
  68-72%) had no dispatch, and `core` was not in
  `rustconfig._rust_modules()`, the hand-kept list the one switch
  toggles. Both kernels now dispatch to fastrace, which matches numpy
  to ~5e-17 on both JVP forms. Measured, same inputs, switch off vs on:
  `update_winner_full` 34.1 -> 14.4 ms at K=8 (2.4x) and 197.8 -> 72.2
  ms at K=20 (2.7x), agreement 1e-15.

  Dispatch requires two or more factor nodes. At a single node the
  compiled JVP is SLOWER (0.59 vs 0.42 ms at K=8, a ~0.5 ms fixed cost)
  and the forward is a wash once the argument copies are counted;
  `nway.update_winner_correlated` calls both 119 times with one node
  each and ran 25% slower with rust on before the guard. With it, that
  path is bit-for-bit unchanged (1.00x, max diff 0). Rust wins from two
  nodes (1.2x) to fifty (3.2x). Deletions, per-node windows and the
  unnormalized JVP have no compiled form and fall through to numpy.

  `methods.native` and `alternatives.reprs` imported fastrace directly
  and tested `is not None`, so `use_rust(False)` and `WINNING_PURE` never
  reached them; both now carry the standard flag and are registered.
  `tests/test_rust_dispatch_core.py` pins rust == numpy for both core
  kernels, that one node stays on numpy, that the switch reaches every
  registered module, and -- the guard that would have caught all three
  -- that every module importing fastrace is in `_rust_modules()`.
  The structural speedup for correlated updates, one batched kernel call
  instead of 119 per-node calls, is nway's algorithm and is not done here.

- `race_probabilities` now takes the closed form for a two-runner normal
  race instead of building the lattice. A pair is a single Gaussian
  contrast, so with `Sigma = V V' + diag(D)` the answer is
  `p0 = Phi((mu1 - mu0)/sd)`, `sd^2 = S00 + S11 - 2 S01`. The identity was
  always exact -- the engine agreed with it to 2.6e-12 -- but nothing
  dispatched to it, so a pair paid 7.0 ms against 1.1 us for the `ndtr`
  call, about 6,000x, and most of that was the grid-window bisection
  running in Python before any compiled kernel. The inverse was already
  asymmetric with this: `core.py` shortcuts `N == 2`.

  Both tails are computed directly, `ndtr(u)` and `ndtr(-u)`, rather than
  as `1 - ndtr(u)`: the subtraction rounds the loser to exactly zero at a
  9 sd contrast where the true value is 1.1e-19, and the pair then depends
  on which runner is listed first. That defect was found by review of the
  first cut while all 775 verifier checks passed, and it is what the
  extreme-tail test pins.

  Measured over 2000 random pairs at rank 1-4: max deviation from the
  analytic value 2.2e-16, permutation equivariance exact, normalisation
  exact, own-slopes against finite differences 1.3e-11. Guarded to
  `n == 2` and `base == "normal"`, after the structure and temperature
  dispatches, so every other race is unchanged.
- The loading-shape contract is now ONE function, `winning.shapes.as_loadings`,
  and every public verb in the package goes through it. It was previously
  re-implemented at twenty-odd call sites in three incompatible ways:
  `np.atleast_2d` alone (silently misread a length-n vector as one
  contestant with n factors -- #66), `np.atleast_2d` plus a hand-rolled
  transpose (seven sites in `winning.ratings`, correct but unvalidated),
  and an explicit row-count check (`winning.probit`, which REJECTED the
  spelling the rest accepted). Measured before the change, on a field of
  5 with non-constant rank-one loadings: `winning.factor` panicked out of
  ndarray or silently priced the independent race; `fastmvn`,
  `likelihood` (2), `methods` (12) and `probit` raised useless errors
  ("not enough values to unpack", "tuple index out of range", "setting an
  array element with a sequence") several frames below the call. All of
  them now accept a scalar, an `(n,)` vector, an `(n, rank)` matrix and a
  `(rank, n)` matrix as the same race, and raise one `ValueError` naming
  `(n, rank)` for anything else. `as_idio` does the same for `D`.
  `winning.methods.registry.register` enforces it for all twelve
  integration methods at once, since they share one signature.
- `as_loadings` returns a C-contiguous array for every spelling. The bare
  transpose view handed BLAS an F-ordered matrix, and the `(rank, n)`
  spelling then answered the `(n, rank)` one to 1 ULP rather than exactly.
- `winning.factor.core.win_probabilities_factor` and
  `winning.factor.topk`'s five correlated entry points did not normalise
  loadings at all; `topk` additionally refused a `(rank, n)` matrix as
  "factor rank > 2". Found by the sweep below, not by a user.
- `tests/test_shape_contract.py`: the harness that would have caught #66.
  It DISCOVERS every public function taking a `V` and fails if one is
  neither driven nor exempted with a stated reason, so a new entry point
  cannot join the package without joining the sweep. For each of the 46
  it drives, it checks that every spelling of the same loadings gives the
  same answer, that a mis-shape raises the one contract error, and -- the
  guard the old suite lacked -- that the loadings move the answer at all
  before any equality is believed. A static check keeps `np.atleast_2d`
  off loadings package-wide. With the #66 bug reinstated, 231 of its 288
  checks fail; the suite as it stood passed 462 of 462 with that bug
  present.
- `python/fastmvn` vendors `winning/fastmvn.py` byte-for-byte, and that
  file now says `from .shapes import as_idio, as_loadings` -- which
  resolves to `winning.shapes` in one tree and `fastmvn.shapes` in the
  other, so the vendored copy stays identical. `winning/shapes.py` is
  vendored alongside it and a second sync test pins the two against each
  other, so the shape contract cannot drift between the packages.
- Belief variances are validated, which closes a second brittleness in
  the same signatures. `v` (belief variance) and `V` (loadings) differ
  by case alone in five public verbs, and at rank one they have the SAME
  shape, so exchanging them was silent: measured at 1.09 on the
  posterior mean in `update_winner_correlated`, 0.71 in
  `update_order_correlated`, 0.053 in `predictive_win_probabilities`,
  and a `sqrt` of a negative -- NaN, with a RuntimeWarning and nothing
  else -- in `simulate.correlated_draws`. Exchanging `m` and `v` was
  silent too, and nothing in the package rejected a negative variance
  anywhere. `winning.shapes.as_variance` now requires finite entries of
  the right length that are non-negative to a relative 1e-12; every
  public verb taking a belief variance uses it, and `as_idio` shares the
  check. The discriminator is free: loadings are gauge-fixed to mean
  zero, so a non-trivial loading vector always carries a negative entry
  and a variance never can. All eleven measured swap and
  negative-variance cases now raise a `ValueError` naming the likely
  cause.
- That check is a TOLERANCE, not `>= 0`. A variance that is negative
  only by round-off IS zero, and a strict test would have turned a
  1e-18 eigenvalue residue out of `fit_covariance` into an exception in
  a caller that was doing nothing wrong. Entries within 1e-12 of zero
  (relative to the largest entry) are clipped to zero; anything below
  that raises. Loading entries are O(0.1-1), so the swap is still
  caught with twelve orders of margin.
- The sweep skips a verb whose optional backend is absent, visibly and by
  name, rather than failing or quietly vanishing: CI installs only
  `.[test]` (pytest, pandas, matplotlib), so `cdf_gradient_shares` has no
  jax there. `driver(..., requires="jax")` turns that into five named
  SKIPs whose reason says the contract is unverified in that environment,
  not waived.
- The sweep's discovery guard now decides "optional backend absent" from
  the CAUSE, not a list of module names. The first version allowlisted two
  paths that did not exist, so `winning.bench.season_ranked` needing
  `trueskill` failed CI on five platforms while passing locally, where
  trueskill happens to be installed. `winning` itself needs only numpy and
  scipy, so a `No module named X` where X is neither a hard dependency nor
  a `winning.*` module is an optional backend: recorded, warned about by
  name, and not fatal. Any other import failure still fails, because a
  module that stops importing takes its verbs out of the sweep silently.

- Factor loadings given as a bare length-n vector were read by
  `np.atleast_2d` as `(1, n)` -- one contestant carrying n factors --
  rather than as n rank-one loadings. Nothing downstream caught the
  difference: the gauge-fix subtracted the single row from itself, so
  the loadings became zero and the rank was read as n. With `fastrace`
  the compiled kernel then indexed n rows that were not there and
  panicked out of ndarray; a panic is not an `Exception`, so a sweep
  wrapped in `try/except Exception` aborted instead of skipping the
  race. Without `fastrace` there was no symptom at all: the call
  returned the INDEPENDENT race, to 4e-15, discarding the correlation
  it was given. `winning.factor.core.as_loadings` now normalises
  loadings to `(n, rank)` at the Python boundary -- a scalar is a
  common column, a length-n vector is rank one, an `(rank, n)` matrix
  is transposed, and anything else raises a `ValueError` naming the
  expected shape -- and `as_idio` does the same for `D`. Both run in
  `_setup`, so `race_probabilities`, `abilities_from_race`,
  `ordered_probabilities`, `polish_race`, the tempered paths,
  `jacobian_vector_product` and
  `abilities_from_probabilities_factor` inherit one contract and the
  compiled side is no longer reachable with a mis-shaped array.
  Reported in #66.
- The tests could not have found the above. Every one of them spells
  loadings `(n, 1)` or `(n, r)`, so no equivalent spelling was ever
  compared against another; and the shapes that do appear are usually
  CONSTANT columns, which the gauge-fix sends to zero, so a factor race
  that had silently become the independent race looked right.
  `tests/test_loading_shapes.py` pins the representation invariance
  with non-constant loadings, on both the compiled and the numpy path,
  and asserts up front that the loadings move the race at all.
- `ordered_probabilities` took neither `structure=` nor `cov=` while
  `race_probabilities` and `abilities_from_race` took both, so a
  covariance described declaratively for the inverse had to be
  re-expressed as `V=`/`D=` for the forward ordered-prefix call. It now
  accepts both, sharing the forward call's fit-and-warn path.
  `structure=Blocks/Nested/Tree` raises `NotImplementedError` naming the
  grammar: those are win-race recursions with no ordered-prefix pass,
  and pricing them through a factor fit would answer a different
  question than the argument asks. Also from #66.
- `structures.Factor` now documents that its keyword names are `V` and
  `D`; `Factor(loadings=..., idio=...)` is the natural guess and raises
  `TypeError`. Also from #66.

- The verifier's documented-approximation envelope for
  `update_ranking` was passing on a seed count. At the shipped 25 seeds
  the failure base reads robust sd 1.55 and coverage 0.799, inside both
  bounds; at 100 seeds it reads 1.6058 and 0.7596, outside both, so a
  better-powered run of unchanged code would have failed. The envelope
  branch also returned before P6's power test, and that test measures
  against the 0.35 threshold rather than the envelope edge seven times
  closer, so the cell could report the cost as documented with no power
  to say otherwise. And the envelope's basis table lists five bases, none
  of which reaches it; the only cell that does is the failure base, which
  has no row there. The failure base now has its own envelope set four
  standard errors clear of the 100-seed measurement, the power test runs
  against the bound in force, an envelope verdict that cannot be resolved
  reports UNDERPOWERED, and every envelope verdict carries its margin.
- The verifier's P8 referee gates on `dm > mark + 4 se`, so what it
  enforces is the mark plus the referee's own Monte Carlo noise, and
  that allowance is budget-dependent while the report printed the
  nominal mark at every profile. Measured across six bases, the enforced
  threshold was 0.0182 at the fast draw count, 0.0116 at full and 0.0071
  at exhaustive, against a documented 0.005: the mark was the smaller
  term of its own threshold everywhere, and a reader of `marks.py` was
  wrong by 3.6x at the profile that gates CI. The report also paired the
  statistic `max(dm, dv)` with `dm`'s tolerance. No mark moves: `dm` and
  `dv` are now separate results, each statistic against its own bound;
  the reported tolerance is the enforced one, with the mark and the
  noise allowance in the detail line; and the full profile draws
  1,000,000 rather than 400,000, where the allowance falls to 0.0044 and
  the mark binds. Adjudication in
  `research/adjudications/ratings_verify.md`.
- `race_probabilities(..., temperature=tau)` convolved each base with
  the tau-scaled min-Gumbel kernel on the base's own asymmetric grid and
  took numpy's central slice, which aligns a kernel's middle sample with
  zero lag. The grid runs from -12 sd - 30 tau to 12 sd + 8 tau, so its
  midpoint is -11 tau and the convolved base came out shifted by that
  much: its mean read 8.16 where the exact value is -0.577. A common
  shift cancels in a race, which is why win probabilities stayed nearly
  right, but the shifted density is then truncated against a grid that
  no longer reaches it, and that does not cancel. The kernel now has its
  own zero-centred grid and the full convolution is sliced at the lag
  that matches. Against a 600,000-draw reference the worst error on the
  reported grid falls from 6.7e-2 to 5.9e-5, and every cell at base sd
  0.5 or below improves by between 26x and 1146x; unit scale was already
  at Monte Carlo noise and is unchanged. A base too narrow to resolve on
  the tempered grid now raises instead of returning NaN, naming
  `softmax_probabilities` as the closed form for that limit. Reported by
  the bandits session.
- `winning.probit` chose its own quadrature nodes and evaluated through
  `winning.factor.core` directly, so its three entry points missed both
  of the race layer's escalations and its window. A fixed Gauss-Hermite
  rule has no sharpness branch: at K = 8 with loadings 1.5 at rank 2
  `shares` carried total variation 1.2e-3 where the adaptive rule
  carries 2.8e-6. It has no node budget either, so the pruned tensor
  reached 24 s per forward pass at rank 6 against 55 ms capped. And the
  core lattice spans the abilities rather than the winner bulk, which
  cost 5x at rank 3 for the same answer. `shares`,
  `utilities_from_shares` and `removal_shares` now route through
  `race_probabilities` and `abilities_from_race`. Two tests asserted
  bit-exact agreement with a hand-built node set, which pinned the node
  rule rather than the reflection claim; they compare against the race
  entry point at two ulp, and the escalation and the cap have
  regression tests.
- `winning.ratings.tuning`: `select` returns a swept parameter's argmin
  together with whether the sweep did anything, separating INERT (the
  grid does not move the metric, so the choice is a tie-break and must
  not be reported as tuned), AT GRID EDGE, and edge-but-SATURATED (the
  curve has converged, benign). `select_grid` profiles each axis of a
  product grid separately and `require_live` raises where a fairness
  claim depends on the tuning being real. `sweep_offset_ridge` and
  `tune_block_rho` now report `inert` from it.
- The chess measurements are corrected. Glicko-2's `tau` was swept and
  cited as evidence of symmetric tuning, but it is inert over one month
  of games, and the parameters that do move its loss were never swept.
  With both sides tuned the factor form still beats Glicko-2 per time
  control on all three splits (0.008 to 0.012 nats, intervals excluding
  zero), the estimator is worth -0.0080 rather than -0.0199, the factor
  structure is unchanged at -0.0033, and stratification's value to
  Glicko-2 flips from +0.0074 to -0.0011, so it helps neither side. The
  module docstring, README and ratings page are corrected;
  `papers/chess_ratings` is marked under correction pending a rewrite.
- `julia/MvNormalCDFFast` is now `julia/FactorMvNormalCDF` (new UUID),
  with MvNormalCDF.jl as a hard dependency: every case outside the
  exact factor path (rank > 2, sharpness > 3, or a covariance that is
  not factor-plus-diagonal) is delegated to `MvNormalCDF.mvnormcdf`
  with `m` and `rng` forwarded, so every call is answered. The package
  exports its own names only; `mvnormcdf_factor` carries MvNormalCDF's
  signature. The name follows the General registry review of the first
  submission (#167315) and the arrangement agreed with the MvNormalCDF
  maintainer (MvNormalCDF.jl#20): companion package, own exports,
  delegation on refusal.
- `julia/winning` gains a test suite: `Pkg.test` runs the parity
  scenarios against `parity/vectors.json` plus intrinsic invariants,
  through `test/parity.jl`, which `parity/check.jl` now also calls. The
  Julia CI matrix had failed on `winning` since 2026-09-08 for want of a
  `test/runtests.jl`. A TagBot workflow with one `subdir` step per Julia
  package tags registered versions as `<Package>-v<version>`; the
  README lists the two packages now in General and the third pending.
- Block correlation on Formula 1, resolved (bandits exp34b-h): the term
  is sound (correctly specified on synthetic data it beats independence
  on ability recovery, winner and joint log-loss), the teammate
  correlation is real at matched noise (+29.2 nats), and the held-out
  loss comes from a rho that moves between seasons (same-team 1-2 rate
  0.632 in 2015, 0.045 in 2021), with the damage on the fit channel.
  Docs rewritten accordingly. New caution wherever rho is documented:
  under an order likelihood rho and beta2 trade off, because a shared
  component changes the ratio of within- to between-group difference
  scale whichever way it is normalised (an equal-loading global factor
  cancels entirely: independence at idiosyncratic 0.6 and a global
  factor at 0.6 return the same evidence to 4e-15), so a rho selected
  at fixed beta2 is a blended estimate. `tune_block_rho` gains
  `beta2_grid` for a joint sweep, returning the evidence matrix and the
  argmax pair; the single-axis default is unchanged.
- `winning.ratings.orders`: `order_from_positions`, `order_from_performance`,
  `order_from_times` and `positions_from_order` name the conversion into
  the best-first `order` every ranked update takes. A rank array
  (position of entrant i) is the inverse permutation, passes every
  runtime check, coincides with the order at K = 2 and is silently wrong
  at K >= 3, where it produces a plausible walk that learns much less
  (reported by the bandits session, hit twice). The ranked-update
  docstrings now say so and point at the module's self-test.
- Block-correlation docs (tracker docstring, README, ratings page) now
  record the measured Formula 1 verdict on the current engine in place
  of the provisional numbers: fitted and priced under one model, the
  blocked arm's held-out winner log-loss is worse by +0.0427 [+0.0089,
  +0.0769] against a 0.005 bound and its joint advantage -0.0224 spans
  zero (bandits exp34 rerun, main 62f80c0). The engine repairs moved the
  gap by about a fifth and left the independent arm bit-identical. The
  feature ships as a modelling option, not a measured improvement.
- `return_se` docstrings caution that the Laplace standard errors are
  for inference, not for tempering predictions: pricing at
  1 + Var(mu) measured worse on Lichess (+0.0015 [+0.0003, +0.0026],
  bandits exp39).
- The mixture engines behind the correlated and full-covariance updates
  recentre and rescale their scrambled-Sobol node cloud (rank >= 3) on
  the factor posterior after one pass at the prior nodes, with the
  importance correction in the weights. Under a diffuse dense belief the
  prior-centred cloud carried 13-35 percent effective weight and the
  order update's variances were 13 percent off at prior sd 10 (verifier
  baseline); recentred they are within a percent at the same node count,
  for about a tenth more work per update.
- The diagonal moment updates (`update_winner`, `update_ranking_exact`,
  the correlated mixture) now difference their curvature at steps
  relative to each coordinate's predictive sd, as the full-covariance
  path already did, so posterior variances are covariant under a
  common rescaling of means, prior sds and noise sds (verifier:
  2.8e-12, was 1.6e-4 at a tenth of unit scale). At unit scale this is
  the historical step; results move at the fourth decimal.
- Verifier marks set from the baseline run for the factor fitter's
  calibration and recovery checks.
- `Qf` was silently ignored by the correlated updates at factor rank
  >= 3 (`_factor_nodes` hard-coded 1024 Sobol nodes there), so it did
  nothing in exactly the regime a user would reach for it (found by the
  bandits session: Qf 2..7 bit-identical at r = 10). The node count at
  rank >= 3 is now `nodes_log2` (default 10, the historical count) on
  `update_winner_correlated`, `update_order_correlated`, `update_race`,
  `ranking_loglik_and_score`, `predictive_win_probabilities` and
  `AbilityTracker`; docstrings say which parameter applies at which
  rank, and the tracker's records the measured cost ceiling of block
  correlation at double-digit group counts (15-29 s per update for a
  20-entrant field with ten pairs).
- Factor ratings (`winning.ratings.factor_ratings`): an entity's
  ability as a vector over observed conditions, MAP through the exact
  ranking likelihood. `fit_factor_ratings(events, n_entities, n_cov,
  ridge=...)` takes `(subset, order, x)` events (full, partial or
  winner-only orders; contest-level or per-entrant covariates) and a
  ridge that is scalar, per-covariate or per-coefficient -- the
  per-covariate form ("levels free, offsets shrunk") is the measured
  mechanism, and a shared penalty is recorded as a confound.
  `fit_design_ratings(events, n_feat, ...)` is the general form over
  `(Z, order)` design rows, dense or sparse, with per-event recency
  `weights=` and `base=`. Likelihoods run batched by field size on the
  engine's lattice pass (closed form at K = 2 under the normal base).
  Also `factor_loglik` / `design_loglik` for validation scoring,
  `predict_factor`, `factor_design` for mixing entity blocks with
  shared columns, `sweep_offset_ridge` (tunes the offset penalty on a
  validation slice and flags the scalar limit) and
  `covariate_contrast_report` (the exogeneity check: observed spread
  of per-entity covariate shares against the no-choice null).
- Block correlation in `AbilityTracker`: `rho=` and per-contest
  `groups` give same-group entrants a shared performance component,
  variance-preserving (`block_loadings`), on every observer -- prices
  inverted under the blocked model, scores through the full-covariance
  conjugate node, orders and winners through the correlated moment
  updates -- and `predict(..., groups=)` prices the same model. The
  tracker now accumulates `evidence`, the sum of log P(observation)
  over everything it consumed (including the independent order path,
  which did not return it before); `tune_block_rho` selects `rho` by
  that marginal likelihood. `walk_forward` passes `groups` through and
  returns the evidence.
- `order_loglik`'s docstring said min-wins; the function is max-wins
  like the rest of the module, and now says so.
- `winning.ratings.verify`: the ratings layer's own verification suite,
  deterministic under a root seed, with smoke / fast / full / exhaustive
  profiles (`python -m winning.ratings.verify --profile full --workers 4
  --out reports/`). Exact identities every Bayesian moment update must
  satisfy (the martingale of the posterior mean, the law of total
  variance with and without the curvature clamp, normalisation),
  enumerated over every outcome and weighted by the engine's own
  probabilities; reductions between code paths, closed forms, coarsening
  of partial orders, gauge / permutation / scale / team invariances,
  predict-versus-evidence; simulation-based calibration of the filters
  on pairwise contrasts; Monte Carlo referees on fixed and seed-drawn
  configurations; and the bandits property audit (P1-P8) ported with
  fixed seeds (its legacy world reproduces the historical digits). Every
  tolerance lives in `marks.py` with provenance; verdicts distinguish
  UNDERPOWERED and documented EXPECTED_APPROX from ok; the fast profile
  gates CI and the full profile runs nightly (`ratings-verify.yml`).
  Adjudicated runs are recorded in `research/adjudications/ratings_verify.md`.
- `winning.ratings.simulate`: one home for synthetic worlds and base
  noise; every base factory carries an exact `.sample`.
- Laplace standard errors for the MAP factor ratings:
  `fit_factor_ratings(..., return_se=True)`, `fit_design_ratings(...,
  return_se=True)`, `factor_se`, `design_se`, from the Hessian of the
  penalised objective (analytic for two-player normal, central
  differences of the analytic gradient elsewhere); calibrated on the
  ridge prior (verifier `factor.se_calibration`).
- `python -m winning` (and the `winning` console script setup.py has
  declared since the start) now exists: it prints the installed version
  and runs `winning.contract.verify()`, exit status 1 on failure. The
  install smoke test exercises it as a subprocess.

- Fixed `winning.probit.fit_factor_model`, and so `shares(..., Sigma=,
  k=)` and `utilities_from_shares(..., Sigma=, k=)`: the contrast
  heuristic applied principal-factor analysis to P Sigma P although the
  idiosyncratic part in contrast space, P diag(D) P, is not diagonal,
  so an independent race (Sigma = I, k = 1) came back with an invented
  factor and shares off by 0.03-0.05. The wrapper now uses the
  projected fit (`factor_model_projected`), which reproduces the
  contrast covariance exactly; the issue's example agrees with
  independent quadrature to 3e-10. Found independently while
  evaluating Bernd Johannes Wuebben's gaussian-correlated-choice (#27).
- Ordered finishing prefixes: `ordered_probabilities(mu, k, ...)` prices
  every ordered k-prefix (exacta/trifecta-style permutations) from one
  shared field pass, with a Rust kernel (`ordered_prefixes`);
  `plackett_luce_prefix_logprob` and `plackett_luce_order_logprob` give
  the stagewise Plackett--Luce likelihood. Lives in
  `winning.factor.permutations`.
- Terminology, with deprecated aliases so nothing breaks: the
  ordered-prefix module is `winning.factor.permutations` (was
  `exotics`); `harville_*` are now `plackett_luce_*` (Harville 1973 is
  the Plackett--Luce model for ordered finishing); and
  `plackett_luce_topk_probabilities` replaces `place_probabilities`
  (win/place/show is US racing jargon that does not travel). The old
  names keep working with a deprecation intent.
- Rust parity and tooling: `test_rust_parity.py` now verifies the whole
  39-scenario parity contract instead of ~7 hand-written cases (top-k,
  loc_scale, win/second and the inversions were never checked against
  Rust); all match numpy to tolerance. `winning.use_rust()` now toggles
  all six compiled modules (it had omitted `topk` and `permutations`).
  `fastrace` bumped to 0.2.0 for the top-k and ordered-prefix kernels.

- Top-k calibration: `abilities_from_topk` inverts a top-k
  membership curve (any depth, logit residuals against the saturated
  favorites, factor rank <= 2); exact-rank targets are refused as
  standalone inputs (two-branched) with `abilities_from_rank_marginal`
  as the eyes-open branch-picking alternative.
- Two-parameter calibration: `loc_scale_from_topk_pair` and the
  market-facing `loc_scale_from_win_and_second` jointly identify per
  runner (location, scale) from two membership curves on the exact
  stacked (dq/dmu, dq/dsigma) Jacobians — square on the double gauge
  quotient (translation, joint rescaling), optional log-sigma ridge
  (`ridge=` multiplies the squared penalty) and warm start (`mu0=`).
- JS and R ports of the full top-k module (forward, both Jacobians,
  rank marginals, all three inversions), parity-locked: 15 new embedded
  scenarios match the python reference at ~1e-15 in both languages.
- `winning.factor.topk` now honors `WINNING_PURE` like races/blocks.
- fastrace kernels for the inversion hot passes: `top_k_slopes`,
  `top_k_jacobians` and `top_k_window` (the python bisection was ~2.6 ms
  of a ~5 ms small-field forward), with the rayon fan-out gated on a
  work estimate so small fields run serial. Measured at n = 9:
  loc_scale_from_win_and_second 48 -> 4.5 ms/race, abilities_from_topk
  13 -> 2.4 ms/race; kernels match numpy at 1e-12 on the shared window.
- loc/scale warm start runs at loose tolerance (the LM loop refines);
  mirrored in the JS and R ports so parity trajectories agree.
- `julia/GMRFExtremes`: the order-statistic layer for Gauss-Markov
  chains and tridiagonal-precision GMRFs -- argmax marginals, max-CDF,
  expected max, first passage, excursion probabilities -- by
  restricted transfer-operator passes (exact where Bolin-Lindgren's R
  `excursions` simulates). Fractional-occupancy boundary cells for
  O(dx^2); refusal contract on wider bandwidth and general sparsity;
  dispatch on GaussianMarkovRandomFields.jl types via extension.
  MC-refereed at 400k paths; reproduces the tridiagonal track's
  boundary-pile-up and discrete-arcsine laws.
- `julia/MvNormalCDFFast`: deterministic MVN rectangle probabilities
  for factor-structured covariance, the MvNormalCDF.jl companion (port
  of `winning.fastmvn`, fixture-pinned). Exact GH at rank <= 2 and
  sharpness <= 3 with a deterministic Laplace-recentered tail path
  (validated to ~1e-31); everything past that is REFUSED and routed to
  the genuine incumbent via a package extension rather than shipped on
  degraded nodes. Agreement with a 200k-point MvNormalCDF run: 8e-8.
- `julia/MultinomialProbit` gained inference and a machine-precision
  score referee: `vcov`/`stderror` in three flavors (observed
  information via central differences of the exact analytic score,
  OPG, sandwich from per-observation scores), a ForwardDiff package
  extension upgrading the Hessian to dual-mode exactness, and -- since
  the likelihood path is now type-generic -- the whole likelihood is
  AD-able by consumers. The analytic score is refereed by ForwardDiff
  duals at 1e-10 on both node branches, superseding step-tuned
  differences.
- `julia/MultinomialProbit`: the first multinomial probit for Julia
  (the ecosystem has logit families, binary and ordered probit, and no
  MNP at all). Two engines, one interface: the exact
  factor-conditional likelihood with analytic score (port of
  `winning.likelihood`/`winning.mnprobit`, pinned to python fixtures at
  1e-9) and common-random-numbers GHK (port of the rust core's
  simulator) for reference and head-to-heads. Dependency-free;
  self-contained BFGS; Halton escalation past sharpness 3 as in
  `r/mlogitfast`.
- Julia joins the parity lock: `julia/winning` (dependency-free but for
  stdlib LinearAlgebra) ports the factor races (forward, slopes,
  inversion, adaptive quadrature) and the complete top-k module; all 24
  in-scope scenarios of `parity/check.jl` match the python reference at
  machine precision. Blocks/nested/tree, polish and the classic lattice
  remain the Julia roadmap.
- fastrace rank-marginal kernels (`rank_marginals`,
  `rank_marginal_jacobian`); `abilities_from_rank_marginal` drops from
  14.5 to 2.7 ms/race at n = 9. The pinning test caught an r = n defect
  in every language: the Jacobian's P(N = n-1) term is identically zero
  but was computed as window-sensitive deconvolution junk; all four
  implementations now use the identity.
- `top_k_jacobians` accepts factor correlation (`V=`, rank <= 2) as an
  exact Gauss-Hermite mixture of independent Jacobians, in python and
  both ports; `loc_scale_from_topk_pair` REFUSES fixed loadings with
  the dimension count (no rescaling gauge means two curves carry
  2n - 2 numbers against 2n - 1 unknowns — a third depth would close
  it as overdetermined least squares).
- `winning.thurstone` renamed `winning.research` (honest labeling of
  research-grade machinery); the old name remained as a
  `DeprecationWarning` alias that also served submodule imports.
  (Superseded: the alias is removed by the entry at the top of this
  section.)

- Dense-covariance front door: `race_probabilities(mu, cov=Sigma)` and
  `winning.factor.core.fit_covariance(Sigma, k, m)` package the paper's
  dense pipeline (certified quotient factor fit, blocks and residual
  promotion on the projected residual, closing `(P∘P) d = diag(P R P)`
  diagonal solve). In-grammar inputs are returned undistorted.
- Grammar-wide inversion: `abilities_from_race` accepts `structure=` and
  `cov=`; blocks/nested/tree invert through the exact forward dispatch
  with a damped, variance-matched-preconditioner fixed point.
- `factor_model_projected` D-step collapsed to its n-dimensional normal
  equations (Gram is exactly `P∘P`): 67 s → 0.27 s at n=300, identical
  minimizer.
- The classic lattice API moved to `winning.classic`; the old top-level
  imports keep working as aliases that raise a `DeprecationWarning`.

## 1.2.0 (2026-08-27)

The structured-covariance engine.

- One race, five covariance grammars: `Independent`, `Factor`, `Blocks`,
  `Nested`, `Tree` dataclasses accepted as `structure=` by the front-door
  verbs; `Tree.from_linkage(Z)` builds the race whose implied correlation
  is exactly the (floored) cophenetic matrix of a hierarchical clustering.
- Block, nested and tree kernels with exact block/nested Jacobians, an
  (approximate-across-clusters) tree Jacobian, and hybrid fixed-point +
  Newton inversion (`abilities_from_block_race`).
- `polish_race`: the nearest race satisfying linear constraints on
  probabilities (concentration caps), with a finite-difference fallback
  when the analytic Jacobian is approximate.
- Winner-bulk lattice window: 3-4x narrower lattices at equal guarantee;
  default `points` lowered to 257. `window="span"` preserves old behavior.
- Sharpness-adaptive factor quadrature: the default Gauss-Hermite order
  now scales with max ||V_i||/sqrt(D_i) (a fixed 15-node rule silently
  lost up to 5% total variation on sharp fields).
- Sharp-field rescue in `core.win_probabilities_factor`: fields whose
  density spikes fall between span-window lattice points retry once
  through the bulk-window front door instead of raising.
- Compiled kernels: `pip install winning[fast]` pulls the `fastrace`
  abi3 wheels (pyo3 over the pure-rust `winning` crate); every python
  path keeps a numpy fallback that remains the spec. `WINNING_PURE=1`
  or `winning.use_rust(False)` forces pure python.
- Parity harness: `parity/vectors.json` embeds inputs and outputs of 22
  scenarios; Python (reference), R, Rust and JavaScript replay them, most
  at machine precision.
- Base-R package (`r/winning`, v0.3.0) and a zero-dependency browser
  port (`docs/js/winning`) mirror the full API.

## 1.1.1

- N=2 inversion closed form; damping 0.7 for general bases at N=2.
