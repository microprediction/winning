# Changelog

## Unreleased

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
  research-grade machinery); the old name remains as a
  `DeprecationWarning` alias that also serves submodule imports.

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
