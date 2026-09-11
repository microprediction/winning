# Changelog

## Unreleased

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
