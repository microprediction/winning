# Changelog

## Unreleased

- `predict_proba` checks the design instead of guessing it, in Julia and
  in Python (#195, #261).
  It decided whether to prepend generated alternative-intercept columns
  from the COLUMN COUNT alone, and `MNProbit` recorded neither whether
  it had generated them nor how many covariates the caller supplied. New
  data with the wrong number of features were therefore silently
  REINTERPRETED rather than refused.

  A model fitted on two covariates without intercepts, handed one
  covariate, prepended two synthetic intercept columns and then used
  only the first `p` of the three. It applied coefficients fitted to the
  covariates to the intercepts, IGNORED the supplied feature entirely,
  and returned a plausible probability row.

  The model now records `intercepts` and `p_raw`, so the count is
  checked: `p_raw` columns means raw covariates and the intercepts are
  generated exactly when the fit generated them, `p` columns means the
  design is already assembled, and anything else raises with both
  numbers named. The alternative count is checked too.

  The python reference had the identical defect and it is fixed here
  too. Review found it, not me: I fixed the port the issue named and did
  not sweep, which is the rule I had just written down. Python's version
  is sharper -- the guess is wrong exactly when the wrong width plus the
  `J - 1` generated columns EQUALS the fitted width, so three covariates
  fitted without intercepts, handed one, gained two synthetic intercepts
  and returned `[0.198, 0.605, 0.198]`.

- The exact likelihood refuses a choice it cannot account for, in all
  three ports (#194). The cores iterate over the LEGAL alternative
  labels and gather the rows matching each, so a row whose choice is
  outside the range is never visited: it contributed nothing to the
  log-likelihood and a zero row to the score, and the fit silently
  optimised a SUBSET of the data while reporting it as the whole.

  The direction is what makes it dangerous. Dropping observations RAISES
  the log-likelihood, because there is less of it -- five bad rows out
  of forty moved it from -55.699 to -49.134 -- and nothing downstream
  can tell that from a better fit. The tests assert that property
  directly rather than describing it.

  Julia additionally accepted a choice vector SHORTER than `T` and
  dropped the tail; R left `logp == 0` for `NA` rows, which adds zero
  negative log-likelihood and zero gradient. All three now name the
  first offending row and count how many there are. An integer-valued
  float is still accepted, because refusing `1.0` would be pedantry
  rather than a guard.

- Prediction and likelihood choose the same quadrature (#213).
  `choice_loglik_and_score` gauge-centers `V` and dispatches on the
  pairwise-safe bound `sqrt(2) max_i ||(PV)_i|| / sqrt(min D)`.
  `MNProbit._prob_of` dispatched on `max_i ||V_i||`: no centering, no
  `sqrt(2)`, no `D`. A fitted model could therefore optimise and report
  under scrambled Sobol and then be PRICED on the 7-point Hermite
  tensor. Julia already matched its own likelihood.

  Two ways it showed. It was not gauge-invariant -- only loading
  DIFFERENCES decide a race, yet adding a common offset of 5 to every
  row moved the statistic from 2.0 to 7.0 and swapped 49 nodes for 1024
  on an unchanged race. And it was too weak: a centred spread of 2.2
  scored 2.2 against the likelihood's 3.11, so prediction took the
  Hermite tensor and priced 2.9e-2 away from the rule the fit used.

  The rule now lives in one place, `likelihood.sharpness_bound`, because
  it was written twice and the copies drifted. Below the threshold
  nothing changes: the same field measures a 3.161e-3 row-sum defect
  before and after. Above it the sharp field improves from 1.6e-3 to
  6.2e-4, which is the escalation doing its job.

- GMRFExtremes answers a threshold out in the tail (#197). The grid was
  built from the chain's means and marginal sds alone, so a threshold
  above the last cell left the occupancy identically one and every
  answer stopped depending on `u`: 9, 10 and 20 sd out all returned the
  same 1.11e-15, which is quadrature noise rather than a tail. The
  excursion was then taken as `1 - max_cdf`, a subtraction of two
  numbers that agree to every bit, so it went NEGATIVE at 10 sd, and
  `first_passage` differenced the same quantities and returned a
  negative passage mass.

  Three things, because one alone was not enough and I measured each:

  1. the grid covers the threshold, with the SPACING preserved -- the
     point count grows with the range, up to a memory budget that
     accounts for the O(n L^2) transitions, and past that the caller is
     told rather than left with a silently coarser lattice;
  2. the exceedance is summed over the cells ABOVE `u` instead of
     subtracted from one, so it is a sum of small positive terms with
     full relative accuracy;
  3. the cell mass falls back to the midpoint density where `ndtr` has
     saturated, since the double-precision CDF reaches exactly 1 around
     8.3 sd and the difference underflows to zero there.

  With (1) and (2) alone the answer was exactly 0.0 at 9 sd and at 12
  sd alike -- no longer negative, still independent of `u`.

  | threshold | before | after | exact |
  |---|---|---|---|
  | 6 sd | 9.911e-10 | 9.878e-10 | 9.869e-10 |
  | 9 sd | 1.110e-15 | 1.128e-19 | 1.1286e-19 |
  | 12 sd | 1.110e-15 | 1.780e-33 | 1.7765e-33 |
  | 20 sd | 1.110e-15 | positive, decreasing | -- |

  `first_passage` at 9 sd was `[-3.1e-15, 1.0]` and is now
  `[1.09e-19, 1.0]`.

- Malformed portfolio limits are refused, not dropped (#247). `nameCaps`
  and `groups` encode name and sector caps, so a typo that DROPS a
  constraint returns an ordinary-looking result that is simply
  under-constrained -- worse than an error. In the browser:

  | input | before |
  |---|---|
  | `nameCaps` of length n-1 | last name uncapped; an 80% position came back with `maxViolation` 0 |
  | `nameCaps` of length n+1 | silently truncated to n |
  | a group index of n | wrote past the row; every runner NaN, reported feasible |
  | a negative group index | stored as a non-element property, member silently ignored |
  | an `A` row of the wrong width | the constraint vanished |

  Two of these needed the python side too, because the ports disagreed
  about what the input MEANT rather than both being wrong. Numpy read a
  negative group index as counting from the end, so python silently
  capped the LAST name where the browser dropped the member. Neither is
  documented and neither is what a group membership means, so both
  refuse it now.

  A non-finite `name_caps` ENTRY is left alone: "NaN/None entries
  skipped" is the documented contract, meaning no cap for that name, and
  my first cut broke it. That is a feature in both ports.

- A worthless dividend prices at zero, in the browser AND in R (#242).
  `prices_from_dividends` maps only a MISSING quote to `nan_value`. The
  browser used `Number.isFinite(x) ? x : nanValue`, which conflates every
  non-finite value with a missing one, and both ports divided by the
  dividend unconditionally. So an infinite-dividend entrant took
  `1/2000` of the book instead of nothing, a dividend of `0` produced
  `Infinity` and then `NaN` after normalising, and a NEGATIVE dividend
  came back as a negative probability.

  The issue reported the browser. R shares two of the three: only the
  `Inf` case was right there, because `1/Inf` is 0 on its own. Both now
  follow python exactly -- a non-positive dividend (and `-Inf` with it)
  prices at 0, `+Inf` prices at 0, a missing quote still takes
  `nan_value`, and the book is normalised only when its total is
  positive, so an all-infinite book is zeros rather than `0/0`.

  | dividends | was (browser) | was (R) | now, all three |
  |---|---|---|---|
  | `[2, 4, Inf]` | 0.666, 0.333, 0.00067 | correct | 2/3, 1/3, 0 |
  | `[2, 4, 0]` | 0, 0, NaN | 0, 0, NaN | 2/3, 1/3, 0 |
  | `[2, 4, -5]` | 0.909, 0.455, **-0.364** | same | 2/3, 1/3, 0 |
  | `[Inf, Inf]` | 0.5, 0.5 | NaN, NaN | 0, 0 |

- The browser differentiates and polishes the factor law it was GIVEN
  (#209). `raceProbabilities` has always accepted caller-supplied factor
  nodes and weights, `{V, D, F, W}`. `raceJacobian` and `polishRace`
  rejected `F` and `W` outright, and `raceJacobianExplicit` built its own
  standard-normal Hermite rule whenever `V` was nonzero. So the browser
  could PRICE a discrete or otherwise non-Gaussian factor and could not
  differentiate it: dropping `F, W` is not a workaround, it silently
  changes the model.

  This was not a cosmetic gap. On a two-point law `F = [[-3], [3]]`,
  `W = [0.5, 0.5]`, the Jacobian the browser returned differed from the
  true one by 0.205 in absolute terms. It now agrees with finite
  differences of the same forward to 8.1e-12, and `polishRace` threads
  the law through its forward, its Jacobian AND the inverse that builds
  `mu0` -- polishing under a different law than the caller priced with
  optimises the wrong model. A binding cap of 0.35 reprices to
  (0.35, 0.30, 0.35) under the caller's own nodes.

- The browser prices the one-leaf tree (#241). An EMPTY linkage is the
  valid scipy-style spelling of a hierarchy with one leaf, and
  `treeFromLinkage([])` computed the leaf variance as
  `1 - rho[parent[i]]` with `parent[i] = -1`. Javascript has no negative
  indexing, so that is `undefined` and `D` came out `[NaN]`; pricing the
  tree then failed instead of returning the certain `[1]`. The guard was
  already written correctly one loop above, for the node strengths.

  Python reaches the right answer by accident: numpy wraps `rho[-1]` to
  the sole zero entry. Both now give `D = [1]` for the empty linkage and
  `D = [0.5, 0.5, 1]` for a three-leaf one.

- A coordinate with no idiosyncratic variance is treated as
  deterministic, in all three structured-MVN ports (#206). Given the
  factor draw such a coordinate is a CONSTANT, `X_i = mu_i + v_i . f`, so
  its conditional cell is an INDICATOR, not a gaussian interval. Every
  port divided by `s_i = 0` anyway. Off the boundary that happened to give
  the right answer; on an inclusive boundary it is `0/0`, so the
  documented `P(X_1 <= 0) = 1` for `X_1` identically zero came back as
  `nan` from python and Julia, and as "missing value where TRUE/FALSE
  needed" from R.

  A constant that lies outside its own interval now returns exactly
  `0.0` with method `outside-support`, rather than the `1e-300` the cell
  floor gave it and a trip through the recentered path looking for a
  probability that is not there. Checked against one-dimensional
  quadrature for the harder case where `D_i = 0` but `V_i != 0`, so the
  coordinate is constant only GIVEN the factor and the indicator
  restricts the factor region instead: filter 0.32379233, quadrature
  0.32379181.

- Julia broadcasts a scalar bound, as the specification does (#184).
  `winning.fastmvn` broadcasts with `np.broadcast_to` and the R port with
  `rep_len`, but `julia/FactorMvNormalCDF` called `collect` on the
  argument. A scalar `Float64` is not iterable, so the ordinary joint-CDF
  spelling `mvn_cdf_fast(V=V, D=D, upper=0.0)` threw before evaluating
  anything, on the structured and the delegated dense path alike. A
  length that is neither 1 nor `n` now raises with both numbers named,
  rather than recycling silently.

- Quadrature weights are RELATIVE, in every verb that takes them (#208).
  `W` and `c*W` describe the same factor law, and every verb normalises
  its own result, so the common scale cancels -- except in the
  pre-normalisation mass checks, which compared an unnormalised row sum
  against 1. `removal_shares` and `ordered_probabilities` therefore
  REJECTED `W = [5, 5]` while accepting the identical law
  `W = [0.5, 0.5]`, with a "mass defect 9.00e+00" that is just the
  weight total minus one.

  The issue named `removal_shares`. Sweeping every public verb that takes
  `mu`, `F` and `W` found two more: `ordered_probabilities` raised the
  same way, and `plackett_luce_prefix_logprob` returned a
  log-probability shifted by exactly `log(sum W)` -- the same law spelled
  `[5, 5]` read -1.5071 as +0.7955 -- because it skips `_setup` entirely
  when the caller supplies the law.

  `_setup` is where the factor law is assembled, so it is the one place
  the rule is decided, as `as_loadings` is for `V`. Every rule the
  library generates already sums to one, so nothing it produces changes;
  a zero or negative weight total is now refused with a reason instead of
  being carried into a mass check. The sweep lives in
  `tests/test_shape_contract.py`, which fails if a new verb takes a
  factor law and is not covered.

  The R and Julia helpers clamp in floating point before converting
  (#228). Both computed the uncapped requirement in a fixed-width integer
  and overflowed BEFORE `min(need, 8193)` -- in the very regime the cap
  exists for. A narrowest sd of 1e-10 needs about 5.9e11 points: R's
  `as.integer()` returned NA, so the `if (need > 8193)` meant to warn
  errored with "missing value where TRUE/FALSE needed", and Julia threw
  `InexactError`. Python was unaffected only because its integers are
  unbounded, which is precisely why a port cannot inherit a numeric
  argument from it. All four now return 8193 there and warn; the browser
  then rejects the field on the mass check, which is the right layering.

- Top-k and rank refine the lattice to the NARROWEST runner, in all four
  engines (#224). The window is set by the widest runner and the grid by
  `points`, so a heterogeneous field left the narrowest density between
  samples: sds of 0.093 and 6.02 at the default 513 points gave a spacing
  of 0.174, nearly twice the narrow runner's whole standard deviation,
  and its top-4 membership came out 4.4e-3 wrong.

  The mass check could not see it, and no tightening of that check would
  have. It is ONE SCALAR -- the memberships sum to k -- and runner-level
  errors of opposite sign cancel in it: the raw total was 3.99440 against
  a tolerance of 0.02, comfortably inside, and the routine then rescaled a
  wrong vector to sum to four. The cure is resolution, not a tighter
  aggregate.

  Same rule the win race has had all along (`races.forward_grid`): about
  two points per narrowest sd, capped at 8193, warning when the cap still
  leaves the lattice coarse. Measured on the reported field, the error
  falls from 4.4e-3 to 4.8e-9 at the default `points`, and stops moving
  with resolution at all -- 513 and 2049 now agree to 1e-9 where they
  differed by 4.4e-3.

  It subsumes two earlier rejections. The #203 field promotes itself to
  2171 points and the #221 field to 6845, both then agreeing with an
  8193-point answer to 5e-11, so neither has to be refused any more. The
  guards from #221 stay as the backstop for what refinement cannot reach:
  a field whose narrowest sd is 5e-5 needs about a million points, and it
  warns and is rejected. Tests cover all three bands -- resolved, warned,
  rejected. Parity vectors are unchanged, since the refinement does not
  fire on fields with mild scale spread.

- GMRFExtremes integrates its transition kernel over each lattice cell
  instead of sampling the density (#244). `npdf(z) * dx` is a probability
  only where the density is flat across a cell. An innovation sd far
  below the lattice spacing makes the kernel a SPIKE between samples, and
  the spacing is set by the MARGINAL sd, so a chain with innovation
  sd 1e-4 gave transition columns summing to about 80: `max_cdf` returned
  79.988 for a probability, `excursion_probability` -78.988, and
  `first_passage` a vector with a -79.488 in it.

  The cell mass is a CDF difference with the variance deflated by
  `dx^2/12`, which is exactly what averaging over a cell adds. That
  matters: the plain CDF difference is exact in mass but smooths by a
  boxcar at every step, and it measured WORSE than the old midpoint rule
  where the old rule worked. With the deflation the resolved regime is
  unchanged to the digit, and the unresolved one gets its correct
  degenerate limit.

  | innovation sd | before, 200 points | after | before, 3200 | after |
  |---|---:|---:|---:|---:|
  | 1.0 | 1.3e-4 | 1.3e-4 | 5e-7 | 5e-7 |
  | 0.25 | 3.6e-4 | 3.6e-4 | 1.4e-6 | 1.4e-6 |
  | 0.01 | **1.1** | 1.6e-3 | 3.4e-5 | 3.4e-5 |
  | 1e-4 | **160** | 1.6e-5 | **9.5** | 1.6e-5 |

  Renormalising the columns would have produced numbers in [0, 1] and
  hidden the resolution failure, which is what #217 and #221 are about.

- The Euler-Maclaurin end correction is gone from both browser copies of
  the tabulated base (#216). #211's description and this changelog both
  said the correction measured worse and was removed; it was left in the
  code, so an undocumented numerical change landed on `main` and it
  touched every tabulated base, not just the `t4` span widening that PR
  was about. Measured on the committed fixtures, removing it:

  | fixture | with correction | without |
  |---|---:|---:|
  | skew forward vs scipy | 4.518e-7 | 4.472e-7 |
  | t4 forward vs scipy   | 8.439e-7 | 8.423e-7 |
  | skewNormalBase(0) = normal | 2.68e-7 | 3.52e-7 |

  Two of the three improve and the third degrades, all of them a fifth or
  less of their tolerances -- which is the point: the correction never did
  anything, and it could not have. The cumulative is chained across three
  pieces with two different step sizes, corrected per piece with that
  piece's `h` and with nothing at the joins, and `f'(a)` was taken as
  `fpL[0]`, which the centred-difference helper never writes, so it was
  identically zero rather than the derivative at the left end. What
  actually closed the 1e-5 survival gap in #211 was the span widening,
  `BASES.t4` `[12,12] -> [24,24]`. Both copies stay byte-identical.

- `polishRace` accepts `mu0` again (#207). It reads the option as
  `const { mu0: mu0In = null } = opts`, and #204's allowlist named the
  destructured LOCAL, `mu0In`. So a call that had worked since the
  function was written threw `unknown option 'mu0'`, while `mu0In` was
  accepted, ignored, and then failed with `give p0 or mu0`.

  The surface test that exists to catch this searched the function body
  for the allowlisted string and found the local name, so it passed. It
  now parses the destructuring and compares PUBLIC keys, and a second
  test covers the opposite direction -- a key the function reads that its
  allowlist rejects, which is the half that refuses valid callers and the
  half #207 actually was. Sweeping all 21 guarded browser APIs that way
  turns up no others. `parity/check_js_api.mjs` calls `polishRace` both
  ways, because neither direction is visible without making the call.

  The first cut of that sweep exempted forwarding wrappers outright, and
  review pointed out the hole (#237): a wrapper reads nothing off `opts`,
  so the audits had nothing to compare its allowlist against and skipped
  it -- leaving `bottomKProbabilities` and `locScaleFromWinAndSecond`
  unguarded in exactly the direction that refuses valid callers. Dropping
  `D` from the first one's list turned `bottomKProbabilities(mu, k, {D})`
  into `unknown option 'D'` with all three source audits still green. The
  exemption now names the API each wrapper forwards to and requires the
  two allowlists to be equal, which is a checkable claim rather than a
  waiver, and the checker calls both wrappers with every option at a
  non-default value.

- The browser inverse honours the two options it advertised (#226).
  `targetFloor` and `returnInfo` were on `abilitiesFromRace`'s allowlist
  and read by nothing, so they passed validation and vanished -- the
  precise failure the allowlist exists to prevent, reintroduced by
  deriving that list from python's signature rather than from what the
  function reads. It now matches python: a zero share raises, because it
  has no finite inverse; `targetFloor` floors deliberately and reports
  which entries were floored; `returnInfo` returns the record rather than
  the bare vector, and does not change the answer.

  A test now requires every allowlisted key to be read by the function
  that advertises it. Two functions legitimately forward their whole
  options object onward, and that is DECLARED rather than inferred --
  "the body mentions opts somewhere" would have excused this very bug,
  since `abilitiesFromRace` forwards on its structure branch while
  `targetFloor` vanished on every other path. Sabotage-tested by putting
  an unread key back.

- The browser honours the loading-shape contract, in every module that
  takes `V` (#232). `winning.shapes.as_loadings` is the one place the rule
  is decided -- `V` is `(n, rank)`, one ROW per contestant, and a scalar,
  a length-n vector, `(n, rank)` and `(rank, n)` are all the same race.
  Four browser modules decided it for themselves instead, by indexing `V`
  and taking the rank from `V[0].length`. So a length-n vector threw
  `V[i].reduce is not a function`, a valid `(rank, n)` matrix was read as
  rank n and rejected with a message about the rank cap, and a RAGGED `V`
  was truncated to the first row's width and answered all-NaN with no
  complaint at all. `core.mjs` now exports `asLoadings`, and `races.mjs`,
  `topk.mjs` and `polish.mjs` call it at the door. All four spellings now
  agree with the python reference to 1.4e-16, and `(rank, n)` is
  bit-identical to `(n, rank)`.

- Halton bases are generated, not tabulated (#233). The low-discrepancy
  nodes read their primes from a literal array: 24 entries in
  `races.mjs`, 16 in `demo.mjs`. One factor past the end gave
  `base = undefined`, and the node loop produced NaN rather than an error,
  so a rank-25 race and a rank-17 demo returned NaN for every runner in
  silence. `firstPrimes(d)` generates them. This is the same defect as
  R's 30-prime table in #190, in a tree I did not grep when I fixed that
  one; the browser checker now exercises ranks 8, 17, 25 and 40.

  Both fixes are guarded behaviourally in `parity/check_js_api.mjs`, which
  CALLS the functions rather than reading the source. Rehearsing four
  sabotages against it found a fifth defect in the checker itself: an
  exception on a happy path killed the file, so the run failed with no
  named check and every later check silently went unrun. A new `accepts()`
  helper turns a thrown exception into a named FAIL.



- The pre-renovation `src/` package is gone, all but the one part still
  used. It had sat since the August renovation: not packaged (`setup.py`
  lists only `winning.*`), not collected (`pytest.ini` does not name it),
  and four of its filenames -- `lattice_calibration`, `lattice_conventions`,
  `skew_calibration`, `std_calibration` -- shadowed live ones under
  `winning/classic/`, so a grep returned two hits with nothing saying
  which was current. Those are deleted, with `elo`, `kernels`, `shims`,
  `thurstonerating`, the old test tree and the benchmarks: nothing
  referenced any of them.

  What survives is `attic/src/winning`, holding the dependency closure of
  Glicko-2 -- `glicko2.py`, `exact.py`, `ratingsystem.py` -- because six
  chess experiments compare against it and the live package has no
  Glicko-2.

  **It had stopped working.** `exact.py` imported the external `thurstone`
  package, which is retired, and `winning.thurstone` is now a tombstone
  that raises on import, so every one of those experiments would have
  failed at the Glicko-2 step. It imports `winning.research` now, which is
  the migration that tombstone's own message prescribes. Verified end to
  end: A beating B three times gives A 1753, B 1247, win probability 0.82.
  A test covers it, since nothing else looks at that directory -- which is
  precisely how it broke.

- The sampler goodness-of-fit test uses a STABLE seed (`zlib.crc32`
  rather than `hash`). `str.__hash__` is salted per process unless
  `PYTHONHASHSEED` is set, so `np.random.default_rng(hash(name) % 2**32)`
  drew a different sample on every run and re-rolled the test's own 9e-5
  tail risk each time. Thirteen bases across six CI platforms is 78 draws
  per round, which is a failure every few rounds: one hit a macos runner
  at 0.010262 against the 1e-2 bound, on a pull request that changed only
  browser javascript. The comment in the test blamed platform differences;
  those are why the scipy-backed samplers consume the stream differently,
  not why the draw changed between two runs on the same machine.

  Each (base, platform) pair now has one fixed draw: it passes forever or
  fails immediately, rather than being re-rolled. Locally the worst base
  sits at 0.006956, a 1.44x margin, and over 120 seeds the previously
  failing base has median 0.0038 and max 0.0067.

- All four structured-MVN ports reject an empty rectangle instead of
  pricing it at the underflow floor (#235). `P(lower <= X <= upper)` with
  `lower_i > upper_i` is an EMPTY event. Python, the vendored standalone
  python, R and Julia each formed the negative conditional cell --
  `Phi(0) - Phi(1) = -0.341344746...` -- and then clamped it with
  `max(cell, 1e-300)`. So an impossible observation came back as a finite
  probability, and in log-likelihood code as about `-690.8` rather than
  `-inf`. On the recentered path it is worse than a clamp: importance
  integration then integrates the artificial constant cell.

  `mvtnorm::pmvnorm`, which `r/mvtnormfast` is a drop-in for, raises on
  reversed bounds and returns 0 when a coordinate has `lower == upper`.
  All four ports now do exactly that, and report the degenerate case as
  method `degenerate-rectangle` with a value of exactly `0.0`, not
  `1e-300`. The error names the offending coordinate.

  The check sits in front of the dense routes too, not only the factor
  one. `scipy.stats.multivariate_normal.cdf` returns the NEGATIVE number
  `-0.341...` for a reversed rectangle rather than raising, so the python
  fallback branch needed its own guard; the julia dense delegation and
  the R `mvtnorm` fallback are covered the same way. Distinct from #196,
  which is about cancellation on VALID upper-tail intervals.

- `rprobit_fast` and `mlogit_fast` reject a malformed choice set instead
  of reshaping it (#230). Both price a field by a POSITIONAL reshape --
  `matrix(X %*% beta, nrow = Tn, ncol = J, byrow = TRUE)` -- so row order
  alone decides which alternative a row is priced as, and the only guard
  was a count: `nrow(df) == Tn * J`. An observation that duplicates one
  alternative and omits another still has J rows, so it passed, and the
  duplicate row was priced as the alternative that was missing. The
  reported case fits a covariate value of 10 onto an alternative that
  does not exist for that observation and returns an ordinary-looking
  coefficient.

  The contract is one row per (observation, alternative), which is a
  table and not a total, so both packages now check the contingency table
  and name the offending observation and alternative. A well-formed panel
  is untouched.

- The browser inverse honours the two options it advertised (#226).
  `targetFloor` and `returnInfo` were on `abilitiesFromRace`'s allowlist
  and read by nothing, so they passed validation and vanished -- the
  precise failure the allowlist exists to prevent, reintroduced by
  deriving that list from python's signature rather than from what the
  function reads. It now matches python: a zero share raises, because it
  has no finite inverse; `targetFloor` floors deliberately and reports
  which entries were floored; `returnInfo` returns the record rather than
  the bare vector, and does not change the answer.

  A test now requires every allowlisted key to be read by the function
  that advertises it. Two functions legitimately forward their whole
  options object onward, and that is DECLARED rather than inferred --
  "the body mentions opts somewhere" would have excused this very bug,
  since `abilitiesFromRace` forwards on its structure branch while
  `targetFloor` vanished on every other path. Sabotage-tested by putting
  an unread key back.

- `rank_probabilities` keeps BOTH guards, before and after normalisation
  (#221). The entry above replaced the raw check with a post-normalisation
  one, and that was wrong: they are independent, not alternatives. Row
  normalisation can ERASE a gross raw defect. On a field whose sds span
  232x the raw matrix is off by 0.249 in a row and 0.199 in a column, and
  dividing by those same row sums leaves a column defect of 0.0047 --
  inside the 5e-3 tolerance -- so it returned a false success, while
  `top_k_probabilities` rejected the same field at 0.056. The raw check
  sees the quadrature; the post check sees what the caller gets; neither
  implies the other.

  The two fields now on record prove the independence, and each slips past
  the guard that does not catch it: #221's is caught only before
  normalisation, #203's only after. A test asserts that, and another
  asserts what the user actually saw -- that rank and top-k agree about
  whether a field resolves at all. Python, R, Julia and the browser all
  carry both; R and the browser agree with python to 9e-10 at 2001 points.

- Binary choices fit in `rprobitfast` and `mlogitfast` (#183). The
  triangular-loading loop read `for (row in (col + 1L):J)`, and at J = 2
  with the default rank 2 the second pass evaluates `3:2` -- which in R
  is the two-element vector `c(3, 2)`, counting DOWN, not an empty range.
  The first assignment was therefore `V[3, 2]` on a 2x2 matrix and every
  binary-choice fit died with "subscript out of bounds"; `nw` counts one
  free loading there, so `wfree[2]` was past the end of the parameter
  vector as well. `col + seq_len(J - col)` is empty exactly when it should
  be, and identical elsewhere -- checked, not assumed: at J = 5, rank 2
  the two loops visit the same seven cells in the same order.

  Fixed in all three copies of the loop (`mlogitfast/R/mlogit_fast.R`,
  `rprobitfast/R/engine.R`, `rprobitfast/R/rprobit_fast.R`). A 180-subject
  binary probit now recovers a true slope of 0.8 as 0.757, where it raised
  before.

- The browser factor race's own parity suite runs in CI (#140, second
  half). `js/factor/test_parity.mjs` checks the tabulated bases against
  scipy-generated vectors, and nothing ran it -- which is why it sat
  failing on `main` at 1.01e-5 against its own 2e-6 tolerance until #211.
  An ungated suite stays broken for exactly as long as nobody happens to
  run it. It is now a step in the `parity` job, alongside the vector
  check, both port checkers and the browser API guards.

- `rank_probabilities` validates the matrix it RETURNS, in all four
  engines (#203). It checked the raw quadrature matrix, divided by the raw
  row sums and returned that, and row normalisation is not neutral: it
  makes every row exact and MOVES the columns. On a field whose sds span
  73x (0.19 to 14.3) at 513 points, the column excess grew from 4.5e-3 to
  5.4e-3 through that division, past the tolerance the check had just
  applied -- and the documented identity, that cumulative rank rows
  reproduce `top_k_probabilities`, broke by 2.6e-3. Nothing looked at the
  matrix afterwards.

  The columns are deliberately NOT forced. Alternating row/column scaling
  would make both identities exact and would do it by hiding what caused
  the defect: the same field at 2001 points has a column error of 3.5e-9,
  so the quadrature is what is wrong, not the normalisation. Raising says
  so, and the error now names both errors and where they were measured; a
  doubly stochastic answer to a question the lattice could not resolve
  does not. Above 2001 points that field's cumulative rows reproduce
  `top_k_probabilities` to 3.5e-9, which is the quadrature's own error.

  Python, R, Julia and the browser all carry it, and R and the browser
  agree with python to 1e-15 on the fixture at 2001 points.

- The browser factor race's Student-t4 base integrates over a window wide
  enough for its tails, and `js/factor/test_parity.mjs` is green for the
  first time in a while. Its `t4 forward shares vs scipy` check had been
  failing at 1.01e-5 against a 2e-6 tolerance, and the cause was neither
  the tabulated survival (accurate to 1e-9 against scipy's exact `sf`,
  measured) nor the lattice resolution (python at 257, 501, 2001 and 8001
  points spans 1e-7): it was the `spans` constant. A t4 keeps real mass
  far out, and 12 sd truncated it. Measured on that fixture at the default
  501 points:

      span   forward error   inverse over 501..4001 points
        12       1.01e-5     2.9e-7 .. 2.3e-6
        20       1.75e-6     2.9e-7 .. 5.0e-7
        24       8.42e-7     2.9e-7 .. 3.5e-7
        32       2.37e-7     2.9e-7 .. 5.3e-7
        40       1.39e-7     NaN at 501, 1001 and 4001

  Both ends cost something, which is why this is a table and not a maximum:
  too narrow drops tail mass, too wide spreads a fixed point budget until
  the bulk is under-resolved. 24 is where the forward clears its tolerance
  with margin and the inverse is better than it was at every resolution.
  The NaN past 40 is the inverse's own fragility rather than this constant
  and is filed separately (#210); nothing ships near it.

  Recorded because it was nearly fixed wrongly twice: an Euler-Maclaurin
  correction to the cumulative, which measured 1.39e-7 against 1.21e-7
  without it and was removed, and a reading that blamed the 257-versus-501
  default mismatch, which accounts for 1e-7 of a 1e-5 gap.

- R's `fit_covariance` threw for every 2x2 covariance (#181). The closing
  solve is against `P o P = a I + b 11'` with `a = 1 - 2/n`, and at n = 2
  that `a` is exactly zero, so the matrix is rank one and `solve()` failed
  on any two-runner input, public `cov=` calls included. The python
  reference has had the two-runner branch all along -- one contrast, so
  the total is spread evenly -- and R now has it too, matching python to
  1e-14 at rho = 0, 0.6 and 0.95.

- The browser factor race no longer invents a longshot floor (#182), and
  its deployed copy is the same file as its source (#139). `tabulatedBase`
  clamped the lookup at the table edge for the density as well as the
  survival, so beyond 40 sd the density stopped decaying: a Student-t4
  two-runner race reported 3.5e-7 for gaps of 80 and 100 alike. Heavy
  tails are where that bites, since a t4 survival decays polynomially and
  no span makes the omitted mass negligible -- and the old normalisation
  divided by the inner mass alone, which is the same error twice. There is
  now a coarse outer tier to 4000 sd and one cumulative sweep across all
  of it, so the tail falls as it should: 4.9e-7, 8.8e-8, 2.6e-8, 1.0e-8,
  5.7e-10 at gaps 40 to 200, a ratio of 18 from 100 to 200 against the
  2^4 = 16 that t4's exponent predicts.

  `docs/assets/js/factor_race.mjs` is what three doc pages load and is a
  copy of `js/factor/factor_race.mjs`. It had silently missed the loading
  gauge fix (#139) and would have missed this one the same way, so a test
  now requires the two to be byte-identical.

- Every browser entry point validates its own options, not two of twenty.
  The guard added for `cov=` covered `raceProbabilities` and
  `abilitiesFromRace`; review walked into the rest inside a day, which is
  fair, because a javascript object swallows a key nobody reads and
  eighteen exports were still relying on the language to catch what it
  cannot. `checkOpts` now lives in `core.mjs` with the reasons worth
  giving, and each of the twenty declares its own key set. A test fails
  if any options-object export does not call it, and
  `parity/check_js_api.mjs` CALLS all twenty-one cases rather than reading
  the source for markers -- which is how the first round shipped.

  Two of those swallowed keys were changing the answer:

  `rankProbabilities` ignored `V` and `qa` and returned the INDEPENDENT
  rank matrix (#199) -- plausible, doubly stochastic, and for the wrong
  model. It now takes the same factor-node mixture `topKProbabilities`
  does, matching python to 2e-16 with and without loadings; the loadings
  move that fixture's matrix by 0.096, which is the error that used to be
  silent. A parity scenario pins it, declared for the browser because R's
  `rank_probabilities` does not take `V` yet (#202) -- the metadata says
  so in the open rather than the scenario being left out.

  `locScaleFromTopkPair` and `locScaleFromWinAndSecond` ignored `V` and
  could report `converged: true` (#200). Python RAISES there, because with
  fixed loadings two curves carry 2n - 2 numbers against 2n - 1 unknowns
  and a flat direction survives the lost rescaling gauge. The browser
  carries that reason now instead of turning a required refusal into a
  confident answer.

- Six defects in the same day's work, all found by review within hours of
  merging, all fixed here. Recorded plainly because the pattern is the
  lesson: a safety net built quickly has holes, and four of these six were
  in the net rather than the engine.

  **The GHK route crashed above 31 runners** (#190). `.halton_unit` held a
  30-prime table and called `stop()` past it, so a degraded `cov=` at n=32
  terminated a call that used to return a factor-fit answer. Primes are
  generated now. The family's high-dimensional projections were the real
  question, so the agreement was measured rather than assumed: max|R -
  python| is 2.7e-4 at n=16, 5.6e-4 at n=32 and 3.6e-4 at n=48, all far
  inside the 4.6e-3 that pricing the degraded fit costs.

  **R routed on a predicate that missed two of python's three failure
  classes** (#189). `close_fit` clamps `D` at `1e-3 * mean(diag(C))` while
  the degradation test compared it against `1e-6 * diag(C)` -- three orders
  of magnitude apart, so a clamp-bound fit counted ZERO bound entries and
  was priced rather than routed, silently restoring the divergence #188
  had just closed. The clamp is now reported by `close_fit` and tested
  against, and the pairwise-contrast residual python also keys on is
  computed. On the n=8 fixture `bound` goes 0 -> 8.

  **A `ports` typo made a scenario skip everywhere and still pass**
  (#191). `"r"` or `"bogus"` matched neither checker, both printed a skip,
  and CI exited 0: a committed fixture became a no-op. The names are a
  closed set now, validated when vectors are written and again when they
  are checked.

  **A NaN passed the staleness gate** (#185). `d = max|a - b|` is NaN if
  either side is, and IEEE makes both `d <= tol` and `d > tol` false, so
  the row printed STALE and the process exited 0 -- a false negative for
  precisely the catastrophic regression the gate exists for. Non-finiteness
  is now its own failure, on either side and on the tolerance.

  **Each browser API accepted the other's options** (#186).
  `raceProbabilities({nIter: 1})` and `abilitiesFromRace({returnSlopes:
  true})` returned plausible answers while ignoring the key, because one
  shared allowlist served both -- recreating the exact failure mode the
  guard was added to remove. Separate allowlists, and the inverse hands the
  forward an explicit object instead of spreading its own options through.
  `parity/check_js_api.mjs` now CALLS these, in the gated job: the previous
  tests read the source for markers, which is why this shipped.

  **The surface registry covered two modules of seven** (#187).
  `top_k_jacobians`, `race_jacobian`, `polish_race` and `fit_covariance` --
  three exported by R, two by the browser -- had no declared status while
  this file claimed to cover the surface. Discovery now walks every factor
  module (53 verbs, up from 28), and a second test audits the other
  direction: an R or browser export that answers to no python verb must be
  declared too. It immediately found two undeclared browser exports.

- R serves the `cov=` route itself, forward and inverse (#173 in part).
  The warning added above said the packages disagree by design; they now
  agree. `r/winning/R/ghk.R` ports the GHK the python package routes a
  degraded fit to, with both properties that route learned the hard way:
  the runners are sorted into a canonical order before the estimate and
  mapped back, so the answer is exactly permutation-equivariant where a
  fixed point set was 8.8e-3 label-dependent; and everything stays in log
  space, because a runner far behind underflows to zero and the inverse
  Newton-steps on log residuals. Halton uniforms keep the package
  dependency-free where python uses scrambled Sobol.

  Measured against the python route on its own n=8 fixture: 3.4e-5 at
  4096 nodes, against 4.6e-3 for the fit R used to price -- a factor of
  135 -- and exactly equivariant. GHK accumulates d log p / d mu in its
  conditioning pass, so the inverse Newton-steps against the routed map
  with those slopes rather than against the fit; routing only the forward
  would make the two front doors describe different races, which is
  python's #164. The round trip closes to 2.6e-10.

  The degraded-fit warning stays for the paths that cannot route -- the
  slopes, a non-normal base -- exactly as in python.

- The parity harness can express partial coverage. A scenario may declare
  which ports implement it, and the others print a visible skip instead of
  a failure; before this, a capability one port lacked could only be
  expressed by leaving the scenario out entirely, which is how the
  browser's silent `cov=` survived. Two scenarios pin the route
  cross-language (R: 3.4e-5 forward, 1.4e-10 on the inverse; the browser
  declares the skip), and `bottom_k_probabilities`, `hermite_nodes` and
  the inverse under a non-normal base are now covered in all three.

- Closing the port gaps the surface audit made visible, starting with the
  one that was silent. R serves `cov=` and has no GHK, so where python
  routes a degraded fit to scrambled-Sobol GHK, R prices the fit -- 4.6e-3
  apart on the n=8 exactly-rank-3 correlation -- and `fit.R` contained no
  `warning()` at all. `fit_covariance` now reports the two failure classes
  python keys on (the fit degenerating to full rank or driving D onto its
  floor, and a bad residual), and `race_probabilities`/`abilities_from_race`
  warn on them, naming the measured gap and saying the two packages
  disagree here by design. A healthy fit is still priced in silence, and
  agrees with python to 4.5e-4 (Halton against Sobol nodes). Tested on
  both sides: R's own suite, and the surface test, which now requires the
  R warning exactly as it requires the browser's guard -- a waiver for
  `cov=` is only allowed while every port says what it is doing.

  The browser keeps rejecting the key rather than gaining a fit: its
  `fitGrammar` is a reduced pipeline (blocks omitted for latency), so
  wiring it in would trade one silent divergence for another.

- Cross-language divergence is now detected rather than stumbled upon.
  The parity comparison has existed for a long time and had **never been
  gated**: `parity/check.R` and `parity/check.mjs` were run by hand, which
  is why three divergences reached `main` before anyone noticed -- the
  rank-one handover (#153), the pair closed form's weight normalisation
  (#171) and the inverse's damping (#178), each ported only after the
  fact. A CI job now runs both checkers on every pull request, and first
  asserts the committed vectors still describe the python reference: a
  python change that moves a scenario without regenerating them leaves the
  ports compared against yesterday's answers, which was true of eleven
  scenarios when the job was written.

  `tests/test_port_surface.py` covers what no scenario can, since the
  failures that hurt were code nobody called. Every public race verb
  carries a declared status per port -- covered by parity, present but
  uncovered, or absent with a reason -- and a new verb fails the suite
  until someone decides. Every keyword of `race_probabilities` and
  `abilities_from_race` must be exercised by a scenario or waived by name.
  The rule that matters is that a declared gap must be a LOUD gap: `cov=`
  may be waived only because the browser now rejects the key, having
  previously accepted it and returned the independent race. Sabotage-
  tested: an undeclared public verb, a port dropping a verb it claims, a
  new unexercised keyword and the browser quietly accepting `cov` again
  each fail.

  Two gaps the audit found and closed rather than waived: `bottom_k_
  probabilities` and `hermite_nodes` are exported by all three engines and
  were exercised by none, and the inverse had never been run under a
  non-normal base in any port. Scenarios added for all three, and
  `hermite_nodes` immediately earned its keep -- the ports pruned the
  product grid without renormalising, so their weights summed to
  1 - 2e-9 against exactly 1 in python. The R comment asserted it matched
  "exactly as the reference", which stopped being true when python started
  renormalising and the ports did not follow. Both fixed; 44 scenarios now
  agree to 1e-15 there.

- Port audit against the python reference, prompted by three divergences
  the 39 parity scenarios did not catch (#153's handover, #171's pair
  weights, #178's damping). Every decision constant in the three race
  engines was extracted and compared: 27 of 27 agree. Two real defects
  turned up outside that set.

  `race_probabilities(cov=)` in the browser port **silently returned the
  independent race**. `cov` is not among the keys the options object
  destructures, and a javascript object swallows a key nobody reads, so
  the answer was bit-identical to `D = 1` -- including for an all-zero
  covariance matrix. Unknown option keys now throw, in both
  `raceProbabilities` and `abilitiesFromRace`, with `cov` named
  explicitly and pointed at `fitGrammar`. Python and R get this from
  their languages (TypeError, unused argument); only an options object
  needs it spelled out, and a test pins all three.

  R's `fit_covariance` died with "subscript out of bounds" on an n = 8
  exact-rank-3 correlation -- the #118 fixture. The greedy allocation
  (k global + m eigendirections + one per block) can ask for more
  columns than there are eigenvectors, 10 at n = 8, and where python's
  `_top_eigen` truncates silently, `seq_len()` ran off the end. Clamped
  to n. On a healthy fit R now agrees with python to 4.5e-4, which is
  the Halton-versus-Sobol node families, not a rule difference.

  Known and deliberate gaps, recorded rather than closed: the ports have
  no GHK dense route (#161/#164), no temperature, and no ordered
  prefixes, so `cov=` on a DEGRADED fit differs by design -- 4.6e-3 on
  that fixture, where python routes to GHK and the ports price the fit.
  The browser port also accepts a narrower set of `V` shapes than
  `as_loadings` does; it raises rather than mispricing, so it is safe.

- The inverse's damping can go back up, and a monotone mode is summed
  rather than waited out (#178). `_jacobi_sweeps` gave two mechanisms one
  number: the Richardson damping, a persistent fact about the Jacobian's
  spectrum, and a safety net that undoes a sweep failing to contract and
  halves the damping, a response to a transient. Sharing `alpha` meant an
  early transient drove it to the 0.1 floor with nothing able to raise it
  again. On a dense short-length-scale correlation at n = 30 the residual
  fell to 5.7e-1 by sweep 10, ROSE back to 1.1 by sweep 30 as the floor
  took hold, then crawled at 0.981 a sweep and was still 3.2e-1 short at
  120 -- while the same sweeps with no damping at all converged in 93.
  They are now `alpha_base` (persistent) times `penalty` (transient,
  restored a third of the way back on each contracting sweep). Restoring
  `alpha_base` instead does not work: a contracting sweep does not repeal
  a negative eigenvalue, and the #149 heterogeneous cases then stop
  converging at all. Both directions are pinned by one test.

  The remaining mode is not the oscillation the damping was built for.
  Measured on the stalled iterate, the cosine between consecutive steps
  is exactly 1.0 and the residual falls monotonically: two runners 1.5e-5
  apart have a contrast sd of 0.014, so their difference is nearly
  unidentified and the own-slope preconditioner -- which sees each
  runner's own marginal, not the contrast -- understates the step by the
  same factor every sweep. Eigenvalue near +1, not -1, so damping was the
  wrong medicine. Collinear steps decaying geometrically are a geometric
  series, so the step is scaled by 1 / (1 - ratio) and the pair measured
  fresh: Aitken extrapolation in the iterate. It is gated to residuals
  above 1e3 tolerances, because nearer than that the ratio is noise and a
  hundredfold extrapolation overshoots (the top-k pair at n = 10 stalled
  at 1.3e-6 that way).

  Dense n = 16 / 30 / 40 now converge in 43 / 65 / 69 sweeps to ~1e-10 in
  probability where all three failed at 120. Unchanged: the pinned counts
  (n = 8: 10 sweeps, n = 150: 7), the heterogeneous cases (19, 18), the
  effective pairs (17) and the `(V, F) -> (V/c, cF)` invariance (14).

  **The "known limit" recorded against this case in the previous entry is
  withdrawn.** It was described there as two near-duplicate runners
  trading a residual, a mode no diagonal preconditioner could contract.
  That was wrong on the mechanism -- the iteration is monotone, not
  oscillating -- and wrong on the conclusion, since removing the damping
  entirely already converged.

  R and browser ports carry the split and the extrapolation. The
  persistent term is `alpha_base`/`alphaBase`, not `base`: `base` is the
  density argument in every port, and shadowing it made the R inverse die
  inside `.BASES[[base]]`. A test asserts both names and the shared gate
  across the three files.

- The factor-node rule is per rank, and rank 3 gets the order it needed
  (#156). One sharpness threshold of 3.0 served every rank, and at rank 3
  it was defending against a Gauss-Hermite cap of 15 that was itself worse
  than the scrambled-Sobol rule it escalated to: at sharp 4 the cap
  carries total variation 4.4e-4 against Sobol's 1.3e-4, while Q=31 --
  4067 nodes after pruning, half of Sobol's 8192, and 29791 before it,
  inside the same 1e5 budget -- carries 1.8e-5. The cap was the defect,
  not the family. `GH_RULE` now pairs each rank with its cap and the
  sharpness past which even that order loses: rank 2 (41, 3.75), rank 3
  (31, 4.75), rank 4 and up unchanged at (15, 3.0), because a rank-4
  tensor is already 10929 nodes and there is no cheap side to reach for.
  The tensor budget is measured on the cap rather than on a hard-coded 15,
  so raising a cap cannot quietly breach it.

  Measured at n = 250: a rank-3 field at sharp 4 takes 1.50s where the
  escalation took 3.04s, for total variation 2.1e-5 where it was 1.6e-4 --
  twice as fast and several times more accurate. This is the common case,
  not an edge case: a caller reported a calibration costing the same 76s
  at rank 2 and rank 3 because a realistic correlated field crosses the
  threshold at every rank. In correlation terms the old threshold is a
  communality of 0.82, and it is ONE runner's exposure that decides it,
  the statistic being a max -- which is also why the same field could cost
  four times more from one draw to the next.

  Both thresholds are the last sharpness at which Gauss-Hermite beat Sobol
  on EVERY field over 8 seeds, not the median one. That distinction is the
  whole of it: an earlier cut of this table used 3-seed medians and put
  rank 3 at 6.0, and the regression test written to defend it immediately
  found a field losing at 5.6. At rank 3, sharp 5.3 the median says
  Gauss-Hermite wins by 1.6x while the worst of eight fields loses by
  1.35x. The table in `races.py` carries the win rates.

  The R and browser ports carry the same table, and a test reads the caps
  and thresholds out of all three files and asserts they agree, as the
  rank-one handover test does. Not changed, and not measured here:
  `fastmvn._nodes_for` and `likelihood.nodes_for_likelihood` have their
  own `sharp > 3.0` rules over different integrands -- and `fastmvn`
  spells `sharp` without the sqrt(2), so three places compare different
  statistics to the same constant.

- `parity/vectors.json` is regenerated. The committed file had gone stale:
  eleven iterative scenarios (the inversions, the polish and loc-scale
  solvers) differed when regenerated on `main` itself, all below the
  checkers' tolerances, which is why both ports still passed.

- The compiled kernels now take the gauge-fixed loadings and return the
  normalised derivative, as the numpy spec does (#114, #124). The
  kernel does not center `V`; the numpy path subtracts each factor's
  mean loading (a common shock cannot move an argmin) before building
  its lattice window, and the compiled dispatch in `factor.core` and
  `methods.native.lattice` passed `V` uncentered -- a common shift of
  100 moved the forward by 3e-2 and the JVP by 1e-1 with rust on, 3e-16
  off. Centering now happens before the dispatch. The compiled JVP also
  returned the raw derivative of the unnormalised rectangle sum where
  `normalized=True` promises the quotient-rule derivative; the two
  differ by 3.5e-4 at 25 lattice points (the parity test ran at 3001,
  where the total mass is 1 to 1e-13 and the difference is invisible).
  The quotient is now applied from the compiled forward's masses, and
  the rust JVP sums to zero on the simplex as the numpy one does.
- `WINNING_PURE` is decided before the extension loads (#113). Nine
  modules each imported `fastrace` and then consulted the environment;
  under pure mode the extension still ran its loader, so a wheel with an
  ABI or dependent-library failure could take the package down when the
  user had asked for python. `rustconfig.load_fastrace(*kernels)` is
  now the one door: it returns `(None, False, False)` without importing
  under `WINNING_PURE`, and otherwise the module, whether it has every
  named kernel, and the active flag. `factor.permutations` gets its own
  ceiling (it borrowed `races`' flags, and `use_rust(True)` defaulted a
  missing `_RUST_OK` to True, reporting a kernel it could not call). A
  test asserts no module imports `fastrace` directly and that
  `import winning` under pure mode leaves it out of `sys.modules`.
- The `cov=` GHK route is permutation-equivariant and inverted by the
  same map (#162, #164). GHK conditions the runners in the order given,
  so one 8-runner field priced 8.8e-3 apart under two labelings and
  5.6e-3 from a 12M-path Monte Carlo reference at 1024 nodes; the route
  now sorts the runners canonically (by ability, ties by covariance
  row) before the estimate and maps back, which makes relabeling exact
  and, as it happens, halves the error, and spends 4096 nodes: 7.2e-4
  on that field in 13 ms (16384: 1.6e-4 in 46 ms). `abilities_from_race
  (cov=)` used to iterate against the fitted lattice while the forward
  returned GHK, so the two front doors described different races, 4e-3
  to 8e-3 apart on exact rank-1/3/5 fixtures. GHK now accumulates
  `d log p_i / d mu_i` in its conditioning pass (the score of the
  conditional product, holding the truncated draws fixed: 5-25% above
  central differences, a preconditioner not a gradient) and the inverse
  Newton-steps against the routed map itself with those slopes, in log
  space throughout (`qmc_ghk` returns `info["logp"]`, finite where `p`
  underflows: a runner displaced from its near-duplicates on a
  near-singular n=30 correlation has log p = -16000 at the warm start,
  a real value the step recovers from). The three fixtures and the
  #162 field round-trip to 1e-9 in 24-36 sweeps; a dense n=16
  correlation in 43. The degenerate fit's own-slopes are used for
  nothing: with `D` on its floor the conditional race is a near-step
  and the lattice inverse of the rank-1 fixture ran to a log residual
  of 690.
- The inverse damps by the contraction it observes and reads its scale
  off the nodes it is given; `abilities_from_topk(k=1)` shares both
  (#149, #151). Heterogeneous variances make the pair's negative
  Jacobi eigenvalue with no top-two share to gate on (`p = [.6, .2,
  .2]`, `D = [.03, .03, 1]`: share 0.8 exactly, 60 sweeps and 1.7e-4
  short; `[.45, .3, .25]` with `D = [.1, 30, 1]`: share 0.75). The
  sweeps now estimate the dominant mode from consecutive steps and take
  the Richardson damping for it, 2 / (2 - lambda) -- 2/3 at the pair's
  -1, which is why the fixed 0.7 was right where it applied -- and undo
  a sweep that contracts neither the max nor the rms residual. Both
  cases converge in 19; the dominant pair goes 19 -> 17, an ordinary
  n=8 field 13 -> 10, n=150 stays at 7. The warm start and step cap's
  scale is the median idiosyncratic variance plus the factor variance
  under the represented nodes, so `(V, F) -> (V / c, c F)`, the same
  forward map, gives the same inverse in the same 13 sweeps (reading
  `V` alone put the start 33x off at c = 0.03 and returned `[0, 4e-10,
  1]` for `[.6, .25, .15]`); the pair closed forms, forward and
  inverse, use the same node covariance when nodes are supplied (the
  forward gave 0.51 for an exact 0.70 at c = 0.03). `abilities_from_
  topk(q, 1)`, documented as `abilities_from_race`, kept the `n > 2`
  gate after #150 and failed on the same fields; it now runs the same
  sweeps with the win race's gate at k = 1 and the pair's at k >= 2.
  Known limit, stated in the sweeps' docstring -- WITHDRAWN, see the
  #178 entry above: two near-duplicate runners inside a large field were
  said to trade one residual between themselves, a mode no diagonal
  preconditioner contracts. The mechanism was misread (the iteration is
  monotone, not oscillating) and the conclusion was wrong: the damping
  itself was causing the stall.
- R and browser ports carry the rank-one handover at 80 and the
  inverse's safeguards (#153): the node-aware scale, the top-two gate,
  the adaptive damping and the pair closed form, each marked "matching
  python". A test reads the handover threshold out of all three files
  and asserts they agree, and that the port files carry the safeguards.
- `hermite_nodes` builds the product rule one dimension at a time and
  prunes as it grows (#155): a partial product that cannot reach the
  threshold whatever the remaining factors contribute is dropped then,
  so the k-fold tensor is never materialised. Q=41 at rank 5 was 116M
  nodes and ~9 GiB before pruning; it is now 0.2 s and 1.06M nodes, and
  the kept set, its order and its weights are exactly the tensor's
  pruned once (tested at ranks 2-4). The factor_ghk experiment's fixed
  Gauss-Hermite orders are now chosen by rank to keep the pruned tensor
  under ~3e5 nodes.
- The `cov=` degeneration warning names a recipe that runs (#158): it
  recommended `winning.methods.qmc_ghk` (not an attribute) with
  `V=chol(cov)` and no `D` (TypeError). It now says the forward normal
  race and `abilities_from_race` route the case automatically, which
  calls cannot (slopes, ordered prefixes, a temperature, a non-normal
  base), and gives `get_method('qmc_ghk')(-mu, cholesky(cov), zeros(n))`
  for win probabilities alone; a test executes the recipe out of the
  warning text.
- The `winning.thurstone` tombstone and the `winning.research`
  deprecation say that `calibrate_abilities` is the NORMAL race by
  default, name `base=` for the other built-in races, and say that a
  custom research `Density` has no front-door equivalent and calibrates
  a different model (#126). The dangling changelog bullet left by
  #150's rebase is folded (#127).

- `abilities_from_race` converges on small-scale fields, and the normal
  pair is now closed form. Two names with near-identical high loadings
  (b = 0.99, `D = 1 - b^2`, one contrast with sd 0.199) came back with
  gap 1.2755 for an exact 0.1679 and a forward map of [1, 0] after 60
  non-converging sweeps; b = 0.80 and 0.95 were exact. Measured, the
  same divergence appeared at n = 2 and n = 3 with `D` = 0.005 / 0.001
  and no loadings at all, while n >= 4 held -- so it was scale, not the
  pair: the warm start `-(log p - mean)/2` and the step cap of 2 were
  written in unit-variance units, and on a field with sd 0.1-0.2 that
  start is many sd off, the forward saturates, and a capped step is
  10-20 sd. Both now scale with the field's contrast sd
  (`sqrt(median(D) + mean ||V_i - mean||^2)`): every failing case
  converges in 12-21 sweeps, and fields near unit scale are untouched
  (the pinned sweep counts did not move). Separately, the normal pair
  with no temperature takes the inverse closed form, mirroring the
  forward one from #83: `mu1 - mu0 = sd_d Phi^-1(p0)`, exact, zero
  sweeps. Handed over by a client whose two-name factor race hit it;
  #149's effective-pair case was already fixed by #150.
- `race_probabilities(cov=)` is now correct where it was warned. When the
  grammar fit is degraded -- either failure class: it reproduces `cov`
  badly (the residual warnings), or it reproduces `cov` by degenerating
  to full rank with `D` on its floor (the #118 case, which the residual
  checks cannot see) -- the forward normal race with no slopes is routed
  to scrambled-Sobol GHK (`winning.methods.qmc_ghk`, 1024 nodes, fixed
  seed) instead of the fitted lattice. Measured against 4M-path Monte
  Carlo on dense correlations: 4.7e-4 at n=16 and 5.0e-4 at n=30 where
  the fitted lattice was 8.9e-3 and 6.5e-3, in a tenth of the time,
  deterministic, and smooth in its inputs (a common shift moves it by
  3e-17). No amount of extra fitting closed that gap -- more factors send
  `D` to the floor and the quadrature error returns; a fractional `D`
  floor bottoms out at ~2e-3. A healthy fit (exact rank 3 at n=40) is
  untouched and still equals the V=/D= lattice answer. Calls that need
  the factor form -- `return_slopes=True`, a non-normal base, a
  temperature, `abilities_from_race`, `ordered_probabilities` -- keep the
  fit with its warnings; `abilities_from_race(cov=)` used to call
  `fit_covariance` directly and warned on nothing, and now warns like the
  rest. The #118 pins flip from strict xfail to passing at GHK's
  accuracy; the mechanism test stays, so the routing stays necessary.

- The shape sweep now covers `n == 2` as well as `n == 5`. Ten factor
  verbs that accept a pair get the same three contract checks, which
  matters because the pair is where the arguments are most confusable: at
  rank one `V` is `(2, 1)` and `D`, `mu` and the belief variance are all
  `(2,)`. When the two-runner closed form landed, its spellings were
  checked BY HAND, which is what this file exists to replace.

  What the sweep actually guards is narrower than it first appears, and
  worth stating: sabotaging the closed form to read `V` with
  `np.atleast_2d` -- the original #66 bug, planted in the new code --
  does NOT fail the sweep, because `_setup` normalises `V` upstream and
  the call is a no-op on an already-`(2, rank)` array. The real risk is
  an `n == 2` fast path added ABOVE `_setup`, bypassing the contract to
  skip its cost; sabotaging THAT fails three checks at once.

  `removal_shares` is deliberately absent from the pair set: removing one
  of two runners leaves one who wins with probability 1, so the result is
  the constant permutation matrix and the loadings cannot enter it. The
  loadings-move-the-answer guard found that rather than it being assumed.
- Regression test for the `ordered_probabilities` points floor reported
  in #66: a negative-correlation block (regular simplex, rank k-1) at
  k=3 prefixes succeeds at `points=501` and raises below it for every
  k = 2..5. The test also pins WHY lowering `points` did not help: a
  request below `need = ceil(span / (sd_min/8)) + 1` is silently
  promoted to `need` (184/216/258/320 for k = 2..5), so 65 and 257
  points give bit-identical results at k >= 4. `need` was calibrated
  for the win race's spectral lattice; the 3-prefix kernel's inner
  cumsum is first-order and 8 points per sd leaves 2-5e-3 of mass
  against a 1e-3 tolerance. No code change; the test names the fix.
- A `D` entry of exactly zero reached the lattice and died as
  `OverflowError: cannot convert float infinity to integer` in
  `forward_grid`'s grid sizing, which divides by the smallest sd. A zero
  variance is legal as a belief (`as_variance`: a perfectly known
  quantity) but not as performance noise on a lattice, so `as_idio` gains
  `positive=True`, the lattice kernels (`_setup`, `win_probabilities_
  factor`, `jacobian_vector_product`, `abilities_from_probabilities_
  factor`) use it, and the error now says what the zero is and what to
  do.
- `cov=` warns on the case its residual checks cannot see. Those checks
  judge how well `V V' + D` reproduces `cov`, and they do fire on a dense
  correlation the grammar fits badly (dense n=16: projected residual
  8.3e-2). What they miss is the opposite failure: the fit reproduces
  `cov` to ~1e-6 by going to full rank with `D` on its floor, and the
  conditional race is then a near-step the factor nodes cannot resolve
  -- an exactly 3-factor correlation at n=8 prices 6.6e-3 off its own
  exact V/D answer this way (#118). The new warning keys on the fit's
  shape (full rank, or the floor bound) and names the alternative. Pushing the fit further does not help -- more factors
  drive `D` to the floor and the quadrature error returns; a fractional
  `D` floor bottoms out at ~2e-3. `winning.methods.qmc_ghk` on the same
  inputs is 4.7e-4 / 5.0e-4 at 1024 nodes in 24 / 67 ms, deterministic
  and smooth in its inputs (a common shift moves it 3e-17), against the
  fitted lattice's 8.9e-3 / 6.5e-3 in 219 / 495 ms; the warning names
  it. Routing `cov=` to it on degeneration is the natural next step and
  is not done here. An exactly low-rank `cov` at n >= 30 fits at its
  true rank with `D` healthy and does not warn.
- The rank-1 node rule hands over from Gauss-Hermite to the midpoint-
  quantile grid at Q > 80 (sharpness ~10) instead of Q > 201 (sharpness
  ~25). Swept on an 8-runner rank-1 field against 3M-path truth (se
  ~3e-4): GH at its scaled order was 7.3e-4 at sharpness 6.7 and 8.3e-4
  at 10.5, then 2.8e-3 at 14.9 (Q = 120) and 4.0e-3 at 21.1 (Q = 169)
  -- erratic past Q ~ 100 rather than slowly worsening (Q = 201 gives
  4e-5 at 14.9 and 2.4e-3 at 21.1) -- while the midpoint grid at the
  same Q sat at the truth floor (3.3-3.5e-4) throughout, and is also at
  least as good as GH below the threshold (6.1e-4 vs 9.0e-4 at 6.7). So
  the default rule was ten times LESS accurate at sharpness 21 than at
  47, because only the sharper field had escalated. Found while
  prototyping GHK-style importance sampling in factor space, which at
  rank one does not beat the grid: a single truncation removes little,
  and per-winner draws multiply the lattice passes by n.
  `tests/test_rank1_node_handover.py` pins the former gap at the floor
  and GH's regime below it.

- `abilities_from_race` now converges when two runners hold nearly all
  the mass, at any field size. The solver already damped the N = 2 case
  (alpha 0.7) because the K_2 Jacobi update has eigenvalue -1 and
  two-cycles -- but keyed on N == 2 literally, so an EFFECTIVE pair at
  any N got alpha = 1 and two-cycled the same way: with two dominant
  runners and the rest at 1e-4 the residual after 60 sweeps was 5.8e-2
  at N = 3, 7.5e-3 at N = 10 and 1.1e-2 at N = 30, 500 sweeps was worse
  (oscillation), and under the skew-normal base a 3-player field with
  one missing price came back with the two favourites SWAPPED while
  only warning. Three substantive runners always converged. The damping
  is now keyed on the target's top-two share (> 0.8): the failing cases
  converge in 19-20 sweeps to ~4e-9, and every field below the
  threshold takes exactly the sweeps it took before -- "0.7 always"
  would have tripled the sweeps on a 150-runner field. Measured before
  choosing: four damping rules on the failing, degrading, ordinary and
  large fields. `tests/test_inverse_effective_pair.py` pins the fixed
  cases, the favourites' order, the skew round-trip, the untouched
  sweep counts below the threshold, and that three or more contenders
  still converge. Found migrating a client whose late-contest fields are
  exactly this regime.

- `winning.thurstone` now fails honestly: importing it raises an
  `ImportError` that names `winning.research` (where the code went),
  `calibrate_abilities` (what to prefer, and why) and the external
  `thurstone` package (the usual route in), instead of a bare
  `No module named 'winning.thurstone'`. It was a deprecation alias for
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
