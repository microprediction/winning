# Claim-to-script manifest

Every empirical claim in main.tex, mapped to the seeded script or test
that produces it. The title footnote promises this; this file is the
index. Paths are repo-relative.

## Beside the paper (this directory)

| Claim | Source |
|---|---|
| GHK table: 200x / 5x at n=200, times and TV | `bench.py ghk` |
| GHK cost law n^2.8 points | `bench.py law` |
| Scale: 0.18 s / 2.7 s / 29 s at 1e4..1e6 | `bench.py scale` |
| Against-the-field table (Genz, frequency, ME, Clark) | `bench.py alt` |
| Genz n=200 single probability: 0.43 s at maxpts=250k, agreement to 4-5 digits (six digits needs multi-second budgets) | `bench.py genz200` (pinned 2026-08-27; supersedes an earlier 4.6 s / six-digit note taken at a different tolerance) |
| Referee agreement (mvtnorm within own bound; Botev 2.2e-4 at 3e-10; quadrature 1.9e-15; invariance battery) | `referee_cases.py` + `referee.R` + `referee_check.py` |
| Stress boundaries (tails to 1e-119, round-trip 1e-9, Gumbel=softmax 1e-15, duplicates, JVP 8e-12, sharpness escalation) | `break.py` |

## Elsewhere in the repo

| Claim | Source |
|---|---|
| Bulk window: 33 points beat 500, span floor 6e-11, 100x on hopeless fields | `research/lattice_window/run_window_savings.py` |
| Fixed 15-node rule silently loses up to 5% (sharp factors) | `research/fuzz/fuzz_races.py` (+ logs) |
| Sharp-factor family escalation pinned | `tests/test_sharp_factor_escalation.py` |
| GH tensor beats Sobol at rank 2, wide margin | `research/experiments/exp24_factor_rqmc/` |
| Depth-one tree reproduces block race; common root invisible | `tests/test_race_invariants.py` |
| Tree vs 2^22 CRN referee (RMS z 0.75-1.1, depths 1-4); Botev 0.3% to 7e-8 | `research/tree_exactness/run_exactness.py` |
| HRP cophenetic identity (floored) | `tests/test_race_invariants.py::test_cophenetic_tree_race_identity`, `research/hrp_finish/run_finish.py` |
| Jacobian off-diagonal symmetry/signs, zero row sums | `tests/test_race_invariants.py` |
| Inversion to 1e-8; sub-resolution floored | `break.py` section F; `research/experiments/exp35_independent_inversion/` (methodologically independent reference) |
| Ensemble accuracy table (1-4e-5 on 12/15; 1-4e-4 dense-strong; 4e-3 Matern/RBF) | `research/general_sigma/run_ensembles.py`, `run_ensembles2.py` (+ logs) |
| Dense n=2000: 5 s vs 36 s, 82% zero-win tails priced | `research/general_sigma/log_large_n*.txt` (script: `run_cv.py` family) |
| Estimation table (exact MLE 0.0187/4.9 s vs MSL) | `research/mnp_estimation/run_mle.py` |
| Parity: 22 scenarios, four languages | `r/winning/tests/testthat/` golden files + counterparts per port |

## Gaps: CLOSED (2026-08-27)

1. Rust scale: `bench.py tenmillion` committed and run on a quiet
   machine: 244.9 s at n=1e7, 1e4 clusters, 257-point lattice
   (streaming field). The research-note 64 s figure was a different
   configuration; the paper now cites the pinned 245 s.
2. Mislocation: `bench.py mislocate` pins it. Measured: max 2.49 vs
   spread 8.24 = 30.2% (median 0.34); paper updated from "a fifth" to
   "three tenths" with the configuration stated.
3. Genz n=200 single probability: `bench.py genz200` pins it. Measured:
   0.43 s per probability at maxpts=250k with estimated error 2e-4
   (agreement with the lattice to ~4-5 digits); the lattice prices all
   200 in 25 ms. Six-digit agreement needs multi-second maxpts budgets.

## Ninth review (referee 3), pinned measurements (2026-08-28)

1. Identified-objective fit: `factor_model_projected` minimizes
   min ||P(Sigma - VV' - diag D)P||_F (ALS; D-step collapsed to its
   n-dim normal equations, Gram exactly P∘P — 67 s → 0.27 s at n=300,
   identical minimizer to 1e-14). Packaged one-call:
   `fit_covariance` / `race_probabilities(cov=)`. In-grammar truths
   return at the 2e-4 node-noise floor (the raw-residual pipeline
   scored 1.7e-2 — the bug the ninth review's point 5 flushed out).
2. Multi-seed ensemble battery: `research/general_sigma/run_ensembles4.py`,
   20 seeds per ensemble, n=300, 1M-draw MC referee per seed, both arms
   (raw eigenfit pipeline vs identified pipeline), kernel stratified
   RBF/Matérn-3/2 × length scale {0.08, 0.2, 0.4} × promoted rank
   {5, 12}. randomcov pinned at commit 0d27a51.
3. Per-grammar inversion round trips: `bench.py invert`, n=400.
   independent 0.04 s, tree 0.37 s, blocks 0.67 s, factor r2 1.7 s,
   nested 3.7 s; max log-probability residual < 1e-8 in every grammar
   through the shipped `abilities_from_race(structure=)`.
4. Per-alternative factor-quadrature comparator (Butler–Moffitt run
   per alternative): `bench.py bm`, n=200 rank 2. Agrees with the
   shared field to TV 3e-15 and max log-ratio 3e-10; costs 21.9 s
   against 36 ms — 601×. Same conditioning, no shared field.
5. Cross-language fitter parity: R `fit_covariance` matches python's
   fitted model covariance VV'+diag(D) to 3e-15 on a mixed factor/AR
   test matrix (priced probabilities differ TV 2e-3: Halton vs Sobol
   nodes, by design).
6. Kernel stratification (final, two-arm fitter): med TV over 20 seeds —
   RBF ls 0.08/0.2/0.4 at m=5: 3.4/2.6/1.6e-2 (m=12: 3.1/2.6/1.7e-2;
   m=5 @2^14 nodes: 3.3/1.2/0.82e-2); Matérn-3/2: 2.9/2.5/2.1e-2
   (m=12: 2.5/2.3/1.7e-2; @2^14: 2.4/1.6/1.2e-2). Two regimes: short
   scale = representation (rank helps, nodes don't), long scale =
   quadrature (nodes help, rank doesn't; 2^16 reaches 6e-3 on the
   diagnostic draw). The old kernel row (up to 7e-2) was greedy rank
   misallocation, repaired by the eigen arm (stage vi).
7. n=2000 one-call: fit 0.6 s + price 3.6 s (fit was 41 s before the
   double-centering/subspace/water-filling rework, commit db111e3;
   NNLS closed form verified to 1e-14 vs scipy).

## Inversion at a million (2026-08-29)

`bench.py invertmillion`. The paper's headline inversion benchmark is
N=1e4 "in under a minute"; that was the table's limit, not the
method's. Measured on one laptop:

| n | structure | forward | inversion | max abs mu error |
|---|---|---|---|---|
| 1e5 | independent | 0.7 s | 6.8 s | 1.2e-11 |
| 1e5 | factor rank-1 | 3.4 s | 111.6 s | 2.8e-09 |
| 1e6 | independent | 7.7 s | 80.3 s | 7.5e-11 |
| 1e6 | factor rank-1 | 31.2 s | 1319.9 s | 2.6e-09 |

So the per-operation scale statement is now: forward block field 1e7,
forward rank-one factor 1e6, INVERSION 1e6 (80 s independent, 22 min
rank-one factor), full rank-two factor inversion 1e4 in under a minute.
Dense covariance remains a separate claim because it must first be
fitted into a scalable grammar.

## Fourth review, adjudicated (2026-08-30)

Ran their counterexamples before believing them:

1. **Heterogeneous-scale standardization (their must-fix 1): right about
   the paper, wrong about the code.** The shipped `fit_covariance` fits
   covariance coordinates directly (their recommended repair), and their
   n=2 example passes to 9e-9 through `cov=`. The paper's Section 6 TEXT
   described a standardize-then-restore pipeline the code never
   performed -- and which their counterexample correctly kills (the
   choice-irrelevant direction in correlation coordinates is S^-1 1,
   not 1). Text rewritten to match the code, with their refutation kept
   in the paper as the reason. Their example also exposed a genuine n=2
   division by zero in the water-filling diagonal solve; guarded, with
   regression tests (analytic binary probability; exact-grammar round
   trip with variances over two decades).
2. **Removal grid (their must-fix 2): right in principle, aimed at the
   wrong window.** `removal_shares` uses the SPAN window, not the
   winner bulk, and passes their dominant-contestant case exactly at 20
   and 200 sigma. Hardened anyway: spacing now refines to the sharpest
   runner, and unnormalized row masses (fixed at one by the continuum
   identity) are checked, raising on defect instead of renormalizing it
   away. Abstract claim qualified per their suggested wording; the
   second-finisher window is the one-failure term of the SIAM paper's
   multiplicity union calculus (Peter's pointer), noted in the
   docstring.
3. Superseded Genz figures found still in the paper (4.6 s / six
   digits) and replaced by the pinned 0.43 s / 4-5 digits; full-vector
   factor corrected from 3e4 to 3e3.
4. TV definition matched to what run_ensembles4.py computes:
   full-vector TV against empirical frequencies (zeros included, so the
   referee's own noise bounds it from above); per-entry column
   restricted to >= 25 wins.
5. Prose: sign-convention warning at Section 3; HRP remark now states
   the zero-root normalization and the contrast-equivalence vs raw
   correlation distinction; "certified/minimize" softened to monotone
   alternation to a coordinatewise-stationary point; frozen-vs-moving
   grid FD statement made explicit; coercivity made uniform via
   compactness of the unit sphere; zero-target statement moved to
   contrasts; solver explicitly disclaimed from the theorem; tree
   sibling-cancellation sentence added; "GHK replaced" replaced by
   their per-alternative-repetition wording; 63-percent premium
   demoted to a sign-stable diagnostic at 8 replications.
6. Also pinned from the discussion: within-cell tie mass lives in the
   quadrature error and decays SPECTRALLY (5.6e-3 at 9 points, machine
   zero by 33, asymmetric near-tied fields) -- why the factor engine
   carries no multiplicity bookkeeping while the classic integer-lattice
   engine, whose dead-heat mass is real and fixed, rightly does.

## Fifth review, adjudicated (2026-08-30)

Every claim checked by running it. Four hit real defects in the code:

1. **Jacobian normalization quotient (their blocker): CONFIRMED.** The
   grid JVP returned the derivative of the UNNORMALIZED rectangle sum;
   the quotient its own docstring described was never applied. Measured
   against finite differences of the returned map: 2.06e-3 at L=25.
   Now applied (`normalized=True` default, `normalized=False` for the
   raw form): 3.1e-11 at L>=101. A residual 6.2e-4 remains at L=25
   because the JVP's own window moves with mu -- a grid-motion term, now
   stated in the paper rather than claimed away.
2. **Saturated MLE has no finite maximizer: CONFIRMED, exactly.** min p
   = 2.534e-5, expected count 1.27, and seeds 103 and 105 do have an
   empty cell. run_mle2.py now adds a Jeffreys pseudocount (counts +
   1/2, ALPHA configurable, ALPHA=0 reproduces the old unpenalized run)
   and the paper calls the estimand a MAP maximizer. Rerun numbers are
   BETTER, not worse: exact 0.0148 in 2.4 s against MSL-100 0.0275 in
   3.5 s and MSL-1000 0.0161 in 39.9 s.
3. **Bulk window hardcoded the normal survival: CONFIRMED.** Heavy tails
   were clipped -- Student-t(2.5) at n=40 lost 5.1e-3 of total variation
   against the span window. `_bulk_window` now takes the caller's base
   survival and honors a base's declared span: 5.2e-5 after (Laplace
   2.0e-6, normal/Gumbel unchanged at 1e-16).
4. **hermite_nodes did not renormalize after pruning: CONFIRMED.**
   Weights summed to 1 - 2.4e-8 (k=2) and 1 - 3.1e-7 (k=3). Forward
   races renormalize downstream and were unaffected, but direct weighted
   mixtures consume W as-is. Now renormalized.

Prose and scope corrections from the same review: abstract now scopes
matrix-free JVPs and removals to the factor grammar and calls the
removal table Omega(n^2); the removal passage carries their own
mu=(-20,0,1) counterexample (3e-28 of unnormalized mass on the original
window) and states that removals integrate on their own span window;
the node policy is described as it actually is (tensor GH under 1e5
nodes, Sobol when sharp, midpoint-quantile at extreme rank one).
Package bumped to 1.3.0 with python_requires >=3.10 and corrected
classifiers (3.7/3.9 were advertised while the code uses `X | None`).

## Sixth review (2026-08-30)

Both indispensable items fixed in code rather than by softening the text.

1. **"Exact gradient" was false at the numerical-map level.** Confirmed
   and repaired. `race_jacobian`/`race_jacobian_row` built their own
   ability-span lattice while probabilities came from the adaptive bulk
   window, imposed the diagonal from the continuum zero-row-sum
   identity, and skipped the normalization projection with a comment
   calling it second order. Now: the lattice comes from the shared
   `races.forward_grid`, the own coordinate is integrated
   (`-f'_i/sd^2`), and the quotient rule is applied,
   `J = (A - p 1'A)/T`. Verified against central differences of
   `race_probabilities` itself: 2e-11 relative at 65 points (surrogate
   1e-8) and 1e-6 under student-4 (surrogate 1e-5). Identical on a fine
   Gaussian lattice, which is why five reviews missed it. Row sums now
   vanish by quadrature (7e-17 normal, 5e-11 gumbel) rather than by
   construction.

2. **Jeffreys/MAP description was wrong.** Confirmed. Dirichlet(1/2) has
   posterior exponents c_i - 1/2, whose mode is boundary-seeking at
   empty cells; add-half is the Jeffreys posterior MEAN, or the
   Dirichlet(3/2) mode read as a penalized likelihood. Text corrected,
   table row renamed to "exact-gradient smoothed inversion".

Smaller items, all confirmed and closed: the inversion contract reached
only Independent/Factor (now validated once before the split, one exit
for all five grammars, true iteration counts); the bulk window bisected
an unbracketed interval (now bracketed geometrically, with delta
relaxed and reported when the point budget cannot resolve the requested
quantile -- honoring 1e-12 literally under student-4 costs 0.071 TV
against a 4e-3 tolerance, so the relaxation is the accurate choice, not
a concession); `winning.probit.removal_shares` bypassed the safe path;
Blocks/Nested/Tree silently dropped base=/temperature=/return_slopes=;
the introduction's matrix-free claim was broader than the abstract's;
`from_linkage` retains the root (paper had it backwards); the flux
proof needed C^1-with-decay rather than integrability; the coercivity
bound is F(mu) <= -kappa||mu||, not W(0) - kappa||mu||.

### Open: the saturated table's timing column

The RMSE column is CONFIRMED unchanged after the Jacobian repair --
re-running `research/mnp_estimation/run_mle2.py` on the new code gives
0.0148 / 0.0275 / 0.0161, identical to the published table.

The timing column is stale and needs re-measurement on a quiet machine.
The new Jacobian does strictly more arithmetic (the fp own-derivative
term and the total-mass derivative), measured at 1.37x the old
surrogate's cost interleaved back-to-back at n=30, r=2, 257 points. The
re-run's wall clock (6.5 / 9.1 / 69.3 s against the published 2.4 / 3.5
/ 39.9) cannot be attributed: it was taken under load average 10.7 with
five other CPU-saturating processes, and the MSL arms, which never
touch the Jacobian, slowed by 2.6x and 1.7x. The three arms remain
internally consistent within each run, so the paper's comparative claim
(the exact path is both faster and more accurate than MSL at either
replication count) holds in both. Re-run the script on an idle machine
and replace the column before submission.

### Closed since: three of the four disclosed gaps

- **Replications.** The saturated study ran at eight, unpaired. Now
  forty, paired (every arm sees the same draw). Exact beats MSL R=100 by
  0.0112 +/- 0.0006 (18.7 SE, 39/40) and R=1000 by 0.0017 +/- 0.0002
  (6.9 SE, 33/40). Point estimates moved slightly upward
  (0.0148/0.0275/0.0161 -> 0.0168/0.0286/0.0184): the small sample was
  mildly favourable, the ordering was never in doubt. Empty cells are 13
  of 40 against the 11.2 the paper's own rate predicts.

- **Multistart dispersion**, offered as a diagnostic and never reported.
  Now measured, and the obvious version of it is the wrong one. Across
  four ensembles at n=300 with eight random starts, every start reaches
  the same objective to 4e-16 and none beats the shipped start. But on
  block equicorrelation the starts agree on the objective AND on D to
  every digit while pricing races that differ by TV 0.25: centering a
  six-block matrix leaves a FIVE-fold tied eigenvalue, so a rank-three
  fit picks an arbitrary three-of-five subspace, every pick exactly
  optimal, each implying a different race. At the multiplicity the fit
  is exact and the spread collapses to 8e-4. Diagnostic: disperse the
  priced RACE, not the objective. Remedy: rank, not optimization.

- **Zero-win tail at n=2000**, disclosed as untested. Sample splitting
  now run: model mass 1.13e-4 on a set fixed by the first million paths,
  111 wins from an independent second million, 95% CI
  [9.13e-5, 1.34e-4], model inside. Bulk scored on the second half alone
  at TV 4.4e-3 against a 5.4e-3 split-half noise floor.

Still open: the timing column (see above) and, newly, whether the
degeneracy finding deserves a rank-selection rule rather than a
diagnostic. The fitter picks k=3 by default; on a matrix whose centered
spectrum has a tied leading block, that default is silently arbitrary.

## Seventh review (2026-08-31)

All five leads confirmed and fixed; none touched the central results,
which the review re-checked and passed (flux sign/geometry, uniform
coercivity, projected objective, n=2 counterexample, removal grid,
sample splitting).

1. **Eq (2) under correlation** — confirmed. The marginal-survival
   product was asserted before independence was assumed. Now: the
   conditional identity first, factorization as the independent case,
   grammars as conditioning routes to the same product.
2. **Window envelope endpoints** — confirmed as a PAPER error only. The
   code always used mu_lo for the lower endpoint and mu_hi for the
   upper (races.py G/H functions); the prose said favourable node at
   both ends, which is not an envelope. Prose fixed to match code.
3. **"Exact derivative of the adaptive map"** — confirmed. The
   implementation returns J_grid, exact conditional on the selected
   grid, and omits grid motion. Taxonomy is now four objects and the
   L=25 residual of 6e-4 is labelled as the grid-motion omission.
4. **Row/column conflation** — already fixed in v9 (columns exact by
   construction, rows quadrature-limited); the review read v7.
5. **Sharpness not choice-invariant** — confirmed by running the
   counterexample: identical loading rows give stated s = 30.0,
   centered s = 0.0, and the factor moves probabilities by 2.8e-13.
   Paper now defines s_c = max ||(PV)_i||/sqrt(D_i) as the
   characterizing statistic and keeps raw s as the conservative
   trigger (false positives cost nodes, not error). CODE still
   triggers on raw s — acceptable per the review, logged as a
   possible improvement (centered trigger would avoid needless
   escalation on common-loading factors).
6. **Tree transpose** — v_i' a_c(i); loading gloss added.
7. **Root contradiction** — pricing gauge-fixes the root; reporting
   constructors may retain it; from_linkage is the latter.
8. **Theorem 1 scope** — Gaussian where stated; generalization
   conditions stated at the invocation, read as such.
9. **Flooring** — zero-target case proved; coordinatewise bounds for
   small positive targets explicitly need a monotonicity argument not
   made.
10. **Parity bug** — confirmed: the tied-eigenvalue warning postdates
    paper-r3 (added in 36f7d6d). Footnote now names paper-r4, cut at
    the commit containing everything the paper describes.
11. **Empty cells** — measured rather than argued: all 13 empty-cell
    replications are the smallest cell (index 29) and no other cell
    ever empties, so the 40 x 0.28 = 11.2 expectation compares to the
    right event.

Deferred as improvements, not defects: centered sharpness as the CODE
trigger; alphabetizing the bibliography; generalizing Theorem 1
formally rather than stating the conditions.

## Eighth and ninth reviews (2026-08-31)

Two genuine blockers, both verified by running them, both fixed in code.

1. **The s >= s_c inequality was false and the dispatcher could miss a
   sharp race.** P contracts Frobenius norms, not maximum row norms.
   Verified on both counterexamples: the 4-runner rank-2 case (raw 2.977
   vs centered 4.426; shipped TV 9.5e-3 against a 20M-draw referee, and
   the choice-irrelevant shift V -> V + 1c' moved the answer by the same
   9.5e-3) and the 3-runner case with an analytic orthant referee
   (p_i = 1/4 + arcsin(rho_i)/2pi; raw 2.90, pairwise 4.10, shipped TV
   6.1e-3). Fix: V is gauge-fixed to PV at every entry point (races
   _setup, core forward + JVP, likelihood; mirrored in the R and JS
   ports), and the dispatcher triggers on sqrt(2) max|(PV)_i|/sqrt(D_i),
   which bounds the pairwise sharpness above by the triangle inequality
   and so cannot false-negative. After: 4-runner TV 3e-5 (referee noise),
   gauge shift bit-identical zero, 3-runner TV 2.2e-7. Parity vectors
   regenerated; R, JS and JS-local checks green.

2. **The absolute D-floor broke pass-through and the residual
   diagnostic missed a 0.48 choice error.** diag(1e-8,1e-8,1) with
   mu=(0,0.001,10): floor 1e-3 x mean variance forced Var(X1-X2) from
   2e-8 to 6.7e-4, head-to-head 1.0 -> 0.515, zero warnings, global
   residual 100x under threshold. Fixes: the closing floor is relative,
   l_i = 1e-6 Sigma_ii (water-filling unchanged, vector shift); the fit
   report carries max_{ij} |d'Rd|/(d'Sigma d) over pair contrasts d,
   which reads 0.49 on the failure and 5e-9 after the fix, and cov=
   warns on it; and fixing this exposed a THIRD defect the example was
   masking -- the alternation's default start stalls on exactly
   in-grammar small-n spread-variance matrices (objective 0.88 where 0
   attainable), now rescued by two further deterministic starts
   (diagonal-heavy, eigen-residual), each reaching 1e-13.

Also actioned: theorem cites (tiecorr) with independent reduction;
boundary-target claim now proved (restriction + dominated convergence +
compactness converse); Case V history corrected (zero correlations
originally; Mosteller relaxed to common); "sits exactly in the null
space" corrected to the common-correlation component; window envelope
scoped to retained nodes and achieved delta; "fixed-grid exact"
propagated to docstrings, table labels and site; parity claim replaced
by the measured tolerances (19 of 22 within 1e-7, three fit-mediated at
5e-4..5e-3 -- BOTH prior public claims, "20 at machine precision" and
"17 at machine precision", were wrong); homepage example now runs
(structures and polish_race exported top-level); site scope fixes
(matrix-free = factor grammar, million-scale timings split by model,
floored cophenetic, photo-finish densities not exact dead heats,
Gaussian-only hierarchical kernels); versioned PDF filename adopted.

Deployment blocker reported against the live PDF was already resolved
by the time of checking (the r4 Pages deploy was cancelled as
superseded and the next push redeployed); versioned filename guards it.

Open: the cottonsurvey bibliography entry has no SSRN number or stable
locator -- only Peter can supply it (or relabel as unpublished
manuscript).

## Tenth review (2026-08-31): the novelty reframe

Fresh prior-art search, accepted in full and written in.

- **Domencich & McFadden (1975), p. 52**: the general random-utility
  choice probability as a one-dimensional integral over the winning
  utility. Cited at the crux sentence; one-dimensionality is no longer
  implied to be new anywhere.
- **Muller, Nesterov & Shikhman, MOR 47(1):485-507, 2022 (arXiv
  1909.05591, 2019)** -- VERIFIED against the journal listing before
  citing: cross-Hessian of the surplus = tie-for-maximum density, with
  the zero-row-sum symmetric structure, for general joint laws. Cited at
  the head of the Jacobian section, with the independent-arrival
  statement (we were unaware of it when this work was done; same
  structure, different route, and the boundary-flux route is what the
  paper keeps).
- **The claim that survives**: not one-dimensionality, not convex
  duality, not the existence of tie densities -- the cavity organization
  that composes them at scale. New "to be plain about what is and is not
  claimed" paragraph in the introduction names all four lineages
  (domencich1975, cotton2021, muller2022, li2018) and states the
  contribution as: forward vector O(nL) via the field (Cotton 2021), the
  SAME cancellation collapsing the derivatives (this paper), surviving
  correlation by conditioning, feeding a matrix-free convex program at
  n = 1e6.
- **The Gram identity elevated** to its own displayed passage
  (sec:jaccomp): w_ij = int G A_i A_j, hence
  (Jh)_i = int G A_i [sum_j A_j h_j - h_i sum_j A_j] with the inner sums
  shared across i -- the reviewer's candidate for the real leap, now
  stated as the identity the scale rests on rather than a parenthesis
  about "Gram structure".
- Methodology page: same lineage, same identity, and the flagged
  sentence ("the shared field underneath it is the new part") replaced
  by the defensible composition claim.
- The 2001 ZEW observation (independence -> one-dimensional even for
  large n) was NOT cited: the reviewer did not name authors and a
  search-verified record was not obtained. Domencich-McFadden carries
  the point.

Abstract originality scores from the review, for the record: ~2/5 new
foundational mathematics, ~4.5/5 numerical/computational methodology,
4/5 overall. The reframe matches that assessment.

### Eleventh round: the reviewer read the papers

Refinements to the reframe, all applied. McFadden 1975 credit made
precise: the GENERAL 1-D form is a representation whose integrand
conceals a multivariate calculation (their own caveat below eq. 4.26);
the genuine 1-D quadrature is the INDEPENDENT case (their p. 68
independent Cauchy example has exactly the density-times-product form).
fosgerau2013 (Choice probability generating functions, JoCM 8:1-18,
2013 -- verified) added for the expected-max/gradient/common-level
representation lineage. MNS credit now states their route
(differentiating the gradient identity) and their scope (convexity
moduli for prox-functions; no graph, no algorithm). Methodology page:
the (n-1)-dimensional-orthant sentence now says CORRELATED probit, since
independent probit has had 1-D quadrature for decades; MNS cited
immediately at "the derivative is the tie density" with the
independent-arrival note, and the later duplicate credit trimmed.
Reviewer's final scores: ~1.5-2/5 new mathematics, ~4.5/5 algorithmic,
with the defensible claim being the shared rank-one/Gram exploitation
making p and Jh both O(nL) and inversion matrix-free at scale -- which
is now the paper's stated claim, verbatim in spirit.

### MNS read against source (arXiv 1909.05591v1, in docs/assets/literature)

Verified page by page. Eq (1) is the gradient identity, which THEY
attribute to Williams-Daly-Zachary [their refs 9, 10] -- now named at
our Danskin step. Page 7: the mixed partial interpreted as the density
that i and j both attain the maximum; diagonal as the sum over j != i.
Page 8: the class A (symmetric, a_ii >= 0, a_ij <= 0, zero row sums)
and Hessian membership. Pages 8-10: the structure is used solely to
bound ||Hessian||_{inf,1} for prox-function convexity moduli; no graph
vocabulary, no algorithm, no shared field anywhere. Our
characterization ("no graph drawn, no algorithm built") is accurate as
written. New distinction added to the paper from the source read:
their cross-derivative remains an (n-2)-fold integral of the joint
density, while the face parametrization reduces the same object to a
ONE-dimensional integral of a pair density against the conditional
survival -- the form a lattice can price. That is the precise sense in
which the flux derivation is not just prettier but computationally
load-bearing... rather: computationally necessary.

### Domencich-McFadden read against source (ch. 4 scan, docs/assets/literature)

Verified page by page, and the source is stronger for the paper than the
reviewer's summary. Eq (4.7) p. 52 is the one-dimensional form, restated
as (4.26) p. 66. The caveat sits below (4.26), in their words: extremely
difficult to evaluate without numerical multivariate integration, only
the Weibull convenient, multiple-choice probit "computationally
intractable." Their (4.27)-(4.32) is the sequential-conditioning
recursion GHK later simulates, set aside in 1975 as too cumbersome for
iterative estimation -- now cited in the GHK paragraph as the
deterministic ancestor. P. 68 eq (4.38): independent Cauchy in exactly
the density-times-product one-dimensional form, priced per alternative,
"straightforward, but costly." So 1975 contains the representation, the
independent 1-D quadrature, and proto-GHK, all judged impractical --
which is precisely the gap the shared field fills, and the crux passage
now says so with equation numbers and their own words. (Note: the
reviewer's "Eq. 4.26" and "p. 52" both check out -- same formula, two
appearances.)

## Twelfth review (2026-08-31): solver causality, Pearlmutter, competing risks

The main blocker was real and our own experiment proves it: exp23
measures naive Newton-CG with exact grid JVPs at 387 s NOT converged
against Jacobi's 3.7 s converged (n=200), and 2056 vs 19 s (n=1000). So
the demonstrated million-inversion is powered by the field's
own-coordinate slopes, not the Jv oracle. Abstract, contribution
paragraph, the Li paragraph and the Newton-Krylov footnote all now say
exactly that, with exp23's numbers in the footnote as the honest
negative result. The site's section 4 says which oracle does what.

Pearlmutter (Neural Computation 6(1):147-160, 1994 -- verified) added,
and written as a rebuttal rather than a concession, per Peter's
direction that no one else invented this algorithm: the meta-theorem
prices Hv at gradient cost but needs a cheap gradient to differentiate;
applied to the per-alternative evaluators the literature had, it
returns the O(n^2 L) it started with. The missing piece was the object
to differentiate, and that object is the field.

Competing-risks precision: p_i = int G (f_i/S_i) is the classical
cause-probability formula (Kalbfleisch-Prentice 2002, added); Cotton
2021's claim is now stated as the log-domain lattice engine that
evaluates and inverts that classical field for all n at once.

Small: the one-alternative-at-a-time sentence scoped to smooth methods
(frequency simulation returns whole vectors, unsmooth); strict
definiteness conditioned on positive pairwise tie densities / connected
photo-finish graph, with Gaussian-full-support as the guaranteed case.

Deployment complaint checked and NOT reproduced: live page last-modified
matches the latest push and serves the new sections; the reviewer's
fetch predated the honesty-pass pushes by minutes.

Declined, per Peter: the reviewer's suggested novelty paragraph, which
frames the Jv oracle as "may support future solvers" and drops the
where-each-stopped framing. The paper keeps the composition claim with
the missing-insight-per-lineage structure, which the same review's own
table supports (matrix-free Jh from the shared field: "strong
originality candidate"; the pipeline: "strongest originality claim").

## Thirteenth round (2026-08-31): two blockers, both verified by running

1. **Sharp hierarchical fields distorted**: block kernel at cluster
   sharpness 18 measured 5e-2 TV vs a 4M-draw referee ON GROUP SHARES;
   qa 9 -> 31 moves the answer a further 8e-2 (GH fails at any order --
   the factor path's old lesson). Interim: kernels warn past sharpness 3
   and direct to the factor grammar; escalation port tracked in issue
   #11. Paper 4.2 states the limitation with the numbers.
2. **Rank-r advertised, rank-one implemented**: forward kernel DOES
   price rank-r (TV 1e-3 vs MC -- the claim was half wrong); inversion
   crashed in structure_variances on (n,r) loadings, fixed (now
   round-trips exactly); block_race_jacobian is rank-one and now refuses
   cleanly instead of mis-broadcasting. Paper 5.1 scopes the block
   Jacobian to rank one.
Both pinned in tests/test_blocks.py.

## Fourteenth round (2026-08-31): window blocker verified, then the mass check caught a second bug

1. **Hierarchical window blocker (reviewer's counterexample, reproduced
   exactly)**: 400-runner single cluster, loadings 1.0/0.9, D=0.01 --
   the independent-marginal window proxy captured raw mass 0.719 and
   silent normalization returned group shares 0.683/0.317 where symmetry
   forces 0.50/0.50. Fixed with a node-aware envelope (_window_nodes):
   per-runner conditional extremes mu +/- amp bracketed then bisected on
   the idiosyncratic scale. Referee regression now 0.5 exact at 1025
   points, 0.499957 at 257. Ported to R and the JS site engine; all
   parity vectors regenerated, R/JS checks green.
2. **Mass check promoted to error**: raw lattice mass is a diagnostic;
   _checked_mass raises on |mass-1| > 5e-3 instead of normalizing.
   Nested kernels in all three languages now average RAW conditional
   masses and normalize once (per-node normalization hid defects).
3. **The check earned its keep immediately**: it caught a tree
   traversal-order bug present in Python, Rust, R and JS -- message
   passes ordered by accumulated |strength| as a depth proxy, which TIES
   at zero strengths (exactly what Tree.from_linkage produces for merges
   beyond the correlation horizon), visiting children before parents and
   reading unwritten cavities. A 6-leaf zero-strength linkage priced at
   raw mass exactly 3.0; pre-fix output was wrong for any tree with tied
   or non-monotone strength path sums, correct otherwise (why the
   MC validations at random strengths passed). Fixed by ordering on hop
   depth; zero-strength trees now match the independent race to 1e-9+
   (rust/numpy agree to 1e-16).
4. **R polish converged to a different local solution** than the python
   reference on the linkage-tree scenario once tree pricing changed
   (both feasible, R's objective higher, 1e-2 apart in p). The
   projection manifold is nonlinear, so the algorithm PATH selects the
   local solution: replaced R's augmented Lagrangian with the same SQP
   path the python SLSQP takes (Hildreth dual QP per step, from mu0),
   plus fd Jacobian for Tree structures (the analytic tree Jacobian's
   cross-cluster bias steers the iterates). polish_tree_p parity
   restored to 2e-7.
5. **Sharpness defect confirmed independent of the window defect**:
   sharp-blocks TV 0.0495 unchanged under the fixed window; the
   existing warning text and issue #11 stand.
6. Paper edits: abstract scopes distributional generality (non-normal =
   independent/factor; hierarchical kernels Gaussian); 4.x adds the
   hierarchical envelope + mass-check paragraph with the measured
   numbers, including the traversal catch; 5.1 states the hierarchical
   materialized Jacobians are the SECOND object (continuum tie densities
   by quadrature), not the grid derivative; stage-5 substitution written
   with the vector floor (d = ell + x componentwise); Newton--Krylov
   footnote updated with exp23 round 3 (proper assembly converges on
   benchmarks at ~60x wall clock, diverges on the constructed stall
   case) and the convergence-evidence sentence now names the stall case;
   added Kitagawa-Merigot-Thibert JEMS 2019 (damped Newton for
   semi-discrete OT, global convergence, Hessian from boundary mass --
   the nearest proven result for a problem of this shape; both refs
   verified at publisher) and Levy-Mohayaee-von Hausegger MNRAS 2021
   (the method at 1e7 cells). Tag paper-r6.

## Fifteenth round (2026-08-31): the tree grammar was written wider than the shipped algorithm

Peter's blocker, verified by running: the manuscript's tree equation used
the block construction's rank-r leaf term v_i' a_c(i) ("the leaf-cluster
loadings of the block construction"), but the shipped tree kernels are
rank-one -- scalar v_o against a 1-D Gauss-Hermite node. Measured
behavior on a rank-r tree loading: python crashed with a raw broadcast
ValueError; the R port SILENTLY flattened the matrix via as.numeric()
and returned normalized wrong shares; the JS engine produced NaN which
sailed THROUGH the mass check (NaN compares false against any
tolerance). Fixes: (1) paper tree equation rewritten with scalar
v_i a_c(i) and an explicit sentence that the shipped tree grammar is
rank-one where blocks support rank r, with refusal stated; the depth-one
containment scoped to the rank-one block race; (2) clean refusals in all
three languages (python NotImplementedError in both tree kernels and the
inverter's variance surrogate; R stop() in tree forward/jacobian AND in
block_race_jacobian, which had the same silent flatten; JS throw), with
(n,1) columns accepted as the scalars they are; (3) mass checks in all
three languages now test finiteness explicitly. Pinned in
tests/test_blocks.py and the R testthat suite. Tag paper-r7.

## r8 (2026-08-31): one stale fact

The title footnote pinned the revision to "package version 1.3.0" -- but
v1.4.0 (the window/traversal fixes) had shipped and IS the code at
paper-r7, so the provenance claim was internally false. Corrected to
paper-r8 / 1.4.0. No other change. Peter's pass 19 cleared v26 for
publication; v27 (= r8) is the copy to publish.
