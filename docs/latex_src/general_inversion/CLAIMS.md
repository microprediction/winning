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
