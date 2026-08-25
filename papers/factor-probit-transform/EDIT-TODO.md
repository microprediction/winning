# Deferred edits (from author-rewrite session)

## From the 2026-08-24 deep-research review (external)

Wording fixes APPLIED 2026-08-24: "oldest in choice theory" softened +
Daganzo 1977 credited in intro; GHK -> "classic econometric evaluator";
BLP04 J^2 tied to asymptotic normality (consistency weaker); Freyberger
conditional on draw growth; "no probit counterpart" -> "does not
transfer in the same closed form"; Sawtooth corrected to mode-dependent
idiosyncratic term (Gumbel logit / normal probit -- per the VERIFIED
1999 PDF quote, contra the review which claimed always-Gumbel);
"dominant" dropped; PyBLP 4^K2 reattributed to C&G's own accounting,
generic m^K2 stated; abstract 140x relabeled "per-alternative
factor-conditioned RQMC baseline" + hardware qualifier + self-convergence
vs independent-validation distinction; Newton-Krylov global-convergence
claim softened to safeguarded-methods-apply; six-order speedup labeled
back-of-envelope vs plain MC; "simulation owns coarse accuracy" fixed
(EP/tilting/Vecchia cited as high-accuracy per-orthant).

Still open from the review (bigger items, need Peter or new work):
- REPRODUCIBILITY: RESOLVED per Peter 2026-08-24 ("just use whatever is
  the latest at winning package"). The local repo is consistent with
  the paper: setup.py 1.1.0, tag v1.1.0 exists and contains
  winning/factor, README describes winning.thurstone (vendored home) +
  winning.factor. The reviewer saw the stale PUBLIC state -- origin/main
  is behind local. Remaining action: push main + tags publicly (and to
  PyPI) so the public repo matches; git ssh, not gh CLI. Note v1.1.0 is
  226 commits behind local HEAD, so retag or cut v1.1.1 at submission.
- Quote verification: DONE 2026-08-24 -- Torgerson, Conlon slides, and
  both Sawtooth quotes verified verbatim against primary sources, no
  corrections needed (quote-verification.md). Tex comments upgraded to
  VERIFIED. Still unverified: Chiang 1961 and Mukherjea-Stephens 1990
  bibitem TITLES (flagged in the bibliography).
- Independent large-N validation: spot-check 20-50 stratified shares at
  N=1000 against an independent code path (EP / Fasano-Denti, Vecchia /
  Cao-Katzfuss, or heavily replicated scrambled RQMC), three covariance
  regimes (benign rank-2, heterogeneous D near grid stress, small
  photo-finish gap). Report share, log-share, AND post-inversion utility
  error.
- Promote the O(RN) conditional all-N Monte Carlo estimator (Section 7)
  into the headline comparison table, run at N=1000+, with an
  accuracy-vs-wall-time frontier (1e-2..1e-5) rather than one matched
  point.
- Conditioning diagnostics: report lambda_2 / spectral gap + iteration
  count per calibration benchmark; plot Jacobi contraction vs lambda_2
  over the stress suite (turns Prop 4 into a predictive result).
- k>4 factor integration: multiple independent Sobol scrambles in
  research benchmarks to measure integration variability (keep fixed
  seed for the production map).
- Citations to verify then add/elevate: Looi, Loaiza-Maya & Nibbering
  2026 skew-MNP (arXiv ~Aug 11 2026 -- verify it exists); elevate
  Huch-Keane (amortized, cited) and Loaiza-Maya-Nibbering (factor MNP
  estimation, cited) with explicit division-of-labour sentences; check
  Chang-Narita-Saito citation placement (should support cross-menu
  restrictions, not covariance identification); Grieco-Murry-Pinkse-Sagl
  (consumer+product data) and recent nonparametric-identification work
  for the Section 8 estimation agenda.
- Real-data cross-menu validation (calibrate on one menu, predict a
  held-out menu) remains the empirical gap the review and our own notes
  agree on.

## From the 2026-08-24 second (technical) review

FIXED 2026-08-24, code and paper:
- N=2 Jacobi two-cycle CONFIRMED empirically (shipped inversion returned
  silently with log-share errors up to ~1 at mid-range targets; the flat
  oscillating residual never triggered the 20% damping rule). Gaussian
  kernel now returns the closed form mu_2 - mu_1 = s Phi^-1(p_1),
  s^2 = D1+D2+||v1-v2||^2 (verified numerically, min-wins sign); general-
  base inversion uses alpha=0.7 at N=2. tests/test_factor_n2.py added,
  12 tests; full suite 74 passed. Algorithm 2 gained the N=2 branch and
  Prop 4 discussion notes K_2 bipartite / lambda_2 = 2.
- Warm-start recursion was NOT circular in code (bottoms out at analytic
  log-share start when V=0); Algorithm 2 pseudocode now says so.
- Representation invariance: paragraph added after the reflection note
  (exact map invariant to V->VO and V->V+1c^T; finite implementation
  not; caller should center + SVD-orient). Measured at production
  resolution: |c|=10 moves log shares <1e-9, rotation <1e-6 (spot check
  N=30, k=2) -- real but small.
- Wording: exp29 "larger, not smaller" -> not directly comparable;
  "applies to all of them" -> mutually exclusive argmax/first-arrival
  scope; astronomy "precisely" conditioned on unchanged scores; "nobody
  would posit Gumbel" softened; "Every one of these" -> "these
  orthant-probability methods"; abstract "errors" -> "agree with
  higher-resolution references ... in log share"; pruned-mass equality
  "so" -> "consistent with" + signed-cancellation caveat.

Still open from review 2 (experiments / bigger):
- Exp17 factor sweep confound: envelope recomputed per factor order, so
  the sweep mixes factor and spatial error. Fix: fixed envelope from the
  highest-order rule, or show doubling L moves results far below the
  factor discrepancy.
- Top-two O(RN) all-share estimator: promote to headline benchmark with
  an accuracy-vs-time frontier at N=200 and N=1000 (shared with review 1
  item). Restrict "gap widens linearly in N" to the per-alternative
  formulation (abstract now says this).
- One actual large-N matrix-free JVP/projected-CG solve experiment
  (iterations, residual, wall time); state the gauge implementation (no
  dense B). Also state exactly which equation Newton-Krylov solves
  (continuum Laplacian vs frozen-grid derivative).
- Rotation / common-shift / near-duplicate regression tests in the
  package (spot check done; make them tests).
- L auto-scaling rule (envelope width / smallest conditional sd) with
  under-resolution warning; note nonneg-weight assumption breaks for
  negative-weight sparse grids.
- Removal ensemble: label full NxN ensemble O(QN^2L), selected removals
  at forward cost (check current wording).
- "Arbitrary densities" scope sentence (continuous, evaluable
  density+CDF; accuracy distribution-dependent).
- N=2 and stress cases (tiny D, near-duplicates, Dirichlet targets,
  large common loadings) added to the robustness suite.
- Reviewer suggests months/year + six-order passage still reads as a
  strawman even labeled; Peter's call whether to delete outright.
- Footnote 10 stranded across pages 11-12 (typesetting); number and
  caption the tables on pp. 13/21/28.

- Vasicek citation: DONE 2026-08-24 — credit-risk sentence added to the
  Related Work dimension-reduction paragraph, after Dunnett, no priority
  claim over Butler.
- Racing citation: DONE 2026-08-24 — henery1981/lo1994 already attached
  to the second-place sentence in Potential applications.
- Prior-art citations from papers/prior-art-inversion-and-shared-field.md:
  DONE 2026-08-24 — Li 2018 (complement framing), Lambert 1975 (earliest
  shared-field), Chiang 1961 (deletion ensemble), Anderson-Ghurye 1977 +
  Mukherjea-Stephens 1990 (identification), Thurstone 1945 + Guilford
  1937 (psychometric forward integral / backward problem). Two bibitem
  titles flagged % VERIFY (Chiang, Mukherjea-Stephens: titles from
  memory).
- Intro rewrite: DONE 2026-08-24 — leads with the problem then the
  hardness record (Torgerson, GHK, Conlon, PyBLP, BLP 2004, Freyberger,
  Sawtooth, McFadden-Train); panorama moved to Potential applications.
  Three quote sets flagged % VERIFY pending the verification agent.
- China/US line in intro: author to decide (flagged as venue-tone risk,
  not logic error).

## From the 2026-08-16 full referee review (deferred; need Peter's call)
- Title: DECIDED 2026-08-16 — stays "Scalable Probit Share Calibration".
- Intro color cuts (China/US, indexes, biology, second-place horses,
  "built into every neural network", "universal for one reason"): the
  reviewer wants nearly all of it gone. Peter's prose; his call.
- Full 10-section restructure proposal (algorithmic spine first, all
  interpretations moved to a late Connections section). Conflicts with
  Peter's own 6-section structure from this week.
- Extend exp24 per-alternative RQMC comparison: several N, k=2..4,
  share buckets, heteroskedastic designs; move from Related Work to §4.
- Experiment numbers out of prose into an appendix table (reviewer);
  conflicts with the committed-script ethos in the prose.
- Ordered-probit/Lazear-Rosen "same model" looseness in intro ¶1.
- D-range reporting for the original exp16 robustness suite (new exp33
  covers ratios 1e2/1e3; retrofitting old suite optional).
- Factorial replication: DONE 2026-08-17 (experiment 36, 20 seeds,
  factor probit wins 20/20 in every stratum). Remaining caveat: single
  truth family.
- Optional: 1000-problem randomized stress suite for empirical global
  convergence (failure count, worst iterations vs spectral gap).
- Real-data inner-inversion demo (reviewer suggestion): take V from
  actual product characteristics or an existing fitted factor MNP and
  demonstrate ONLY the inner inversion/removal problem. Removes the
  "synthetic algorithm looking for a use case" objection without
  expanding claims.
- Reviewer prefers Further structure compressed to a paragraph with the
  transport/trace material in an appendix (currently its own section
  after the numerics). Editorial; Peter's call.
- Intro trim: reviewer asks again for 40-50% cut of pp. 1-2 color.
