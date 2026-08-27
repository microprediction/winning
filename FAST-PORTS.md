# The -fast ports: meeting users inside the tools they already use

Strategy (2026-08-27): rather than asking users of the dominant probit
tools to switch packages, ship drop-in accelerators inside the
interfaces they already trust. Evidence for the targets' dominance and
download counts are in the general-inversion paper and its commit
history (Stata's cmmprobit/asmprobit: GHK with no non-GHK option;
mlogit 19k downloads/mo; mvtnorm 335k/mo).

## Shipped (Python, in the winning package itself)

- **winning.fastmvn**: scipy.stats.multivariate_normal.cdf drop-in for
  factor-structured covariance (numpy has no MVN-probability surface;
  scipy is the ecosystem target). 0.3-1 ms where scipy needs 4 ms-4.6 s;
  scrambled Sobol past rank 2 (plain Halton degrades by dim 6:
  measured 1e-4 -> 2.6e-6); Laplace recentering for deep tails; strict
  factorization verification with scipy fallback. Five tests.
- **winning.likelihood**: the estimation core the package lacked --
  observation-level exact likelihood + analytic score (posterior-
  weighted Mills ratios), validated vs finite differences at 1e-8 on
  both node branches. Rust port of this kernel is the natural next
  fastrace addition.
- **winning.mnprobit**: MNProbit / MNProbitClassifier -- the model
  statsmodels does not have (it has binary and ordered probit, no
  multinomial) and sklearn cannot express (softmax only). Not a
  replacement: a first. Fishing: referee logLik -1212.82 +- 0.10 in
  9.5 s (vs GHK -1215.7 in 22.2 s), with the likelihood reported from
  independent Sobol scrambles rather than the optimizer's own
  landscape, and a boundary_ flag: the unrestricted-covariance MLE on
  Fishing is boundary-seeking (true logLik still rising at ||v||~4000,
  verified 2^16-2^18) -- GHK's simulation noise was accidentally
  regularizing a ridge with no interior maximum.

## Shipped (R)

- **mvtnormfast** (r/mvtnormfast): pmvnorm drop-in for factor-structured
  covariance. 1-2 ms vs 31 ms - 8.2 s at matched accuracy; auto-detects
  exact structure, strict refusal + fallback for dense sigma; Laplace
  recentering for deep tails. Eight tests.

- **rprobitfast** (r/rprobitfast): exact MNP for the model class of
  Bauer's Rprobit (MACML/Mendell-Elston engine, whose measured tail
  drift is the field table's ME row). Plain data-frame interface,
  shared engine with mlogitfast (sync-guarded), boundary detection.
  Direct head-to-head pending an Rprobit Rcpp build that succeeds.

## Next

- **mlogitfast** (R): probit estimation behind mlogit's interface.
  Baseline captured: mlogit(probit=TRUE) on its own Fishing vignette =
  22.2 s via GHK; our engine prices those per-observation vectors in ms
  with exact scores. Care needed mapping mlogit's differenced-covariance
  parameterization to factor form (representable for J=4; identification
  bookkeeping is the real work). Deliverable: same formula/data in,
  coefficients + higher exact logLik out, order-of-magnitude faster.

- **asmprobit_fast / cmmprobit_fast** (Stata): ado wrapper over the
  Python package via Stata 16+'s native python integration. Scaffold
  only until someone with a Stata license runs it.

- **Live convergence pages** (JS): the winning JS port already drives
  winning.microprediction.org; port the COMPETITORS (GHK with growing R,
  Mendell-Elston, frequency counting) to JS and animate error-vs-time
  against the lattice on the same case, in the browser. The bench.py
  "alt" numbers become something a visitor can watch happen.

- **pyblp adjacency** (Python, research-grade): the Berry share
  inversion is literally abilities_from_race for RC-logit; a
  probit-family demand variant ("Berry inversion for probit, exact,
  with Jacobian") is a paper-sized contribution aimed at the IO
  audience, not a port. Berry-Pakes pure characteristics is the model
  class; the extreme repo's paper already cites it.
