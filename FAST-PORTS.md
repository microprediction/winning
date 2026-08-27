# The -fast ports: meeting users inside the tools they already use

Strategy (2026-08-27): rather than asking users of the dominant probit
tools to switch packages, ship drop-in accelerators inside the
interfaces they already trust. Evidence for the targets' dominance and
download counts are in the general-inversion paper and its commit
history (Stata's cmmprobit/asmprobit: GHK with no non-GHK option;
mlogit 19k downloads/mo; mvtnorm 335k/mo).

## Shipped

- **mvtnormfast** (r/mvtnormfast): pmvnorm drop-in for factor-structured
  covariance. 1-2 ms vs 31 ms - 8.2 s at matched accuracy; auto-detects
  exact structure, strict refusal + fallback for dense sigma; Laplace
  recentering for deep tails. Eight tests.

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
