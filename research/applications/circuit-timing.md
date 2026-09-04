# Statistical timing criticality is a correlated race
(Peter's pointer, 2026-09-01: Mogal, Qian, Sapatnekar, Bazargan, "Fast
and accurate statistical criticality computation under process
variations", IEEE TCAD 2009 [locator-verified via IBM Research page;
unread in full].)

## The mapping, term by term
Statistical static timing analysis (SSTA) asks, under manufacturing
process variation, for each gate/edge the CRITICALITY: the probability
it lies on the critical (maximum-delay) path of a manufactured die.

- The industry-standard "canonical first-order delay model" -- delay =
  nominal + sum of sensitivities times shared variation sources plus an
  idiosyncratic term -- IS the factor grammar, literally: a linear
  factor model with spatially correlated common factors.
- "Tightness probability" (their Clark-formulation ingredient) is the
  pairwise win probability. Criticality over a CUTSET is the n-way
  membership: which of the cutset's path delays is the maximum -- a
  correlated Gaussian race, priced by the shared field in O(nLQ)
  instead of chained pairwise moment matching.
- Their error source is named in their own abstract: tightness
  probabilities under Clark's statistical-maximum formulation. The
  winning paper ALREADY benchmarks Clark-type recursions: drift to
  3e-2 total variation at n=30 with tail entries off by a factor of
  ~3.7 and nothing in the output announcing it. Criticality lives in
  the tails (rare critical paths), so that silent error is the
  interesting one.
- The Jacobian is the optimization payload: d(criticality)/d(mean
  delay) is the gate-sizing gradient, exact and matrix-free, where the
  2009-era flow needed criticality just to RANK gates for sizing.
- Non-Gaussian variation and the skew of max-of-Gaussians intermediate
  distributions meet the density-agnostic bases.

Their numbers to beat/match: ~250x speedup over pairwise pruning at
~5 percent error against Monte Carlo. Ours would be exact against the
same referee; the question is wall clock at circuit-scale cutsets.

## Candidate experiment
An ISCAS-style benchmark netlist (or a synthetic timing graph at
realistic scale), canonical-form delays with spatial factors, cutset
criticalities: exact shared-field versus Clark-chain versus Monte
Carlo referee, scored on bulk TV and tail log-error, plus one
sizing-gradient check against finite differences.

## Literature gate before writing anything
SSTA is a large literature: Visweswariah et al. 2004 (canonical form),
Clark 1961 (already cited in the winning paper), the criticality line
this paper sits in, and whatever replaced it after 2009 [all U;
read-in-full required].

## The 2024 confirmation, and the observation that sharpens everything
Mishagli, Koskin, Blokhina, "Statistical Static Timing Analysis of
VLSI as the Statistics of Correlated Extremes", arXiv:2401.03559
(2024) [locator-verified; unread in full]: SSTA reframed explicitly as
correlated extreme-value statistics, solved by analytic corrections to
the Gumbel distribution valid for WEAK correlations, with stated
applicability limits, plus an algorithm for estimating a timing
graph's covariance. So the framing is live in 2024, and the current
tool is a weak-correlation expansion where the engine is exact at any
correlation strength (with the sharpness escalation carrying the
strong end).

The observation their covariance algorithm hands us: path-delay
covariance in a timing graph is IN-GRAMMAR BY CONSTRUCTION. A path's
delay is a sum over its edges, so Cov(path a, path b) = shared-edge
variance: the covariance is the Gram matrix of path-edge incidence
vectors weighted by edge variances -- exactly a factor model whose
loadings are incidence rows, before any fitting. With the industry
canonical form (few global spatial sources plus per-gate idiosyncratic
terms) the rank is the number of variation sources, which is small.
Nothing needs approximating: SSTA criticality under the canonical
model is a low-rank factor race, natively.

Ties to standing issues: their Gumbel-correction machinery is the
analytic-extremes toolbox of issue 10 (control variates and exact
tests), and the tail-exponent-controls-extremes-growth story of issue
13 is the same mathematics from the base-family side.
