# Julia Bayes-net ecosystem: where our contribution lands (2026-09-06)

## The landscape
- RxInfer.jl / ReactiveMP.jl (ReactiveBayes): the flagship, very
  active. Factor graphs + reactive message passing, state-space
  models to ~1e5 variables. Outputs: posterior marginals, free
  energy. No order-statistic queries.
- BayesNets.jl (Stanford SISL): canonical discrete BN package;
  maintained, slower-moving. Discrete, so a weak fit for us.
- GaussianMarkovRandomFields.jl (Tim Weiland): modern, active GMRF
  package -- SPDE/FEM constructions, sparse precision, AD-compatible
  Cholesky, extends Distributions.AbstractMvNormal. Issue #27
  discusses merging with jojal5/GMRF.jl (Rue-Held style): a
  community-consolidation moment, i.e. a maintainer receptive to
  contributions.

## The gap (same shape as the tridiagonal track's thesis)
Every one of these computes marginals, joints and samples. NONE
computes the order-statistic layer: P(X_t is the max), the max-CDF /
expected max, first-passage to a barrier, top-k membership. In the R
world this query class has a name and an incumbent: Bolin & Lindgren's
`excursions` (JRSS-B 2015) -- excursion sets and exceedance
probabilities for latent Gaussian models, via sequential importance
sampling. Julia has NO equivalent. Meanwhile our machinery computes
these EXACTLY for banded precision (research/tridiagonal exp1-exp4:
transfer operator, b=1 measured, b=2 lifted and verified 5.6e-16
against the b=1 embedding) and to quadrature accuracy for factor
covariance (the engine itself).

## The suggestion
An extension package (or PR series) against
GaussianMarkovRandomFields.jl -- working name `GMRFExtremes.jl` --
offering, on their AbstractMvNormal-extending GMRF type:
  argmax_marginals(gmrf)      P(X_i is the max), all i
  max_cdf(gmrf, u)            P(max <= u), expected max
  first_passage(gmrf, u)      distribution of the first index over u
  excursion_probability(...)  the Bolin-Lindgren query, exact on bands
Dispatch: exact transfer operator when the precision is banded
(detected from their sparse structure), our factor engine when the
covariance verifies as V V' + diag(D), and an honest refusal (or SIS
fallback citing excursions) otherwise -- the refusal-plus-fallback
contract of the -fast ports.

## The one-plot pitch
exp2's measured law: on a stationary AR(1) at phi = 0.9 the endpoints
are ~2.6x likelier than the middle to be the field maximum, while a
stationary Kalman filter reports IDENTICAL marginals everywhere --
the marginal layer literally cannot express where the extreme lives.
That plot, plus "exact where excursions simulates," is the whole
announcement post.

## Sequencing
After (or alongside) registering the three existing Julia packages;
the tridiagonal exp4 argmax lift (open item in research/tridiagonal/
NOTES.md) is the natural build-out this would ship on top of.
