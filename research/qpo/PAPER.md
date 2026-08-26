# Scope: this is a numerical methods paper

**They did it this way.** qPO scores a candidate by the probability it is the
best in the library, `p_i = P(Y_i = max_j Y_j)`, under the joint GP posterior.
The released implementation estimates that by drawing 10,000 samples from the
joint posterior and counting winners. Sampling a dense N-dimensional Gaussian
costs O(N^2) per draw, so the pool is first truncated to the top 10,000 by
UCB.

**We do it this way.** `p_i` is a race: the probability that competitor i beats
a correlated field. Approximate the posterior as `Sigma = V V' + D` with V of
rank r; conditional on the r factors the candidates are independent, the N
idiosyncratic dimensions integrate in closed form, and only r dimensions need
quadrature. One lattice pass over a field product, with the cavity division
that removes each competitor from its own field.

**What that buys.**

| | released sampler | this |
|---|---|---|
| cost per candidate | O(N) per draw, O(N^2) covariance | O(1) per lattice point |
| whole-library run (133,702) | needs a 143 GB covariance | 3 minutes, 2 GB |
| accuracy at equal wall time (N=1000) | top-100 recall 0.58 | **0.93** |
| vs factor MC at matched accuracy | 88 s | **0.10 s (880x)** |
| relative error on rare candidates | 1/sqrt(M p_i), unbounded | independent of p_i |
| exact zeros | yes, for unlucky candidates | never |
| output | step function of the draws | smooth in (mu, V, D) |

**Smoothness is not a footnote.** A counted estimate is a step function of the
random draws, so nothing downstream can differentiate it, calibrate against
it, or reuse it across rounds. The lattice answer is smooth in the posterior
parameters, which is what lets it be fitted, warm-started, and updated.

**Rank is the only question, and it is settled.** The kill test: if qPO needed
rank in the hundreds the method would stop being fast. Rank 2 preserves
eta > 0.995 and rank 4 reaches 0.998, on five posteriors across two datasets,
including a mid-campaign posterior and an unprefiltered pool. A rank-2 model
whose off-diagonal covariance error is still 40% delivers essentially the
whole objective, and Plackett's relation says why: the argmax weights the few
pairs that compete for the lead, not the N^2/2 pairs that Frobenius counts.

## Target: short, narrow, JCIM

qPO was published in J. Chem. Inf. Model. 65(10):4808-4817 (May 2025), and
again in TMLR (2025). JCIM is the natural home: same readership, same
application area, and a short methods note that makes a recently published
method 800x faster is exactly the shape of paper that journal takes. Target
4-6 pages: the race formulation, the lattice, the rank kill test, the
timing/accuracy table, the whole-library demonstration. No new acquisition
function, no re-litigation of the objective, no closed-loop claims.

## Why this reaches past one paper

`p_i = P(i is best)` is one computation with many names, and in every one of
them the standard practice is to sample:

- **Thompson sampling.** TS selects arm i with probability exactly
  `P(i optimal)`; the usual implementation draws a posterior sample and takes
  an argmax. With correlated arms that draw is O(N^2), which is precisely the
  regime this solves.
- **Bayesian A/B testing.** "Probability to be best" is the headline number in
  every commercial testing platform, and it is computed by Monte Carlo --
  closed forms exist only for two to four independent variants.
- **Response-adaptive randomisation.** Allocation proportional to
  `P(treatment best)`; the regulatory setting wants reproducibility, which a
  counted estimate does not give and a lattice does.
- **Entropy search and its variants.** All are built on the distribution of
  the argmax, `p(x*)`.
- **Best-arm and top-k identification**, and the discrete-choice/ranking
  literature this machinery came from.

The claim is narrow and portable: wherever the arms are correlated and the
field is large, the win probability need not be sampled.

**Out of scope, deliberately.** Whether `P(x* in batch)` is the right
objective, whether the loop should be exploitative, and how acquisition
functions compare in closed-loop discovery are all questions about the method,
not the arithmetic. This paper replaces the arithmetic. The closed-loop
measurements in README.md are reported for completeness and are not the claim.


## Scaling addendum (2026-08-26)

The block-structured extension of the same kernel (winning.factor.blocks,
fastrace hybrid backend, winner-bulk lattice window with an exact
omitted-mass bound) prices a correlated race over 10,000,000 candidates in
200,000 clusters in 64 s on a laptop at 4.7 GB peak -- flat ~4-6 us per
candidate from 1e5 to 1e7, boards summing to 1.000000 throughout, every
probability smooth and positive. The dense-covariance alternative does not
exist at any budget (an 800 TB matrix); the sampling alternative would need
>1e7 draws for one expected win count per candidate. For the paper's scope
this extends the operating-point claim by two orders of magnitude beyond
the whole-library molecular run.
