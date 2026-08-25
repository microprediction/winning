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

**Out of scope, deliberately.** Whether `P(x* in batch)` is the right
objective, whether the loop should be exploitative, and how acquisition
functions compare in closed-loop discovery are all questions about the method,
not the arithmetic. This paper replaces the arithmetic. The closed-loop
measurements in README.md are reported for completeness and are not the claim.
