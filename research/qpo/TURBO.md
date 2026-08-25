# The Thompson step is a race, and races need not be sampled

Trust-region BO (TuRBO, Eriksson et al. 2019) is the workhorse for
high-dimensional problems. Its inner loop draws q joint posterior samples over
a candidate set and takes each sample's argmin -- Monte Carlo estimation of

    p_i = P(candidate i is the minimiser),

which is exactly the race this directory solves. Two things follow.

## The candidate cap is a computational artifact

TuRBO caps its candidate set at `min(100d, 5000)`; qPO prefilters to 10,000.
Both caps exist because joint sampling needs an N x N covariance: O(N^2)
memory and O(N^3) to factor. Measured, d = 100, batch q = 20:

| candidates N | cov + Cholesky | ours, rank 0 | ours, rank 2 | dense memory |
|---:|---:|---:|---:|---:|
| 5,000 (the cap) | 1.3 s | 0.04 s | 2.1 s | 0.2 GB |
| 20,000 | 120.5 s | 0.18 s | 9.3 s | 3.2 GB |
| 50,000 | not attemptable | 0.61 s | 23.9 s | 20 GB |

Against a 200,000-draw dense reference at N = 5,000: rank 0 gives TV 0.097
with 18/20 of the top-20 batch and correlation 0.984; rank 2 halves the TV to
0.047 at the same overlap. Covering a 100-dimensional trust region with 5,000
points is extremely sparse, so the cap binds exactly where relief is wanted.

## Sampling the latent, not the field

The factor model is generative, not merely a compression:

    Y_i = mu_i + V_i . f + sqrt(D_i) eps_i,     f in R^r, eps independent.

So a Thompson draw needs r numbers and one pass: no Cholesky, no N x N.
Measured at N = 20,000, d = 100, q = 20 draws:

| route | setup | 20 draws | total | memory |
|---|---:|---:|---:|---:|
| Cholesky (what TuRBO does) | 88.3 s | 2.405 s | **90.7 s** | 3.2 GB |
| factor r = 2, sample the latent | 0.9 s | **0.0053 s** | **0.9 s** | O(N r) |
| factor r = 4 | 1.0 s | 0.0046 s | 1.0 s | O(N r) |

A hundred-fold on the step, and 450x on the draws themselves, at TV 0.12 from
the dense posterior. The approximation is the rank, not the sampling.

**Three uses of the same object, in increasing order of what they ask for.**
If you want an ACTION, sample f and eps and take an argmin: O(N r), and no
probabilities are needed. If you want the PROBABILITIES, note that each draw
of f yields an entire conditional probability vector, because conditional on
the factor the candidates are independent -- averaging those is
Rao-Blackwellisation, and the lattice is its limit (integrate f on a grid
rather than sampling it). If you want UNCERTAINTY ABOUT the probabilities,
sample the abilities themselves from their posterior: each draw gives a whole
probability vector, so the acquisition ranking acquires error bars. The third
is standard practice nowhere and falls out of the same model.

## What is NOT claimed

Not that this fixes high-dimensional GPs. With a stationary kernel, few
observations and candidates spread over the domain, the posterior goes
vacuous: measured on Matern 5/2 with n = 30, `max_p * N` falls from 72 at
d = 1 to 1.16 at d = 20, i.e. every candidate is equally likely to be best and
every acquisition rule is the same rule. That is a statistical wall and no
arithmetic touches it.

But that is the naive configuration, not how high-dimensional BO is run. With
fitted ARD lengthscales inside a trust region -- the d = 100 Ackley posterior
above -- `max_p * N` is 79 with effective support 355 of 5,000: strong
opinions and real correlation. The regime practitioners actually use is
exactly the regime where this computation applies.
