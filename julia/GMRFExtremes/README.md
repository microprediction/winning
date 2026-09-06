# GMRFExtremes.jl

The order-statistic layer the graphical-model stack does not compute.
A Gauss-Markov chain's Kalman filter/smoother — or its GMRF, in
[GaussianMarkovRandomFields.jl](https://github.com/timweiland/GaussianMarkovRandomFields.jl)
form — gives marginals, joints and samples. It does **not** give:

- `argmax_marginals(c)` — P(X_t is the maximum), for every t
- `max_cdf(c, u)`, `expected_max(c)` — the law of the maximum
- `first_passage(c, u)` — the distribution of the first index over u
- `excursion_probability(c, u)` — the Bolin–Lindgren query
  (JRSS-B 2015; their R package `excursions` answers it by sequential
  importance sampling — here the chain case is deterministic and
  exact to quadrature)

All by restricted transfer-operator passes on the chain (the
machinery measured in the winning repo's `research/tridiagonal`
track), O(n L^2) with precomputed transitions, fractional-occupancy
boundary cells for O(dx^2) accuracy, for ANY Gauss-Markov chain —
non-stationary, drifting, inhomogeneous — including any GMRF whose
precision is tridiagonal (the chain is read off the precision's
bidiagonal Cholesky).

```julia
c = GaussMarkovChain(mu, sd0, phi, s)          # explicit chain
c = chain_from_precision(mu, Q)                 # tridiagonal Q
using GaussianMarkovRandomFields                # arms GMRF dispatch
c = GaussMarkovChain(gmrf); argmax_marginals(gmrf)
```

**The refusal contract**: bandwidth > 1 needs the lifted-state pass
(measured in the winning repo, roadmap here) and general sparse
precision belongs to sampling methods; both are refused with a
pointer, never approximated silently.

The one-plot fact this package exists to compute: on a stationary
AR(1) at phi = 0.9, the endpoints are ~2.6x likelier than the middle
to be the field's maximum — while the stationary Kalman filter
reports identical marginals everywhere. The marginal layer literally
cannot express where the extreme lives; this layer is that answer.
Tests: closed forms, a 400k-path Monte Carlo referee, the measured
boundary-pile-up and discrete-arcsine laws, precision-to-chain round
trips, and the refusal contract.
