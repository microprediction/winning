# MultinomialProbit.jl

The model no Julia package has: multinomial probit estimation with
correlated alternatives. Julia's discrete-choice ecosystem is
logit-family throughout (DiscreteChoiceModels.jl, DCM.jl, MultiLogit)
and probit stops at binary (GLM) and ordered
(OrdinalMultinomialModels.jl); nothing estimates the correlated
random-utility probit. This does, with two engines behind one
interface:

- `method = :exact` (default) — the factor-conditional
  product-of-CDFs likelihood with an **analytic score**: smooth,
  deterministic, no simulation anywhere. Port of the python
  reference `winning.likelihood` / `winning.mnprobit` (itself
  translated from the R package `mlogitfast`), pinned to it by
  embedded fixtures at 1e-9.
- `method = :ghk` — the field's incumbent, for reference and for
  head-to-heads: the GHK sequential importance sampler
  (common-random-numbers, so the simulated likelihood is a
  deterministic, optimizable function of the parameters). Port of
  the `winning` rust core's `ghk_prob_one`.

Model: utilities `U_tj = x_tj' beta + (V f_t)_j + z_tj` with
`f_t ~ N(0, I_r)`, unit idiosyncratic noise, `choice = argmax_j U_tj`.
Loadings carry a zero reference row and a strictly-lower-triangular
free block; at J alternatives and r = 2 this spans the identified
differenced covariance with the same dof count as mlogit's
differenced-Cholesky parameterization.

```julia
m = MNProbit(X, choice; intercepts = true, r = 2)  # X: (T, J, p)
fit!(m)                       # exact likelihood, analytic score, BFGS
fit!(m; method = :ghk)        # the simulation incumbent, same API
loglikelihood(m); m.beta; m.V
P = predict_proba(m)
```

Dependency-free but for stdlib `LinearAlgebra` and `Random`.
Sharpness rule inherited from the reference: past
`sqrt(2) max_j ||v_j|| / sqrt(min D) = 3` Gauss-Hermite
under-integrates a near-step integrand and an optimizer will exploit
the holes; evaluation escalates to Halton nodes automatically.

Tests: `julia --project=julia/MultinomialProbit
julia/MultinomialProbit/test/runtests.jl` — python-fixture parity
(likelihood, both score blocks, forward probabilities at 1e-9),
analytic score vs central differences, GHK vs the exact engine, and
an end-to-end planted-parameter fit.
