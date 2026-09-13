# FactorMvNormalCDF.jl

Deterministic multivariate-normal rectangle probabilities for
factor-structured covariance, the
[MvNormalCDF.jl](https://github.com/PharmCat/MvNormalCDF.jl) companion
for the `V V' + diag(D)` slice. Port of the python `winning.fastmvn`
(itself a port of the R package `mvtnormfast`); the python reference
is the spec and embedded fixtures pin this port to it.

Conditional on the r-dimensional factor the coordinates are
independent, so `P(a <= X <= b)` is a low-dimensional smooth integral
of a product of univariate normal CDFs. The package evaluates it by
exact Gauss-Hermite quadrature, deterministic (same input, same bits),
with a Laplace-recentered path for deep tails validated to
probabilities near 1e-31 against the reference.

```julia
using FactorMvNormalCDF
p, e = mvnormcdf_factor(mu, Sigma, a, b)     # MvNormalCDF's signature, own name
p = mvn_cdf_fast(V = V, D = D, upper = b)    # structure supplied directly
p, how = mvn_cdf_fast_info(sigma = Sigma, upper = b)   # how: "factor", "factor-recentered", "fallback"
```

The exact path covers factor rank at most 2 and sharpness
`max ||v_i|| / sqrt(D_i)` at most 3. Every other case, including a
covariance that does not verify as factor-plus-diagonal to strict
tolerance, is delegated to `MvNormalCDF.mvnormcdf`, which is a
dependency, so every call is answered and the caller need not know
which path ran. Keyword arguments such as `m` and `rng` are forwarded
to MvNormalCDF on that path.

The package exports its own names only, so `using MvNormalCDF,
FactorMvNormalCDF` is unambiguous. The error estimate on the exact path
is the difference against a reduced-order quadrature (measured near
1e-7 where the true error is near 1e-8); delegated results carry
MvNormalCDF's own estimate.

Tests: fixture parity with the python reference (tight on the exact
path, relative on recentered tails), delegation of dense and rank-3
cases, keyword forwarding, and agreement with MvNormalCDF on factor
structure (measured 8e-8 at n = 6 against a 200k-point QMC run).
