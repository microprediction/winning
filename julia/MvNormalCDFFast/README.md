# MvNormalCDFFast.jl

Deterministic multivariate-normal rectangle probabilities for
factor-structured covariance: the
[MvNormalCDF.jl](https://github.com/PharmCat/MvNormalCDF.jl) companion
for the `V V' + diag(D)` slice. Port of the python `winning.fastmvn`
(itself a port of the R package `mvtnormfast`); the python reference
is the spec and embedded fixtures pin this port to it.

Conditional on the r-dimensional factor the coordinates are
independent, so `P(a <= X <= b)` is a low-dimensional smooth integral
of a product of univariate normal CDFs — exact Gauss-Hermite
quadrature, deterministic (same input, same bits, unlike QMC), with a
deterministic Laplace-recentered path for deep tails (validated to
probabilities ~1e-31 against the reference).

```julia
using MvNormalCDFFast
p, e = mvnormcdf(mu, Sigma, a, b)        # incumbent-compatible signature
p = mvn_cdf_fast(V = V, D = D, upper = b)  # structure supplied directly
```

**The refusal contract.** The fast path claims only what it computes
exactly: factor rank <= 2 and sharpness `max ||v_i||/sqrt(D_i) <= 3`.
A covariance that does not verify as factor-plus-diagonal to strict
tolerance — or that exceeds those bounds — is REFUSED rather than
approximated (the reference measured plain low-discrepancy nodes
degrading to 1e-4 at rank 6). Load the incumbent and refused cases
route to it transparently:

```julia
using MvNormalCDF        # arms the fallback via a package extension
p, e = MvNormalCDFFast.mvnormcdf(mu, dense_sigma, a, b)  # -> incumbent
```

Both packages export `mvnormcdf`, so qualify the name when both are
loaded. The returned error estimate on the fast path is the difference
against a reduced-order quadrature (measured ~1e-7 where the true
error is ~1e-8); fallback results carry the incumbent's own estimate.

Tests: fixture parity with the python reference (tight on the exact
path, relative on recentered tails), the refusal contract, armed
fallback routing, and agreement with the incumbent on factor
structure (measured 8e-8 at n = 6 against a 200k-point QMC run).
