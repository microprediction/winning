# fastmvn

Fast multivariate normal rectangle probabilities for factor-structured
covariance -- and a transparent scipy accelerator.

```python
from fastmvn import mvn_cdf_fast
p = mvn_cdf_fast(upper=b, mean=mu, V=V, D=D)      # sigma = V V' + diag(D)
p = mvn_cdf_fast(upper=b, sigma=S)                # auto-detect structure

import fastmvn
fastmvn.patch_scipy()                              # sklearnex-style: existing
# scipy.stats.multivariate_normal.cdf calls now route factor-structured
# covariances through the fast path, everything else through scipy.
```

Conditional on the low-dimensional factor, coordinates are independent,
so the rectangle probability is a small smooth integral of a product of
univariate normal CDFs: deterministic, no simulation. Measured on one
laptop against scipy's Genz-style integrator at matched inputs:

| case | fastmvn | scipy | agreement |
|---|---|---|---|
| n=10 one-sided | 0.2 ms | 3 ms | 4.5e-6 |
| n=30, p ~ 6e-8 | 0.3 ms | 39 ms | 4.1e-10 |
| n=200 moderate p | 0.9 ms | ~4.6 s | inside referee error |
| n=200, p ~ 3e-20 | 35 ms (auto-recentered) | -- | 5e-4 rel vs minimax tilting |

Three regimes handled automatically: adaptive Gauss-Hermite for mild
loadings; scrambled Sobol for sharp loadings or rank above two; Laplace
recentering with importance reweighting for deep tails. Dense,
genuinely unstructured covariance is handled by refusal: the exact
factorization check is strict (reconstruction to 1e-11 relative,
verified with a recomputed factor), and anything failing it goes to
scipy unchanged, so a loose fit can never masquerade as the structured
case. The patch caches factorization attempts by covariance content, so
repeated calls with the same matrix pay the eigendecomposition once.

Part of the winning project
(github.com/microprediction/winning); the R sibling is mvtnormfast.
