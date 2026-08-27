# mvtnormfast

Drop-in accelerator for `mvtnorm::pmvnorm` on the structured slice:
covariances of the form `VV' + diag(D)` (factor plus diagonal), which
are the estimable covariances of multinomial probit and the workhorse
structures of applied work. Conditional on the low-dimensional factor,
rectangle probabilities are products of univariate normal CDFs, so the
whole computation is a small smooth integral -- deterministic,
milliseconds, no simulation.

```r
pmvnorm_fast(lower, upper, mean, V = V, D = D)   # structured, fast path
pmvnorm_fast(lower, upper, mean, sigma = S)      # auto-detect; falls back
                                                 # to mvtnorm::pmvnorm if S
                                                 # is not exactly structured
```

Measured on one laptop (rank-2 factor covariance, vs `mvtnorm` at
matched accuracy and Botev's `TruncatedNormal` as the tail referee):

| case | mvtnormfast | reference | agreement |
|---|---|---|---|
| n=30 one-sided | 1 ms | mvtnorm 31 ms | 9e-12 |
| n=200, p ~ 6e-3 | 2 ms | mvtnorm (abseps 1e-12) 8.2 s | 2e-4 rel, inside Botev's own error |
| n=200, p ~ 9e-18 | 130 ms (auto-recentered) | Botev relerr 1.6e-3 | 5e-4 rel |

Three regimes, handled automatically:
- mild factor loadings: adaptive-order Gauss-Hermite;
- sharp loadings or rank > 2: Halton nodes (polynomial rules converge
  slowly on sharp integrands at any order);
- deep tails (p < 1e-8): Laplace recentering of the node set with
  importance reweighting.

Dense, genuinely unstructured covariances are not this package's
business: the exact-decomposition check is strict (reconstruction to
1e-11 relative), and anything that fails it goes to `mvtnorm::pmvnorm`
unchanged, so a loose fit can never masquerade as the structured case.
