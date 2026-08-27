# rprobitfast

Exact multinomial probit where MACML approximates.

The Rprobit package (Bauer, github.com/dbauer72/Rprobit) estimates MNP
models by MACML (Bhat 2011), whose engine is the Mendell-Elston /
Solow-Joe family of analytic approximations. Those approximations are
fast and centrally accurate, and they drift silently in the tails: in
the general-inversion paper's field benchmark the ME recursion is
within 2e-3 of truth at n=10, then 3e-2 at n=30 with tail
probabilities off by a factor of ~4 and nothing in the output
announcing the change. This package prices the same likelihood exactly,
to quadrature accuracy, with the analytic score and the
sharpness-escalating node rule.

```r
fit <- rprobit_fast(df, covariates = c("price"), r = 2)
```

df in long format: id, alt, chosen, covariates. Synthetic J=3, T=1500
MNP fits in ~4 s with correct recovery. Boundary-seeking covariance
(the pathology documented in r/mlogitfast and winning.mnprobit) is
detected and flagged.

Status: engine shared with mlogitfast (sync-guarded); a direct
head-to-head against Rprobit itself awaits a machine where its Rcpp
build succeeds -- until then the field-table measurement of MACML's
core ingredient is the standing comparison. Interface parity with
Rprobit's own S4 workflow is not attempted; this is the same model
class behind a plain data-frame interface.
