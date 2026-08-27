# mlogitfast

Exact multinomial probit estimation behind the mlogit interface: the
GHK simulator replaced by a deterministic factor-conditional product
integral, vectorized across observations.

```r
Fish <- dfidx(Fishing, varying = 2:9, shape = "wide", choice = "mode")
fit  <- mlogit_fast(mode ~ price + catch, Fish)
```

Measured on mlogit's own Fishing vignette (J=4, T=1182):

|  | logLik | time | note |
|---|---|---|---|
| mlogit(probit=TRUE), GHK | -1215.655 | 22.2 s | simulated likelihood |
| mlogit_fast | **-1214.613** | 94.2 s | exact to quadrature accuracy |

Same identified model space (rank-2 factors, zero reference loadings,
unit idiosyncratic variance covers every positive-definite differenced
covariance up to scale, with the same five covariance parameters at
J=4). The scale-invariant coefficient ratio catch/price agrees with
mlogit to 1% (-45.0 vs -44.6); the one-nat likelihood gain is the
expected sign and size of removing simulation bias at the optimum.

Slower than GHK in this version: numeric gradients cost 21 likelihood
evaluations each, and the fitted covariance is sharp enough that every
evaluation runs on the escalated Halton branch. The analytic score
(closed-form derivative of the product integral) is the named fix.

One war story worth reading in the source: without sharpness-aware
node escalation the OPTIMIZER EXPLOITS QUADRATURE HOLES -- the first
fit ran the loadings to ||w|| ~ 300 and reported a fake 20-nat
likelihood gain that collapsed under denser rules (GH-7: -1195.1,
GH-15: -1243.4, GH-31: -1229.9, non-monotone = noise). The likelihood
now switches to Halton nodes past sharpness 3, the same
family-escalation rule as winning::race_probabilities, and the runaway
disappears.
