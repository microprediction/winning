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
| mlogit_fast | **-1214.613** | **17.2 s** | exact to quadrature accuracy, analytic score |

Same identified model space (rank-2 factors, zero reference loadings,
unit idiosyncratic variance covers every positive-definite differenced
covariance up to scale, with the same five covariance parameters at
J=4). The scale-invariant coefficient ratio catch/price agrees with
mlogit to 1% (-45.0 vs -44.6); the one-nat likelihood gain is the
expected sign and size of removing simulation bias at the optimum.

The analytic score (posterior-weighted Mills ratios, one extra pass
over arrays the likelihood already computes, validated against finite
differences to 1e-8 on both node branches) replaced numeric gradients
and cut the fit from 94 s to 17 s. Remaining headroom if wanted:
sharp-branch node count, warm starts.

One war story worth reading in the source: without sharpness-aware
node escalation the OPTIMIZER EXPLOITS QUADRATURE HOLES -- the first
fit ran the loadings to ||w|| ~ 300 and reported a fake 20-nat
likelihood gain that collapsed under denser rules (GH-7: -1195.1,
GH-15: -1243.4, GH-31: -1229.9, non-monotone = noise). The likelihood
now switches to Halton nodes past sharpness 3, the same
family-escalation rule as winning::race_probabilities, and the runaway
disappears.
