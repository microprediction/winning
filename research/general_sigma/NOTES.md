# General Sigma: the estimator ladder (Aug 26, 2026)

Estimand: all-n win probabilities under dense Sigma. Reference: 2^18
Sobol. "VRF" below is equal-DRAW variance reduction vs plain MC.

| rung | estimator | bias | VRF | verdict |
|---|---|---|---|---|
| 1 | indicator CV (factor/tree surrogate, CRN) | 0 | 1.1-4.6 | binary disagreement noise; dead end |
| 2 | Rao-Blackwell, scalar alpha split | 0 | <1 | alpha collapses, integrand sharpens; dead |
| 3 | positive-part residual (full D analytic) | 3e-2 | ~2500 | variance solved, bias fatal |
| 4 | + seriated composed fit (rank3 + blocks) | 3e-2 | ~1000 | fit better (resid -40%) but bias barely moves |
| 5 | diagonal transfer (unbiased) | ~8e-4* | ~100 | *only when lambda_min(E) << min D |
| 6 | deconvolution adjustment (Peter) | 1.4e-3 | ~80 | full-D smoothing + exact analytic delta-sharpening correction; bias second-order |

Composed fit (ELI5): rank-3 global factors, then HRP-seriated blocks
with one rank-1 effect each fitted to the WITHIN-block leftover
(diagonal ignored), alternated to convergence; beats global rank-3 by
~40% in residual mass on the French 30 correlation.

Wall-clock caveat: per-draw wins are 80x but a lattice race costs ~1e5
Gaussian draws, so plain MC still wins per second at n <= 30. The
crossover is large n (plain draw O(n^2), race eval near input-bound in
rust) and any use needing the analytic side products. Scaling run TBD.

Candidate package verbs: fit_structure(Sigma) (the composed fit);
price_sigma(mu, Sigma) = rung-6 estimator.

## Large n: the one-call estimator (run_large_n.py)

Residual-as-extra-factors makes the whole ladder deterministic: fit
(rank-3 + seriated blocks + top-5 residual eigendirs), then ONE race
call with Sobol factor nodes. n=2000, dense truth: 5.2s total vs 36.4s
for 1M-draw MC; bulk agreement at MC's noise floor; node
self-convergence 4.6e-4. MC saw zero wins for 82% of the 1563 tail
runners; we price down to 7e-95.

Open (the last piece): deep-tail RELATIVE accuracy across factor nodes
is quadrature-limited (a 1e-20 runner is priced by whichever node lands
in its favorable corner). Fix: exponential tilting of the factor
Gaussian per runner-group (analytic shift + reweight), turning the tail
into a bulk calculation. Then the large-n claim is complete: uniform
relative accuracy at any n, no draws, no Cholesky of Sigma.

## Named-ensemble battery (run_ensembles.py, randomcov 0.1.0)

One-call estimator (k3+m5+blocks20, n=300) vs 2M-draw MC per named
measure. Median absolute error:

  ~1e-5..4e-5   factor+sparse, sparse_precision, wishart, marchenko_
                pastur, spiked, onion, vine, lkj, hierarchical,
                block_equicorr, walk, ar1        (12 of 15: excellent)
  ~1e-4..4e-4   residuals, archakov_hansen, animals  (dense-strong corr)
  ~4e-3         kernel (Matern/RBF point cloud)      (FAILS relatively)

The kernel ensemble is the honest adversary: smooth spatial correlation
has a polynomially-decaying spectrum with no low-rank-plus-blocks
structure, so a fixed m=5 residual captures little. Fix candidates:
adaptive m chosen from the residual spectrum (fit until captured
variance passes a threshold), or the rung-6 deconvolution top-up.
Claims about "general Sigma" now read per-measure, reproducible by seed.
