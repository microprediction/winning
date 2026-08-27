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
