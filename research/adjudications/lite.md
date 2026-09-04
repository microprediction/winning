# Adjudication: winning vs LITE (Track A)
(Agent report, 2026-09-01. Verdict: PURSUE.)

## What LITE computes and how
LITE (Menet et al., AISTATS 2025, arXiv:2501.13535) estimates PoM
p_x = P[F_x >= F_z for all z] for Gaussian F AFTER DISCARDING ALL
OFF-DIAGONAL COVARIANCE (their Assumption 2, "independence
assumption"). Two variants: F-LITE -- a single global threshold
kappa* by binary search so sum_x Phi((mu_x - kappa*)/sigma_x) = 1;
A-LITE -- fits prod_{z!=x} Phi((f - mu_z)/sigma_z) by a Gaussian CDF
via quartile matching. Complexity Theta(|X| log(log|X|/eps)) time,
inputs are means and MARGINAL variances only (confirmed in source,
flite.py). Guarantees (Props 1-2) bound convergence to LITE's own
independence target, not to the true PoM; no error bound under
dependence (stated limitation). Prop 3: closed-form gradients w.r.t.
(mu, sigma) only. Ground truth in their experiments is TS-MC
(exhaustive Thompson sampling), Theta(|X|^3 + |X|^2/eps^2); their
10k-point 1000-dim linear-kernel task: "21 days to 30 seconds" vs
TS-MC. Accuracy: TV to TS-MC of 3.76% (synthetic) to 14.1% (1D GP)
for A-LITE. Repo lasgroup/LITE: JAX, MIT; estimators are plain
functions (nie_poo, cme_poo, vapor_poo); a new estimator slots in
trivially. Nomenclature: TS-MC<->E-TSE, A-LITE<->NIE, F-LITE<->CME.

## Correspondence
LITE PoM <-> win probability (exact for factor/block/tree). TS-MC
ground truth <-> exact O(nLQ) evaluation replaces the 21-day
reference where Sigma is grammar-form (their 1000-dim linear-kernel
task is literally VV'+D; any inducing-point GP posterior is).
Prop 3 diag-only gradients <-> full Jacobians incl. factor params.
Absent from LITE entirely: top-k, rank marginals, inversion.

## Leverage
Covariance-structure gap is total, not partial. Local research/qpo/
benchmark (QM9, N=1000, UCB pool) quantifies it downstream: F-LITE
top-100 recall 0.61 vs 0.94 for rank-4 exact probit; redundant-pair
fraction 3x higher; LITE sits within TV 0.006-0.010 of exact
independence -- LITE approximates its target well, and its target is
the thing the correlation replaces. Exact reference resolves small
probabilities MC cannot (relative MC error ~ 1/sqrt(M p_i)).

## Kill risks
- LITE's flagship domains (SE-kernel GP posteriors) are dense, not
  grammar-form; the factor-fit error competes with LITE's
  independence error (qpo: rank 2-4 suffices on real posteriors;
  rank-1000 linear kernel exceeds tensor-quadrature reach).
- Closed-loop parity: in qpo's twenty-round QM9 loop F-LITE (0.09s)
  converged as fast as exact correlated qPO (58s) -- more accurate
  probabilities may not buy downstream wins on saturating benchmarks.
- JAX/GPU F-LITE wins raw wall-clock on n=1e6 diagonal problems.

## Decisive experiment
Rerun LITE's Table-1 protocol with Sigma restricted to factor/tree
structure (inducing-point GP posteriors; their linear-kernel task at
modest rank), reporting each method's TV to winning's EXACT answer
rather than to TS-MC -- winning as the ground-truth oracle their
21-day baseline approximates, and LITE's independence bias
quantified per structure.

## Positioning
"LITE is a fast, well-converged estimator of the independence
approximation to PoM; for the factor, block, and tree covariances
that dominate applied use, winning computes the true PoM -- and its
gradients and inverse -- exactly, in the same linear time."

## Caveats
arXiv:2501.13535v3 read via summarized fetch (equation numbering not
eyeball-verified against PDF); LITE's asymptotic-exactness
(exchangeability) conditions unverified.
