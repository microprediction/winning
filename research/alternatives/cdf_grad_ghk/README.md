# The CDF-gradient evaluator: the strongest alternative we could build

A reference implementation of the best GHK-family all-share evaluator
we know how to construct for factor covariance, built so that the
paper's comparison is against the strongest alternative rather than a
textbook protocol, and so that a referee can run it.

## What it computes

All N winner probabilities from ONE rectangle estimator. With
H(b) = P(U <= b) and M = max_i U_i,

    p_i = int dH/db_i (x 1) dx,

the integrated boundary gradient of the CDF of the maximum along the
diagonal. `run_cdf_grad.py` estimates H by GHK and takes the whole
N-vector gradient by reverse-mode differentiation.

## The three upgrades over published GHK protocols

1. **Factor state.** For Sigma = VV' + diag(D) the sequential
   conditioning runs a Kalman state over the k factors: O(N k^2) per
   sample, no dense per-alternative Cholesky. (Stata's factor()
   option parameterizes the covariance this way but still evaluates
   with dense per-difference GHK.)
2. **All alternatives at once.** One reverse-mode sweep per grid
   point returns the full gradient, so the complete share vector is
   linear in N -- against the O(N^2) of every per-alternative
   protocol, including reduced-rank Genz (mvtnorm::lpRR) called once
   per winner.
3. **Common scrambled Sobol** across all grid points, the smoothness
   needed for calibration by inversion.

We found no published method combining these three; the assembly
appears to be new. It is given away here because the shared-field
engine still dominates it, for a statistical rather than an
engineering reason.

## Why the shared field still wins

Conditional on the factors, the alternatives are independent and the
survival product prod_j F_j(x|F) is exact. The GHK draws simulate
precisely those idiosyncratic dimensions, and their conditional
expectation given the factors is the product: the shared field is
the Rao-Blackwellization of this evaluator. Its extra randomness
cannot add information at any rank; rank only changes the factor
integral, which both methods face equally once the engine uses QMC
factor nodes (`winning.factor.qmc_nodes`).

## Measured (run_qmc_defense.py; N = 200, truth = 131k-node QMC)

| rank | engine qmc m=10 | engine qmc m=13 | this evaluator (R=512) |
|---|---|---|---|
| k=4  | TV 0.0012, 0.17s | TV 0.0001, 1.4s | TV 0.0078, 1.85s |
| k=8  | TV 0.0039, 0.19s | TV 0.0006, 1.5s | TV 0.0082, 1.89s |
| k=16 | TV 0.0079, 0.18s | TV 0.0022, 1.5s | TV 0.0103, 1.92s |

The engine dominates the time-accuracy frontier at every rank
tested. Read the trend honestly: the QMC factor error grows with k
faster than this evaluator's, so a genuine contest may appear near
k ~ 32-64 at loose accuracy targets; high-accuracy work belongs to
the field at any rank by the Rao-Blackwell argument. At rank 2 the
engine's margin is 54-76x (results.json, run_cdf_grad.py).

## Limitations, honestly

Gaussian only (the field runs any smooth base); requires the factor
structure to be supplied (as does the engine); unoptimized CPU JAX,
so its constants would improve on GPU without changing the
statistics; the diagonal-gradient route gives the current shares but
not the N removal counterfactuals the cavity yields directly; and
mass(p-hat) = 1 identically, so total mass is NOT a diagnostic --
certify against Monte Carlo argmax counts.

## Reproduce

    python run_cdf_grad.py        # rank-2 head-to-head vs the engine
    python run_qmc_defense.py     # ranks 4, 8, 16 vs QMC factor nodes

Requires jax (CPU) and scipy. Full context: the manuscript-side
discussion in docs/latex_src/general_inversion/ALTERNATIVES.md and
issue 20.
