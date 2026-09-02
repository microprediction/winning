# Adversaries: the CDF-gradient route and structured MVN methods
(Peter's analysis, 2026-09-02, recorded verbatim in substance.
Released version pinned at tag arxiv-2609.01133v1 = paper-r11; all
changes below are arXiv v2 territory. Kill tests tracked in issue
20.)

## The real threat: one common CDF and all its boundary derivatives
With H(b) = Pr(U <= b) and M = max_i U_i,

    dH/db_i = f_{U_i}(b_i) Pr(U_{-i} <= b_{-i} | U_i = b_i),

and on the diagonal b = x 1 this is the density that i finishes at x
with everyone else below. Hence

    p_i = int dH/db_i (x 1) dx                       [the identity]

-- ALL winner probabilities are the integrated boundary gradient of
the maximum CDF. An adversary could: put x on a 1-D grid; estimate
the single rectangle probability H(x 1) by structured GHK/SOV; take
the full N-vector gradient in ONE reverse-mode sweep (a modest
multiple of the scalar cost); integrate over x. Possible complexity
O(LRNk) for a factor-state GHK against O(RN^2 k) for per-alternative
factor RQMC. GHK with analytic derivatives w.r.t. rectangle limits
has existed for years; the construction is assemblable from standard
parts even though no published method combines derivative-capable
GHK, diagonal-threshold integration, and all-N argmax probabilities.

## The revealing fact
Under factor conditioning H(x|F) = prod_j F_j(x|F) and its boundary
derivative is f_i(x|F) prod_{j!=i} F_j(x|F) -- EXACTLY the shared-
field integrand. The CDF-gradient route is not a different identity;
it is a GHK implementation of the same flux identity. Reassuring
(validates the geometry) and threatening (erases any claim that
orthant methods intrinsically need one integral per alternative).

## Would it beat winning in the pure factor case? Probably not at
## low rank and high accuracy
Conditional on F the shared field integrates all N idiosyncratic
shocks ANALYTICALLY; factor-state GHK simulates sequential truncated
values whose conditional expectation given F is that same product.
The shared field is a RAO-BLACKWELLIZATION of factor-state GHK: the
simulation adds randomness that cannot add information. Competitive
regimes for the adversary: higher factor rank (tensor quadrature
cost), very sharp conditional races (fixed Gauss-Hermite struggles),
loose accuracy, GPU constants, approximately-factor covariance.

## Block structure is more exposed
Blocks independent, covariance ARBITRARY within blocks: H(x) =
prod_c H_c(x), and for i in block c the winner density is the local
gradient h_{i,c}(x) times the cavity product of the other blocks'
CDFs. A dense-GHK / minimax-tilting LEAF inside each block plus the
global cavity gives O(L (N/m) cost_GHK(m)) -- linear in the block
count, with richer within-block covariance than the private-factor
block grammar. Not a refutation: it makes GHK a leaf evaluator
inside the hierarchical calculus. For the exact rank-1/2 block
grammar the conditional product stays preferable.

## Hierarchy: near-linear structured MVN is REAL
Hierarchical low-rank decompositions cut per-sample SOV cost to
~O(mN + rN log(N/m)); tile-low-rank QMC with block reordering is
demonstrated to dimension 65,536; Vecchia + minimax exponential
tilting gives linear-cost approximate MVN probabilities beyond
20,000 dimensions. Any claim that "GHK cannot exploit hierarchical
covariance" is FALSE. Correct distinction: standard dense GHK does
not; modern structured SOV/QMC/MET does; those have been used for
ONE rectangle, not the all-share vector. The CDF-gradient identity
potentially bridges that final gap. [Citations to be located and
read before v2 quotes: Genton et al. hierarchical/tile-low-rank
line; Botev minimax tilting; Ridgway; Vecchia-MET paper; EIS.]

## Arithmetic scaling is not the whole story
Ridgway: GHK is sequential importance sampling in a state-space
model; normalized variance can grow exponentially with dimension
(AR(1) example); SMC repairs it. Botev's minimax tilting has
vanishing relative error in an important regime. EIS contains GHK as
a special case. The strongest hierarchical-regime threat is
structured SOV-QMC / minimax tilting / particle-SMC / Vecchia-MET,
not vanilla GHK.

## Our exposed flank, in our own words
The block/nested/tree code uses fixed Gauss-Hermite nodes; the
sharpness warning marks where the conditional integrand approaches a
step (recorded sharpness-18 example: ~5e-2 TV at nine nodes, and
raising the order does not converge cleanly). Adaptive importance
sampling / tilting / SMC could win exactly there. The right
comparison in that regime is adaptive SMC/MET, not a bigger fixed
tensor.

## What remains genuinely protected
Low-rank factor at high precision (Rao-Blackwell argument);
non-Gaussian formula bases and lattice atoms (Gaussian machinery
does not natively cover them); ALL single-removal counterfactuals
(one diagonal CDF gradient gives current shares, not the N removal
re-pricings; the cavity gives those directly); repeated inversion
(deterministic fixed construction vs approximation error carried
through every step); the explicit graph-Laplacian Jacobian
(positivity, reduced SPD system -- stronger than differentiability).

## Claims discipline for v2
DO NOT claim: GHK intrinsically cannot exploit factors/blocks/
hierarchy; any orthant method must price one orthant per
alternative. DO claim: for supplied low-rank-plus-diagonal
covariance at low rank, factor conditioning yields a deterministic
all-share calculation integrating the idiosyncratic dimensions
analytically; standard per-alternative GHK and factor-conditioned
RQMC retain quadratic all-share cost while the shared field is
linear. ADD: structured MVN methods greatly accelerate an individual
rectangle under block/hierarchical covariance; whether a
diagonal-CDF-gradient implementation turns them into a competitive
all-share evaluator is untested.

## The three kill tests
A. cdf_grad_ghk (HIGHEST PRIORITY): p = int grad_b Phi_N(b)|_{x1} dx
   via factor-state GHK (no dense Cholesky), common scrambled Sobol,
   reverse-mode differentiation, common adaptive x-grid; then add
   minimax tilting. N in {50, 200, 1000, 5000}, k in {2,4,8,16};
   vary sharpness, share concentration, target log-share accuracy.
   (Peter's exploratory version: identity verified numerically, mass
   near one, not competitive at modest N, prototype unoptimized.)
B. block_leaf_met: arbitrary dense within-block covariance, m in
   {4,8,16,32}; dense GHK / QMC-GHK / minimax tilting leaves; global
   cavity combination. Does the hybrid extend the block method
   materially at linear block scaling?
C. tree_smc: depth/branching/sharpness grid; fixed-node tree
   quadrature vs adaptive factor-node rules vs sequential GHK vs
   particle SMC vs tilted/Vecchia -- focused on the sharpness-
   warning regime.
