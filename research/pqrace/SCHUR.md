# The Schur race: block covariance by recursive conditioning

Peter's Schur complementary portfolio construction splits a covariance into
blocks, solves each block's sub-problem against a Schur-adjusted environment
(the conditional covariance A - gamma B D^-1 B'), and interpolates with gamma
between the hierarchical end (independent blocks) and full Markowitz. The
race admits the same architecture, and the correspondence is tight enough to
carry the intuitions both ways.

## The rungs, all validated against Monte Carlo

**Rung 1 -- the block race** (`blockrace.block_race`). Clustered covariance
(one shared effect per block, arbitrary loadings, heteroskedastic diagonal).
Across-block independence factorizes the field product EXACTLY by cluster:

    G(x) = prod_c G_c(x),   G_c(x) = E_a[ prod_{j in c} F_j(x - v_j a) ]

one 1-d quadrature per cluster, and the winner's own cluster handled by
leave-one-out inside its block -- cavity division at the block level, which
is the Schur move (condition on the block, integrate it out). Cost
O(N L Q_a) for ARBITRARILY MANY blocks: the geometry a global low-rank
factorization represents worst costs the same as rank one.
Validated: N=300, 40 clusters, rho=0.7: TV vs 2M-draw MC = 0.0031 against
MC self-noise 0.0036, corr 0.99998, 67 ms.

**Rung 2 -- the nested race** (`blockrace.nested_race`). Blocks whose
cross-covariance is carried by one rank-1 coupling:

    Sigma = gamma^2 g g' + blockdiag(v_c v_c') + D

gamma interpolates exactly as in the portfolio construction: 0 = independent
blocks (the Harville/HRP end), 1 = full nested covariance (the
Markowitz end). Cost: Q_f outer nodes x the block field assembly.
Validated: TV 0.0025 vs MC noise 0.0033, corr 0.999993, 1.4 s; the coupling
is doing real work (TV between gamma=0 and gamma=1 boards: 0.25).

**Rung 3 -- the tree race** (`blockrace.tree_race`, BUILT and validated). Recurse the same move:
blocks of blocks, one rank-1 coupling per split -- an HODLR-style
hierarchical covariance. The field integral is then message passing on the
tree (each node's field is a function of (x, ancestor shift)), giving
O(N L Q log C). This is the full Schur-portfolio analogue, and it prices a
race whose covariance is hierarchical at any depth without ever forming it.

## Why this matters beyond retrieval

Every clustered-competition problem this repo has touched hits the same wall:
analogue series in molecular libraries, near-duplicate candidates in
retrieval shortlists, running-style groups in racing, sibling variants in
A/B tests. Global low-rank is the wrong geometry for all of them; the block
and tree races are the right one, at the same cost. And gamma is not
decoration: the portfolio finding that intermediate gamma is more robust
than either end has a direct racing analogue worth testing -- shrink the
coupling when it is estimated from noise, exactly as shrunk_cavity shrinks
tilts.


## The inversion (completing the generalization)

`block_abilities_from_probabilities`: given a win vector, recover centred mu
under block or nested covariance -- the block-structured analogue of the
winning package's factor inversion, closing the loop so the Schur race is a
full forward/inverse pair like every other covariance class in this
programme. Damped log-space fixed point on the exact forward map; eta adapts
by backtracking.

Validated round trips: block (N=200, 30 clusters, rho=0.65) recovers mu to
7e-9 in 4.4 s; nested (global coupling + blocks) to 4e-6 in 49 s. And the
distortion measurement that motivates it: inverting block-generated
probabilities under an ASSUMED-independent model mislocates abilities by up
to 1.31 against a true spread of 5.96 -- 22% of the field's range. Ignoring
known block structure at inversion time is not a small error.


## Done properly: the exact Jacobian, and hybrid Newton

Per Peter: the black-box fixed point was the lazy version. The proper one
uses what the forward pass already computes -- and the Jacobian inherits the
Schur structure of the model:

    same block   J_ij = -int dx sum_a w_a f_i f_j exp(S_c - lF_i - lF_j) R_c
                 (i and j coupled through the shared block effect per node)
    cross block  J_ij = -int dx h_i h_j G/(G_c G_d): a GRAM over lattice
                 points -- block-diagonal plus a factored field coupling,
                 never an N x N surprise
    diagonal     rows sum to zero (a common shift moves nothing)

`block_race_jac` returns p and the exact J from one pass; validated against
finite differences to 4e-5 (FD noise). `block_invert_newton` solves in
log-residual space with the ones-projector fixing the gauge.

Globalization mattered, measurably: pure Newton from the Luce start diverges
under strong correlation, and so does an UN-damped fixed point -- the
adaptive (backtracking) fixed point run to a loose tolerance is the correct
globalizer, after which Newton contracts quadratically (machine precision
from |mu err| <= 0.5 in 3-4 steps).

    hybrid (adaptive FP to 0.2, then Newton):  0.8 s to 1e-14
    fixed point alone:                          4.4 s to 2e-10

5x faster and four orders tighter. The same pattern as one_pass_polished in
exotics: the cheap estimator is the initializer that makes the exact solver
well-posed, not the answer.


## Rung 3, measured

The tree race runs as designed: an upward pass building subtree fields
G_t(y) = E_a[prod_children G_c(y - lam_t a)] and a downward pass
distributing outside-fields R_c(y) = Smooth[R_parent] * prod_siblings, with
the leaf step exactly block_race's under R. Uniform per-node strengths are
what keep every message a function of one lattice variable; leaf clusters
retain per-member loadings. Validated:

- 3-level hierarchy (2 supergroups x 5 clusters x ~20 members): TV 0.0017
  vs a 2M-draw MC whose own noise is 0.0025, corr 0.999998, **50 ms**.
- Structural invariance: adding a strong COMMON root effect (lam = 1.5)
  moves the board by 2.8e-7 -- the argmax cannot see a shared shift, and the
  message passing respects that to numerical precision.
- Asymmetric depth-4 tree with heterogeneous strengths: TV 0.0022 vs MC
  noise 0.0025, corr 0.999996.

The ladder is complete: block (rung 1), nested (rung 2), tree (rung 3),
plus the exact Jacobian and hybrid Newton inversion for rung 1. Remaining
gaps, recorded rather than hidden: the tree Jacobian (same Gram-through-
messages structure, one level up) and per-leaf loadings on INTERNAL node
effects, which would break the one-variable message property and need the
2-d message tables the uniform restriction avoids.


## Inversion for all three rungs

`invert_race` is the generic hybrid (adaptive fixed point into Newton's
basin, then Newton on the log residual, with a monotone fixed-point fallback
so a failed Newton step never strands the iterate). Jacobians: block exact
(one pass); nested exact for free (the nested race is a finite mixture over
the global node, so J is the mixture of block Jacobians at shifted mu); tree
approximate (same-cluster exact under the message R_c, cross-cluster by the
Gram h_i h_j R_c R_d / G_root -- exact in the flat case) with the residual
always measured on the exact forward map.

Round trips (N = 200):

    block   3 Newton steps   1.1 s   residual 9e-14
    nested  3 Newton steps   7.4 s   residual 1e-12
    tree   converged          19 s   residual 4e-5 (approx J: superlinear,
                                     not quadratic -- as expected)

Two failures on the way, both instructive. The monotone fallback exists
because a failed Newton backtrack previously ABORTED the solve. And the
nested case exposed the resolution-floor lesson in a sharper form: the
target board contained probabilities down to 2.5e-38 -- not zeros -- and the
un-floored Luce start inherited a mu-spread of ~86 from them. A probability
below resolution BOUNDS an ability; it does not measure one. `invert_race`
floors targets at max(1e-14, min-positive/1000) and labels the floored
entries as upper bounds.

Provenance, honestly: the Luce start, damped log-space fixed point, and
centred gauge are the original factor inversion's ideas reused; the full
Schur-structured Jacobian, monotone hybrid and resolution floor are new here
and could flow back. NOT yet borrowed and worth it at scale: the package's
coordinate-Newton with analytic per-coordinate slopes from the same pass
(no N x N matrix, ~10 forward equivalents), with the full Jacobian reserved
for final polish or for when J itself is wanted.
