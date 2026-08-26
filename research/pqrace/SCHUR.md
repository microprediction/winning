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

**Rung 3 -- the tree race** (designed, not built). Recurse the same move:
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
