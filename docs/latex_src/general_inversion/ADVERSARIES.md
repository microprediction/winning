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

## The Genz/lpRR analysis (Peter, 2026-09-02): sharper than GHK
The most important prior art is the Marsaglia-Genz-Bretz REDUCED-RANK
rectangle method, exposed as mvtnorm::lpRR/slpRR. The k+1-rank
difference construction: for winner i, Y_j = U_j - U_i has covariance
B_i B_i' + diag(D_{-i}) with B_i of k+1 columns, so ONE factor-race
share is a standard reduced-rank Genz rectangle probability -- PRIOR
ART, decisively (Marsaglia 1963 for the reduction; Butler-Moffitt
1982 for factor quadrature; Stern 1992 for conditional independence
and the univariate-CDF product; slpRR for the scores). NO novelty
claims for: factor conditioning; N-dim to (k+1)-dim reduction; the
CDF product; quadrature of it; smooth one-winner probabilities or
gradients.

What remains distinctive (and matches the manuscript's own "the
contribution is the combination"): the COMPLETE N-winner vector from
one shared product field (per-winner lpRR costs O(RN^2) for the
vector, shared field O(QNL)); the full Jacobian-vector product from
the same field via the two shared hazard sums (Lambda = sum g_j/F_j,
A = sum h_j g_j / F_j) -- no Genz/GHK derivative variant found that
obtains the complete JVP in linear work (slpRR rows stay quadratic);
and the combination into scalable inversion. Present the Laplacian
structure cautiously (roots in random-utility/WDZ and tie-boundary
theory); the claim is the matrix-free linear-work REALIZATION.

Encroachment ranking (Peter's table, condensed): lpRR/reduced-rank
and Butler-Moffitt/Stern = high ingredient overlap; slpRR =
medium-high (derivatives of ONE rectangle); HMR derivatives = medium;
tile-low-rank/hierarchical Genz and Peterlin's doubly blocked Genz
(reported ~100x over some packages; still one rectangle, pivoted
Cholesky) = low math overlap but HIGH wall-time relevance to
benchmark rhetoric; Botev tilting, Ridgway SMC = low core overlap;
Mendell-Elston/MACML/EP = different tradeoff.

FOUR CHANGES for the factor-probit manuscript (it cites Butler-
Moffitt and Stern but lacks the Marsaglia/lpRR treatment the
general-inversion manuscript already has): (1) add the k+1-rank
difference construction and the prior-art acknowledgment; (2)
benchmark per-i lpRR with common scrambled Sobol as the strongest
factor-aware baseline, complete vectors not single shares; (3) add
Peterlin's doubly blocked Genz to the wall-time comparisons; (4)
qualify the website language (docs/converge.html fixed 2026-09-02).
Avoid "first" until the literature review is formalized. Also
consider offering "minimally modified GHK that exploits the cavity"
for readers who prefer that presentation.

## Kill test A: first measured result (2026-09-02, research/
## adversaries/cdf_grad_ghk/)
The CDF-gradient adversary built honestly in ~100 lines of JAX:
factor-state GHK (Kalman state over factors, O(Nk^2) per sample, no
dense Cholesky), scrambled Sobol R=512 common across the grid,
reverse-mode N-vector gradient at the diagonal, trapezoid over L=96.
Two bugs en route worth remembering: mass = 1 is NO diagnostic (it
is identically 1 for any H); and the convention clash (CDF gradient
prices the max, engine min-wins). Measured, TV to the exact engine:
  N=50  k=2: TV 0.0059, 0.43s vs engine 0.008s   (54x engine)
  N=200 k=2: TV 0.0060, 1.85s vs engine 0.032s   (58x)
  N=1000 k=2: TV 0.0043, 9.5s vs engine 0.125s   (76x)
  N=200 k=4: TV 0.0078, 1.86s vs engine 1.81s    (TIED)
The Rao-Blackwell prediction holds at low rank; THE CROSSOVER IS AT
RANK ~4, where the engine's tensor quadrature (Q^k nodes) meets the
adversary's O(LRNk^2). Rank 8+ goes to the adversary unless the
engine gets sparse-grid/QMC factor nodes. This is unoptimized CPU
JAX; a GPU implementation moves the crossover down. For v2: the
"untested" sentence can now cite a measured prototype, and
sparse/QMC factor nodes are promoted from nicety to the identified
defense of the high-rank flank.

## Stata's factor(#) is parameterization, not computation (Peter,
## 2026-09-02)
Modern Stata's multinomial-probit commands accept factor(#), but the
manual shows this only PARAMETERIZES the covariance as I + C'C; the
evaluator still Cholesky-factorizes each (J-1)-dimensional difference
covariance and runs ordinary GHK. "Factor model" in Stata does not
mean factor-conditioned low-dimensional integration. [Verify the
exact manual wording before quoting in v2.] Useful for benchmark
rhetoric: the flagship applied implementation that KNOWS the factor
structure still pays the dense per-alternative cost -- the
structure-exploiting computation is absent from practice, not merely
from theory.

## The high-rank flank holds: QMC factor nodes (2026-09-02,
## run_qmc_defense.py)
The defense was already shipped (winning.factor.qmc_nodes into the
same shared-field pass). Against the CDF-gradient adversary on the
kill-test instance, truth = m=17 QMC (131k nodes):
  k=4: qmc m=10 reaches TV 0.0012 in 0.18s -- ten times faster AND
       six times more accurate than the adversary (0.0078, 1.90s);
       m=13 is 65x more accurate at similar time. The default tensor
       rule (TV 1e-5, 1.87s) caused the apparent rank-4 tie -- the
       crossover was an artifact of the node rule, not the field.
  k=8: qmc m=10 is twice as accurate as the adversary at a tenth of
       the time; every QMC tier dominates. Rao-Blackwell holds at
       high rank once the factor integral is QMC.
Consequence for v2 and the package: at k >= 4 the engine should
default to QMC factor nodes (tracked in issue 21); the adversary
paragraph can now cite both the constructed adversary AND its
measured defeat by the shipped defense.

## The lpRR complete-vector baseline, measured (2026-09-02,
## research/adversaries/lprr_baseline/)
Manuscript change 2 executed: mvtnorm::lpRR (v1.4.1) on the
k+1-column reduced-rank difference representation, one call per
winner, common draws across winners (pseudo-random commons; qrng not
installed -- rerun with scrambled Sobol before the paper table). The
construction was validated against pmvnorm at N=5 to four decimals,
so the reduced-rank equivalence is confirmed operational. Complete-
vector results, k=2, TV against the exact engine:
  N=50:   engine 0.007s | lpRR R=512 TV 0.0063 in 0.05s
                        | R=4096 TV 0.0007 in 0.33s
  N=200:  engine 0.025s | R=512 TV 0.0094 in 0.67s
                        | R=4096 TV 0.0019 in 5.5s
  N=1000: engine 0.124s | R=512 TV 0.0229 in 17.4s
                        | R=4096 TV 0.0061 in 148s
The O(RN^2) analysis is measured: five times N costs lpRR
twenty-six times the time (engine: five), AND its accuracy at fixed
R degrades with N (TV 0.006 -> 0.023 at R=512). At N=1000 the gap is
140x at worse accuracy or 1200x at TV 0.006. Tail finding: at finite
R, lpRR returns -Inf for deep-tail winners (floored and counted in
the harness) -- the per-winner simulator cannot resolve the longshot
probabilities the tail-accuracy claim concerns, the same wall as
TS-MC and LITE. Lab note: the first run showed TV 0.166 flat in R --
a data-handoff scramble (jsonlite auto-parses nested lists; do not
re-wrap), caught because flat-in-R error cannot be Monte Carlo.
