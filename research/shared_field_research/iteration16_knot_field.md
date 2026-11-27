# Iteration 16: local polynomial boundary meshes, global shared cavities

## Repository and scope

Read main README, CLAUDE.md, winning/factor/blocks.py (_block_max_r),
winning/factor/races.py (_fit_cov and _race_dense), tests/test_blocks.py,
research/general_sigma/NOTES.md and papers/general_inversion/CLAIMS.md.
AGENTS.md absent. Main base reported by draft #1093: 898a9f685bec35f88f6eeef10227bd6d302d7f83.
Previous research head 20b3bff55558e5b18f55327780a2feca6615c5c2.
The main block kernel still integrates local factor nodes with private
Gaussian noise and products of survival functions; dense degraded fits
can route to GHK (which adds a ridge on failed Cholesky). The existing
residual-as-extra-factors ladder was reviewed; this is not a covariance
fit or residual promotion. Open PR search for shared found #1093 only.
Re-read shift_field.py, test_shift_field.py and iteration 15. This is
a numerical improvement of the exact signed planar-contrast grammar,
not a new covariance class or an arbitrary PSD solution.

## Conditional convolution

In a block X_j = mu_j + R_j Z + sigma Y, Z standard planar and Y
independent standard normal. Let r=R_i, s=||r||, d=sqrt(s^2+sigma^2),
e=r/s, q perpendicular, and define z=(t-mu_i)/d. Conditioning on
X_i=t rotates the Gaussian coordinates to

    e'Z = (s z - sigma u)/d,   Y = (sigma z + s u)/d,
    q'Z = v,

where u,v are independent standard normals. The unshifted winner
performance is x=mu_i+s^2 z/d - sigma*s*u/d.
Every other contestant restricts v via an affine bound in x:

    v (R_j q) > (1 - R_j e/s)(x-mu_i) + mu_i-mu_j.

Hence the feasible interval (ell(x), upper(x)) is piecewise affine
and its switches occur at pairwise crossings of the affine bounds.
For block observation h'Z + score*Y (h=R' beta, score=sigma sum beta),
set A=(s e'h+sigma score)/d, B=(-sigma e'h+s score)/d, H=q'h.
The exact boundary density at frequency k is

    g_i(t,k) = phi(z)/d * E_u[
      exp(i k (A z+B u)) interval_CF(ell(x),upper(x),k H)].

This avoids an FFT and retains every positive local common-shift
variance, no matter how small. At sigma=0 the u integral vanishes
(B=0 for the specified score) and the analytic interval transform
is recovered. Smooth u rows use Gauss-Hermite; rows near switches or
steep normal thresholds use Gauss-Legendre split at the switches and
thresholds in [-pad,pad]. All rules converge as order increases; no
classification discards covariance. The method is not an analytic
polynomial convolution of the unsmoothed density as proposed last run:
it evaluates the exact conditional convolution at selected nodes.

## The crucial improvement: separate local and global meshes

Naively putting every boundary knot on one shared grid and evaluating
every expensive complex local transform at every node was too slow.
Initial version at 24 near-singular contestants: 4136 shared nodes,
495528 conditional nodes, 3.40 s at coarse resolution; 4242 / 869350
nodes, 8.09 s at moderate resolution. Refinement agreed well but
this was not competitive.

Instead construct a separate knot-resolved mesh for each runner.
For each local interval interpolate g_i(t,k) with a degree (q-1)
polynomial through q Gauss-Legendre nodes, in a Legendre basis.
The same polynomial yields an EXACT polynomial antiderivative.
Specifically V is the Legendre Vandermonde at the Gauss nodes,
C=V^{-1}; at a point with panel coordinate x, the density weights
are P(x)'C and the remaining-panel tail weights are the integrated
basis from x to 1, times the panel half-width. Reverse cumulative
panel integrals give the rest of the tail. Precompute these two
transfer matrices for the union of the local knots; reuse at every
frequency. Sum the individual tails within each block to obtain S_B,
then prefix/suffix products on the union grid give all cavities.
Use Gauss-Legendre on the union panels for the final probability
integration. No FFT, periodization, aliasing, or extra covariance
error. No clipping or normalization. Integrating each local
interpolant exactly does not imply exact integration of the
PRODUCT of fields; the latter has greater polynomial degree and
remains a separate quadrature error.

With Q observation nodes, q local polynomial order, M union nodes,
L_i local Gauss nodes, J_i average number of conditional nodes per
local node, expensive transforms cost O(Q sum_i L_i J_i), not
O(Q M sum_i J_i). Interpolation/tail transfer cost O(Q n M q),
cavities O(Q n M); memory O(n M q + sum_i L_i J_i + n M batch).
Union size can grow with n and block size; no overall linear-in-n
claim. All-pair crossing enumeration is still O(b^3) per b-runner
block and previous hull pruning not used. Future optimization:
lazy local transfer/streaming, remove inactive crossings, reuse
geometry at a block level, or exact outer polynomial product
integration at sufficiently high order (degree <= B q-1).

## Convergence and ties

Same exact target as iteration 15: C0 = blockdiag(R R' + sigma^2 11'),
target Sigma=C0-C0 beta beta' C0, with beta'C0 beta<=1. Input
explicit roots, no small singular direction discarded. For zero
observation noise require at least three active independent blocks;
then conditioning cannot make a nonzero pairwise contrast vanish.
Proportional planar rows and zero planar marginals are refused
(including permanent ties). No claim about the general API's tie
policy; this prototype avoids those cases. At positive observation
noise a nondegenerate contrast remains nondegenerate. Roundoff-only
rescaling of observation variance above 1 as in the previous code.

For fixed finite observation cutoff K and pad, the bounded continuous
interval characteristic function makes the local conditional integral
converge under increasing GH/GL order (GH quadrature measures converge
weakly to the moment-determinate Gaussian; GL integrates a continuous
integrand on finitely many split compact panels). Conditional tail
omission in split branch <=2 Phi(-pad) times marginal density.
The smooth branch integrates all u, so the switch between rules can
introduce a jump of at most that tail size in the limiting integrand.
Marginal t truncation adds at most 2 Phi(-pad) per runner; the
implemented local live window may produce tiny jumps there too.
At fixed q, local interpolation has a bounded Lebesgue constant,
and as maximum panel width step->0 converges in L1 for piecewise
continuous bounded functions, with the above tail error. For
fixed K,pad and finite q, first refine inner and frequency rules and
mesh jointly (diagonal sequence; local uniform convergence on compact
pieces, or choose sufficiently accurate inner order for each finite
mesh); polynomial primitives converge uniformly by L1 convergence.
Cavity product errors follow telescoping; bounded exact fields have
modulus <=1 and interpolants remain bounded on finite windows. Outer
quadrature converges with mesh refinement for piecewise continuous
limits, independent of the polynomial-product finite resolution
error. Then increase pad and K so K Phi(-pad)->0 and omitted
frequency tails vanish. At positive observation noise Gaussian
decay suffices; at zero noise two untouched active blocks provide
the previous O(k^-2) integrated absolute tail bound. This establishes
a refinement path, not a finite-resolution certificate. No structural
or sampling error within scope. Tiny negative estimates remain
possible from interpolation/Fourier cancellation; do not use as a
log-probability engine. Near-coincident floating-point knots and
very large common shifts need more robust treatment.

Primary literature checked: Genz (1992), Numerical Computation of
Multivariate Normal Probabilities, DOI 10.1080/10618600.1992.10477010,
publisher abstract and author manuscript abstract; previous
iteration checked Dutt (1973), DOI 10.1093/biomet/60.3.637 for
Gaussian Fourier inversion. Gaussian conditioning, polynomial
interpolation, Gauss quadrature and integration of polynomials
are standard. The contribution is reuse of cheap transferred local
boundary polynomials and their primitives within the all-winner
signed shared-field computation, not a claim of novelty for the
individual numerical methods.

## Experiments

Local NumPy 2.3.5, SciPy 1.17.0, single BLAS thread. Five standalone
tests passed in 3.97 s. Independent four-contestant trivariate
orthant arcsine formula: max absolute error 4.22e-15. Combined
permutation/translation/scale test at atol 1e-11; exact local and
global singularity on four equilateral blocks with positive shares,
block symmetry at 2e-8; single-block common-shift invariance at
1e-10. No full-suite CI or general common-shock API claim.
Source blob SHA 2c63952cea51ab0bee54f2688e85365bcb7c1da3 and test
82d1bd9e347c17f79cc1de5765c3af853ef6063b verified equal locally
and remotely.

Reproduce in this directory:
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest -q -s test_knot_field.py
    python -u test_knot_field.py 24 near
    python -u test_knot_field.py 24 arbitrary
    python -u test_knot_field.py 60 near

Fixtures match iteration 15 random_fixtures at n=24, seed 926;
random orthogonal roots diag(1.3,.8,.003) for near case, arbitrary
iid normal roots for broad shift case; random means N(0,.5^2) and
observation variance 1-1e-8. Heterogeneous mixed-sign covariance,
with very small winning probabilities. Compiled SciPy _qmvn Genz/SOV
greedy ordering and CBC randomized lattice, eight batches per
winner, normalize full vector, four independent seeds 0..3 and
100..103 for 32768. Timings include setup, no amortization. No ridge.
All field RMSE figures below relative to most refined field
(24 near: step=.25,q=8,inner=16,Q=48,K=24; others: .35/8/12/40/20).
No independent large-dimensional certificate; high-budget Genz
means support the results at their much larger uncertainty.

case | method step/q/inner/Q/K or budget | seconds | RMSE | pooled Genz SD
24 near | field 1/4/6/24/12 | .2110 | 6.57e-6 | -
24 near | field .5/6/10/32/16 | .5229 | 9.13e-10 | -
24 near | field .35/8/12/40/20 | 1.1096 | 1.43e-12 | -
24 near | field .25/8/16/48/24 | 1.8210 | ref | -
24 near | Genz 2048 | .1176 | 2.08e-4 | 2.19e-4
24 near | Genz 8192 | .3251 | 1.39e-4 | 1.51e-4
24 near | Genz 32768 | 1.1330 | 5.27e-5 | 5.31e-5
24 arbitrary | field 1/4/6/24/12 | 1.976 | 4.62e-6 | -
24 arbitrary | field .5/6/10/32/16 | 6.556 | 2.06e-10 | -
24 arbitrary | field .35/8/12/40/20 | 11.815 | ref | -
24 arbitrary | Genz 2048 | .1284 | 1.13e-4 | 1.22e-4
24 arbitrary | Genz 8192 | .3316 | 4.05e-5 | 4.33e-5
60 near | field 1/4/6/24/12 | .6938 | 8.01e-6 | -
60 near | field .5/6/10/32/16 | 1.8055 | 1.61e-10 | -
60 near | field .35/8/12/40/20 | 3.6735 | ref | -
60 near | Genz 2048 | 1.1465 | 8.75e-5 | 7.17e-5

24 near moderate: 1602 union nodes, 6708 total boundary nodes
(sum over 24 local grids), 40620 expensive conditional nodes, vs
219140 for shared-grid-only precursor at exactly same union mesh,
and 2.026 s for that precursor. The separate meshes reduce costly
conditional node count by 5.39x and runtime by 3.87x; moderate
RMSE vs refined 9.13e-10 instead of 1.20e-11 (interpolation cost).
Previous iteration's FFT on same fixture at M8192 still had mass
defect 2.13e-6, and at M16384 the defect was 2.25e-7; note that
those are earlier measurements, not this run's timed baseline.
New raw mass defects coarse/mid/fine/final: -8.36e-5, +1.44e-8,
-2.21e-11, +4.31e-14; final max refinement difference 3.39e-12.
Minimum numerical p at moderate -3.36e-80 (tiny cancellation, not
clipped); final -1.55e-91. 60 near moderate mass defect -4.21e-9,
final -1.27e-11, max refinement difference 6.13e-10; final minimum
-2.01e-48. Coarse mass defects too large to use for strict
normalization. 24 near higher-budget Genz mean RMSE 2.57e-5,
comparable to its mean uncertainty; 60 near 2048 mean RMSE
6.16e-5. 60 higher-budget Genz and 24 arbitrary 32768 were not
completed in this run, so no claim there. The latter cases were
tested through the first three field levels only; bench command
does an additional refinement. Broad-shift conditional convolution
is much slower than iteration 15's active-spectrum algorithm:
use a hybrid strategy in future. No deep-tail relative accuracy
claim, no certified error bars, no universal speed claim.

Next: mix active-spectrum broad-shift boundaries and knot-local
narrow-shift boundaries in one field, retaining exact covariance.
Consider choosing meshes per block and using exact outer product
quadrature or error estimates independent of mass conservation.
Then explore multiple global negative modes and automatic signed
decomposition. General PSD target and production API integration
remain open.
