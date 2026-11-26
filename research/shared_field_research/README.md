# Shared-field research log

## 2026-11-26 — Gaussian precision trees, not latent clade trees

**Status: substantive prototype, opt-in research only.** No existing production
interface is changed. This is a broader exact structured class, not a solution
for arbitrary covariance.

### What was already there

Read main README, CLAUDE.md, winning/factor/races.py, structures.py, blocks.py,
methods/native.py, tests/test_fit_tree.py, research/pqrace/SCHUR.md and
research/general_sigma/NOTES.md. The dense route falls back to QMC-GHK.
The old general-Sigma ladder has already tried CRN indicator control variates,
scalar diagonal splits, positive residuals, adaptive residual factors and
chessboard approximations; avoid rediscovering those as advances. The existing
Tree is a covariance of additive shared clade effects (not a tree precision).
Crucially, GMRFExtremes.jl and research/tridiagonal already implement restricted
transfer-operator passes for chains (and research on banded lifted-state
passes). Do NOT claim the chain as new. The Julia README explicitly refuses
general sparse precision and bandwidth > 1. The new scope here is arbitrary
branching precision trees, and the threshold-adapted Gaussian quadrature
rather than the existing fractional-cell chain lattice. Code/PR search
found no existing arbitrary-branching precision-tree price engine; open
PR #1035 has a different aggregation title (not inspected in detail).

### Model and exact shared field

Each contestant is a vertex, including internal vertices. Write
X_i = mu_i + sd_i Z_i, sd_i > 0, where Z_root ~ N(0,1) and, along each
directed edge, Z_i = rho_i Z_parent + sqrt(1-rho_i^2) e_i with independent
standard normals, |rho_i| < 1. Then Corr(Z_i,Z_j) is the product of edge
correlations on their path. This parameterizes all SPD Gaussian tree
precision graphs (zero edge correlations permit independent components).
Covariance is generally dense and full rank; negative edges are allowed.
The topology differs from Tree's uniform ancestor shifts: a positive AR
path has correlations rho, rho, rho^2 rather than cophenetic levels;
signed branching trees are a useful further extension.

For threshold x let c_i(x)=(x-mu_i)/sd_i. On edge i--j define
m_{i->j}(x,z) as the conditional probability that every performance on
i's side of the cut exceeds x, given Z_j=z. Conditional independence and
reversibility of the standardized normal pair give the recurrence

    m_{i->j}(x,z) =
        integral_{c_i(x)}^infty
          phi((t-rho_ij*z)/sqrt(1-rho_ij^2))/sqrt(1-rho_ij^2)
          prod_{k in N(i) except j} m_{k->i}(x,t) dt.

At the winner's boundary the cavity is exactly the product of all
incoming messages:

    p_i = integral_R phi(c_i(x))/sd_i
                 prod_{j in N(i)} m_{j->i}(x,c_i(x)) dx.

The threshold is shared. Two passes over the edges per threshold compute
ALL n shares, not n distinct orthants. At each threshold use prefix/suffix
products at high-degree nodes, no division by vanishing fields. This is
ordinary sum-product on a tree applied to the winning boundary integral;
no claim of inventing graphical inference. See Kschischang, Frey and
Loeliger (2001), Factor Graphs and the Sum-Product Algorithm,
DOI 10.1109/18.910572; and the repo's own chain engine.

### Refinement and cost

Prototype uses L Gauss-Legendre threshold nodes, M quadrature nodes in
[clip(c_i(x),-B,B), B] for each vertex, and one extra exact boundary
evaluation per vertex. Restrict every standardized marginal to [-B,B].
The omitted total race mass is <= 2n Phi(-B), by union bound; this is
ONLY the truncation error, not a quadrature certificate. On the compact
domain with |rho|<1 kernels are continuous and bounded, the moving
endpoints are continuous, and the outer boundary mask is piecewise
continuous. Positive Gauss-Legendre quadrature converges on these
integrals (polynomial approximation and bounded kernels, induction
over the tree, then outer quadrature). Take M and L to infinity at
fixed B, then B to infinity. No approximation of the covariance.
For a fixed nondegenerate tree the limit is the desired win vector.
Singular edges and zero sd explicitly refused; no hidden jitter.
One-node case exact. A common additive shock is choice-irrelevant but
adding it to the covariance may move out of this grammar; do not fit
the contaminated precision. Conditioning an explicitly supplied common
shock is unnecessary.

Work O(n L M^2), memory O(n L M + L M^2) with batched thresholds; could
stream thresholds to reduce memory. For fixed resolution this is linear
in n, unlike per-alternative GHK. No claim resolution independent of
correlations/ability spread. Input uses O(n) tree parameters; deriving
them from a dense covariance is NOT free and is not benchmarked. A sparse
precision-to-tree constructor is a future step. Raw shares returned;
mass defect is a diagnostic, not an accuracy bound; never normalize away
an unresolved quadrature. Close to singularity the density quadrature
needs many nodes. All equations here use min-wins.

### Reproducible experiments

From this directory:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_markov_tree.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench_markov_tree.py

Local standalone prototype: 14 tests passed in 1.46 s (not repository-wide
CI). Binary closed form; 10 independent three-way bivariate orthants by
adaptive 1-D integration; independent survival products; branching
18-node dense Gaussian MC at 400k paths; inverse covariance sparsity;
permutation, common location, utility rescale, re-rooting, refusals.
One negative-edge case at seed 7 failed at (96,72), error 1.35e-6,
but at (144,96) error 2.31e-10. Do not silently lower the test order.

Single-thread local Python timings, seeded branching binary trees:
means N(0,.6^2), marginal sd U(.8,1.2); mixed edge rho U(-.85,.85)
or strong U(.85,.97). Seed 23. Reference is higher-order same algorithm
(160,112), cross-checked against (192,144), change <= 6.8e-14 for
completed cases; independent dense MC 2^20 paths has rms standardized
errors 1.08, .78, .96 for the rows below. GHK is log-domain scrambled
Sobol with greedy conditional ordering (expected truncated draw
heuristic), four scrambles seeds 0..3. Per-call timing includes ordering
and factorization; tree uses explicit edge parameters and includes grid
setup. No compiled or amortized GHK comparison yet.

| n / edges | tree (L,M) | sec | max abs err | GHK draws/runner | sec | median max abs err |
|---|---:|---:|---:|---:|---:|---:|
|32 mixed|64,48|.034|1.59e-7|1024|.103|2.43e-4|
|32 strong|64,48|.039|6.63e-8|1024|.094|1.42e-4|
|100 mixed|64,48|.117|1.84e-6|1024|1.009|2.82e-4|
|100 mixed|96,72|.411|5.40e-10|256|.616|5.94e-4|

These show advantage over a competent Python SOV baseline in these
specific cases, not universal or matched compiled performance. MC
validates bulk but cannot independently resolve 1e-10. No high precision
independent reference for large n yet. Benchmark stopped after third
case due execution limit; 100 strong branching remains to complete.
Chain-only scratch experiments were also promising, but chain isn't new.

Near-singular stress (seed 99, n=8 binary tree, signed rho with absolute
values U(.98,.995), means N(0,.2^2), sd U(.8,1.2), condition number 1779):
raw mass 1.03443 at (64,48), .99718 at (96,72), .999794 at (144,112),
.99999778 at (192,160), .999999965 at (256,224). Last two vectors differ
about 1.6e-6. This is a warning about fixed defaults, not a victory;
need an independent orthant reference here and adaptive resolution.
No claims for arbitrary precision graphs or extreme tails.

### Next research iteration

1. Improve transition quadrature: analytic terminal-edge integration,
   conditional-normal-coordinate quadrature with interpolation or
   Gaussian cell probabilities, adaptive order by edge sharpness.
   Benchmark against the existing Julia chain kernel and compiled GHK
   if feasible. For near-singularity use independent orthant references.
2. Exact Gaussian covariance = Markov-tree covariance + arbitrary
   low-rank factor: condition on the few global factors, run these two
   passes on shifted means. Distinct from existing latent clade tree.
   Derive and test sparse precision elimination to edge parameters.
3. Generalize to bounded treewidth (separator state, not just a scalar).
   Conditioning a feedback vertex set turns the remaining precision into
   a forest exactly; integrate the conditioned vertices, which are also
   contestants and can win. This is a possible general convergent
   route, but dense graphs require a large separator and the practical
   gain is not established. Explore low-rank-plus-tree covariance
   residual corrections rather than claiming universality prematurely.
4. Preserve this branch/log between runs; no changes to main. If an
   improved direction merits a second PR, prefer extending the existing
   research PR rather than opening duplicate drafts.

## 2026-11-26 — Close the loop: precision cycles and a route to junction trees

**Status: tested research extension on the same PR #1093.** Current main
README, CLAUDE.md, dense dispatcher and general-Sigma notes were rechecked.
The prior chain engine does not solve the closed cycle; no overlapping
cycle PR found in targeted search. This is not a covariance fitting method.

### One shared partition field, all boundary insertions

Let Q be SPD cyclic tridiagonal precision on n>=3 vertices in cycle
order; input is its diagonal and the n cyclic edge entries, with either
sign and arbitrary magnitudes consistent with SPD. Compute marginal
standard deviations s and logdet Q in O(n) using tridiagonal LDL of
the first n-1 vertices, a scalar Schur complement for the last, and
the diagonal inverse recurrence. No dense matrix is constructed.
In standardized coordinates let R=diag(s) Q diag(s), c_i(x)=(x-mu_i)/s_i.
Write the joint density as a normalizing constant times the product
of pair factors

    K_i(u,v)=exp(-R_ii u^2/4 - R_i,i+1 u v - R_i+1,i+1 v^2/4).

At fixed threshold x, quadrature on [max(c_i,-B),B] at every vertex
turns the truncated survival integral into trace(A_0 ... A_{n-1}),
A_i=diag(w_i)K_i (cyclic indices, w_i are positive quadrature weights).
Append one extra node c_i, clipped to [-B,B], with ZERO weight.
The derivative with respect to that node's weight inserts the winner's
boundary, rather than integrating it. This is exact for the finite
quadrature of the other n-1 vertices; it is NOT a numerical derivative
of a moving quadrature rule. With P_i=A_0...A_{i-1} and
S_{i+1}=A_{i+1}...A_{n-1}, the derivative is

    (K_i S_{i+1} P_i)[boundary,boundary].

Two sweeps (prefix products and suffix products) compute all n
derivatives. Multiply by the Gaussian normalizing constant and
1/s_i, mask the winner's boundary outside [-B,B], then integrate x.
No per-alternative orthants. No loopy belief propagation, no fitted
covariance. The trace identity is a standard transfer-matrix/
tensor-network identity, not new graphical inference. See Baz,
Alonso and Perez-Fernandez (2023), DOI 10.1016/j.laa.2022.12.028
for Gaussian cycle models; Robeva and Seigal, *Duality of graphical
models and tensor networks*, DOI 10.1093/imaiai/iay009, for the general
junction-tree connection.

Fixed-resolution work O(n L M^3) and current memory O(n L M^2).
Preprocessing O(n) from explicit precision edges, not from dense
covariance. Positive integrands and compact-domain polynomial
approximation justify convergence of nested positive quadrature as
M,L increase (at fixed B, then B increases). The omitted probability
of any standardized coordinate outside [-B,B] is <=2n Phi(-B);
the bound is for truncation only. The outer boundary mask introduces
finitely many jumps; Gauss-Legendre converges on these bounded
piecewise continuous functions. For any fixed SPD Q all ties have
probability zero. Indefinite/singular Q is refused; no jitter.
Scaled matrix products and exponent shifts help but extreme exponent
cancellation can underflow before the final rescaling. No guarantee
of robust finite-precision convergence at arbitrary condition number.
Raw mass is diagnostic only and not forced to one. Arbitrary permutation
requires reordering into a cycle; rotation and reversal were tested.
An explicitly supplied independent common shock can be ignored by
choice invariance, but adding it to Q^{-1} need not preserve the cycle
precision grammar.

### Experiments

Commands from this folder (single-thread, isolated prototype):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_markov_cycle.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench_markov_cycle.py

14 standalone tests passed in 0.87 s. Sparse marginal variances and
logdet match dense inverse; six independent bivariate orthant
integrations plus a frustrated non-diagonally-dominant SPD triangle
agree. Rotation, reversal, common location, unit scaling, exchangeable
and independent cases, and indefinite refusal tested. Repository-wide
CI not run: local network does not permit cloning, so sources were
inspected via GitHub connector and new files tested in isolation.

Seed 41 per row: mixed edge Q entries U(-.6,.6), diagonal 1 plus
incident absolute edges; strong negative edges -U(.7,1.2), diagonal
.3 plus incident absolute edges; means N(0,.5^2). Reference (112,64)
vs (96,56) max change <=1.34e-9 across all four cases. Four GHK
scrambles 0..3, 1024 nodes per alternative, previous ordered/log-domain
implementation, includes ordering and factorizations; dense C already
provided (its construction excluded). Cycle timing includes O(n)
preprocessing. Errors below against the refined quadrature reference.

| n / edges | cycle (64,32) s | max abs err | ordered QMC-GHK s | median max abs err |
|---|---:|---:|---:|---:|
|32 mixed|.0397|2.95e-7|.1001|1.92e-4|
|32 strong|.0417|9.79e-11|.0959|1.64e-4|
|100 mixed|.1237|2.31e-6|.9670|1.55e-4|
|100 strong|.1167|1.76e-7|.9857|1.91e-4|

At (80,40), 100 mixed: .243 s and 8.01e-8 max error; 100 strong:
.232 s and 3.59e-10. Four independent direct dense Gaussian Sobol
scrambles (2^18 paths each, seeds 100..103) agreed in bulk (mean
vector max differences 2.34e-4, 2.33e-4, 5.16e-4, 5.15e-4);
they do not establish the fine quadrature accuracy independently.
The comparison is single-thread Python, not compiled GHK or an
amortized preordered baseline. It is evidence for these matrices,
not universal superiority. The benchmark also includes 256-node GHK.

Near-singular positive-correlation stress, seed 99:
    n=8; e=-rng.uniform(.7,1.2,n)
    d=.005+abs(e)+abs(np.roll(e,1)); mu=rng.normal(0,.2,n)
Q condition number 808.5. Four ordered GHK scrambles at 2^15 per
runner mean sum 1.000002916. Cycle (64,32) mass 1.000010797, max
difference from GHK mean 5.55e-6. Cycle (96,64) mass 1+1.8e-13,
max difference 1.21e-6, and (128,96), (160,144), (192,192) agree
with that GHK mean to the same 1.21e-6. This is a useful stress,
not a general near-singular guarantee.

### Generalization and next experiment

The zero-weight boundary insertion is the useful unifying idea.
For ANY Gaussian precision graph, the shared survival partition
function can be represented as a product of pair potentials and
unary quadrature weights. Reverse-mode differentiation of a single
variable-elimination contraction with respect to each boundary weight
extracts all winners, even though those weights are zero. Equivalently
use a calibrated junction tree of tensor messages; no division by
a zero boundary weight. For treewidth t, elimination with explicit
clique tables costs O(n L M^(t+1)) at fixed resolution, assuming a
supplied bounded-width elimination order. This is an exact integral
representation with convergent quadrature for SPD, but on dense
graphs t=n-1, so not a practical universal competitor to GHK.

Next: (a) implement a generic bounded-width contraction with reverse
boundary insertion (small width 2 and 3); (b) compress the closing
edge's quadrature kernel by adaptive low-rank separation, reducing
cycle work toward O(n L M^2 r), but rigorously monitor signed
cancellation and approximation error; (c) for arbitrary dense
matrices pursue structured backbones with unbiased residual
corrections, not the already failed fixed-rank approximations. Limit
memory by streaming outer thresholds; current cycle reference
allocates large batched matrices. Keep this isolated from production.
