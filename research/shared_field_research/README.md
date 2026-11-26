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


## 2026-11-26 — Arbitrary bounded-width precision graphs

**Status: tested extension of PR #1093, not a universal dense competitor.**
Main README, CLAUDE.md, factor/races.py, general-Sigma research notes, and
the previous tree/cycle code and PR were rechecked. Main's dense route
still includes GHK; the old residual-factor fitting ladder already has
substantial negative results. This iteration implements the generic
elimination idea rather than fitting another covariance grammar.

### Conditional-factor elimination and reverse cavities

Use a supplied elimination order, and perform sparse Gaussian Schur
elimination without dropping fill. At vertex i let S_i be its remaining
neighbors in the filled graph, a clique; pivot d_i>0 and regression
b_i=-Q_i,S_i/d_i are the conditional precision and coefficients in the
CURRENT Schur complement. For centered X,
X_i | X_{i+1:n} ~ N(b_i X_{S_i},1/d_i). Thus the joint density is a
product of normalized conditional Gaussian kernels, with scopes
B_i={i} union S_i. Unlike raw pair potentials these have nonpositive
quadratic exponents. Selected covariance entries can be computed in
reverse order: C_{i,S_i}=b_i C_{S_i,S_i},
C_ii=1/d_i+b_i C_{S_i,S_i} b_i^T. All needed pairs have been retained
because S_i is a clique. Sparse elimination and selected inversion
cost O(n t^2) arithmetic for maximum filled width t, given the order
and sparse input (excluding permutation validation/sorting and CSR
conversion, dense input scanning and finding the order). No dense
inverse inside the solver.

Standardize using these marginal variances. At threshold x, use the
same truncated positive quadrature as the previous cycle work, with an
extra zero-weight boundary node. Attach bucket i to the smallest member
of S_i; if S_i empty it is a root. Every child's separator lies in
the parent's bucket scope. In the forward direction form

    H_i = K_i * product_{j child of i} m_j
    m_i = sum_{z_i} w_i(z_i) H_i

where K_i is the conditional density in standardized coordinates.
In reverse, let A_i be the derivative of the complete scalar
partition function with respect to the outgoing message m_i.
Root A_i is the product of all other root messages (disconnected
components are independent). Child context A_j is the sum over
B_i minus S_j of A_i K_i w_i times all sibling messages except j.
The boundary derivative is sum_{S_i} A_i H_i(z_i=c_i(x)).
Multiply by 1/sd_i and integrate in x to price winner i. This is
the finite-quadrature partial derivative with respect to the boundary
weight, not a derivative of quadrature nodes or normalized messages.
The code keeps explicit logarithmic scales and uses prefix/suffix
products without division by weights or zero-valued messages.

This is exact for the finite product quadrature, and has the same
compact-domain convergence argument as previous iterations: at fixed
span B, positive product Gauss-Legendre quadrature converges for the
continuous joint density on the bounded integration domains, then
outer quadrature converges (boundary mask has finitely many jumps).
The omitted total probability is <= 2n Phi(-B); increase B after
resolving the quadrature. SPD only, no ties, no hidden ridge, no
covariance fit. A dense graph with width n-1 is formally covered but
exponential and explicitly refused by the default width cap. The code
does not solve singular covariance. A good order is the caller's
responsibility. Adding a common shock to the covariance can destroy
precision sparsity; explicitly specified common shocks can be ignored
by choice invariance. Complexity O(n L (M+1)^(t+1)) for fixed t, with
batched threshold memory O(n batch (M+1)^(t+1)). Prefix/suffix means
even high-degree buckets need only linear-in-degree contractions.
The Gaussian factors are not individually log-scaled; finite precision
can underflow at sharp conditionals or many children. No automatic
resolution or error certificate; raw probabilities are never renormalized.

Prior art: Darwiche, *A Differential Approach to Inference in Bayesian
Networks*, UAI 2000 / JACM 2003, https://arxiv.org/abs/1301.3847,
explicitly differentiates a network polynomial to obtain all marginals.
Do not claim reverse-mode graphical inference or zero-weight evidence
derivatives as new in themselves. The contribution here is the
winning-specific all-boundary integral and practical Gaussian
precision solver, including sparse selected inversion.

### Experiments

Local isolated prototype, not full repository CI (network cloning
not available in prior runs). Commands from this directory:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_markov_width.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench_width.py

12 tests passed in 1.52s locally (benchmark comparator duplicated the
previous tree benchmark locally; committed benchmark imports that same
GHK function). Five independent bivariate contrast orthant integrations
for n=3: largest error 1.17e-14. Selected inversion compared with dense
inverse for width 1,2,3; cycle with nontrivial fill, arbitrary
permutation with corresponding order, scale/location invariance,
independent disconnected unequal-variance race, width-limit and
singular/indefinite refusals all tested. No production dispatch changes.

Seed 42 random branching k-trees (not lifted banded chains), mixed-sign
precision edges U(-.65,.65) and diagonal 1+sum absolute edges; strong
negative edges -U(.7,1.2) and diagonal .15+sum absolute edges; means
N(0,.5^2). Single-thread Python; GHK includes greedy ordering and
contrast factorization, 1024 Sobol draws per alternative, four
scrambles seeds 0..3. Precision input given, covariance inverse for
GHK excluded; each shared-field timing includes sparse prep. Error
against highest-resolution same-method reference, not a certificate.

| n,width,mode | (L,M) | shared sec | max abs error | GHK sec | median GHK max abs error |
|---|---|---:|---:|---:|---:|
|32,2,mixed|48,16|.148|5.57e-6|.132|1.47e-4|
|32,2,mixed|64,24|.587|3.90e-8|.132|1.47e-4|
|100,2,mixed|48,16|.617|9.88e-5|1.335|2.17e-4|
|100,2,mixed|64,24|2.171|1.87e-7|1.335|2.17e-4|
|32,2,strong|64,24|.724|1.13e-8|not completed|not completed|
|16,3,mixed|48,14|.867|6.31e-6|.0354|1.19e-4|

Reference levels 112,44 (32 mixed), 88,36 (100 mixed and 32 strong),
80,24 (16 width 3). Preceding finer-grid changes 1.06e-11, 3.49e-9,
3.56e-13, 2.61e-8 respectively. Direct dense Gaussian Sobol (4 scrambles
2^17 each) corroborated bulk for 32 and 100 mixed, max differences
2.10e-4 and 2.02e-4 from refined quadrature; not an independent
fine-accuracy reference. Benchmark timed out before strong GHK and
width-3 MC portions, so no claims there. Width 2 is competitive
in these examples, particularly at high accuracy, but width 3 is
uncompetitive at tested small n. The initial log-space implementation
was substantially slower: 32 mixed at (48,16) .431s vs .148s,
same answer; use positive scaled contractions for moderate cases.
No compiled or amortized GHK comparison.

Near-singular stress: graph(8,2,seed=99,strong=True), subtract .145 I
from Q, multiply mu by .25, condition number 1368.83. Four ordered
GHK scrambles with 2^14 samples/alternative, max spread 8.17e-6.
(L,M)=(64,24): mass 1.002577, max error vs mean GHK 2.94e-3;
(80,40): mass .999993987, max difference 1.39e-6;
(112,64) and (144,80): mass essentially one, max difference
1.24e-6, and errors at last two refinements agree to displayed digits.
Batch=2 for this stress. Fixed defaults are not safe in this regime.

### Next research priorities

1. Do not simply increase graph width: the explicit tensor cost becomes
   uncompetitive already at width 3. Exploit analytical leaf elimination:
   a leaf's forward message is a univariate normal interval CDF
   conditional on the separator; its boundary factor is a density.
   This removes an entire quadrature axis for leaf buckets and preserves
   the exact target (with truncation). More generally integrate or
   compress the conditional Gaussian kernel in a separator basis,
   and monitor both partition and boundary derivatives under refinement.
2. Investigate a convergent residual correction based on a low-width
   precision backbone, rather than solely fitting covariance or using
   the already failed binary-indicator control variates. One option:
   exact Gaussian density ratio on the winner boundary, conditional
   quadrature under the backbone, with bounds on second moments or
   a tempering path. Need real experiments and comparison, not a
   formal exponential algorithm or trivial importance sampling claim.
3. Add independent high-accuracy width-3 references, more graph/order
   families, analytic leaf performance, adaptive resolution, and
   compiled GHK comparison. The arbitrary covariance goal remains open.


## 2026-11-26 — Shared affine lower envelope: a dense route, with negative results

**Status: exact conditional estimator for any Gaussian square root; not yet
a convincing general GHK replacement.** Research files envelope.py,
test_envelope.py, bench_envelope.py, on existing draft PR #1093. Rechecked
main README, CLAUDE.md, races.py, general_sigma/NOTES.md, previous
bounded-width code and PR. No production dispatch changes. Targeted PR
search for preintegration/envelope found no overlapping implementation
in titles; not an exhaustive code search. This direction deliberately
does not repeat the failed scalar diagonal split or indicator control
variate in general_sigma/NOTES.md.

### Derivation

Let X=mu+A Z, Z standard normal, A any exact square root of PSD Sigma,
including rectangular/singular. Remove one common loading row and mean
from every contestant (choice-invariant). Rotate the latent coordinates
orthogonally, with the leading right singular direction of the centered
contrast loadings first. Write X_i=a_i(Y)+b_i T, T~N(0,1) independent of
Y. For fixed Y, every contestant is an affine line of T. The minimum
envelope is concave, so the active slopes appear in decreasing order
and each active line owns a single interval (l_i,r_i). Its conditional
winning credit is Phi(r_i)-Phi(l_i); absent lines have zero credit.
Coincident lines split the credit equally, and crossings of distinct
slopes have zero conditional probability. Parallel lines are reduced
to minimum intercepts, retaining multiplicities for ties.

Sort slopes ONCE. Scan the lines, popping the previous line whenever
the new intersection lies before its entry breakpoint. Each line is
pushed and popped at most once. Hence all n conditional winning credits
cost O(n) per outer draw after O(n log n) sorting; no per-alternative
orthant or lattice. The shared field here is the lower envelope, not
the existing survival product. Gaussian interval masses use survival
differences in the positive tail; cancellation for extremely short
intervals and overflow in extreme contrasts remain possible.
The rank-one case is exact with no outer samples. For rank r>1 the
integrand is a probability vector bounded by one. IID outer sampling
would give strong consistency by the law of large numbers and
Rao-Blackwell variance no larger than the plain indicator for the same
rotation. The implemented outer sampler is scrambled Sobol: in the
ideal unbounded-bit fixed-dimensional limit, the integrand is Riemann
integrable on the unit cube (discontinuities confined to cube boundary
and finitely many equal-parallel-line hyperplanes, unless those lines
are permanently identical, in which case tie splitting is constant).
Thus low-discrepancy quadrature converges; practical finite-bit Sobol
has a finite resolution, not an unlimited asymptotic guarantee. No
covariance approximation or quadrature of T. Refinement is outer sample
count, not rank. Dense factorization/rotation costs O(n^3) for square
input and residual draw multiplication O(n r) per sample; O(n) is
only the shared-envelope scan. Memory batched. Current explicit-A API
avoids a ridge or rank cutoff and supports singular laws, including
identical rows and deterministic races. Equal splitting is the declared
tie convention, consistent with classic fractional dead-heat credit;
the existing continuous factor dispatcher does not promise such atoms.
A covariance-to-root constructor preserving exact clone identities and
rejecting indefinite matrices is still needed. Float-rounding can make
nearly equal lines numerically indistinguishable; not certified.

### Prior art (important)

This is NOT a new convex-hull algorithm. Frazier, Powell and Dayanik
(2009), The Knowledge-Gradient Policy for Correlated Normal Beliefs,
DOI 10.1287/ijoc.1080.0314, section 3.1 Algorithm 1
(https://people.orie.cornell.edu/pfrazier/pub/CorrelatedKG.pdf),
already gives the linear-time upper-envelope scan for Gaussian expected
maxima. Liu and Owen (2023), Preintegration via Active Subspace,
DOI 10.1137/22M1479129, explicitly study Gaussian rotation followed by
preintegration and RQMC. The current experiment combines these known
ideas to price the whole Gaussian win book; do not claim novelty for
either ingredient or claim a literature-first combination without
further search. The interesting goal is practical all-share pricing.

### Reproducible measurements

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_envelope.py
passed 6 standalone tests in 1.88s: brute per-line pairwise intervals
over 400 rows with repeated slopes/intercepts; exact rank-one and binary
answers; common shift/scale/permutation in rank one; deterministic and
singular duplicate rows. Repo-wide CI not run; local standalone copies
of the research files and existing ordered-GHK comparator were used.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_envelope.py
(seed 53 per model). Numba optional, installed locally, warmed before
timings; Python fallback is slower and unbenchmarked. Direct baseline
uses same PCA rotation, not naive unrotated normals. GHK is existing
Python log-domain greedy reordered Sobol; its times include per-runner
preprocessing. Envelope/direct times exclude common Cholesky+SVD prep
(.0006-.0064s for reported cases), include sampling, batched matrix
products and scan. No compiled GHK baseline. Reference is mean of 4
independent GHK scrambles at 2^14 per alternative (16/48 wishart and
48 factor), 3 at 2^14 (16 near), or 3 at 2^12 (96 wishart). Errors
are differences from those stochastic references, NOT true errors;
max spreads respectively 3.11e-5, 1.64e-4, 2.42e-4, 1.04e-5,
2.66e-4. Test scrambles seeds 0..3 or 0..2, refs seeds 100+.
Wishart covariance .3I+.7 normalized Gram, factor V V'+.3I (rank3 V),
near-singular vv'+.01I (condition number 765 for n16). Mixed signs.

| n/model | method (log2 draws) | seconds | median max difference | RMSE |
|---|---|---:|---:|---:|
|16 wishart|GHK 10|.0256|1.24e-4|6.51e-5|
|16 wishart|envelope 15|.0252|6.90e-4|3.20e-4|
|16 wishart|direct 15|.0185|1.41e-3|8.03e-4|
|16 wishart|envelope 18|.1899|1.90e-4|9.35e-5|
|48 wishart|GHK 10|.2401|3.11e-4|1.14e-4|
|48 wishart|envelope 18|.5707|4.21e-4|1.37e-4|
|48 wishart|direct 18|.3713|4.81e-4|1.75e-4|
|48 factor|GHK 10|.2447|3.63e-4|8.04e-5|
|48 factor|envelope 15|.0663|4.07e-4|1.28e-4|
|48 factor|direct 15|.0561|1.73e-3|4.83e-4|
|16 near|GHK 10|.0293|9.21e-5|4.22e-5|
|16 near|envelope 15|.0224|2.27e-4|7.09e-5|
|16 near|direct 15|.0205|1.26e-3|6.04e-4|
|16 near|envelope 18|.1701|3.94e-5|1.97e-5|
|96 wishart|GHK 10|1.0034|4.62e-4|1.28e-4|
|96 wishart|envelope 18|1.2030|4.72e-4|1.31e-4|
|96 wishart|direct 18|.7755|5.48e-4|1.38e-4|

Both local benchmark invocations hit execution time limits before all
rows (48 factor at 2^18 and 96 factor comparison not completed).
The committed script contains all case commands. Envelope and direct
sum to one by construction (envelope deviations <=4.5e-16 observed
in first run); GHK raw mass deviations up to .00584. This is not an
accuracy certificate. Strong low-rank directional signal gives useful
variance reduction against direct simulation. On dense high-rank
Wishart, it does NOT currently beat ordered GHK at matched accuracy or
direct Sobol reliably at matched time; n96 is close to GHK but reference
noise is material. On a true factor model existing survival pricing is
likely better still; no comparison against that path yet. No claim of
practical general breakthrough or useful rare-tail reliability.

### Next iteration

The precision-tree/cycle/width extension remains the strongest new
structured result. For universal dense pricing, try integrating two
Gaussian coordinates via a planar lower-envelope power diagram
(Gaussian polygon probabilities), or use a pilot to select a direction
that maximizes changes in the likely WINNER, not total contrast
variance. The latter can be learned from near-tied pairs of pilot
draws: maximize a weighted sum of squared projected contrasts with
weights depending on closeness and relevance to the minimum. Train on
independent pilot draws, freeze direction, then integrate exact
intervals as above. Compare to direct QMC with same rotation and to
ordered GHK, including pilot cost. A second exact improvement may be
to integrate out analytically all low-width leaf buckets in the earlier
precision code; not performed this run. Do not promote the current
envelope estimator without stronger matched-time tests on large,
high-rank dense cases and independent reference precision.
