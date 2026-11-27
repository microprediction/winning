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


## 2026-11-26 — Collapse Gaussian leaf quadrature axes exactly

**Status: opt-in improvement to existing draft PR #1093, not a new
universal solver.** Rechecked main README, CLAUDE.md, factor/core.py,
races.py, general_sigma/NOTES.md, existing envelope and bounded-width
research. PR #1093 open on this branch; targeted search for leaf/precision
PRs did not reveal an overlapping implementation. Continued the prior
bounded-width route because PCA-envelope results were unconvincing on
dense Wishart. No production changes, no default change even in the
research width solver.

At an elimination bucket i with no children, the message is exactly
the truncated conditional normal mass:

    a = (clip(c_i,-B,B) - b_i z_S)/tau_i
    b = (B - b_i z_S)/tau_i
    m_i(z_S) = Phi(b)-Phi(a).

Its winner boundary insertion is phi(a)/tau_i, masked when c_i is
outside [-B,B]. Store these TWO separator tables, rather than the full
conditional density tensor with the extra i-axis. Reverse context A_i
already excludes m_i, so its contraction with the boundary table gives
the same continuous boundary integral for i. No division by a possibly
zero survival message. Other contestants use the exact CDF message in
their forward and reverse contractions. Disconnected singleton roots
also collapse, and the independent race therefore needs only the outer
threshold quadrature, as in the original shared-survival-product engine.

For a leaf with s separator variables the local forward cost and
storage drop from O(M^(s+1)) to O(M^s) per threshold (fixed s).
The parent reverse contractions still have their original scopes;
this does NOT imply an M-fold end-to-end gain or remove the general
treewidth barrier. The remaining quadrature still converges under
the previous compact-domain argument (exact continuous leaf messages).
No covariance approximation; SPD only, singular refused, no ridge.
Stable tail reflection for interval CDF; extremely tiny interval
differences can still round to zero. Near-singular narrow boundary
densities may be missed on separator axes even though the leaf axis
has been integrated exactly. The numerical gain is not a novelty claim
about Gaussian conditioning: e.g. Genz (1992),
DOI 10.1080/10618600.1992.10477010, and Genz & Trinh (2016),
DOI 10.1007/978-3-319-33507-0_13, study conditional normal probability
integration. Here it is the application to shared all-winner cavities.

### Experiment

New option width_shares(..., collapse=True); default False.
Bench and test files bench_analytic_width.py / test_analytic_width.py.
Run from this folder with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1:
    python -m pytest -q test_analytic_width.py
    python -u bench_analytic_width.py

Local standalone equivalent implementation (copied prior prepare and
_align, then full contraction with same option) passed all six tests
in 1.14 s: four independent 3-way orthant references (adaptive
one-dimensional integral), mixed disconnected graph versus full tensor,
mass/nonnegativity, permutation, common location and scale, independent
case with M=2 and one contestant with M=1. No repository-wide CI; cannot
clone in local environment, connector used for source inspection and
branch writes. The local benchmark used the same copied ordered GHK
routine from bench_markov_tree.py. Timings from separate single-thread
invocations, not compiled GHK; include sparse prep and GHK reordering,
exclude construction of Q and dense inverse for GHK.

Seed 79: k core contestants with covariance G G'/k+.5I; all others
conditionally independent with Gaussian regression on the core,
mixed-sign loadings A=strength*N(0,1)/sqrt(k), D=U(.5,1.5).
This is a deliberately favorable leaf-rich graph (width k; n-k
analytic leaves), not a generic bounded-width graph or a new covariance
grammar claim. Means N(0,.35^2). GHK: four scrambles, 1024 per runner.
Differences against refined same-method quadrature, not independently
certified errors at large n. Reference changes (64,28)->(88,36) for
100/2 weak: 2.05e-6; (64,24)->(80,32) for 64/3 weak: 1.82e-6;
(80,32)->(112,44) for 32/3 strong: 7.96e-7;
(96,56)->(128,80) for 32/2 strong: 3.77e-8.
Here weak/strong mean loading strength .2/1 (not the sharp-noise case).

| n,width,strength | L,M | analytic sec | max diff | full tensor sec | max diff | GHK sec | median max diff |
|---|---|---:|---:|---:|---:|---:|---:|
|100,2,.2|48,12|.180|6.63e-5|.195|3.30e-4|1.066|2.73e-4|
|100,2,.2|64,20|.356|2.05e-6|.632|2.06e-6|1.066|2.73e-4|
|64,3,.2|48,12|.485|5.03e-5|.655|1.36e-4|.391|4.23e-4|
|64,3,.2|64,20|2.562|1.82e-6|4.186|1.82e-6|.391|4.23e-4|
|32,3,1|48,16|.475|9.75e-4|.689|6.65e-3|.094|1.19e-4|
|32,3,1|64,24|1.792|1.70e-5|4.125|1.42e-4|.094|1.19e-4|
|32,2,1|48,16|.073|4.38e-3|.099|1.32e-2|.095|2.30e-4|
|32,2,1|64,24|.129|3.75e-4|.231|8.57e-4|.095|2.30e-4|

100/2 weak reference raw mass 1.000000120; 64/3 weak 1.000000520.
At 100/2, 64,20 the error is mainly outer quadrature (the result
matches 64,28 closely); 2.05e-6 is not a certified error. 16/3 strong
(64,24) analytic .784s vs full 1.956s, both about 8.25e-7 from
(96,56) reference, GHK .0227s / 2.06e-4; no second high-order
reference. Strong loading 100/3 high resolution attempt timed out,
no reported comparison. GHK is much faster in the small strong cases
and no comparison against the existing factor engine on these
leaf-rich examples was performed. They can be represented by
conditioning on a small core, so don't claim they are a new class
or the best approach. The meaningful result is the exact local
axis removal in arbitrary existing bounded-width graphs.

**Important negative result:** sharp residual variance D *= .015
for n32 width2 strength1, condition number 100965. Analytic mass
1.062106 at (64,24), full tensor mass .126055; analytic high orders
(96,64) were not used in the completed run; (96,56)->(128,80)
changed 5.66e-3 and masses .994852->.997368. GHK median difference
from the last unresolved reference 9.80e-4. This reference is not
accurate enough to compare errors. Analytic leaves fix the underresolved
leaf kernel but not the sharp separator boundary. Do not normalize
or call this solved.

### Next research direction

Instead of raising all M, consider eliminating *blocks* of terminal
vertices with analytic low-dimensional Gaussian CDF and boundary
derivatives, and choose elimination orders by the actual number of
remaining tensor axes rather than width alone. More ambitious: for
leaf-rich arrowheads condition on the core contestants directly and
integrate the independent leaves up to the exact minimum of the core.
Each sample of the core gives the earliest core contestant credit
equal to the shared leaf survival at that minimum; all leaf credits
come from the usual shared product integral below the cap. This
avoids sharp boundary densities on core separator grids. Compare
against existing factor code and ordered GHK, including high-rank
limits. It is an exact decomposition for a given separator, not a
claim of efficient arbitrary-matrix pricing. A dense graph has no
large independent set, so the universal objective remains open.


## 2026-11-26 — Probability-coordinate shared field with an exact random cap

**Status: research-only addition to draft PR #1093.** Rechecked main
README, CLAUDE.md, races.py (_setup requires strictly positive D;
dense cov route fits and falls back to GHK for degraded Gaussian fits),
core.py, methods/native.py, general_sigma/NOTES.md, earlier width and
analytic-leaf implementations and benchmark, and the open PR head.
Targeted PR title searches for quantile and conditional minimum had
no matches. No production changes. Do not confuse this with a new
low-rank covariance fit; the factor law is exact and includes zero
conditional variances, which the current continuous lattice refuses.

### Derivation: integrate in the distribution of the minimum, not time

Write X = mu + A Z + diag(s) eps, with s>=0. Given Z, the contestants
with s>0 are independent Gaussians, while those with s=0 have known
performances. Let c be the minimum of the deterministic contestants
(infinity if none). For the independent leaves let

    G(x) = product_i S_i(x), H(x) = -log G(x),
    h_i(x) = f_i(x)/S_i(x), F(x)=1-G(x).

The original winning shared field gives leaf i mass
integral_{-infty}^c G(x) h_i(x) dx. Substituting u=F(x), for which
du=G(x) sum_i h_i(x) dx, gives exactly

    p_i | Z = F(c) integral_0^1
                 h_i(F^{-1}(F(c)*t))/sum_j h_j(F^{-1}(F(c)*t)) dt.

The earliest core contestant(s) get G(c), shared equally on exact
ties. All leaf credits are fractions of a *single* hazard field,
and all conditional credits sum to one for EVERY t, not by
post-normalizing a quadrature result. This is the existing survival
product expressed in probability coordinates. There is no tail cutoff
in the mathematical integral or a narrow density at the core's
winning boundary. The min CDF is strictly increasing if at least one
leaf has positive s. For n_l leaves, quantile x(u) lies between
min_i [a_i+s_i Phi^{-1}(u/n_l)] and
min_i [a_i+s_i Phi^{-1}(u)], by the union and maximum bounds.
Safeguarded Newton on sum log S_i uses this bracket, with log-domain
hazard fractions; one leaf needs no root solve. Use either one
scrambled-Sobol coordinate t together with Z or positive
Gauss-Legendre integration in t. Conditional credit stays in [0,1].
For L=1 ideal iid sampling would give unbiased strong consistency;
the implemented RQMC is low-discrepancy integration of a bounded
function continuous away from cube boundaries and core tie planes
(permanent clones split exactly). It converges in the ideal
fixed-dimensional infinite-resolution limit. L>=2 fixed quadrature
can have deterministic bias; send L and the outer sample count to
infinity, for example iteratively refining one then the other.
Root numerical tolerance and finite-bit Sobol/clipping impose
practical floors; this is not an arbitrary-precision certificate.
Extremely tiny masses, underflow and cancellation in severe utility
scales remain possible, and a nonconverged root raises. No covariance
approximation. For a supplied rectangular exact root with d=0,
deterministic tie splitting works including singular laws, but
degenerates to direct QMC (no general speed claim).

For SPD precision Q, any independent set I gives an exact conditional
decomposition with diagonal conditional covariance 1/Q_ii and
regression B=-Q_II^{-1}Q_IC. Core precision is the Schur complement
Q_CC+Q_CI B. Use its Cholesky inverse transpose R; A_C=R, A_I=B R.
The helper precision_factor requires exact structural zeros; no
thresholding or ridge. Dense preprocessing O(k^3+n k^2), k=core
size, per sample O(n k+n J L) with J root iterations; finite
root iteration count is empirical, not a uniform accuracy bound.
Dense graph has independent set size one: exact but not an efficient
general solution. Main novelty claim is limited to this application
of shared survival pricing; cause-specific hazards and the cumulative
incidence integral are standard competing-risks identities (e.g.
Beyersmann et al., Understanding competing risks: a simulation point
of view, 2011, https://doi.org/10.1186/1471-2288-11-86). Inverse-CDF
simulation and Rao-Blackwellization are standard as well.

### Tests and results

Research files capped_field.py, test_capped_field.py,
bench_capped_field.py. Local standalone source passed 7 tests in
0.71s (binary closed form, adaptive independent capped quadrature,
single leaf, deterministic ties, permutation/common loading/location/
scale with same nodes, equal leaf hazards, exact precision conversion
and non-independent set rejection). No full repo CI.
Run OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
test_capped_field.py from this folder, and python -u bench_capped_field.py
with the same environment. Local benchmark copied ordered_cholesky/ghk
from bench_markov_tree.py; committed benchmark imports it. Same
arrowhead generator as iteration 5, seed 79, weak strength .2 or
strong 1, sharp residual variances multiplied by .015. Mean of three
ordered log-domain GHK scrambles at 4096 per alternative (seeds
100..102) is the noisy reference, NOT truth. Test seeds 0..2;
times single-thread means, given A,d and C, preprocessing excluded
except per-runner GHK ordering. No compiled GHK or production factor
comparison; production rejects D=0 so it cannot directly price these
exact laws. Table reports median max abs difference and pooled RMSE.

| n,k,mode | method (m,L) | seconds | median max diff | RMSE |
|---|---|---:|---:|---:|
|32,2,weak|GHK 10|.0929|1.79e-4|9.15e-5|
| |direct 16|.0854|8.28e-4|3.20e-4|
| |cap 10,1|.0114|2.63e-4|4.27e-5|
| |cap 12,1|.0437|6.61e-5|1.50e-5|
| |cap 10,4|.0414|7.57e-4|1.63e-4|
|32,2,sharp|GHK 10|.0988|3.69e-4|9.38e-5|
| |direct 16|.0669|6.22e-4|1.92e-4|
| |cap 10,1|.0750|6.22e-4|1.88e-4|
| |cap 12,1|.2875|2.15e-4|6.43e-5|
| |cap 10,4|.2620|7.08e-4|1.48e-4|
|100,2,weak|GHK 10|.9719|3.74e-4|8.80e-5|
| |direct 16|.2644|8.37e-4|2.31e-4|
| |cap 10,1|.0615|1.07e-4|2.41e-5|
| |cap 12,1|.1241|7.37e-5|1.67e-5|
| |cap 10,4|.1068|6.85e-4|1.02e-4|
|48,8,strong|GHK 10|.2024|3.67e-4|1.25e-4|
| |direct 16|.1146|1.23e-3|3.69e-4|
| |cap 10,1|.0180|9.55e-4|3.97e-4|
| |cap 12,1|.0664|5.07e-4|1.61e-4|
| |cap 10,4|.0578|1.26e-3|3.57e-4|

Reference max spreads respectively 1.23e-4, 2.16e-4, 1.82e-4,
2.92e-4; reference raw mass defects 4e-5, 1.33e-4, 5.36e-4,
2.53e-4. All cap/direct mass defects <=2.3e-16. GHK test defects
up to .00615. Mass conservation is NOT an accuracy certificate.
The n100 weak case is favorable and promising but the cap's small
differences are below reference noise; a higher-budget n100 GHK
attempt timed out before completion. No claimed true error ratio.
Four-point deterministic t quadrature was worse than random t at
these budgets. Increasing the outer sample count is preferable.

Sharp condition-number-100965 case refined against three 2^14 GHK
scrambles (spread 1.03e-4, raw reference mass 1.00011949):
GHK10 .0987s / 4.10e-4 median max diff; cap12,1 .2783s /
2.21e-4; cap14,1 1.123s / 6.12e-5; cap10,8 .5217s /
5.17e-4; cap10,16 1.109s / 5.27e-4. Here exact mass and
absence of a sharp core lattice are useful, but no matched-time
win versus GHK demonstrated, and fine reference uncertainty remains.

Dense Wishart negative control: seed87, n48, C=G G'/48+.25I
(where G is unscaled standard Gaussian), cond15.0, use one leaf
with largest conditional variance, exact precision_factor prep
.00085s (dense inverse excluded), k47. Three 2^13 GHK refs,
spread 9.94e-5. GHK10 .2137s / 2.88e-4; cap16,1 .1038s /
1.00e-3; cap18,1 .4087s / 4.80e-4; direct16 .0856s / 1.19e-3;
direct18 .3146s / 5.17e-4. One-leaf conditional split is marginally
better than direct per sample but not competitive with GHK at
matched accuracy. No arbitrary-matrix breakthrough.

### Next iteration

The positive result is a shared probability-coordinate field that
admits exact deterministic conditional contestants, and a useful
randomized quadrature with only ONE extra integration coordinate
regardless of number of independent leaves. The dense negative
control rules out claiming universal advantage just from one-leaf
conditioning. The natural next extension is to a separator whose
conditional components are small correlated groups: each component
has its own minimum-survival function and boundary densities;
the identical hazard-ratio identity works for all groups and all
contestants. Seek efficient bivariate/block minimum CDFs and
derivatives, plus an adaptive choice between group conditional
quantile integration and explicit tensor cavities. Another worthwhile
direction is choosing a Gaussian rotation to integrate a whole
coordinate along the earlier affine envelope, targeted at near ties,
or using a tractable backbone with an exact correction. The root
solver is the bottleneck in sharp cases: avoid 64 iterations across
already converged rows; compare with a capped fixed-x field at equal
time. No need to increase tensor width indiscriminately.


## 2026-11-26 — Sample the minimum, forget the sampled winner

**Status: research-only in existing draft PR #1093.** Read current main
CLAUDE.md, README, factor/core.py, races.py, general_sigma/NOTES.md,
and the preceding capped_field.py and ordered GHK benchmark. PR #1093
is open, on this branch. Searches of open PR titles for hazard,
diagonal transfer and conditional minimum found no matches. No production
dispatch changes.

Earlier general_sigma experiments already attempted a scalar diagonal
split plus lattice Rao-Blackwellization, with poor time/accuracy. Do
not claim the split is new. Iteration 6's quantile method costs many
hazard evaluations in the Newton solver. This iteration removes both
the root solve and the lattice, at the price of more QMC coordinates.

Given the exact law X=a(Z)+diag(s) eps, sample all eps and retain only
the minimum T. Conditional on Z and a continuous minimum T=t, the
winner label distribution is

    w_i(Z,t)=h_i(t)/sum_j h_j(t),

where h_i=f_i/S_i is the individual conditional Gaussian hazard.
This follows by cancelling the common survival field in the joint
density G(t) h_i(t). For deterministic conditional rows, if the
minimum reaches their cap, divide credit equally among the earliest
deterministic rows instead. Conditional expectation of the simulated
winner credit gives this vector, so its expectation is the exact
winning vector. No truncation, inversion, lattice, post-normalization,
or covariance approximation. For iid nodes the law of total variance
gives a positive semidefinite covariance reduction relative to direct
winner indicators on the SAME split, not a guarantee for different
QMC rotations or equal time. Sampling the minimum in high dimension
adds dimensions compared with iteration 6. Each row is a simplex
vector. The formula is the standard competing-risks conditional
cause identity, not a novel identity; see Beyersmann et al. (2011),
doi:10.1186/1471-2288-11-86, and Lindqvist (2021), Conditional Monte
Carlo revisited, doi:10.1111/sjos.12549. Novelty here is limited to
the computational experiment on the shared race field.

For a covariance positive definite in contrast space, set B to an
orthonormal basis of 1-perp, S=B' C B, and d=.999 lambda_min(S).
Then S-dI is PSD and A=B U sqrt(Lambda-dI) reproduces the full
contrast covariance with independent noise d per row. The .999
is NOT a ridge: the removed variance is restored as independent
noise. It avoids an exactly zero eigenmode in this prototype.
Descending spectral order for QMC. Preprocessing O(n^3), per draw
O(nr+n) with r=n-1 here. The hazard calculation is O(n) for all
contestants with log-domain evaluation. Dense sample generation
still costs O(n^2); there is no near-linear general claim.
For supplied exact rectangular roots, including singular ones,
zero d is allowed and degenerates to direct sampling, with exact
equal credit for identical rows/means. The dense helper refuses
nonpositive contrast eigenvalues: no numerical rank threshold or
ridge that silently changes the target. Singular covariance intake
is therefore not a complete API; the caller must provide an exact
root and exact deterministic duplicate rows. Near ties and floating
underflow/overflow can still cause failures at extreme scales;
this is not a rare-tail solver.

For fixed exact input law the bounded conditional credit integrand
is continuous off Gaussian-cube boundaries, tie surfaces and the
cap transitions (permanent identical rows split exactly). Ideal iid
sampling is unbiased and strongly consistent; ideal infinite-bit
scrambled Sobol with increasing m converges for this bounded
almost-everywhere continuous integrand. Finite precision, clipping,
finite Sobol bits and covariance roundoff give practical floors.
No deterministic quadrature bias; exact covariance split up to
roundoff. Permuting rows changes the Sobol coordinate order in
general, so finite QMC estimates need not be identical.

### Tests and measurements

Files hazard_draw.py, test_hazard_draw.py, bench_hazard_draw.py.
Local standalone copy passed seven tests in .86s:
contrast reconstruction/common covariance shock, independent
three-way adaptive quadrature, exact equal independent hazards,
deterministic cap/clones, singular exact root, common shift/loading
and scaling, binary exact normal contrast. No full repository CI.
Run OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
test_hazard_draw.py and python -u bench_hazard_draw.py in this folder.

Seed 731, means N(0,.4^2), dense G iid normal. Wishart GG'/n+.25I,
weak .15 GG'/n+I, rank3 factor V=G[:,:3]/sqrt(3) plus .5I or .01I
(sharp). Four test scrambles 0..3. Reference mean of three ordered
Python GHK scrambles 100..102 at 8192 points/runner for n32, 4096
for n64/128. Errors below are DIFFERENCES from noisy references,
not certified true errors. GHK times include ordering; split
preprocessing .0003-.0022s excluded from hazard/direct times,
included (recomputed) for PCA. Times are means, single-thread,
local copy of current research comparator, no compiled GHK.
PCA is direct Sobol with exact contrast root, d=0; direct is the
same split and same nodes as hazard. Report median max abs diff
across seeds, pooled RMSE.

| n/case | method (log2 draws) | seconds | med max diff | RMSE |
|---|---|---:|---:|---:|
|32 wishart|GHK 10|.08993|2.09e-4|6.48e-5|
||hazard 14|.05041|8.45e-4|3.49e-4|
||hazard 16|.18967|3.81e-4|1.36e-4|
||direct 16|.11305|8.46e-4|4.23e-4|
||PCA 18|.21359|5.10e-4|1.80e-4|
|64 wishart|GHK 10|.38742|2.48e-4|1.09e-4|
||hazard 14|.12716|8.36e-4|2.90e-4|
||hazard 16|.47921|4.11e-4|1.31e-4|
||direct 16|.33544|1.20e-3|3.52e-4|
||PCA 18|.43821|4.59e-4|1.50e-4|
|32 factor|GHK 10|.09618|1.87e-4|5.75e-5|
||hazard 14|.05320|1.80e-4|6.43e-5|
||hazard 16|.18971|6.80e-5|2.63e-5|
||direct 16|.11581|9.29e-4|3.40e-4|
||PCA 18|.21478|5.43e-4|1.92e-4|
|32 sharp|GHK 10|.09925|6.30e-4|1.49e-4|
||hazard 14|.05526|4.41e-4|1.08e-4|
||hazard 16|.22413|1.27e-4|3.85e-5|
||direct 16|.11411|8.79e-4|2.79e-4|
||PCA 18|.21067|2.14e-4|8.58e-5|
|64 weak|GHK 10|.42457|2.53e-4|1.33e-4|
||hazard 14|.11621|1.25e-4|4.35e-5|
||hazard 16|.39259|7.65e-5|3.34e-5|
||direct 16|.25296|1.04e-3|3.26e-4|
||PCA 18|.42907|5.12e-4|1.61e-4|
|128 weak|GHK 10|1.6898|6.23e-4|1.13e-4|
||hazard 14|.25819|2.93e-4|5.11e-5|

Reference max spreads respectively 8.58e-5, 3.41e-4, 7.92e-5,
1.46e-4, 2.38e-4, 1.51e-4. Sharp condition number 1186.
Hazard/direct raw mass errors <=1.2e-16; GHK mass defects up to
.00945 (not necessarily systematic bias). Initial benchmark hit a
too-strict elementwise symmetry check on weak C; fixed to use
matrix-scale tolerance, then reran weak cases. n128 completed the
first two rows before execution timeout. Factor/strong examples
may be better served by the production structured engine; not
benchmarked here. No relative rare-tail assessment or compiled GHK
comparison. The weak dense case is promising but reference noise
is material, and stronger Wishart is a negative result against
GHK. This is a useful lower-cost alternative to root inversion
and lattice evaluation, not an unconditional general winner.

### Next step

Compare against the previous quantile field for the SAME exact
diagonal split, and against a much finer reference, rather than
comparing across different covariance decompositions. Optimize
heterogeneous diagonal noise allocation (e.g. PSD-constrained
diagonal packing in contrast space): a tiny minimum eigenvalue
can make scalar smoothing useless even when nearly every runner
has large independent conditional variance. A particularly
interesting extension is to analytically smooth only promising
contenders while sampling others to set a random cap, selecting
the subset using Z alone, not their unobserved independent noise.
Compare equal-time on strong dense spectra, not merely weak
Wishart. The bounded-width precision extension remains a
separate valid structured result. No claim of solving the
arbitrary-matrix performance objective yet.


## 2026-11-26 — Heterogeneous diagonal packing: localized vs diffuse near-null modes

**Status: research-only on existing draft PR #1093.** Current main README,
CLAUDE.md, factor/races.py, methods/native.py and general_sigma/NOTES.md
rechecked, as well as the preceding hazard_draw.py and ordered GHK
benchmark. PR #1093 open; targeted open PR title search for diagonal
found none. Earlier diagonal transfer and scalar smoothing experiments
are not new. This iteration changes the exact independent-noise
allocation for the sampled-minimum shared field; no production dispatch.

### Exact split, not a fitted covariance

Let B be an orthonormal basis of the contrast subspace, S=B' C B SPD.
In normalized units write S=U diag(lam) U', and define v_i to be column
i of diag(lam)^(-1/2) U' B'. We can assign any d_i>=0 such that

    R = I - sum_i d_i v_i v_i' >= 0.

The residual T=S-B' diag(d) B is then PSD. Its full square root,
mapped back via B, gives A so that X=mu+A Z+diag(sqrt(d)) eps has
exactly the desired contrast law. The sampled-minimum hazard ratio
of the previous iteration is unchanged. In particular, a finite
number of optimization iterations incurs no model or quadrature
bias: the entire residual is retained, not truncated. IID estimates
are unbiased and strongly consistent by conditional expectation;
ideal scrambled Sobol refinement converges for the bounded a.e.
continuous integrand, as in iteration 7. Singular contrast input is
refused; a supplied exact root with d=0 still uses direct sampling
and explicit equal tie credit (see previous iteration). No ridge or
silent clipping. Dense cost O(sweeps*n^3) preprocessing, O(n^2)
sample generation and O(n) shared hazard evaluation. Permutation
changes the finite coordinate-ascent order and QMC rotation, not the
target. Common covariance shocks and units are removed in contrast
space, subject to floating-point limits.

We use weighted proportional fairness with a logdet barrier:
maximize sum_i w_i log(d_i)+beta logdet(R), w_i,beta>0.
For fixed other coordinates, write h=v_i' R^(-1) v_i. The matrix
determinant lemma gives the exact coordinate maximizer

    d_i_new = w_i/(w_i+beta) * (d_i + 1/h).

Sherman-Morrison updates the inverse in O(n^2). Use 12 sweeps,
beta=.25, all w_i=1 (race-specific weights are NOT validated).
Initialize d=.5 lambda_min(S), not zero: the first zero-start
sweep of the Wishart example drove R into numerical singularity
through successive downdates; the positive initialization corrected
this in completed cases. Rebuild R each sweep; reject infeasible
or nonpositive residual rather than repairing covariance. No claim
that 12 sweeps solve the optimization or that this objective
minimizes estimator variance. Diagonal-plus-PSD decomposition and
minimum-trace factor analysis are old (Shapiro 1982,
https://doi.org/10.1007/BF02296274; Saunderson et al. 2012,
https://arxiv.org/abs/1204.1220). The experiment is the contrast-space
allocation for shared-minimum conditional credit.

**Structural limitation.** For any unit contrast vector u with
u'C u=lambda, every feasible diagonal satisfies
sum_i d_i u_i^2 <= lambda. If u=(e_0-e_1)/sqrt(2) is a localized
near-null mode, only d_0+d_1<=2 lambda is forced; all other rows
may retain substantial smoothing. If every |u_i|=1/sqrt(n), then
mean(d)<=lambda. No diagonal optimization can fix the latter
case; a genuinely different conditioning/control-variate approach
is needed. For diffuse null modes with unequal coordinates the
weighted budget still applies. This is a mathematical obstruction,
not a solver-tuning problem.

### Reproducible experiment

Files packed_split.py, test_packed_split.py, bench_packed_split.py.
Standalone local copies passed seven tests in .70s: reconstruction
(random 2,8,32 dimensions and permutations), localized mode bound,
common shock and scale, exact binary reference, singular/indefinite
contrast refusal. No full repo CI. Local copies of hazard_draw and
ordered GHK use the same computational paths as the branch (the
local hazard copy omitted input validation; committed benchmark
imports the full branch code). Run from this folder:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q test_packed_split.py
and OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u bench_packed_split.py.

Seed 812, means N(0,.4^2); C=.2 GG'/n+I; clone case duplicates
row and column 0 into 1, adds .0005I and mu_1=mu_0+.008; hetero
rescales rows/columns by geomspace(.04,1.5); Wishart GG'/n+.05I;
diffuse I-.9995 uu' for a random centered unit u. Four test
scrambles 0..3. Reference is mean of four ordered log-domain
QMC-GHK scrambles 100..103 (8192 points/runner for n32, 4096
for n64/128), NOT exact truth. Single-thread local Python. GHK
includes ordering and factorizations; scalar/packed preprocessing
reported separately, excluded from per-race estimate below.
Direct uses identical packed split and Sobol nodes, keeps sampled
winner instead of hazards. Med max = median maximum absolute
difference from noisy reference, pooled RMSE over all seeds/rows.

| n/case | method (log2 draws) | sec | med max | RMSE |
|---|---|---:|---:|---:|
|32 clone|GHK 10|.09372|1.53e-4|1.09e-4|
||scalar 14|.05589|2.33e-3|8.72e-4|
||packed 14|.05036|9.59e-4|2.35e-4|
||packed 16|.19397|4.02e-4|9.45e-5|
||direct 16|.11335|9.36e-4|4.13e-4|
|64 clone|GHK 10|.36488|2.02e-4|1.20e-4|
||scalar 14|.11681|2.09e-3|7.23e-4|
||packed 14|.10377|5.19e-4|1.14e-4|
||packed 16|.43580|1.73e-4|4.61e-5|
||direct 16|.26841|1.07e-3|3.28e-4|
|32 hetero|GHK 10|.08851|2.19e-5|6.32e-6|
||scalar 14|.05992|8.72e-4|3.46e-4|
||packed 14|.04893|6.72e-4|2.24e-4|
||packed 16|.20178|3.15e-4|8.10e-5|
||direct 16|.11444|2.38e-3|6.28e-4|
|32 wishart|GHK 10|.09246|3.81e-4|1.21e-4|
||scalar 14|.05329|1.53e-3|5.10e-4|
||packed 14|.05427|1.18e-3|5.07e-4|
||packed 16|.20831|6.15e-4|2.30e-4|
||direct 16|.11685|1.13e-3|4.42e-4|
|32 diffuse|GHK 10|.09544|2.54e-4|1.15e-4|
||scalar 14|.05549|2.95e-3|9.43e-4|
||packed 14|.05481|1.85e-3|7.42e-4|
||packed 16|.22609|9.58e-4|3.76e-4|
||direct 16|.11257|1.08e-3|4.13e-4|
|128 clone|GHK 10|1.60018|2.84e-4|9.50e-5|
||scalar 14|.25363|2.01e-3|5.91e-4|
||packed 14|.23080|4.10e-4|7.04e-5|
||packed 16|.89414|1.84e-4|3.67e-5|
||direct 16|.55434|1.01e-3|2.52e-4|

Packed prep (in table case order) .00985,.01702,.00563,.00536,
.00552,.09823 seconds; scalar .0002-.00345s. Median d for clone
cases .8844,.8891,.8854 vs scalar .0004995; for hetero .05378
vs .002123; Wishart .06863 vs .05509; diffuse .0005494 vs
.0004995. All estimated packed/direct mass defects <=2.3e-16;
GHK defects up to .01016. Reference max spreads 8.20e-5,2.62e-4,
9.89e-6,2.29e-4,1.09e-4,4.28e-4 respectively; the 128 clone
reference mass .99890. The apparent advantage vs GHK at 128
is encouraging, but reference spread exceeds both methods'
reported differences. Do not infer a certified true error ratio.
The comparison does not support a general dense win. Large
Wishart run timed out after preparation; no accuracy claim
for that case. No compiled GHK comparison or production factor
comparison. Exact split has practical value on localized near-null
modes, but heterogeneous variance alone does not deliver a
universal competitor. No extra PR.

### Next experiment

Test grouping the near-null mode rather than treating each row as
a separate independent noise source. For a conditional independent
block B, the survival of its minimum is G_B(t), and the winning
boundary numerator for row i in B is
f_i(t) P(X_{B\{i}}>t | X_i=t). Shared conditional credit at
a sampled global minimum is this numerator divided by G_B(t),
normalized across all rows/blocks. For bivariate blocks the
numerator is analytic; the denominator is a bivariate Gaussian
survival probability. Need stable vectorized denominator and
matched-time results. This can allow correlated smoothing in
directions a diagonal split cannot touch. However a diffuse
near-null mode can still force each small block to remove variance
along its restricted direction; do not assume block packing
solves that adversary. Another route is low-width precision
backbone with an exact likelihood-ratio correction, or the
previous affine envelope along a race-specific contrast direction.
Test with higher-budget independent references and a compiled
GHK baseline where available. The arbitrary-covariance goal is
still open.


## 2026-11-26 — Sparse line cavities: overcome diffuse near-null modes

Research-only in existing open PR #1093; current main README, CLAUDE.md,
factor/races.py and methods/native.py rechecked. Previous general_sigma
notes path attempted but unavailable at that path; prior durable log
already records the diagonal transfer experiments. Open PR title
search for conditional found no matches. No production dispatch.
Rather than compute correlated bivariate survival denominators, this
iteration combines the earlier affine-envelope idea with an
alternative-specific Gaussian direction and shared order statistics.

Let P=I-11'/n, A A'=P C P, A full column rank n-1, and let Q be
the inverse of P C P restricted to contrast space (zero on 1).
For entrant i choose a single companion j, and v=P(e_i+a e_j).
Set a=-Q_ij/Q_jj, which minimizes v'Qv over a. Choose j maximizing
Q_ij^2/Q_jj. This choice is fixed before sampling, independent of
the Gaussian draw. Let s=1/sqrt(v'Qv) and w=s A^+ v. Then w is a unit
latent direction, A w=s v. For sample z, conditional on its
orthogonal projection away from w, replacement of the scalar
z'w by an independent N(0,1) is exact. Modulo common translation
it moves entrant i by r and j by a*r, with
r | projection ~ N(-s z'w,s^2) when measured from the sampled x.
The winning interval is the intersection of

    r <= min_{l != i,j} x_l - x_i
    (1-a) r <= x_j - x_i.

If 1-a<0, the latter supplies a lower bound; if zero, it is a
deterministic comparison. Evaluate the interval mass by Gaussian
CDF differences, reflecting positive-tail intervals. The three
smallest sampled performances suffice for all leave-two-out minima.
One shared sample matrix and one shared score matrix compute all
entrant integrands; O(n) additional work per sample after O(n^2)
dense matrix multiplications, O(n^3) preparation. Unlike the
shared-minimum hazard vector, columns condition on DIFFERENT
sigma fields, so their sum need not be 1 at finite budget.
Each entry is bounded and nonnegative, unbiased under iid or
ideal uniformly scrambled nodes and strongly consistent under iid.
Increasing ideal Sobol resolution gives convergence for the bounded
a.e. continuous integrand (finite bits/roundoff impose floors).
Normalizing the final vector would be consistent but not unbiased;
raw mass defects reported, no normalization performed.
No model approximation, ridge or numerical quadrature.
Conditional covariance reduction holds coordinatewise for iid
draws, not necessarily jointly or at matched QMC time.
This is Gaussian preintegration (Liu/Owen 2023,
https://doi.org/10.1137/22M1479129; constrained active subspaces,
Liu 2024, https://doi.org/10.1137/23M1548918); the proposed
contribution is sparse entrant-specific directions and shared
leave-two-out minima, not the general preintegration identity.

**Diffuse near-null escape.** Consider contrast covariance
P-(1-eps)uu' with centered unit u. Its restricted inverse is
Q=P+(1/eps-1)uu'. For any i except a largest-|u_i| entry, take
a=-u_i/u_j with j a largest-|u_j| entry. Then |a|<=1 and
v'u=0, hence v'Qv=||v||^2<=2 independently of eps. Optimizing a
can only lower energy. Thus s>=1/sqrt(2) for at least n-1
entrants, even as eps tends to zero. For the balanced +/-1/sqrt(n)
mode, this holds for every entrant. In contrast, single-entrant
directions have s tending to zero and independent-noise packing
has mean variance <=eps. This is a useful mathematical distinction:
each entrant gets its own conditional integral, so simultaneous PSD
packing is no longer required. It does not establish a GHK speed win
or solve arbitrary many diffuse null modes.

The code requires SPD on contrast space. For truly singular laws
one can work with a supplied exact root: a sparse v must lie in
range(A), and use w proportional to A^+ v; otherwise direct
equal-tie credit is a valid fallback (the existing exact-root
hazard_draw direct mode already does this). This generalization
has NOT been implemented here. Never silently ridge, drop positive
eigenvalues or infer ties from a tolerance. Extreme units, very
small interval mass and eigensystem accuracy remain limitations.

### Reproducible measurements

sparse_cavity.py, test_sparse_cavity.py, bench_sparse_cavity.py.
Run from this folder with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q test_sparse_cavity.py and python -u
bench_sparse_cavity.py (set the same environment). Local standalone
copies passed 6 tests in .76s: exact binary, direct dense line
inequality cross-check, balanced diffuse near-null, independent
three-runner adaptive quadrature, credit gauge/scale/permutation and
singular refusal. The exact dense directions check matches A*w
against sparse contrast v. No full repository CI.

Seed 812, same matrix and means generator as bench_packed_split.
Four test scrambles 0..3. Reference mean of three ordered log-domain
GHK scrambles 100..102, 8192 points/runner for n32 and 4096 for n64.
Same fetched ordered_cholesky and ghk functions in local benchmark.
All differences below are relative to noisy reference, not exact
errors. Single-thread Python, mean seconds over seeds; GHK includes
preprocessing, cavity prep separately .00027-.00185s. Direct uses
same PCA contrast root and nodes as single/pair, keeps winner label.
Medmax is median maximum absolute difference, RMSE pooled across
rows and scrambles. Raw mass defects up to .00878 for cavity, .00681
for GHK; direct sums to one.

| n/case | method log2 draws | seconds | medmax | RMSE |
|---|---|---:|---:|---:|
|32 diffuse|GHK 10|.09401|2.48e-4|1.15e-4|
||single 14|.06342|2.68e-3|7.45e-4|
||pair 14|.06159|5.01e-4|1.91e-4|
||pair 16|.20611|1.68e-4|6.88e-5|
||direct 16|.05436|1.50e-3|5.16e-4|
|64 diffuse|GHK 10|.38108|2.46e-4|1.26e-4|
||single 14|.13109|1.70e-3|5.19e-4|
||pair 14|.11483|5.11e-4|1.23e-4|
||pair 16|.47175|2.40e-4|5.95e-5|
||direct 16|.10668|1.08e-3|3.53e-4|
|32 Wishart|GHK 10|.09479|3.83e-4|1.23e-4|
||single 14|.05913|1.28e-3|4.02e-4|
||pair 14|.05388|9.44e-4|3.66e-4|
||pair 16|.21530|3.22e-4|1.34e-4|
||direct 16|.05068|1.05e-3|3.98e-4|
|32 hetero|GHK 10|.09279|2.19e-5|6.28e-6|
||single 14|.04981|4.23e-4|1.12e-4|
||pair 14|.04869|4.07e-4|1.03e-4|
||pair 16|.19747|2.43e-4|5.39e-5|
||direct 16|.05419|8.59e-4|2.51e-4|
|64 clone|GHK 10|.37177|2.25e-4|1.29e-4|
||single 14|.12072|4.77e-4|1.28e-4|
||pair 14|.11954|3.92e-4|1.14e-4|
||pair 16|.47729|1.19e-4|4.07e-5|
||direct 16|.11201|8.54e-4|3.37e-4|

Reference max spreads 1.09e-4, 2.55e-4, 1.35e-4, 8.08e-6,
2.62e-4 respectively; reference raw mass 1.000246, 1.001608,
1.000129, .99999957, 1.001923. Median smoothing SD single/pair
for diffuse n32 .18560/.98615, n64 .24601/.96505, n128
.41976/.99234. n128 reference attempt timed out; no n128
accuracy or runtime claim. Cavity can improve over direct and
scalar cavity, and is competitive in pooled RMSE in the diffuse
n64 case, but worst-coordinate error is not decisively better
at equal time; heteroscedastic case strongly favors GHK.
No compiled comparison or certified accuracy. No new PR;
preserved as research in existing draft.

### Next experiment

Generalize the sparse direction by choosing a small companion set
J and solving a=-Q_JJ^{-1} Q_Ji. For r diffuse near-null modes,
r suitably selected companion rows can cancel the dangerous modes
if their restriction is nonsingular. Use the k+2 smallest sample
values for deleting k+1 changing contestants; intersect all the
linear inequalities with the shared cap. A pivoted rank-revealing
selection on the near-null eigenvectors may be much better than
greedy pair gain, with explicit conditioning safeguards. Also test
a mass control variate with coefficients fitted on an independent
pilot (the sum of individual conditional credits has expectation
one), and compare full-vector error including rare shares. For
singular laws require explicit exact support and tie semantics.
The arbitrary-matrix performance objective remains open.


## 2026-11-26 — Multi-companion sparse cavities: cancel several small eigenmodes

Current main README, CLAUDE.md, factor/races.py (including _race_dense
and fit_covariance routing), methods/native.py inspected via GitHub;
AGENTS.md not found. Existing PR #1093 confirmed in open-PR search;
no separate open cavity PR found. Earlier diagonal splitting, envelope
and single-companion experiments remain in this log. No changes to
production or main. This iteration is not an alternative covariance
fit: all contrast eigenvalues are retained.

For contrast basis B and exact root A=H sqrt(lambda), set
F=H / sqrt(lambda), where H=B U contains orthonormal contrast
eigenvectors. For contestant i and k distinct companions J excluding
i, minimize ||F_i + a' F_J||^2, equivalently a=-Q_JJ^{-1} Q_Ji
for Q=F F'. Solve the least-squares problem directly rather than
forming the squared-condition-number normal equations. Put s=1/||v||
and w=s v, v=F_i+a'F_J. Then A w=s P(e_i+sum_j a_j e_j), so
preintegration in this unit latent direction changes only i and
J modulo a common shift. Conditional on the orthogonal latent
residual, displacement relative to a sampled performance is
N(-s z'w,s^2). The interval is bounded above by the smallest
performance outside {i} union J minus x_i, and by the inequalities
(1-a_j) r <= x_j-x_i. Negative coefficients of r produce lower
bounds; zero coefficients require the fixed comparison. The k+2
smallest sampled performances suffice for every complement minimum.
One shared draw and score matrix serve all contestants. Current
broadcast cap search is O(n k^2) per draw; a membership/rank
implementation could reduce this. Dense products still cost O(n^2)
per draw; eigensystem O(n^3); per-i pivoted QR and least squares
are extra preprocessing. No rare-tail or compiled speed claim.

**Multiple near-null modes theorem.** For C=P-(1-eps) U U' on
contrast space, U centered with r orthonormal columns, choose k=r.
The rows of U excluding i span R^r: their sum equals -U_i, so
removing i cannot reduce rank. Thus some r companion rows J are
invertible and a solving U_J' a=-U_i exists for every i. The
resulting direction is orthogonal to all r near-null modes, and
its precision energy is ||P(e_i+sum a_j e_j)||^2, independent of eps.
The optimized energy is no larger. In particular, take a globally
maximum-absolute-determinant r-row submatrix of U. For any i
outside these r rows, Cramer's rule bounds every coefficient of
a in absolute value by 1 (else replacing a selected row increases
the determinant). Hence at least n-r entrants admit s >=
1/sqrt(1+r), uniformly as eps decreases. For the other r entrants,
choose a separate full-rank companion set; the bound need not be
as good. This is an existence guarantee, not a bound for arbitrary
pivoted QR. The prototype selects the k smallest-eigenmode rows
by pivoted QR excluding i (or greedily orthogonalizes F rows);
then optimizes energy with all eigenmodes retained. k<r in general
cannot cancel all r dangerous modes. The eigenmode selector is a
heuristic for spectra without a clear gap. Preintegration and
constrained subspaces are prior art (Liu/Owen 2023,
doi:10.1137/22M1479129; Liu 2024, doi:10.1137/23M1548918);
volume subset selection is established (Osinsky 2023,
doi:10.1002/nla.2525). The race-specific sparse direction and
shared complement minima are the experiment, not those techniques.

For SPD contrasts each column is an exact conditional probability,
unbiased under iid/ideal uniform randomized nodes and strongly
consistent under iid. Bounded a.e.-continuous integrands also
converge under ideal equidistributed Sobol refinement. No quadrature
or structural bias in exact arithmetic; floating point factorization,
finite Sobol bits and rare intervals impose floors. Different
conditioning sigma fields imply finite-sample mass defects, not
a simplex vector. No renormalization. For truly singular inputs
the prototype explicitly refuses; an exact-root generalization
must constrain directions to its support, coalesce permanent
identical rows and means with equal credit, and use direct exact-root
credit if no admissible sparse direction exists. Not implemented;
no arbitrary PSD intake or GHK dominance claim.

### Experiments

multi_cavity.py, bench_multi_cavity.py, test_multi_cavity.py.
Local standalone copies (same GHK comparator copied from fetched
bench_markov_tree.py) passed 5 tests in 0.74 s: dense line geometry
vs shared minima, independent three-way adaptive quadrature,
3/6-mode smoothing, shift/scale of sample credits, singular refusal.
No whole-repo CI or compiled GHK comparison. To reproduce from
this directory use OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m pytest -q test_multi_cavity.py and the same environment
with python -u bench_multi_cavity.py.

Seed 812, means N(0,.4^2), centered Gaussian QR modes, eps=1e-5.
Wishart GG'/n+.05 I; hetero .04 to 1.5 row scales and weak dense
base. Four test scrambles 0..3; reference mean of ordered log
GHK scrambles 100..102, 8192 points/entrant at n32 and 4096 at n64.
RMSE pooled over all entries and scrambles; max is median of
per-scramble max absolute difference. These are differences from
noisy references, NOT exact errors. Mean single-thread seconds;
GHK includes preparation, cavity preparation separately 0.0017
to 0.0067 s. GHK comparator uses greedy conditional ordering and
Sobol nodes. No postnormalization.

| n/case | k/mode/log2 draws | median s | seconds | RMSE | medmax |
|---|---|---:|---:|---:|---:|
|32 null3|GHK 10|—|.0871|1.872e-4|6.887e-4|
||1 greedy 14|.0494|.0601|8.405e-4|2.188e-3|
||3 pivot 14|.8185|.0668|2.376e-4|6.665e-4|
||3 greedy 14|.7459|.0709|2.987e-4|8.493e-4|
||6 pivot 14|.8740|.0979|2.235e-4|5.988e-4|
||3 pivot 16|.8185|.2783|1.236e-4|3.283e-4|
|64 null3|GHK 10|—|.3601|1.192e-4|3.264e-4|
||1 greedy 14|.1306|.1179|6.505e-4|1.942e-3|
||3 pivot 14|.8193|.1338|2.888e-4|6.870e-4|
||3 greedy 14|.7853|.1384|3.086e-4|1.039e-3|
||6 pivot 14|.8783|.2076|2.474e-4|6.334e-4|
||3 pivot 16|.8193|.5686|9.425e-5|3.134e-4|
|32 null6|GHK 10|—|.0933|2.211e-4|7.218e-4|
||1 greedy 14|.0137|.0496|8.517e-4|2.454e-3|
||3 pivot 14|.0115|.0648|8.403e-4|2.279e-3|
||3 greedy 14|.0689|.0740|7.821e-4|2.164e-3|
||6 pivot 14|.7447|.1013|2.562e-4|6.335e-4|
||3 pivot 16|.0115|.2551|3.921e-4|1.010e-3|
|32 wishart|GHK 10|—|.0954|1.229e-4|3.831e-4|
||1 greedy 14|.6065|.0505|3.664e-4|9.440e-4|
||3 pivot 14|.5956|.0666|3.716e-4|1.161e-3|
||3 greedy 14|.7148|.0708|3.080e-4|8.437e-4|
||6 pivot 14|.6288|.0983|3.407e-4|1.016e-3|
||3 pivot 16|.5956|.2686|1.265e-4|3.034e-4|
|32 hetero|GHK 10|—|.0888|6.280e-6|2.187e-5|
||1 greedy 14|.2687|.0475|1.026e-4|4.075e-4|
||3 pivot 14|.2690|.0641|1.213e-4|5.207e-4|
||3 greedy 14|.2697|.0640|9.995e-5|3.463e-4|
||6 pivot 14|.2713|.1023|1.195e-4|4.541e-4|
||3 pivot 16|.2690|.2602|5.654e-5|2.818e-4|

Reference max spreads in order 3.516e-4, 1.982e-4, 4.075e-4,
1.346e-4, 8.082e-6. Worst raw cavity mass defect .019998 (64 null3,
3 pivot 14), GHK .006043. The improvement in smoothing is real,
but the measured GHK errors are comparable or better at similar
time; particularly poor on hetero and generic Wishart. No new
standalone PR; preserved in existing research PR #1093.

Next: do not optimize conditional SD alone, since that is not
equivalent to smoothing the actual winner event. Test per-entrant
directions using pilot near-tie contrasts and a fixed sparse
support; consider mass control variate with independent pilot
rather than discarding mass defects. A more distinct approach is
an unbiased correction around an exact shared-field structured
backbone using common random numbers and control variates:
choose the coupling to minimize changes of the winner label,
not Frobenius covariance residual. This might be more fruitful
than adding yet more companions. The arbitrary-matrix performance
objective remains open.


## 2026-11-26 — Paired smooth shared-field residuals

Current main README, CLAUDE.md, factor/races.py (dense fit and
GHK routing), methods/native.py (including contrast construction and
its fallback ridge), and current PRs checked. AGENTS.md not found.
Search also found general_inversion/main.tex's dense grammar plus
residual-factor route and #1064/#1067 fitting work. This experiment
does not replace covariance by that fit: it prices a cheap independent
backbone and estimates a correction to the *original* dense law.
Existing draft PR #1093 is the only working PR; no production change.

### Derivation

For SPD C, take d = .5 lambda_min(C), C0=diag(C), and factor
C-dI=A A', C0-dI=A0 A0'. The independent backbone is deliberately
simple; A0 is diagonal. Couple X=mu+A Z+sqrt(d) E and
X0=mu+A0 Z+sqrt(d) E with independent standard normal Z,E shared
between models. Let R be the symmetric root of C-dI; choose the
right orthogonal rotation A=R O by SVD of R A0, O=U V'.
This minimizes E||A Z-A0 Z||^2, NOT winner-switch probability.
For each race retain its own minimum T,T0, and form conditional
credits H_i = h_i(T)/sum_j h_j(T) with h_i the independent Gaussian
hazard given Z. Compute

    p_hat = p0 + mean(H(X,Z) - H(X0,Z)).

p0 is obtained by deterministic common survival-field quadrature:
integrate exp(sum log S_j - log S_i + log f_i) simultaneously for
all i. The prototype uses quad_vec with breakpoints and two infinite
tails, not the production lattice; no claim that quadrature error
estimates are certified. In exact arithmetic E H=p(C), E H0=p(C0),
by factoring the shared minimum density. Thus the correction is
unbiased with exact p0 under iid draws or ideal uniformly scrambled
nodes. It is NOT a Rao-Blackwellization of the paired hard difference
under a *single* joint conditioning field, so do not claim automatic
variance dominance of that difference. It is the difference of two
separately smoothed credits.

Why smoothing matters: let delta=(A-A0)Z. The minima differ by at
most ||delta||_infty when sharing E. Hence the standardized hazard
arguments differ by at most 2||delta||_infty/sqrt(d).
The derivative of log Gaussian hazard at z is phi(z)/Phi(-z)-z,
positive and bounded by a constant plus |z| (for positive z the
standard normal Mills ratio bounds it; for negative z use
h(z)<=h(0)). Softmax is globally Lipschitz. For fixed n, means,
uniformly bounded covariance and d bounded away from zero, the
squared norm of the credit difference has expectation
O(||A-A0||_F^2): apply the derivative bound along the line segment,
then Cauchy-Schwarz and Gaussian fourth moments to delta and the
standardized minima. This is a local second-order variance bound,
not a dimension-uniform or QMC rate guarantee. The analogous hard
difference has squared norm 2 on a winner flip (when no ties) and
zero otherwise, so its second moment is 2 P(flip). Both the numerical
results and the bound indicate an advantage for weak residuals
with substantial common independent variance. For small d or poor
backbone there is no such guarantee. Common random numbers and
control variates are classical (Glasserman & Yao 1992,
doi:10.1287/mnsc.38.6.884); Gaussian preintegration is established
(Liu & Owen 2023, doi:10.1137/22M1479129). The experiment combines
shared minimum hazards and a deterministically priced shared
field as a smooth control variate; no priority claim.

Preparation O(n^3) for eigensystem and SVD, plus backbone pricing.
Each draw uses one dense multiply O(n^2), then two O(n) survival
credit vectors. Covariance preparation is reusable for changes in
mu; p0 must be recomputed for changed mu. Fixed backbone introduces
no structural approximation error; refine both sample count 2^m
and p0 quadrature separately. With exact p0, iid convergence is
almost sure by bounded credits; ideal equidistributed Sobol
converges since the bounded integrands are continuous except
measure-zero boundaries (including cube edges). Finite bit/FP
floors remain. A supplied exact covariance root, including singular,
can instead use hard_root_correction with equal tie credit for exact
identical rows and means, zero-padding the roots to common latent
dimension. That bounded integrand has discontinuities only on
finitely many nontrivial affine equality hyperplanes (permanent
ties are treated as a single equal-credit group); same convergence.
It is not a competitive dense performance solution for singular
cases. No silent ridge, tolerance ties or eigenvalue deletion.
The SPD prepare routine explicitly refuses singular input; it
does not implement contrast-space root intake. Equal credit agrees
with the classic dead-heat convention; the main factor Gaussian
engine does not promise Dirac ties. Target law invariant under
permutation/common shift/common shock/positive scale; fixed Sobol
coordinates, choice of backbone and smoothing can change finite
sample estimates under permutation or common shock. Shift/scale
tested at fixed nodes. Contrast-space canonicalization remains
to do. Signed correction preserves total mass (up to p0 quadrature)
but can yield negative coordinates at finite m; optional Euclidean
simplex projection ensures nonnegative normalized output, is
consistent and cannot increase L2 error to the true simplex
vector, but introduces bias. Do not silently apply it.

### Experiments

coupled.py, test_coupled.py, bench_coupled.py. Local standalone
copies passed seven tests in .94s (root reconstruction, zero
residual, exact binary, independently integrated bivariate orthants
for 3 entrants, shift/scale, singular refusal, exact-root permanent
tie and simplex projection). No full suite, compiled GHK, or
production factor-lattice comparison. Run from this directory:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q test_coupled.py
and same environment python -u bench_coupled.py [--refined].
Seed 812, means N(0,.4^2), W=GG'/n, C=I+.04W (weak) or
I+.25W (moderate), Wishart W+.05I, diffuse I-.99999 uu' with
random centered unit u, hetero rescales the moderate matrix by
geomspace(.04,1.5). Mixed-sign off-diagonals; all full rank.
Four evaluation scrambles 0..3. First reference mean of
three log-domain ordered GHK scrambles 100..102, 8192 points
per entrant at n32 and 4096 at n64. RMSE pooled across
all contestants/scrambles, medmax median worst coordinate.
Times mean per scramble, GHK includes ordering/factorization;
CV prep includes baseline field and root setup, separately
.032,.059,.065,.030,.028,.028s in table order. No postnormalization
in this table; all differences against stochastic references.

| n/case | method log2 points | sec | RMSE | medmax |
|---|---|---:|---:|---:|
|32 weak|GHK 10|.09069|1.056e-4|1.431e-4|
||hard 14|.03106|1.033e-3|3.112e-3|
||hard CV 14|.03163|3.147e-4|7.602e-4|
||smooth 14|.05280|1.383e-4|4.133e-4|
||smooth CV 14|.07654|1.479e-5|4.208e-5|
||smooth CV 16|.27843|6.649e-6|2.025e-5|
|64 weak|GHK 10|.36916|1.240e-4|1.918e-4|
||hard 14|.05975|6.389e-4|1.732e-3|
||hard CV 14|.06036|2.224e-4|7.105e-4|
||smooth 14|.10900|9.487e-5|2.977e-4|
||smooth CV 14|.15833|2.956e-5|5.944e-5|
||smooth CV 16|.54115|2.812e-5|4.601e-5|
|64 moderate|GHK 10|.38892|1.342e-4|2.257e-4|
||hard CV 14|.08423|4.815e-4|1.297e-3|
||smooth 14|.10201|1.184e-4|3.505e-4|
||smooth CV 14|.14034|6.251e-5|1.939e-4|
||smooth CV 16|.52680|4.266e-5|1.040e-4|
|32 diffuse|GHK 10|.08861|1.170e-4|2.531e-4|
||smooth CV 14|.07791|6.752e-4|2.079e-3|
||smooth CV 16|.28342|3.414e-4|8.071e-4|
|32 hetero|GHK 10|.09039|6.607e-6|2.536e-5|
||smooth CV 14|.09005|2.043e-4|5.252e-4|
||smooth CV 16|.33111|8.585e-5|2.637e-4|
|32 Wishart|GHK 10|.09127|1.229e-4|3.831e-4|
||smooth CV 14|.07915|6.584e-4|1.879e-3|
||smooth CV 16|.30880|2.974e-4|8.442e-4|

Reference max spread respectively 8.865e-5,2.452e-4,2.527e-4,
1.152e-4,8.548e-6,1.346e-4. Reference mass 1.000015,
1.001626,1.002105,1.000258,.9999992,1.000129. CV mass within
5e-16 in these runs. Negative finite-sample CV in hetero:
min -3.908e-5 at m14, -7.077e-6 at m16; rare GHK share
1.94e-90, so CV is not a rare-tail solution. Other cases positive
at the tested budgets. At n32 weak, empirical trace variance
of hard, hard CV, smooth, smooth CV respectively .94746,.05191,
.18163,.0003738 (m14), paired winner flip .02596. At n64 weak,
smooth .15833 versus CV .0003418. At n64 moderate, .20001
versus .0089429. On diffuse near-singularity smooth CV variance
.2520 and Wishart .4621; coupling is not enough.

Because GHK reference uncertainty obscured the favorable cases,
second experiment (--refined) used four independent 2^18 smooth
CV scrambles 1000..1003 as reference (same method, NOT an
independent accuracy certificate), compared raw and normalized
GHK at higher budgets. n32 and n64 weak completed; moderate
timed out after reference only (spread 8.10e-5).
CV reference max spread n32 9.16e-6, n64 1.48e-5; backbone
alone RMSE 1.396e-4 and 4.474e-5, respectively, against that
reference. Thus the correction does meaningful work rather
than simply returning independent shares.

| n | method log2 | sec | RMSE vs refined CV | medmax | pooled between-scramble SD |
|---|---|---:|---:|---:|---:|
|32|GHK normalized 10|.0963|6.423e-5|2.008e-4|7.406e-5|
|32|GHK normalized 12|.2431|1.078e-5|3.919e-5|1.201e-5|
|32|smooth CV 14|.0810|1.508e-5|3.901e-5|1.446e-5|
|32|smooth CV 16|.2908|7.019e-6|1.981e-5|7.151e-6|
|64|GHK normalized 10|.3637|7.860e-5|3.908e-4|9.025e-5|
|64|GHK normalized 12|1.0041|3.203e-5|4.646e-5|3.022e-5|
|64|smooth CV 14|.1688|9.700e-6|3.562e-5|1.010e-5|
|64|smooth CV 16|.6122|4.415e-6|1.602e-5|4.379e-6|

Even including the .059s initial backbone cost, weak n64
is promising against normalized Python GHK; the independent
scramble SD gives evidence not relying on a possibly favorable
reference. n32 is a more modest gain. No universal superiority;
near-null, hetero and Wishart are negative results. Full rank
does not imply a useful independent backbone. No independent
large-scale certificate, no compiled comparison, and no
finite nonnegativity without optional projection.

### Next experiment

Replace diagonal backbone by existing tree/factor/block price
with a *shared positive diagonal split on both laws*. The
same pair of hazard credits then applies exactly conditional
on each coupled factor root. Work in contrast coordinates and
fit the diagonal/backbone by pairwise variograms (not full
covariance Frobenius) to cancel irrelevant common shocks.
Optimize the shared diagonal or coupling against pilot
variance of the smooth residual, not raw winner flips. Explore
the previously implemented heterogeneous split for ill-conditioned
localized modes, and for diffuse modes try shared sparse
preintegration instead of diagonal smoothing. Use an independently
priced low-rank structured model for the backbone, and compare
with compiled GHK before promoting the result. Preserve negative
cases as adversarial tests. The universal performance objective
remains open; this is the strongest weak-residual result to date.


## 2026-11-26 — Spectral factor backbone with an exact smooth correction

Rechecked main README, CLAUDE.md, factor/core.py (fit_covariance,
factor_model_projected, win_probabilities_factor), factor/races.py,
methods/native.py and existing draft PR #1093. Current dense fit
has projected NNLS and global factors, block clustering, residual
eigendirections, diagonal matching, capped-fit polish, and a GHK
route for dense cases. Recent related PRs #1064, #1068, #1050
concern the fit itself. This experiment is NOT a new covariance
fit offered as the target model: the backbone is a control variate,
with the entire dense residual retained in the stochastic correction.
No production dispatch change. Existing draft #1093 remains the
only PR. No checkout/installed winning or fastrace in the sandbox;
tested standalone copies of the code and the ordered GHK from the
previous iteration, not the full repository suite.

**Derivation.** Write P=I-11'/n, B orthonormal basis of contrasts,
and S=B' C B with eigenvalues lambda_1>=...>=lambda_(n-1)>0 and
eigenvectors in ambient contrast coordinates U. For fixed
0<d<lambda_min, set a_j=sqrt(lambda_j-d). Fit a rank-r spike plus
isotropic variance backbone C0 = V V' + s^2 I. Define
b=mean(a_(r+1),...,a_(n-1)), s^2=d+b^2 and
V=U[:,1:r] diag(sqrt(lambda_j-s^2)). Then
A=U diag(a), A0=U diag(a_1,...,a_r,b,...,b).
The coupled laws X=mu+A Z+sqrt(d) E and
X0=mu+A0 Z+sqrt(d) E have exact contrast covariances
P C P and P C0 P. Neither adds a ridge to the target. Their
residual-mean displacement is minimized in squared Frobenius norm
within this isotropic-plus-rank-r class for fixed d: for any chosen
backbone residual eigenvalues, the optimal Gaussian root coupling
aligns ordered eigendirections (von Neumann trace inequality /
Gaussian Bures distance); the leading r root eigenvalues are
matched exactly, and least squares for the common trailing root
gives b above. Since b<=a_r, this respects the positive-spike
constraint. Objective is sum_(j>r)(a_j-b)^2, not raw covariance
Frobenius or directly hazard variance. This is a shifted
square-root analogue of PPCA (Tipping & Bishop 1999,
doi:10.1111/1467-9868.00196), which averages trailing covariance
eigenvalues instead; optimal Gaussian coupling and CRN are not
novel (Glasserman & Yao 1992, doi:10.1287/mnsc.38.6.884).
The research contribution is to combine this particular backbone
with the existing shared minimum hazard correction and the existing
deterministic shared survival field. No claim of optimal race
variance or priority for the spectral fit.

Use previous H_i = h_i(T)/sum h_j(T), with each law's own T,
and p_hat=p0+mean(H-H0). Exactly the same conditional hazard
identity and local O(||A-A0||_F^2) variance result apply for fixed
d>0. Expected correction plus exact p0 equals the original
target, no structural approximation error at any rank. The target
law and chosen contrast representation are invariant to common
shocks, common translation, positive scale and permutation (at
a spectral cutoff tie, a nonunique backbone can be selected);
finite scrambled samples need not be equivariant. Signed finite
estimates may be negative; mass equals p0 mass. Optional prior
simplex projection is consistent but biased; not used here.
The SPD-on-contrasts prototype refuses singular contrasts; the
previous supplied-exact-root hard_root_correction is a consistent
fallback with equal permanent-tie credit, not a competitive
singular method. No ridge. Iid strong consistency and ideal
Sobol equidistribution as in previous iteration, refining sample
count independently of backbone quadrature. Prototype field is
the same conditional product integral as factor/core.py with
per-node trapezoidal grids, tensor Gauss-Hermite for the factors,
but not a call to the installed production code. Fixed 10-sigma
tails omit at most (n+1) Phi(-10) conditional total probability
(by union bound on the left, one marginal on the right); to
remove even this tiny fixed truncation in a convergence statement,
increase the window padding alongside GH order and grid resolution.
No quadrature certification; check refinement explicitly. Cost
O(n^3) eigensolve once, O(Q n L) backbone field per mu, O(M n^2)
correction for all contestants, two dense multiplies, no per-runner
orthant computation. In future use main win_probabilities_factor
with explicit F,W or race_probabilities(V=V,D=...) and validate
quadrature independently. Root prep reusable across mu; price
p0 again when mu changes.

**Experiments.** spectral_backbone.py, test_spectral.py,
bench_spectral.py, compare_diagonal.py. Local standalone tests:
4 passed in .95s (root reconstruction, common shock/scale and
permutation of backbone covariance, exact rank-one zero correction,
field shift/scale/permutation, independent three-runner zero-mean
arcsine orthant formula, singular refusal). Run from directory:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q test_spectral.py
and same environment python -u bench_spectral.py
and python -u compare_diagonal.py. The committed benchmark imports
the identical ordered_cholesky/ghk functions from bench_coupled.py;
the local run used a verbatim copy normalized to unit mass.
Seed 9011, mu~N(0,.4^2), W=GG'/n; weak=I+.12 W;
spike adds vv' with centered v~N(0,.65^2); Wishart=W+.05I;
diffuse=I-.99999 uu' centered normalized v. Mixed sign correlations,
all SPD. d=.8 lambda_min on contrasts; rank 0 vs rank 1.
GH order 21, L=513, refinement order 41,L=1025.
Four evaluation scrambles 0..3, independent normalized ordered
GHK reference four scrambles 100..103, 2^14 nodes/runner at n32,
2^13 at n64. GHK includes all ordering/factorization, CV prep
includes field + roots separately; times per scramble, RMSE over
all coordinates and scrambles, SD pooled between-scramble SD,
medmax median max coordinate error. Ref not exact; reference
pooled SD 8.10e-6,3.10e-5,8.03e-6,2.46e-5 in case order;
max spread 6.79e-5,2.33e-4,5.20e-5,1.57e-4. Large n
reference uncertainty is substantial. All estimates in table
positive and mass defect <=4.5e-16. Benchmark completed four
cases before environment timeout, diffuse only separately below.

| n/case | method log2 | prep s | run s | RMSE | medmax | scramble SD | trace integrand var |
|---|---|---:|---:|---:|---:|---:|---:|
|32 spike|GHK 10|0|.08959|1.117e-4|4.035e-4|1.244e-4|–|
||GHK 12|0|.23742|2.420e-5|9.705e-5|2.550e-5|–|
||rank0 CV 13|.00409|.03625|1.485e-4|4.374e-4|1.332e-4|.08033|
||rank0 CV 15|.00409|.13709|8.218e-5|2.593e-4|8.494e-5|.08029|
||rank1 CV 13|.01287|.03154|5.050e-5|1.177e-4|4.853e-5|.003186|
||rank1 CV 15|.01287|.12072|2.403e-5|6.385e-5|2.308e-5|.003204|
|64 spike|GHK 10|0|.37166|1.227e-4|3.107e-4|1.267e-4|–|
||GHK 12|0|.97422|5.036e-5|1.528e-4|5.141e-5|–|
||rank0 CV 13|.00607|.07274|1.419e-4|4.467e-4|1.443e-4|.05352|
||rank0 CV 15|.00607|.24311|6.487e-5|2.050e-4|6.409e-5|.05356|
||rank1 CV 13|.02893|.07622|4.295e-5|1.325e-4|4.174e-5|.003240|
||rank1 CV 15|.02893|.25085|2.297e-5|6.472e-5|1.961e-5|.003227|
|32 weak|GHK 12|0|.22599|1.989e-5|6.661e-5|1.920e-5|–|
||rank0 CV 15|.00144|.11945|2.412e-5|8.661e-5|2.556e-5|.003306|
||rank1 CV 15|.01277|.12159|2.089e-5|7.543e-5|2.215e-5|.002815|
|32 Wishart|GHK 12|0|.23808|6.074e-5|1.929e-4|5.787e-5|–|
||rank0 CV 15|.00112|.12942|4.921e-4|1.200e-3|5.074e-4|.50193|
||rank1 CV 15|.01307|.13035|5.058e-4|1.222e-3|5.299e-4|.45709|

Backbone refinement max diff at rank1 spike n32/n64 respectively
1.061e-6,1.037e-6; rank0 near machine precision. Rank1 backbone
alone RMSE 1.039e-3 and 6.612e-4, so the correction is substantial.
Root residual squared falls 6.9945->.3010 (n32) and
25.4066->.6490 (n64). Rank1 improves substantially over rank0 on
spiked dense cases and reaches similar/lower differences at
roughly half/third the GHK time (including prep), but no universal
claim; Wishart is much worse. Weak no-spike has little benefit
from added rank. In a separate same-fixture comparison at m15
with previous diagonal CV (fraction .5 on full C, 4096 batch)
versus new rank1 (fraction .8 on contrast), excluding p0 price
and all prep, four scrambles:

| n/case | prior diagonal SD | rank1 SD | prior trace variance | rank1 trace variance |
|---|---:|---:|---:|---:|
|32 spike|8.123e-5|2.308e-5|.068745|.003204|
|64 spike|1.087e-4|1.961e-5|.116789|.003227|
|32 diffuse|4.407e-4|4.092e-4|.254058|.283191|

Run s .146/.137, .258/.262, .130/.133 respectively. The
comparison changes both d and QMC coordinate basis, so does not
isolate the effect of backbone rank vs the prior diagonal estimator;
the rank0 vs rank1 comparison above holds d and basis fixed.
Diffuse negative result: trace variance actually increases.
No new PR, existing draft only; no compiled comparison or
independent large-n accuracy certificate.

**Next experiment.** Fit the existing heterogeneous factor grammar
as a control rather than replace it: choose D0,V from main's
projected fit, pack a common positive diagonal d_i feasible in
both contrast covariances (previous heterogeneous coordinate
ascent), then use contrast-space Procrustes residual root
coupling. Compare with spectral arm on heteroscedastic spiked
and localized near-singular matrices. A further promising
direction for poor backbones: deterministic first-order
Gaussian covariance derivative from shared field and an exact
likelihood-ratio remainder; investigate second-order residual
variance and near-singular likelihood-ratio instability before
benchmarking. Need compiled GHK and independent accurate
references before production promotion. General performance
objective remains open.


## 2026-11-26 — All-winner covariance-score likelihood remainder

Current main README, CLAUDE.md, factor/core.py (shared product,
win_probabilities_factor and fit_covariance), factor/races.py (dense
fit routing), methods/native.py (log GHK), paper index and recent PRs
rechecked. AGENTS.md absent. Open #1093 is the existing research
PR; no overlapping open estimator PR found. This continues the
spectral backbone iteration's first-order covariance correction,
not the existing dense fit or residual-factor approximation.
No production changes. Standalone local prototype, not full CI.

**Derivation.** For SPD on contrasts choose P=I-11'/n and
C*=PCP + tr(PCP)/(n-1) 11'/n. This adds noise only to the
unidentifiable common direction, not to contrasts. Let
D=diag(C*), R=D^(-1/2) C* D^(-1/2), K=R-I. Price the independent
backbone with mean mu and diagonal D. Under it, let T be the
minimum, z_j=(T-mu_j)/sqrt(D_j), and h_i(T) the normalized hazard
share. Conditional on T and winner i, standardized performance
i is z_i, and all others are independent standard normals
truncated below z_j. Generate one shared vector y of such
truncated normals and replace coordinate i by z_i to form y^(i).
This gives the correct marginal conditional law for every i
even though the n conditional samples are coupled.

For the density ratio L=det(R)^(-1/2) exp(.5 y'(I-R^-1)y),
the covariance path I+alpha K has score at alpha=0
S=.5(y'Ky-tr K). Then

    p_i = p0_i + d_i + E[h_i(T) (L(y^(i))-1-S(y^(i)))],

where d_i=E_0[1_{winner=i} S] is a deterministic directional
covariance derivative. Compute it in the same survival field:
with m_j=phi(z_j)/Phi(-z_j), v_j=1+z_j m_j-m_j^2,
delta_i=z_i-m_i,

    E[S|T,i] = .5*(m'Km + diag(K)'v - tr K
                 + 2 delta_i (Km)_i + K_ii(delta_i^2-v_i)).
    d_i = integral f_i(t) prod_(j!=i) S_j(t) E[S|t,i] dt.

All candidates share the same dense matvec and quadratic
form at each t. The stochastic L and S evaluations likewise
use two matvecs on y then O(n) single-coordinate updates,
rather than n different quadratic forms. Prep O(n^3), deterministic
field O(n^2 L), O(n^2) per all-winner correction sample. Prep is
amortizable across mu except field price and derivative.
No structural approximation error; derivative is exactly
subtracted from and restored to the likelihood correction,
not a truncation of the target model. Integrate field using
quad_vec over infinite tails and finite breakpoints; not certified.
Refine its tolerance independently of sample count.

For fixed n and small K, ||L-1-S||_(L2(P0))=O(||K||_F^2)
by Taylor expansion in a neighborhood of I (Gaussian polynomial
moments times a uniformly integrable exponential envelope).
For each component h_i^2<=h_i, so summing its second moment
over i is bounded by E_0[(L-1-S)^2]; hence correction trace
variance O(||K||_F^4), vs O(||K||_F^2) for the first-order
unsubtracted correction. This is a local fixed-dimension bound,
not a universal QMC convergence rate. Exact criterion for
E_0[L^2]<infty is lambda_max(R)<2, by integrating Gaussian
quadratic exponent 2R^-1-I. Outside this regime the raw
likelihood correction can have infinite variance; subtraction
of a quadratic score cannot repair it. Even within it the
determinant factors may make variance large, especially
near singularity. Iid strong consistency holds on SPD
contrasts by integrability of L and score even without finite
variance. Ideal uniformly scrambled nodes give unbiased sample
averages (exact field); no unqualified deterministic QMC
convergence assertion for unbounded integrands. Finite bit
uniform and floating-point limits remain. The earlier exact-root
hard correction supplies the singular/permanent-tie fallback
with equal dead-heat credit; no silent ridge. This prototype
does not reliably detect nearly singular inputs at machine
precision or implement that fallback itself. Common-shock,
shift and scale invariance tested; finite sample permutation
equivariance not claimed, deterministic field equivariance tested.
Unlike paired hazards, raw LR corrections do not preserve mass
or nonnegativity. Can project to simplex consistently with
finite-sample bias; no projection in the results below.

Covariance differentiation itself is classical (Plackett 1954,
doi:10.1093/biomet/41.3-4.351; Gassmann 2003,
doi:10.1198/1061860032283). Likelihood score/control variates
are established, e.g. Capriotti 2008 arXiv:0808.2332.
The investigated combination is a single shared truncated draw
for all winner-conditioned likelihoods, rank-one quadratic
updates, and deterministic covariance score from the existing
conditional survival product. No priority claim.

**Experiments.** score_field.py, test_score_field.py,
bench_score.py. Local equivalent standalone files, tests
3 passed in .78s: independent adaptive bivariate orthant
reference and central covariance derivative, zero residual,
common shift/shock/scale and field permutation invariance.
Committed code adds a basic input check, benchmark imports
the previously verified ordered GHK instead of the identical
local copy. No production lattice or compiled GHK tested.
Run from this directory:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q test_score_field.py
and same environment python -u bench_score.py.
Seed 812, mu~N(0,.4^2), W=GG'/n; weak I+.04 W, moderate
I+.25 W, Wishart W+.05I, diffuse I-.99999 uu' centered
normalized u, hetero rescaled weak by geomspace(.1,1.5,n).
Four test scrambles 0..3, reference mean of four independent
normalized ordered QMC-GHK scrambles 100..103 (2^13 points
per runner at n32, 2^12 at n64). Reference uncertainty material.
Mean per-scramble seconds; GHK includes prep, LR prep separately
.0238,.0161,.0149,.0151,.016,.0177 seconds in case order.
Reported RMSE across runners and scrambles, pooled between-
scramble SD, trace integrand variance. LR0 and LR1 computed
together in same call, so reported runtime includes both.
Numbers are differences against noisy references, not exact errors.

| n/case | method log2 | sec | RMSE | scramble SD | trace var |
|---|---|---:|---:|---:|---:|
|32 weak|GHK 12|.2575|1.051e-5|1.196e-5|–|
||LR0 12|.0220|2.715e-5|2.944e-5|.0011763|
||LR1 12|.0220|9.732e-6|9.667e-6|.00001513|
||LR0 14|.0591|1.192e-5|1.222e-5|.0011887|
||LR1 14|.0591|4.500e-6|4.146e-6|.00001606|
|64 weak|GHK 12|1.0388|2.649e-5|3.035e-5|–|
||LR0 14|.1173|1.928e-5|6.412e-6|.0013829|
||LR1 14|.1173|1.892e-5|4.259e-6|.00002936|
|32 moderate|GHK 12|.2273|1.195e-5|1.315e-5|–|
||LR0 14|.0575|1.421e-4|1.380e-4|.048831|
||LR1 14|.0575|1.378e-4|1.286e-4|.018158|
|32 Wishart|GHK 12|.2402|6.110e-5|5.535e-5|–|
||LR1 14|.0653|.03731|.03688|660.29|
|32 diffuse|GHK 12|.2343|2.272e-5|1.896e-5|–|
||LR1 14|.0552|.004150|.003801|11.8298|
|32 hetero|GHK 12|.2356|1.562e-6|1.660e-6|–|
||LR1 14|.0592|.01157|.01290|86.915|

Reference pooled SD respectively 7.738e-6,1.994e-5,9.355e-6,
3.150e-5,1.358e-5,5.584e-7. Eigenvalue ranges of R:
[.9570,1.0994], [.9587,1.0943], [.7841,1.5201],
[.0555,3.6486], [1.123e-5,1.2517], [.2867,7.5341].
First-order deterministic price alone RMSE respectively
3.357e-6,2.061e-5,6.580e-5,.001537,.0001885,.0001216;
for weak cases correction may not be distinguishable from
reference noise. LR1 m14 worst mass defect respectively
.0001044,.0003183,.003658,.3976,.04058,.2850; weak positive
at these budgets; hetero negative estimates (min -5.77e-6).
Weak trace variance reduction LR0/LR1 ~74x and 47x;
weak n32 time including prep .0829s vs .2575s GHK, and
n64 .1334s vs 1.0388s GHK. Weak reference noise prevents
precise accuracy claims; between-scramble SD corroborates
variance gain. Raw mass error means the previous paired-hazard
estimator may remain preferable. The moderate case is already
worse and near-null, Wishart, hetero are catastrophic. The
hetero gauge itself induces an unnecessarily strong common
component in standardized coordinates. No universal performance
claim; no independent large-n certificate or compiled comparator.

**Next experiment:** condition on low-rank common factors
in both laws and do the likelihood remainder only on their
residual covariance (choose factor backbone such that residual
R spectrum close to I, with target exact). First-order score
formula survives unchanged at each factor node. For mass,
subtract p0 times the sampled residual mass (zero expectation)
as an exact linear control variate; use held-out pilot for
alternative coefficients, and distinguish from normalization.
Also consider integrating deterministic second-order score via
truncated fourth moments, but only after addressing spectral
importance-weight degeneracy. Avoid repeated small-perturbation
experiments on the same fixture without testing hetero and
Wishart. Existing #1093 research branch only.


## 2026-11-27 — Whitened contrast lift and factor-conditional score remainder

Current main README, CLAUDE.md, factor/core.py, factor/races.py,
methods/native.py and research/general_sigma/NOTES.md rechecked;
AGENTS.md absent. The existing projected factor fit and residual
ladder are prior work, not claimed here. Draft #1093 still open;
main base 898a9f685bec35f88f6eeef10227bd6d302d7f83 at inspection.
Recent shared-field PR search included #1061 (adaptive ranking
quadrature), #1055 (sharp tree warnings); no competing correction
PR identified. No production changes. This continues the previous
score-field iteration, with a better common-direction gauge and
factor conditioning; not merely a dense fit.

**Exact lift.** Let P=I-11'/n and A=P C P. For positive diagonal
preconditioner D, put w=D^(-1/2)1/||D^(-1/2)1||, H=I-ww',
and W=H D^(-1/2) A D^(-1/2) H + ww'. Then
C*=D^(1/2) W D^(1/2) has exactly the original contrast covariance:
P D^(1/2) H=P D^(1/2), H D^(1/2) P=D^(1/2) P, and
P D^(1/2)w=0. This is a choice-equivalent lift, not a ridge.
If A=P diag(d) P and D=diag(d)>0, W=I exactly. The initial d
solves the diagonal equations of P diag(d) P:
d_i=(A_ii-tr(A)/(n(n-1)))/(1-2/n), n>2. A positivity floor
only chooses coordinates, never changes A. This fixes the
artificial standardized correlation caused by the previous
isotropic common-direction lift on heterogeneous independent races.

For rank r, eigen-decompose W on w-perpendicular; remove from the
r largest eigenvalues their positive excess above the median
of the remaining eigenvalues. Map the corresponding loadings V
back through D^(1/2). The residual E=C*-VV' is SPD on the full
space for SPD contrasts. Re-estimate d from diag(P E P) using
the same equation, damp 1/2 and repeat 12 sweeps. This is a
heuristic preconditioner/fit, not a claimed new factor-analysis
algorithm or guaranteed optimum. Every sweep retains the complete
residual E; hence the iteration count causes no structural error.
It is the whitened quotient lift and subsequent correction that
matter. It can in principle use main's fitted diagonal instead.

**Conditional correction.** Set S=diag(sqrt(diag E)), R=S^-1 E S^-1,
K=R-I. Conditional on F~N(0,I_r), the original Gaussian has means
mu+VF and standardized residual law N(0,R). The independent
proposal has the same means and N(0,I) residual. Apply the previous
all-winner score-field formulas conditional on F; integrate p0
and dp over F by Gauss-Hermite, then sample F as the first r
coordinates of the scrambled Sobol correction. Each residual
likelihood is exact. Write H_i for the conditional hazard share,
Y^(i) for the winner-conditioned truncated Gaussian draw, and
b_i=H_i[L(Y^(i))-1-score(Y^(i))]. E sum_i b_i=0 (by conditioning
on F and summing winner events). For fixed vector c with sum c=1,
replace b by b-c sum b; this is a zero-expectation linear
control variate, not ratio normalization. We use c=p0/sum(p0)
and enforce the deterministic field's known unit mass with the
same vector (only numerical quadrature error). This preserves
mass samplewise without changing the exact target; does not
guarantee nonnegative prices. Exact-arithmetic iid SLLN applies
on SPD contrasts, even for unstable likelihoods; finite variance
requires max eig(R)<2 and can be enormous near min eig(R)=0.
For fixed n and R near I the previous fourth-order variance
bound still holds, including after the bounded linear projection.
Ideal uniform scrambles are unbiased for the stochastic term,
but no deterministic QMC rate asserted. Independent refinement:
increase GH order, tighten threshold tolerance, increase sample
count. No structural approximation. Singular or numerically
near-singular contrasts explicitly refused; earlier exact-root
hard correction (equal dead-heat credit, no ridge) is needed for
a general dispatcher. This prototype does not implement it.

Prep O(sweeps*n^3) dense, field O(order^r * L*n^2) for L
threshold evaluations, correction O(n^2+nr) per all-winner
sample. Prep reusable across abilities; field and correction
are not. No claim of universal speed. Factor conditioning
and Gaussian likelihood scores are classical; previous log's
Plackett and Capriotti citations apply. Related general
orthant methods: Ridgway (2016), arXiv:1411.1314, SMC rather
than this shared-field conditional derivative; Fasano and
Denti (2025), doi:10.1093/biomet/asaf060, expectation propagation.
No novelty priority claim.

**Experiments.** factor_score.py; run from this directory:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python factor_score.py --test
and same environment python -u factor_score.py --bench.
Local equivalent functions executed with threadpoolctl=1; committed
file imports identical moments and ordered GHK from existing
research scripts instead of local copies. Standalone assertions
passed: reconstruction on spike, heterospike, Wishart, diffuse
(max tolerance 1e-12), deterministic field common shift/shock/
scale and permutation (atol 1e-10), singular refusal, residual
Frobenius <.3 on heterospike, samplewise mass and positivity on
4-runner mixed-sign nontrivial zero-mean exact orthant reference
(trivariate arcsine formula, max error 2.587e-6, m16 seed41).
Three-runner bivariate adaptive orthant reference also matched
to 2.22e-16, but that fixture admits exact diagonal quotient
so it is not a nontrivial correction test. No full CI or
production factor engine run. The committed entry point adds
basic validation. Finite-sample permutation equivariance not
claimed (factor eigenspace sign and Sobol coordinates).

Seed 812; weak I+.04 GG'/n, spike adds vv' (v~N(0,.65^2)),
hetero scales by geomspace(.1,1.5,n), heterospike both,
Wishart GG'/n+.05I, diffuse I-.99999 uu' with centered
unit u. Mixed signs. Four eval seeds 0..3, refs mean of four
independent normalized greedily ordered log-QMC-GHK scrambles
100..103, m13 at n32, m12 at n64. GHK test m12 (4096 per
runner), includes its ordering. GH21, quad_vec tol1e-10;
GH41 refinement largest differences 1.26e-7 (32 spike),
7.75e-7 (64 spike), 3.88e-9 (heterospike), <1e-16 (hetero),
2.66e-9 (Wishart), 7.14e-12 (diffuse). These are convergence
checks not rigorous certificates. Times per scramble; LR
run computes LR0, LR1 and mass control jointly. Prep includes
factor fit and deterministic field; excludes refinement.
RMSE against stochastic reference, SD pooled between-scramble,
medmax median per-scramble max absolute difference. Masscv
all mass defects <=1.2e-15. Negative estimates in Wishart;
all other listed 12-sweep cases positive, tails down to ~1e-27.
Reference SD material: 1.18e-5,2.84e-5,2.35e-6,5.58e-7,
3.15e-5,1.36e-5 in case order.

| n/case | method m | prep s | run s | RMSE | SD | medmax |
|---|---|---:|---:|---:|---:|---:|
|32 spike|GHK12|included|.253|2.56e-5|2.35e-5|1.18e-4|
||masscv14|.045|.062|1.00e-5|9.33e-6|3.30e-5|
||masscv16|.045|.220|6.18e-6|2.76e-6|3.14e-5|
|64 spike|GHK12|included|1.035|4.61e-5|4.61e-5|1.68e-4|
||masscv14|.068|.131|2.37e-5|3.64e-6|7.78e-5|
||masscv16|.068|.473|2.29e-5|2.31e-6|7.86e-5|
|32 heterospike|GHK12|included|.257|4.07e-6|3.11e-6|1.47e-5|
||masscv14|.056|.068|6.79e-6|6.54e-6|1.71e-5|
||masscv16|.056|.230|5.19e-6|3.68e-6|1.38e-5|
|32 hetero|GHK12|included|.234|1.56e-6|1.66e-6|5.80e-6|
||masscv14|.047|.064|4.49e-6|4.86e-6|1.50e-5|
||masscv16|.047|.227|1.47e-6|1.54e-6|4.25e-6|
|32 Wishart|GHK12|included|.249|6.11e-5|5.53e-5|1.82e-4|
||masscv14|.038|.065|4.55e-2|4.40e-2|6.04e-2|
|32 diffuse|GHK12|included|.257|2.27e-5|1.90e-5|7.07e-5|
||masscv14|.039|.065|4.07e-3|3.25e-3|1.22e-2|

The preconditioner sweeps are crucial: n32 spike residual R
range [.726,1.538] -> [.958,1.097], Frobenius K .736 -> .202;
n64 spike [.835,1.586] -> [.959,1.093]; heterospike
[.713,1.475] -> [.955,1.100]. Rank1 zero-sweep masscv14
trace variance .009762 (spike) and .006637 (heterospike);
12-sweep .00002020 and .00002440 (~483x,272x). Rank0
spike had max eig 5.890 and trace var 36.92. Hetero rank0
whitened lift already has range [.957,1.101] vs previous
isotropic lift [.287,7.534]; finite variance regained but
not a speed win over GHK. 40 sweeps scarcely changed
spike/heterospike eigen ranges vs 12. At m14, 12-sweep
unsubtracted LR0 / raw score LR1 / masscv trace variances
respectively .002675/.00003112/.00002020 (32 spike);
.002324/.00004712/.00002791 (64 spike);
.004123/.00005190/.00002440 (heterospike);
.002456/.00002456/.000009003 (hetero).
Wishart max eig 3.331 (infinite theoretical second moment);
diffuse min eig 1.308e-5 (finite but large variance). Negative
results confirm that simply conditioning on top modes does not
solve diffuse near-null modes. n64 reference noise dominates
the new estimator's between-scramble SD; no exact error claim.
No compiled comparator, no independent large-n certificate.
No new PR; existing draft branch updated.

**Next useful experiment:** stop tuning positive factors for
negative spectral modes. A finite-variance defensive proposal
can be constructed as a mixture of the exact target and
structured backbone, with bounded likelihood weights, but
winner-conditioned independent truncated draws no longer
sample that mixture directly. Explore combining a shared-field
control with target draws using multiple importance sampling,
or exact low-rank *precision* perturbations (negative covariance
modes) via the Fourier/conditional Gaussian representation;
test variance versus the earlier sparse cavities. Also test
second-order deterministic score on the well-conditioned
residual, and compare against compiled GHK. Need honest
relative-error reporting for nonvanishing probabilities and
full singular tie dispatcher before production integration.


## 2026-11-27 — Negative factors by a complex shared survival field

Rechecked current main README, CLAUDE.md, papers/general_inversion/CLAIMS.md,
winning/factor/core.py (fit_covariance and win_probabilities_factor),
factor/races.py (cov fit/degraded dense dispatch and actual shared
log-survival cavity), methods/native.py, and benchmark runner/shape tests.
AGENTS.md absent. Main still uses positive factor columns, block clustering,
residual eigendirection promotion and diagonal matching/flooring; the
degraded Gaussian race can use GHK. This is NOT fitting dense covariance
to the existing grammar. Open PR #1093 is still the isolated research
branch, and the other open PRs do not implement negative factors.
No production dispatch or API changes. This is the first prototype in
this log to use Fourier inversion for a negative covariance mode.

**Exact class and derivation.** Let D=diag(s_i^2), s_i>0, and
a in R^n with a'a<=1. The target covariance is
C=D^(1/2)(I-aa')D^(1/2), or additionally +vv' (positive ordinary
factor). With independent Y_i~N(0,1) and E~N(0,1-a'a), define
U=a'Y+E. U~N(0,1). Conditional on U=0, X=mu+D^(1/2)Y
has exactly covariance C and the stated mean. E can be identically
zero: no ridge or covariance fit. For a threshold t, put
z_j=(t-mu_j)/s_j and
F_j(z,k)=integral_z^infty phi(y) exp(i*k*a_j*y) dy.
Fourier inversion of the density of U at zero, after fixing
winner i at t and requiring all other contestants >t, gives

 p_i = sqrt(2/pi) integral_R dt integral_0^infty dk
       exp(-(1-a'a) k^2/2) Re[
       phi(z_i)/s_i * exp(i*k*a_i*z_i) * product_(j!=i) F_j(z_j,k)].

The joint density insertion follows by independence before conditioning;
divide by phi(0) to condition; conjugate symmetry halves the Fourier
domain. At each (t,k) this is precisely the independent survival product
with a complex-valued survival and boundary insertion. Compute all
cavities with prefix and suffix products, not division by possibly zero
complex factors. F_j is evaluated via scipy.special.wofz in the upper
half plane: for z>=0, F=.5 exp(-z^2/2+i*b*z) w((b+i*z)/sqrt(2)),
b=k*a_j; for z<0 use exp(-b^2/2)-F(-z,-b). Thus no exponentially
large negative-half-plane wofz calls. The Fourier axis has a fixed
natural scale even as a'a->1, unlike a Gaussian likelihood concentrated
on a thin hyperplane. Complexity O(n L K) for L performance and K
frequency nodes; batch storage O(n K batch). No per-winner multivariate
orthant integrals, no eigensolve for supplied parameters; includes node
construction. Positive factors are combined by the SAME outer
Gaussian integration as existing factor races (mixed_field implements
one positive factor, with the common loading removed). Complexity
O(Q n L K), Q outer nodes. Not yet production integration or a dense
parameter fitter.

**Convergence and singularity.** If a'a<1, Fourier inversion and
Fubini hold by Gaussian damping. At a'a=1, if at least three a_j
are nonzero, at least two nonzero a_j remain after any winner insertion.
Integration by parts bounds |F_j|<=min(1,sqrt(2/pi)/|k a_j|)
uniformly in z. Therefore the product is integrable in k at
infinity, and integrating the bound over t uses integral f_i=1.
Conditioning on U=0 at this singular limit is valid by continuous
densities of the remaining weighted truncated sum. Every pair
contrast has strictly positive variance since its two-sparse vector
(s_i e_i-s_j e_j) cannot be parallel to a with >=3 nonzero
coordinates. Hence even singular C in this class has zero probability
of ties and prices the exact original race. A sparse singular case
with <=2 nonzero coefficients is explicitly refused (also within
8 machine eps of the boundary), not silently regularized. For a
normalized input rounding just above unit norm by <=8 eps, renormalize
to unit norm; larger invalid norms rejected. General permanent-tie
handling and arbitrary PSD matrices are not claimed by this
prototype. Main classic dead-heat calculus and the earlier hard
exact-root equal-credit fallback remain separate.

For fixed finite cutoffs, Gauss-Legendre converges for the continuous
integrand in t and k; then send both cutoffs to infinity, with enough
nodes at each cutoff (e.g. a diagonal sequence of accuracies).
No structural or sampling error. Truncation and numerical
quadrature are separate. In particular the finite performance
window [min(mu-pad*s),min(mu+pad*s)] omits at most
(n+1) Phi(-pad) total race mass by union bound and marginal
conditional variance <=s_i^2. For Fourier truncation at K0,
take r_j=max(1,K0 |a_j|/sqrt(2/pi)), let q count r_j>1
excluding i, and use the bound sqrt(2/pi)*K0*prod_(j!=i)(1/r_j)/(q-1)
when q>=2. For noise>0 also use erfc(K0*sqrt(noise/2))/sqrt(noise).
Take the smaller bound, plus the performance tail bound.
tail_bounds implements these conservative real-arithmetic bounds;
not a floating-point certificate or a quadrature error bound.
For fixed rank r negative update, A has n rows and r columns,
A'A<=I, U=A'Y+N(0,I-A'A), and replace ka_j with k'A_j
and Fourier integral with r-dimensional inversion. For strictly
positive observation-noise covariance, Gaussian damping suffices;
at singular noise additional directional integrability conditions
are needed, not simply total count of nonzero rows. Cost exponential
in r for tensor quadrature, linear in n per node. Not implemented
in this iteration. Independent positive factors can be combined
by conditioning whenever the negative residual is itself PSD.

Why this is a genuinely different grammar: e.g. take equal s
and centered a with every coordinate nonzero and a'a=1.
For any exact contrast representation P(diag(d)+VV')P,
the centered null vector a forces sum d_i a_i^2=0, hence
d_i=0 for all i; positive factors alone must then span
n-2 contrast dimensions. Our one negative factor prices the
same race with one Fourier axis, including the singular endpoint.
At small positive noise, the prior diagonal packing obstruction
returns for positive smoothing. This construction bypasses it
rather than making a better approximate factor fit. The Gaussian
conditioning and Fourier inversion themselves are classical;
see Steck (1962), Biometrika 49:433–445,
doi:10.1093/biomet/49.3-4.433 (equicorrelated orthants; abstract
record only), Sun (1988), doi:10.1080/03610928808829845 (orthant
Fourier reduction), Genz (1992), doi:10.1080/10618600.1992.10477010
(separation-of-variables), Cong et al. arXiv:1607.04751 (efficient
simulation for low-rank covariance downdates/precision updates).
The investigated combination is the all-winner Fourier cavity and
low negative rank in the existing field; no priority claim.

**Experiments.** signed_field.py, test_signed.py, bench_signed.py.
Local identical standalone code on SciPy 1.17.0; not full repo
CI, no installed winning or fastrace. Five tests passed in 1.83s:
CF against direct complex adaptive integration (20 points),
four-runner zero-mean trivariate contrast arcsine orthant formula
at residual eigenvalues .3, 1e-5 and exactly zero (max absolute
errors 5.42e-13, 5.52e-13, 5.51e-13; centered a=(1,2,-1,-2)/sqrt(10),
unit s), independent adaptive field with nonzero means and
heterogeneous s, shift/scale/sign/permutation invariance, positive
factor common-shock invariance and mixed-sign-factor orthant
reference (atol 1e-10), non-PSD/singular sparse refusal.
Initial test with four heterogeneous s and nearly singular
observation, cutoff 256 and 160 t nodes was ~1.75e-6 from
exact orthant, not a passing fine-accuracy test; increased
performance refinement needed for oscillations, and do not
infer dense-case accuracy from this small sparse-ish case.
The final mixed orthant test uses nonzero noise .3.

Reproduce from this directory:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s test_signed.py
and same environment python -u bench_signed.py [--large].
Seed 812; mu~N(0,.4^2), centered Gaussian a normalized so a'a=1-1e-5,
unit s; hetero s=geomspace(.12,1.6,n), sparse only 4 nonzero a,
mixed additional v~N(0,.3^2). All target matrices PSD, dense
mixed-sign off-diagonals in diffuse, hetero and mixed cases.
Bench references use same deterministic field at (L,K,K0)=(256,120,96)
and compare with (192,80,48); positive factor orders 41 and 31.
Differences versus this refined field are not independent error
certificates. Independent compiled Genz/SOV baseline from
SciPy 1.17 _qmvnt._qmvn uses permuted Cholesky, CBC lattice
and compiled _qmvn_inner; 8 randomized batches per runner,
4 seeds 0..3, raw covariance contrast per runner, normalized
full vector for fair comparison. Includes all per-runner
preparation and lattice construction (may be internally cached);
no fastrace comparison. All field timings include construction,
no prep amortization required. RMSE over all contestants and
scrambles, max and relative max are medians over scrambles
(relative only on reference p>1e-4); SD pooled between-scramble.
Single-thread runtime per complete vector. Field settings
96/24/12/15 and 128/48/24/21 respectively (last GH order for
mixed only). First run timed out before completing n128
compiled comparator; reran hetero, sparse, mixed separately at
2048. Do not report incomplete 8192 results for those.

| case | method | sec | RMSE vs refined | med max | med relmax | scramble SD |
|---|---|---:|---:|---:|---:|---:|
|32 diffuse|field coarse|.02086|7.45e-12|1.25e-11|4.44e-10|–|
||field fine|.05439|4.09e-15|1.00e-14|1.25e-13|–|
||compiled Genz 2048|.32314|1.09e-4|3.31e-4|.01407|1.16e-4|
||compiled Genz 8192|.78386|3.48e-5|9.22e-5|.00459|3.28e-5|
|64 diffuse|field coarse|.04059|2.10e-10|3.58e-10|4.13e-8|–|
||field fine|.10341|8.20e-15|1.22e-14|1.30e-11|–|
||compiled Genz 2048|2.02806|9.89e-5|3.55e-4|.04451|9.85e-5|
||compiled Genz 8192|4.37633|3.26e-5|1.10e-4|.01222|3.36e-5|
|32 hetero|field coarse|.02159|2.03e-8|1.08e-7|9.24e-5|–|
||field fine|.05130|1.02e-11|4.82e-11|7.87e-9|–|
||compiled Genz 2048|.32386|6.73e-6|1.87e-5|.00416|6.88e-6|
|32 sparse|field coarse|.01240|1.08e-7|4.13e-7|2.57e-5|–|
||field fine|.03084|3.06e-8|1.49e-7|5.08e-6|–|
||compiled Genz 2048|.33157|1.00e-4|3.21e-4|.01153|1.03e-4|
|32 mixed|field coarse|.30752|1.63e-10|5.88e-10|2.73e-8|–|
||field fine|1.18627|6.04e-13|1.98e-12|1.30e-10|–|
||compiled Genz 2048|.31296|9.29e-5|2.83e-4|.01222|9.68e-5|

Refinement max differences: 2.51e-14, 2.17e-14, 5.43e-14,
1.42e-8, 2.42e-14 in case order. Coarse field mass defects
2.13e-10,1.19e-8,7.08e-8,2.11e-10,3.08e-11; fine
1.0e-14,3.25e-13,6.12e-11,1.03e-14,8.88e-15.
All positive in diffuse, hetero (minimum ~1.33e-22), mixed.
Sparse negative coarse min -4.14e-7 and refined min -1.20e-9,
so this is NOT a reliable rare-share evaluator; mass alone
is not a diagnostic. Fourier truncation bound at K0=12 is
9.92e-11 (n32 diffuse/hetero/mixed), 1.04e-12 (n64 diffuse),
but .273 for sparse; at K0=24 the first three are ~5.07e-18,
n64 ~7.34e-18, sparse .0533. Sparse needs separate fallback
or explicit failure/refinement rather than normalization/clipping.
At n128 an earlier standalone field run (96/24/12) took .0792s
and differed by 7.76e-9 max from 256/256/96; (128/48/24)
took .2008s, 1.83e-12 max difference; no compiled comparison
at n128. Singular exact-zero observation noise n32/n64
unit sd refinements also stable: 128/128/32 vs 192/192/64
max diff 2.46e-14 and 4.01e-14; no independent high-dimensional
singular certificate. The positive+negative factor extension
is slower than the pure-negative field but produces much
smaller refinement differences at roughly the same budget
as compiled Genz in the n32 fixture. Independent closed form
and compiled baseline corroborate the new formula; larger-n
accuracy numbers are refinement, not exact error guarantees.

**Next experiment:** move to rank two negative modes with
mixed positive/negative factor grammar, aiming for an adaptive
low-rank Fourier rule rather than tensor explosion. The real
mathematical issue is cancellation when the conditional event
is rare or the negative loadings are localized. Try contour
tilting of each Fourier integral by the conditional truncated
sum saddlepoint (reuse shared tilted field and cavity, possibly
one saddlepoint per t rather than per runner); derive a stable
tail bound and check sparse fixtures. Need a dispatcher with
explicit accuracy diagnostics or a stable exact-root fallback,
and a dense signed-decomposition fit before production API.
General arbitrary-PSD performance remains open, but this is
a substantiated exact broader matrix class under route 1.
Existing research draft #1093 updated; no speculative new PR.


## 2026-11-27 — Two negative factors: polar quadrature and directional tail bounds

Rechecked main README, CLAUDE.md (AGENTS.md absent), factor/core.py
(fit_covariance, factor cavity), factor/races.py (dense fit and GHK
dispatch), methods/native.py, bench/runner.py, papers/general_inversion/
CLAIMS.md and research/general_sigma/NOTES.md. Main base
898a9f685bec35f88f6eeef10227bd6d302d7f83 at PR inspection. #1093
still open/draft on this branch; recent open PRs do not overlap.
The fit/residual promotion ladder is prior work, not this result.
Continue the negative-factor field of the preceding iteration.

**Exact class.** For A of shape n by r, A'A <= I, s_i>0,
X=mu+diag(s)Y conditional on U=A'Y+E=0, where Y~N(0,I_n),
E~N(0,N=I_r-A'A) independently, has covariance
diag(s)(I-AA')diag(s). U is standard r-variate Gaussian.
For t the winning threshold, z_i=(t-mu_i)/s_i, use

  p_i=(2*pi)^(-r/2) int_R dt int_R^r dk
      exp(-k'N k/2) (phi(z_i)/s_i) exp(i z_i k'A_i) prod_(j!=i) F_j(z_j,k'A_j),

where F_j(z,b)=int_z^infty phi(y) exp(i b y)dy. The integral
is real. As in the rank-one implementation the survival products
are shared across all i; no division by near-zero complex survival.
For r=2, write k=rho u(theta), theta in [0,pi], rho>=0.
Conjugate symmetry and Gaussian density normalization give the
prefactor 1/pi; the radial Jacobian is rho. The code uses Gauss
rules per angular sector and radial interval, with cutoff allowed
to differ per sector. O(n L m a q) for L performance nodes,
m sectors, a angular and q radial nodes per sector; n-linear
per node, storage O(n m a q batch). Singular pair validation
has an O(n log n) leverage screening and worst-case O(n^2)
pairwise fallback. No dense fitting or covariance approximation.
Positive factors can mathematically be combined by outer
conditioning, as before, but this mixed rank-two implementation
and timing are not yet done.

**A sufficient general-rank singularity theorem.** Suppose every
r-row submatrix of A has full column rank (full spark), n>2r.
For any unit u, at most r-1 rows have A_j'u=0. The (r-th)
smallest absolute projection has strictly positive minimum over
the compact sphere. After fixing any one winner, at least n-r
remaining projections are uniformly bounded below by a positive
constant. Integration by parts gives |F_j| <= min(1,c/|k'A_j|),
c=sqrt(2/pi), uniformly in z. Consequently the radial tail of
the absolute integrand is bounded by const*rho^(r-1-(n-r)),
which is integrable iff n>2r. This proves absolute Fourier
inversion even when N=0. This is a sufficient condition, not
necessary; non-full-spark cases can be covered by directional
bounds below. No contrast can be deterministic in the stated
full-spark class: a null vector must lie in col(A), and a
nonzero such vector has at least n-r+1>2 nonzero components.
Conditional Gaussian means shift continuously with U; since
there are no ties, the conditional race probabilities are
continuous at U=0. The absolutely integrable Fourier transform
of the restricted race measure has continuous density and hence
prices exactly that conditional law at zero. If n>2r is false
or the rows are not full spark, no general assertion of failure
or convergence is made. In r=2 at least n-2 uniform decaying
factors suffice for n>=5. The test with n=20 regular polygon
rows is not full spark (opposite rows parallel) but is covered
by the sector bound and no-tie check.

**Directional tail bound (implemented).** For sector centered
at u_l, half-width pi/(2m), let delta=2sin(pi/(4m)).
For any u in sector, |A_j'u| >= b_lj =
max(0,|A_j'u_l|-delta||A_j||). Let K_l be its radial cutoff,
r_lj=max(1,K_l b_lj/c) and q_li the number with r_lj>1 excluding i.
Integrating rho times the product bound gives an omitted-frequency
bound per winner and sector of

  (1/m) K_l^2 prod_(j!=i)(1/r_lj)/(q_li-2), q_li>2.

Take infinity otherwise; also bound the noise damping by
ell_l=max(sqrt(lambda_min(N)), ||N^(1/2)u_l|| -
delta sqrt(lambda_max(N)))^2 (nonnegative by the first term).
The alternative bound is (1/m) exp(-ell_l K_l^2/2)/ell_l
when ell_l>0. Take the smaller, then sum across sectors.
The integration in t of phi(z_i)/s_i is <=1, so this is an
absolute per-share frequency truncation bound. Add
(n+1)Phi(-pad) for the omitted performance window, as in rank1.
Numerical roundoff in evaluating the bound is not certified;
small eigenvalues within tolerance are treated as zero for
the bound (not a ridge). Input eigenvalue excess <=16eps can
be repaired to the PSD boundary, otherwise refuse. For actual
singular/near-machine-singular inputs, require finite sector
bounds and refuse possible deterministic pair contrasts; this
is a conservative scope restriction, not a tie-rule change.
For fixed domain increase quadrature orders, then expand
cutoffs/pad and, if necessary, narrow angular sectors.
This converges to the exact race by the bounds and continuity
on finite domains; no structural bias, no sampling noise.
The adaptive cutoff chooser doubles K per sector up to a cap
and requests <=tol/m per sector. This certifies only the
omitted-frequency term in real arithmetic, not angular/radial
or performance discretization, not a floating-point certificate.
The observed adaptive choice often equals a uniform cutoff and
did not improve runtime in these isotropic fixtures.

**Experiments.** signed_rank2.py reuses signed_field.truncated_cf.
Local identical standalone code, NumPy 2.3.5, SciPy 1.17.0, one
thread; not full repo CI or fastrace. Commands from this folder:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s test_rank2.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u bench_rank2.py diffuse 32
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u bench_rank2.py diffuse 64
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u bench_rank2.py hetero 32
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u bench_rank2.py sparse 32

Five tests passed in 4.67s. Independent zero-mean four-contestant
trivariate contrast arcsine formula, seed 52, sd (.8,1.2,.95,1.4),
singular values (.75,.91): initially max error 1.047e-8 at
96/36/24/2/24; increasing angular nodes to 48 sectors x3,
112 performance, 40 radial gives max error 7.202e-12 (actual
refinement, not relaxed test). Independent heterogeneous
four-runner adaptive one-dimensional integral at A=0 also
passes. Exact singular n20 circulant covariance with regular
polygon A (two eigenvalues zero): uniform p_i=1/20 by symmetry,
max error 6.063e-11; omitted-frequency bound 1.904e-9.
Seed 812 n24 heterogeneous near-singular: permutation, shift,
scale and rotation of negative loadings agree at atol 1e-9.
Invalid PSD and sparse singular ties refused, bounds selection
verified. Common positive shock invariance follows from race
contrasts but not separately tested for this wrapper.

Bench seed 812, mu~N(0,.4^2), two centered Gaussian columns
orthonormalized, both singular values sqrt(1-1e-5); diffuse
unit sd, hetero sd=geomspace(.12,1.6,n), sparse only 8 nonzero
rows. Mixed signs and two near-null directions. All field
timings include preparation and node creation; no amortization.
Reference (L,radial,sectors,angular,cutoff)=(144,40,28,2,32)
for diffuse and (160,40,28,2,32) for hetero/sparse, compared
with (128,32,24,2,24). Reference max refinement discrepancies:
8.35e-12 (32 diffuse), 9.52e-12 (64 diffuse), 2.71e-10
(hetero), 4.68e-8 (sparse). These are not independent error
certificates. Compiled SciPy _qmvn SOV with permuted Cholesky,
8 randomized batches per winner, four seeds 0..3, normalized
for comparison, timing includes per-winner preparation.
A long full benchmark timed out during n64 Genz, so no n64
Genz timing/accuracy claimed. Hetero/sparse independently
rerun, only 2048 completed; adaptive cutoff chooser tested
but not integrated on those reruns. RMSE over contestants and
scrambles, median maximum and max relative difference on
p>1e-4, pooled scramble SD. Representative results:

| case | method (L/q/m/a/K) | sec | RMSE vs refined | med max | med relmax | scramble SD |
|---|---|---:|---:|---:|---:|---:|
|32 diffuse|64/16/12/1/10|.11724|6.09e-8|1.77e-7|2.44e-5|–|
||80/20/16/1/12|.25014|1.47e-9|2.91e-9|1.63e-7|–|
||96/24/24/1/16|.61616|1.27e-11|4.10e-11|5.50e-9|–|
||Genz 2048|.33896|1.19e-4|3.27e-4|.01276|1.17e-4|
||Genz 8192|.79307|3.67e-5|9.26e-5|.00355|3.88e-5|
|64 diffuse|64/16/12/1/10|.22460|5.46e-7|8.08e-7|1.35e-4|–|
||80/20/16/1/12|.45862|1.09e-8|3.33e-8|2.33e-6|–|
||96/24/24/1/16|1.07127|2.54e-10|5.10e-10|4.47e-8|–|
|32 hetero|64/16/12/1/10|.11715|6.18e-6|3.49e-5|.03887|–|
||96/24/24/1/16|.44851|4.90e-8|2.74e-7|.000305|–|
||144/32/24/1/16|.92574|9.81e-13|3.30e-12|3.68e-9|–|
||Genz 2048|.29658|1.03e-5|3.40e-5|.00271|1.05e-5|
|32 sparse|64/16/12/1/10|.06617|9.34e-8|2.76e-7|.000150|–|
||96/24/24/1/16|.30014|3.27e-8|1.05e-7|4.70e-5|–|
||144/32/24/1/16|.58728|3.27e-8|1.05e-7|4.70e-5|–|
||Genz 2048|.28911|1.55e-4|5.34e-4|.01427|1.71e-4|

32 diffuse coarse omitted-frequency bound 2.11e-5, fine
9.86e-10; 64 diffuse coarse 1.00e-5, fine 2.59e-13.
Observed fine 64 difference exceeds frequency bound: expected,
the latter does not bound quadrature. Raw field mass defects
at coarse/fine respectively 2.19e-7/1.39e-11 (32 diffuse),
3.17e-5/1.39e-8 (64 diffuse), 3.71e-5/3.20e-7 (hetero),
1.27e-6/1.29e-10 (sparse). Sparse fine mass is excellent but
there is unresolved truncation, illustrating why mass alone
is insufficient. Sparse adaptive cutoff refused at cap 64,
rather than assert tolerance; all reported field probabilities
positive in these four fixtures. Hetero refined minimum
~2.83e-23; no relative claim there. Initial n5 random centered
two-near-null fixture was negative (-.0051) at coarse cutoff
10 despite near-perfect total mass; very slow Fourier decay
and poor directional bounds (choose_cutoffs refused cap256).
Not a production-ready universal method. Exact singular
n32/n64 diffuse field also evaluated at three levels in try2
exploratory script, mass defects similar to near-null, but
no independent asymmetrical singular accuracy certificate.
No full repo or Rust test. Sun (1988), DOI
10.1080/03610928808829845, and Genz (1992), DOI
10.1080/10618600.1992.10477010, publisher abstracts checked;
Fourier reduction and SOV are established. The specific
contribution is low-negative-rank all-winner product reuse
and a directional cutoff bound, not priority for inversion.

**Next experiment:** rather than growing a tensor in negative
rank, exploit an alternative representation for localized
negative modes. If loadings supported on a small group, use
an ordinary independent-block survival field with a stable
low-dimensional orthant solver within that group; its
boundary densities can be reused for every outside winner.
For diffuse modes, use angular/radial anisotropic refinement,
or shared exponential tilt (Lunde, Kleppe, Skaug,
arXiv:1811.05678) to address cancellation in rare shares.
For arbitrary matrices, investigate decomposing signed
spectral residuals into progressively refined exact
structured backbones plus a provably convergent correction;
do not claim tensor quadrature at rank n is practical.
Existing draft #1093 updated, research only, no production
API changes. The rank-two exact class is demonstrated, while
general arbitrary-PSD competitive performance remains open.


## 2026-11-27 — Localized negative modes: real positive triad-block survival

Main rechecked via GitHub: CLAUDE.md (AGENTS absent), README,
factor/structures.py (Blocks explicitly rank-one plus diagonal,
Nested and Tree scalar leaf loadings), factor/core.py (fit_covariance
promotes positive residual eigenvectors; win_probabilities_factor
uses shared cavities), factor/races.py (_fit_cov and GHK route),
methods/native.py, bench/runner.py, research/general_sigma/NOTES.md,
papers/general_inversion/CLAIMS.md. PR #1093 open/draft, base
898a9f685bec35f88f6eeef10227bd6d302d7f83 at inspection. Recent
PRs unrelated. Prior signed_field.py, signed_rank2.py and
envelope.py inspected. This is not a new dense fitting algorithm,
nor a claim that block independence or Gaussian CDFs are novel.

**Exact broader block grammar and derivation.** Partition the
contestants into independent blocks B of size 1, 2 or 3, with
arbitrary supplied PSD covariances C_B, positive marginal variances,
no perfectly correlated pair in a block (conservative refusal
including possible permanent ties). An ordinary low-rank positive
factor V V' can be added by outer conditioning; implemented rank 1.
For every i in B, let g_i(t)=f_i(t) P(X_j>t for j in B minus i |
X_i=t). Gaussian conditioning gives at most a bivariate CDF,
which is computed using Owen's T (or analytic singular limit).
All g_i are nonnegative, are densities of the block winning
threshold marked by the winner, and sum to G_B(t), the density
of the block minimum. Hence

  S_B(t)=integral_t^infty G_B(u) du,
  p_i=integral_R g_i(t) prod_(A != B) S_A(t) dt.

No Fourier inversion, no cancellation between oscillatory terms,
no per-winner large-dimensional MVN integrals. The essential
reuse is integrating the boundary insertions once to construct
the survival of each block, instead of pricing the block CDF
separately at every t, then the original multiplicative cavity
over blocks. Prefix/suffix products avoid division by zero.
A block of size 3 with all three off-diagonals strictly negative
cannot be rank-one plus diagonal: product of the three offdiag
entries in a rank-one representation is nonnegative. In particular,
B independent equal-scale equicorrelated triples with rho=-1/2
have B exact null modes. Differences of group-indicator null
vectors are centered and force every independent D_i=0 in any
exact positive-factor contrast representation when B>=2, since
a'Da + ||V'a||^2=0. Then rank(V) >= 2B; no low positive-factor
rank shortcut. These localized negative modes require no global
Fourier axis. At least three fixed blocks also forbid a nontrivial
contrast-equivalent change of covariance that preserves zero
off-block entries: 1a'+a1' with cross-block a_i+a_j=0 implies a=0.
The earlier bounded-width precision method covers nondegenerate
independent triples too; here the boundary formula is a simpler
alternative with no state tensor and works at exact rank-two
singularity. This is a useful special-purpose extension, not a
claim to supersede every existing tree/precision representation.

**Implementation / convergence.** triad_field.py. Conditional
standardized upper bounds are affine in t; for r=+/-1 use
Phi(min(h,k)) or max(0,Phi(h)-Phi(-k)), with symmetry for tails.
For |r|<1 use Owen's T formula for the bivariate CDF, including
explicit threshold-zero cases. Cancellation in the real CDF
formula still limits absolute/relative accuracy for very tiny
probabilities; checks CDF bounds, clips roundoff excursions,
does NOT normalize the race vector. Initial implementation
had a wrong sign-quadrant coefficient (1/4 instead of 1/2);
caught by direct adaptive CDF check and corrected before commit.
Reject invalid blocks and perfectly correlated pairs, no ridge.
Small negative block eigenvalues at the floating point boundary
are tolerated (32 eps in correlation scale), and computed
conditional correlation rounded to [-1,1] within a tolerance.
Very near-perfect pairs may be numerically unresolved; not a
certified floating-point method.

On a common performance mesh, evaluate g at endpoints and
quarter points of each panel. Integrate each half-panel by
positive Simpson weights, and accumulate these integrals
backwards to get S_B at endpoints/midpoints. Apply Simpson
again to g_i times the product of other block survivals. This
keeps the quadrature nonnegative without assuming a smooth
interpolated primitive is positive. Add mesh breakpoints where
conditional thresholds cross zero or where the singular wedge
changes (h=k or h=-k); add +/-{.5,1,2,4,8} times transition
width if narrow compared to the nominal mesh spacing.
The latter addresses near-singular smoothed kinks, but mesh
counts and error need not be monotone in the base panel count.
Cost O(n L) and storage O(n L) for L ACTUAL panels; fixed size
block preprocessing O(n), sorting knots O((L+n) log(L+n));
there can be O(n) additional knots, so worst-case O(n(L_base+n))
rather than claiming global linear time at fixed base resolution.
Global positive rank-one adds Q outer nodes, each with its own
conditional mesh; no preprocessing amortization in timings.

With positive pairwise conditional variance, there are no
permanent ties, even with a rank-two singular triple. All
boundary densities are continuous (piecewise smooth at singular
wedge changes). On fixed finite domain the maximum mesh size
tends to zero as panels increases, so the positive composite
rules converge, and the finite product and integrals converge
to the truncated shared field. Use lo=min_i(mu_i-pad*s_i),
hi=max_B min_(i in B)(mu_i+pad*s_i), so every block's
omitted upper-tail mass <=Phi(-pad). Union bound below lo
<=n Phi(-pad), above hi <=Phi(-pad). Removing other block
upper-tail masses changes any share by <=B Phi(-pad), using
telescoping bounded products and integral g_i<=1. Thus per-share
domain error <=(n+B+1)Phi(-pad), distinct from discretization.
Take pad and panels to infinity in a diagonal sequence. For
global positive factor, condition on factor, integrate true
bounded continuous p_i(f) against standard Gaussian, and
refine the GH order (positive Gauss-Hermite measures converge
weakly to Gaussian by moment determinacy); for each finite
order refine inner meshes. No structural or sampling error;
finite-resolution diagnostics and floating-point error are
not certificates. Scope excludes exact perfect-pair ties
rather than silently defining a different dead-heat rule.
Arbitrary PSD outside this structure not solved.

**Evidence.** Owen (1956), Ann. Math. Stat. 27:1075-1090,
DOI 10.1214/aoms/1177728074, and Plackett (1954), Biometrika
41:351-360, DOI 10.1093/biomet/41.3-4.351, bibliographic/publisher
records checked. Established bivariate CDF and reduction formulas,
not a priority claim. Genz SOV is the independent stochastic
comparator. Eight standalone tests passed (0.66s), NumPy 2.3.5,
SciPy 1.17.0, single-thread local. Bivariate CDF against direct
adaptive integral, zero threshold and r=+/-1 endpoints; 3-runner
asymmetric contrast integral at rho=-.5,-.49999,.35 (atol 4e-9);
four-runner independent arcsine trivariate orthant formula
with hetero singular triple and singleton (atol 2e-9), and with
global positive factor (atol 2e-8); shift/scale/permutation,
factor common-shock invariance, nonnegative raw mass, sixty
exact-singular symmetric runners (atol 1e-8), invalid/possible
tie cases refused. Not full repository CI or fastrace.

Commands in this directory (one thread for all):
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m pytest -q test_triad.py
  python -u bench_triad.py 30 near
  python -u bench_triad.py 30 hetero
  python -u bench_triad.py 30 mixed
  python -u bench_triad.py 60 near
  python -u bench_triad.py 120 singular --no-genz
  python -u bench_triad.py 30 dense

Seed 90213: means N(0,.5^2), hetero sd uniform [.75,1.25]
or geometric [.12,1.6] within each triple. Near rho=-.49999,
exact singular rho=-.5, mixed random 3x3 Wishart+.4I standardized
with mixed-sign correlations. Dense additionally seed 90314
global loading N(0,.25^2). Refined deterministic reference
4096 vs 8192 base panels, dense 512/25 vs 1024/31 (panels/GH).
Reference differences not independent accuracy certificates.
Compiled SciPy _qmvn (Genz permuted Cholesky, CBC lattice,
8 randomized batches per winner, four seeds 0..3) normalized
for comparison. Full vector, all preparation and mesh generation
included in field timing, Genz per-winner prep included; no
amortization. RMSE pooled across runners/scrambles; relmax
median per scramble for reference p>1e-4. Representative actual
measured results:

| case | method | sec | RMSE vs refined | max/median max | relmax | scramble SD |
|---|---|---:|---:|---:|---:|---:|
|30 near|field 128|.03375|3.02e-8|7.47e-8|3.03e-6|–|
||field 256|.04032|9.43e-10|2.78e-9|1.01e-7|–|
||Genz 2048|.21437|2.08e-4|8.00e-4|.02674|2.10e-4|
||Genz 8192|.55992|9.51e-5|3.57e-4|.00795|7.88e-5|
|30 hetero|field 256|.05899|2.45e-8|1.05e-7|9.20e-5|–|
||field 1024|.09775|7.69e-10|2.41e-9|3.14e-8|–|
||Genz 2048|.21319|2.07e-5|8.74e-5|.00372|2.09e-5|
||Genz 8192|.54884|1.09e-5|4.77e-5|.00198|1.24e-5|
|30 mixed|field 128|.01531|7.50e-8|2.05e-7|6.24e-6|–|
||field 256|.02411|1.41e-9|3.57e-9|2.07e-7|–|
||Genz 2048|.27322|3.69e-5|1.21e-4|.00424|3.54e-5|
||Genz 8192|.68769|1.77e-5|7.29e-5|.00119|1.60e-5|
|60 near|field 256|.11628|8.36e-10|3.20e-9|3.51e-8|–|
||Genz 2048|1.23185|1.64e-4|6.67e-4|.03217|1.69e-4|
|120 singular|field 256|.02219|2.06e-9|1.17e-8|1.28e-7|–|
||field 512|.03123|1.10e-10|6.26e-10|6.93e-9|–|
|30 dense|field 128/11|.44680|3.91e-8|1.13e-7|6.64e-6|–|
||field 256/15|.64530|2.10e-9|6.26e-9|6.05e-7|–|
||Genz 2048|.20973|2.14e-4|9.00e-4|.01699|1.57e-4|
||Genz 8192|.53913|7.58e-5|2.36e-4|.00690|8.20e-5|

Refinement max differences in case order 30 near, hetero, mixed,
60 near,120 singular,30 dense: 4.44e-10,1.33e-10,7.33e-15,
3.70e-10,2.25e-13,2.13e-10. Reference raw mass defects
3.08e-10,2.28e-9,5.33e-15,7.73e-10,2.51e-13,4.41e-9.
At 256 base panels raw mass defects 1.80e-8,8.33e-8,3.21e-8,
2.61e-8,1.20e-7 and dense 256/15 3.68e-8. All shares nonnegative,
but no tail relative claim for heterogeneous minimum ~1.39e-47.
60 Genz 8192 timed out, no timing or accuracy reported. No
Genz comparison for 120 exact singular, only symmetry small
reference and refinement asymmetric fixture. 30 dense is slower
than coarse Genz but much smaller disagreement with refined
field at comparable budget. 30 near coarse 64 used 1014 actual
panels vs 518 at 128 (sharp-knot inclusion); 120 singular
64 used 2768 vs 432 at 128, partly due rounding conditional
r to just inside +/-1. Nonmonotone cost is real, no uniform
speedup assertion. Original Fourier sparse fixtures with
support 4/5/8 not directly covered here; this is a stable
special-case alternative rather than a fix to Fourier
inversion itself. Earlier independent-block width-2
implementation not timed against triad solver this run.
No production interface changes, existing draft PR updated.

**Next:** generalize localized blocks to size four through
deterministic trivariate boundary CDFs (e.g. Plackett reduction
or one conditional Gaussian quadrature). Investigate direct
low-rank geometry for singular blocks, and hybrid complex
survival for one global diffuse negative constraint with
small correlated local blocks. Separate local adaptive meshes
and positive interpolation could remove O(n^2) shared-knot
worst case. A general dense convergent estimator and comparison
with fastrace remain open. Do not pretend that increasing
block size without bound makes the local orthant cost practical.


## Iteration 11: block-Fourier hybrid (one diffuse negative mode on correlated pairs)

At main 898a9f685bec35f88f6eeef10227bd6d302d7f83 the actual
blocks.py conditions on positive cluster factors; races.py _fit_cov
routes degraded dense normal fits to GHK. test_cov_routes_to_ghk
confirms the distinction. PR 1093 remains open/draft. AGENTS.md not
present at root. No production interface changes.

For SPD independent blocks C_B of size 1 or 2, let C=blockdiag(C_B),
U=beta'(X-mu)+sqrt(1-beta'C beta)*epsilon, var(U)=1.
The target covariance given U=0 is C-cc', c=C beta. No approximation.
This represents local positive/negative dependence and a global
negative mode, at the cost of just one frequency axis. For each
block define S_B(t,k)=E[exp(ik U_B) 1{all X_B>t}], U_B=beta_B'(X_B-mu_B).
Let g_i(t,k) be the boundary density with the same complex weight.
For a pair, condition on X_i=t: U_B has deterministic part
c_i(t-mu_i)/C_ii and random part beta_j q Y, where q is the
conditional SD of X_j. Thus g_i is marginal normal density times
exp(ik c_i(t-mu_i)/C_ii) times the truncated characteristic function
F((t-E[X_j|X_i=t])/q,k beta_j q). Singleton is the same without F.
Reconstruct S_B by reverse integrating sum_i g_i over performance.
Fourier inversion at U=0 gives p_i=sqrt(2/pi) int_0^infty
exp(-noise*k^2/2) Re int g_i prod_{A != B} S_A dt dk.
Prefix/suffix cavities avoid division by complex near-zero quantities.

Useful general bound: condition on the Gaussian residual of X_B given
U_B. The truncation event then restricts the scalar U_B to an interval
(possibly empty, unbounded or all R); by integration by parts the
absolute truncated Gaussian Fourier transform is <= min(1,
sqrt(2/pi)/(k sd(U_B))). This holds for ANY Gaussian block, not just
pairs. Therefore if >=3 blocks have nonzero U_B variance, the product
outside any winning block supplies >=2 factors of 1/k and inversion
is absolutely convergent even with no observation noise. For the
frequency bound at K use active r_B=max(1,K sd(U_B)/sqrt(2/pi))
and integrate the power-law bound from K to infinity, or use the
Gaussian observation noise bound, whichever is smaller. A positive
observation noise also suffices without 3 blocks. In exact singular
case a two-contestant contrast cannot be proportional to the
observation vector when it has nonzero variance on 3 independent
blocks; hence no permanent ties in scope. Refuse singular local
blocks and sparse singular constraints rather than invent a tie rule.
Repair only observation variance excess within roundoff, no ridge.

Finite performance window: omitted race winner tail <=(n+1)Phi(-pad)
since target marginal variances are no larger than base. Omitting
the upper tail when reconstructing each S_B changes it by at most
Phi(-pad), in modulus. Telescoping products and integrating
|g_i|<=f_i gives extra finite-frequency bound
sqrt(2/pi)*K*(B-1)*Phi(-pad). Bounds in exact arithmetic only; not
quadrature bounds. On finite rectangle composite Simpson and
Gauss-Legendre converge by continuity; choose K,pad increasing
with K Phi(-pad) -> 0 and refine quadrature for each finite rectangle.
No sampling error or structural error. Cost O(n L Q) for actual
panels L and frequency nodes Q; memory O(n L batch). Added local
knot refinement can make L larger than base request. Complex
quadrature can return negative tiny shares; no clipping/normalization.
The block-level decay bound also suggests generalization to arbitrary
small correlated blocks if their complex boundary densities can be
evaluated cheaply. Dutt (1973), Biometrika 60:637–645, uses Fourier
transforms for MVN integration; Sun (1988), DOI
10.1080/03610928808829845, gives orthant reductions. Genz (1992),
DOI 10.1080/10618600.1992.10477010, describes SOV. Bibliographic
abstracts checked; no priority assertion for Fourier conditioning.

Standalone implementation hybrid_pair.py; test_hybrid.py (4 passed
in 1.07 s; independent four-runner trivariate arcsine reference at
eps=.2, exact singular symmetric 16-runner case, 12-runner
heterogeneous near-null shift/scale/permutation, invalid inputs).
No full repo CI, no common-shock API test, no fastrace comparison.

Commands in research/shared_field_research:
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m pytest -q test_hybrid.py
  python -u bench_hybrid.py 32 mixed
  python -u bench_hybrid.py 32 hetero
  python -u bench_hybrid.py 64 singular
  python -u bench_hybrid.py 32 hard

Seed 2026: alternating local correlation signs with magnitude .65-.9,
global observation beta normalized to variance 1-1e-5, means N(0,.4^2),
hetero sd uniform [.7,1.3]. Full-vector normalized SciPy _qmvn
compiled SOV with greedy permuted Cholesky and CBC lattice, 8 batches
per contestant, 4 seeds 0..3. Preparation included; no amortization.
Reference deterministic (1024,96,32) vs (1536,112,40) base panels,
frequency order, cutoff: max differences 1.75e-12 and 2.40e-12.
Not an independent large-dimensional certificate. Measured results:
case | method | seconds | RMSE vs refined | med max | med relmax p>1e-4 | pooled scramble SD
32 mixed | field 96/24/10 | .1656 | 3.94e-7 | 9.53e-7 | 1.16e-5 | -
32 mixed | field 192/32/14 | .1757 | 1.36e-9 | 2.99e-9 | 7.86e-8 | -
32 mixed | field 384/48/20 | .5343 | 2.88e-11 | 5.79e-11 | 2.32e-9 | -
32 mixed | Genz 2048 | .2489 | 8.09e-5 | 2.26e-4 | .01069 | 8.43e-5
32 mixed | Genz 8192 | .6650 | 2.85e-5 | 8.45e-5 | .00227 | 2.81e-5
32 hetero | field 96/24/10 | .1669 | 2.24e-7 | 5.95e-7 | 3.13e-5 | -
32 hetero | field 192/32/14 | .1744 | 3.42e-9 | 1.02e-8 | 5.59e-7 | -
32 hetero | field 384/48/20 | .5090 | 2.63e-10 | 8.22e-10 | 2.29e-8 | -
32 hetero | Genz 2048 | .2499 | 1.02e-4 | 3.58e-4 | .00913 | 1.04e-4
32 hetero | Genz 8192 | .6346 | 3.08e-5 | 8.43e-5 | .00239 | 3.26e-5

At 192/32/14, mass defects 3.55e-8 and 3.22e-8; omitted-frequency
bounds 1.26e-7 and 4.26e-8 respectively, not bounds on quadrature.
All reported shares positive; reference minimum .000523 and .000100.
Fine reference times 4.51 and 4.74 s. Combined benchmark invocation
timed out during the 64 singular case, no result there or hard case.
Do not infer favorable performance for severe local near-singularity,
localized global loadings, or arbitrary covariance. The exact grammar
is useful, but automatic decomposition and production integration
remain open. Next: stress localized observation and very sharp pairs
at moderate budgets; extend g_i to triples by a one-dimensional
conditional complex quadrature, or use direct complex CDF formula.
Also investigate positive factors outside the hybrid and independent
validation in larger dimensions. 


## Iteration 12: trivariate blocks with a global negative mode; reusable observation quadrature

Main still at 898a9f685bec35f88f6eeef10227bd6d302d7f83 according to PR base.
Read main CLAUDE.md (root AGENTS.md absent), README, factor/races.py and
blocks.py and general_sigma/NOTES.md; dense degraded fits go to GHK,
positive block engine is conditional on cluster factors, residual rank
ladder already exists. Reviewed prior hybrid_pair.py, test_hybrid.py and
bench_hybrid.py; recent shared-field PR search showed our existing draft
1093 and no obvious competing triple Fourier implementation. No production
files modified. This is an extension of the previous exact hybrid grammar,
NOT the universal covariance goal.

Let C be block diagonal with SPD blocks of sizes 1--3, c=C beta,
beta'C beta<=1. The target C-cc' is unchanged. For a triple winner i at
performance t, write the residual two-vector after conditioning on X_i=t
as L E, with E standard normal in R^2, and let h=L' beta_rest,
a=||h||. Rotate E so its first coordinate z is parallel to h; then
L E = v z + w y with independent standard normal z,y and observation
residual a z. The two inequalities v_j z + w_j y > d_j(t)
give an interval [ell(t,z),u(t,z)] for y (hard condition if w_j=0).
The complex boundary density is the marginal density at t times
exp(ik c_i(t-mu_i)/C_ii) times

    integral phi(z) exp(ik a z) [Phi(u)-Phi(ell)]_+ dz.

If a=0 any rotation works. This is exact; no complex bivariate CDF
needed. Use the real interval probability and a stable survival-CDF
difference when both bounds positive. At most one kink occurs when
two affine bounds cross, at z=(d1*w2-d2*w1)/(v1*w2-v2*w1).
The first implementation uses separate GL nodes per t row, split at
this kink and around step transitions. The improved method uses a
common z grid, composite 4-point GL with 4*nz cells and BLAS product
of the real t-by-z weight matrix and complex z-by-frequency phase
matrix. Remove the one panel containing the moving kink and replace
by split quadrature for that row. A sharp step (|w_j|<.06|v_j| or
zero) falls back to row-dependent rule. This is the key new reuse:
complex phases computed once per boundary instead of at every t.
No claim that the factorization itself is new Gaussian conditioning.
Reuse previous reverse Simpson primitive for block survivals and
prefix/suffix products. For fixed local block size, O(n L Qz Qk)
arithmetic at actual performance panel count L; common-grid method
uses BLAS plus O(n L Qk) kink replacements; fallback may use more
memory O(L Qz Qk) for one contestant at a time. Full frequency batch
helps but raises peak memory; batch as appropriate. No amortized
preparation in measured timings. Adaptive knots can make L much
larger than requested base panels.

Second candidate: a Price/Plackett differential equation for the
complex truncated characteristic function T(r)=E_r[exp(i(aY+bZ))
1{Y>h,Z>j}]. Distributional mixed derivative gives
T'(r)=-ab T(r)+B(r), where B is the corner density times phase
plus ia times the Z=j edge transform and ib times the Y=h edge
transform. Each edge uses one univariate truncated characteristic
function. Thus T(r)=exp(-ab(r-s0))*T(s0)+ integral_s0^r
exp(-ab(r-s))*B(s)ds. Choose s0=0 if ab*r>=0, otherwise sign(r),
using the analytic degenerate one-dimensional interval at +/-1.
Then the integrating factor never exceeds 1 (frequencies in a
single boundary have common product sign); r=sin(theta) cancels
the corner density Jacobian singularity. This is an exact formula,
but fixed angular order can miss narrow transitions near endpoints.
It was slower than the common-grid rotation in the tested case.
Price (1958) and Voigtlaender (2021), DOI
10.1007/s10959-020-01017-w (distributional generalization),
Plackett (1954), DOI 10.1093/biomet/41.3-4.351, checked publisher
abstracts. No novelty claim for Gaussian covariance differentiation.

Convergence: same frequency bound as iteration 11 (at least three
active blocks at zero observation noise); no permanent ties in that
scope, because a two-contestant contrast cannot span the observation
on three independent blocks. Local singular C refused by Cholesky;
no ridge. Near-boundary variance excess repaired only within roundoff.
The inner z truncation in rotation methods changes the integrated
boundary transform by <=2 Phi(-zpad) per runner, and a reconstructed
block survival by <=2 |B| Phi(-zpad). Exact truncated transforms have
modulus <= the corresponding unweighted probability (integrated
over their boundary-defined subevent), hence telescoping products
gives extra per-share bound sqrt(2/pi)*K*2n Phi(-zpad) on finite
frequency interval [0,K]. This is separate from performance and
frequency truncation, not a quadrature certificate. For fixed finite
windows refine inner z quadrature then t and frequency quadrature;
piecewise smooth interval probabilities and finite domains imply
convergence by bounded integration. Move cutoffs outward in a
diagonal sequence with K Phi(-pad) and K Phi(-zpad) ->0, and
frequency bound ->0. No structural approximation or sampling error
in the field. Complex cancellation may produce negative tiny values;
no clipping or normalization. The reference code does not itself
report all these truncation bounds yet. Finite precision and
near-singular conditional Cholesky remain limitations.

Measured on local NumPy/SciPy 1.17.0 (single BLAS thread) using
separate modules, consolidated equivalent functions and tests
committed as triple_fourier.py and test_triple_fourier.py; the
consolidated remote checkout has not been executed as a full repo
suite. Direct local checks: eight Price sign/correlation cases
(r=+/- .4,.9; a=1.4,b=+/-2.3, h=.2,j=-.6) against scalar adaptive
conditional integral, max 1.39e-16; four-contestant independent
trivariate arcsine reference, max error 1.24e-10 at 256/64/32/24
(panels/nk/K/nz); 12-contestant exactly singular symmetry error
3.08e-9; 24 mixed/hetero shift, scale and reversal of blocks and
within-block contestants combined, max 1.11e-16. Positive raw shares
on reported fixtures. No common-shock API test.

Reproduce in this directory:
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m pytest -q test_triple_fourier.py
  python -u test_triple_fourier.py 24 mixed
  python -u test_triple_fourier.py 24 hetero
  python -u test_triple_fourier.py 36 mixed --no-genz
  python -u test_triple_fourier.py 24 hard --no-genz

Seed 34: 8/12 triples drawn as Wishart+.3I standardized, mixed signs;
heterogeneous scales uniform [.5,1.5]. Means N(0,.4^2), global
observation variance 1-1e-5. Full vector normalized compiled SciPy
_qmvn Genz/SOV with greedy permutation and CBC lattice, 8 batches
per winner, 4 evaluation seeds 0..3 and reference 4 seeds 100..103
at 32768 points per winner. Reference pooled between-scramble SD
9.78e-6 mixed and 1.85e-5 hetero; uncertainty dominates field error
comparisons. Actual timings from local prototypes:
case | method | seconds | RMSE vs mean Genz ref | pooled scramble SD
24 mixed | Genz 2048 | .1274 | 7.96e-5 | 8.20e-5
24 mixed | Genz 8192 | .3419 | 3.08e-5 | 2.81e-5
24 mixed | row rotation 96/28/14/10 batch8 | 2.6568 | 5.58e-6 | -
24 mixed | row rotation 192/40/20/14 batch8 | 6.3064 | 5.28e-6 | -
24 mixed | Price 96/28/14/16 batch8 | 4.1086 | 5.29e-6 | -
24 mixed | fast 96/28/14/10 batch28 | .5735 | 5.29e-6 | -
24 mixed | fast 192/40/20/14 batch40 | 1.0638 | 5.28e-6 | -
24 hetero | Genz 2048 | .1199 | 1.00e-4 | 1.05e-4
24 hetero | Genz 8192 | .3261 | 5.11e-5 | 4.17e-5
24 hetero | fast 96/28/14/10 batch28 | .7465 | 9.04e-6 | -
24 hetero | fast 192/40/20/14 batch40 | .7712 | 9.04e-6 | -

Fast maximum disagreement vs stochastic reference ~1.30e-5 mixed,
2.13e-5 hetero; max relative for p>1e-4 .000575/.000476.
Fast coarse/fine maximum differences 1.04e-7 (mixed), 9.09e-8
(hetero), 3.62e-6 (36 mixed; .986s vs 1.767s). This is refinement,
not independent certification. Raw mass defects coarse/fine
3.55e-7/2.55e-7 mixed, 6.76e-7/1.65e-7 hetero, 1.40e-6/2.61e-7
36 mixed. Fast vs row reference at fine resolution had max
5.93e-8 mixed, but 1.14e-6 hetero (row reference itself had
mass defect 1.04e-6). Both require better convergence diagnostics.
Hard triple fixture (rho=-.49995) did not finish during combined
benchmark before timeout; no robustness or speed claim. 36 did
not get independent Genz comparison. The fast method is much
faster than the original triple prototype but slower than coarse
Genz; accuracy advantage at matched runtime remains suggestive
not established, since reference is noisy and budgets not matched.
No production deployment claim. Existing PR already open; preserve
these additions on its research branch, do not open duplicate.

Next: cache real quadrature weights across frequency batches,
trim performance knots whose boundary densities are negligible,
and tackle near-singular local blocks by reducing to their
intrinsic rank, rather than brute mesh escalation. The Price ODE
is a possible general recursive complex CDF tool but not yet the
fast choice. Automatic signed covariance decomposition, fully
general PSD, compiled fastrace comparison and full CI remain open.


## Iteration 13: intrinsic-rank local blocks and endpoint continuation

Re-read main README, CLAUDE.md (root AGENTS.md absent), factor/races.py,
factor/blocks.py and research/general_sigma/NOTES.md. Current code
still prices positive private factors and routes degraded dense fits to
GHK; its dense fallback has a ridge for failed Cholesky. This experiment
does not change that API. PR 1093 is open on this research branch;
reviewed prior triple_fourier.py and test_triple_fourier.py and searched
recent shared-field PRs. No competing intrinsic-root implementation found.
Dutt (1973), DOI 10.1093/biomet/60.3.637, and Genz/Trinh (2016),
DOI 10.1007/978-3-319-33507-0_13 publisher abstracts checked; these
establish Fourier and bivariate conditioning precedents, not novelty
of either. The contribution is use of exact local root geometry in
the shared all-winner boundary field.

Write independent local X_B=mu_B+R_B Z_B, allowing rectangular roots
with 1--3 rows and full column rank. Let h_B=R_B' beta_B. Observe
U=sum h_B' Z_B+sqrt(1-sum||h_B||^2)*epsilon=0, so target
C-C beta beta' C, C=blockdiag(R_B R_B'). Conditional on X_i=t,
take an orthogonal basis Q with first column parallel to row R_i.
The conditional residual root is L=R_rest Q[:,1:], computed without
subtracting nearly equal covariance matrices. Its SVD gives independent
dominant and minor directions v z+w y, with score a z+b y, plus the
usual deterministic regression score. For given y the two inequalities
v_j z+w_j y>d_j(t) restrict z to an interval (ell(y),u(y)).
The boundary transform is exactly
  integral phi(y) exp(i k b y) [F(ell(y),ka)-F(u(y),ka)] dy
where F is the univariate truncated characteristic function. In exact
rank-two triples there is NO y at all: use just the interval transform.
This removes the entire local quadrature dimension, even though
the target may have arbitrarily many null modes and dense correlations.
Reflect left-tail intervals to avoid subtracting two total masses.
Use explicit rectangular roots, not a rank threshold to discard small
positive eigenvalues; numerical rank validation refuses unresolved roots.
Reject proportional local rows (even if their means differ) and zero
marginal variances rather than silently choose a tie rule. At zero
observation noise require three active independent blocks, which makes
every two-runner contrast nondegenerate in this scope. No ridge; only
roundoff-level rescaling if the observation variance is just above one.

For full-rank near-singular triples, an alternative to quadrature over
y is continuation from the nearest rank-one endpoint of the Price ODE
from iteration 12. Let r=s cos(theta), s=sign(r), q=sin(theta), solve
[v/q_j, w/q_j]' (A,B) = (a,b) where q_j is marginal conditional sd
(not sin(theta) here), then multiply A,B by frequency k. The derivative
in theta uses -s times the corner and q-weighted edge terms. The factor
exp(s A B (delta(theta_end)-delta(theta))) uses delta=2 sin(theta/2)^2.
Computing the conditional thresholds via j-s*h+s*delta*h avoids
subtracting nearly equal correlations. Split theta near |j-s*h|.
Refuse the endpoint route when its integrating factor exceeds exp(2)
and use direct minor-axis quadrature instead. Initial prototype
incorrectly projected the observation score onto the standardized
residuals rather than solving the linear system; mass defect .26
exposed that error, fixed before commit. Direct quadrature over the
minor axis was also much too slow on the 24-runner hard fixture; the
first unoptimized run timed out. The corrected endpoint route works
but is still slower than Genz; no production recommendation.

The same shared prefix/suffix cavity and reverse primitive as iterations
11--12 apply. Cost O(n L Q) for exact rank-two triples at actual mesh
count L and Q frequencies, no local quadrature; full rank O(n L Q J)
where J is actual local nodes (angular or minor-axis). Memory O(n L
batch), plus local temporary arrays for full-rank minor integration.
Preprocessing is O(n) for fixed block sizes, knot sorting adds a log
factor; the mesh may have O(n) additional knots, so L is not fixed
independent of n. Convergence follows the earlier block Fourier bound
(the proof uses only the variance of the scalar block score, not
invertibility of its root). For fixed finite K,pad, refine performance,
frequency and, where applicable, local quadrature. Piecewise continuous
boundary functions are integrable; at a rank-two triple's moving
interval endpoint they are continuous, including where the interval
vanishes. Then enlarge K,pad so K Phi(-pad)->0 and frequency bound
vanishes. Direct minor-axis truncation adds at most 2 Phi(-pad) per
boundary integrated over t and the earlier telescoping product bound.
Endpoint angular integral has no minor truncation. No structural or
sampling error within this grammar; finite-frequency and performance
truncation and quadrature distinct from floating-point cancellation.
No formal finite-resolution certification or arbitrary-matrix claim.

Local tests: 5 passed (1.10 s, NumPy 2.3.5, SciPy 1.17.0, one thread).
Interval CF vs scalar adaptive integration; exact local singular
4-contestant trivariate arcsine reference (512 panels, atol 5e-9;
256 panels missed that tolerance by reaching 1.90e-8); simultaneous
global and local singularity symmetry/mass; joint translation, scale,
block and within-block permutation; near-singular Price endpoint
against scalar adaptive integral at six points (max 1.80e-9 at
angular order 64). No full repository suite or common-shock API test.
The remote source has equivalent arithmetic to the locally tested
prototype; no production files changed.

Reproduce from this directory:
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m pytest -q test_intrinsic_fourier.py
  python -u test_intrinsic_fourier.py 24 exact
  python -u test_intrinsic_fourier.py 24 exact --root-reference
  python -u test_intrinsic_fourier.py 60 exact --root-reference
  python -u test_intrinsic_fourier.py 24 hard

Seed 184, rank-two local equilateral root (cos theta,sin theta) for
each triple; hard adds 1e-5 common variance to each local block
(and rescales the other two directions). Means N(0,.4^2), global
observation variance 1-1e-5. Exact local rank 16 at n=24 and rank 40
at n=60; dense mixed-sign cross-block dependence after conditioning.
Four Genz evaluations, seed 0..3, normalized, compiled SciPy _qmvn
with greedy Cholesky permutation/CBC lattice and eight batches per
winner, prep included; reference seeds 100..103 at 32768 points per
winner. Independent root sampling reference (four scrambled Sobol
sets, 2^18 each, seeds 0..3) uses y=Z-h(h'Z+sqrt(noise)*epsilon),
which has exact conditional latent covariance without ridge.
No amortization. Final optimized exact rank measurements:
case | method | seconds | RMSE vs mean Genz ref | pooled scramble SD
24 exact | Genz 2048 | .1262 | 3.558e-4 | 3.382e-4
24 exact | Genz 8192 | .3340 | 1.787e-4 | 1.722e-4
24 exact | field 96/28/14/6 | .1262 | 4.994e-5 | -
24 exact | field 192/40/20/8 | .2440 | 4.994e-5 | -
24 exact | field 384/56/28/8 | .5158 | 4.994e-5 | -
Genz reference SD 8.19e-5 (mean uncertainty roughly 4.09e-5);
thus field RMSE against this reference is noise-limited. At matched
.126 s budget the field agrees much better with independent Genz
reference than low-budget Genz. Median-relative metric not computed;
max relative difference for reference p>1e-4 is .00255 (noise-limited).
Field raw mass defect 5.57e-8/6.64e-9/3.55e-9. Successive refinement
max differences 2.00e-8 and 1.14e-9 (before pruning empty intervals;
arithmetic unchanged). All shares positive, minimum .0018591.

Independent root reference, earlier unpruned implementation:
24 exact | field coarse/fine .193/.915 s | RMSE 7.69e-5
60 exact | field coarse/middle/fine .595/1.211/2.557 s | RMSE 8.09e-5
Root reference pooled SD 1.65e-4/1.62e-4; refined 60 field differences
7.16e-8 then 4.38e-9, mass defects 1.17e-6/6.86e-8/1.91e-9.
60 exact low-budget Genz timings 1.068/2.600 s (2048/8192),
pooled SD 2.83e-4/1.13e-4; higher-budget Genz reference timed out,
so no matched-accuracy claim at 60. These refinement comparisons
are not independent accuracy certificates.

24 hard (corrected endpoint, before final empty-interval pruning):
Genz 2048 .1204 s RMSE 3.58e-4, SD 3.40e-4;
Genz 8192 .3378 s RMSE 1.80e-4, SD 1.76e-4;
field 64/24/12/4 3.031 s RMSE 5.07e-5, raw mass defect 6.45e-6;
field 128/32/16/6 7.241 s RMSE 5.07e-5, mass defect 1.85e-7.
Reference SD 8.05e-5. Refinement max difference 6.29e-7.
The final integrating-factor guard may change full-rank timings if
activated; these are not a competitive result. Need robust performance
for near-rank-deficient but nonsingular local blocks and heterogeneous
scales, strong observation along minor directions, very small shares,
and arbitrary PSD. Next useful experiment: Taylor-expand the analytic
dominant-axis interval endpoints in the minor coordinate, split at
their crossing, integrate the complex truncated Gaussian moments
analytically; then refine polynomial order to recover exact transform
on finite windows. It may remove local quadrature work for small
minor modes without discarding them. Alternatively, exploit roots of
rank two for larger blocks: conditional residual dimension one no
matter block size, so intersect all other inequalities to form one
interval. This latter extension is particularly promising, although
all-winner local boundary construction would naively be O(b^2) for
block size b. Investigate planar lower-envelope geometry to avoid that.


## Iteration 14: sparse planar hull cavities for arbitrary-size rank-two blocks

See [full derivation, experiments, reproduction commands, limitations and next steps](iteration14_planar.md). New `planar_field.py` and `test_planar_field.py`: arbitrary-size exact rank-two blocks with a scalar negative global mode, convex-hull neighbor pruning and affine envelopes. Five standalone tests pass, independent 13-way cell reference agrees to 7.19e-10. At 301 runners in one regular polygon, 90,300 directed constraints reduced to 602, with 2.88x measured speedup; dense 32/64 benchmarks and independent randomized root checks recorded in the linked note. No general PSD or production claim.


## Iteration 15: common-shift convolution with knot-split active spectra

See [derivation, reproducible tests and results](iteration15_shift_spectrum.md).
New shift_field.py and test_shift_field.py: any generic trivariate root
rotates into two contrast directions plus an independent common shift;
large rank-three blocks with planar contrasts also fit. The shift is
integrated by an exact complex Gaussian Fourier multiplier, not another
local quadrature axis. Naive FFT aliases sharp planar knots; integrating
only active Fourier coefficients on knot-split panels resolves that for
broad shifts. Seven standalone tests passed; four-way independent arcsine
reference max error 4.68e-11. 24 random full-rank triples and a global
near-null mode: .280 s and 5.18e-8 RMSE vs refined field; compiled Genz
.332 s and 4.05e-5 RMSE (four randomizations). At 60 contestants, .465 s
and 5.85e-9 refinement RMSE vs Genz 1.203 s and 1.02e-4. These are
refinement comparisons, not independent accuracy certificates; independent
stochastic checks noise-limited. Narrow shifts remain alias-sensitive and
can yield tiny negative shares. No ridge, structural fit, or production
changes. General PSD remains open.
