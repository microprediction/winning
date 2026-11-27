# Iteration 15: common-shift convolution, with knot-split active spectra

## Context and scope

Base reported by open draft #1093: 898a9f685bec35f88f6eeef10227bd6d302d7f83.
Previous head 0524dda6047e70cbe0e8714c36223225439f482d. Re-read main
README, CLAUDE.md, factor/blocks.py (_block_max_r), factor/races.py
(_race_dense, _fit_cov), tests/test_blocks.py, research/general_sigma/NOTES.md,
papers/README.md and papers/general_inversion/CLAIMS.md. Root AGENTS.md
absent. The existing rank-r block engine conditions on local factors and
uses private densities; dense fallback uses GHK, with ridge if Cholesky
fails. Existing residual factor ladder is not this. Read the planar
prototype, tests and iteration 14 note. Open shared-survival PR search
returned existing #1093 only. No production dispatch or main changes.

## Exact decomposition

Let a block have root T, with contrast span E=span{T_i-T_1} of
dimension 2. In an orthonormal basis of E and E-perp, the projection
onto E-perp is identical for every row. Rotate that common vector into
one coordinate; all remaining latent directions vanish. Thus exactly

    X_B = mu_B + R_B Z_B + sigma_B 1 Y_B,

with R_B of width two, Z_B standard normal in R^2, Y_B independent
standard normal. The block covariance is R_B R_B' + sigma_B^2 11'.
This includes generic full-rank triples (whatever the covariance,
not just equicorrelation), and arbitrary-size rank-three blocks with
rank-two contrasts. It also includes earlier exact rank-two blocks.
The independent common shift does not alter the winner within B,
only its performance relative to other blocks.

With scalar global observation U = sum_B (h_B' Z_B + a_B Y_B)
+ sqrt(1-sum_B (||h_B||^2+a_B^2)) epsilon, h_B=R_B' beta_B,
a_B=sigma_B sum beta_B, the target covariance is exactly

    C - C beta beta' C,  C=blockdiag(R_B R_B'+sigma_B^2 11').

The global observation may have zero noise: no unexplained ridge.
At zero noise require at least three blocks with nonzero score variance
(as in earlier iterations); then no nonzero two-runner contrast can
be proportional to U. Nonproportional, nonzero planar rows required;
permanent ties excluded rather than silently allocated. Cases with
proportional rows, unresolved contrast rank, deterministic performances
and general PSD outside the stated grammar are not supported.
Roundoff-only rescaling at total observation variance just above 1.
The decompose helper rotates a 3-by-3 explicit root using SVD of its
two contrast rows, retaining all directions; larger blocks accept
explicit planar roots and shifts rather than discarding small singular
values in an automatic rank test.

## Convolution and why naive FFT was not good enough

Let g_i^0(t,k) be the complex boundary density for the planar block
alone (iteration 14 analytic interval transform). Let g_i(t,k) include
the common shift and its contribution to U. Independence gives

    g_i(t,k) = E_Y [ exp(i k a_B Y) g_i^0(t-sigma_B Y,k) ].

With Fourier transform convention exp(-i omega t),

    ghat_i(omega,k) = ghat_i^0(omega,k)
                     exp[-(sigma_B omega-k a_B)^2/2].

No local quadrature dimension in Y, even if its variance is not small.
For sigma=0 this is the original unshifted boundary. Recover g on the
shared performance grid by inverse FFT, reverse integrate each block
sum for S_B, and multiply other blocks' survivals with prefix/suffix
products. Finish the usual scalar observation Fourier integral
sqrt(2/pi) int_0^infty Re ... exp(-noise k^2/2) dk.

The first implementation used uniform samples and FFT of g^0, then
the Gaussian multiplier. It worked for simple near-singular cases but
converged nonmonotonically at sharp planar envelope knots. For the
random arbitrary-triple fixture, 512/2048/8192/16384 grid point runs
had raw mass defects +.001929, -.0001782, +.0001114, -.0001316;
even 16384 was not a good reference. This is pre-convolution spectral
aliasing; smoothing AFTER aliasing cannot remove incorrect low modes.

Fix: integrate only the ACTIVE Fourier band directly, before smoothing.
For finite observation cutoff K, modes outside
|sigma_B omega| <= pad + K |a_B| have Gaussian multiplier <=
exp(-pad^2/2). For each runner use standardized winner coordinate z;
the other inequalities are affine lower and upper bounds in z.
Split a local Gauss-Legendre rule at pairwise crossings, where each
line hits normal-scale thresholds, and phase-resolution knots. Evaluate
the analytic interval transform at those nodes. For each retained omega
and quadrature node form matrix E with entries

    E[omega,z] = exp(-i omega (mu_i+s_i z-left)) w_z phi(z) / dx.

A single BLAS multiplication E @ values(z,k) for a batch of k gives
the DFT-scaled coefficients; multiply by the exact Gaussian filter,
zero omitted modes, inverse FFT. This has no uniform-sampling aliasing
in the active low modes. At sigma >= .05 (after global scale normalization)
use this branch; for smaller sigma the active band grows and uniform
FFT is currently cheaper. Threshold only selects numerical method,
not covariance. Both branches converge to the same convolution.
Spectral order is configurable (local_order); pad controls both local
normal truncation and Gaussian multiplier cutoff. This supersedes
the failed raw-FFT result; the latter remains selectable via
spectral=False for comparison. The prototype has not reused convex
hull pruning yet: all-pair bounds and pairwise crossing knots, so cost
can grow badly with block size. This run focuses on fixed small blocks.

For fixed b, M performance nodes, Q observation nodes, J local GL
nodes and F retained performance frequencies per runner:
uniform branch O(n M Q log M); spectral branch O(n F J Q + n M Q log M)
including inverse FFT, plus E setup O(n F J), with F,J dependent on
sigma, pad,K, not on M once Nyquist exceeds the active band. Memory
O(n F J+n M batch), currently stores E for each spectral runner.
In broad-shift cases F much less than M; raising M mostly improves
the final cavity integration and does not require repeated local
boundary evaluation at all M nodes. No preprocessing amortization
claimed. For variable block size initial all-pair arrays O(sum b^2 M)
work and spectral crossing construction up to O(sum b^3); use previous
envelopes/hulls to remove this overhead later.

## Convergence and limitations

For each fixed finite K and period T=[left,right], raw g is continuous
in t within scope (at empty interval the integral tends to zero), and
bounded by marginal Gaussian density. Periodize the truncated base
boundary on this interval. At sigma>0, the exact circular convolution
has absolutely summable Fourier coefficients because the Gaussian
multiplier decays in |omega|; direct coefficients converge under
local_order refinement on finite split panels. The uniform FFT
coefficients converge for each fixed mode by Riemann sums, and
Gaussian attenuation bounds the high-mode remainder uniformly as
M increases. Sigma=0 uses original boundary and no smoothing.
At fixed pad, the spectral branch omits a finite Gaussian-weighted
frequency tail; as pad increases that omitted tail vanishes, e.g.
each unshifted coefficient has modulus <= 1/T, and a sum of Gaussian
tail values bounds the omitted series. Nyquist truncation separately
vanishes under M refinement. Truncating z costs <=2 Phi(-pad) in
integrated raw absolute boundary mass; periodization/wrap error in
L1 bounded by truncated base mass plus convolved marginal Gaussian
tail mass outside the period. The chosen t window includes pad
marginal sds of every shifted runner, and thus also the unshifted
base. These tails vanish with pad. Reverse Simpson and outer Simpson
converge on finite rectangles; endpoint zero convention contributes
vanishing mesh error (and negligible tail in tested cases).
Telescoping products of block survivals bound finite-K error by
sums of boundary integration errors; true weighted survivals have
modulus at most one. Thus refine M, local_order and nk on finite
windows, then increase pad and K in a diagonal sequence with
K Phi(-pad)->0 and spectral omitted tails ->0. No structural or
sampling error within this grammar. At zero noise earlier bound
|S_B(t,k)|<=min(1,sqrt(2/pi)/(|k| sd(U_B))) still holds by conditioning
on residual latent directions; at least two untouched active blocks
for each runner yield integrable O(k^-2) decay or faster after
integrating |g_i|<=f_i. At positive noise Gaussian decay suffices.
This is a convergence argument, not a certified finite-resolution
error bound. Floating-point root/hull decisions, cancellation,
aliasing, frequency cutoffs and quadrature errors distinct.
One narrow-shift fixture produced a negative -6.97e-14 numerical
share at M=1024; not clipped or normalized. Tiny share relative
accuracy is not established. Global common shock invariance follows
by contrasts mathematically; arbitrary global common-shock input
is outside this prototype interface, not separately tested.

Primary sources checked: Dutt (1973) Biometrika 60:637–645 DOI
10.1093/biomet/60.3.637 (Fourier transforms of MVN integrals),
Genz (1992) DOI 10.1080/10618600.1992.10477010 (MVN integration
transformation), Genz and Trinh (2016) DOI 10.1007/978-3-319-33507-0_13
(bivariate conditioning), Tiwari et al. arXiv:2107.03958 (periodic
FFT convolution, smoothness/aliasing issues). No novelty claim for
Gaussian convolution, SVD, or Fourier transforms themselves.

## Experiments

Local source blobs verified equal to committed source; NumPy 2.3.5,
SciPy 1.17.0, single BLAS thread. Seven standalone tests passed
(3.23 s); no full repository CI. Analytic complex Gaussian multiplier
test, exact root rotation, single-block common shift invariance,
block exchange symmetry at exact global singularity, permutation/
translation/scale invariance, four-way independent trivariate
orthant arcsine reference (max absolute 4.68e-11), arbitrary triple
refinement, exact singular rank-three five-runner blocks.
The original symmetry test at M=2048 without spectral correction
failed its 2e-6 tolerance (block error 1.58e-5); refined uniform
M=8192 passed before spectral fix.

Commands from this directory:
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest -q test_shift_field.py
    python -u test_shift_field.py random
    python -u test_shift_field.py extra
    python -u test_shift_field.py 24 3 .00316227766
    python -u test_shift_field.py 24 3 .5

Seed 926 random fixture: eight independent trivariate roots first
unrestricted iid normals, then random orthogonal rotation of
diag(1.3,.8,.003), means N(0,.5^2), heterogeneous mixed signs,
global observation variance 1-1e-8. Each root is rotated into planar
and common components, preserving covariance to 1e-13 in code.
Seed 825 fixture: polygon angles jittered, row scales .6–1.4,
means N(0,.5^2), shift multiplier uniform [.8,1.2], global score
variance 1-1e-8. Extra 30-runner case uses 6 rank-three five-row
blocks and exact global singularity (rank <=17 of 30).
Compiled SciPy _qmvn Genz/SOV with greedy permutation, CBC lattice,
8 batches per winner, normalized full vector, 4 independent seeds
0..3, 32768 budget seeds 100..103; per-run mean wall time.
No ridge in baseline. Root sampling uses 4 scrambled Sobol
sets 2^18 (or 2^19 for singular) seeds 100..103; exact conditional
latent root via y=Z-h(h'Z+sqrt(noise)*epsilon). All timings include
preparation. For Genz the covariance supplied is the original target,
not the FFT approximation.

case | method (M/Q/K or Genz budget) | seconds | vector RMSE* | pooled Genz SD
24 arbitrary | hybrid 512/24/12 | .2804 | 5.18e-8 | -
24 arbitrary | hybrid 1024/32/16 | .4067 | 3.24e-9 | -
24 arbitrary | hybrid 2048/40/20 | .5080 | 2.02e-10 | -
24 arbitrary | hybrid 4096/48/24 | .6868 | 1.19e-11 | -
24 arbitrary | hybrid 8192/56/28 | .9887 | ref | -
24 arbitrary | Genz 2048 | .1230 | 1.13e-4 | 1.22e-4
24 arbitrary | Genz 8192 | .3324 | 4.05e-5 | 4.33e-5
24 arbitrary | Genz 32768 | 1.2287 | 1.68e-5 | 1.82e-5
24 rotated near | hybrid 512/24/12 | .0795 | 1.76e-5 | -
24 rotated near | hybrid 1024/32/16 | .1945 | 7.39e-6 | -
24 rotated near | hybrid 2048/40/20 | .4710 | 7.69e-7 | -
24 rotated near | hybrid 4096/48/24 | 1.0919 | 1.98e-7 | -
24 rotated near | hybrid 8192/56/28 | 2.6160 | ref | -
24 rotated near | Genz 2048 | .1261 | 2.08e-4 | 2.19e-4
24 rotated near | Genz 8192 | .3186 | 1.39e-4 | 1.51e-4
24 rotated near | Genz 32768 | 1.1106 | 5.27e-5 | 5.31e-5
60 shift .5 | hybrid 512/24/12 | .4650 | 5.85e-9 | -
60 shift .5 | hybrid 1024/32/16 | .4442 | 3.44e-10 | -
60 shift .5 | hybrid 2048/40/20 | .6244 | ref | -
60 shift .5 | Genz 2048 | 1.2030 | 1.02e-4 | 1.03e-4
60 shift .5 | Genz 8192 | 2.7590 | 4.18e-5 | 4.42e-5

* Against highest-resolution hybrid field for each fixture, not
certified or independent. Independent higher-budget Genz mean vs
field RMSE 5.61e-6 arbitrary, 2.56e-5 rotated near; root sampling
mean RMSE 9.85e-5/1.12e-4, pooled SD 1.96e-4/2.18e-4 respectively.
60 mean Genz RMSE 4.91e-5/1.67e-5 at the two budgets. At n=24
arbitrary, final max refinement difference 2.62e-11; raw mass defect
1.56e-11; changing local order 8->12 at M4096/K28/Q56 differs
1.33e-13, K36/Q72 differs 1.38e-13, pad12 differs 2.99e-11.
Max relative error vs refined for p>1e-4: coarse 4.59e-6 arbitrary,
.00142 rotated near. Near case unresolved aliasing remains:
M8192 mass defect 2.13e-6; separate M16384 run mass defect 2.25e-7,
RMSE 1.46e-7 vs M8192; reference not a high-precision truth.
The broad-shift spectral algorithm fixes the arbitrary fixture's
initial .00013 mass error even on a much smaller grid. These are
favorable small-block cases, not a universal Genz superiority claim.

30 exact-singular rank-three block fixture: .3335/.3383/.3718 s
for 512/1024/2048 mesh, max refinement differences 4.31e-8 and
2.70e-9, final mass defect 1.79e-9, min p=.00043197. Independent
root sampling mean RMSE 7.34e-5, pooled SD 1.27e-4; no Genz claim
for this singular case.

Next: eliminate narrow-shift aliasing with knot-local conservative
spectral correction or an analytic piecewise-polynomial convolution
in performance space; use previous convex hull/envelope pruning
for large blocks. Add diagnostics for retained band and local GL
error. Then try more than one global negative mode. General PSD
and production integration remain open.
