# Iteration 17: contrast-gauge preconditioning of signed observations

## Scope and source verification

Main base per draft #1093: 898a9f685bec35f88f6eeef10227bd6d302d7f83; previous research head 0bd137b99f3025249eb0114622ef246b35191042. Rechecked main README, CLAUDE.md (AGENTS.md absent), winning/factor/blocks.py, races.py, tests/test_blocks.py, research/general_sigma/NOTES.md and papers/general_inversion/CLAIMS.md. Main shared block fields integrate private independent noise given positive factors, and degraded dense fits route to GHK. Existing residual-promotion route is not this proposal. Open PR search for shared returned #1093. Read signed_field.py, prior negative factor note, iteration 15/16 notes and interfaces. The earlier whitened contrast lift in the likelihood-score experiments is related; this is its specific application to remove unidentifiable negative observation modes and improve Fourier conditioning. No new production API or general dense claim. This is a detour from the mixed-mesh plan: a cheaply fixable singularity obstruction was identified.

## Derivation

For backbone X=mu+R Z, Z standard Gaussian, condition on U=A'Z+E=0 with independent E~N(0,N=I-A'A), N PSD. The resulting performance covariance is R(I-AA')R'. Let P=I-11'/n. In a square invertible backbone choose unit q parallel to R^{-1}1 (diagonal R=diag(sd): q proportional to 1/sd). Write b=q'A, A_p=A-q b. Then

    PR A_p = PR A
    N_p = I-A_p'A_p = N+b'b >= N.

The contrast covariance is IDENTICAL, and N_p remains PSD. Since the entire Gaussian contrast law agrees, every race probability, including any specified tie-credit functional, agrees mathematically. This is not a ridge. Unlike a covariance approximation, even deterministic contrasts are unchanged. For arbitrary rectangular R the generalization is H = orthogonal projector onto row(PR); replace A by H A, and add A'(I-H)A to N. This gives the Loewner-maximal observation noise among all loadings B satisfying PR B=PR A: B=H A+K with columns of K in ker(PR), so B'B=A'H A+K'K. The prototype only implements a supplied common direction or square invertible root, not automatic rank decisions for rectangular roots. A block-diagonal root with a known common direction can use this preprocessing without changing its survival field grammar; multi-axis and hybrid end-to-end integration remain to be tested.

Fourier observation damping exp(-k'N k/2) is pointwise no weaker. If N was singular rank r-1 and the removed row has nonzero projection on ker(N), N_p is SPD. For rank one, the added observation variance is b^2. If |b|=1 the original downdate is pure common shock and can be removed entirely. In a singular two-sparse rank-one case a nonzero b makes the previous field's integrability obstruction disappear: the new observation noise is positive, the new covariance nonsingular, and the same contrast probabilities are priced. The singular case b=0 can have a permanent tie; conservatively refuse near-unit sparse centered input rather than rely on roundoff in a projection to create artificial noise. Diffuse centered modes get no damping improvement. A previously sparse loading can become dense, increasing special-function cost. General supplied-root solve can be ill-conditioned and is not certified; diagonal q uses min(sd)/sd for scaling.

For rank one, original signed_field.py quadrature/convergence and Gaussian omitted-frequency bound apply unchanged to the new parameters: erfc(K sqrt(N_p/2))/sqrt(N_p) when N_p>0, plus marginal t truncation. Finite K, pad, converging quadrature then increasing K,pad gives the exact race under the same conditions as original field, not a finite-error certificate. No structural or sampling error. The stronger damping bound is independent of the still-separate time-grid/quadrature errors. Finite precision and sparse centered tie handling remain limitations.

The contrast gauge is standard Gaussian choice identification, not claimed novel. Genz (1992), Numerical Computation of Multivariate Normal Probabilities, doi:10.1080/10618600.1992.10477010, publisher abstract checked; Fourier inversion literature already checked in prior iterations. Contribution here is the precise observation-noise monotonicity, extension to formerly refused non-tied singular input, and measured utility in the existing signed shared-field implementation.

## Reproducibility and results

New gauge_field.py, test_gauge_field.py. Standalone local NumPy 2.3.5/SciPy 1.17.0, single BLAS thread. Used equivalent arithmetic of signed_field.field/truncated_cf from fetched source (local copy omitted only validation unrelated to these valid fixtures and comments). Five standalone tests passed in 1.25s, not full CI. Independent four-runner zero-mean arcsine orthant for exact singular two-sparse a=(.8,.6,0,0), sd=(1,.8,1.3,.6): max absolute error 1.82e-14, all shares positive. Three-mode algebraic noise PSD/contrast check; common-only shock removal vs independent field, sign/permutation/translation/scale invariance; centered sparse permanent tie refusal. For the independent arcsine test the original signed_field refuses the input (zero noise, two nonzero a). No general permanent-tie API support.

Reproduce in research/shared_field_research:
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python -m pytest -q test_gauge_field.py
    python -u test_gauge_field.py 32 pair
    python -u test_gauge_field.py 32 sparse
    python -u test_gauge_field.py 32 common
    python -u test_gauge_field.py 32 centered
    python -u test_gauge_field.py 64 pair

Seed 382, sd uniform(.65,1.35), means N(0,.45^2), observation noise 1e-6 before gauge. Pair a=(.8,.6,0,...), sparse a proportional to (.8,.5,-.3,0,...), common a=.98*q+sqrt(1-.98^2)*v with v unit centered latent vector, centered sparse projected before normalizing. Heterogeneous scales, mixed signs in sparse/common/centered cases, min reference p around 1e-5 (n32) or 4.24e-5 (n64). Reference field uses gauge nt=512 nk=384 K=192, max difference from nt=384 nk=256 K=128 <=4.34e-13; not independent error certificate. All benchmark raw/gauge runs nt=128, all raw shares positive in these fixtures, no normalization. Compiled SciPy _qmvn randomized CBC lattice, greedy ordering, eight batches per winner, four seeds 0..3, normalized vector. Timings per full vector include setup/no amortization. Genz errors pooled over contestants and scrambles; SD pooled between scrambles.

case | observation noise after gauge | method nk/K or budget | seconds | RMSE vs refined | raw mass defect or Genz SD
32 pair | .06207 | raw 24/12 | .0418 | 6.65e-8 | +7.49e-7
 | | gauge 24/12 | .0477 | 1.40e-11 | +6.74e-13
 | | raw 64/32 | .0774 | 3.39e-6 | -2.62e-5
 | | gauge 64/32 | .1397 | 9.92e-14 | +6.00e-13
 | | Genz 2048 | .403 | 6.52e-5 | SD 6.54e-5
 | | Genz 8192 | .741 | 1.07e-5 | SD 9.87e-6
32 sparse | .03835 | raw 24/12 | .0184 | 6.78e-10 | +3.75e-9
 | | gauge 24/12 | .0281 | 3.75e-10 | +2.36e-13
 | | raw 64/32 | .0455 | 6.81e-7 | -3.62e-6
 | | gauge 64/32 | .0726 | 7.59e-14 | +1.31e-13
 | | Genz 2048 | .340 | 5.27e-5 | SD 5.34e-5
 | | Genz 8192 | .843 | 1.86e-5 | SD 1.72e-5
32 common | .96040 | raw 24/12 | .0282 | 3.96e-12 | +9.84e-11
 | | gauge 24/12 | .0244 | 7.47e-14 | +6.48e-14
 | | Genz 2048 | .294 | 5.46e-5 | SD 5.26e-5
 | | Genz 8192 | .767 | 1.28e-5 | SD 1.27e-5
32 centered | 1e-6 | raw 64/32 | .0631 | 3.56e-13 | +1.27e-13
 | | gauge 64/32 | .0662 | 3.56e-13 | +1.27e-13
 | | Genz 2048 | .306 | 6.55e-5 | SD 6.46e-5
 | | Genz 8192 | .766 | 1.77e-5 | SD 1.71e-5
64 pair | .03184 | raw 24/12 | .0247 | 1.49e-8 | +1.89e-7
 | | gauge 24/12 | .0487 | 6.07e-12 | +2.34e-11
 | | raw 64/32 | .0649 | 2.92e-8 | +3.78e-7
 | | gauge 64/32 | .1321 | 4.08e-14 | -1.71e-12
 | | Genz 2048 | 2.042 | 9.13e-5 | SD 9.08e-5

At 64, Genz 8192 did not finish within the run budget. Run-to-run timings across the commands vary; only within-fixture comparisons should be emphasized. Raw quadrature on the sparse cases is not monotonically accurate with increasing both cutoff and nodes, as oscillation and t-grid errors interact; this is not a proof that the raw method cannot attain high accuracy. Gauge often costs more per node but can greatly improve accuracy at fixed resolution, and recovers an exact formerly unsupported singular pair class. No high-budget independent n32/64 reference; Genz corroborates only to its sampling precision. Relative-error table not computed; min reference shares and absolute error suffice for these moderate tail examples. No high-rank dense experiment or general PSD convergence claim. No production dispatch changes.

Next: integrate the maximal contrast projection in the rank-two and hybrid signed-block prototypes (preserving explicitly supplied roots and observation scores), with a damping-driven cutoff choice and independent rare-share checks. If common direction absent or removed projection negligible, retain original grammar and tail bounds. Continue earlier mixed-mesh plan, and investigate automatic signed decompositions without sacrificing contrast accuracy.
