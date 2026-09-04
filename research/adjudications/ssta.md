# Adjudication: SSTA criticality (Track B)
(Agent report, 2026-09-01. Verdict: PURSUE, one experiment deep.)

## Incumbents, errors, costs
Mogal-Qian-Sapatnekar-Bazargan, TCAD 2009 [read in full,
people.ece.umn.edu/users/sachin/jnl/tcad09hm.pdf]. Criticality
T_i = Pr(e_i >= max of other cutset edges) -- a win probability over
correlated Gaussians. Delay model (their Eq. 2): e_i = mu_i +
sum_j a_ij p_j + zeta_i r_i -- k PCA factors from a spatial grid
(Chang-Sapatnekar) + global inter-die + independent per-edge term:
the factor grammar. Headline indictment of the analytic incumbent:
Clark's MAX (also under Visweswariah tightness probabilities) gives
criticality errors EXCEEDING 50% (Table I: 56.7% toy; Table III:
48-60% max on ISCAS89). Their fix: clustering-based cutset pruning
(5680 -> 15 edges on s38584) + localized Monte Carlo (N=1000),
~2-16% (avg ~5%) error, 0.01-0.25s on 20k-gate circuits. So the
accurate incumbent is itself sampling; the analytic one errs by
half. Cutsets: raw eta = 451-10742; post-pruning 2-66 dominant.

Mishagli-Koskin-Blokhina arXiv:2401.03559 [read in full]:
perturbative Gumbel corrections, valid only for weak correlation
(blows up at rho ~ 0.8, Fig. 7); max distribution only, no
criticality; leaves "covariance topology on realistic circuits"
open. Visweswariah 2004 not fetched; corroborated via Mogal refs.

## Leverage
Analytic incumbent >50% error; MC incumbent ~5% at 1000 samples per
cutset; winning exact, O(nLQ), cutsets only 1e2-1e4. Vs localized MC
the pitch is exactness + Jacobians (sizing gradients; cf. INSTA's
autodiff of approximations) + the free circuit-delay distribution.

## Kill risks
1. THE RANK QUESTION (primary): rank-k + diagonal holds at edge
   level but fails at path level -- per-gate random terms are shared
   across paths through the same gate (residual = A diag(zeta^2) A',
   A the path-gate incidence, not diagonal). Mogal Sec. VI-C:
   ignoring it caused errors to 60%. NO published measurement of
   path-covariance rank found -- a genuine gap. If near-critical
   bundles share hundreds of gates, effective factor count may
   exceed quadrature reach unless tree/block captures reconvergence.
2. Industry relevance: SSTA is live (NVIDIA INSTA won DAC 2025 Best
   Paper: differentiable STA, 15M pins, 3nm, open-sourced), but
   signoff runs POCV/LVF per-arc sigmas; criticality is a
   research/optimization-loop object, not a signoff bottleneck.
3. Non-Gaussian arcs at advanced nodes (LVF moments).

## Corpus
EDA-Schema-V2 VERIFIED: arXiv:2605.06952, 18 IWLS'05 circuits, 7776
instances, 36M OpenSTA timing paths, SkyWater130/Nangate45/IHP130/
ASAP7, CC BY-NC-SA, github.com/drexel-ice/EDA-schema. It contains
DETERMINISTIC STA only -- a process-variation model (Chang-
Sapatnekar grid + per-gate random) must be added on top.

## Decisive experiment
Extract near-critical path bundles from EDA-Schema-V2, build the
exact incidence-Gram covariance under a declared variation model,
measure (a) effective rank of the residual after k spatial factors,
(b) winning-exact criticality vs 1e5-sample MC vs Clark. Kill if
effective residual rank routinely exceeds interactive-cost grammar.

## Positioning
"Where SSTA's analytic criticality errs by up to 60% and its
accurate mode is Monte Carlo, winning computes the same
criticalities, the circuit-delay distribution, and exact sizing
Jacobians in one closed-form lattice pass over the industry's own
canonical delay model."

## Not verified
Visweswariah 2004 text; production PC count k; EDA-Schema-V2 hosting
links; any published path-covariance rank measurement (none found).
