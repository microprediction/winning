# Equilibrium generalization: adjudicated framing (2026-09-01)

Distilled from Peter's dialogue + session corrections. Claims marked
[M] are measured in this repo; [V] verified at source; [U] unverified.

## The two-directions split (accepted)
(a) Project-activation equilibrium (metals clearing, m ~ 10-30 prices,
    on/off or small-R menus): flux is an exact-Jacobian and structural
    insight (PSD supply response, elasticity decomposition by marginal
    tonne), NOT an asymptotic speedup. AD + dense m x m Newton suffices
    there. The remaining flux value: distributional derivatives of hard
    regime boundaries that pointwise AD returns zero for, registered as
    custom JVPs -- "AD globally, flux locally at boundaries."
(b) Global allocation equilibrium (each unit of demand chooses among
    n large competing sources; endogenous shadow price per source):
    the true generalization. The relevant large number is the number of
    competing alternatives / endogenous node prices, not the number of
    commodities.

## Key correction: (b) is winning's inversion re-labeled
suppliers = contestants; capacity shares = target win probabilities;
shadow prices psi = abilities mu. The n=1e6 inversion [M: ~20 min
rank-1, ~1 min independent, bench.py invert] IS a single-buyer-type
capacity-clearing solve. Adjacent proven work: Kitagawa-Merigot-Thibert
JEMS 2019 [V] (damped Newton, global linear convergence, Hessian =
boundary mass = our tie densities); Levy-Mohayaee-von Hausegger MNRAS
2021 [V] (1e7 cells). The "134M points / 34 Newton iterations / timing
breakdown" benchmark quoted in the dialogue is [U] -- find the source
before citing.

## The new object: heterogeneous buyers
Q(psi) = sum_b w_b p^(b)(psi + delta_b), delta_b = buyer-type cost
offsets (freight, tariffs, quality). Q = grad_psi sum_b w_b W_b(psi),
W_b = E min_i(psi_i + delta_bi + eps_i): a sum of concave potentials.
Hessian = sum of tie-density Laplacians = a Laplacian. Theorem 1 of the
winning paper carries over verbatim: clearing prices exist, unique on
contrasts, iff capacities interior. Exposures generalize the winning
Jacobian to DQ = A' L A (congruence; winning is A = I).

## Solver lesson (exp23, [M])
Jh at O(nLQ) exists but Newton-Krylov lost to the diagonal iteration
(387s diverged vs 3.7s converged at n=200 naive; proper assembly
converges at ~60x wall clock, diverges on the constructed stall case).
The practical winning advantage is own-slopes FREE in the forward pass
-> diagonal damped log-residual iteration, no linear solve. This is
the hard-max analogue of Sinkhorn scaling (entropic case = literal
diagonal iteration). The AD comparison in the dialogue stands: JVP
through the O(nL) forward matches Jh asymptotically [plausible, not
measured], but neither changes the solver hierarchy.

## Functionals that ride the field (catalog for the paper)
- shares / allocation probabilities (forward pass)
- own-slopes (same pass, free) -> the equilibrium preconditioner
- welfare / expected winner cost: integrals of x against dG on the
  same lattice (WDZ: gradient of welfare = shares)
- full tie-density Jacobian as operator: cross-price substitution,
  DQ = A' L A, PSD/NSD by side of the market
- removal counterfactuals: entry/exit of a supplier (span-window
  integration + mass check, per the winning paper)
- dead-heat densities: mass at indifference boundaries = which
  boundaries carry the elasticity (attribution)
- inversion: capacities -> shadow prices (= market clearing)

## Honest scoping for the paper
- No published metals-equilibrium benchmark uses this machinery; we
  create the first, synthetic, seeded.
- For small-m commodity clearing we claim exactness and structure,
  not speed. Say so plainly (the winning paper's bounded-claim style).

## Atlas models and local times (Peter: include this connection)
The r11 intro teases it "for another day"; this paper is the day. The
static tie density w_ij = int phi_ij(t,t) P(field above t | tie) dt is
the one-shot analogue of the COLLISION LOCAL TIME of rank-based
diffusions: in Atlas models the semimartingale local time accumulated
at coincidences of adjacent RANKED particles is exactly what governs
how drift/occupation passes between ranks, as our photo-finish flux
governs how probability passes between winning regions. Make it
literal by dynamizing the race: performances as diffusions, winner =
argmin at horizon T; then dp_i/dmu_j at the horizon should be an
expected boundary local-time density at the leading-pair coincidence
{X_i = X_j = min}, and the Laplacian is the generator projected onto
rank space. What needs working out: the precise statement (a
Tanaka/occupation-times-formula derivation of eq. flux from the
dynamic model), and whether the shared-field lattice computes
horizon-T local-time densities at O(nL) the same way.
Candidate literature, ALL [U] until read at source: Fernholz,
Stochastic Portfolio Theory (2002); Banner-Fernholz-Karatzas, Atlas
models of equity markets (Ann. Appl. Probab. 2005); Ichiba-Karatzas
(-Banner-Fernholz-...) on collision local times and triple-collision
degeneracies; Pal-Pitman on stationary gaps of competing particles.
Portfolio-weights-as-choice-probabilities (the HRP remark in the
winning paper) is the bridge: market weights in SPT are literally a
choice-probability vector evolving by rank collisions.

## The third paper (Peter, 2026-09-01)
Working title: "Calculation of Order Statistics at Scale". Scope: top-k
/ bottom-k membership probabilities, rank distributions, order-statistic
CDFs (the shared count DP is the cache), and their FLUX derivatives --
the rank-boundary Jacobian eq (topk) of the Atlas note, i.e. the
leakage adjustment. The winning paper is k=1 throughout; this paper is
general k. Implementation begins at winning/factor/topk.py; measured
results go here first.

## Base-density claim, bounded (bandits exp24, final, 2026-09-01)
If cited, cite the committed harness (bandits/experiments/
exp24_select_the_tail.py), and cite the NARROW version: the base
matters mostly through SKEW. F1 across seven bases spans 0.45 nats but
0.42 of that is gumbel alone; the four symmetric families span 0.026.
Two caveats travel with it: (1) validation selection over bases fails
at ~100-race samples (normal ranked first on validation, second-to-last
on test); (2) adding candidates degrades selection monotonically. The
honest motivation sentence: any-smooth-base support lets a model avoid
one specific expensive error (wrong skew) that a Gaussian-only engine
cannot express -- NOT "tune the tail weight". Their earlier 0.088-0.119
figure must be paired with the 0.026 symmetric-family span if ever
quoted.
