# Cavity calculus for extremal portfolios
(Peter's notes, 2026-09-02. One coherent project: a single
conditionally independent shared field yields extremal value,
probabilities of optimality, deletion values, Shapley/Banzhaf
allocations, robustness margins, and top-K probabilities. All
literature pointers [U] until read at source.)

## 1. Cavity-Shapley for extremal portfolios (the strongest
## paper-sized idea)
Coalition value v(A) = E max_{i in A} Y_i (batch of experiments,
algorithm portfolio, suppliers, forecast pool). Independent
availability B_i ~ Bernoulli(a_i) gives the multilinear extension

  V(a) = E max_i B_i Y_i
       = E_z int_0^inf [1 - prod_j {1 - a_j Fbar_{j|z}(y)}] dy,

and its partial derivative is ANOTHER shared-field cavity:

  dV/da_i = E_z int Fbar_{i|z}(y) H_{z,a}(y)
            / (1 - a_i Fbar_{i|z}(y)) dy,
  H_{z,a}(y) = prod_j {1 - a_j Fbar_{j|z}(y)}.

Three quantities drop out: deletion value V(1) - V(1 - e_i);
Banzhaf = dV at a = 1/2; SHAPLEY by Owen's (1972) diagonal formula
phi_i = int_0^1 dV/da_i (t 1) dt -- so ALL N Shapley values need one
extra one-dimensional quadrature over t, not 2^N coalitions:

  phi_i = E_z int_0^inf int_0^1 Fbar_{i|z}(y)
          prod_{j != i} {1 - t Fbar_{j|z}(y)} dt dy.

Peter verified against exhaustive enumeration, five heterogeneous
Gaussian players: max error 1.6e-15. Owen's diagonal-of-multilinear
trick is old; the factor-correlated extremal specialization and the
all-player winning-style computation appear substantially less
explored (novelty gate: literature check before claims).

Applications: batch Bayesian optimization (Y_i = (b - X_i)_+ makes
v(A) the batch expected improvement -- qEI, grand-coalition
marginals, and Shapley attribution distinguishing frequently-optimal
from irreplaceable; multipoint-EI formulas still carry serious cost
at batch size); chemical-library screening (qPO prefilters to 10k
candidates because posterior sampling needs a big factorization --
its antibiotic example starts at 39,312 compounds; low-rank-plus-
diagonal posterior would attribute value over the unfiltered
library; connects to research/qpo/); algorithm/forecast portfolios
(pay contributors by expected contribution to the best result,
discounting near-duplicates -- the microprediction contributor-
selection connection); supplier redundancy under common shocks
(deletion = immediate irreplaceability, Shapley = fair average
contribution).

CONNECTION ALREADY IN THE REPO: V(1_S) = E max over S is exactly the
group objective select_race_group optimizes (research/selection/) --
the availability field prod_j [1 - a_j F_j(x)] at binary a IS that
experiment's miss product. Shapley here is attribution for the same
objective the optimizer maximizes; the submodularity proof carries
over (multilinear extension of a submodular-like value).

## 2. Winner-margin spectrum (flux becomes robustness)
m_i(delta) = P(X_i + delta < min_{j!=i} X_j), the shifted cavity
m_i(delta) = E_z int f_{i|z} G_z(x+delta)/S_{i|z}(x+delta) dx.
Identities: m_i(0) = p_i; sum_i m_i(delta) = P(X_(2) - X_(1) >
delta); m_i'(0) = -sum_j w_ij with w the photo-finish conductances,
so TOTAL CIRCUIT CONDUCTANCE = the infinitesimal hazard of an
ambiguous result, and node degree = how fast a candidate's
robust-win probability dies as the required margin grows. Peter
verified the second identity numerically.

STATUS: the computation is already implemented and MC-verified in
research/orderstats/exp4_spacings (winner margin with per-winner
decomposition, n to 2e4); what is new here is the DERIVATIVE
identity tying it to tie_densities and the robustness framing.
Applications: gap-based stopping in ranking and selection
(indifference-zone/BIZ connection -- stop on P(gap > delta) >=
1-alpha, extending the exact_pom stopping experiment); catalyst
selectivity (lowest uncertain activation barrier, gap to next
barrier = robustness, edges = the competing pathways); reliability
(gap = interval between first and second failures, w_ij = pairs at
risk of near-simultaneous failure); abstention rules for
classification/routing (wins-by-at-least-delta).

## 3. The rest of the table
- Top-K rank probabilities: SHIPPED (winning/factor/topk.py,
  O(QNLK)); the same polynomial gives correlated k-out-of-n
  reliability and second-order auction statistics.
- Second-price/second-best: same polynomial field keeping degrees 0
  and 1 -- auction revenue, second-lowest bid, supplier-removal
  costs. Concrete, unbuilt.
- Random availability field prod_j [1 - a_j F_j]: supplier outages,
  solver downtime, expert dropout; doubles as the multilinear
  extension above. Strong.
- Effective resistance R_ij = (e_i - e_j)' L^+ (e_i - e_j):
  substitution clustering, inversion conditioning, active pairwise
  measurement -- adjacent choice-design/Laplacian literature exists.
- Transform cavity for sums (divide one factor out of a
  characteristic-function product): all-removal aggregate-loss;
  crowded, actuarial prior art exploits generating-function reuse.
- Matrix inverse cavity (Schur/rank-one downdates): established.

## API plan (issue to file with the first implementation)
extremal_value, extremal_deletion_values, extremal_shapley_values,
margin_probabilities(delta), rank_probabilities(k) [last one ships
already]. Decisive experiment: a factor-Gaussian candidate set with
correlated duplicates and rare-upside specialists; show PoM,
deletion value, and Shapley value produce radically different
rankings, each answering a different operational question. Frame as
an EXTENSION of the exact-PoM manuscript (papers/exact_pom), not a
competitor.

## Standing request
Track cavity-method prior art (Owen/multilinear-extension lineage,
extremal-value attribution, availability-field constructions) before
any novelty claim -- same discipline as the rollout-pruning tracker
in research/design/rollout_control.md.

## exp1_shapley: the attribution twin, measured (2026-09-02)
Owen-diagonal Shapley matches exhaustive 2^5 enumeration to 1.7e-16
(efficiency to 4e-16; value and deletion match 2M-draw MC to four
decimals). The decisive experiment delivers: with five
near-duplicates and one rare-upside specialist, PoM ranks the
duplicate block first collectively (0.30) and the specialist 0.20;
deletion gives the duplicates 0.008 EACH and the specialist 0.28;
Shapley pays duplicates 0.127 each and the specialist 0.37. Three
rankings, three questions. The pairing runs end to end: greedy entry
selects two of the five duplicates + the specialist; within-group
Shapley splits the duplicate pair equally (0.356 each, symmetry
emerging numerically) and pays the specialist most (0.483).
RESOLVED (same day): composite geometric t-panels -- [0, 1/(2Neff)]
doubling to 1, eight Gauss-Legendre nodes per panel -- bring the
n = 2000 efficiency error from 2.5e-2 to 2.1e-12 at 61s for all 2000
Shapley values, with n = 5 still at machine precision. Two failed
attempts recorded in git history for the avoidance of repetition: a
single exponential warp (wrong direction, worse) and the
worst-case-uniform substitution t = 1-(1-u)^{1/N} (singular Jacobian
at u = 1 poisons smooth regions). The layer scale varies with y, so
no single change of variables serves; panels do. Deletions are 0.6s
at n = 2000.

## Pass@k is an extremal portfolio: TailSFT (noted 2026-09-02)
TailSFT (arXiv:2608.25756, Malladi-Jelassi-Foster-Ash-Krishnamurthy,
Aug 2026 [U beyond abstract]): supervised fine-tuning that filters
out already-fit sequences, concentrating learning on the
UNDER-MODELED TAIL of the data distribution; up to 17 percent
absolute pass@16 gains on OLMo-3 7B math/coding, up to 4 percent
pass@1 after GRPO initialized from TailSFT checkpoints, plus a
diagnostic for when it helps.

Why it is up our alley: pass@k = P(at least one of k correlated
samples succeeds) = the extremal value of a sample portfolio -- the
availability-field object of this note with the samples as
candidates and the model's own correlation as the factor structure.
The duplicates-versus-specialist experiment (exp1_shapley) is the
mechanism in miniature: mode mass is the near-duplicate block (high
pass@1 credit, replaceable draws), tail mass is the rare-upside
specialist (what actually moves E[max] and pass@k). TailSFT is
empirical post-training confirmation that shifting mass into the
tail buys the extremal objective -- training toward the Shapley-
valuable region rather than the PoM-heavy one. Connects to:
research/design/rollout_control.md (best-of-N as a race; killing
duplicates), the posttraining Luce-vs-probit thread (pass@k
optimization is E[max] under correlated probit, not independent
Luce), and papers/exact_pom (the vector that prices which samples
plausibly win). Candidate question for the tracker: does a
factor-probit model of per-sample success predict WHICH prompts
TailSFT helps -- their diagnostic, derived rather than fitted?

## TailSFT gate check (agent, 2026-09-02): PARTIAL, favorably
No code, checkpoints, or eval logs released (absence confirmed on
the abs and HTML pages; GitHub/HF searches empty). BUT their
diagnostic does not need their model: rho_16 = L/G, coverage lost
vs gained by STANDARD SFT relative to the BASE model on the
base-reachable set (base pass@16 in [0.05, 0.95], pass@16 estimated
from pass@1 by 1-(1-p)^16); decision rule rho_16 > 1. Inputs = base
+ standard-SFT per-prompt pass rates only -- both computable from
public AI2 checkpoints (allenai/Olmo-3-1025-7B and
Olmo-3-7B-Think-SFT, Apache-2.0; no hosted API for the base, so
self-hosted vLLM or Apple-silicon inference; AI2 ships no per-sample
generative logs).

Reshaped experiment, feasible with public artifacts: generate 16+
samples/prompt for base and Think-SFT on their eval suite, fit the
factor-probit success model, compute the tail-value decomposition,
and test whether it REPRODUCES rho_16's dataset-benchmark ranking
against the paper's printed per-pair coverage gains. Full per-prompt
validation against actual TailSFT behavior requires reimplementing
the recipe or author contact -- gated on a GPU/effort decision.

Fallback runnable TODAY at zero generation cost: correlated probit
vs beta-binomial independence for pass@k prediction on
CL-From-Nothing RLVE Pass8-Rollouts (released per-sample records,
k=8, 9k prompts; HF). Turnkey regeneration pair if more is wanted:
LeapLabTHU/limit-of-RLVR pins Qwen2.5-Math-7B base/RL pairs with
eval code, logs regenerated not shipped.

## exp2_passk measured (2026-09-02): the plug-in extrapolation
## inside TailSFT's diagnostic costs 13 points at k=8
Released Pass8-Rollouts (9k prompts x 8 scored samples, Qwen3-4B-
Thinking, 18 RLVE environments; 3 GB streamed down to a 41 KB
outcome table). Train on samples 0-3, score on 4-7:
- Per-prompt held-out pass@4 (a single Bernoulli): plug-in
  independence 1-(1-s'/4)^4 scores log loss 3.37 (it says
  "impossible" for every zero-success prompt that then succeeds);
  hierarchical probit-normal 0.4825; beta-binomial 0.4830. SEVEN
  TIMES better log loss, 20 percent better Brier, for two fitted
  parameters.
- Aggregate extrapolation from the half-sample to k=8: truth 0.682;
  plug-in 0.549 (13.3 points under); probit 0.663; betabin 0.661.
  At k=4: plug-in 7.5 points under, probit within 0.9.
- Family verdict: probit-normal and beta-binomial are a statistical
  tie here; the probit costs nothing and speaks the engine's
  language.
Caveats stated plainly: the bias magnitude depends on the per-prompt
sample count (4 here; more samples shrink it), and this is Qwen3-4B
on RLVE, not their OLMo math/code setup -- transfer plausible,
unshown. The implication for the derived-diagnostic project: the
f_k(p) = 1-(1-p)^k step inside rho_16 is exactly the plug-in
predictor measured here, so a posterior-predictive version of their
own diagnostic is the natural first upgrade, before any Shapley
machinery enters.
