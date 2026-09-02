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
