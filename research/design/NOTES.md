# Active tournament design: D-optimality in the photo-finish geometry
(Peter, 2026-09-01)

Select the next field S from the population by

    S* = argmax_S [ logdet' { H + I_S(mu_hat) } - lambda C(S) ],

with H the accumulated information (plus prior precision) and I_S the
expected information of racing field S once, evaluated at the current
estimate (locally optimal design).

## What I_S is, exactly, in this machinery
Winner-only feedback: the outcome is categorical with probabilities
p(mu_S), so the Fisher information over the field's locations is

    I_S = sum_i p_i grad(log p_i) grad(log p_i)^T
        = sum_i J_i.^T J_i. / p_i,

J the photo-finish Jacobian -- one O(|S|^2 L) computation per candidate
from rows the engine already prices. Translation invariance puts 1 in
the null space, hence logdet' (pseudo-determinant on contrasts), and
I_S is PSD there. Full-order feedback has strictly larger information
(the order likelihood factorizes stagewise); winner-only is v1.

## Why greedy is principled, not just cheap
log det of a sum of PSD atoms is submodular in the atom set, so greedy
field construction (add the runner with the largest marginal logdet'
gain) inherits the standard near-optimality guarantees of submodular
maximization under cardinality constraints. [U: check the exact
statement and attribution -- Shamaiah-Banerjee-Vikalo 2010ish for
sensor selection; D-optimal design literature; active ranking /
dueling bandits (Jamieson-Nowak); TrueSkill matchmaking is the m = 2
draw-probability special case.]

## Measured (exp1_scheduler)
Fill in after the run: identification speed of D-optimal greedy vs
random fields vs a max-uncertainty heuristic, winner-only feedback,
moment-update posterior as the running estimator.

## Hardware races (Peter, 2026-09-01): CPU routing and friends
Consider applications where the "contest" is physical latency: which
core/server/route/cache answers first. Racing hedged requests IS a
min-race (send the job to S replicas, first response wins), so the
machinery prices completion-time contests directly, and the design
question above becomes REPLICA SELECTION: choose the set S whose
first-response distribution best trades latency against load, or --
the inference side -- learn per-node latency profiles from observed
winners only (which node answered first is often all the scheduler
logs). Concrete objects: p_i = P(node i answers first) with correlated
delays (shared switch/rack = block covariance; datacenter = tree);
tie densities = how often two replicas nearly tie, i.e. wasted
duplicate work priced exactly; the D-optimal scheduler = active
probing (which subsets to hedge to learn the fleet fastest). Adjacent
literature to sweep [U]: hedged requests / tail at scale (Dean-Barroso
2013), replica selection (C3, Prequal), work stealing, and the
straggler-mitigation line -- none of it, at a guess, prices correlated
first-response probabilities exactly or infers latency locations from
winner-only logs. Base families fit too: latencies are skewed
(lognormal-ish -> skew families on log-time) with failure lumps
(timeouts) -- the failure_base story transfers whole.

## Preferential Batch Bayesian Optimization (Peter, 2026-09-01)
Siivola, Dhaka, Riis Andersen, Gonzalez, Garcia Moreno, Vehtari,
"Preferential Batch Bayesian Optimization", arXiv:2003.11435 (2020)
[locator-verified via abstract page; unread in full]. Their setting:
BO with feedback only as rankings/winners over queried BATCHES, GP
surrogate, "custom likelihood" for the batch preference. Three exact
correspondences, making this a consumer of the engine rather than
adjacent work:
1. The batch-preferential likelihood is a correlated Gaussian race on
   the batch's GP marginals (mu_B, Sigma_B): the lattice prices it
   exactly with gradients, at batch sizes approximation methods cannot
   reach, and rankings use the exact order likelihood.
2. Their posterior update is update_winner_full / update_order_full on
   the batch block plus linear-Gaussian conditioning of the rest --
   the exact structure of exp1_scheduler's full-covariance arm. The
   tournament scheduler IS discrete PBBO; PBBO is the scheduler with a
   kernel prior and continuous designs.
3. Their batch acquisition meets the D-optimal criterion above: the
   photo-finish Laplacian of a candidate batch under the kernel
   covariance, logdet' on contrasts.
Candidate demo for the design note/paper: exact-likelihood PBBO versus
their approximation on their own synthetic suite -- measure both the
likelihood accuracy (ours vs their approximation vs MC) and regret.
Caveat to carry: nearby batch points are strongly correlated, so the
race is sharp -- the sharpness escalation and dense-covariance handling
are the relevant machinery, and batch smallness keeps it cheap.

Also: Gonzalez, Dai, Damianou, Lawrence, "Preferential Bayesian
Optimization", arXiv:1704.03651 (2017) [locator-verified; unread in
full] -- the pairwise predecessor: duels only, GP with a Bernoulli
likelihood on comparisons, and their headline finding is that
CORRELATION MODELLING is what drives the efficiency gain over discrete
dueling bandits. That finding is the motivation sentence for the exact
treatment: a duel is the two-runner race (closed-form probit of the
difference), PBBO's batch is the n-way race, and the winning engine is
what removes the reason the literature stopped at pairwise -- N-way
preference factors lacked tractable likelihoods and moments, which the
shared field supplies exactly. Lineage for the related-work section:
PBO 2017 (duels) -> PBBO 2020 (batches, approximate) -> exact n-way
(here).

## prefGP (Peter, 2026-09-01)
github.com/benavoli/prefGP -- Benavoli (Trinity) and Azzimonti (IDSIA),
JAX/PyTorch, the reference implementation for GP preference/choice
learning: nine likelihood models including Thurstonian ORDERING and
Plackett-Luce, plus choice-from-menu ("rational choice") models, with
the companion tutorial "A tutorial on learning from preferences and
choices with Gaussian Processes", arXiv:2403.11782 (2024)
[locator-verified; both unread in full]. Three uses here:
1. THE comparator for benchmark claims: their Thurstonian order and
   choice likelihoods under a correlated GP are computed by
   approximation (EP/variational/MC -- verify which); the engine's
   order_loglik and win probabilities are exact for the
   factor-conditional structure and near-exact through the dense fit.
   Same data, likelihood accuracy and wall clock, their inference
   against exact: the natural experiment.
2. The tutorial is the related-work map for the preference-learning
   positioning of the design note (model zoo: consistent, JND, probit
   error, Thurstone, PL, rational choice).
3. Their "choice from a menu" data IS winner-only race feedback, so
   the moment updates and the D-optimal scheduler apply to their
   problem class directly.
