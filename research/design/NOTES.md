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
