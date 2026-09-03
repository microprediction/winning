# JEPA and latent-space prediction: where the race is, and isn't
(Peter's question, 2026-09-03. Honest dividing-line assessment; the
one empirical gate is out to an agent. References [U] until read.)

## What JEPA is, and what it is NOT (for us)
JEPA (I-JEPA, V-JEPA, LeCun's energy-based line) predicts in a
learned LATENT space: an encoder embeds the input, a predictor
predicts the embedding of a masked/future part from context, trained
by regression in embedding space with collapse prevention (EMA
target encoder, VICReg-style variance/covariance terms). The core
object is a POINT prediction (or its energy), fit by regression.
That is not a race, not an argmax over competitors, not an order
statistic. The training objective has no winning content, and
pretending otherwise would be a force-fit. Say so plainly.

## Where the race actually appears: the decision on top of the latent
The dividing line (as everywhere): does a Gaussian/latent MODEL
already exist and does the user need the argmax/argmin distribution
of it? Three places downstream of JEPA where the answer is yes:

1. SELECTION OVER A CANDIDATE SET UNDER LATENT UNCERTAINTY. Having
   predicted a latent, systems then SELECT: which of N codebook
   entries / retrieved memories / next tokens / candidate actions
   best matches the predicted latent. If the prediction carries a
   covariance (a distributional JEPA: mean + low-rank-plus-diagonal
   uncertainty) and the candidates are correlated in embedding
   space, the probability each candidate is the argmax of
   similarity-plus-noise is EXACTLY the factor-probit PoM vector.
   Same relationship as exact_pom: a Gaussian belief exists, winning
   gives the argmax distribution over it. This is the "token/snapshot
   selection at scale" bullet made concrete.

2. JEPA-PLANNER ROLLOUT SELECTION. JEPA world models plan by
   model-predictive control: roll out predicted latent trajectories
   under candidate action sequences and pick the argmin-cost one.
   The rollouts are CORRELATED latent trajectories (shared model,
   shared context), and choosing among them is a correlated
   min/max race -- and pruning which correlated rollouts to keep is
   precisely research/design/rollout_control.md (the n=2/n=3/n=16
   free-boundary work). The candidate rollouts in a JEPA planner ARE
   the correlated evolving trajectories that thread solves.

3. ENERGY-BASED ARGMIN. LeCun frames JEPA as an EBM: inference is
   find-the-argmin-energy target/action. If energy is quadratic
   (Gaussian) in the latent with structured covariance, the argmin
   over candidates is a race and its identity+magnitude are the
   engine's outputs.

## The honest catch
The fit is conditional on the JEPA producing a COVARIANCE, not just
a point. Most shipped JEPAs predict points (that is the design). A
distributional / uncertainty-aware JEPA with low-rank-plus-diagonal
latent covariance is where the machinery bites, and that is somewhat
aspirational -- it may be a small research contribution in itself
(predict the factor covariance, not just the mean) before winning
plugs in. So: real fit for distributional JEPA + downstream
selection; no fit for vanilla point-prediction JEPA training.

## The empirical gate (out to an agent)
Does anyone (a) do DISTRIBUTIONAL / probabilistic JEPA or latent
prediction with a STRUCTURED (low-rank-plus-diagonal / factor)
covariance, and (b) do argmax/best-of-N selection over correlated
latent candidates accounting for that covariance -- or is selection
done by plain top-k cosine similarity ignoring correlation (the
independence assumption again)? If (b) is done by independent
similarity, that is the same gap as everywhere: correlated argmax
computed as if independent.

## Verdict (provisional, pending the gate)
Not a flagship like first-failure -- the core JEPA objective is out
of scope and the fit needs a distributional JEPA that mostly does
not exist yet. But two genuine threads: the rollout-selection one
connects to work we have already built (rollout_control), and the
codebook/retrieval selection-under-correlated-uncertainty one is the
same argmax-of-a-given-model story as the whole applications program.
Park as research, revisit if the agent finds distributional latent
prediction is a live area.
