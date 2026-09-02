# Research plans: the argmax-of-a-given-model applications
(2026-09-01, from Peter's application-scouting notes. Every plan leads
with its kill test; every external claim carries its verification
status.)

## The dividing line, stated once
The target applications are NOT "people make choices and probit might
fit better." They are settings where a Gaussian, survival, simulation
or physical model ALREADY EXISTS and the user needs the induced
distribution of its argmax or argmin -- both the identity I and the
magnitude M. There, factor logit does not approximate the quantity; it
silently changes it (Boltzmann exploration is not posterior probability
matching; a Gumbel delay law is not a timing yield). The engine's
native outputs cover both halves: the win/membership vector is I, and
the winning-value distribution G(x) = 1 - prod S_j(x) -- free on the
lattice -- is M.

---

## Track A: position against LITE (first external move; days)
LITE, "Efficiently Estimating Gaussian Probability of Maximality",
Menet-Huebotter-Kassraie-Krause, AISTATS 2025, arXiv:2501.13535, code
at github.com/lasgroup/LITE [both locator-verified; paper unread in
full]. A named literature on exactly the forward problem, with an
almost-linear APPROXIMATION motivated by existing methods being too
expensive -- and we ship an exact evaluator whose cost is also linear
per lattice point for the factor grammar.

Kill test: none needed conceptually (same problem); the empirical
question is accuracy-vs-time head to head. Run LITE's own experimental
setup from their repo: low-rank-plus-diagonal Gaussian vectors, N from
1e3 to 1e6; compare Monte Carlo, LITE, winning on PoM accuracy
(including small-probability entries), maximizer-entropy accuracy, and
their downstream bandit-control task. Deliverable: a benchmark section
(site + the order-statistics paper), possibly a short standalone note.
Package needs: none -- max-wins negation of the forward pass.

## Track B: SSTA criticality (flagship; kill test before anything)
See research/applications/circuit-timing.md (two papers locator-
verified). The named industrial target is literally the win
probability of a correlated race, the canonical delay model is the
factor grammar, and path covariance is the Gram of edge-incidence
vectors -- in-grammar by construction under the canonical variation
model.

KILL TEST FIRST (the notes' own instruction): topological correlation
from shared gates need not stay low-rank-plus-diagonal at real cutsets.
Take 1e4-1e5 near-critical paths at one endpoint group, build the true
path covariance from a variation model, run the rank/grammar ladder
(factor rank ladder, blocks, tree, factor-plus-residual) against Monte
Carlo criticalities. If accuracy plateaus before the computation is
attractive, abandon. Corpus candidate: EDA-Schema-V2 (7,776 OpenROAD
instances, 36M extracted timing paths) [U -- verify it exists and what
it carries before planning around it]. If the ladder passes:
criticality kernel -> gate aggregation -> one sizing loop, benchmarked
against Clark/tightness and localized sampling. Yield curves P(M <= t)
come free from the field.

## Track C: weakest-link failure localization (cleanest paper; weeks)
"A Bayesian Weakest-Link Framework for Joint Estimation of Material
Strength and Stress Profile", arXiv:2608.01261 [locator-verified;
unread in full]: the authors REPLACE the argmin zone-failure likelihood
with a Softmin (a Gumbel race) explicitly for differentiability, then
must fix its sharpness k for identifiability. The engine computes the
exact argmin likelihood with exact gradients under the actual flaw
distribution -- and the base families needed shipped this week
(exponential-power, skew-logistic, failure lump, student).

Kill test: data availability. Read the paper; determine whether the
crossarm data (200 rejected + 50 accepted, mapped knots, failure loads
and zones) is public or obtainable. If yes: reproduce with four
location likelihoods (their Softmin; Gaussian log-strength race;
Weibull/generalized-normal race; heteroskedastic or defect-mixture
race), score held-out zone log loss, load likelihood, recovered stress
profile, sensitivity to k, zone-probability calibration. The paper's
question: does the convenient Softmin distort the inferred stress
profile? Deliverable: a short methodological paper answering it.

## Track D: ranking & selection under common random numbers (+ Thompson)
The commercial vertical. CRN replications make the shared factors
literal (F_r IS the scenario), so estimated system performances are
Gaussian with low-rank-plus-diagonal covariance BY CONSTRUCTION, and
the R&S literature reportedly lacks the full correlated
probability-of-best vector [U: verify against the R&S literature and
the 2026 parallel-R&S package before claiming a gap]. Same machinery
serves finite Gaussian Thompson sampling (the policy IS the share
vector; linear-bandit posteriors give Sigma = X S X' + D natively) --
references to chase: ToSFiT, VAPOR [both U].

First experiment: 1e4-1e5 simulated inventory/queueing policies on
shared scenario streams; compare MC, LITE, Bonferroni/pairwise R&S,
winning; score PoM, probability of correct selection, expected
opportunity cost, and a budget-allocation loop driven by the Jacobian.
Deliverable: the downstream half of the Track A benchmark plus a
vertical demo. Package needs: a thompson_probabilities convenience
wrapper at most.

## Track E: model/prompt/agent selection on shared evals (best demo)
N models scored on the same R questions: S_ir = mu_i + v_i' F_r + eps,
so "probability this model is actually best" is a factor-race PoM, and
an agent-router removal is the removal counterfactual -- the exact
Harville/Benter story from the winning paper retold for routers
(softmax renormalization IS the Luce move the paper's own footnote
indicts). First experiment: a public per-item eval matrix (HELM-style),
fit the factor covariance, report P(best) with uncertainty, leaderboard
gap reality-check, and the removal counterfactual against softmax
renormalization. Deliverable: site demo page plus a section in the
order-statistics or applications paper. References to place: MODEL
SELECTOR, LLM-judge uncertainty work [U].

## Parked
- Probit output layer (Google HET/HET-XL parameterize Sigma(x) =
  V'V + diag d and then Monte-Carlo a tempered softmax -- the exact
  conceptual fit, but needs a batched GPU kernel and rank 1-4 at
  billions of observations; revisit after A/D/E establish the PoM
  positioning). [HET references U.]
- EV battery limiting-cell prognosis (Nature Energy fleet study [U]):
  right shape (rolling-horizon min-race, removal counterfactual =
  module replacement), needs an OEM/fleet data partner; hold a
  one-pager for partner conversations.
- Engineering first-failure maps: needs alternative-specific
  conditional CDF/hazard interface (a real API extension); after C
  proves the survival-side story.
- qPO: already ours; the notes rank it canonical but done.

## Demoted, per the notes, and agreed
Product choice, ballots, RLHF, routing-as-choice, classifier heads as
mere predictors: factor logit remains admissible there, so winning must
win on fit rather than by disqualification -- weaker ground. Keep the
preference_probit line (MoPLEx notes) as research, not positioning.

## Sequence
A (days, code exists both sides) -> C kill test (read paper, chase
data) -> D (a week, machinery exists) -> E (a week, data public) -> B
kill test (corpus verification, then the ladder) -> B flagship if the
ladder passes. F/G/engineering-hazards parked. LITE head-to-head is
the first external-facing number.
EOF
git add research/PLANS.md && git commit -q -m "$(printf 'Research plans: five tracks behind one dividing line\n\nThe argmax of an already-specified model is the target class; factor\nlogit is disqualified there rather than outcompeted. LITE (AISTATS\n2025, code public) and the weakest-link Softmin paper (2608.01261)\nboth locator-verified; kill tests lead every track; SSTA stays\nflagship pending its grammar-ladder kill test; parked and demoted\nlists recorded with reasons.\nCo-Authored-By: Claude Fable 5 <noreply@anthropic.com>')" && git push origin main 2>&1 | tail -1