# External audit and framing notes (August 2026)

Two external assessments Peter passed in, stashed verbatim in substance
with the actionable items pulled out at the top. Audit was taken at
commit `1aec3bb` (the high-rank tensor fix).

## Actionable items

1. **The site overstates demonstrated scale.** winning.microprediction.org
   says the method scales to "millions of variables"; the paper
   demonstrates 10^4 alternatives at rank 2. Replace with:
   *"Demonstrated at 10^4 alternatives; linear in N at fixed modest
   rank."* Conceivable is not demonstrated.
2. **CI is red on Ruff** even though functional tests pass across
   Linux/macOS/Windows and Python 3.10-3.13.
3. **Abstract undersells the object.** Lead with counterfactual
   selection, not share inversion (see framing note below).
4. **Framing: the repository should lead with "counterfactual choice
   under support changes", not horses.** A high-level API such as
   `CorrelatedChoice.fit(shares, loadings, idiosyncratic_variance)` /
   `.restrict(survivors)` / `.jacobian_vector_product(direction)` would
   make the general contribution legible. Racing stays as derivation and
   heritage, not primary identity.
5. **The deciding experiment** is a preregistered cross-menu
   counterfactual benchmark on real support changes (AI routing masks
   are the cleanest first domain because availability can be
   randomized), with (V, D) estimated from training information and
   FROZEN, mu calibrated on full-menu shares, and held-out restricted
   menus predicted. Primary metric: fraction of released probability
   assigned to the wrong survivors,
   TV(q_pred, q_obs) / sum_{i in R} p_i. Baselines: proportional
   softmax renormalization, nested logit, mixed logit, independent
   probit, sampled/variational MNP, a neural choice model, embedding
   heuristics.

## Assessment 1: where the contribution sits

**The novelty claim that survives audit.** None of the ingredients is
new (random utility, MNP, factor covariance for MNP, social-surplus
gradient identity, convex demand inversion, identification up to an
additive constant, leave-one-out products, conditional factor
integration, deletion counterfactuals). The defensible contribution is
the COMBINATION: factor conditioning + all-N computation + derivatives
+ inversion + implementation at N = 10^4. A synthesis-and-scale
breakthrough, not a new choice model -- which is the right claim, and
does not diminish it: much of applied mathematics advances when an old
model crosses from "conceptually attractive but computationally
unusable" to "routine behind an API".

**Why it could matter beyond racing: prediction -> intervention.**
Prediction asks which alternative wins under the current system; almost
every operational decision asks what happens after the system changes
(stockout, filtered search result, agent unavailable on cost or
permissions or outage, prohibited portfolio constituent, contraindicated
treatment, ruled-out candidate, supplier exit, masked token, model
removed from an ensemble). Proportional renormalization is immediate,
but it is a substantive claim about substitution. The factor-race
transform permits the same immediate support restriction while
preserving a specified low-dimensional dependence structure. Measured
in the paper's synthetic deletion experiments: plain logit misallocated
~22.9-30.5% of released mass across deletion-size strata versus
~4.5-11.6% for factor probit with oracle loadings; correctly specified,
~25% versus ~1% for large deletions.

**Strongest prospective application: resilient AI routing.** Hundreds
or thousands of models, tools, databases and specialist agents whose
performance is correlated through a modest number of task dimensions.
A router must keep working as availability changes. A softmax router
renormalizes surviving scores; a factor race reroutes toward genuine
substitutes. Qualification: valid only when latent scores exist BEFORE
the menu restriction -- if showing a different menu changes how agents
are evaluated, the fixed-race counterfactual is incomplete.

**Portfolio application** (benchmark weights -> latent abilities ->
correlated race -> repaired weights) is structurally legitimate,
especially after HRP or Schur recursion where cross-block dependence
was compressed, but ranks behind routing and assortment substitution as
a validation domain: there "probability of being selected" is literally
the object, whereas portfolio weights are choice probabilities only
instrumentally.

**Where the evidence is not yet strong enough.**
- *The covariance geometry is supplied.* For any fixed admissible
  (V, D), every interior share vector is matched exactly by some
  utility vector, so a perfect same-menu fit is NO evidence the
  covariance is right. The engine is an excellent inner solver; a
  complete application still needs a credible outer procedure for
  (V, D) from repeated menus, covariates, features, micro-choices or
  independent performance data.
- *The substitution evidence is synthetic*, with oracle loadings.
- *Scalability is in N, not in rank.* Where the choice-relevant
  covariance admits no good low-rank-plus-diagonal approximation,
  direct simulation remains appropriate.
- *The shipped iteration has local, not global, convergence theory.*
- *Zero and tiny shares need statistical treatment* (pseudocounts,
  regularization, censoring), being outside the interior-simplex
  theorem.

**The ecosystem pattern.** skaters transports a process into
standardized predictive coordinates and back; inventory transports an
imbalanced control problem to a balanced one; Schur allocation restores
information discarded across blocks; winning conditions on factors to
turn correlated alternatives independent, computes in the easy space
and reconstructs; mechanisms moves between scoring rules and markets by
convex duality. The recurring move: hard problem -> structure-preserving
transform -> easy problem -> exact/controlled inverse -> answer in the
original space. "Decompose without pretending the discarded dependence
was irrelevant."

Bottom line offered: Schur complementary allocation has contributed
more already (it is in skfolio with attribution); the open AI network is
the more expansive vision; but the inverse-race transform is the piece
most likely to look, in retrospect, like the underappreciated
invention -- PROVIDED the next stage demonstrates it on real, held-out
support changes rather than oracle factor models.

## Assessment 2: the paper undersells the package

The paper presents winning as a fast numerical solver for share
inversion in factor MNP. The package is closer to a general engine for
inference, selection, substitution and counterfactual routing in large
correlated races.

Omitted from the abstract: the Jacobian is a weighted photo-finish
graph Laplacian; that graph measures substitution; the same field
produces removal counterfactuals; hard and soft selection are one
engine; Gumbel gives softmax exactly; factors give correlated softmax;
arbitrary bases permit non-Gaussian races; forward and inverse maps
form a general choice/routing primitive.

**The photo-finish graph as an ecology map**: dense cluster = redundant
contributors; strong edge = close substitutes; weak edge = distinct
niches; removal response = actual dependence on a contributor; bridge
node = contributor connecting predictive regimes. Potentially more
useful than a one-dimensional leaderboard.

The applications section reads as a list of things that can be
interpreted as races; it should identify the common operation --
*selecting among a large population of correlated contributors and
understanding the counterfactual consequences of that selection.*

"This paper's sole point is that probit's computational barrier is not
real for this covariance class" is too restrictive; there are three
points: (i) factor-MNP computation at large N is practical; (ii) a
shared field encodes shares, derivatives and removal counterfactuals
simultaneously; (iii) correlated selection has an explicit graph
geometry -- possibly the most generative.

**But do not turn this paper into the grand theory.** Narrowness is
partly a virtue: "we removed a longstanding computational barrier in
factor MNP" is recognizable and publishable. Split instead into:
the paper (theorem, algorithm, numerics, the barrier); the package (the
broader race/selection engine); and a companion conceptual paper
(correlated contributor selection and routing in modular machine
intelligence).

**Concrete abstract edit proposed.** Open with: *"We give a scalable
engine for inference and counterfactual selection in large correlated
races, specializing it to multinomial probit models with supplied
low-rank-plus-diagonal covariance."* Add: *"The exact share Jacobian is
a weighted graph Laplacian whose edge weights are photo-finish
densities, giving the geometry of substitution between alternatives.
The shared field also yields all single-removal counterfactuals without
repeating the underlying integration."* Close with: *"These objects
support not only discrete-choice estimation but also selection and
routing among large correlated populations of models, experts, or
autonomous agents."*

**Caveat to keep.** winning is not yet a complete contributor-selection
system. Where full realized log scores are available for every
contributor, direct exponential weighting is simpler and uses more
information. winning's distinctive value arises when only selections or
winner frequencies are observed; contributors face different menus or
task populations; correlations materially affect substitution;
counterfactual removal matters; or routing must retain exploration.
Real routing would also require learning task-dependent abilities and
covariance from repeated tasks -- a research program, not a delivered
feature.
