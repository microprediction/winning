# Randomized First Choice: the incumbent, and why it is an asset

**Summary: this is demand evidence, not a threat, and it should be cited as
such.** A first draft of these notes filed it as a prior-art risk. That was
wrong, and the correction matters enough to record.

## What it is

Huber, Orme & Miller (1999), "Dealing with Product Similarity in Conjoint
Simulations", Sawtooth Software Conference Proceedings; also Springer DOI
10.1007/978-3-540-71404-0_17. It has been the default market-simulation engine
in Sawtooth Software — the dominant commercial conjoint platform — ever since.

The specification, verbatim [LIT, quoted by a research agent that read the
primary text; verify before citing]:

    U_i = X_i (beta + E_A) + E_P
    Pr(i | S) = Pr(U_i >= U_j for all j in S)

with `E_A` attribute-level variability and `E_P` product-level variability.
The induced error covariance is

    X Sigma_{E_A} X' + sigma^2 I

i.e. **low rank plus diagonal, with the conjoint design matrix as the
loadings.** That is the factor probit model of this paper, evaluated by
brute-force Monte Carlo, shipped commercially for 27 years.

**VERIFIED (2026-08-24, from the original 1999 PDF on Joel Huber's Duke
faculty page).** E_A is a genuine shared random vector: the glossary reads
"E_A = Variability added to the part worths (SAME FOR ALL ALTERNATIVES)" and
"E_P = Variability added to product i (unique for each alternative)". So
Cov(U_i, U_j) = X_i Sigma_EA X_j' + delta_ij sigma_EP^2 -- low rank plus
diagonal. It IS the model class. Two qualifications, both favourable on
novelty: the loadings are FROZEN to the conjoint design matrix and only one or
two scalar variability levels are tuned (a far more restrictive member of the
class than a fitted factor model), and "For logit, EP has a Gumbel, while for
Probit, it has a Normal distribution" -- they knew both bases.

## Why this is an asset

They did not stumble into the model class; they chose it, and said so
[LIT, verbatim]:

> "numerous ways researchers have attempted to solve this problem, from nested
> logit to correlated error terms within probit... We will show that a simple
> first choice simulation with suitable variability added... provides a robust
> way"

So:

1. Practitioners with commercial stakes evaluated correlated-error probit and
   the alternatives, and **selected this model class**.
2. They were **forced to simulate it**, because exact evaluation was not
   available. The incumbent states the gap on our behalf.
3. There is an **installed base** already committed to the model, which is
   worth more than a hypothetical audience.

Prior art on a *model* is not prior art on an *algorithm*. The novelty claim
must therefore be about exact, fast, differentiable evaluation and inversion —
never about the model class. Anyone who reads this literature will know the
model class is theirs, so concede it first and clearly.

## The concrete improvement over the incumbent

RFC's variability is **tuned**. Huber, Orme & Miller grid-search the
attribute-level and product-level variance against holdout data until the
simulator reproduces observed choice shares. In this paper's language that is
a hand-tuned factor-to-idiosyncratic variance split.

With an exact forward share map *and* an exact Jacobian, that split can be
**fitted by maximum likelihood and differentiated through**, rather than
tuned. That is a specific, checkable improvement on the incumbent using
machinery that already exists here, and it is a better pitch than raw speed.

## The validation claim, and its scope

Sawtooth's own holdout validation [LIT, verbatim]:

> "within 10% of what one would get asking the same question again... There
> simply is not much room for further improvement"

Two qualifications, both of which have to be stated honestly and both of which
limit where this paper can claim value.

- **It is at N = 3–5 concepts per task and 6–38 in simulation scenarios.** At
  that size Monte Carlo is free and exact evaluation buys nothing. Any claim
  must be made where simulation actually breaks — assortment and retail
  problems at N = 10^3–10^4.
- **SETTLED (2026-08-24): it is MODEL fit to holdout shares, not simulator
  error.** The metric is "mean absolute error predicting holdout stimuli as a
  percent of the test-retest MAE for repeated choice sets", at N = 5
  alternatives, 352 respondents, against a 19% human test-retest noise floor.
  It says nothing about whether the Monte Carlo computes the model's own
  probabilities accurately. Better still: the paper NEVER STATES the number of
  draws and never mentions simulation error -- "Equation 2 is estimated by
  using a simulator to draw Ui's ... and then simply enumerating the
  probabilities" is the only procedural sentence. The incumbent has never
  quantified its own numerical error. That is the open flank: an exact
  evaluator can measure RFC's simulation error for the first time.

## The related inversion, which also exists

Sawtooth's "external effect adjustment" — Orme & Johnson (2006), *External
Effect Adjustments in Conjoint Analysis*, Sawtooth Software Research Paper
Series [LIT, verbatim]:

> "find an adjustment to each brand's part worths... so that the simulated
> shares become equal to the target shares... add those ratios or differences
> to each respondent's brand part worth and re-simulate"

That is share inversion: an ad hoc ratio fixed-point, no Jacobian, worked at
N = 6. Its own authors disown it as "fudge factors" whose "widespread practice"
they decline to encourage [LIT, verbatim].

So the inversion problem is real, is being solved badly, and the people
solving it badly are uncomfortable about it. Cite it as motivation.

## What would actually be fatal, and is still open

Someone *failing* to compute this is evidence for the work. Someone
*succeeding first* is the only real collision. The open item is:

**SETTLED (2026-08-24): Sohn is NOT a collision.** The published version is
Sohn, Kim, Kang (+ Kock, Bansal), "Scalable variational inference for
multinomial probit models under large choice sets and sample sizes",
Statistics and Computing, DOI 10.1007/s11222-025-10789-2 = arXiv 2507.10945
(the SSRN item is the working paper). Read in full. It ESTIMATES (a, Sigma)
from individual choice data -- "The parameters to be learned are Sigma and a,
while pairs (X_i, y_i) ... are observed"; a full-text grep for "market share",
"aggregate share", "invert" returns zero. "Regardless of the number of
alternatives" means Gumbel-Softmax straight-through gradients avoid sampling
high-dimensional truncated Gaussians; covariance is FULL (differenced, trace-
restricted), not factor; largest d = 20; stochastic VAE-style VI, not
deterministic. No overlap with share inversion at large N.

The useful by-product: it names Loaiza-Maya & Nibbering (2023) as
"state-of-the-art" for scalable MNP VI, with covariance "factorized as
B B^T + D^2" -- the low-rank-plus-diagonal class, already cited in
scalable-share-calibration.tex as loaiza2022/loaiza2023. Their benchmark:
d = 20, 1M observations, 982 minutes (vs 28 for Sohn's CVI). Position the
covariance parameterisation against LM-N, and the computation against both.

## Related

- `mcfadden-train-notes.md` — the other standing objection, that mixed logit
  already approximates any random utility model.
- The identification point that outranks both: N observed shares give N-1 free
  numbers, so the covariance must be supplied exogenously (Keane 1992). Every
  pitch must lead with "we invert mu GIVEN Sigma."
