# Response to the third review

We are grateful for a report that is unusually specific and, on the mathematics, unusually
useful. The independent recomputation of the reverse-hazard identity, the Gumbel cumulants, the
location-scale invariance and the concentrated-share example all agree with ours, and the
deterministic small-$\varepsilon$ experiment showing the local exponent approaching three is a
better piece of evidence for the cubic order than anything we had. Our own browser check suite
reaches the same conclusion by a different route, and we will cite the agreement.

Below we separate what we accept, what we contest, and three places where the report describes a
manuscript we did not write. We ask for more care on the last of these, because in one case the
imprecision produced a recommendation that would introduce an error into the paper.

## Accepted, and already changed

1. **The second-place exercise was cited as a check against truth in Scope and in the
   Conclusion, while the section presenting it says neither rule there is the ordering law of its
   model.** This was the most valuable finding in the report. It was a live contradiction and it
   was ours, introduced when the two diagnostics were first named together. Both passages now
   check against truth only in the after-stimulus arms, where an observed restricted probability
   exists, and state why the second-place table is not a second such check.

2. **The excess is a centred diagnostic, not a de-biased structural effect.** The protocol
   sentence saying every gain is "reported net of" the null did invite the stronger reading, and
   the report is right that no deconvolution is being performed. The sentence now says the excess
   is centred on that particular fitted null and that under a non-Luce population there is no
   reason for the null's bias to match the alternative's.

3. **"The covariance relevant to utility differences must be spherical" was wrong.** Independent
   equal-variance errors give $\operatorname{Cov}(D) = \sigma^2(I + \mathbf{1}\mathbf{1}^{\top})$
   in reference-difference coordinates, which is not diagonal; isotropy holds on an orthonormal
   contrast basis. The paper now states the algebra rather than the slogan.

4. **"Correlated probit handles near-substitutes exactly" was too strong.** It can represent
   near-substitution through a positive covariance. It does not thereby represent
   menu-dependent attention or the dimensional retuning that Getty et al. report, which is a
   boundary condition of ours and not a covariance.

5. **"Any standardized noise shape gives such a map" needed a regularity condition**, now stated
   as invertibility of the share map up to a common translation.

6. **"No map from old shares to new ones can be right" is too absolute.** We adopt the report's
   reformulation: no menu-invariant transport rule is identified or structurally justified from
   the old shares once the intervention changes latent utilities, discriminability or the
   measurement process.

7. **Wikipedia is not an impression-level exclusive-choice table.** The release is monthly
   aggregate (referrer, resource) transition counts with a censoring threshold. Describing it as
   a reader picking one outbound link exclusively overstates what the file contains, and the row
   will be recast as transport of conditional transition shares.

8. **The row count is descriptive.** The abstract now says so, and we accept that a study-level
   table should be primary, with the forced-choice subgroups shown as nested within their
   experiments rather than as separate rows of equal standing.

## Contested

**The truncation argument does not work, and the recommended replacement text propagates the
error it is meant to fix.** The report's premise is correct and trivial:
$P(X=i \mid X\in T) = p_i / \sum_{j\in T} p_j$ by the definition of conditional probability. The
inference drawn from it is not. That identity concerns conditioning on an event inside a fixed
probability space. The operation this paper is about is an intervention on the mechanism that
generates the support: a decoder confined to a sub-vocabulary, a field with scratchings, a
response set the experimenter disallows. In those cases the old experiment's conditional
probability is not an answer to a question about the new experiment, and the two maps are the
candidates. The report's own suggested wording, that the problem "should not be conflated with
ordinary probabilistic conditioning or truncation, for which proportional renormalisation follows
directly from conditional probability", reads as though truncation were always conditioning. That
is the very conflation the report elsewhere and rightly warns against, and Section 3 of the
manuscript already draws the distinction with the Monty Hall case. We have kept the example and
said which of its two senses we mean, rather than deleting it.

**The blanket demand for $9{,}999$ replicates is not costed.** We accept the direction and the
reasoning: $B=200$ gives a Monte Carlo standard error of about $0.015$ at a true tail near
$0.05$, which is large against a $0.05$ boundary, and $B=60$ on the news row supports a floor and
not an estimate. We will raise $B$, report exceedance counts alongside adjusted tails, and match
the simulation unit to the actual clustering. But a uniform $9{,}999$ across the ranking
collections means rerunning a pipeline that already evaluates $1{,}013$ subsets across five folds
per replicate, and the report does not say what that buys. Our proposal is to raise $B$ until the
Monte Carlo interval excludes the $0.05$ boundary for every row whose verdict could turn on it,
to state $B$ per row, and to leave coarse any row whose tail is not load-bearing. If the report
believes a specific row needs more than that, we will do that row exhaustively.

**One point of agreement worth making explicit, since it strengthens rather than weakens the
paper.** The report observes that under an exact Luce population with unlimited calibration data
the Gaussian-minus-Luce gain must be negative, so the null median mixes finite-sample shrinkage
with Gaussian misspecification. That is right, and it is what our null medians show: they are
negative wherever the shares rest on more than about two thousand observations. We will say so
in those terms.

## Three descriptions of a manuscript we did not write

We ask for more care here, and we say why it matters rather than merely noting it.

1. **The report states that the manuscript places the code "under `research/restriction` in the
   `microprediction/winning` repository" and then reports checking "the public repository's `main`
   branch".** The draft under review names neither a repository nor a branch. It names a path
   only. The report's substantive finding is correct and we are acting on it, since nothing is
   public and the reproducibility sentence therefore cannot be exercised by anyone. But the
   specifics were supplied by the report, not quoted from the paper, and a reader of the report
   would reasonably conclude we had pointed at a branch that contradicts us. The underlying defect
   is ours and is worse than the report says: a path with no repository is less useful than a
   wrong repository. We are fixing it with an archived snapshot.

2. **Table numbers.** The report cites Tables 3, 6 and 9 and Section 5.6. The manuscript was
   under active revision on the day of the audit and those numbers have since moved. We would
   find the next round considerably easier to act on if the report quoted the commit hash it
   audited, as the internal numerical audit of the same date did. Two of the items in the present
   report had already been corrected before it was written, which is nobody's fault but is
   avoidable.

3. **"Case V is structurally wrong unless the maps coincide"** is stated as though it bore on
   whether our comparison is meaningful. It does not: that both maps are misspecified for most
   populations is a premise of the paper, not a discovery about it. The conclusion already states
   that both err in the same direction, in the one setting where an observed restricted
   probability makes the check possible, and by more than they differ from one another.

## Accepted but deferred, with reasons

The following are correct and are not in this revision.

- **A formal local-expansion lemma for the cubic order**, with interior shares, a smoothness
  topology on the noise law and a nonsingular contrast Jacobian. The prose currently carries the
  hypotheses informally. We agree a lemma is preferable and will state one.
- **Refitting calibration inside the ranking bootstrap.** The intervals presently hold the fitted
  training models fixed and are therefore too narrow, most seriously in the two smallest
  collections. The menu experiment already does this correctly and is the pattern to follow.
- **Menu weighting as part of the estimand.** Uniform-over-subsets is a choice, the gain varies
  sharply with the size of the surviving menu, and a sensitivity analysis over uniform-over-size
  and application-weighted alternatives is warranted.
- **Sensitivity to the add-$\alpha$ convention.** The structural maps fit nothing, but
  $\alpha = 1/2$ is an implementation convention, one collection is scorable only because of it,
  and the two maps need not respond to pseudocounts identically.
- **Discovery and validation labels for the boundary rule.** We accept that the rule is
  hypothesis-generating and that the datasets used to form it should be distinguished from those
  used to test it. On the one case where chronology is claimed to carry weight, the line-length
  data, the classification was fixed in a committed draft before the data was obtained, and we
  will cite that commit so the order is independently checkable rather than asserted.

## On the recommended headline

We accept the substance of the proposed replacement and have moved most of the way to it. The
paper now claims parity rather than superiority, states that the count describes this corpus, and
presents the boundary as motivated rather than established. We will not describe the corpus as
"searched" without qualification, however, because the search was over the literature that asked
this question rather than over datasets that would answer it favourably, and the resulting corpus
contains the four collections on which our own default loses. A convenience corpus that includes
its own counterexamples is still a convenience corpus, and we will say so; but the distinction is
worth keeping.
