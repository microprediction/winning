# Response to the third review

We are grateful for a report that is unusually specific and, on the mathematics, unusually
useful. The independent recomputation of the reverse-hazard identity, the Gumbel cumulants, the
location-scale invariance and the concentrated-share example all agree with ours, and the
deterministic small-$\varepsilon$ experiment showing the local exponent approaching three agrees
with what our own browser check suite finds by a different route. We take the report's point that
this is not the right evidentiary move: two numerical experiments agreeing is not a proof, and
citing the agreement would not make it one. We will archive both calculations in a versioned
supplement and state the cubic rate as a local expansion under explicit smoothness and
nonsingularity conditions until the lemma is written.

Below we separate what we accept, what we contest, and three attributions the audited version
does not support. On the last we ask for more care, because in one case the imprecision produced
a recommendation that would have introduced an error into the paper.

A note on the copy circulated to us: the file we sent is a single document of one hundred and
sixty-odd lines with no repeated sections. If the version received restarts midway and ends
mid-bullet, that happened in transit rather than on disk, and the intact file is in the
repository alongside the manuscript.

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

**On truncation we were both partly right, and a two-way split was not enough.** Our objection
stands in one direction: $P(X=i \mid X\in T) = p_i / \sum_{j\in T} p_j$ is conditioning inside a
fixed probability space, and it does not answer a counterfactual question about a mechanism that
has been altered, so the report's suggested wording read as though truncation were always
conditioning. But the report is right that our two-way split understated the concession, and the
four-way distinction it proposes is better than ours. We adopt it.

Conditioning a fixed categorical law is proportional renormalization by definition. Masking a
fixed softmax and rescaling is the same arithmetic on a score vector, and we now say the stronger
thing the report is owed: that operation does not merely permit the linear transport, it *is* the
linear transport, so IIA is built into the operator rather than being one of two readings of it.
Intervening on the feasible set changes the choice-generating experiment, and there the original
shares do not determine the new ones. Recomputing scores with the permitted set as an input is
not a transport of the old vector at all. The manuscript now separates the four, says the paper
is about the third, and locates its practical relevance in the second: that is where the linear
transport is already deployed without anyone having chosen it.

We also accept the smaller correction that follows. Where the manuscript said the two maps are
"the candidates" it now says they are two conventional candidates, since any regular standardized
shape supplies another.

**The blanket demand for $9{,}999$ replicates is not costed.** We accept the direction and the
reasoning: $B=200$ gives a Monte Carlo standard error of about $0.015$ at a true tail near
$0.05$, which is large against a $0.05$ boundary, and $B=60$ on the news row supports a floor and
not an estimate. We will raise $B$, report exceedance counts alongside adjusted tails, and match
the simulation unit to the actual clustering. But a uniform $9{,}999$ across the ranking
collections means rerunning a pipeline that already evaluates $1{,}013$ subsets across five folds
per replicate, and the report does not say what that buys. Our first proposal, to raise $B$ until the interval excludes the
$0.05$ boundary, was wrong and we withdraw it: inspecting a fixed-sample interval and stopping
when it clears a threshold is optional stopping and destroys the coverage that made the interval
worth quoting. The report is right to catch it. We will instead pre-specify $B$ from a target
Monte Carlo error, print the exceedance count $b$ alongside $(b+1)/(B+1)$ and an interval for
every row, and stop forcing rows through a binary threshold at all. The high-cost gamble row near
$0.035$ is the one that most needs the extra simulation, and rows with zero exceedances can be
resolved more cheaply provided the $0/B$ is printed rather than hidden behind a floor.

**One point of agreement, stated neutrally.** Under an exact Luce population with unlimited
calibration data the Gaussian-minus-linear gain must be negative, since linear renormalization is
then the correctly specified restricted-menu map. So the null median mixes a finite-sample
estimation effect with Case V misspecification under Luce. That is what our null medians show:
negative wherever the shares rest on more than about two thousand observations. We had described
this as strengthening the paper. It does not. It clarifies what the diagnostic contains, and
subtracting a composite null median does not isolate a structural effect under a non-Luce
population.

## Three attributions not supported by the audited version

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

## One correction we found ourselves after drafting this

Writing a second, independent implementation of the two maps in JavaScript, so that the
mathematical claims can be checked in a browser from nothing, turned up a defect the reports did
not. The two implementations agree to $10^{-4}$ on fifteen of the sixteen tone rows and disagreed
sharply on one. That row contains an exact zero, which has to be floored, and the resulting
location is far enough out that two independently written calibrators do not agree on it. The row
is now excluded, and the wide ten-to-eight gain moves from $-0.0057$ to $-0.0041$. The direction
is unchanged and linear renormalization still wins all four tone conditions, but the figure that
appeared in the scoreboard was resting on an ill-conditioned inversion and should not have been
quoted to four decimals. We mention it because it bears on a theme of both reports: the
calibration tolerance is a real source of uncertainty that our intervals do not carry.

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
