## Citation

Getty, D. J., Swets, J. A., Swets, J. B., & Green, D. M. (1979). On the prediction of
confusion matrices from similarity judgments. *Perception & Psychophysics*, 26(1), 1-19.
DOI 10.3758/BF03199856. Authors and pagination confirmed via Crossref. Bolt Beranek and
Newman Inc.

## Domain and stimuli

Visual representations (spectrogram-like displays) of eight complex sounds. Three
observers (B.F. or S.F., J.K., J.S.) did both a pairwise similarity-judgment task and an
identification task. INDSCAL multidimensional scaling of the similarity judgments gave a
three-dimensional perceptual space, with periodicity as a candidate fourth dimension.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**Nested, with an important twist: the stimulus set is held constant and only the
response set shrinks.** Two experiments.

"8 by 8": all eight stimuli, all eight responses. Master matrix.

"8 by 4 partial identification": all eight stimuli still presented, but only four
responses allowed. Verbatim:

> "We may test the model in another way by considering how well it is able to predict the
> pattern of identification confusions when an observer is limited to responses associated
> with only a subset of the eight stimuli. In this section, we apply the model to three
> conditions of an 8 by 4 partial identification task, in which only four of eight
> stimuli - referred to as the 'signals' - correspond to allowable identification
> responses, a different set of four stimuli in each of the three conditions."

Signal sets: Condition 1 = {1,2,5,6}; Condition 2 = {3,4,5,6}; Condition 3 = {1,3,5,7}.
Labels are the stimulus numbers, identical across conditions, so genuinely nested.

Caveat: the four non-signal ("noise") stimuli were still presented but had no correct
response, so those rows are forced errors. The clean nested comparison lives in the four
signal rows of each condition, matched against the same rows and columns of the 8x8.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

**Raw confusion frequencies, printed, per observer.** Counts, not proportions.

Table 6, "8 by 8 Experiment: Matrix of Confusion Frequencies for Each Observer" — three
8x8 matrices with row totals. Verified row totals 284-332 per stimulus per observer
(e.g. B.F. row 1: 273 0 1 1 0 0 12 0, total 287). Overall error rate 18%, 1,304 errors in
7,263 trials.

Table 8, "8 by 4 Experiment: Matrix of Confusion Frequencies for Each Observer in Each of
Three Conditions" — 3 observers x 3 conditions x 8 stimuli x 4 responses, with row totals.
Verified row totals 20-45 per stimulus per observer per condition; "The analyses that follow
are based on 230 trials" per observer per condition.

Both tables extract cleanly with `pdftotext -layout` — I recovered the full numeric content
of both without hand transcription.

Sample-size asymmetry worth planning around: the 8x8 has ~300 trials per stimulus per
observer, the 8x4 has ~30. Pooling the three observers gives ~90 per stimulus per condition.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Open** at the publisher. Springer's legacy Psychonomic archive serves this free;
Unpaywall confirms `is_oa: true` with a publisher OA location. Fetched, HTTP 200,
application/pdf, 2,412,622 bytes, 19 pages:

    https://link.springer.com/content/pdf/10.3758/BF03199856.pdf

A browser may show a cookie-consent redirect; fetching the URL directly returns the PDF.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Usable now.** Counts already extracted mechanically, no digitizing needed. The
constant-stimulus-set / shrinking-response-set design is the cleanest available isolation
of the response-set manipulation, because nothing about the stimuli changes. Two limits:
only three observers, and small n in the restricted conditions.

## What the authors concluded, quoted verbatim where possible

The abstract states the restriction effect as a change in the perceptual representation,
not just in the choice rule:

> "Three conditions of the identification task, calling for identification of different
> subsets of the eight stimuli, led the observers to vary the weights they placed on the
> dimensions; they apparently adjusted the weights to maximize the percent correct
> identification."

Expanded in the discussion:

> "the stimuli remained constant across all conditions, the subset of stimuli that we
> required the observer to identify - the 'signals' - changed from condition to condition.
> Furthermore, it was only for this subset of stimuli that the observer received
> discriminative feedback that indicated which stimulus had been presented. We believe that
> the observer was engaged in an adaptive tuning process in which the relative weighting of
> dimensions was adjusted in order to maximize the discriminability of the subset of
> stimulus patterns to be identified in that condition. This tuning process probably takes
> place gradually, over many trials, based on the feedback given the observer regarding the
> correctness of identification."

> "In general, the observed patterns of dimension salience weights seem consistent with the
> hypothesis that observers are tuning their weighting of dimensions in order to maximize
> the probability of a correct identification."

This is a substantive warning for our use. If restricting the response set retunes the
perceptual space, then *no* fixed-representation map — renormalization or Gaussian race —
should fit the restricted matrices out of sample, and the discrepancy is not evidence
between the two maps. The feedback structure is the likely driver, so the effect should be
weaker in designs where feedback is unchanged or absent. Worth checking whether the
Gaussian race fits Condition 2 (signals {3,4,5,6}, a contiguous middle block, largest
observed weight change) worse than Condition 1.

On the forced-error rows:

> "the observer most likely reinterprets the task to respond with the stimulus among the
> allowable four which is 'most similar' to the presented stimulus, even though it is
> clearly incorrect."
