# Getty, Swets, Swets & Green (1979)

## Citation

Getty, D. J., Swets, J. A., Swets, J. B., & Green, D. M. (1979). On the prediction of
confusion matrices from similarity judgments.
*Perception & Psychophysics*, 26(1), 1–19.
doi:10.3758/BF03199856
(Bolt Beranek and Newman Inc., Cambridge MA.)

## Stimuli and master response set

Eight **visual displays** — contrast/spectrogram-like "visual representations of eight
complex sounds." The stimuli were viewed, not heard: "Three observers viewed visual
representations of eight complex sounds in both a pairwise similarity-judgment task and an
identification task." Stimulus duration 2 s. The four physical dimensions the authors
measured were locus of low-frequency energy, locus of midfrequency energy, visual
contrast, and periodicity (waxing/waning) — i.e. properties of the printed pattern.

Three trained observers, designated **B.F., J.K., J.S.**

**Master response set = all eight stimuli, {1,…,8}** — the "8 by 8 experiment," a complete
closed identification task. Overall error rate 18% (1,304 errors in 7,263 trials).
Presentation counts per stimulus (identical for all three observers): 287, 325, 286, 284,
292, 332, 308, 307 — 2,421 trials per observer.

## Restricted response sets (nested, overlapping, or a relabelling)

**Three OVERLAPPING restricted response sets, all four-element, with the stimulus set held
fixed at eight.** This is the "8 by 4 partial identification task":

| Condition | Allowed responses ("signals") | Stimuli presented |
|---|---|---|
| 1 | {1, 2, 5, 6} | all 8 |
| 2 | {3, 4, 5, 6} | all 8 |
| 3 | {1, 3, 5, 7} | all 8 |

The authors' own description:

> "we apply the model to three conditions of an 8 by 4 partial identification task, in
> which only four of eight stimuli—referred to as the 'signals'—correspond to allowable
> identification responses, a different set of four stimuli in each of the three
> conditions."

This is the cleanest available instance of the **response-set restriction** form of the CRR
test: the stimulus ensemble is unchanged, only the menu of permitted responses shrinks from
8 to 4. Because all eight stimuli continue to be presented, CRR is testable on **all eight
rows** of each condition, not merely the four "signal" rows. For any stimulus *i* and any
allowed response *j*, the zero-parameter prediction is
P(j | i, R) = P(j | i, master) / Σ_{k∈R} P(k | i, master).
Note that the three sets are pairwise overlapping (all contain response 5; 1 and 2 share
{5,6}; 1 and 3 share {1,5}; 2 and 3 share {3,5}), which yields cross-condition consistency
constraints in addition to the master-to-subset constraint.

230 analysed trials per observer per condition (an initial 30-trial practice block
discarded). Per-condition error rates over the signal subsets: 8%, 21%, 9% for Conditions
1, 2, 3; per-observer 5%, 24%, 8% for B.F., J.K., J.S. Overall 12%, "somewhat lower than
that in the full 8 by 8 experiment (18%)."

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

Both the master and the restricted matrices are printed as **raw integer confusion
frequencies, separately for each of the three observers. Nothing is pooled.**

- **Table 6 — "8 by 8 Experiment: Matrix of Confusion Frequencies for Each Observer."**
  Three 8×8 count matrices (B.F., J.K., J.S.) plus a printed row Total column.
  *All 24 rows verified: every row of every observer sums exactly to its printed total
  (287/325/286/284/292/332/308/307).* The OCR of this table is clean.

- **Table 8 — "8 by 4 Experiment: Matrix of Confusion Frequencies for Each Observer in
  Each of Three Conditions."** Nine 8×4 count matrices (3 observers × 3 conditions) with a
  printed Total column per condition. Column headers give the allowed response labels
  (1,2,5,6 / 3,4,5,6 / 1,3,5,7). Verified against a 200-dpi render of article page 14.
  Row totals per condition sum to 230 per observer.
  **One printed typo found:** Observer J.K., Condition 3, Stimulus 1 shows entries
  29, 0, 0, 4 but a printed Total of **23**; the entries sum to **33**. The printed total is
  the error (33 also makes the condition sum to 230). Flag this cell when ingesting.

- Figures 7–10 replot the same matrices as conditional response-probability distributions
  with the authors' model predictions overlaid; Figure 6 gives per-stimulus error
  probability. These are redundant with Tables 6 and 8.
- Tables 7 and 9 give the authors' fitted parameters (sensitivity *a*, salience weights
  w₁–w₄) and proportion-of-variance values, per observer and per condition. Figure 11 plots
  the estimated salience weights per condition against the accuracy-maximising weights.
- Tables 1–5 concern the MDS/INDSCAL similarity-judgment side (14 additional
  similarity-only subjects) and the physical stimulus measurements.

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

**Wayback-only, full text with embedded text layer, fetched successfully
(2,412,622 bytes, 19 pages):**

https://web.archive.org/web/20170909232732if_/https://link.springer.com/content/pdf/10.3758%2FBF03199856.pdf

The live Springer page (link.springer.com/article/10.3758/BF03199856) is paywalled and
bounces through idp.springer.com. The Wayback capture is of the publisher PDF taken while
Springer was serving it openly, and is complete.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**USABLE NOW — and it is the most valuable *unexploited* dataset in this literature,
because the authors never framed it as a CRR test at all.**

Strengths:
- Raw **counts**, not proportions, so likelihood scoring is exact with no reconstruction
  step.
- **Per-observer**, three observers, master plus three restricted sets each → 3 master
  matrices (8×8) and 9 restricted matrices (8×4) = 12 matrices, 3×8×8 + 9×8×4 = 480 cells
  of which 288 are restricted-set cells available as held-out forecasting targets.
- The restriction is of the **response set only**, with the stimulus ensemble fixed. This
  is the purest form of the "remove alternatives, preserve odds between survivors" test —
  it removes the confound that troubled Townsend & Landon, where changing the response set
  necessarily changed which stimuli were shown.
- Overlapping (not nested) sets give extra cross-condition consistency checks.
- Table 6 fully arithmetic-verified; Table 8 verified against the page image with one
  benign printed-total typo identified.

Caveats:
- Modest cell counts in the 8×4 matrices (~28 trials per row), so per-cell noise is real.
  The authors themselves note "a 10% deviation between obtained and predicted probabilities
  corresponds to a difference of only about three responses, given that each obtained
  distribution is based on about 30 trials." Scoring should be likelihood-based on counts,
  not on proportion differences.
- Presentation counts are not perfectly balanced across conditions for a given stimulus
  (e.g. stimulus 6: 30, 45, 31), which matters only if you weight rows.
- The master 8×8 came from a *separate, earlier* set of sessions rather than being
  interleaved with the 8×4 blocks, so practice/drift between master and subsets is not
  controlled the way it is in Townsend & Landon.

Only OCR transcription is needed and both tables have already been checked here.

## What the authors concluded about CRR, quoted verbatim where possible

**The authors never mention the constant-ratio rule.** The phrase does not occur in the
paper; Clarke, Hodge, Morgan and Egan are absent from the reference list. Luce appears only
as the functional form of their decision rule:

> "Equation 3 is essentially Luce's (1963) choice model, with the added assumption that
> there are no differential response biases. While it would be a simple matter to include
> measures of response bias in the model, we have chosen to exclude them here for reasons
> of simplicity (fewer parameters to estimate) and because we have no reason to expect
> strong response biases."

So their model is a **bias-free Luce/similarity choice model whose similarity terms are
re-estimated in every condition** — precisely Townsend & Landon's *weak* SCM, the version
that does *not* imply the CRR. They state this design choice explicitly and up front:

> "We will not assume, however, that the particular set of salience weights determined for
> each observer by INDSCAL in the similarity-judgment task necessarily applies to the
> identification tasks. In fact, we will show later that, within observers, the set of
> salience weights changes in predictable ways across different conditions of the
> identification task."

**This is a CRR/IIA violation reported as a substantive finding rather than as a failure.**
From the abstract:

> "Three conditions of the identification task, calling for identification of different
> subsets of the eight stimuli, led the observers to vary the weights they placed on the
> dimensions; they apparently adjusted the weights to maximize the percent correct
> identification."

In the Results:

> "The pattern of estimated weights clearly changes from one condition to another,
> suggesting that the observers modified their set of salience weights from condition to
> condition according to the composition of the set of four stimuli defined as signals."

And named as a mechanism, "adaptive tuning":

> "We believe that the observer was engaged in an adaptive tuning process in which the
> relative weighting of dimensions was adjusted in order to maximize the discriminability
> of the subset of stimulus patterns to be identified in that condition. This tuning
> process probably takes place gradually, over many trials, based on the feedback given the
> observer regarding the correctness of identification."

The residual is deliberately confined to the metric while stimulus positions are held
fixed — i.e. the authors absorb the context effect into a re-scaling, the same place
Townsend & Landon found it (η, not β):

> "Of particular interest is the fact that the model accounts for these changes in
> performance in terms of changes in the relative salience of perceptual dimensions rather
> than in terms of changes in the structure of the perceptual space. In the model, a given
> stimulus is assumed to have a fixed location in the perceptual space, regardless of
> changing stimulus context."

**A systematic residual is reported and treated as nuisance** (in the master 8×8 fit):

> "There are occasional deviations between predicted and obtained probabilities that appear
> to be systematic in that two of the three observers show the same pattern of deviations.
> For example, when Stimulus 3 was presented, both observers J.K. and J.S. made Response 6
> more frequently and Response 7 less frequently than predicted. This, and other such
> examples, may result from the observers' use of dimensions that were available in the set
> of patterns but not included in the model analyses."

Their summary claim for the restricted-set predictions — note that this is a *fitted*
prediction with per-condition free weights, not a zero-parameter one:

> "Overall, however, the model, using the perceptual space derived from similarity
> judgments within the context of the full stimulus set, predicts quite well the confusion
> matrices for partial identification tasks in which different subsets of the stimuli are
> identified."

**Implication for the present project:** the zero-parameter CRR forecast of Table 8 from
Table 6 has, as far as this reconstruction can tell, never been computed. Getty et al.
established that a *free-weight* Luce model fits each restricted matrix, and that the
weights must change across restrictions — which is exactly the statement that the
*constrained* (CRR) version must fail. The size and sign of that failure is sitting
un-scored in Tables 6 and 8.
