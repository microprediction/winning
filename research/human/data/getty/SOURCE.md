# Getty, Swets, Swets & Green (1979) — eight stimuli, four allowed responses

Getty, D. J., Swets, J. A., Swets, J. B., & Green, D. M. (1979). On the prediction of
confusion matrices from similarity judgments. *Perception & Psychophysics*, 26(1), 1-19.
doi 10.3758/BF03199856. Authors, pagination and DOI confirmed via Crossref.

Open access at the publisher; fetched 2026-08-17, HTTP 200, 2,412,622 bytes, 19 pages:

    https://link.springer.com/content/pdf/10.3758/BF03199856.pdf

## Why this is the cleanest restriction design in the corpus

The stimulus set never changes. Three observers identify the same eight complex sounds
throughout. What changes is the *response* set: the "8 by 4" experiment allows only four
responses, a different four in each of three conditions, and the labels are the stimulus
numbers themselves, so the smaller menus nest inside the larger with nothing inferred.
Every other nested design in this project varies the stimuli, the subjects, or both.

Signal sets: condition 1 = {1,2,5,6}, condition 2 = {3,4,5,6}, condition 3 = {1,3,5,7}.

All eight stimuli were presented in every condition, so the four non-signal rows are
forced errors — the correct answer has been withdrawn. Those rows are not spoilage; they
are the restriction test in the case where the favourite itself is removed, and they are
reported as their own split.

## Files here

| File | Source | Contents |
|---|---|---|
| `master_8x8.csv` | Table 6 | 3 observers x 8 stimuli x 8 responses, confusion frequencies |
| `restricted_8x4.csv` | Table 8 | 3 observers x 3 conditions x 8 stimuli x 4 responses |

Transcribed by hand from `pdftotext -layout` output. **Every row sum reproduces the
printed row total**, which is the transcription check; the one discrepancy is the printed
total for observer J.K., condition 3, stimulus 1, which reads 23 where the cells sum to 33
and the 8x8 row makes 33 the only consistent reading. Treated as a typographical or OCR
slip in the total, not in the cells.

Sample sizes are asymmetric by design: about 300 trials per stimulus per observer in the
8 by 8, about 30 in each 8 by 4 condition ("The analyses that follow are based on 230
trials" per observer per condition).

## The authors' own warning, which is why this is a boundary case

From the abstract:

> "Three conditions of the identification task, calling for identification of different
> subsets of the eight stimuli, led the observers to vary the weights they placed on the
> dimensions; they apparently adjusted the weights to maximize the percent correct
> identification."

and from the discussion:

> "We believe that the observer was engaged in an adaptive tuning process in which the
> relative weighting of dimensions was adjusted in order to maximize the discriminability
> of the subset of stimulus patterns to be identified in that condition. This tuning
> process probably takes place gradually, over many trials, based on the feedback given
> the observer regarding the correctness of identification."

If restriction retunes the perceptual space then the survivors are not the same
alternatives before and after, which is this project's second boundary condition. Feedback
was given only on the signal subset, which is the likely driver, so the effect should be
weaker wherever feedback is absent or unchanged.

## What the run found

`research/human/getty.py`, output in `research/human/results/getty.txt`. The race wins
overall (+0.0272 nats per cell, excess +0.0343) and on the signal rows (+0.0111, cell
bootstrap [+0.0021, +0.0225]). It loses in condition 2 alone (-0.0127, excess -0.0330).

Condition 2 is the condition whose four survivors are mutual confusions. Measuring, from
the 8 by 8 alone, the fraction of a signal stimulus's errors that land on another signal of
the same condition gives 0.103, 0.790 and 0.335 for conditions 1, 2 and 3. The only
condition the race loses is the one where that fraction is high, and the statistic is
computable before any restricted data is seen.
