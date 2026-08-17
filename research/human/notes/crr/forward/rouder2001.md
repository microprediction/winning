# Rouder 2001 — Absolute identification with simple and complex stimuli

## Citation

Rouder, J. N. (2001). Absolute identification with simple and complex stimuli.
*Psychological Science*, 12(4), 318–322. doi:10.1111/1467-9280.00358

Full text obtained and read. This is the source of the six-choice/two-choice letter data
that Rouder (2004) later uses to reject the constant ratio rule; see `rouder2004.md`.

## Domain and stimuli

Two experiments, both absolute identification with a set-size manipulation.

**Experiment 1 — line lengths (unidimensional).** Six students. Two-choice condition:
identify which of two line lengths was presented. Six-choice condition: which of six.
Line presented for 333 ms after an 800-ms blank foreperiod, position jittered by a
6-pixel two-dimensional uniform perturbation. Numbered keys on the top row of the
keyboard, 60-pixel line paired with "1" and so on, with feedback after each response.
Participants did 300 trials of the two-choice condition, then 700 of the six-choice
condition.

**Experiment 2 — letters (complex/multidimensional).** 15 participants. Two-choice
condition: letters W and E. Six-choice condition: letters Q, W, E, R, T, Y. Letters
degraded with both forward and backward masks made of an array of symbol characters, so
accuracy was well below ceiling. Conditions alternated across 16 blocks of 50 trials in a
single session; feedback given.

## Master and restricted response sets

Nested, within-subject, in both experiments, and this is the largest single removal found
anywhere in this branch — five of six alternatives dropped to two.

| Experiment | Wide menu | Narrow menu | Nesting |
|---|---|---|---|
| 1 (line lengths) | 6 lengths | the 2 most intermediate of the 6 | nested, and deliberately the interior pair |
| 2 (letters) | Q, W, E, R, T, Y | W, E | nested |

"The two line lengths in the two-choice condition were the two most intermediate line
lengths in the six-choice condition" (p. 319) — the interior choice matters, because it
means the removed alternatives flanked the survivors on both sides rather than being
tacked on at an end.

Because the task is keyed absolute identification with feedback and the participant is
told the condition, the response set is genuinely restricted, not merely the stimulus set.

Both experiments give a 6x6 matrix and a 2x2 matrix per participant over the same stimuli
and the same subjects, which is exactly the master/restricted pairing the project needs.

## What numbers are printed or deposited

**No confusion matrices are printed.** What is printed:

- **Table 1** (p. 321): per-participant SCM similarity estimates (eta) between W and E in
  the two-choice and six-choice conditions, plus chi-square goodness-of-fit values for
  SCM, and for two mixture models MM1 and MM2 (MM1 also reports its estimated encoding
  probability D). 15 rows, participants A–O. Parameters and fit statistics only.
- **Figure 1**: scatter of six-choice against two-choice psychological distance, one
  lettered point per participant for the letter task and one numbered point per
  participant for the line-length task.
- **Figure 2**: the same for Townsend & Landon's (1982) data, one panel per participant,
  points labelled by letter pair.

The matrices are held back, with an offer, footnote 1 (p. 319):

> "The frequency matrices from Experiments 1 and 2 may be obtained from the author."

No supplementary file, no repository deposit. Treat as not deposited — the offer is 25
years old and the hosting lab site (`pcl.missouri.edu`) is gone.

Trial counts are recoverable in outline (300 two-choice and 700 six-choice trials in
Experiment 1; 8 blocks of 50 per condition in Experiment 2), which would let counts be
reconstructed *if* the proportions were available. They are not.

## Access with a fetched url

Fetched successfully:

    http://web.archive.org/web/20170808070230if_/http://pcl.missouri.edu/sites/default/files/Rouderpsysci2001.pdf

HTTP 200, 97,864 bytes, 5 pages, born-digital PDF with a text layer — the published
two-column Psychological Science typesetting, pages 318–322. The live host no longer
resolves. Located via the Wayback CDX index for `pcl.missouri.edu*`, which also lists
`rouder.psyrev.2004.pdf` and about fifty other Rouder reprints.

## Usability verdict

**CRR-TEST-BUT-NUMBERS-NOT-PRINTED.**

The design is ideal and the numbers are absent, which is the worst possible combination.
A 6→2 nested contraction with 15 participants, degraded masked letters, and feedback would
be the single best test in this branch — better than Townsend & Landon, because the
removal is drastic enough that renormalisation and a Gaussian race must diverge visibly.
But the paper prints only fitted SCM similarity parameters and chi-square values, and
neither can be inverted to recover a 6x6 matrix.

Two further cautions if the matrices are ever obtained:

1. **Order is confounded in Experiment 1.** All participants did the 300 two-choice
   trials before the 700 six-choice trials, so practice favours the six-choice condition.
   Rouder addresses this in footnote 3, arguing the practice effect works against his
   conclusion rather than for it. Experiment 2 alternated conditions and is clean.
2. **Line lengths are near-substitutes by construction, and the removed pair flanked the
   survivors.** Experiment 1 gives the opposite set-size effect from Experiment 2 — that
   is the paper's whole point — so the two experiments should never be pooled.

The paper does **not** test CRR. It fits SCM separately to each menu and asks whether the
fitted similarity parameter is invariant across menu size, which is a parametric proxy for
the same question. That the fitted-model route and the direct odds-ratio route (Rouder
2004) reach the same answer is a mild reassurance, but a fitted-SCM non-invariance is not
the out-of-sample failure the project needs to demonstrate. Do not cite this paper as a
CRR test; cite Rouder 2004 for that, and this one for the data and for the direction of
the effect.

Its independent value: it establishes the sign of the effect for letters in fifteen
subjects, and it independently replicates it on Townsend & Landon's four subjects (21 of
24 points above the diagonal), which corroborates the printed-matrix result recorded in
`townsend1982.md`.

## Conclusion about CRR quoted verbatim

**Not discussed.** The phrase "constant ratio rule" does not appear in the paper, and
Clarke (1957) is not in the reference list. The CRR framing of these data is supplied
three years later by Rouder (2004). What the paper concludes about the CRR-equivalent
question — invariance of the similarity parameter across choice-set size — is:

Abstract, p. 318:

> "The experiments reported here tested this theoretical dissociation using Luce's (1963)
> Similarity Choice Model to measure the psychological distance between stimuli in
> line-length-identification and letter-identification tasks. The psychological distance
> between line-length stimuli decreased with the number of to-be-identified stimuli; this
> result is concordant with capacity limits in unidimensional absolute identification.
> Surprisingly, the opposite result held in letter identification. Psychological distance
> between letters increased with an increased number of to-be-identified stimuli. This
> result indicates an opposite type of processing deficit: People process letters more
> efficiently with more choices."

On the letter result, p. 321:

> "As shown in Table 1, the estimated similarity between W and E was greater in the
> two-choice condition than in the six-choice condition for 14 of the 15 participants
> (participant B is the exception)."

On the reanalysis of Townsend & Landon, p. 321 — worth recording because it is an
independent count on the matrices that *are* printed elsewhere:

> "The psychological distances obtained by analyzing data from the three-choice and
> five-choice conditions are plotted in Figure 2. Each plot is for a different
> participant, and each point is labeled by the letter pair it represents. As can be seen,
> 21 of the 24 points are above the diagonal, indicating a processing inconsistency with
> poorer performance in the three-condition case relative to the five-choice condition.
> Although Townsend and Landon did not publish their parameter estimates, they noted that
> similarities were greater in the three-choice case than in the five-choice case."

Closing statement, p. 322:

> "There is no explanation of how or why the psychological distance between any two
> letters should increase with increasing numbers of stimuli. The present finding with
> letter identification is not yet well explained and provides a meaningful constraint on
> current and future theories of letter recognition."

That last sentence is the opening the project is aiming at, stated by the author as an
open problem.
