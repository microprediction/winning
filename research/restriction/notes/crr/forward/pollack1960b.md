# Pollack, Rubenstein & Horowitz 1960 — Communication of verbal modes of expression

Negative file. Recorded so a later pass does not have to rediscover it: this is the **largest
menu contraction in the auditory cluster (16 -> 8 -> 4 -> 2)** and a real CRR test, but the master
matrix was never printed.

## Citation

Pollack, I., Rubenstein, H. and Horowitz, A. (1960). "Communication of Verbal Modes of
Expression." *Language and Speech* 3(3): 121-131 (article begins p. 121).
Operational Applications Office, Air Force Command and Control Development Division, Bedford,
Massachusetts. Issued as Technical Note 60-24 (Project 7684, Communications in Noise).

Same journal volume as Pollack & Decker (1960) — see `pollack1960.md`.

## Domain and stimuli

Auditory, but **not** speech-sound identification: recognition of a talker's intended **mode of
verbal expression**. Four talkers (untrained, not actors) each read the two neutral sentences
"The lamp stood on the table" and "His friend is coming by train" in each of **16 modes of
expression**, in randomised order, monitoring their level on a VU meter. The 16 modes were the
best-recognised of a larger set from a literature search, and included: confidential
communication, objective statement, fear, uncertainty, disbelief, happiness, impatient
repetition, sarcasm, threat, approval, disgust, surprise, anger (Table 1 lists all 16).

Recordings were played to three crews of six listeners each, in a quiet room over loudspeaker,
no added noise, no correctness feedback. Each point in Figure 1 is the average of 2300
observations (32 observations by each of 18 listeners with each of 4 talkers).

## Master and restricted response sets

Yes, and it is a nested sequence of true response-menu restrictions over the same recorded
stimuli, each listener holding the printed list of modes:

> "After the listeners were exposed to the entire set of materials, the same materials were played
> back, but with only eight response alternatives available per item ; then with four response
> alternatives per item ; and, finally, with two response alternatives per item." (p. 123)

- **Master menu:** all **16** modes, 16 x 16 confusion matrix.
- **Restricted menus:** **8**, then **4**, then **2** alternatives, over the same stimulus
  recordings.
- **Prediction direction:** each menu predicted from the menu twice its size — "Predictions for n
  response alternatives were based on the obtained result for 2n response alternatives.
  Specifically, predictions for 2, 4, and 8 response alternatives were based upon the obtained
  results with 4, 8, and 16 response alternatives respectively" (p. 123). So there are three
  independent full-menu-calibrate / restricted-menu-predict steps, not one.

Two design caveats. (i) The restricted conditions were always run **after** the master, in fixed
order, and the authors say so plainly: "The direction of the discrepancy is consistent with the
assumption that the listening crew became more experienced in proceeding from the 2n- to the
n-alternative tests" (p. 123). Practice is confounded with menu size. (ii) The second half of the
paper uses CRR *generatively* rather than as a hypothesis under test — successively deleting the
worst mode and redistributing its mass by CRR, 16 -> 15 -> ... -> 8, to pick a "best eight". Only
the leftmost point of Figure 2 is data: "It is important to note that only the points above the
abscissa value 0 in Fig. 2 represent experimental observations. Thereafter, successive changes are
introduced by application of the constant-ratio rule" (p. 124). Figure 2 is therefore not
evidence about anything.

## What numbers are printed or deposited

**The 16 x 16 master matrix is not printed, and neither is any restricted-menu matrix.** The paper
prints only:

- **Table 1 (p. 122)** — the 16 modes with their rank order under 16 alternatives, and asterisks
  marking which were eliminated during the CRR-driven reduction to eight. Ranks, not proportions.
- **Figure 1 (p. 123)** — average per cent correct recognition against number of response
  alternatives (16, 8, 4, 2), obtained (circles) versus CRR-predicted (squares). Four pairs of
  aggregate numbers, read off a small plot.
- **Table 2 (p. 124), "Prediction by the constant-ratio rule from 2n- to n-alternative tests"** —
  four summary rows for each of the three prediction steps (labelled 16, 8, 4):
  mean absolute difference over all entries **2.9% / 4.0% / 4.6%**;
  mean absolute difference over diagonal entries only **6.5% / 6.4% / 6.5%**;
  obtained correct minus predicted correct **3.3% / 3.9% / 0.8%**;
  per cent of differences greater than 0.10 **6.2% / 10.9% / 0.0%**.
- **Figure 2 (p. 125)** — the CRR-generated trajectory of per-mode performance under successive
  deletions; only the first column is observed.
- No cell-level entry of any matrix appears anywhere; no data deposit (1960).

Aggregate over 18 listeners and 4 talkers throughout; no per-subject or per-talker breakdown.

## Access with a fetched url

Full text read (OCR of the microfilm scan of *Language and Speech* vol. 3, which contains the
whole year; this article sits at pp. 121-131 of the scan):

- https://archive.org/download/sim_language-and-speech_january-march-1960_3_1/sim_language-and-speech_january-march-1960_3_1_djvu.txt (fetched, 594,721 bytes — abstract, Procedure, "Test of Constant-Ratio Rule" section, Table 1, Table 2, Figure 1 and 2 captions, the mode-selection section, and references all read directly)
- https://archive.org/metadata/sim_language-and-speech_january-march-1960_3_1 (fetched; item is unrestricted, page-image PDF available at 48.7 MB should anyone want to re-read Figure 1 by eye)

The SAGE version of *Language and Speech* returns HTTP 403 to automated fetches.

## Usability verdict

**Not sufficient.** The design is the most aggressive menu contraction anywhere in this cluster —
a 16-alternative master reduced to 8, 4 and 2 with the same stimuli and the same listeners, giving
three chained out-of-sample prediction steps — but **no confusion matrix is printed at any menu
size**, so there is nothing to calibrate a Thurstone model on and nothing to score against at
cell level. What survives in print is four aggregate percent-correct pairs (Figure 1) and twelve
summary discrepancy numbers (Table 2). Those twelve numbers do permit one weak comparison: a
model calibrated on a recovered 16 x 16 could be scored on whether it beats CRR's printed mean
absolute differences of 2.9 / 4.0 / 4.6 per cent and its printed signed diagonal bias of
+3.3 / +3.9 / +0.8 per cent — but the 16 x 16 cannot be recovered, so that comparison is
unavailable.

Worth noting for the paper's argument even so: **the signed bias is in the same direction as
everywhere else in this literature**. Obtained correct exceeds CRR-predicted correct at every step,
i.e. CRR *under*-predicts accuracy on the smaller menu here — and the authors attribute it to
practice rather than to a failure of the rule. Combined with Engstrand & Moeller's arithmetic
(small-count denominator bias) and Hodge & Pollack's response-label confusion account, that makes
three distinct nuisance explanations offered for restricted-menu discrepancies in this literature,
none of which is separable from an IIA violation using published summary statistics alone. This is
an argument for the project's own machine-generated data rather than for mining these papers.

Classification: **CRR-TEST-BUT-NUMBERS-NOT-PRINTED**.

## Conclusion about CRR quoted verbatim

From the "Test of Constant-Ratio Rule" section, p. 123:

> "The average discrepancy between the average obtained and predicted correct percentage (line 3)
> ranges from 1 to 4%. The direction of the discrepancy is consistent with the assumption that the
> listening crew became more experienced in proceeding from the 2n- to the n-alternative tests. The
> mean difference, sign ignored, between all of the predicted and obtained entries (line 1) ranged
> from 3 to 5%. The mean difference, sign ignored, for the correct entries only (line 2) was about
> 6-5%. The accuracy of prediction is considered to be good."

And the sentence that licenses the rest of the paper, p. 124:

> "With some assurance that the constant-ratio rule was applicable to modes of expression, we
> attempted to select the most discriminable set of eight modes from the initial sixteen."

Also from Table 1's note, p. 122:

> "The constant-ratio rule was successively applied upon the initial 16x16 confusion matrix to
> achieve the final ratings."

There is no separate Conclusions section; the paper's abstract makes no claim about CRR, reporting
only that "Reasonably high levels of performance may be achieved under conditions of reduced
acoustic information."
