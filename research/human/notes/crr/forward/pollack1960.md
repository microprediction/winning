# Pollack & Decker 1960 — Consonant confusions and the constant-ratio rule

## Citation

Pollack, I. and Decker, L. (1960). "Consonant Confusions and the Constant Ratio Rule."
*Language and Speech* 3(1): 1-6. doi:10.1177/002383096000300101.
Also issued as Technical Note 59-17, Air Force Cambridge Research Center (Project 7681,
Auditory Presentation of Information).

## Domain and stimuli

Auditory / speech. Eight word-initial English consonant contexts, each paired with /a/ as in
"father" (e.g. /fa/, /ha/): **/f, h, l, r, w, y/, the cluster /hw/, and the absence of an
initial consonant /#/**. Read in the carrier phrase "you will trah, ______" by a live talker
and presented in noise to a crew of five university-student listeners (six subjects served
alternately as talker and listeners). Four speech-to-noise ratios: **-17, -13, -9, -5 dB**.

Each listener had a bank of response buttons wired into a scoring system that accumulated the
full stimulus-response confusion matrix in real time. 360 observations per stimulus
alternative per matrix (about 2900 observations per 8x8, 1450 per 4x4, 700 per 2x2).

## Master and restricted response sets

This is a true master-plus-restricted design, and the restriction is applied to the response
alternatives, not just to the stimuli. Quoting the Procedure (p. 2):

> "The message sets were defined for the listeners such that the listeners' response
> alternatives agreed with the talker's possible message alternatives."

- **Master menu:** the full **8 x 8** matrix over all eight consonants, run at all four S/N
  ratios.
- **Restricted menus, three 4-item subsets:** **/l,r,w,y/**, **/f,h,l,r/**, and
  **/f,h,hw,#/**. All three were run at the highest three S/N ratios; the first two were also
  run at -17 dB.
- **Restricted menus, six 2-item subsets:** **/f,h/ /l,r/ /h,#/ /l,w/ /w,hw/ /f,w/**, all at
  a single S/N ratio of -13 dB.

Predictions for the 4x4 menus were computed from the corresponding 8x8 matrix by the
constant-ratio rule (i.e. proportional renormalisation); 2x2 predictions were made both from
the 8x8 and from the 4x4 matrices.

## What numbers are printed or deposited

**The master matrices are printed; the restricted-menu observed matrices are not.**

- **Table 2 (pp. 4-5) prints the complete 8 x 8 confusion matrix for all four S/N ratios**
  (-5, -9, -13, -17 dB), each cell being "the nearest rounded percentage entry of the
  confusion matrix", rows = stimuli, columns = responses. These are the full-menu shares the
  project needs for calibration. (Caveat: this is a microfilm scan and the OCR of Table 2 is
  partly mangled — the -13 and -17 dB blocks read cleanly, the -5 and -9 dB blocks need the
  page image re-read by eye. The page images are in the same Internet Archive item.)
- **Table 1 (p. 4)** prints only *summary* statistics: the mean absolute deviation between
  obtained and CRR-predicted cell entries, one number per (subset x S/N ratio) cell for the
  8x8 -> 4x4 predictions, and one number per 2-item subset for the 8x8 -> 2x2 and 4x4 -> 2x2
  predictions. Values run 1.8 to 5.9 percentage points for the 4x4s and 2.2 to 15.4 for the
  2x2s.
- **Figure 1 (p. 3)** is a scatterplot, one point per 4x4 cell, of (observed 4x4 percentage on
  the abscissa, obtained-minus-predicted deviation on the ordinate), panelled by S/N ratio and
  coded by which 4x4 subset. In principle this figure is digitisable and would recover both
  the observed and the predicted restricted-menu cell values — but the authors state
  explicitly that "(Some points in the densely packed region have been omitted)", so the
  recovered set would be incomplete.
- **Figure 2 (p. 5)** is a confusion-vector diagram derived from Table 2, thresholded; it adds
  no new numbers.
- No 4x4 or 2x2 observed matrix is tabulated anywhere in the paper, and there is no data
  deposit (1960).

Reported accuracy: "Approximately 92 per cent of the predictions are within 10 percentage
points of the obtained percentage score" (p. 2), mean overall discrepancy about 4 percentage
points for the 4x4s and almost 8 per cent for the 2x2s, with a noted **systematic
over-prediction** of the average intelligibility (diagonal) scores.

## Access with a fetched url

Full text obtained (OCR of the microfilm scan of the whole bound issue, *Language and Speech*
vol. 3 no. 1, Jan-Mar 1960; the Pollack & Decker article occupies pp. 1-6 of the scan):

- https://archive.org/download/sim_language-and-speech_january-march-1960_3_1/sim_language-and-speech_january-march-1960_3_1_djvu.txt (fetched, 594,721 bytes, complete article text including Table 1, Table 2, both figure captions, conclusions and references)
- https://archive.org/metadata/sim_language-and-speech_january-march-1960_3_1 (fetched; confirms the item is unrestricted and lists a 48.7 MB page-image PDF for re-reading Table 2 by eye)
- https://api.openalex.org/works/doi:10.1177/002383096000300101 (fetched; abstract)
- https://api.unpaywall.org/v2/10.1177/002383096000300101?email=peter.cotton@gsmc.ai (fetched; confirms the SAGE version is closed, no OA copy)

The SAGE page (journals.sagepub.com/doi/10.1177/002383096000300101) returns HTTP 403 to
automated fetches.

## Usability verdict

**Partially sufficient.** The full-menu (8x8) confusion matrices needed to calibrate a
parameter-free Thurstone model are printed in full, at four S/N ratios — four independent
master matrices over eight stimuli, aggregate over the listening crew (aggregate-only; no
per-subject data). The restricted-menu conditions were genuinely run with the response
alternatives cut down to the surviving subset, which is exactly the manipulation the project
needs. **But the observed restricted-menu shares are never tabulated**, so a Thurstone-vs-CRR
scoring on the restricted menus cannot be done from the printed tables alone. Two partial
routes exist: (a) digitise Figure 1 to recover most 4x4 cells (48 cells per S/N ratio minus
omitted points), and (b) score at the coarser grain the authors do print, i.e. compare the
model's predicted mean absolute deviation against the CRR's printed mean absolute deviations
in Table 1 and against the printed "92% within 10 points" statistic. Route (b) is a weak but
real test; route (a) needs the page images.

Classification: **CRR-TEST-BUT-NUMBERS-NOT-PRINTED** (master matrices printed, restricted-menu
matrices not printed).

## Conclusion about CRR quoted verbatim

From the Conclusions, p. 6:

> "The constant-ratio rule of Clarke was examined and was supported for the set of eight
> initial English consonants /f,h,l,r,w,y,hw,+/. Inter-confusion analysis suggests the
> following representational structure for these consonants: /l,r,w,y/ over a wide range of
> S/N ratios ; /f,h,+/ at the most unfavourable S/N ratios ; and a marked asymmetric confusion
> between /w,hw/."

(In the OCR the null-consonant symbol /#/ renders as "+".)

And from the Results section, p. 3, after reporting the poorer 2x2 predictions:

> "Despite the difficulties with the smaller matrices, we interpret the results in support of
> the constant-ratio rule."

Note for the project: the authors' own printed numbers show the failure mode is *systematic,
not noise* — "the large deviations in Table 1 associated with the 2 X 2 matrices were due
primarily to a systematic overestimation of the predicted scores relative to the obtained
scores" (p. 3), and CRR shows "systematic over-prediction of the average intelligibility
scores (filled points) relative to the observed intelligibility scores" (p. 2). That is the
signed direction a Gaussian/Thurstone model would be expected to correct.
