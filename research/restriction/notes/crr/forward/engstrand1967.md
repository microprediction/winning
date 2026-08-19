# Engstrand & Moeller 1967 — Confusion matrix analysis for form perception

## Citation

Engstrand, R. D. and Moeller, G. (1967). "Confusion Matrix Analysis for Form Perception."
*Human Factors* **9(5): 439-446**. doi:10.1177/001872086700900507.
Submarine Medical Research Laboratory, Naval Submarine Medical Center, Naval Submarine Base
New London, Groton, Connecticut.

Two corrections to the record before anything else:

1. **The page range is 439-446, not 475-481.** The running heads in the reprint read
   "440 — October, 1967", "October, 1967 — 441", ... "446 — October, 1967", and the first line
   of the document is "HUMAN FACTORS, 1967, 9 (5), 439-446".
2. **DTIC AD0668614 is not a separate technical report — it is a reprint of the journal
   article**, deposited 1968-01-29 (8 scanned pages, identical text including the Human Factors
   running heads and the journal's own abstract). So there is no longer, more detailed technical
   report behind this paper to go hunting for; the reprint *is* the paper. The fuller data write-up
   is a different, earlier item: Engstrand & Moeller (1962), "The relative legibility of ten
   simple geometric figures", *American Psychologist* 17: 386 — itself only a meeting abstract.
3. **Engstrand & Moeller do not cite Anderson (1959).** Their reference list is Bowen et al.
   1960; Clarke 1957; Clarke 1959; Clarke & Anderson 1957; Engstrand & Moeller 1962; Hodge 1962;
   Hodge & Pollack 1962; Hodge, Crawford & Piercy 1961; Hodge, Piercy & Crawford 1961; Pollack &
   Decker 1960; Shepard 1963. The Anderson (1959) AFCRC TN 58-60 lead comes from Pollack & Decker
   (1960) and Hodge & Pollack (1962), not from here.

## Domain and stimuli

Visual form perception, tachistoscopic single-character recognition. Subjects were U.S. Naval
enlisted candidates for basic submarine school, all with normal or corrected-to-normal acuity.

Stimuli were upper-case English letters in "Leroy lettering guide" style plus 10 simple geometric
figures drawn with a matching template, stroke-width-to-height ratio 1:7, reproduced as high
contrast positive and negative 2" x 2" slides.

Three experiments:

- **Study I, Experiment I** — master set of **36 elements**: the 26 letters of the alphabet plus
  the 10 geometric figures. 12 subjects. Two brightness contrasts (3.6% and 2.2%) x two modes
  (light figures on dark surround, dark figures on light surround); each subject in one mode and
  both contrasts. 100 ms exposure, 7 Ft-L background, 1°30' vertical visual angle. Eight
  responses per brightness contrast per stimulus.
- **Study I, Experiment II** — **20 elements**: the letters **B D H J K L N V W Z** plus the same
  10 geometric figures (Figure 1). 28 subjects. Brightness contrasts 3.2% and 2.5% — i.e.
  *different* task difficulty from Experiment I, deliberately chosen "to be intermediate between
  those for the first" — same two modes.
- **Study II (called "experiment III" in the Discussion)** — 10 subjects, tested individually on
  four consecutive days, all stimuli dark-on-light at a single brightness contrast, exposure
  40 ms, Scientific Prototype Model G tachistoscope. **Condition 20** = the 20-element set;
  **Condition 10** = the 10-element subset of geometric figures only. Day order counterbalanced
  as 10,20,20,10 or 20,10,10,20.

## Master and restricted response sets

There are three master/restricted comparisons in the paper, and they are **not** of equal quality.
Only the third is the clean design the project needs.

**(a) 36 -> 20, across experiments (Study I).** CRR applied to the Experiment I 36-element data to
predict the 20-element matrix, checked against the Experiment II empirical 20-element data. But
Experiment II used **different subjects (28 vs 12) and different brightness contrasts**, so this
is a generalisation-across-conditions test, not a pure menu-restriction test. The authors are
explicit that this was the point: "We were concerned with the ability of the rule to predict the
discriminability of a set of stimuli which were to be presented under conditions different from
those of the original data collection."

**(b) 36 -> 10 and 20 -> 10 (Study I).** Predicted only. The paper states flatly:

> "CRR was also applied to the data of experiments I and II to predict 10 X 10 matrices common to
> both sets of data. **There was no empirical data for the 10 X 10 matrices in these experiments.**"
> (p. 443)

So the two 10-element columns of Table 1 are CRR output compared *against each other*, with no
observed matrix anywhere. This is the exact trap flagged in the assignment: predicted subset
proportions with no observed subset proportions.

**(c) 20 -> 10, within subjects, conditions held constant (Study II).** This one is right:

> "Study II (experiment III) was conducted in part to obtain further evidence on the
> overprediction question. **All experimental conditions were held constant except for the size of
> the stimulus set.** Data were obtained in separate sessions for the 20 element set used in
> Study II and the 10 element subset of geometric figures." (p. 444)

Same 10 subjects, same apparatus, same brightness, same exposure, master menu of 20 and restricted
menu of the 10 geometric figures, run in separate counterbalanced sessions. Trial counts are
generous: in Condition 10 each character appeared **50 times per day**; in Condition 20 each
character appeared **25 times per day**; 500 stimuli per subject per day, 2000 test observations
per subject over four days, **with only the last two days used in the analysis**. Aggregated over
10 subjects that is on the order of 500 observations per figure in the restricted condition and
250 per character in the master condition — ample cell counts.

**One caveat that matters for the project.** Responses were **verbal**, not keyed: "Subjects
responded verbally at any time following the presentation of the stimulus using words of the
international phonetic alphabet for the letters and descriptive names for the geometric figures
(circle, star, zig-zag, etc.)", with a *recognition field* displayed between trials. The menu is
therefore defined by that displayed field, and the paper describes it only once, without
qualifying by condition: "Between trials the subject viewed a recognition field consisting of the
20 possible stimuli" (p. 444). **It is not stated whether the recognition field in Condition 10
showed 10 items or all 20.** If it showed all 20, Condition 10 restricted the *stimulus* set but
not the *response* set, which would make it the wrong kind of restriction for scoring CRR. The
authors plainly treat it as a genuine 10-alternative condition — they renormalise onto the 10 and
call it a prediction — but the text does not settle it. In Study I the corresponding phrasing is
"the subject's task was to recognize items presented from an available listing of stimuli", and
the paper elsewhere notes that CRR "presumes a square matrix... Accordingly subjects would respond
on every trial, know the set of stimuli from which each stimulus is drawn, and limit their
responses to responses associated with items in this set."

Also note the response set was not strictly closed even in the master conditions: "Some subjects
did not respond to all stimuli; these failures to respond were categorized as 'blank' responses",
handled three ways (omit from tabulation; a 36x37 / 20x21 dummy column; and a small side
experiment with blank slides as stimuli). The authors report this "did not unduly affect the
predictive power of the rule."

## What numbers are printed or deposited

**No confusion matrix is printed anywhere in the paper — not the 36x36, not the 20x20, not the
10x10, and neither observed nor derived.** This is weaker than "half a dataset": both halves are
missing at cell level. What is printed:

- **Table 1 (p. 445), "Derived proportions of correct responses for the 10 letters and 10 figures
  common to both experiments."** Forty numbers, all of them **CRR predictions, none observed**:
  for each of the 10 letters (Delta .97/.81, Victor .95/.76, Kilo .92/.71, November .91/.63,
  Whiskey .89/.77, Lima .86/.70, Zulu .86/.67, Bravo .78/.52, Juliett .69/.72, Hotel .64/.56) and
  each of the 10 figures (Circle .98/.94, Triangle .96/.82, Semi-circle .94/.74, Pentagon
  .88/.76, Rectangle .83/.64, Ellipse .76/.72, Arrow .68/.51, Cross .67/.52, Star .66/.52,
  Zig-zag .64/.59), the proportion derived from the 36-element data and the proportion derived
  from the 20-element data. **Diagonal (correct-response) entries only.** Rank-order agreement
  between the two derivations: rho letters = +.617 (p<.05), rho figures = +.918 (p<.01).
- **Figure 2 (p. 442)** — scatter of *observed* 20x20 correct-response proportions (Exp. II)
  against CRR-*derived* values from the 36-element set. Fitted line y = .79x - .02, r = .81, slope
  not significantly different from unity. Diagonal cells only.
- **Figure 3 (p. 444)** — the one figure that carries the clean Study II test: mean percentage of
  responses **obtained** in Study II against mean percentage **predicted by CRR** for the
  geometric figure subset. Fitted line **y = 1.01x**, **r = .99 over all responses**, r = .83 for
  correct responses only (p<.01). "In this case, CRR tended to predict exact proportions of
  responses" (p. 445). Because the correlation is over *all* responses, this scatterplot should
  contain on the order of 100 points — i.e. an unlabelled but complete (predicted, observed)
  point cloud for a 10x10 restricted matrix. Digitising it would recover the *joint distribution*
  of predicted and observed cell values but **not** which cell is which, so it cannot be
  reassembled into a labelled matrix for Thurstone calibration.
- **Figure 4 (p. 444)** — observed-vs-observed reliability check: Exp. II empirical correct
  proportions against Study II empirical correct proportions for the common 20-element master.
  y = .76x + .08, r = .64 (p<.01), slope not significantly different from 1. The authors
  themselves call this reliability "low for measures to be used as predictors."
- **Off-diagonal cells are excluded from the whole of Study I by design:** "the elements lying in
  the triangular matrices above and below the negative diagonal, have been omitted purposefully
  from the analysis presented here. The mass of those points lie near the origin of the scatter
  diagram" (p. 443).
- **A deposit exists but it is the wrong content.** Footnote 2: "Tables of these intercorrelations
  have been deposited with the American Documentation Institute. Order Document No. ADI9671,
  remitting $1.75 for 35-microfilm or $2.50 for 6 by 8 in. photocopies." These are the 153
  Study I comparisons and 306 subset correlations — **tables of correlation coefficients, not
  confusion matrices**. The ADI auxiliary-publication service is long defunct (its deposits passed
  through NAPS and the Library of Congress Photoduplication Service); recovering ADI9671 is a
  long shot and, even if recovered, would yield correlations rather than cell-level counts.

## Access with a fetched url

Full text obtained, complete, free, and OCR-clean:

- https://archive.org/download/DTIC_AD0668614/DTIC_AD0668614_djvu.txt (fetched, 39,066 bytes — the entire 8-page article: abstract, introduction with the CRR formula, Method, Study I, Study II, Discussion, Table 1, all four figure captions, and the full reference list)
- https://archive.org/metadata/DTIC_AD0668614 (fetched; confirms creator "Defense Technical Information Center", date 1968-01-29, imagecount 8, and lists the page-image PDF at https://archive.org/download/DTIC_AD0668614/DTIC_AD0668614.pdf, 547 KB, for re-reading Table 1 and the figures by eye)
- https://api.openalex.org/works?filter=cites:W1993689353 (fetched; this is where the Human Factors journal version, doi 10.1177/001872086700900507, was identified as a forward citation of Clarke & Anderson 1957)

The SAGE journal version (journals.sagepub.com/doi/10.1177/001872086700900507) was not fetched —
SAGE returns HTTP 403 to automated fetches, as it did for Pollack & Decker. It is not needed: the
DTIC reprint is the same text, and the abstract in the DTIC record matches the journal abstract
word for word.

## Usability verdict

**Not sufficient.** The Study II design (item (c) above) is genuinely excellent for the project's
purposes — same subjects, same viewing conditions, a 20-element master menu and a 10-element
restricted menu run in separate counterbalanced sessions, ~50 trials per stimulus per day, and a
CRR fit so tight (y = 1.01x, r = .99) that it is the strongest pro-CRR result in either cluster.
That is precisely the comparison a parameter-free Gaussian model should be scored against. **But
not a single cell of it is printed.** The 20-element master matrix needed for calibration does not
appear in the paper; the observed 10-element matrix does not appear either; and the only Table
in the paper (Table 1) reports CRR *predictions* for a 10x10 for which, in the authors' own words,
"There was no empirical data". The archived deposit contains correlation tables, not matrices.

Downgrade note for the sweep: this was flagged as potentially the second usable source after
Townsend & Landon (1982). It is not — free full text turned out to mean free access to a paper
that never printed its matrices. Its value to the project is as (i) the strongest published
pro-CRR datapoint, useful as a benchmark claim to engage with, and (ii) an unusually clear
statement of *why* CRR mispredicts, quoted below.

Classification: **CRR-TEST-BUT-NUMBERS-NOT-PRINTED**.

Residual leads this paper's reference list opens up, all of them USAF technical reports whose
titles promise exactly this design and which, being technical reports rather than 8-page journal
articles, have room to print matrices — none is in the archive.org DTIC mirror and DTIC itself
serves a WAF block page to automated fetches:

- Hodge, M. H. "The constant-ratio rule and identification tasks." USAF: ESD TDR-62-2, January 1962.
- Hodge, M. H., Crawford, M. J. and Piercy, M. L. "The constant-ratio rule and visual displays." USAF: ESD-TDR-61-56, December 1961.
- Hodge, M. H., Piercy, M. L. and Crawford, M. J. "The constant-ratio rule and lifted weights." USAF: ESD-TN-61-61, June 1961.
- Bowen, H. M., Andreassi, J., Truax, S. and Orlansky, J. "Optimum symbols for radar display." *Human Factors* 1960, 2: 28-33 — an *application* of CRR to symbol selection, so probably not a validation test.

## Conclusion about CRR quoted verbatim

The authors' overall conclusion, final paragraph, p. 446:

> "Within the limits of this study, CRR was found to be of positive value for the prediction of
> response patterns to visual form stimuli which were to be presented under a variety of stimulus
> conditions and viewed by different samples of subjects. The rule accurately predicted exact
> numeric values when experimental conditions were held constant. When experimental conditions
> were allowed to vary, the rule did not accurately predict numeric values, but the use of the
> rule did provide sufficient information to enable the selection of a subset of stimuli from a
> larger set which sufficiently satisfied pre-established criteria and with far less manipulation
> of data than other existing techniques."

On the clean within-condition test, p. 445:

> "All experimental conditions were held constant except for the size of the stimulus set. Data
> were obtained in separate sessions for the 20 element set used in Study II and the 10 element
> subset of geometric figures. Figure 3 shows the mean proportion of responses obtained for each
> of the 10 geometric figures plotted on the proportion of responses predicted by CRR. In this
> case, CRR tended to predict exact proportions of responses."

**The signed failure mode, which is the part most relevant to the project** — p. 445:

> "This apparent tendency of the rule to overpredict, as size of subset (relative to size of
> master matrix) decreases, has been observed by other investigators."

and the authors' arithmetic account of it, p. 446:

> "Clarke (1957) noted that the rule tended to overpredict cells with 'large' initial probability
> estimates and underestimate cells with 'low' initial probability estimates. From basic
> probability theory it can be shown that these tendencies may result from built in bias
> dependent on (1) the number of observations in each row of a submatrix, (2) the number of
> stimuli (rows) in both the master and submatrix, and (3) the shape of the true distribution of
> responses within a row."

> "In the case of estimating the probability of a correct response this factor will result in
> overprediction. In the case of estimating the probability of an incorrect response this factor
> will result in underprediction."

This is worth flagging for the paper: Engstrand & Moeller attribute the systematic overprediction
of diagonal entries to a **small-sample estimation artefact in the denominator** of the
renormalisation, not to a failure of IIA. That is a competing explanation for the same signed
discrepancy that Pollack & Decker (1960) and Hodge & Pollack (1962) report, and any claim that
restricted-menu overprediction is evidence against IIA has to rule it out — which a
parameter-free model fitted to large-count full-menu data, predicting large-count restricted-menu
data, is well placed to do.

Also recorded, p. 440, the authors' summary of the state of play as of 1967:

> "The few formal tests of CRR conducted to date have tended to substantiate its predictive
> usefulness. Clarke and Anderson (1957), Clarke (1957, 1959), Pollack and Decker (1960) and
> Hodge and Pollack (1962) successfully predicted performance on auditory tasks."

Note that this lumps Clarke (1959) in with the successes, which is **wrong** — see the additional
section in `clarke1957.md` on Clarke (1959) reporting a CRR *failure* for tonal displays.
