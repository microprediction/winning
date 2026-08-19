# Townsend (1971a, 1971b) — DOES NOT QUALIFY (but is the field's reference master matrix)

## Citation

Townsend, J. T. (1971a). Theoretical analysis of an alphabetic confusion matrix.
*Perception & Psychophysics*, 9(1A), 40–50.

Townsend, J. T. (1971b). Alphabetic confusion: A test of models for individuals.
*Perception & Psychophysics*, 9(6), 449–454.

Both from Purdue University; data gathered at the University of Hawaii under an intramural
research grant. These are the two halves of one dataset — (a) the group-average analysis,
(b) the individual-subject analysis — and are treated together here.

This is the single most-reused visual confusion matrix in the literature. Townsend & Landon
(1982) use it to justify their choice of the F, H, X triple; Lupker (1979) estimates his bias
and similarity parameters "according to the formulas given in the Appendix of Townsend (1971)";
Keren & Baggen (1981) re-analyse it against Gilmore et al. (1979).

## Stimuli and master response set

**The complete 26-letter uppercase English alphabet, simple sans-serif font, tachistoscopic
presentation.** From the 1971a abstract:

> "A study was undertaken to acquire a confusion matrix of the entire upper-case English
> alphabet with a simple nonserifed font under tachistoscopic conditions. This was accomplished
> with two experimental conditions, one with blank poststimulus field and one with noisy
> poststimulus field, for six Ss run 650 trials each."

**1971a:** six subjects, 650 trials each, two conditions — **Condition I** (blank poststimulus
field) and **Condition II** (noisy poststimulus field). Reported as **group averages**.

**1971b:** two new subjects (**M.J.** and **V.F.**, University of Hawaii coeds, paid,
practised and calibrated for 4 days beforehand), Gerbrands T-2B-1 tachistoscope, 5.5 fL for
prestimulus/stimulus/poststimulus fields, letters ½° at the subject's eye, IBM Executive
Directrix typewriter face, presented one at a time over a fixation point from a shuffled deck
of five alphabets. Each subject run for **30 days at 130 trials per day, giving 150
presentations of each letter per subject.** Response was made and then feedback given.
1971b replicates **Condition I only** ("without postceding visual noise").

**Master response set = all 26 letters. It never changes in either paper.**

## Restricted response sets (nested, overlapping, or a relabelling)

**NONE. Neither paper restricts the response set, and this is why the pair does not qualify.**

The two matrices in 1971a differ in **stimulus degradation** (blank vs noisy poststimulus
field), not in the alternative set: all 26 letters are presented and all 26 responses are
available in both conditions. The two matrices in 1971b differ by **subject**. In no
comparison is an alternative removed from the menu.

No relabelling either — the letter names are the response labels.

This is worth stating plainly because the citation record invites the mistake: papers in this
literature say "Townsend (1971) obtained two alphabetic confusion matrices" (Keren & Baggen),
and "two matrices over the same stimuli" sounds like a CRR design. It is not. It is a
d′-manipulation over a fixed 26-alternative menu.

Townsend & Landon (1982) confirm from the inside that no suitable prior visual dataset existed,
including their own earlier work:

> "The present experiment was designed to generate data that could be used explicitly for such
> tests as proposed here, there being little, if any, data extant in the literature suitable for
> the purpose."

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

- **1971a:** two **26×26 group-average** confusion matrices, one per condition. Pooled over the
  six subjects. Also correlation coefficients among model parameters, the MDS solution, and a
  crude physical-similarity measure. The paper's appendix gives the closed-form estimators for
  the choice model's η and β that the rest of the literature subsequently reuses.
- **1971b:** **Tables 1A and 1B — one full 26×26 confusion matrix per individual subject**
  (M.J., V.F.), Condition I only, at 150 presentations per letter. Table 2 gives representative
  model predictions of probability correct and confusions for the letter "P" as stimulus.

Both papers are **scanned images with no embedded text layer**, so the matrices exist only as
bitmap tables at present.

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

**Wayback-only, from the author's own (now dead) Indiana University lab page. Both fetched,
both HTTP 200 `application/pdf`, both verified as complete 6-page scans:**

1971a — https://web.archive.org/web/20040831091706if_/http://www.indiana.edu:80/~psymodel/papers/tow71theoretical.pdf
(1,027,990 bytes, 6 pages, image-only)

1971b — https://web.archive.org/web/20040831091659if_/http://www.indiana.edu:80/~psymodel/papers/tow71al.pdf
(493,982 bytes, 6 pages, image-only)

A third related file from the same directory, apparently a note version, also returns 200:
https://web.archive.org/web/20040831073740if_/http://www.indiana.edu:80/~psymodel/papers/tow71anote.pdf

Publisher versions are paywalled at Springer. Note that `~psymodel/papers/` is a rich seam:
the Wayback CDX index lists ~150 Townsend PDFs there, including `towlan82.pdf`,
`towash82.pdf`, `towlan83.pdf`, `tow72/74/75/76/81/84.pdf`, `ashtow80.pdf`, `ashtow86.pdf`.
CDX query used:
`http://web.archive.org/cdx/search/cdx?url=indiana.edu/~psymodel/papers*&output=text&collapse=urlkey&fl=original,timestamp,statuscode`

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**NOT USABLE for a CRR / restricted-response-set test — excluded on design, not access.**
There is no restricted response set to forecast.

**But usable, after OCR, as a large within-set benchmark**, and it is the natural companion to
Townsend & Landon (1982) since it shares the author, the paradigm and (approximately) the font:

- 1971b gives **two independent 26×26 per-subject matrices at a known 150 trials per row**, so
  proportions convert to counts exactly. That is 2 × 676 = 1,352 cells of individual-level data
  — by far the largest per-subject visual confusion dataset in this directory.
- 1971a's two conditions provide a **discriminability contrast** (blank vs noisy poststimulus
  field) at the group level: a Gaussian and a Gumbel race make different predictions about how
  the whole confusion structure must deform as noise is added, and that is testable without any
  response-set manipulation.

Cost: **needs digitizing.** Both PDFs are image-only, no OCR binary is available in this
environment, and a 26×26 table of 3-decimal proportions is a large vision-transcription job
(676 cells per matrix, four matrices). Two useful validation checks exist: rows must sum to
1.000, and proportions × 150 must be near-integers for 1971b.

Before investing in that, note that Townsend & Landon (1982) already supplies per-subject
master **and** subset matrices at 240 trials/row in a PDF that *does* have a text layer, and
Getty et al. (1979) supplies per-observer raw counts. Townsend (1971) should be queued behind
both.

## What the authors concluded about CRR, quoted verbatim where possible

**Neither paper mentions the constant-ratio rule.** Townsend & Landon (1982) state explicitly
that the CRR had never been tested in visual letter recognition before their own study:

> "It has been employed or tested in other contexts (e.g., auditory recognition; Clarke, 1957)
> but not, to our knowledge, in visual letter recognition (Anderson, Note 3, employed visually
> presented monosyllables)."

What Townsend (1971) concluded was about **within-matrix** model comparison, with the choice
model (Luce) and the overlap-activation model tied and both beating all-or-none. From the
1971a abstract:

> "Briefly, the results were: (1) the finite-state model that assumed stimulus similarity (the
> overlap activation model) and the choice model predicted the confusion-matrix entries about
> equally well in terms of a sum-of-squared deviations criterion and better than the
> all-or-none activation model, which assumed only a perfect perception or random-guessing
> state following a stimulus presentation; (2) the parts of the confusion matrix that fit best
> varied with the particular model..."

And he flags the pooling problem himself, as the explicit motivation for 1971b:

> "One limitation, of course, in the present type of study is that the models purport to
> explain individual behavior. Hence, the work reported here must be viewed as testing the
> ability of the models to handle a large amount of averaged human confusion data but not as
> providing a critical test of the models' assumptions that are meant to apply at the level of
> the individual. An experiment is in progress collecting long-term confusion data at the
> individual level."

The 1971b abstract reports that individual-level fits did not degrade, and that the
**bias** parameters differed between individuals while the **similarity** parameters did not —
the same η/β dissociation that reappears in Townsend & Landon (1982) across response sets, and
that Keren & Baggen (1981) report across studies:

> "Individuals and the group were consistent in their sensory confusions as represented by
> similarity parameters in the choice and overlap models but differed in their response biases.
> A simple measure of physical similarity explained 50% of the variance of the similarity
> structure in the confusion data."

Also recorded there, as a directional asymmetry that a symmetric similarity model cannot
produce:

> "There are some interesting asymmetries in the confusion matrix that may point to
> sensory-bias interactions. For example, both of the present Ss showed more confusions of 'E'
> to 'F' than from 'F' to 'E'."
