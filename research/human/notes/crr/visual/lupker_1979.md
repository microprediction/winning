# Lupker (1979) — DOES NOT QUALIFY

## Citation

Lupker, S. J. (1979). On the nature of perceptual information during letter perception.
*Perception & Psychophysics*, 25(4), 303–312.
doi:10.3758/BF03198809
(University of Western Ontario; based on parts of a doctoral dissertation submitted to the
Department of Psychology, University of Wisconsin–Madison.)

Cited by Townsend & Landon (1982) as a study in which the similarity choice model "found
acceptable" fits to alphabetic confusion matrices — not as a CRR test.

## Stimuli and master response set

**Twelve visual characters**, purpose-built to have "little or no featural redundancy",
subtending about 0.2° × 0.3°, drawn on a Tektronix RM 503 oscilloscope (P15 phosphor) at
1.54 m, PDP-8 controlled. Transcribed from Table 1:

- **Four single line features:** `|` (vertical), `-` (horizontal), `/` and `\` (two diagonals)
- **Four letters:** T, L, X, V
- **Four two-feature non-letter characters:** `⊢`, `⊣`, `Λ`, and a differently-slanted `Λ`

Masking paradigm: ~4–5 ms stimulus display, then a 250-ms mask (a grid of hexagons
subtending 0.6° in both dimensions, deliberately much larger than the stimuli "to prevent the
subject from using some sort of relative position information"). Display duration, brightness
and room illumination were adjusted per session and per subject to hold each subject at
about 50% overall accuracy.

Eleven University of Wisconsin undergraduates (3 male, 8 female). Five consecutive days, two
half-hour sessions per day, each session containing two replications of a 192-trial set
(12 stimuli × 8 ISIs × 2). Response was made on a box of four buttons in two stages: first
whether the stimulus was a single feature, then which.

**Master response set = all 12 characters, and it never changes.**

## Restricted response sets (nested, overlapping, or a relabelling)

**NONE. This is why the paper does not qualify.**

The "eight confusion matrices" that make this paper look like a candidate are **eight
interstimulus-interval conditions**, not eight response sets. The ISIs were 10, 20, 30, 40,
50, 75, 100 and 200 ms. All twelve stimuli were presented and all twelve responses were
available in every one of the eight matrices; only the amount of processing time before the
mask differed.

> "Using a masking paradigm, confusion matrices were generated at each of eight interstimulus
> intervals." (abstract)

This is a *discriminability* manipulation, not an *alternative-set* manipulation. It varies
d′ while holding the menu fixed — the exact complement of what the CRR predicts about. It
would be a good dataset for asking whether a Gaussian or Gumbel race better tracks confusions
as sensory evidence is titrated, but it contains no removal of response alternatives and
therefore no CRR test.

The two "subsets" in Table 2 are **not separately collected data**. They are submatrices
lifted out of the master:

> "In order to analyze the other confusions predicted by the global-to-local model (see
> Figure 2), two subsets of the overall matrix were extracted from Table 1 and are listed in
> Table 2."

Renormalising an extracted submatrix of the master and comparing it to itself is vacuous —
it is the CRR prediction, with no independent observation to test it against.

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

Only two tables, and both are **pooled over all 11 subjects and over all 8 ISI conditions,
and both are model-transformed rather than raw.**

- **Table 1 — "Idealized Overall Stimulus Confusion Matrix."** A single 12×12 matrix. Verified
  against a 170-dpi render of article page 309. Values are large integers (diagonal
  919, 1178, 1157, 1192, 967, 1088, 686, 986, 870, 906, 994, 701). **They are not raw
  frequencies:** row sums are not constant (row 1 = 1765, row 2 = 1762), because the entries
  have been passed through Luce's choice model to strip out response bias:

  > "The similarity parameters were next used to estimate what the eight confusion matrices
  > would have been if the subjects had been unbiased."

  The eight per-ISI matrices are **not printed at all**, on the grounds that they look alike:

  > "A visual examination of these eight matrices revealed that the pattern of errors changed
  > very little over lSI conditions. Thus, only the overall confusion matrix is reported in
  > Table 1."

- **Table 2 — "Idealized Error Matrices for Selected Subsets of Stimuli."** Two small
  off-diagonal blocks (a 5-stimulus set `|`,T,L,`⊢`,`⊣` and a 4-stimulus set X,V,Λ,Λ-slanted),
  extracted from Table 1 as stated above.

- Figures 3–5 give "idealized masking functions" — bias-corrected proportion correct against
  ISI, for the three stimulus types and for individual stimuli. Figure 1 shows the stimulus
  set; Figure 2 the model's predicted confusions.

Bias and similarity parameters were estimated per ISI condition using the formulas in the
appendix of Townsend (1971), giving 65 parameters per condition
(*n*(*n*−1)/2 − 1 with *n* = 12 — which is how the stimulus count is confirmed). RMS deviation
between observed and predicted P_ij rose monotonically from .0127 at 10 ms to .0259 at 200 ms.
Chi-square tests were not run "because in every condition a number of expected cell frequencies
were below five."

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

**Wayback-only, full text with embedded text layer, fetched successfully
(1,268,017 bytes, 10 pages):**

https://web.archive.org/web/20180727230501if_/https://link.springer.com/content/pdf/10.3758%2FBF03198809.pdf

Publisher page paywalled at Springer (doi:10.3758/BF03198809).

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**UNUSABLE for a CRR test — excluded on design, not on access.**

Three independent disqualifications:
1. **No restricted response set anywhere in the paper.** The eight matrices are ISI
   conditions over a fixed 12-alternative menu. There is nothing to renormalise *to*.
2. **The one printed matrix is not data.** Table 1 is "idealized" — bias-removed via a fitted
   Luce choice model. Using it to adjudicate between Luce and Thurstone would be circular,
   since a Luce model has already been used to transform it. The raw matrices were not
   published and (1979, pre-supplementary-materials) are not recoverable.
3. **Pooled over 11 subjects and 8 conditions**, so even the transformed matrix carries the
   pooling artefact.

Retained in this directory as a **negative result**: it closes off a lead that looks
promising from the citation record (Townsend & Landon cite it; "eight confusion matrices"
sounds like eight response sets) but is not one.

Residual value elsewhere: the eight per-ISI matrices, had they been printed, would be an
excellent dataset for a *discriminability-sweep* test of Gaussian vs Gumbel — comparing how
each family's implied similarity structure must move as sensory evidence is degraded. Lupker's
own finding that "the pattern of errors changed very little over lSI conditions" while RMS
deviation from the choice model **doubled** across the same range (.0127 → .0259) is a hint
worth remembering, but it cannot be pursued from the published numbers.

## What the authors concluded about CRR, quoted verbatim where possible

**Nothing. The constant-ratio rule is never mentioned.** The phrase does not occur in the
paper, and Clarke, Hodge, Egan, Morgan and Anderson are absent from the reference list.

Lupker's use of Luce is purely as a **bias-removal device**, not as a choice-set theory:

> "The exact nature of confusion matrices is always strongly determined by the biases subjects
> have for guessing certain stimuli. Thus, in order to analyze the perceptual factors involved,
> the effects of these biases must somehow be removed. What appears to be the best way of doing
> this is to appeal to the choice model of Luce (1963; Townsend & Ashby, Note 1)."

His verdict on the choice model is descriptive adequacy only, with an explicit acknowledgement
of the parameter count:

> "This model has [n(n-1)]/2 - 1 parameters (here 65 for each lSI condition) and generally
> provides a very good fit to the data."

> "...these deviation values are quite in line with those obtained for the choice model by
> Townsend (1971), and, thus, the model appears to do a fairly good job of mimicking the data."

Note the word "mimicking" — Lupker treats the Luce model as a curve-fitting convenience, and
the paper's actual argument (feature-accumulation vs "global-to-local" perception) is
orthogonal to the CRR question entirely.
