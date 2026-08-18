## Citation

Townsend, J. T., & Landon, D. E. (1982). An experimental and theoretical investigation
of the constant-ratio rule and other models of visual letter confusion. *Journal of
Mathematical Psychology*, 25(2), 119-162. DOI 10.1016/0022-2496(82)90009-8.
Authors and pagination confirmed via Crossref.

## Domain and stimuli

Visual letter identification, tachistoscopic. Five block capitals A, E, F, H, X drawn
with equal-length line segments on a square template, X being the two diagonals. Gerbrands
T-2B two-field tachistoscope, letters at ~1 deg visual angle. Four Purdue undergraduates,
daily 1-hour sessions for 16 days, after a 4-day calibration period that stabilised each
subject's accuracy between .55 and .65. Responses spoken aloud as letter names.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)

**Genuinely nested, labels identical throughout** — responses are spoken letter names,
so no rank recoding is possible. Verbatim:

> "Four different sets of the five letters (A, E, F, H, X) were employed as independent
> stimulus sets in the present study. One set served as the master set from which the CRR
> and the SSCM predictions were obtained, and consisted of all five letters. A four letter
> subset (A, E, F, H) was employed... Two three letter subsets, one consisting of the
> letters, A, E, X and the other of the letters F, H, X, were also employed as independent
> stimulus sets."

Master {A,E,F,H,X}; restricted {A,E,F,H}, {A,E,X}, {F,H,X}. Each was a separate block, and
the four block types were counterbalanced across sessions so each appeared equally often in
each within-session serial position. Subjects were reminded of the letters in each block
before it began.

The subsets were chosen deliberately to vary similarity structure: A/H are highly similar
and E/F are highly similar in this font, so {A,E,X} is a deliberately dissimilar subset.
This is the near-substitute case the branch README warns about, here on purpose.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)

**Obtained response proportions, printed, per subject** — Tables 1-4, one table per
subject, each containing all four matrices (5x5 master, 4x4, and two 3x3).

Each confusion cell is quartered: obtained proportion upper-left, WSCM prediction
upper-right, SSCM prediction lower-right, CRR prediction lower-left. A footnote guide
appears beneath each table.

Counts are recoverable exactly. Verbatim: "The letters for each block were randomly
presented for a total of 15 trials per letter per block... When summed over sessions, each
letter appeared 240 times for each block in which it occurred." So every row of every
matrix sums to n=240, and obtained_proportion x 240 must be an integer — which
self-corrects OCR slips.

Model parameter estimates were **not** printed: "In order to save space, the parameter
estimates of all of the models fit to the data, via STEPIT, have been omitted... available
from the authors upon request." Irrelevant for our purpose; the obtained proportions are
what matters.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)

**Wayback-only.** The publisher copy is paywalled (ScienceDirect returns 403 to
automated fetches). Unpaywall/Semantic Scholar list a green OA copy at
`indiana.edu/~psymodel/papers/towlan82.pdf`, but that path is dead — it now redirects to
the Indiana University homepage.

Working URL, fetched, HTTP 200, application/pdf, 2,699,621 bytes, 44 pages:

    https://web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf

`pdftotext -layout` renders Tables 1-4 legibly.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**Needs digitizing.** Four tables x four matrices x quartered cells is roughly 4 x
(25+16+9+9) = 236 cells of obtained proportions, in 1982 OCR. Entirely tractable, and the
x240 integer check makes verification cheap. Highest-value digitizing job in this set,
because it is the only source with a master matrix, three nested restrictions, four
individual subjects, and n=240 per row.

## What the authors concluded, quoted verbatim where possible

This is the paper that states the CRR-approximately-right-with-a-systematic-residual
result most explicitly. Verbatim, from the summary of results:

> "Overall, we can characterize the results with the present class of choice models (and
> the CRR) as establishing that the CRR holds to a reasonable first approximation. Further,
> the SSCM performed about as well as the CRR, even though it is a special case of the CRR.
> ... However, the change in the alphabet in the subsets brought about an alteration in the
> similarity structure of the resultant confusion matrices so that a high degree of
> prediction was possible only with the WSCM, fit as it was to each separate confusion
> matrix."

And from the numbered conclusions:

> "(1) The CRR and SSCM model both provided good first approximations in predicting the
> three subset matrices (A, E, X, F, H, X, A, E, F, H) from the master matrix (A, E, F, H,
> H). However, stricter statistical tests showed that, if a high degree of prediction
> precision is required, both the SSCM and the CRR are inadequate. The lack of a high
> degree of predictive precision of the CRR and SSCM appeared to be due to the varying set
> contexts across the four matrix conditions."

**The residual has a stated direction and shape**, which is the part worth testing a
Gaussian race against:

> "The pattern of results presented in Table 5 suggests that both the CRR and SSCM provided
> a good first, but not second, order approximation in the prediction of the A, E, X and
> F, H, X subset matrices with F, H, X being especially suspect. The CRR and SSCM are
> apparently unable to predict the patterns of the results obtained in the A, E, X and
> F, H, X subset matrices with high accuracy as they both predict that the confusions for
> the missing stimuli (i.e., A, E or F, H) would be evenly distributed across the confusions
> with the remaining stimuli. This follows from the assumption of independence from
> irrelevant alternatives that underlies both models."

> "The confusions among the four letters A, E, F, H in the master matrix tended to
> concentrate into the two letters A, E and F, H in their respective subset matrices rather
> than spreading evenly across all three letters in the subset matrices as predicted by the
> CRR and SSCM. Both the CRR and SSCM tended to underestimate the P(E|A) and P(A|E)
> confusion entries in the A, E, X subset matrix and the P(F|H) and P(H|F) confusion entries
> in the F, H, X subset matrix."

They connect this directly to Debreu:

> "The lack of high precision of prediction in the subset matrices containing a highly
> dissimilar object (here, the letter X) may indicate some relevance of Debreu's (1960)
> criticisms of the choice axiom that were mentioned earlier."

They also flag their own confound honestly: "there is a confounding of changes in the
number of stimuli in the subsets with changes in the letter constituency of those subsets."
The {A,E,F,H} 4x4 was predicted well; the two 3x3s were not, so set size alone does not
explain the failure.
