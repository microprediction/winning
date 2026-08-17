# Townsend & Landon 1982 — CRR and other models of visual letter confusion

## Citation

Townsend, J. T., & Landon, D. E. (1982). An experimental and theoretical investigation
of the constant-ratio rule and other models of visual letter confusion. *Journal of
Mathematical Psychology*, 25(2), 119–162. doi:10.1016/0022-2496(82)90009-8

This is the seed of the forward sweep and it is the single best human source found in
this branch. Full text obtained and read.

## Domain and stimuli

Visual letter identification under brief tachistoscopic exposure (Gerbrands T-2B
two-field tachistoscope), block capitals drawn on a square template so that A, E, F, H
are built from equal-length line segments and X from the two diagonals of the same
square. Pre/post fixation field of four dots; letter subtended about 1 degree. Verbal
naming response, recorded by hand.

Four subjects (labelled D.X., M.X., G.X., A.X.), Purdue undergraduates, 16 daily 1-hour
sessions each. A 4-day calibration/practice period preceded the experiment, tuning each
subject's accuracy into the .55–.65 band; calibration and practice data are excluded.
Data are per-subject, not only aggregate.

Note for the project's standing caution: this stimulus set contains deliberate
near-substitutes. A/E and F/H are each highly confusable pairs, and X is dissimilar to
all four. That is exactly the configuration in which removal can concentrate rather than
spread probability mass, and it is what drives the reported CRR failure (see the
verbatim quotes below).

## Master and restricted response sets

Nested, within-subject, and blocked. Each session consisted of four blocks:

| Block | Set | Size | Relation to master |
|---|---|---|---|
| Master | A, E, F, H, X | 5 | — |
| Subset | A, E, F, H | 4 | nested |
| Subset | A, E, X | 3 | nested |
| Subset | F, H, X | 3 | nested |

Blocks were counterbalanced across sessions so each block appeared four times in each
of the four within-session serial positions. 15 trials per letter per block per session,
225 trials per session. Summed over the 16 sessions, **each letter was presented 240
times in each block in which it occurred**, and 45 times per session summed over blocks.

The restriction is a joint stimulus-set and response-set restriction, in the Clarke 1957
tradition: "The subject was informed of the block presentation order before each session
began, and was reminded of the letters in each block prior to their presentation"
(p. 141). Responses were verbal names, so the response set was restricted by instruction
and by the subject's knowledge of the block, not by a physical response device; the
reported subset matrices are square s x s, so no out-of-set responses are tabulated.
Every restriction is nested — no relabelling — and the same four subjects supply the
master and all three subsets, so calibration and held-out target share subjects.

Townsend & Landon do **not** reuse or extend the Townsend (1971) 26x26 alphabetic
confusion matrix. Townsend 1971a/b are cited as prior work and as the source of the
overlap model, and an earlier study of Townsend's is cited for font similarity, but the
data here are a new 5-letter experiment. Their notation section defines M as "the master
matrix" and S as "a confusion matrix obtained with a subset of the stimuli used in the
master matrix M" (p. 121) — the paper is built around exactly the master/subset
distinction the project needs.

## What numbers are printed or deposited

**Tables 1, 2, 3, 4 (pp. 142–145), one table per subject (D.X., M.X., G.X., A.X.).**
Each table stacks all four matrices for that subject: the 5x5 master (A,E,F,H,X), the
4x4 subset (A,E,F,H), and the two 3x3 subsets (A,E,X and F,H,X). Every confusion cell is
quartered:

- upper left  = **obtained response proportion** (OBT)
- upper right = WSCM predicted proportion (weak similarity choice model, refit per matrix)
- lower left  = **CRR predicted proportion**
- lower right = SSCM predicted proportion (strong SCM, master-estimated parameters carried over)

Proportions are printed to three decimals. With 240 presentations per letter per block
per subject, cell counts are recoverable as round(p x 240) per row; grouped data are out
of 960 per row ("The resultant grouped data represented response proportions out of 960
total for each row in each of the four confusion matrices", p. 147). This is enough to
run exact multinomial or chi-square scoring, not just proportion comparisons.

Also printed: Table 5 (p. 146) non-parametric CRR/SSCM summaries and per-subject
chi-square goodness of fit for the three subset matrices; Table 6 (p. 147) Morgan's
likelihood ratio test of CRR predictions per subject and for the group; Table 7 (p. 149)
chi-square fits for WSCM, overlap, all-or-none and Nakatani confusion-choice; Table 8
SNCC fits; Table 9 percentages of non-parametric measurement tests passed.

**Not printed:** all model parameter estimates. "In order to save space, the parameter
estimates of all of the models fit to the data, via STEPIT, have been omitted. Certain
other numerical results have also been omitted. The parameter estimates and other
numerical results that are discussed but not reported are available from the authors
upon request" (footnote 8, p. 141). This does not matter for the project, because the
project fits nothing — the four obtained-proportion quadrants are the whole input.

Because the printed CRR quadrants are exactly proportional renormalisation of the master
row onto the surviving subset ("each appropriate cell of the master matrix M was
normalized by the sum of the appropriate cells in the same row, thus forming a
prediction entry for the subset matrix S", p. 141), the paper hands over both the
calibration input and the CRR benchmark already computed, as a printed cross-check.

## Access with a fetched url

The publisher DOI is paywalled. Semantic Scholar reports a GREEN open copy at
`http://www.indiana.edu/~psymodel/papers/towlan82.pdf`, which is **dead** — it now
redirects to `https://bloomington.iu.edu/` (verified by following the redirect chain).

Fetched successfully from the Wayback Machine:

    http://web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf

HTTP 200, 2,699,621 bytes, 44 pages, PDF 1.3 produced by "Acrobat 4.0 Capture Plug-in",
carrying a usable OCR text layer (Elsevier's own scan; "PII: 0022-2496(82)90009-8" in
the metadata title). The OCR is good on numerals but mangles some words ("Lute" for
"Luce", "tit" for "fit"), so quoted passages below are normalised for obvious OCR
substitutions only.

Metadata cross-checked at
`https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/0022-2496(82)90009-8?fields=title,abstract,year,authors,openAccessPdf,externalIds,venue`
(fetched; abstract elided by publisher, openAccessPdf status GREEN).

The whole Townsend lab reprint archive is recoverable the same way; the Wayback CDX
index for `indiana.edu/~psymodel/papers/*` lists ~150 PDFs including `towlan83.pdf`,
`towash82.pdf`, `towhueva84.pdf`, `tow71theoretical.pdf`.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES.**

The printed numbers are sufficient, with nothing left to request. A parameter-free
Gaussian map can be calibrated on the printed 5x5 master matrix and scored out of sample
against the printed 4x4 and two 3x3 subset matrices, per subject, with the printed CRR
quadrants serving as the competing prediction and as a check on the renormalisation
arithmetic. Twelve scoreable master-to-subset predictions exist (4 subjects x 3 subsets),
plus a grouped version out of 960 per row. Trial counts are known exactly, so
likelihood-based scoring is available and there is no need to treat proportions as if
they carried unlimited precision.

Two design features raise its value above the rest of this branch: the restrictions are
strictly nested rather than relabellings, and the same subjects contribute master and
subsets, so a failure cannot be blamed on between-group differences.

One caution, from the paper's own diagnosis. The residual is directional and
concentrated: CRR and SSCM underestimate P(E|A), P(A|E) in the A,E,X subset and P(F|H),
P(H|F) in the F,H,X subset, and also underestimate confusions into X. So the A,E,F,H
subset (where nothing near-substitutable was removed) is fit well by CRR, while the two
3-letter subsets (where removal deleted the near-substitute of a survivor) are not. Any
Gaussian-race scoring should be reported per subset, because the aggregate result is a
mixture of one easy case and two hard ones. This is the near-substitute failure mode
flagged in `../README.md`, appearing here with printed numbers on both sides.

## Conclusion about CRR quoted verbatim

The authors' final verdict (p. 158):

> "We conclude that although CRR (or SSCM) may be useful for general rough predictive
> purposes, it fails as an underlying principle of visual confusion; in particular as a
> probable generating seed behind the ratio of strengths form of WSCM. It might be
> mentioned in this context, that the SSCM or CRR predictions are as good or better than
> many predictions that are represented in the experimental literature as supporting a
> theory or model but that are not statistically tested."

Summary conclusion (1), p. 157:

> "The CRR and SSCM model both provided good first approximations in predicting the
> three subset matrices (A, E, X, F, H, X, A, E, F, H) from the master matrix
> (A, E, F, H, X). However, stricter statistical tests showed that, if a high degree of
> prediction precision is required, both the SSCM and the CRR are inadequate. The lack
> of a high degree of predictive precision of the CRR and SSCM appeared to be due to the
> varying set contexts across the four matrix conditions."

(The scanned text renders the master set as "(A, E, F, H, H)" at this point; the final
letter is X throughout the rest of the paper.)

The direction and mechanism of the residual, p. 148 — the most useful passage for this
project:

> "The pattern of results presented in Table 5 suggests that both the CRR and SSCM
> provided a good first, but not second, order approximation in the prediction of the
> A, E, X and F, H, X subset matrices with F, H, X being especially suspect. The CRR and
> SSCM are apparently unable to predict the patterns of the results obtained in the
> A, E, X and F, H, X subset matrices with high accuracy as they both predict that the
> confusions for the missing stimuli (i.e., A, E or F, H) would be evenly distributed
> across the confusions with the remaining stimuli. This follows from the assumption of
> independence from irrelevant alternatives that underlies both models. In the present
> study, this assumption suggests that subtracting letters from the master set should not
> have any effect on the relative recognition probabilities between the remaining
> letters, whether they are presented in the context of the master set or in a subset by
> themselves. This result was obtained with the A, E, F, H subset, and to a moderate
> extent with the A, E, X subset, but not at all with the F, H, X subset."

And immediately following, same page:

> "The confusions among the four letters A, E, F, H in the master matrix tended to
> concentrate into the two letters A, E and F, H in their respective subset matrices
> rather than spreading evenly across all three letters in the subset matrices as
> predicted by the CRR and SSCM. Both the CRR and SSCM tended to underestimate the
> P(E|A) and P(A|E) confusion entries in the A, E, X subset matrix and the P(F|H) and
> P(H|F) confusion entries in the F, H, X subset matrix."

On the statistical tests themselves (p. 147, discussing Table 6):

> "The results are presented in Table 6, and support the non-parametric analyses of the
> CRR (upper half of Table 5) in that the A, E, X and A, E, F, H subset matrices were
> predicted well, but the F, H, X subset matrix was not."

And (p. 147, on Table 5's chi-square half):

> "The x2 values in the lower half of Table 5 indicate that neither the CRR nor the SSCM
> predicted the A, E, X and F, H, X subset matrices with a high degree of accuracy. Both
> the CRR and SSCM fitted excellently, with the exception of D.X., in the A, E, F, H
> subset matrices."

Note the tension the authors leave standing: Morgan's likelihood-ratio test (Table 6)
rejects only F,H,X, while the chi-square test (Table 5, lower half) rejects both 3x3
subsets for all four subjects. Footnote 9 (p. 147) explains why the two statistics need
not agree — the likelihood ratio "in essence forms an estimate of the 'true' subset
matrix using both S and M under the null hypothesis", whereas the chi-square treats the
master-derived CRR entries as fixed theoretical values. The project's out-of-sample
framing matches the chi-square convention, not Morgan's.
