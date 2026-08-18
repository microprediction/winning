# Townsend & Landon (1982)

## Citation

Townsend, J. T., & Landon, D. E. (1982). An experimental and theoretical investigation
of the constant-ratio rule and other models of visual letter confusion.
*Journal of Mathematical Psychology*, 25(2), 119–162.
doi:10.1016/0022-2496(82)90009-8

## Stimuli and master response set

Five block capital letters **A, E, F, H, X**, hand-drawn in black ink on 5×8 white index
cards with the aid of a square template. A, E, F, H are built from equal-length line
segments (one template side); X is the two diagonals of the same template. Figure 2 of
the paper shows the font.

Tachistoscopic identification (Gerbrands T-2B two-field), pre/post-stimulus fixation
field of four dots at the corners of a square, letter in the centre. Letter subtended
about 1° of visual angle (printed as "1lo", an OCR artifact of 1.1° or 1°). Verbal naming
response. Four subjects (labelled A.X., G.X., M.X., D.X.), Purdue undergraduates, run
individually for 16 daily 1-h sessions after a 4-day calibration/practice period in which
accuracy was stabilised between .55 and .65.

**Master response set = {A, E, F, H, X}** (the full five-letter block). This is a true
closed identification task: within a block the subject was told which letters were
possible and responded only from that set.

## Restricted response sets (nested, overlapping, or a relabelling)

**Nested subsets, each collected as an independent block of trials.** Four stimulus/response
sets were run in separate counterbalanced blocks:

| Set | Letters | Role | Size |
|---|---|---|---|
| M | A, E, F, H, X | master | 5×5 |
| S1 | A, E, F, H | nested 4-subset (drop X) | 4×4 |
| S2 | A, E, X | nested 3-subset | 3×3 |
| S3 | F, H, X | nested 3-subset | 3×3 |

All four sets are subsets of the master, so this is a strictly **nested** design, not a
relabelling. Crucially the subject was *told the block composition in advance* and
reminded of the member letters before each block, so the response set was genuinely
restricted, not merely unused.

Each session contained all four blocks (one each), counterbalanced so that each block
appeared four times in each of the four within-session serial positions. 15 trials per
letter per block per session; 225 trials/session. Summed over the 16 sessions,
**each letter was presented 240 times in each block in which it occurred.** Row counts
are therefore exactly 240 per subject per stimulus per matrix, and proportions × 240
recovers integer counts.

The A,E,X set was chosen deliberately as a *dissimilar* triple (A–H and E–F are the
similar pairs in this font, per Townsend & Ashby), with F,H,X as a control triple
expected to be more mutually confusable. The authors note the design bears on Debreu's
(1960) objection to the choice axiom, "although the sets were not constructed
specifically with Debreu's criticism in mind."

## What numbers are printed (which tables, counts or proportions, per subject or pooled)

- **Tables 1–4: one table per subject** (D.X., M.X., and the other two), giving the
  **obtained and predicted response proportions for all four matrices** (5×5 master plus
  the three subset matrices) stacked along the ordinate. Proportions to 3 decimals,
  **per subject, not pooled.**
  Each confusion cell is quartered:
  - upper-left = **OBT** (obtained proportion)
  - upper-right = **WSCM** (weak similarity choice model, re-estimated per matrix)
  - lower-left = **CRR** (constant-ratio-rule prediction, renormalised master row)
  - lower-right = **SSCM** (strong similarity choice model, parameters carried from master)

  In the master matrix only two numbers appear per cell (OBT upper-left, SSCM lower-right),
  because CRR makes no prediction for the master and WSCM = SSCM there.

  Verified arithmetic: subject D.X. master row A = (.442, .083, .121, .225, .129);
  renormalising over {A,E,F,H} gives .442/.871 = **.507**, which is exactly the printed
  CRR entry for stimulus A in the A,E,F,H matrix. The table is internally consistent and
  the CRR column is fully reconstructible from the master row.

- **Table 5, upper half:** non-parametric CRR/SSCM analysis *combined across subjects* —
  mean proportion correct, mean absolute difference, percentage of |differences| > .10
  (Clarke's 1957 criterion) and > .05, for each of the three subset matrices.
  CRR: mean |diff| = .039 (A,E,X), .051 (F,H,X), .028 (A,E,F,H); % > .10 = 2.8, 11.1, 0.0.
- **Table 5, lower half:** χ² goodness-of-fit **per subject** for CRR and SSCM
  (df = 6, 6, 12). CRR is significant at p<.05 for **all four subjects** in both A,E,X and
  F,H,X; for A,E,F,H only D.X. is significant.
- **Table 6:** Morgan's (1974) likelihood-ratio test for CRR predictions, **per subject
  plus a Group row.** F,H,X fails for A.X., M.X., D.X. and for the Group (40.61, p<.05);
  A,E,F,H never fails; A,E,X fails only for A.X.
- **Table 7:** within-matrix χ² fits for WSCM, overlap, all-or-none, and Nakatani models.
- Parameter estimates (η, β) were **omitted to save space**, "available from the authors
  upon request" (footnote 8) — now effectively lost, but irrelevant since CRR is
  zero-parameter.

## Access (a DIRECT url you have fetched; open, paywalled or Wayback-only)

**Wayback-only, full text, fetched and text-extracted successfully (2,699,621 bytes, PDF 1.3, embedded text layer):**

https://web.archive.org/web/20160916190658if_/http://www.indiana.edu/~psymodel/papers/towlan82.pdf

An earlier snapshot of the same file also returns HTTP 200 application/pdf:
https://web.archive.org/web/20040831082550if_/http://www.indiana.edu:80/~psymodel/papers/towlan82.pdf

Publisher version is paywalled at Elsevier (doi:10.1016/0022-2496(82)90009-8). The
Wayback copy is the author's own posting from his Indiana University lab page
(`~psymodel/papers/`), which is no longer live.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)

**USABLE NOW — this is the single best item in the visual CRR literature.**

Reasons:
- Master matrix and three nested subset matrices, **same subjects, same font, same
  apparatus, interleaved within session** — context is manipulated cleanly and the
  master is not confounded with practice or drift.
- **Per-subject data**, four subjects, so no pooling artefact (the defect Morgan 1974
  blamed for Clarke's and Egan's over-optimistic conclusions).
- Exact trial counts known (**240 per stimulus per matrix**), so proportions convert to
  integer counts and any likelihood-based scoring is exact.
- The printed CRR quadrant provides a built-in check on any re-derivation.
- df structure (6, 6, 12) confirms CRR is being scored as a **zero-parameter** predictor
  of the subsets from the master — precisely the forecasting contest of interest.

Only work needed: OCR clean-up. The extracted text renders many decimal points as commas
(",442" for ".442") and occasionally splits digits (".03 I" for ".031", ".09 1" for ".091",
".I29" for ".129"), so the four tables need transcription against the page images rather
than naive parsing. Four tables × 4 matrices × ~2–4 numbers per cell is a manageable
hand/vision transcription job. Row-sum-to-1 and the CRR-equals-renormalised-master
identity give two independent validation checks.

## What the authors concluded about CRR, quoted verbatim where possible

From the abstract:

> "The strictest interpretation of the SCM (both the similarity and bias parameters
> constrained), shown to be a special case of the CRR, and the CRR produced nearly
> equivalent across-set predictions that provided a reasonable first approximation to the
> data. However, they proved inferior to the least strict SCM (neither the similarity nor
> bias parameters were constrained; the common interpretation of the SCM in visual
> confusion)."

**The systematic residual — this is the passage that matters most.** The authors report
CRR as approximately successful and then describe the residual as a nuisance produced by
IIA:

> "The pattern of results presented in Table 5 suggests that both the CRR and SSCM
> provided a good first, but not second, order approximation in the prediction of the
> A, E, X and F, H, X subset matrices with F, H, X being especially suspect. The CRR and
> SSCM are apparently unable to predict the patterns of the results obtained in the
> A, E ,X and F, H, X subset matrices with high accuracy as they both predict that the
> confusions for the missing stimuli (i.e., A, E or F, H) would be evenly distributed
> across the confusions with the remaining stimuli. This follows from the assumption of
> independence from irrelevant alternatives that underlies both models."

And the direction of the residual, stated explicitly:

> "The confusions among the four letters A, E, F, H in the master matrix tended to
> concentrate into the two letters A, E and F, H in their respective subset matrices
> rather than spreading evenly across all three letters in the subset matrices as
> predicted by the CRR and SSCM. Both the CRR and SSCM tended to underestimate the
> P(E | A) and P(A | E) confusion entries in the A, E, X subset matrix and the P(F | H)
> and P(H | F) confusion entries in the F, H, X subset matrix. That the CRR and SSCM also
> underestimated the P(j | X) (for j = A, E, F, H) entries in the A, E, X and F, H, X
> subset matrices suggests that the letter X was quite dissimilar to all of the letters
> A, E, F, H in this study."

The residual is located in the **similarity** structure, not the bias structure — a
scale/variance effect, exactly where a Gaussian race would put it:

> "These findings indicate that the change in the alphabet from the master set to the
> subsets is disturbing both the sensory-similarity structure and the bias structure of
> the confusion matrices, but the sensory-similarity structure more so than the bias
> structure. An examination of the estimated η's for the WSCM revealed that similarity
> (as measured by η) increased an average of 69% of the time (within each subject, 12
> comparisons were possible) when descending from the master to the subset matrices. The
> β's, on the other hand, showed no systematic variation."

On the pooling question that had exercised Morgan (1974):

> "These results suggest that Morgan's (1974) rejection of the CRR might have been due to
> the use of grouped data. However, in the present study the grouped data for the A, E, X
> and A, E, F, H subset matrices were predicted fairly well according to the likelihood
> ratio test. The discrepancies between these results and those of Morgan (1974) could be
> due to differences in modality (Morgan's data were derived from auditory confusions)
> and design."

On the Debreu / similarity-of-alternatives reading of the failure:

> "The lack of high precision of prediction in the subset matrices containing a highly
> dissimilar object (here, the letter X) may indicate some relevance of Debreu's (1960)
> criticisms of the choice axiom that were mentioned earlier."

Note on framing: the authors state that no prior visual dataset existed for this purpose —
"there being little, if any, data extant in the literature suitable for the purpose" — and
that they had not set out to break the rule: "The experiment is not simply an attempt to
determine if a visual stimulus set can be discovered that fails to satisfy CRR." They also
record that the CRR had never before been tested in visual *letter* recognition:
"It has been employed or tested in other contexts (e.g., auditory recognition; Clarke,
1957) but not, to our knowledge, in visual letter recognition (Anderson, Note 3, employed
visually presented monosyllables)."

**Nowhere is a Gaussian/Thurstonian race fitted.** The comparison set is CRR, four
similarity-choice-model variants, Townsend's overlap model, the all-or-none model, and a
modified Nakatani confusion-choice model — all Luce-family or finite-state, none
Gumbel-vs-Gaussian.
