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

## Whether a Gaussian model was among those tested

Answering a direct question put by another agent in this sweep. Searched the OCR text for
`thurstone`, `gaussian`, `probit`, `discriminal`, `case v`, `normal`, `yellott`,
`double exponential`, `logistic`, and read Definitions 1–8 and Section V in full.

### 1. Is any fitted model Gaussian, normal, Thurstonian or probit?

**No. Not one.** The seven fitted models are all specified directly at the level of
discrete confusion probabilities, with no latent continuum and no noise distribution:

- **CRR** (Definition 1, p. 121) — row-wise renormalisation of the master matrix.
- **SCM / WSCM / SSCM / SSCM(η) / SSCM(β)** (Definitions 2–4, pp. 122–126) — Luce
  ratio-of-strengths, η similarity and β bias parameters.
- **AON**, all-or-none (Definition 5, pp. 126–127) — `P(j|i) = (1 − p_i)h_j` off-diagonal,
  `p_i + (1 − p_i)h_i` on-diagonal; `2q − 1` parameters. "The simplest interpretation of
  the AON is as a strict template matching perceptual process. Upon presentation of the
  stimulus, the subject either recognizes the stimulus perfectly (which occurs with
  probability p_i), or with probability 1 − p_i the subject is thrown into a null
  information state in which he/she can guess the correct stimulus with probability h_j"
  (p. 126). No distribution anywhere.
- **OVLP**, the overlap model (Definition 6, p. 127) — despite the name, **not** a model of
  overlapping distributions as parameterised here. It is a pairwise-confusion-state model:

  > "The OVLP model assumes that either perfect information or two-way partial information
  > is acquired by the subject at each stimulus presentation. With probability ξ_ii stimulus
  > i is recognized perfectly and the correct response is given. With probability 1 − ξ_ii
  > the subject enters some pairwise confusion state in which the subject is unsure as to
  > which of the two letters in the pairwise state was actually presented. ξ_ij represents
  > the probability that the pairwise confusion state is between stimulus i and stimulus j.
  > When in the pairwise confusion state, the subject responds with j according to the ratio
  > of the biases of the two stimuli involved in the pairwise confusion." (p. 127)

  Definition 6 gives `P(j|i) = ξ_ij × [g_j/(g_j + g_i)]` for i ≠ j, with `ξ_ij = ξ_ji`,
  `Σ_j ξ_ij = 1`. The ξ's are free parameters; nothing generates them from a density. The
  paper's own gloss is that OVLP contributes "simple pairwise interstimulus relationships"
  to be "contrasted with the pure guessing strategy of the AON and the stimulus similarity
  relationships of the WSCM and the SSCM" (p. 127).
- **NCC / WNCC / SNCC / SNCC(δ) / SNCC(ω)** (Definition 7, pp. 128–129) — Nakatani's (1972)
  confusion-choice model, specialised. **SNCC = "Strong Nakatani confusion-choice model.
  The NCC version of the SSCM. Both ω and δ parameters are constrained to not vary between
  matrices"** (Appendix A glossary, p. 158). Structurally `Q = AD`, an acceptance matrix
  `A` of probabilities `π_ik = Π_{j∈𝒞k} ω_ij Π_{l∉𝒞k}(1 − ω_il)` giving the chance that
  confusion set `𝒞k` arises from stimulus i, times a decision matrix `D` in which response
  j is drawn from the confusion set by bias ratio δ_j. `[q(q+1)/2] − 1` parameters.

**The Gaussian in this paper belongs to a model they deliberately gutted.** The one place
a normal distribution appears is Nakatani's *original* NCC, and Townsend & Landon state
explicitly that they replaced the Gaussian generative layer with free parameters
(pp. 128–129):

> "A comparison of the above specialized version of the NCC with the original formulation
> (Nakatani, 1972) will reveal that the probabilities ω_ij are being assumed to be free
> parameters that will be estimated. In the original formulation, the ω_ij probabilities
> are defined (not estimated) relative to a Euclidean multidimensional space. This space is
> occupied by sets of multidimensional Gaussian distributions (or spheres), with the center
> point of these spheres representative of a stimulus/response. A boundary lies within each
> of these spheres that defines an acceptance region. Response j is an acceptable response
> given that an observation falls within the acceptance region of the sphere whose center
> point represents the location of stimulus/response j. Therefore, ω_ij represents the
> probability that an observation will fall within the acceptance region of the sphere of
> response j given that stimulus i was presented."

and the reason for discarding it:

> "The free parameters in the original model are the point coordinates in an L-dimensional
> Euclidean space and the bias probabilities, while the specialized version employed here
> begins on a more macroscopic level. This permits estimation and testing to be comparable
> to that of the other models investigated here." (p. 129)

So the only Gaussian-grounded model in the paper's field of view was stripped of its
Gaussian and reduced to free ω's, on grounds of comparability of fitting — and the
resulting parameter count was matched to WSCM and OVLP on purpose. A parameter-free
Gaussian prediction is not among the things tested, and the paper's stated methodology
(equalise parameter counts, fit everything by χ² STEPIT) has no slot for one.

### 2. Which models are master-calibrated-then-applied, and which are refit per matrix?

Confirmed, and this is unambiguous in the text. **Master-calibrated, applied to the
subsets without refitting:**

| Model | What is carried over from the master |
|---|---|
| **CRR** | the master matrix rows themselves, renormalised — nothing estimated |
| **SSCM** | both η (similarity) and β (bias), estimated on M, applied to all S |
| **SNCC** | both ω and δ, estimated on M, applied to all S |
| **SSCM(β)** | η only; β refit per matrix (Definition 4a, p. 125) |
| **SSCM(η)** | β only; η refit per matrix (Definition 4b, p. 125) |
| **SNCC(δ)**, **SNCC(ω)** | the NCC analogues of the two partial cases (glossary, p. 158) |

**Refit separately on every matrix, including each subset:**

| Model | Statement |
|---|---|
| **WSCM** | "The WSCM predictions were of course elicited by completely re-estimating the η_ij's and β_i's for each matrix" (p. 141) |
| **WNCC** | "The ω and δ parameters are unconstrained" (glossary, p. 158) |
| **OVLP** | fit within matrix; Table 7 reports within-set fits |
| **AON** | fit within matrix; Table 7 reports within-set fits |

The decisive sentence, p. 141:

> "SSCM predictions on the other hand, were obtained by estimating the η_ij's and β_j in
> the master matrix and then carrying them over to predict the subset matrices."

and on CRR, same page:

> "Similarly, CRR generated entries for Table 4 by Eq. (1). That is, each appropriate cell
> of the master matrix M was normalized by the sum of the appropriate cells in the same
> row, thus forming a prediction entry for the subset matrix S. Under the stated procedure
> no CRR predicted proportion existed for the master matrix (A, E, F, H, X)."

and the design intent, p. 125:

> "The SSCM expressed by Eq. (5) is a very strong interpretation of Eq. (4) and the CRR in
> the sense that both the η and β parameter estimates in an experiment are constrained to
> be invariant between the master matrix and the predicted subset matrices. In other words,
> only one set of parameters may be estimated from a master confusion matrix containing all
> of the stimuli used in the experiment, and then the appropriate η and β parameter values
> necessary for a specific subset drawn from the master matrix are used to predict the
> subset."

**So state it accurately: the out-of-sample master-to-subset protocol is not novel.** It is
the organising principle of this 1982 paper, formalised in Definitions 3 and 4, executed on
printed matrices, and scored three ways (non-parametric difference counts, χ², and Morgan's
likelihood ratio). Anyone claiming novelty for the *protocol* will be corrected by a
referee holding this paper.

Three qualifications keep the picture honest, and they are where a contribution can
actually sit.

- **Every master-calibrated map here is either CRR itself or a fitted parametric model.**
  SSCM and SNCC are out of sample only in the sense that their parameters were estimated on
  M; they still require an estimation step, and Theorem 1 (p. 122) proves "The SSCM implies
  the CRR, but the CRR does not imply the SSCM", with the consequence drawn on p. 139: "As
  shown by Theorem 1, the SSCM is a special case of the CRR and so should perform no better
  than the CRR." Empirically they came out "nearly equivalent". So the paper contains
  exactly **one** contraction map tested out of sample — proportional renormalisation — in
  two dresses. There is no second, non-IIA, parameter-free map anywhere in it.
- **The paper's winning models are the refit ones**, and its own closing line about them is
  a within-matrix claim: WSCM and WNCC "both of which did a creditable job with the data
  obtained in the present study" (p. 157) — after refitting on each subset. Summary
  conclusion (1) is explicit that the strong versions fail and that "the failure of the
  model to predict accurately the data from the three subset matrices used in this study was
  due primarily to the restrictions placed on its η_ij similarity parameter estimates"
  (p. 157). Freeing the similarity parameters per menu is what rescues the fit — which is
  precisely the move the project declines to make.
- **Nothing here anticipates predicting the residual.** The residual is described (p. 148,
  the concentration of A/E and F/H confusions quoted above) and then attributed to
  "set contexts" and to loosened perceptual filters, not derived from any map.

### 3. Thurstone, Case V, probit, normal, discriminal, Gaussian — what is actually there

| Term | Occurrences in the paper |
|---|---|
| Thurstone / Thurstonian | **once, reference list only** — inside the title of Yellott (1977), p. 162. Never in the body text. |
| Gaussian | twice, both pp. 128–129, both describing Nakatani's original formulation, which they discard (quoted above). |
| normal (as a distribution) | **zero.** The single hit for the string is "normalized" at p. 141. |
| probit | zero |
| discriminal (process) | zero |
| Case V | zero |
| logistic | zero |
| double exponential | eight times, pp. 155–156 and reference list |

**Instead of a Gaussian they built the Gumbel.** Section V, "Indications for models of
visual confusion", pp. 155–156, contains a full random-utility derivation — a race over
latent match strengths — under the *double exponential*, i.e. the one noise distribution
that regenerates Luce's ratio-of-strengths and hence IIA and hence CRR:

> "One of the more attractive candidates is the maximal match choice model (Townsend,
> Note 6; Townsend, Evans, & Hu, Note 7; van Santen & Bamber, 1981, independently arrived
> at this model). The model assumes that the observer selects that alternative pattern which
> exhibits the maximal comparison, or match, with the stimulus pattern. The match process is
> assumed to be random, utilizing the idea of Holman and Marley (cited in Luce & Suppes,
> 1965), and later Yellott (1977), showing how the double exponential distribution can lead
> to the ratio of strengths form (Eq. (4))." (p. 155)

Assumption 4 is the distributional commitment:

> "4. Assume that Y_ij = log X_ij is distributed as the double exponential with scale value
> log η_ij." (p. 155)

They then carry the algebra through to `P(X_ij = max_k X_ik) = η_ij β_j / Σ_k η_ik β_k`,
"which is the formula sought" (p. 156).

**This is the closest the paper comes to a Thurstonian model, and it is the fork taken in
the other direction.** They set up exactly the architecture a Gaussian race would use —
latent match strengths per alternative, a max selector over the available set, bias added in
log space (Figure 3, p. 155) — and then chose the extreme-value noise that makes the whole
thing collapse back to the model they had just shown fails out of sample. The consequence is
visible in the very next paragraph, where the set-size effect has to be patched by hand
rather than falling out of the noise:

> "Within the context of the present study, the maximal match model would need to account
> for the increase in η_ij values as set size decreases. This could be accomplished by a
> shift in the location of the double exponential distribution of the log X_ij towards
> log X_ii (which has scale value 0 from the constraint that η_ii = 1)." (p. 156)

> "One way in which this might occur is if the observer activates a filter or 'template' for
> each potential pattern to be shown. It might take an amount of processing capacity to
> energize and maintain such filters. With a smaller alphabet, the observer could 'loosen up'
> a little and permit a less finely tuned filter for each memory pattern. On the average,
> this would lead to some degradation in the pairwise matching process but performance would
> still improve overall due to the smaller number of patterns." (p. 156)

So the set-size dependence of the similarity parameters — the entire empirical residual — is
handled by a stipulated location shift plus a capacity story about loosening filters. Nobody
asks whether a different noise distribution over the same max-selector architecture would
produce the shift for free. There is no statement anywhere that a Thurstonian or normal
alternative was considered and rejected; the possibility is simply never raised. The nearest
thing to an acknowledged limit is footnote 13, p. 156, which concerns replacing the maximal
selector with decision criteria rather than changing the distribution:

> "It is interesting that the WSCM has no reasonable representation along a continuum as it
> does here if, instead of permitting a maximal selector, it is supposed that a set of
> decision criteria separates the choice alternatives, except when N = 2 (Marley, 1971)."

### 4. Is Yellott (1977) cited?

**Yes, twice.** In text on p. 155, in the passage quoted above, as the authority for deriving
the ratio-of-strengths form from the double exponential. And in the reference list, p. 162,
verbatim:

> "YELLOTT, J. I. The relationship between Luce's choice axiom, Thurstone's theory of
> comparative judgment, and the double exponential distribution. *Journal of Mathematical
> Psychology*, 1977, 15, 109–144."

This is the sharpest point available for a prior-art discussion. Townsend & Landon had, in
hand and cited, the theorem establishing that the double exponential is what distinguishes
Luce from Thurstone in a random-utility race — and they used it in the Luce direction only,
to justify a Gumbel race, in the same paper in which they concluded that Luce's IIA content
"fails as an underlying principle of visual confusion" (p. 158). The Thurstonian half of
Yellott's title is present in their bibliography and absent from their models.
