# Prior art: has anyone already done this?

**The question.** Has anyone calibrated a parameter-free Gaussian model to full-menu
choice shares and scored its restricted-menu predictions against proportional
renormalization, out of sample, with nothing fitted to the restricted menu?

**The answer, in one line.** No — but the claim is much more crowded than the project's
current framing admits, and three of the four component ideas are already in print.
Nobody has run the specific out-of-sample forecast; somebody has already published the
conclusion it is meant to support.

Read the "What is already published" section before circulating anything. The sentence
"the literature did not score renormalization against a parameter-free Gaussian race out
of sample" in `../README.md` is defensible as written, but only because of the words
*parameter-free* and *out of sample*. Drop either and it becomes false.

---

## Verdict on true prior art

No work was found that does all four of the following together:

1. a Gaussian / Thurstonian / probit choice map,
2. calibrated on full-menu shares **only**,
3. used to predict restricted-menu shares **out of sample**, nothing fitted to the target,
4. scored against proportional renormalization as the competing forecast.

The nearest miss — Wills et al. 2000 — asserts (1) as its *conclusion* and owns the
theoretical framing, but its actual model has rectangular noise rather than Gaussian, four
free parameters, and a decision threshold refitted per menu; and it never computes the
renormalization benchmark at all. So it fails (1), (2), (3) and (4) as executed, while
publishing the hypothesis. That combination is the single most important thing in this
file.

---

## What is already published, and what it costs the project

### The theory has been settled since 1977 — no novelty available here

Yellott, J. I., Jr. (1977). "The relationship between Luce's Choice Axiom, Thurstone's
Theory of Comparative Judgment, and the double exponential distribution." *Journal of
Mathematical Psychology* 15:109–144. doi:10.1016/0022-2496(77)90026-8

Fetched in full and read: https://escholarship.org/content/qt7z91732x/qt7z91732x.pdf?t=nvyhsf

Purely theoretical; no data, no experiment of his own ("complete choice experiment" is a
formal object in his notation, not something he ran). He proves that the double
exponential (Gumbel) is the **unique** discriminal-process distribution making a
Thurstone model equivalent to Luce's Choice Axiom once triples are admitted — for pairs
alone the representation is not unique — and derives Gumbel, and hence the Choice Axiom,
from "invariance under uniform expansions of the choice set."

Consequence for the project: *that a Gaussian Case V model must violate the constant-ratio
rule on menus of three or more is a theorem, not a finding.* It has been known for
forty-nine years. Yellott also records the earlier steps: Adams & Messick (1957) showed
the logistic is the unique difference distribution equivalent to the Choice Axiom for
pairs; Holman & Marley (in Luce & Suppes 1965) showed Gumbel suffices for any menu size;
Luce (1959) already printed a table showing Choice Axiom and Thurstone Case V predictions
are "virtually identical" for pair comparisons. The project's contribution cannot be the
existence of a Gaussian/Gumbel divergence, only its measured size and direction on real
restricted menus.

### The master-to-subset out-of-sample protocol is not novel

Townsend, J. T., & Landon, D. E. (1982), *Journal of Mathematical Psychology* 25:119–162.
See `townsend1982.md` for the full reading, including the recovered Wayback copy.

Their organising distinction is exactly the project's: M is "the master matrix", S is "a
confusion matrix obtained with a subset of the stimuli used in the master matrix M"
(p. 121). Models split into **master-calibrated-and-applied-without-refitting** (CRR,
SSCM, SNCC, and partial-constraint variants) versus **refit per matrix** (WSCM, WNCC,
OVLP, AON). So calibrate-on-master, predict-subset, score-out-of-sample was standard
practice in 1982, and Clarke (1957) did a cruder version in 1957.

What they did **not** do is the Gaussian arm. Confirmed against the full text:
"Thurstone" appears exactly once in the paper, in the reference list, inside Yellott's
title; *normal distribution*, *probit*, *Case V* and *discriminal* appear zero times.
Where Nakatani's original confusion-choice model defined its terms over "sets of
multidimensional Gaussian distributions", Townsend & Landon explicitly **strip the
Gaussian out** and replace it with free parameters "on a more macroscopic level"
(pp. 128–129) so that parameter counts match the rival models. And in Section V
(pp. 155–156) they build a random-utility race over latent match strengths with a max
selector under the **double exponential** — the one noise law that regenerates Luce — then
patch the set-size effect by hand as "a shift in the location of the double exponential
distribution." A normal alternative is never raised, let alone tested and rejected.

They cite Yellott (1977) twice, in text at p. 155 and in the reference list at p. 162,
full title included. So they held the theorem that separates Luce from Thurstone in a
random-utility race, used it in the Luce direction only, in the same paper whose
conclusion was that CRR "fails as an underlying principle of visual confusion" (p. 158).

**This is the real gap the project fills: not the protocol, and not the negative result,
but the untried map.** That is a narrower and more defensible claim than "nobody scored
these out of sample."

### The negative result — CRR fails on human restricted menus — is already in print, twice, prominently

Rouder, J. N. (2004). "Modeling the effects of choice-set size on the processing of
letters and words." *Psychological Review* 111(1):80–93. doi:10.1037/0033-295X.111.1.80
See `rouder2004.md`.

Rouder tests CRR on human restricted menus and rejects it, on two independent letter data
sets — his own 2001 six-choice-to-two-choice data (21 of 28 log-odds points off the
diagonal) and a reanalysis of Townsend & Landon's matrices (39 of 48) — with a **consistent
signed residual: restricted-menu performance is worse than CRR predicts.** He states the
rejection reaches Luce's similarity choice model, Keren & Baggen's letter recognition
model, and Massaro & Oden's FLMP, because CRR is their common decision rule; and that the
opposite-direction failure afflicts guess-only models such as McClelland & Rumelhart's
interactive activation model. The true contraction therefore sits strictly between
renormalization and guess-only.

Consequence: any claim that renormalization fails on real restricted response sets **must
cite Rouder 2004**, and cannot be presented as new. What Rouder does not do is propose a
Gaussian race; he reads the residual as a fact about people ("participants are somewhat
efficient in their conditioning on choice-set restrictions", p. 92), not about the
contraction map, and offers no calibrated competitor.

### The strongest near-miss: the conclusion the project wants has already been asserted

Wills, A. J., Reimers, S., Stewart, N., Suret, M., & McLaren, I. P. L. (2000). "Tests of
the ratio rule in categorization." *Quarterly Journal of Experimental Psychology A*
53A(4):983–1011. doi:10.1080/713755935
Fetched: https://www.andywills.info/assets/pdf/2000Wills.pdf
See `wills2000.md`, including the live trial-level data deposit.

Human subjects, genuine full-menu-versus-restricted-menu manipulation (three-choice versus
the same stimuli with one response disallowed by the experimenter), and this conclusion at
p. 1008:

> "Our central conclusion is that the ratio rule is an inappropriate theory of categorical
> decision and should be replaced by a system based on the principles of Thurstonian
> choice."

with the authors' own restatement on the same page:

> "the Case V double exponential Thurstonian choice process is an inappropriate model of
> categorical decision, but other Thurstonian choice processes are potentially
> appropriate"

and from the abstract:

> "The central feature underlying the success of this model is the assumption that
> categorical decisions are based on a Thurstonian choice process (Thurstone, 1927, Case V)
> whose noise distribution is not double exponential in form."

**This is prior art on the framing and on the conclusion.** The project cannot claim to be
the first to propose replacing the ratio rule with a Thurstonian choice process on the
evidence of restricted-menu human data. Wills et al. published exactly that in 2000 and
put it in their abstract. They also already own the Yellott framing, stated precisely at
p. 985 including the n ≥ 3 restriction: "Yellott (1977) proved for situations involving
three or more choices that the predictions of Thurstone's theory and the ratio rule can be
equivalent if and only if the distributions employed in Thurstone's theory are double
exponential."

Why it is nevertheless **not** prior art on the method. Four findings from the full text,
each independently sufficient:

1. **The noise is rectangular, not Gaussian.** The winner-take-all model that carries the
   paper's argument uses uniform noise: the noise added to ν_i "ranges from +N to −N, has a
   mean of zero, and has a **rectangular distribution** (i.e., all values from +N to −N are
   equally likely)" (p. 1002). Gaussian appears twice and neither is their model — once in
   the Introduction as a *definition* of Thurstone's theory (p. 985), and once as a
   one-sentence unillustrated aside that Gaussian noise "produces comparable results" for
   the stripped-down simple-WTA model (p. 1005) — a model they describe as fitting worse
   and which "does not correctly predict the trend in the q′ statistic." Footnote 3 (p. 1009)
   even detaches the two: "Considering the ratio rule in this way assumes that Gaussian
   distributions are not a defining property of Thurstone's theory."
2. **Four free parameters, self-described.** "The WTA model is a relatively complex system
   with four free parameters (E, D, N, and S). The ratio rule, in contrast, has no free
   parameters" (p. 1005).
3. **Something *is* fitted to the restricted menu.** The decision threshold S is set per
   condition — "S is set to 0.18 for the two-choice condition, 0.65 for the three-choice
   condition, and 0.72 for the novel-elements condition" (p. 1002) — and the authors lean on
   exactly that freedom to rescue the q′ trend (pp. 1005–1006). So the two-choice account is
   a simulation with a menu-specific parameter, not an out-of-sample forecast. The magnitude
   terms are likewise assumed linear from a chosen learning rate, not inverted from observed
   full-menu shares. None of "parameter-free", "out of sample", "cross-validation" or
   "held-out" occurs anywhere in the paper (0 hits each).
4. **The renormalization benchmark is never computed.** The whole test rides on the derived
   statistic q = [P(B|B,C) − P(B|A,B,C)] / P(B|A,B,C) and a *qualitative* same-direction /
   opposite-curvature prediction, assessed by fitted quadratics and F-tests. No discrepancy
   statistic against a renormalized prediction appears anywhere. Their own Equations 3 and 6
   imply the exact parameter-free identity q = P(A|A,B,C) / [1 − P(A|A,B,C)], computable
   from the three-choice condition alone — and they never form or score it.

The menu is also only 3 → 2, the smallest size at which Yellott's non-equivalence bites at
all, the manipulation is between-subjects, and the analysis is on aggregate data (a
limitation they flag themselves at p. 1006).

Practical consequence, and it is a favourable one. The project's novelty must be stated as
*map plus method*: the Gaussian arm specifically, calibrated parameter-free by inverting
observed full-menu shares, forecast out of sample, and scored against renormalization. All
four of those are absent from the one paper that reached the same conclusion. But the
*hypothesis* must be credited to Wills et al. (2000), and the Yellott framing to Yellott
(1977). Note also that the unscored identity above can be evaluated directly from Wills'
live CAM1 deposit, which makes it a cheap and rhetorically strong first result: closing a
gap the original authors left open in their own data.

### The sign of the residual is not consistent across the literature

`../README.md` and Rouder 2004 both invite the assumption that the residual has one
direction. Across the sweep it does not, and a Gaussian race predicts a *specific* sign, so
this matters before any claim is made:

- **CRR over-predicts accuracy on the restricted menu** (observed worse than predicted):
  Rouder 2004 on both his 2001 data and Townsend & Landon's (21/28 and 39/48 log-odds points
  off the diagonal); Engstrand & Moeller 1967, systematic overprediction of diagonal entries;
  Hodge & Pollack 1962 for 2×2 menus drawn from adjacent objects in widely spaced ensembles.
- **CRR under-predicts accuracy on the restricted menu** (observed better than predicted):
  Pollack, Rubenstein & Horowitz 1960 — obtained correct exceeds predicted at every one of
  the 16→8→4→2 steps, signed diagonal bias +3.3 / +3.9 / +0.8 per cent; Clarke & Anderson
  1957 report obtained-minus-predicted articulation scores of +1.0 and +4.2 points.

Townsend & Landon's own diagnosis is subtler still and is the most useful of the three: the
residual is not a uniform accuracy shift but a *concentration* — "confusions among the four
letters A, E, F, H in the master matrix tended to concentrate into the two letters A, E and
F, H in their respective subset matrices rather than spreading evenly across all three
letters in the subset matrices as predicted by the CRR and SSCM" (p. 148). CRR fits the
subset where nothing near-substitutable was removed and fails the two where removal deleted
a survivor's near-substitute.

Implication: the claim should be framed as *structure*, not sign — a Gaussian race predicts
where mass moves as a function of the similarity geometry, and should be scored per subset
rather than pooled. Reporting a single aggregate improvement over CRR would mix cases that
the theory says must differ.

### A confound the old literature cannot resolve, and the project should not inherit

Three *different* nuisance explanations for restricted-menu discrepancies are on the record,
each offered by the authors who found the discrepancy:

- **denominator bias** in the renormalization arithmetic, depending on observations per row
  and the master/submatrix size ratio — Engstrand & Moeller 1967 (`engstrand1967.md`);
- **response-label confusion** rather than stimulus confusion — Hodge & Pollack 1962,
  pp. 138–139 (`hodge1962.md`);
- **practice** — Pollack, Rubenstein & Horowitz 1960 (`pollack1960b.md`).

None of the three is separable from a genuine IIA violation using the published *summary*
statistics alone. That is an independent argument for scoring only sources with cell-level
numbers (Townsend & Landon 1982; the Wills CAM1 deposit), and for generating fresh data
rather than resting the claim on mined summaries.

A related warning about the citation record, documented in `clarke1957.md`: Clarke (1959)
reported CRR **failing** for tonal displays, that failure was attributed by Clarke himself
to "strong contextual effects present in his data" (surviving only as Hodge & Pollack's
footnote 5, "personal communication, 1960"), the data were never published, and Engstrand &
Moeller (1967, p. 440) then list "Clarke (1957, 1959)" among studies that "successfully
predicted performance on auditory tasks." The one negative test by the rule's own author was
silently converted into a success. The apparent historical support for CRR is therefore
weaker than a citation count suggests — which cuts in the project's favour, but must be
argued from the primary sources rather than asserted.

### Gaussian versus Gumbel has been run as a fit comparison, but only on binary choice

Kornbrot, D. E., Georgiou, G. J., & Page, M. (2018). "Choice of choice models: Theory of
signal detectability outperforms Bradley-Terry-Luce choice model." PeerJ Preprints.
doi:10.7287/peerj.preprints.26978v1

Verified at abstract level via a fetched API record:
https://api.semanticscholar.org/graph/v1/paper/DOI:10.7287/peerj.preprints.26978v1?fields=title,abstract,year,authors,venue,openAccessPdf,externalIds,citationCount
Full text **not obtained** — peerj.com returned HTTP 403 to both WebFetch and a
browser-User-Agent curl, so the characterisation below rests on the abstract only.

They compare a probit link (TSD, "based on a normal distribution/probit function") against
a logit link (choice-model theory, Luce 1959; Link 1975) using Generalized Linear Mixed
Models across eighteen psychology data sets, and report that probit "was shown to have a
better Bayesian Goodness of Fit than the logit function for every one of eighteen diverse
psychology data sets."

So the answer to "has anyone framed this as a Gaussian-versus-Gumbel comparison" is **yes**
— and the project should expect a reviewer to raise it. But it is category (B): both models
are *fitted*, the criterion is goodness of fit rather than out-of-sample forecast, and
critically it is **two-choice decision-making throughout**, on "binary proportions." For a
fixed binary menu the constant-ratio rule has no content at all — nothing is removed — and
by Yellott's theorem probit and logit are near-indistinguishable for pairs anyway, which is
exactly why Luce's 1959 table showed them "virtually identical." Their result is a
different and weaker claim than the project's, and saying so crisply is the right response.

---

## Categorised near-misses

Categories: **(A)** true prior art; **(B)** fits both maps, or fits anything to the
restricted menu; **(C)** theory only, no empirical forecast; **(D)** menu/IIA testing
without the Gaussian-versus-renormalization forecast framing; **(E)** irrelevant.

| Work | Cat | Fetched url | Why this category |
|---|---|---|---|
| Yellott 1977, JMP 15:109–144 | C | escholarship.org/content/qt7z91732x/qt7z91732x.pdf | Proves Gumbel uniqueness for triples; no data at all. The divergence is a theorem. |
| Wills et al. 2000, QJEP 53A:983 | B/D, closest miss | andywills.info/assets/pdf/2000Wills.pdf | Asserts the Thurstonian conclusion on human 3→2 restricted-menu data and owns the Yellott framing, but the model has **rectangular** noise, four free parameters, and threshold S refitted per menu (0.18 two-choice vs 0.65 three-choice); renormalization is never scored. |
| Rouder 2004, Psych Rev 111:80 | D | see `rouder2004.md` | Rejects CRR out of sample on human restricted menus with a signed residual; proposes no Gaussian competitor. Prior art on the negative result. |
| Townsend & Landon 1982, JMP 25:119 | D | Wayback copy, see `townsend1982.md` | Runs the master→subset out-of-sample protocol with Luce-family maps only; builds the Gumbel race explicitly; never raises a normal alternative. |
| Kornbrot, Georgiou & Page 2018, PeerJ preprint | B | S2 API record (full text 403) | Real probit-vs-logit comparison, but binary choice only and goodness-of-fit, not menu-restriction forecasting. |
| Ashby, Lee & Balakrishnan 1992, Math Soc Sci 23 | B (title-level only) | S2 API record; abstract elided | "Comparing the biased choice model and multidimensional decision bound models of identification" — Gaussian decision-bound versus Luce on identification data, but both fitted, and no restricted-menu arm found. **Not verified beyond the title; see unresolved leads.** |
| del Castillo 2020, Transportmetrica A | B/E | S2 API record | Gumbel-marginal copula RUM fitted to interurban trip data, compared to nested logit. All Gumbel-family, all fitted. |
| Dagsvik 1994 *Econometrica*; Dagsvik 2014 *Theory and Decision* | C | Crossref records | "What independent random utility representations are equivalent to the IIA assumption?" — characterisation theorems, no forecast. |
| Marley 1982, Math Soc Sci | C | Crossref record; abstract elided | Whether all choice probabilities are functions of the binary ones — the theoretical relative of the question, no data. |
| Chakrabarti 1969, *Econometrica* | C | Crossref record | Note on the relation between binary and multiple choice probabilities. |
| Iverson 1979, JMP | C | Crossref record | Conditions for Thurstone Case III representations of binary choice probabilities. |
| Hausman & McFadden 1984, *Econometrica* 52 | D | Crossref record, doi 10.2307/1910997 | The canonical omitted-alternative IIA specification test. A test statistic, not a forecasting comparison, and no Gaussian competitor is scored. |
| Seshadri & Ugander 2019, ACM EC | D | S2 record | Statistical detectability limits for IIA violations. |
| Gensch & Ghose 1997, *Omega* | D, unresolved | S2 API record; abstract elided | Title promises "single pair vs full choice set" IIA contrast; could not establish whether held-out subset shares are predicted. See unresolved leads. |
| Elliffe & Davison 2009/2010, *Behav Proc* 84:381 | E for this purpose | see `elliffe2009.md` | Title says CRR is violated, but **no alternative is ever removed** — four keys throughout. Operant within-menu ratio invariance, not Luce deletion. |
| Bensemann et al. 2015, *JEAB* 104:7 | E for this purpose | see `bensemann2015.md` | Same fixed four-key design, same six birds, opposite conclusion; the disagreement is procedural. |
| Nosofsky 1985–1991 series | B | see OpenAlex enumeration | Fits exemplar/similarity-choice models to identification and categorization data. Fitting a Thurstone-like or biased-choice model *to* the data is the near-miss the project must disclaim. |
| Brusco & Stahl 2001; Brusco & Steinley 2006; Theise 1989 | E | see agent notes | Integer-programming subset *extraction* from a master matrix. They compute predicted subset performance but never collect restricted-set data to validate it. |
| Kujala, Richardson & Lyytinen 2008, JMP | B | see agent notes | Estimates a full confusability matrix *from* adaptively chosen subset trials — the inverse of the project's direction. |

---

## Unresolved leads — the honest gaps in this negative

Each of these could in principle overturn the verdict. None could be read in this pass, and
the reason is recorded so the next pass starts here rather than repeating the search.

1. **Takane, Y., & Shibayama, T. (1992). "Structures in stimulus identification data", in
   F. G. Ashby (Ed.), *Multidimensional Models of Perception and Cognition*, pp. 335–362,
   Erlbaum.** Rouder 2004 cites this as having "provided more stringent statistical tests
   of the constant ratio rule" and rejected it. **This is the single most important
   unresolved item**: a book chapter, so no DOI; Crossref returns nothing; archive.org has
   no copy of the Ashby volume (advanced-search query returned numFound 0). Likely
   reanalyses Townsend & Landon's matrices. Needs a library.
2. **Ashby, F. G., Lee, W. W., & Balakrishnan, J. (1992). "Comparing the biased choice model
   and multidimensional decision bound models of identification", *Mathematical Social
   Sciences* 23:199–219 (doi 10.1016/0165-4896(92)90016-x).** Categorised (B) on the title
   alone; the abstract is elided by the publisher and the full text was not obtained. This
   is the closest *psychology* analogue of a Gaussian-versus-Luce comparison on
   identification data and it should be read properly before publication.
3. **Smith, J. E. K. (1992). "Alternative biased choice models", *Mathematical Social
   Sciences* 23:199–219 area (doi 10.1016/0165-4896(92)90017-y).** Cited by Rouder as the
   competitor model in the same territory. Abstract elided.
4. **Gensch, D., & Ghose, S. (1997), *Omega* 25(3) (doi 10.1016/s0305-0483(96)00047-3).**
   "Differences in independence of irrelevant alternatives at individual vs aggregate
   levels, and at single pair vs full choice set." The title is the closest match in the
   marketing literature to a menu-size contrast; abstract elided, full text not obtained.
5. **Morgan, B. J. T. (1974). "On Luce's choice axiom", JMP 11:107–123.** One of the
   project's four seed works. Paywalled at ScienceDirect (403), no open copy, abstract
   elided. Its content was never verified in this sweep. Morgan (1972) separately compared
   normal and logistic models for ROC fitting, so Morgan is the one seed author who
   plausibly touched the Gaussian arm.
6. **The marketing "attraction model" thread** (Bell, Keeney & Little 1975 market-share
   theorem; Cooper & Nakanishi) — whether anyone there scores subset-share predictions
   against proportional renormalization out of sample.
7. **The m-alternative-forced-choice thread.** Hacker & Ratcliff (1979) "A revised table of
   d′ for M-alternative forced choice", *Perception & Psychophysics* (doi
   10.3758/bf03208311), and Clarke (1959) "Proportion of correct responses as a function of
   the number of stimulus-response alternatives", JASA (doi 10.1121/1.1930396). This
   literature *does* predict accuracy across menu sizes under Gaussian assumptions and
   compare it to Luce-type predictions — a genuine Gaussian-versus-Gumbel menu-size
   forecast. It predicts only the **diagonal** (percent correct), never the distribution of
   errors across survivors, so it cannot be full prior art; but it is the oldest honest
   ancestor of the project's comparison and deserves a paragraph rather than silence.

---

## What was actually searched

So that the negative can be weighed. Two hard constraints applied: the OpenAlex daily API
budget was exhausted partway through (after the citation enumeration completed, which was
the part that mattered), and the session-wide WebSearch budget of 200 calls was fully
consumed, so all later work used WebFetch plus direct API calls only. Publisher sites
routinely returned 403: AIP, Elsevier/ScienceDirect, SAGE, Taylor & Francis, OUP, ASHA,
Karger, Springer link, APA, PeerJ.

**Citation enumeration (complete, saved to disk).** Forward citations pulled in full from
OpenAlex for Clarke 1957 (W2033675284, 102 citers), Townsend & Landon 1982 (W2020127675,
57), Morgan 1974 (W2019859807, 23) — 156 unique works. Same three seeds via Semantic
Scholar (110 / 58 / 14, 163 unique), yielding 32 records OpenAlex lacked, notably Conrad
(1964). Luce 1959 (W3015812362, 2626 citers) was **not** enumerated wholesale; instead it
was intersected with full-text searches for "constant ratio rule", "constant-ratio rule",
"restricted response set" and "response set size", which is what surfaced the operant/JEAB
vein that cites Luce but not Clarke. Yellott 1977 citations enumerated via Semantic Scholar
(300 works), then filtered by title and by abstract keyword — that set contains no
empirical Gaussian-versus-renormalization forecast.

**Query threads run** (Semantic Scholar search, Crossref `query.bibliographic`, OpenAlex
`search=` until budget exhaustion, arXiv API):

- constant ratio rule × Thurstone / normal model / restricted response set / choice set size
- Thurstone versus Luce predicting triadic choice from binary choice probabilities
- predicting triadic choice probabilities from pairwise probabilities
- Thurstone Case V versus Bradley-Terry-Luce / versus Luce choice axiom, empirical comparison
- probit versus logit predicting market shares when an alternative is removed; holdout
- Gaussian versus Gumbel random utility choice share forecasting; predictive accuracy
- normal versus logistic / double exponential error distribution, discrete choice prediction
- independence of irrelevant alternatives out of sample forecast, menu subset, choice shares
- predicting choice from smaller menus using larger menu shares; proportional renormalization
- forecasting subset choice shares; share reallocation
- Hausman & McFadden specification tests for multinomial logit; omitted alternative
- Yellott double exponential Luce Thurstone; invariance under uniform expansions
- Marley random utility models, choice probabilities as functions of binary probabilities
- similarity between stimuli, experimental test of the Luce and Restle choice models
- m-alternative forced choice: revised d′ tables; proportion correct versus number of alternatives
- arXiv: `all:"constant ratio rule" AND all:choice`; `abs:"independence of irrelevant
  alternatives" AND abs:probit`; `abs:Thurstone AND abs:Gumbel`; `abs:"choice set" AND
  abs:probit AND abs:logit`; `abs:Gaussian AND abs:"choice set" AND abs:renormalization`;
  LLM multiple-choice answer-option calibration

**Deliberately probed and found empty.** arXiv has no work matching
`all:"constant ratio rule" AND all:choice`, and no work matching
`abs:Thurstone AND abs:Gumbel`. The modern machine-learning literature that uses the
Gumbel-max trick, Plackett–Luce and Thurstone estimation (a large fraction of Yellott's
300 citers since 2016) is concerned with sampling, ranking and reward modelling, not with
menu-restriction forecasting; nothing there scores a Gaussian map against renormalization
on held-out menus.

**How much weight the negative deserves.** High for the exact four-part conjunction, and
for the psychology literature specifically, where the enumeration was complete and the two
papers most likely to be prior art (Townsend & Landon 1982, Wills et al. 2000) were read in
full text. Lower for econometrics, transportation and marketing, where the search was by
keyword rather than by citation enumeration and where seven leads remain unresolved. The
single largest hole is Takane & Shibayama (1992), which no online route reached.
