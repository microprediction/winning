# Prior art: has anyone already done this?

**The question.** Has anyone calibrated a parameter-free Gaussian model to full-menu
choice shares and scored its restricted-menu predictions against proportional
renormalization, out of sample, with nothing fitted to the restricted menu?

**The answer, in one line.** No — but the claim is far more crowded than the project's
current framing admits, and every component idea is already in print somewhere. Nobody has
run the specific out-of-sample forecast on real matrices; **Lee (1968) computed the
parameter-free Gaussian departure from renormalization and printed the table**, Wills et al.
(2000) published the conclusion, Townsend & Landon (1982) ran the protocol, and Rouder (2004)
published the negative result.

Read all of "What is already published" before circulating anything. The sentence "the
literature did not score renormalization against a parameter-free Gaussian race out of
sample" in `../README.md` survives only on the words *out of sample* and *race* — the
Gaussian-versus-renormalization comparison itself was tabulated fifty-eight years ago.

**Two things must be pre-empted rather than left to a referee.** Duffy & Smith (2025) is a
published, open-access, contrary result: on induced-value choice with varying set size, errors
were "better described as having a Gumbel distribution rather than a normal distribution", with
"evidence consistent with the independence from irrelevant alternatives (IIA) property."
And Treisman & Faulkner (1985) may already have concluded that signal detection theory beats
choice theory on data where response-set size varies — that one is **unresolved and is the
highest residual risk in this file**.

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

### The parameter-free Gaussian departure was computed and tabulated in 1968

Lee, W. (1968). "Detection theory, micromatching, and the constant-ratio rule."
*Perception & Psychophysics* 4(4):217–219. doi:10.3758/BF03206305
Full text recovered via Wayback and read; see `lee1968.md` for the complete treatment.

**This is the most important citation for the project.** Lee takes sets of three stimuli and
subsets of two, assumes equal-variance independent normals with zero covariance (Case V), and
computes by numerical integration the **constancy**

    c_ij = (r_i / r_j) / (r′_i / r′_j)

— the observed restricted-menu odds divided by the odds renormalization predicts, which the
CRR requires to equal 1.0. *This is exactly the quantity the project measures.* He states the
point plainly (p. 218): "According to the CRR, c_ij = 1.0; however, as we shall see, there is
no necessity that c_ij = 1.0 for detection theory or micromatching." Table 1 prints the
departures at d′ = 0.0 … 3.0 for six configurations, and they are large — univariate detection
theory falls to c = 0.22 by d′ = 3.0 in one configuration, and 0.00 at d′ = 0 in two others.
He then proposes precisely the project's use for them: the differences "can be used in
diagnosis of the basis of empirical confusion matrices."

Three consequences, and they are not all bad:

- **The project cannot claim that a Gaussian model implies a departure from renormalization,
  nor that the departure is large, nor that its size was unknown.** All three are in Table 1.
- **Lee's map is not the project's map, and this distinction must be stated explicitly rather
  than glossed.** His Gaussian arm is a *decision-bound* model — one sample point in a shared
  space, carved by optimal cutoffs re-solved on the surviving menu — not an independent-race
  Thurstone model where each alternative draws its own latent and the argmax wins. His
  "micromatching" is a posterior/ratio rule, also not a race. For the univariate
  three-stimulus case these are genuinely different geometries with different numbers.
- **Lee predicted where the failure would live, and he was right.** The departures concentrate
  in the *univariate* configurations, which he uses to explain Hodge & Pollack's generalization
  that CRR holds better for multidimensional stimuli. Townsend & Landon later found exactly
  this: the subset that deleted a survivor's near-substitute was the one CRR could not predict.
  That is a 1968 directional prediction confirmed in 1982 and never credited as such.

And the sentence that best justifies the whole exercise (p. 219): "I would suggest that
investigators of the CRR give more detail on individual response probabilities for specific
configurations, rather than simply giving gross plots and gross statistics. If this level of
detail is not possible for a journal article, at least the availability of the data could be
advertized." The field did not comply — which is precisely why this sweep found only two
scoreable sources. That is the project's real opening: not a new idea, but the first execution
of a fifty-eight-year-old request.

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

### A published contrary result that must be addressed, not ignored

Duffy, S., & Smith, J. (2025). "An economist and a psychologist form a line: What can imperfect
perception of length tell us about stochastic choice?" *Theory and Decision*.
doi:10.1007/s11238-025-10040-4 — **open access**, PDF at
https://link.springer.com/content/pdf/10.1007/s11238-025-10040-4.pdf
(Abstract verified verbatim via a fetched Semantic Scholar API record. This work appears in the
sweep as a citer of both Clarke 1957 and Morgan 1974.)

An induced-values design — subjects choose among lines of various lengths and are paid in
proportion to the length chosen, so the experimenter knows the true utility and can see whether
the choice was optimal. Choice-set size is varied. Two sentences from the abstract go directly
against the project:

> "Errors in our data are better described as having a Gumbel distribution rather than a normal
> distribution. We find evidence consistent with the independence from irrelevant alternatives
> (IIA) property and we observe dynamic effects."

This is a 2025 economics result, in a good journal, with a cleaner utility measure than any
confusion-matrix study, concluding **for** Gumbel and **for** IIA on choice sets of varying
size. It is not prior art — nothing is held out, the distributional comparison is a fitted
model selection, and IIA is assessed within the design rather than by forecasting a deleted
menu — but it is the most direct published contradiction of the project's expected finding and
it will be the first thing an economics referee reaches for.

Honest handling requires distinguishing two claims that the abstract runs together: whether the
*error distribution* fits better as Gumbel, and whether *menu deletion* preserves odds. Lines
of differing length are a one-dimensional stimulus set — exactly the univariate geometry where
Lee (1968) predicted the *largest* Gaussian departures — so a null IIA result there is
genuinely surprising on the project's own theory and deserves engagement rather than
dismissal. Their data may be re-analysable in the project's framing; that is worth checking
before publication, not after.

### The highest residual risk in this file

Treisman, M., & Faulkner, A. (1985). "On the choice between choice theory and signal detection
theory." *Quarterly Journal of Experimental Psychology* 37A(3):387–405.
doi:10.1080/14640748508400941

**Abstract only; full text not obtained.** **RESOLVED 2026-08-19: this is category (B) and the
threat collapses. See `treisman_faulkner_1985.md` for the full workup.** Treisman & Faulkner
fit d′ and β *within* each m-AFC condition, so nothing is held out; the compared quantity is a
scalar per condition and never the odds between named survivors; no response set is restricted
over shared alternatives; and proportional renormalization never appears. Decisively, *neither*
parameter turned out invariant — their own preferred model drifted too — and they chose signal
detection by a plausibility argument about the sign of the drift rather than by predictive
accuracy. Robinson et al. call the results "somewhat ambiguous". Of the fifteen citing works,
none cites it as settling Gaussian versus Luce under set-size change. Their own experiment is an
auditory m-AFC memory task with n = 6, in which m is confounded with memory load — the project's
own quality-changing-removal failure mode.

**Correction to what this file previously said.** The claim that Treisman & Faulkner tested
invariance "on Miller, Heise & Lichten's vocabulary-size data" was attributed here to Robinson
et al. The published Robinson et al. text does not mention Miller, Heise and Lichten at all;
verified against the article itself, which says only "[Treisman and Faulkner (1985)] reported
evidence for the Gaussian signal detection model, however, their results were somewhat
ambiguous." The Miller-Heise-Lichten link comes from Treisman & Faulkner's own abstract, which
states the procedure "is applied to the data of Miller, Heise and Lichten (1951) and to the
results of an experiment". Two separate claims were run together.

**Robinson et al. (2023) is the closer prior art and should be cited as such.** Verified from
the article: their primary analysis holds d′ and β *fixed across all* m-AFC conditions and
compares log-likelihood, and the Gaussian wins in both experiments, t(29) = 4.26 and
t(29) = 4.42, both p < .001, n = 30 each. Their conclusion, verbatim: "the d′ parameter of the
Gaussian signal detection model was more stable across m-afc conditions than the β parameter of
the softmax model." That is a cross-condition constraint, much nearer this project's map than
anything in Treisman & Faulkner. It remains category (B): parameters are fitted to the same data
that scores them, no response set is restricted, and the constant-ratio rule is never the
competitor. Correct reference is *Journal of Mathematical Psychology* **117**:102805, 2023,
confirmed via Crossref on doi 10.1016/j.jmp.2023.102805.

The superseded characterisation follows, kept for the record:
Treisman & Faulkner favour signal detection theory over choice theory on Miller, Heise &
Lichten's vocabulary-size data, by fitting d′ and β within each m-AFC condition and testing
whether the parameters stay invariant across response-set sizes.

If that characterisation is right, it is category (B): parameters are fitted within each menu,
so nothing is held out, and the test is parameter invariance rather than a scored forecast. But
it would mean **"Gaussian over Luce when the response-set size changes" has been the
psychophysics verdict since 1985**, which would narrow the project's claim to the protocol
alone. Since I could not read it, that inference is not secure in either direction.

**Get a library copy of Treisman & Faulkner (1985) before claiming novelty.** It is the one
item in this sweep that could still change the verdict, and it is cheap to settle.

Related and verified: Robinson et al. (2022), "Revisiting the connection between Luce's Choice
Axiom and Signal Detection Theory", treats Gaussian versus Gumbel signal detection across
m-AFC as "a strong test bed of parameter invariance" — both parameterizations fitted, so
category (B), but it is the live modern thread on this exact contrast and should be cited.

---

## Categorised near-misses

Categories: **(A)** true prior art; **(B)** fits both maps, or fits anything to the
restricted menu; **(C)** theory only, no empirical forecast; **(D)** menu/IIA testing
without the Gaussian-versus-renormalization forecast framing; **(E)** irrelevant.

| Work | Cat | Fetched url | Why this category |
|---|---|---|---|
| **Lee 1968, *P&P* 4:217** | C, **closest on the quantity** | `web.archive.org/web/2020id_/https://link.springer.com/content/pdf/10.3758/BF03206305.pdf` (full text) | Computes and tabulates the parameter-free Gaussian departure from renormalization for 3-sets → 2-subsets, and proposes it "in diagnosis of the basis of empirical confusion matrices". Numerical only, hypothetical configurations, no data. **Decision-bound, not an independent race.** See `lee1968.md`. |
| Yellott 1977, JMP 15:109–144 | C | escholarship.org/content/qt7z91732x/qt7z91732x.pdf | Proves Gumbel uniqueness for triples; no data at all. The divergence is a theorem. |
| **Duffy & Smith 2025, *Theory and Decision*** | **Contrary result** | link.springer.com/content/pdf/10.1007/s11238-025-10040-4.pdf (open access; abstract verified verbatim) | Induced-value line-length choice with varying set size: errors "better described as having a Gumbel distribution rather than a normal distribution", "evidence consistent with the independence from irrelevant alternatives (IIA) property". Fitted, nothing held out — but a direct published contradiction. **Pre-empt this.** |
| **Treisman & Faulkner 1985, *QJEP* 37A:387** | B, **unresolved — top risk** | doi 10.1080/14640748508400941; abstract only, full text NOT obtained | Reportedly favours SDT over choice theory on Miller-Heise-Lichten vocabulary-size data by testing d′/β invariance across m-AFC sizes. If so, "Gaussian over Luce when menu size changes" dates from 1985. **Settle from a library before claiming novelty.** |
| Robinson, DeStefano, Brady & Vul 2022 | B | files.osf.io/v1/resources/n78wz_v1/providers/osfstorage/6310f8d18cc1bd001d168f33 (full text) | Gaussian vs Gumbel signal detection across m-AFC as "a strong test bed of parameter invariance"; both parameterizations fitted. The live modern thread on this contrast. |
| Ashby & Perrin 1988, *Psych Rev* 95:124 | B | OpenAlex record, doi 10.1037/0033-295x.95.1.124 (abstract) | Gaussian general recognition theory versus Luce's similarity choice model, both fitted to the same identification data, in sample. |
| Currim 1982, *JMR* 19(2) | B, framing warning | OpenAlex record, doi 10.1177/002224378201900205 (abstract; protocol not obtained) | Closest marketing near-miss, and important for framing: it groups "the Luce choice axiom, the LOGIT model, and the independent PROBIT model" together as all carrying IIA — i.e. the marketing tradition treats independent probit as an IIA *member*, not the IIA-breaking competitor. Expect this objection. |
| Batsell & Polking 1985, *Marketing Science* 4:177 | B | OpenAlex record, doi 10.1287/mksc.4.3.177 (abstract) | Reviews Generalized PROBIT as a cure for Luce's axiom, rejects it on estimation grounds, proposes an OLS competitive-effects model fitted on the menus of interest. |
| Conlon & Mortimer 2013, *AEJ: Micro* 5(4):1 | B | Crossref record | The genuine "alternative removed from the menu" field dataset (vending-machine stockouts), but availability variation feeds estimation and the horse race is logit vs mixed logit. |
| Wills et al. 2000, QJEP 53A:983 | B/D, closest miss | andywills.info/assets/pdf/2000Wills.pdf | Asserts the Thurstonian conclusion on human 3→2 restricted-menu data and owns the Yellott framing, but the model has **rectangular** noise, four free parameters, and threshold S refitted per menu (0.18 two-choice vs 0.65 three-choice); renormalization is never scored. |
| Rouder 2004, Psych Rev 111:80 | D | see `rouder2004.md` | Rejects CRR out of sample on human restricted menus with a signed residual; proposes no Gaussian competitor. Prior art on the negative result. |
| Townsend & Landon 1982, JMP 25:119 | D | Wayback copy, see `townsend1982.md` | Runs the master→subset out-of-sample protocol with Luce-family maps only; builds the Gumbel race explicitly; never raises a normal alternative. |
| Kornbrot, Georgiou & Page 2018, PeerJ preprint | B | S2 API record (full text 403) | Real probit-vs-logit comparison, but binary choice only and goodness-of-fit, not menu-restriction forecasting. |
| Ashby, Lee & Balakrishnan 1992, Math Soc Sci 23 | B (title-level only) | S2 API record; abstract elided | "Comparing the biased choice model and multidimensional decision bound models of identification" — Gaussian decision-bound versus Luce on identification data, but both fitted, and no restricted-menu arm found. **Not verified beyond the title; see unresolved leads.** |
| del Castillo 2020, Transportmetrica A | B/E | S2 API record | Gumbel-marginal copula RUM fitted to interurban trip data, compared to nested logit. All Gumbel-family, all fitted. |
| **Horowitz 1980, *Transportation Research B* 14** | C, title-level only | S2 API record, doi 10.1016/0191-2615(80)90013-2; abstract elided | "The accuracy of the multinomial logit model as an approximation to the multinomial probit model of travel demand" — **the closest econometrics ancestor**: it asks directly how much Gaussian and Gumbel predictions of choice shares diverge. Almost certainly analytic/simulation rather than a restricted-menu forecast scored on data, but this was **not verified** — see unresolved leads. 79 citations. |
| Horowitz 1981, *Transportation Science* 15:153 | D, title-level only | S2 API record, doi 10.1287/trsc.15.2.153; abstract elided | "Testing the Multinomial Logit Model against the Multinomial Probit Model without Estimating the Probit Parameters" — a specification test, not a forecasting comparison. |
| Bell, Keeney & Little 1975, *JMR* 12 | C | Crossref record, doi 10.2307/3150435 | "A Market Share Theorem" — the axiomatization behind the marketing attraction model, i.e. renormalization as a market-share axiom. Theory; no Gaussian competitor, no out-of-sample subset test. |
| Dagsvik 1994 *Econometrica*; Dagsvik 2014 *Theory and Decision* | C | Crossref records | "What independent random utility representations are equivalent to the IIA assumption?" — characterisation theorems, no forecast. |
| Marley 1982, Math Soc Sci | C | Crossref record; abstract elided | Whether all choice probabilities are functions of the binary ones — the theoretical relative of the question, no data. |
| Chakrabarti 1969, *Econometrica* | C | Crossref record | Note on the relation between binary and multiple choice probabilities. |
| Iverson 1979, JMP | C | Crossref record | Conditions for Thurstone Case III representations of binary choice probabilities. |
| Hausman & McFadden 1984, *Econometrica* 52, and its descendants (Small & Hsiao 1985; McFadden 1987; Rouwendal 2017) | D | Crossref record, doi 10.2307/1910997; S2 citation graph | The canonical omitted-alternative IIA test **re-estimates on the restricted choice set** — the restricted menu is an estimation input by construction, the exact opposite of holding it out. Rouwendal (2017) shows these tests lose power once alternative-specific constants absorb omitted variables, which is itself an argument *for* a forecasting test. **Strongest single negative in this file:** an exhaustive sweep of all 2,666 works citing Hausman & McFadden, intersecting {probit, Gaussian, Thurstone} with {out-of-sample, forecast, holdout, renormalize, market share, red bus, cannibalization, omitted alternative}, returned one false positive; zero citing titles contain "market share". That test never became a forecasting comparison. |
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

0. **Treisman, M., & Faulkner, A. (1985). "On the choice between choice theory and signal
   detection theory", *QJEP* 37A(3):387–405 (doi 10.1080/14640748508400941).** Listed first
   because it is the **only item that could still overturn the verdict**, and because it is the
   cheapest to settle. Abstract obtained; full text not. Characterised at second hand (via
   Robinson et al. 2022) as fitting d′ and β within each m-AFC condition and testing parameter
   invariance across response-set sizes, which would make it category (B) — but if it is more
   than that, then "Gaussian over Luce when the menu changes" is a 1985 result and the project
   is left claiming only the protocol. **Read this before circulating.**
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
6. **Horowitz, J. L. (1980). "The accuracy of the multinomial logit model as an approximation
   to the multinomial probit model of travel demand", *Transportation Research Part B* 14
   (doi 10.1016/0191-2615(80)90013-2), 79 citations, and Horowitz (1981), *Transportation
   Science* 15:153 (doi 10.1287/trsc.15.2.153).** Both abstracts are elided by the publisher
   and neither full text was obtained. **This is the most important unresolved item on the
   econometrics side**, because the 1980 title describes precisely the question of how far
   Gaussian and Gumbel share predictions diverge. It is very likely an analytic/simulation
   approximation-error study rather than a scored out-of-sample restricted-menu forecast — but
   that must be confirmed, not assumed, before the novelty claim is made in an economics
   venue.
7. **The marketing "attraction model" thread.** Bell, Keeney & Little (1975) "A Market Share
   Theorem", *JMR* 12 (doi 10.2307/3150435) axiomatizes renormalization as a market-share
   property; Cooper & Nakanishi's *Market-Share Analysis* (1988) is the standard treatment.
   Whether anyone in that line scores subset-share predictions against proportional
   renormalization out of sample was not established.
8. **The m-alternative-forced-choice thread.** Hacker & Ratcliff (1979) "A revised table of
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
- market share theorem / attraction model (Bell, Keeney & Little; Cooper & Nakanishi)
- multinomial probit versus multinomial logit forecasting mode shares when a mode is removed
- accuracy of logit as an approximation to probit; nested logit versus probit forecasting accuracy
- red bus / blue bus problem with probit prediction
- predicting the share of a new alternative; external validity of discrete choice experiments
- cannibalization and share reallocation, logit versus probit
- validation of choice model predictions on holdout choice sets (conjoint)
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

**Additional threads run in the econometrics/marketing sweep.** Full enumeration of all 2,666
works citing Hausman & McFadden (1984) and all 414 citing Yellott (1977), keyword-intersected
as described in that table row. Small & Hsiao 1985, McFadden 1987, Rouwendal 2017, Fry & Harris
1996; Conlon & Mortimer 2013 stockout data; share reallocation, cannibalization, draw,
substitution; Daganzo and Horowitz 1980/1981; Bell-Keeney-Little, Kotler, Batsell & Polking,
Batsell & Lodish, Currim, Meyer & Eagle, Grover & Dillon, Cooper & Nakanishi; Debreu 1960 and
the IIA counterexample line; the Thurstone-vs-Luce pair-comparison line (Burke & Zinnes 1965,
Hohle 1966, Bradley 1954, Hopkins 1954); the Thurstonian-vs-Plackett-Luce ranking literature;
and Duffy & Smith's four-paper line-length programme.

**How much weight the negative deserves.** High for the exact four-part conjunction. Highest
for the **psychology** literature: forward citation enumeration was complete for three of the
four seeds, the Luce 1959 citation set was filtered by full-text CRR phrasings (which is what
surfaced the operant vein), the Yellott 1977 set was enumerated twice independently, and the
four papers most likely to be prior art — Lee 1968, Townsend & Landon 1982, Wills et al. 2000
and Hodge 1967 — were all obtained and read in **full text**, each with a targeted search for
Gaussian/normal/probit/Case V/discriminal/Thurstone. Those readings are what the verdict rests
on, and two of them (Lee, Townsend & Landon) were recovered only via the Wayback Machine after
Springer and Elsevier refused every direct request.

**Reasonably high for econometrics** now that the Hausman & McFadden citation graph has been
swept exhaustively — that is a genuine enumeration, not keyword matching, and it comes back
empty. **Weaker for marketing**, where publisher abstract elision defeated verification:
Currim 1982's validation protocol, Gensch & Ghose 1997, Horowitz 1980, Ashby/Lee/Balakrishnan
1992, Smith 1992 and Marley 1982 were all characterised from titles and metadata rather than
text.

**Three items to settle from a library before the claim is made in print**, in priority order:

1. **Treisman & Faulkner (1985)** — the only item that could overturn the verdict.
2. **Takane & Shibayama (1992)** — no online route reached it at all; cited by Rouder as the
   most stringent statistical test of the CRR.
3. **Horowitz (1980)** — title is the closest description anywhere of a Gaussian-versus-Gumbel
   share-prediction comparison; almost certainly an approximation-error study, but unconfirmed.

**A note on convergence.** The `auditory/` branch of this project independently located Lee
(1968) and reached the same headline reading of it. Two searches run separately, from different
seeds, converging on the same 1968 paper as the pivotal ancestor is the strongest available
evidence that the ancestor set is now correctly identified. See `../auditory/lee1968.md`
alongside `lee1968.md` here.
