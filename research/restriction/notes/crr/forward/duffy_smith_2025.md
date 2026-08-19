# Duffy & Smith (2025) — line-length choice, Gumbel beats normal, IIA not rejected

## Citation

Duffy, S., & Smith, J. (2025). "An economist and a psychologist form a line: What can
imperfect perception of length tell us about stochastic choice?" *Theory and Decision*
**99**(3), 701–734. https://doi.org/10.1007/s11238-025-10040-4

Accepted 6 April 2025; published online 4 June 2025. Open Access, CC-BY (Springer hybrid;
Unpaywall lists no repository copy). Sean Duffy (Rutgers–Camden, Psychology), John Smith
(Rutgers–Camden, Economics).

Retrieved: main article PDF (27pp) from Springer; the "Appendix for Online Publication"
(Tables A1–A22, 20pp, dated March 29 2025) at
`https://static-content.springer.com/esm/art%3A10.1007%2Fs11238-025-10040-4/MediaObjects/11238_2025_10040_MOESM1_ESM.pdf`;
trial-level data from OSF (below). Page references below are journal pages; appendix
references are "App. Table Axx".

**Companion papers by the same authors, also uncited and pointing the same way** — this is a
coordinated programme, not a single paper:

- Duffy, S., & Smith, J. (2025a). "Stochastic choice and imperfect judgments of line lengths:
  What is hiding in the noise?" *Journal of Economic Psychology* **106**, 102787. — 117
  subjects, **56 triples** with binary choices on all three pairs. This is the dataset that
  supports the **Product Rule** test (below). Directly on the restriction question.
- Duffy, S., Gussman, S., & Smith, J. (2021). "Visual judgments of length in the economics
  laboratory: Are there brains in stochastic choice?" *JBEE* **93**, 101708.
- Duffy, S., & Smith, J. (2025b). "The random thickness of indifference." Working paper.
- Brañas-Garza, P., & Smith, J. (2024). *Imperfect perception and stochastic choice in
  experiments*. Cambridge Elements in Behavioural and Experimental Economics. — survey of
  exactly this space.

---

## Stimuli and dimensionality

**Yes. The stimuli are line lengths — a single physical continuum. This is unambiguous and the
authors make the unidimensionality the central selling point of the design.**

Abstract, p. 701:

> "we design an induced-values choice experiment: objects are valued according to only a
> single attribute with a continuous measure and we can observe whether the choice was
> optimal. Subjects are given a choice set involving lines of various lengths and are told to
> select one."

§3.2 "Line selection task", pp. 707–708:

> "In each trial, subjects were presented a choice set of lines that ranged in number between
> 2 and 6."

> "There were 10 possible longest line lengths, ranging in 16 pixel (0.73 cm) increments from
> 160 pixels (7.4 cm) to 304 pixels (14.1 cm). The lines each had a height of 0.36 cm and were
> the identical shade of grey."

Footnote 2, p. 702, invoking Stevens' near-linear length exponent:

> "Line length is an attractive stimulus type in our setting because length is perceived
> roughly linearly. For example, 100 cm appears to subjects to be nearly twice as long as 50
> cm (Stevens, 1957)"

§2.1, p. 705 — they explicitly rule out every multi-attribute or context mechanism:

> "In our experiment, there is no plausible preference for randomization, there is no
> preference for flexibility, there is no private information, there are not multiple
> attributes that could possibly interact (for instance, as complements or substitutes), and we
> can observe the consideration set."

Conclusion, §5, p. 727:

> "the choice set is populated with objects that are valued according to only a single, static
> attribute with a continuous measure"

Lines are grey, identical thickness, 100–304 px long, randomly offset within invisible 400×150
px regions, **one visible at a time** (clicking a letter label A–F swaps which line is shown),
60 s limit. So it is a unidimensional perceptual continuum *plus* a working-memory / search
component. 112 subjects × 100 trials; 10,989 valid trials after dropping 211 (1.88%) with no
line viewed or none selected (§3.5, p. 709).

**Boundary-rule relevance:** this is squarely inside the stated boundary condition. It is the
same continuum as our twelve-line-length identification dataset, but reached by a completely
different paradigm — incentivized choice with induced monetary value rather than an
identification confusion matrix. See "Does this refute or confirm" below.

---

## What was actually compared

**It is a model-fitting exercise (AIC horse race), not an out-of-sample restriction test.**
This is the crux and the answer is clean.

§4.5, p. 718 — the estimation description, verbatim:

> "In our experiment, we can perfectly observe the objective lengths of the lines and the
> choices made by the subjects. We can therefore run specifications that employ either of
> these assumptions of the error distribution and observe which provides a better fit."

The two models, both invoked through Yellott's theorem by name (§4.5, p. 718):

> "We run one specification where the stochastic component has the Gumbel distribution and is
> independently and identically distributed for every option. As McFadden (1974) and Yellott
> (1977) show, this structure implies the Luce (1959a) stochastic choice model, whereby the
> probability that option j is selected from set K is: P(j) = e^{β∗Length_j} / Σ_{k∈K}
> e^{β∗Length_k}. We refer to this Conditional Logistic model as 'Logit' and denote it as
> specification (1)."

> "We also run a specification where the stochastic component is assumed to have a normal
> distribution and is independently and identically distributed for every option. Yellott
> (1977) shows that this corresponds to Case V of Thurstone (1927a). We refer to this
> Multinomial Probit model as 'Probit' and denote it as specification (2)."

Crucially, §4.5, p. 719 — **each menu size is fitted separately**:

> "We report the Akaike Information Criterion (AIC, Akaike, 1974) for each specification,
> restricted to a particular number of lines treatment to facilitate the comparison of
> different models"

Footnote 35, p. 719: "Each specification was executed with the MDC (multinomial discrete
choice) procedure in SAS. Specifications (1), (3), and (5) were performed with the clogit
option. Specification (2), (4), and (6) were performed with the mprobit option."

Three scale functions crossed with two error laws: Linear `V = β·Length`, Log
`V = β·log(Length)` (Fechner), Power `V = β·(Length)^{1.04}` (Stevens/Teghtsoonian). Result
(Table 6, p. 719; robustness App. Tables A12–A14):

> "Regardless of the specification (Linear, Log, or Power) and regardless of the number of
> lines treatment, we find that the Logit specification has a lower AIC than the corresponding
> Probit specification." (p. 719)

> "In the body and the appendix, we provide 60 trial-level specifications. In each, the model
> assuming Gumbel errors has lower AIC than the model assuming normal errors." (pp. 719–720)

They *do* run a diagnosticity check on this fit comparison (Table 7, p. 720): 600 simulations
on the 2,284 two-line trials, errors drawn Gumbel or normal with SD ∈ {7.5, 8.0, 8.5}, AIC
recovers the true law in 81–86% of cases. So the Gumbel-wins result is not a bare AIC artefact
— it has demonstrated power.

**No out-of-sample restriction exercise anywhere in the paper or appendix.** There is no
calibrate-on-full-menu / predict-restricted-menu step. Because β is re-estimated freely inside
each menu-size cell, the design *absorbs* any menu-size-dependent rescaling rather than testing
it. I ran the missing test on their deposited data — see "Data availability" below.

---

## Set-size variation and whether menus nest

Set size varies **2 to 6**, randomized within subject (§3.2, p. 707):

> "In each trial, subjects were presented a choice set of lines that ranged in number between 2
> and 6. Each of these choice set sizes occurred with probability 1/5 and were drawn with
> replacement."

**Menus do not nest physically, but they nest in induced value by design — and this is
deliberate, and it is exactly a constant-ratio-rule structure.** §4.6, pp. 720–721:

> "Recall that our choice sets always involve a longest line and another line that is a
> specific amount shorter than this longest line. The difference between these lines is 1 pixel
> in the difficult treatment, 11 pixels in the medium treatment, and 31 pixels in the easy
> treatment. Choice sets with more than 2 lines are constructed by including lines that have
> lengths less than or equal to the shorter of these lines."

So the top pair (longest, second-longest) is held fixed at a gap of exactly 1, 11 or 31 px
across every menu size, and everything added is weakly *inferior* to the second-longest. That
gives, at the level of induced value, the pair `{a,b}` and the supersets `{a,b}∪C` with all of
`C` weakly below `b`. Their test is then precisely Clarke's constant-ratio rule (§4.6, p. 721):

> "To test IIA in our setting–as was discussed in Sect. 2.4—we can observe if the probability
> that the longest line is selected varies with the size of the choice set."

Two filters: unique second-longest (because equal-length lines are literal duplicates and
"it is not possible to determine whether the choices in these two trials were the 'same' or
'different'", p. 721), and condition on the choice being the longest or second-longest —

> "we consider only trials in which either the longest or the second longest lines were
> selected. This allows us to interpret the Selected longest variable as the choice probability
> of the longest line conditional on the choice being either the longest or second longest.
> The resulting dataset contains 8,628 observations." (p. 721)

**This is scoreable data.** Yes. See below.

**Important design limitation for our purposes:** additions are *only downward*. There are
never intermediate or superior alternatives added, no attribute structure, no decoy geometry,
no context manipulation. The design is structurally incapable of producing an
attraction/compromise/similarity effect. So it tests CRR under monotone downward menu growth on
a single continuum — the easiest possible case for proportional renormalization.

---

## Evidence for IIA

**A model-comparison significance test that fails to reject — i.e. a null result — not a
positive test, and not an equivalence test.** They estimate a dummy per menu size in a logit
for `Selected longest` (conditional on top-2) and report a **Wald statistic with 4 df** on the
four menu-size dummies.

Table 8, p. 722 (specs 1–4, with controls for longest length + difficulty + trial, ±fixed
effects, ±letter dummies):

| spec | 1 | 2 | 3 | 4 | 5 (no controls) |
|---|---|---|---|---|---|
| Wald | 4.32 | 4.31 | 5.53 | 5.09 | 20.91 |
| p | 0.36 | 0.37 | 0.24 | 0.28 | **0.0003** |

§4.6, p. 721:

> "In every specification that controls for the line lengths, the size of the choice set is not
> significantly related to the relative choice of the longest and the second longest lines. We
> interpret this as consistent with the IIA property."

> "Specification (5) highlights the importance of our controls: without accounting for line
> lengths, the Number of lines variable is significant."

p. 723:

> "In the body and the appendix, we present 24 specifications to test whether our observations
> are consistent with IIA. In each specification, our observations are consistent with IIA."

They concede the narrowness themselves, p. 722:

> "We admit that this analysis is not as general as possible. For example, IIA also predicts
> that all of the fractions of choice probabilities are also independent of the choice set."

Appendix extends to the top **three** lines (App. Tables A17–A19, 15,420 stacked
observations, testing the 1:2, 2:3 and 1:3 fractions jointly): Wald p = 0.78, 0.79, 0.81, 0.93.
Same non-rejection.

**Product Rule** (App. pp. 19–20). They note "Gumbel errors imply both IIA and that choices
satisfy the Product Rule" (fn 42, p. 723), cannot test it here for lack of all three pairs, and
so test it on the companion dataset:

> "Duffy and Smith (2025a) describe a similar line length judgment task with triples.
> Specifically, there are 117 subjects making two binary choices on every component of 56
> triples. ... We test whether the left-hand side of the Product Rule expression is
> significantly different than the right-hand side. They are not significantly different
> according to a t-test (t = 1.24, p = 0.22) and a non-parametric Signed-rank test (S = 185,
> p = 0.13). We therefore find evidence that the choices in Duffy and Smith (2025a) are
> consistent with the Product Rule."

Again a non-rejection, again with no power analysis.

**The raw cells (App. Table A15, p. 10) are the most persuasive part of their case, and they
are strikingly flat.** Fraction of top-2 choices going to the longest line, by difficulty ×
menu size (I reproduced these exactly from their deposited data; the published easy/4-lines
entry of 0.968 is a typo, the data give 0.987):

| gap | n=2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| easy (31 px) | 0.972 | 0.984 | 0.987 | 0.981 | 0.961 |
| medium (11 px) | 0.826 | 0.816 | 0.849 | 0.825 | 0.811 |
| difficult (1 px) | 0.535 | 0.545 | 0.548 | 0.552 | 0.541 |

Note the pooled rejection in spec (5) is a **composition artefact**, not fishing: "Easier
treatments and treatments with smaller choice sets are more likely to have a unique second
longest line" (App. Table A15 note), so the difficulty mix shifts toward easy as n grows. Their
controls are legitimate. Do not attack them on that.

### What they cite for cognitive-process disputes

Footnote 40, p. 722: "neuroscientists can come to different conclusions about the consistency
with IIA, based on the model of cognition employed. For example, see Gluth et al. (2020) and
Webb et al. (2020)." — i.e. the *Nature Human Behaviour* divisive-normalization exchange
(Gluth, Kern, Kortmann & Vitali 2020, "Value-based attention but not divisive normalization
influences decisions with multiple alternatives" vs. Webb, Glimcher & Louie 2020, "Divisive
normalization does influence decisions with multiple alternatives"). **Divisive normalization is
proportional renormalization.** That exchange is directly ours and is worth checking whether we
cite it.

---

## Is the utility induced?

**Yes, induced with real money, and it is a cleaner utility measure than any confusion-matrix
study. This strengthens their result and should be conceded openly.**

§3.2, p. 708:

> "The earnings on this task were increasing in the length of the choice in that trial.
> Specifically, if a line x pixels in length was selected then in that trial the subject
> earned: $5 ∗ (x − 100)/(304 − 100)."

Intro, p. 702:

> "Our induced-values design is advantageous because we can observe—without noise—whether the
> choice was optimal and the objective values of the elements in the choice set."

The payment is *linear in the physical continuum*, which they flag as the economically
important feature (p. 702):

> "This payment scheme shares a key feature with economic choice: the material differences
> between choices are smaller for options closer to indifference and larger for options farther
> from indifference."

§3.5, p. 709: three trials randomly selected for payment, $5 show-up fee, cash, mean earnings
$14.50, no feedback until the end. Lineage acknowledged in fn 1, p. 702: "There is a history of
induced-values experiments, possibly beginning with Smith (1976)".

They also **observe the consideration set**, which kills the standard escape hatch (§4.3,
p. 715): 99.0% of suboptimal choices occurred in trials where the subject had actually viewed
the longest line, so "the bulk of our suboptimal choices can be explained due to imperfect
perception, rather than not considering the longest line."

So: incentivized, single-attribute, objectively known values, observed consideration sets,
observed search history and response times. On the utility-measurement axis this is stronger
than a confusion matrix. We should say so rather than let a referee say it for us.

---

## Data availability

**Fully deposited and I have verified it downloads and reproduces.**

Data availability statement, p. 729:

> "The dataset and the results of our simulations can be downloaded from https://osf.io/f7gu4."

OSF node `f7gu4` contents:

- `NoCLUALData-OSF.csv` — 3.66 MB, **11,200 rows (112 subjects × 100 trials) × 74 columns**,
  trial level. Direct download: `https://osf.io/download/nsya2/`. Columns include
  `LineLength0..5`, `NumLines`, `LineSelected`, `SelLong`, `SelNextLong`, `NumNextLong`,
  `Difficulty` (0=easy/31px, 1=medium/11px, 2=difficult/1px), `LongestLen`, `Trial`,
  `TotalTime`, `ViewClicks`, per-letter view-click counts and dwell times
  (`AViewClicks`…`FTotTime`), `TimeSinceLong`, `TimeViewLong`, `SeenLongest`,
  `EitherNoVieworNOSel`, CRT items, demographics. **Every menu is fully reconstructible.**
- `NoCLUAL-FromPaymentLines-TriplesCalculation-SAS.xlsx` — 16.5 KB, the **56 triples** used for
  the Product Rule test on Duffy & Smith (2025a). Download: `https://osf.io/download/ahvkq/`.
- `NoCLUAL-Survey.pdf`, plus folders `ePrime Code`, `Screenshots`, `MDC Simulations`.

Local copies for this note: `ds.pdf` / `ds.txt` (article), `esm1.pdf` / `esm1.txt` (appendix),
`osf_data.csv` in the session scratchpad — re-download rather than rely on those paths.

### Verification and the analyses they did not run

**(a) Exact reproduction.** From the deposited CSV I reproduce their IIA sample of exactly
**8,628** observations, all of App. Table A15, their Table 8 spec (1) **Wald = 4.32, p = 0.364**
(paper: 4.32, 0.36), and their Table 6 Linear-Logit β row exactly: 0.1258, 0.1394, 0.1440,
0.1354, 0.1308 for n = 2…6. The replication is sound; findings below are not artefacts of
misreading their spec.

**(b) Their IIA test has essentially no power against Thurstone Case V. This is the single
strongest response to the paper.** I generated 500 replications of their own design (their
actual 8,628 line-length configurations) under iid normal errors with SD ∈ {7.5, 8.0, 8.5} —
*their own calibrated range*, and I independently recover σ ≈ 8.05 px from the difficult/n=2
cell — then ran their Table 8 spec (1) on each:

| DGP | rejection rate of their IIA Wald test at nominal 5% |
|---|---|
| **Thurstone Case V**, iid normal SD 7.5 | 5.6% |
| **Thurstone Case V**, iid normal SD 8.0 | 7.4% |
| **Thurstone Case V**, iid normal SD 8.5 | 8.4% |
| Luce, iid Gumbel matched SD 7.5 | 5.0% |
| Luce, iid Gumbel matched SD 8.0 | 4.8% |
| Luce, iid Gumbel matched SD 8.5 | 6.4% |

Power against the leading alternative is 6–8% at a 5% nominal level. The test cannot
distinguish Thurstone from Luce **at all**. Analytically the reason is transparent: with σ ≈ 8
px and a top-pair gap fixed at 1/11/31 px, Case V predicts the top-2 fraction should drift by
only **+0.009 (difficult), +0.013 (medium), +0.001 (easy)** from n=2 to n=6, while the binomial
SE in the difficult n=6 cell (229 obs) is 0.033 — nearly 4× the effect being hunted.

Note the asymmetry in their own methodology: they ran a diagnosticity simulation for the
Gumbel-vs-normal AIC comparison (Table 7), where they had power, and ran none for the IIA test,
where they did not. Their "evidence consistent with IIA" is a non-rejection from an
underpowered test, and their generality claim rests on it.

**(c) The out-of-sample restriction test they never ran — and Luce still wins.** Fitting a
single scale on one menu-size stratum and scoring held-out strata by log-likelihood per
observation (Luce with iid Gumbel vs Thurstone Case V with iid normal, multinomial, full menus,
10,989 valid trials):

| calibrate | predict | fitted β | fitted σ | OOS LL/obs advantage to Luce |
|---|---|---|---|---|
| n=2 (N=2,284) | n=3..6 (N=8,705) | 0.1258 | 10.56 px | **+0.0026** nats/obs |
| **n=6 (N=2,119)** | **n=2 (N=2,284)** | 0.1308 | 11.33 px | **+0.0028** nats/obs (SE 0.0015, t = 1.86) |
| n=5,6 (N=4,338) | n=2,3 (N=4,465) | 0.1329 | 11.33 px | **+0.0046** nats/obs |

The middle row is the restriction direction proper — calibrate on the full menu, predict the
restricted menu. **Luce wins it.** So we cannot dismiss their result as "mere fitting": the
restriction question resolves the same way on this dataset. The margin is thin (t = 1.86,
one-sided p ≈ 0.03) but the sign is against us, and a referee running this will get the same
answer.

**(d) Menu-invariance of the Luce scale, full-multinomial LR test** (more powerful than their
top-2 Wald test, since it uses all alternatives): β per menu size 0.1258, 0.1394, 0.1440,
0.1354, 0.1308 (pooled 0.1353) — a non-monotone hump, ~14% peak-to-trough. LR = 8.17, df = 4,
**p = 0.086**. Marginal; does not reject a menu-invariant Luce scale. This is the *best*
available anti-IIA statistic in their data and it still does not fire. Reporting it is honest
and pre-empts the accusation that we only tried the weak test.

---

## Does this refute or confirm the boundary rule

**It confirms the boundary rule, and it is the best independent confirmation of it that
exists.** But it comes with a real inherited liability, because the authors do not frame it as
a boundary result — they frame it as evidence that IIA is general.

Confirming, on our side of the boundary:

1. The stimulus is a single physical continuum, near-linearly perceived (their fn 2 cites
   Stevens explicitly). This is exactly where the paper predicts proportional renormalization —
   the Gumbel point — should win.
2. Gumbel beats normal in **60 of 60** trial-level specifications, across linear/log/power
   scale functions, with item and subject heterogeneity, with a diagnosticity simulation
   backing the comparison. This is a well-powered result and it goes our way.
3. CRR/IIA is not rejected in 24 specifications, and my own out-of-sample restriction test on
   their data also favours Luce. On this continuum, proportional renormalization is right.
4. It reaches this by a **different paradigm** from ours. Our unidimensional evidence is
   identification/confusion-matrix (tone identification; the twelve-line-length dataset).
   Theirs is incentivized choice with induced monetary value, observed consideration sets, and
   observed search. A referee who suspects our boundary rule is an artefact of confusion-matrix
   methodology is answered by Duffy & Smith. **This is an asset, not a threat — provided we
   cite it as one.**
5. Their reference list is our reference list. §2.4 "Independence from irrelevant
   alternatives" (pp. 706–707) walks the same forward-citation chain we are tracking: Clarke
   (1957), Clarke & Anderson (1957), Hodge & Pollack (1962), Morgan's (1974) reanalysis
   ("Morgan (1974) reanalyzed the original data from Clarke (1957) and found significant
   departures from CRR"), Tversky (1972) on dot estimation vs multi-attribute vs risky choice,
   Huber–Payne–Puto (1982), Crosetto & Gaudeul (2016). Plus Yellott (1977) by name for the
   theorem, Thurstone (1927a,b) including "Case V of Thurstone (1927a)", Luce (1959a,b, 1986,
   1994, 2005), McFadden (1974, 1976, 1981, 2001), Bradley & Terry (1952), Becker–DeGroot–
   Marschak (1963), Falmagne (1978), Echenique–Saito–Tserenjigmid (2018), Kovach &
   Tserenjigmid (2022). We are in the same conversation and cannot pretend otherwise.
   (Not cited: Block–Marschak, Luce 1963, Yellott's later work.)

The liability:

6. They state the **opposite generality claim**, in print, in a peer-reviewed journal. §2.4,
   p. 707: "Our results suggest that IIA violations stem from specific details in the choice
   setting rather than being a general feature of choice." Conclusion, p. 728: "Given the
   general nature of our choice setting, we interpret this as suggesting that IIA could be a
   general feature of choice, and that violations of IIA only occur in specific choice
   settings, such as those with certain profiles of multiple attributes." Intro, p. 704: "This
   suggests to us that choice that violates IIA stems from details in a specific choice setting
   ... rather than being a general feature of choice."
7. Therefore our paper must be scrupulous about the difference between **"IIA/CRR fails as a
   general axiom"** and **"IIA/CRR fails in setting X."** If any sentence in the paper reads as
   "restriction fails everywhere," Duffy & Smith is a counterexample to that sentence. If the
   boundary condition is stated crisply and up front — proportional renormalization wins on
   unidimensional perceptual continua, and fails off them — Duffy & Smith is corroboration.

Verdict on the substance: **not a counterexample to the thesis; a counterexample to an
overstated version of the thesis.** The fix is editorial, not empirical.

---

## Verdict for circulation

Duffy & Smith (2025) is **not a refutation of the thesis; it is the strongest independent
confirmation of the paper's stated boundary condition, and it must be cited as such, promptly
and prominently.** Their stimuli are grey lines of 100–304 pixels, valued by "only a single
attribute with a continuous measure" (abstract, p. 701) with linear cash payoffs — a
unidimensional perceptual continuum, which is precisely the regime where the paper predicts the
Gumbel point should win. It does: Gumbel beats normal in 60 of 60 specifications (pp. 719–720),
IIA survives 24 tests (p. 723), and when I run on their deposited data the out-of-sample
restriction test they never ran — calibrate on six-line menus, predict two-line menus — Luce
still beats Thurstone Case V by +0.0028 nats/observation (t = 1.86). Their design also improves
on our unidimensional evidence in ways worth conceding aloud: induced monetary value rather than
an inferred scale, observed consideration sets (99.0% of errors occur on trials where the
longest line was actually viewed, §4.3 p. 715), and a different paradigm from confusion-matrix
identification, so it answers the objection that our boundary rule is a methodological artefact.

**Now the strongest case that it *is* a refutation, stated as a hostile referee would.** Duffy
& Smith do not present a boundary result; they present a generality claim in the opposite
direction — "IIA could be a general feature of choice, and that violations of IIA only occur in
specific choice settings" (p. 728) — and they earn the right to try, because their setting is
arguably the cleanest IIA test ever built: induced values, one attribute, no preference for
randomization, no private information, no multi-attribute interaction, consideration set
observed, 10,989 incentivized trials, open data. If imperfect perception is the engine of
stochastic choice and that engine breaks Luce's axiom, this is exactly where the breakage should
be visible and measurable — and it is not visible. Their Gumbel-over-normal result is not a bare
AIC horse race: they ran a recovery simulation showing 81–86% correct identification (Table 7,
p. 720), so the comparison has demonstrated diagnosticity, and it goes 60/60 against Thurstone.
The Product Rule, a logically independent Luce implication, also survives on their companion
56-triple dataset (t = 1.24, p = 0.22; signed-rank p = 0.13). And the restriction test — *our*
declared crux — favours Luce on their data too, in the calibrate-full/predict-restricted
direction, so "they only fitted, they never restricted" is not an available escape. On this
reading, a peer-reviewed paper with induced utility and public data has run our own crux test
and got the opposite answer, and our silence about it looks like avoidance.

**The answer to that case, in one line each.** (i) Their IIA evidence is a *non-rejection from a
test with no power*: replicating their Table 8 spec (1) on 500 simulated datasets built from
their own 8,628 configurations, the test rejects 5.6–8.4% of the time when the truth is
Thurstone Case V at their own calibrated σ = 7.5–8.5 px, against 4.8–6.4% when the truth is
Luce — because Case V predicts a top-2 fraction drift of only +0.009 to +0.013 from n=2 to n=6
while the binomial SE in the sparsest cell is 0.033. They simulated power for the AIC comparison
where they had it and simulated none for the IIA test where they did not. (ii) Their menu growth
is *downward only* — "Choice sets with more than 2 lines are constructed by including lines that
have lengths less than or equal to the shorter of these lines" (p. 720) — with no intermediate,
superior, or attribute-structured additions, so the design is structurally incapable of
producing the failures documented elsewhere, and they concede the narrowness themselves ("We
admit that this analysis is not as general as possible", p. 722). (iii) The Gumbel-wins fit
result and the out-of-sample restriction result are conceded outright, because on a
unidimensional continuum that is what our own boundary rule predicts. So: **confirmation of the
boundary condition, plus a citation obligation, plus a discipline requirement** — every claim in
our paper about restriction failure must carry its domain restriction on its face, and Duffy &
Smith's generality claim should be met head-on with the power argument rather than left
unmentioned. Cite the companion papers too (Duffy & Smith 2025a in *J. Econ. Psych.*, Duffy–
Gussman–Smith 2021, Brañas-Garza & Smith 2024): this is a research programme pointing one way on
one continuum, and a referee who knows one of them knows all four.
