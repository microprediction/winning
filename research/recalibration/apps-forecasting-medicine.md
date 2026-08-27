# Applications scouting: forecasting and medicine

Front assignment: weather/climate categorical postprocessing, medical multi-class
diagnosis and recalibration, epidemiology, judgmental forecast aggregation.
Horse racing and betting excluded by instruction.

**The layer under evaluation.** Take a frozen scorer that ranks `N` mutually
exclusive outcomes. Treat its scores as latent Gaussian means. Fit a
low-rank-plus-diagonal covariance `(V, D)` post hoc on held-out realized outcomes
by exact maximum likelihood. Read out exact correlated probabilities. Strictly
generalizes softmax and temperature scaling (softmax is the iid-Gumbel special
case), and additionally delivers correct restricted-menu conditioning and exact
top-k / set probabilities.

---

## 0. The one argument that unifies every front

Before the domains, the generic gap. It is real, it is documented, and it is the
same sentence in every field.

Everyone agrees the thing you actually want is **canonical calibration** — the
whole probability vector correct, not just the top label. Popordanoska, Sayer and
Blaschko (NeurIPS 2022) state the standard position:

> "The most strict notion of calibration, called canonical (or distribution)
> calibration [Bröcker, 2009, Kull and Flach, 2015, Vaicenavicius et al., 2019],
> requires the whole probability vector to be calibrated. The curse of
> dimensionality makes estimation of this form of calibration difficult, and
> current estimators, such as the binned estimator ECE_bin [Naeini et al., 2015],
> MMCE [Kumar et al., 2018] and Mix-n-Match [Zhang et al., 2020], have
> computational or statistical limitations that prevent them from being
> successfully applied in this important setting."
> — <https://arxiv.org/pdf/2210.07810>

Vaicenavicius et al. (AISTATS 2019) show marginal calibration is genuinely not
enough, with an explicit `N=3` counterexample:

> "The model is perfectly calibrated according to Guo et al. (2017) and
> additionally all marginal predictions are calibrated. However, 𝑔 is not
> reliable since (1) is not satisfied."
> — <https://arxiv.org/pdf/1902.06977>, Example 1

And the reason nobody fits the between-class structure is stated plainly by Guo
et al. (ICML 2017), who tried the full-matrix version and abandoned it:

> "Matrix scaling performs poorly on datasets with hundreds of classes (i.e.
> Birds, Cars, and CIFAR-100), and fails to converge on the 1000-class ImageNet
> dataset. This is expected, since the number of parameters scales quadratically
> with the number of classes. Any calibration model with tens of thousands (or
> more) parameters will overfit to a small validation set, even when applying
> regularization."
> — <https://arxiv.org/pdf/1706.04599>, §5.2

So the field's decision tree is: full between-class structure (`O(K²)`, overfits,
abandoned) → vector scaling (`O(K)`, diagonal, no between-class structure) →
temperature scaling (`O(1)`, one knob). Dirichlet calibration (Kull et al.,
NeurIPS 2019) sits at the `O(K²)` end and is described by its own authors as
"equivalent to log-transforming the uncalibrated probabilities, followed by one
linear layer and softmax" — i.e. still a deterministic map producing marginals,
still incapable of a joint.

**Low-rank-plus-diagonal is the missing rung**: `O(Kr + K)` parameters, which is
exactly the regularization Guo says is needed, and it produces a *joint* over
latents rather than a map on marginals — which is what buys restricted-menu
conditioning and top-k. The lineage is multinomial probit with factor-structured
covariance (a known scalability trick in discrete choice); the novelty on offer
is doing it **post hoc on a frozen scorer** as a recalibration layer.

That framing — "the regularized matrix scaling that the calibration literature
gave up on, plus a joint" — is the strongest single-sentence pitch and it is
domain-independent. Every application below is a venue for it.

---

## Ranked candidates

### 1. Verbal autopsy: computer-coded cause-of-death assignment ★★★★★

The best fit found on either front. Two independent search paths converged on it.

**(i) Observable and frozen scorer.** Exactly right. One mutually exclusive
*underlying cause of death* per decedent, from a fixed list (PHMRC analysis list
is 34 adult causes; WHO/InterVA-5 list is ~60). The incumbents are literally
frozen scorers that rank causes: Tariff computes "a Tariff score ... for each COD
k" and, per the openVA toolkit paper, "produces the COD distribution for each
death in terms of their rankings instead of the probability distributions"
(<https://journal.r-project.org/articles/RJ-2023-020/>). InterVA-4/5 emit a
probability vector but then "identifies top 3 most likely causes, truncates
others to zero" and dumps the remainder into an "undetermined" bucket — a
crude, hand-set calibration hack begging to be replaced. InSilicoVA gives MCMC
posteriors. nbc4va gives naive-Bayes scores.

**(ii) Public data.** PHMRC Gold Standard Verbal Autopsy, 2005–2011: 12,530
records with gold-standard cause established from medical records, lab, pathology
and imaging — 7,841 adults, 2,064 children, 1,620 neonates, 1,005 stillbirths,
from Philippines, Mexico, Tanzania, India.
- GHDx record: <https://ghdx.healthdata.org/record/ihme-data/population-health-metrics-research-consortium-gold-standard-verbal-autopsy-data-2005-2011>
  (registration required since Jan 2024 — minor friction, not a gate)
- Loadable from R in one line: `read.csv(getPHMRC_url("adult"))`
- Code: <https://github.com/verbal-autopsy-software/openVA> (InterVA4, InterVA5,
  InSilicoVA, Tariff, nbc4va all on CRAN)
- Standard metrics already defined and implemented: CCC (chance-corrected
  concordance) and CSMF accuracy.

**(iii) Incumbent + documented limitation.** Verbatim:

> "Difficulties in finding sufficient cases to meet gold standard criteria as
> well as problems with misclassification for certain causes meant that the
> target list of causes for analysis was reduced to 34."
> — PHMRC design paper, <https://pophealthmetrics.biomedcentral.com/articles/10.1186/1478-7954-9-27>

> "A high proportion of VA cases (66%) reported respiratory symptoms, but only
> 18% of assigned hospital causes were respiratory-related."

On InSilicoVA's default configuration:

> "at best suboptimal, with poor cause-of-death predictive performance"
> and "the increased sensitivity was at the expense of other causes, which had
> significantly lower concordance."
> — <https://pmc.ncbi.nlm.nih.gov/articles/PMC5907465/>

That last clause is the correlated-error signature stated out loud: gains on one
cause are paid for by named other causes. Nobody models the trade-off; they
report it as a nuisance.

**(iv) Unique advantage.** Three, and all three are load-bearing here:
1. **Confusability is the whole scientific object.** Causes of death confuse in
   clinically meaningful blocks — respiratory causes with each other, infectious
   febrile illnesses with each other, cardiovascular with "other NCD". A fitted
   rank-`r` `V` *is* an estimate of that block structure, readable and
   publishable in its own right. This is a scientific deliverable, not just a
   metric improvement.
2. **Restricted menus are native, not contrived.** Cause lists legitimately
   differ by age module (neonatal vs child vs adult), by country (no malaria in
   Mexico), and by surveillance system. Practitioners *routinely* need "score
   this death against a different cause list" and currently just renormalize —
   the IIA error. This is the strongest real-world instance of restricted-menu
   inference found anywhere in this scouting exercise.
3. **CSMF is a set/aggregate quantity.** Cause-specific mortality *fractions* are
   what the field actually reports, and they are sums over the joint. Correct
   joint uncertainty on CSMF (not just point estimates) is a live open need.

**(v) Minimal demo.** Fit Tariff (or nbc4va) on PHMRC adult data with a fixed
train split. Freeze it. On a calibration split, take its per-cause score vector
as latent means and fit `(V, D)` by exact ML against the gold-standard cause. On
a held-out test split report: (a) log-loss and CSMF accuracy vs softmax and vs
temperature scaling; (b) **restricted-menu test** — delete a random 30% of causes
from the menu, condition, compare against softmax renormalization on the deaths
whose true cause survives; (c) print the top eigenvectors of `V` and show they
recover clinically recognizable confusion blocks. All three are days of work; the
data loads in one R call and the metrics are already implemented in openVA.

**(vi) Venue.** *Population Health Metrics* (published the PHMRC papers and the
Tariff/InterVA validations — home turf), *BMC Medicine*, or *Journal of Clinical
Epidemiology*. A methods-first version fits *Biostatistics* or the
*Annals of Applied Statistics*.

**Risks.** GHDx registration. The VA community is small and R-centric — ship an R
wrapper or it will be ignored. Tariff scores are not on a natural latent scale,
so a per-cause affine pre-transform is needed before the Gaussian reading (this
is fine, but it is a real modeling step, not free).

---

### 2. Winter precipitation type ★★★★☆

The best *weather* fit by a wide margin, and it has a physical story that no
other candidate can match.

**(i) Observable and frozen scorer.** Yes on both. Precipitation type at the
surface is genuinely mutually exclusive and *nominal* (not ordered): rain, snow,
freezing rain, ice pellets. Frozen scorers exist in quantity — operational
diagnostics (Bourgouin, Ramer, NCEP algorithms) plus a recent ML literature.

**(ii) Public data.** Both sides are open:
- **mPING** (NSSL/OU crowdsourced precip-type reports) — "The whole mPING archive
  is freely accessible for researchers through an API" with registration.
  <https://mping.nssl.noaa.gov/> — crucially, mPING "contains reports of ice
  pellets which are not available from fully-automated ASOS sites."
- **NOAA Rapid Refresh (RAP) / HRRR** vertical thermodynamic profiles — fully
  open on AWS/NOAA Big Data Program.
- ASOS/METAR present-weather archives as an independent label source.

**(iii) Incumbent + documented limitation.** Two quotes, one recent and one
canonical.

Recent, directly on target (arXiv 2512.13899, Dec 2025):

> "existing methods struggle in thermodynamically ambiguous regimes, and most do
> not quantify forecast uncertainty from a single model run"

and, from the same paper, an explicit statement that the joint matters more than
the argmax:

> ambiguous conditions are "more robustly represented through the full
> probability distribution than through the dominant predicted class alone."

The canonical coherence complaint in meteorology is Wilks (2009), *Meteorological
Applications*, and it is worth quoting at length because it is the field's own
statement of the failure mode:

> "the most problematic consequence of separate MOS equations for different
> predictand thresholds is that forecasts derived from the different equations
> are not constrained to be mutually consistent."

> "Unless the regression functions f_{1/3}(x) and f_{2/3}(x) are exactly parallel
> (i.e. they differ only with respect to their intercept parameters, b_0) they
> will cross for some values of the predictor(s) x, leading to the nonsense
> result of p_{1/3} > p_{2/3}, implying Pr{q_{1/3} < V < q_{2/3}} < 0."

> "unless the separate logistic regression equations are exactly parallel,
> logically inconsistent sets of forecasts are inevitable for sufficiently
> extreme values of the predictor."
> — <https://civil.colorado.edu/~balajir/CVEN6833/paper-presentations/Wilks-2009.pdf>

**This is the key strategic observation for the weather front.** Meteorology
diagnosed exactly our problem — per-category fits are incoherent as a joint — and
solved it by *adding the threshold itself as a predictor*, which forces the
category boundaries to be parallel. That fix works **only because the categories
are ordered**. For *nominal* categories — precipitation type, weather type, storm
mode, cloud type — there is no threshold to add, and the field has no coherent
answer. The gap is precise and defensible.

**(iv) Unique advantage — and this is the best part.** Precipitation type is
governed by essentially **two continuous latent thermodynamic factors**: the
depth/warmth of the melting layer aloft, and the depth/coldness of the refreezing
layer near the surface. Rain, freezing rain, ice pellets and snow are four
regions of that two-dimensional plane. So the confusion structure among the four
classes is *generated by a rank-2 latent factor model* — which is precisely the
`(V, D)` object being proposed, with `r = 2`.

This makes the demo an unusually strong scientific claim rather than a benchmark
win: fit `r = 2` post hoc on outcomes alone, then show the recovered factor
loadings align with the known melting-layer / refreezing-layer physics. A
calibration layer that *rediscovers the governing thermodynamics from labels
alone* is a genuinely interesting result. No competitor (temperature scaling,
Dirichlet, vector scaling) can produce such a picture, because none of them has
latents.

Secondary: freezing rain is the high-cost, low-frequency class (ice storms), and
the cited paper notes its "reduced freezing rain probability of detection
reflects genuinely ambiguous thermodynamic environments" — a set-probability
readout (`P(precip ∈ {freezing rain, ice pellets})`, the "hazardous ice" set) is
directly decision-relevant for road and power utilities, and is exactly what the
layer computes exactly and softmax computes only by summing miscalibrated
marginals.

**(v) Minimal demo.** Take a published/reimplemented ptype classifier or a raw
NWP diagnostic as the frozen scorer over `{rain, snow, freezing rain, ice
pellets}`. Fit `(V, D)` with `r = 2` on one winter of mPING labels. Test on a
held-out winter. Report: multi-category Brier / ranked probability score and
log-loss vs temperature scaling; the two recovered factor loadings against
melting-layer physics; and calibrated `P(hazardous ice set)`. Small `N = 4` means
the ML fit is trivial and the whole thing is a laptop experiment.

**(vi) Venue.** *Weather and Forecasting* (AMS) or *Monthly Weather Review* — WAF
is the natural home and Wilks published the ordered-category fix there. *Artificial
Intelligence for the Earth Systems* (AIES) is the AMS venue most receptive to a
post-hoc ML layer. Also the NeurIPS/ICML climate workshops.

**Risks.** `N = 4` is small — the correlated layer has less room to beat softmax
than at `N = 34`. Mitigate by extending the class list (mixed types, drizzle,
freezing drizzle, wet snow — mPING supports finer types) or by pooling with a
second nominal weather task. mPING labels are crowdsourced and noisy; the cited
paper's "rigorous physical quality control" that "removed thermodynamically
implausible reports" is a necessary preprocessing step and a confound to handle
honestly.

---

### 3. Multi-class clinical diagnosis: ovarian tumour (ADNEX / IOTA) ★★★★☆

The cleanest *clinical* instance, with a literature that has already built the
evaluation apparatus but stopped short of the fix.

**(i) Observable and frozen scorer.** Yes. The IOTA **ADNEX** model is a
published multinomial logistic regression estimating risk over five mutually
exclusive histological classes: benign, borderline malignant, stage I primary
invasive, stage II–IV primary invasive, and secondary metastatic. It is *frozen
and published with coefficients*, developed on 5,909 patients across 24 centres
in 10 countries, and is in routine clinical use. That is an unusually good frozen
scorer: real, deployed, widely externally validated.

**(ii) Public data.** Weakest link on this candidate. IOTA patient-level data is
not open. Workable substitutes: (a) the many published external-validation
studies report class-wise counts and calibration curves that can be partially
reconstructed; (b) run the demo on a public multi-class clinical or imaging
dataset instead and cite ADNEX as motivation. Public alternatives with genuine
correlated-class structure: HAM10000 / ISIC dermoscopy (7 mutually exclusive
lesion classes; melanoma / nevus / BCC are the classic confusable triple),
DDXPlus (1.3M synthetic patients, 49 pathologies, with a labelled differential
diagnosis per patient — <https://huggingface.co/datasets/aai530-group6/ddxplus>),
and the ISIC archive generally.

**(iii) Incumbent + documented limitation.** The Van Calster / Steyerberg school
has explicitly built the multi-class calibration machinery and explicitly
identified the gap. Van Hoorde, Van Huffel, Timmerman, Bourne and Van Calster,
*J Biomed Inform* 2015:

> "When validating risk models (or probabilistic classifiers), calibration is
> often overlooked."

> "Calibration assessment is often overlooked, but is of importance for several
> applications where risk models may be used."
> — <https://www.sciencedirect.com/science/article/pii/S1532046415000027>
  (author manuscript: <https://lirias.kuleuven.be/retrieve/301205>)

Their proposed remedy is "a calibration framework based on a vector spline
multinomial logistic regression model" — i.e. a flexible *per-class* map on
log-ratios. Flexible in the marginals, still structureless between classes.
No covariance, no joint, no restricted-menu inference. Their own diagnosis of the
worst-calibrated model in the study is telling:

> "the calibration of naive Bayes was disappointing. An explanation for the
> latter might be that the independence assumption is unrealistic for the data
> leading to inaccurate estimates"

They locate the failure in an independence assumption, and then their fix does
not restore dependence.

The calibration hierarchy paper (Van Calster, Nieboer, Vergouwe, De Cock, Pencina,
Steyerberg, *J Clin Epidemiol* 2016) frames the ladder:

> "'Strong calibration' requires that the event rate equals the predicted risk
> for every covariate pattern"

> "Strong calibration is desirable for individualized decision support, but
> unrealistic ... development and external validation should focus on moderate
> calibration."
> — <https://lirias.kuleuven.be/retrieve/577522>

Note the exact parallel to the ML literature in §0: the strongest notion is
declared "utopia" / cursed by dimensionality, and the field retreats. Same
retreat, two vocabularies, twenty years apart. A method that makes a *structured*
version of the strong notion estimable speaks to both audiences at once, and that
cross-field framing is itself a paper.

Also documented, on ADNEX specifically:

> "two patients with the same risk for Stage I OC can have very different risks
> for a borderline tumor because there are a total of five subgroups"
> — ADNEX systematic review / meta-analysis,
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10875560/>

That sentence is a between-class dependence statement written by clinicians who
have no tool to express it.

**(iv) Unique advantage.** The five ADNEX classes have obvious confusion
structure: borderline vs stage I is the notoriously hard boundary; stage II–IV vs
metastatic is another. Restricted-menu inference is clinically real — once
frozen-section or imaging rules out metastatic disease, the clinician needs
`P(borderline | not metastatic)`, and today gets it by renormalizing, which is
wrong under any correlated truth. Set probabilities map onto the actual decision:
the referral decision is `P(malignant) = P({borderline, I, II–IV, metastatic})`,
a set probability, and the surgical-extent decision is a different set.

**(v) Minimal demo.** Preferred: ISIC/HAM10000 with a frozen public dermoscopy
classifier. Fit `(V, D)` on a calibration split. Show (a) log-loss and classwise-
ECE beating temperature scaling and vector scaling, matching or beating Dirichlet
calibration with an order of magnitude fewer parameters (this is the direct
answer to the Guo overfitting quote); (b) restricted-menu conditioning beating
renormalization when classes are deleted; (c) `V`'s leading eigenvectors
recovering melanocytic-vs-keratinocytic lesion structure. Then a short section
arguing the ADNEX use case with published coefficients.

**(vi) Venue.** *Journal of Clinical Epidemiology* or *Statistics in Medicine*
for the clinical framing (this is where the calibration-hierarchy conversation
lives); *Journal of Biomedical Informatics* for the tool framing; MICCAI or a
NeurIPS workshop for the imaging demo. A dual submission strategy is realistic.

**Risks.** IOTA data is closed. The clinical audience is conservative about new
machinery and will ask for the "moderate calibration is enough" objection to be
answered head-on — it must be, by showing a decision (restricted menu, or a set
probability) where moderate calibration demonstrably is *not* enough.

---

### 4. SARS-CoV-2 variant nowcasting ★★★☆☆ — good infrastructure, prior art problem

**(i) Observable.** *Not* an argmax, and this is the core mismatch. The US
SARS-CoV-2 Variant Nowcast Hub target is latent clade *proportions*, scored by
comparing "Ĉ_{l,t} ∼ Multinomial(N_{l,t}, θ̂_{l,t}) to the observed C_{l,t}"
(<https://arxiv.org/html/2606.07129v1>). You observe a count vector per
state-day, not one winner. The layer still applies (it induces a distribution
over the proportion vector), but the "mutually exclusive winner" pitch does not.

**(ii) Public data — excellent, the best of any candidate.**
<https://github.com/reichlab/variant-nowcast-hub> — ~102 rounds of
`target-data/oracle-output/`, `auxiliary-data/modeled-clades/`,
`auxiliary-data/scores/`, and `model-output/` with nine teams posting 100
posterior samples per clade/date/location as parquet (`blab-open_hier_mlr`,
`LANL-CovTransformer`, `UMass-HMLR`, `Hub-baseline`). Weekly cadence since
Oct 2024. GenBank/Nextstrain-derived, so no GISAID gate. Plus `evofr`
(<https://github.com/blab/evofr>) and the evaluation repo
<https://github.com/epiforecasts/evalvariantnowcasthub>.

**(iii) Incumbent + documented limitation.** Multinomial logistic growth (MLR /
hierarchical MLR). The independence assumption is stated by the hub:

> "The use of a multinomial distribution assumes that, conditional on the mean
> prevalence, clade assignments for the sequenced samples are independent and
> have probability of being in each clade equal to the population probabilities
> θ_{l,t}."

And the overconfidence is confessed by the incumbents themselves (Abousamra,
Figgins & Bedford, *PLOS Comput Biol* 2024):

> "However, we observe that coverage is generally lower than ideal with
> predictive coverage under 50% for countries with the most sequencing (S8(B)
> Fig). We believe this may be due to a combination of over-dispersion of
> sequence counts relative to the multinomial sampling assumption as well as
> clade-level growth advantages changing through time as clades evolve."
> — <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012443>

Sub-50% coverage where data is densest is a spectacular miscalibration and would
normally make this candidate #1.

**(iv) Why it drops to #4: prior art.** MacArthur, Robacker, Ray, Rogers, Reich &
Griffin (<https://arxiv.org/abs/2605.22676>) **already fit a full correlation
matrix across variants** — "Σ_p = diag{σ_p} Ω_p diag{σ_p}, where Ω_p is a
(V−1)×(V−1) correlation matrix" under "an LKJ(2) prior" — and already report the
headline effect:

> "Moving from ICD to CCD, and thus from an individual to correlated, leads to a
> rise in uncertainty and a dramatic shift in the mean predictions. The full
> correlation matrix leads to more uncertainty in predictions by allowing trends
> for each variant to depend more on the observed behavior of and uncertainty
> about all variants."

So "correlated latents across lineages" is taken. Their stated open problem is
*spatial* correlation, not phylogenetic: future work is "relaxation of the
independence assumption across locations and predictors by using spatially
correlated models."

**The one genuinely open angle is IIA, via menu churn.** The clade menu changes
every round: "Any clades with prevalence of less than 1% are grouped into an
'other' category," capped at 10, and "The reference phylogenetic tree that
defines clades changes over time." Translating a nowcast onto a *changed* menu —
a clade splits, or collapses into "other" — is a restricted-menu problem that no
MLR handles coherently and that arises 52 times a year by construction. If this
front is pursued, that is the wedge, not correlation per se.

**(v) Minimal demo.** Pull `Hub-baseline` and `blab-open_hier_mlr` samples for
closed rounds; convert samples to latent means; fit `(V, D)` on rounds 1–60
against `oracle-output`; score 61–102 with the hub's own energy score and
categorical Brier score. The scoring harness already exists.

**(vi) Venue.** *Epidemics*, *PLOS Computational Biology*, or a NeurIPS/ICML
time-series forecasting workshop.

**Honest negatives.** Not an argmax observable; correlated-variant modelling is
published; the incumbents name *drift in growth advantage* as a co-cause of
miscalibration, which a static post-hoc `(V, D)` does not fix; and the "frozen
scorer" is a posterior sample set rather than logits.

---

## Weak fits, with reasons (do not pursue)

**Seasonal tercile forecasts and ENSO phase.** `N = 3`, *ordered*, and the
incumbent already uses a Gaussian latent — "a Gaussian fitting method is used for
the estimation of tercile-based categorical probabilities" (IRI). With three
ordered categories generated by thresholding one latent, the correlation
structure is pinned by the ordering and there is essentially nothing for `(V, D)`
to learn. ENSO phase (El Niño / Neutral / La Niña) is the same object. The
documented skill problems there are about the *spring predictability barrier* and
neutral-state skill, which a recalibration layer cannot touch. Skip.

**Continuous-predictand ensemble postprocessing (EMOS, BMA, quantile
regression).** Wrong shape entirely — the flagship review (Vannitsem et al.,
*BAMS* 2021, <https://arxiv.org/abs/2004.06582>) is about continuous variables
end to end; its stated limitation is "the need to select a suitable parametric
family to describe the distribution of the target variable remains a limitation
for parametric postprocessing methods." Worth noting as a *negative* finding:
that review, the field's most cited statement of open problems, contains
essentially **no treatment of nominal categorical predictands at all**. Its five
listed future challenges are about implementation, resources, technique
selection, new datasets and communication — none about categorical calibration.
That is a gap by omission, which is weaker evidence than a complaint but does
confirm nobody is working the problem.

**Severe-weather convective outlook categories (MRGL/SLGT/ENH/MDT/HIGH).**
Ordered, and the SPC hazard probabilities (tornado / hail / wind) are *not*
mutually exclusive — they are independent hazard probabilities for the same
point. Fails the exclusivity requirement. Convective *storm mode*
(supercell / QLCS / disorganized) does qualify and is a plausible small side
demo, but labels are scarce and hand-curated.

**Competing-risks survival with shared frailty.** Adjacent and conceptually
sympathetic — the literature says "a single frailty term for all causes of
failures is not suitable to explain overall randomness, and independent frailty
terms for each cause will ignore the dependency structure", which is precisely
the low-rank-plus-diagonal argument in survival vocabulary. But this is a *time-
to-event* problem with censoring, not a frozen-scorer recalibration problem, and
entering it means competing with a mature copula/frailty literature on its own
terms. Better used as a one-paragraph "related work" citation showing the
low-rank-plus-diagonal idea is independently motivated in biostatistics.

**Syndromic surveillance (NSSP/ESSENCE).** The useful visit-level data is
jurisdiction-gated. Skip on data access alone.

---

### 5. Judgmental forecast aggregation ★★★☆☆ — best story, worst structure

The most *vivid* motivating narrative found anywhere, undercut by a structural
problem that the other candidates do not have.

**(i) Observable and frozen scorer.** Exactly right on paper. Metaculus Multiple
Choice questions "present a set of options that are mutually exclusive and cover
all possibilities, and resolve to exactly one of them. Options can be added or
removed as the question evolves. Forecasters assign a probability to each option,
and those probabilities sum to 100%" (<https://www.metaculus.com/faq/>). Frozen
scorers are free: the Community Prediction, a plain mean/median of individual
forecasts, or a market price vector.

**(ii) Public data — counted, not guessed.**
- **Good Judgment Project / ACE**, Harvard Dataverse, doi:10.7910/DVN/BPCDH5.
  Direct count from `ifps.csv`: **617 questions; option counts {2: 493, 3: 50,
  4: 54, 5: 20} → 124 multi-option, of which 116 closed with a recorded
  outcome.** Individual forecasts in `survey_fcasts.yr1–yr4.tab` (~300 MB), so
  crowd disagreement per question is observable.
- **Metaculus**: API now token-gated — both `api.metaculus.com` and
  `/api2/questions/` return "The API is only available to authenticated users."
  A free account gets a token. Multiple choice launched December 2023. Volume
  proxy: the Q2 2026 AI Benchmark alone was "binary (58%), numeric (21%), and
  multiple-choice (21%)" of 348 questions ≈ 73 MC/quarter; plausibly 500–1,000
  resolved MC overall, unverified without a token.
- **Manifold**: fully public, unauthenticated —
  `GET /v0/search-markets?contractType=MULTIPLE_CHOICE&filter=resolved` works;
  `shouldAnswersSumToOne=true` markets "auto-arbitrage so that probabilities add
  up to 100%". Thousands available; question quality is the trade.
- **Polymarket**: `gamma-api.polymarket.com/events?closed=true` is public;
  negRisk events are mutually exclusive.
- <https://github.com/niplav/iqisa> bundles GJP and related datasets.

**(iii) Incumbent + documented limitation.** The incumbent is per-option
recalibration followed by renormalization, and it is confessed in writing.
Metaculus, in production, verbatim:

> "For Multiple Choice Questions, the Community Prediction is a weighted median
> of the individual forecaster probabilities, renormalized to sum to 1"
> — <https://www.metaculus.com/faq/>

Han & Budescu (*Judgment and Decision Making* 2021) state the gap and then
concede the coherence break in the same paper:

> "Previous work on recalibration has focused on binary forecasts"

> "If we recalibrate specific bin probabilities, we need to add one more step of
> normalization to make the sum of C recalibrated bin probabilities equal to 1."
> — <https://www.sas.upenn.edu/~baron/journal/21/210914/jdm210914.html>

That is Wilks's complaint again, in a third vocabulary: fit each category
separately, discover the result is not a coherent joint, patch it by
renormalizing. The extremizing literature — Baron, Mellers, Tetlock, Stone &
Ungar, "Two Reasons to Make Aggregated Probability Forecasts More Extreme"
(*Decision Analysis* 2014), and Satopää et al., "Combining multiple probability
predictions using a simple logit model" (*IJF* 2014) — is entirely binary.
Satopää's partial-information framework models correlation **across forecasters**,
never across answer options.

**(iv) Unique advantage.** The IIA story is unusually concrete here. Metaculus's
own FAQ says "Options can be added or removed as the question evolves" — so
renormalizing after an option is removed is an explicit, documented,
in-production IIA assumption applied to real money and real reputations.
Near-duplicate options splitting probability mass is endemic in user-authored
Manifold markets. And "one of these three candidates" is a set probability
forecasters actually ask for.

**(v) Minimal demo.** GJP: for each of the 116 resolved `N ∈ {3,4,5}` questions,
freeze the scorer as the log of the renormalized per-option mean of final-week
forecasts. Fit rank-1-plus-diagonal by exact ML, leave-one-question-out. Report
multi-class Brier and log-loss against (a) the raw mean and (b) per-option
Satopää extremized-logit-then-renormalize. Scale up on ~2,000 resolved Manifold
sum-to-one markets using the price vector as the frozen scorer.

**(vi) Venue.** *International Journal of Forecasting* (home of Satopää 2014) is
the natural fit; *Decision Analysis* and *Judgment and Decision Making* are
realistic alternates.

**(vii) Why it does not rank higher — the menu does not repeat.** This is the
decisive objection and it is structural, not fixable by more data. In every
other candidate the menu is *fixed and shared across observations*: every
verbal-autopsy death is scored against the same 34 causes, every precipitation
observation against the same 4 types, every adnexal mass against the same 5
histologies. That repetition is what makes `(V, D)` estimable by exact ML —
thousands of draws from one menu. In forecast aggregation, **option `k` of
question A has nothing to do with option `k` of question B**, and each question
yields exactly one realized outcome. There is no pooling. To fit anything you
must tie the covariance to observables (forecaster-disagreement cross-products,
option-text embeddings), which abandons the clean "exact ML on held-out
outcomes" pitch and turns the project into a different, harder paper.

Secondary objections: 116 high-quality questions is a power problem against a
one-parameter extremizer; markets are already coherent by construction
(Polymarket negRisk, Manifold sum-to-one), so the coherence argument only bites
on poll-style aggregates, where the known miscalibration is longshot bias — which
temperature scaling already handles.

Net: a good methods paper with a modest empirical section, sold on the
option-removal IIA story rather than on benchmark wins. Not the lead.

---

## The selection criterion that fell out of this exercise

Worth recording, because it separated the candidates more cleanly than anything
else and should be applied to future scouting:

> **Does the menu repeat?** `(V, D)` is estimable by exact maximum likelihood
> only when many observations are scored against the *same* `N` outcomes. A fixed,
> shared, repeated menu is the binding requirement — not sample size, not the
> quality of the frozen scorer, not the vividness of the IIA story.

Verbal autopsy (34 fixed causes), precipitation type (4 fixed types) and clinical
diagnosis (5 fixed histologies) pass cleanly. Variant nowcasting passes weakly —
the clade menu is stable within a window but churns weekly by design. Forecast
aggregation fails: one outcome per bespoke menu. Ranking follows this criterion
almost exactly, which is a good sign it is the right one.

---

## Ranking summary

| # | Candidate | Argmax? | Menu repeats? | Public data | Incumbent gap | Overall |
|---|---|---|---|---|---|---|
| 1 | Verbal autopsy cause-of-death | Yes | Yes, 34 causes | Yes (PHMRC, registration) | Strong, quoted | ★★★★★ |
| 2 | Winter precipitation type | Yes | Yes, 4 types | Yes (mPING + RAP, open) | Strong, quoted | ★★★★☆ |
| 3 | Multi-class clinical diagnosis | Yes | Yes, 5 classes | Proxy only (ISIC/DDXPlus) | Strong, quoted | ★★★★☆ |
| 4 | SARS-CoV-2 variant nowcast | No (counts) | Weakly — churns weekly | Excellent | Strong, but prior art | ★★★☆☆ |
| 5 | Forecast aggregation | Yes | **No** — bespoke per question | 116 GJP + ~2k Manifold | Strong, quoted | ★★★☆☆ |
| — | Seasonal tercile / ENSO | Yes | Yes, 3 ordered | Yes | Weak — Gaussian already | ★☆☆☆☆ |
| — | Competing risks / frailty | No | N/A | Mixed | N/A — cite as related work | ☆☆☆☆☆ |

**Recommended play.** Lead with **verbal autopsy** — it is the only candidate
where the argmax observable, the frozen scorer, the confusable-class science, and
a *native* restricted-menu use case all coexist, and the target journal already
publishes exactly this kind of validation study. Pair it with **precipitation
type** as a second, physically interpretable demo: `N = 4` with a two-factor
thermodynamic ground truth turns the covariance from a nuisance parameter into a
verifiable scientific claim. Use §0 (canonical calibration is cursed; matrix
scaling overfits; low-rank-plus-diagonal is the missing rung) as the framing for
both, since it is the argument that makes a single method paper out of two
unrelated application domains.
