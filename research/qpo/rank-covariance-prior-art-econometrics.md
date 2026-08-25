# Prior art: estimating Sigma of a Gaussian random-utility model from rank-level data

Front: econometrics and statistics of ranked data. Question: has anyone identified or
estimated the covariance (or factor loadings) of a multinomial-probit / Thurstonian
random-utility model from multi-rank data — win + place/show frequencies, top-k inclusion,
or ranking distributions — especially from a single market/menu?

Compiled 2026-08-25. Fetched texts in scratchpad `rankhunt/` (Maydeu-Olivares & Bockenholt
2005 full PDF; Maydeu-Olivares IMPS 2001 proceedings full PDF; Nair-Bhat et al. rank-ordered
probit full PDF; Bunch 1991; Marden lecture notes; Fok-Paap-van Dijk working paper; Stata
rologit manual). Hausman & Ruud (1987) itself is paywalled (ScienceDirect 403; no OA copy of
the J. Econometrics version or the 1986 UC Berkeley WP 8605 found); its content below is
reconstructed from the publisher abstract and from detailed descriptions in citing papers,
flagged as such.

**Bottom line.** The psychometric Thurstonian literature (Maydeu-Olivares, Bockenholt, Tsai,
Yao, Yu, Dansie) has done, for small menus, essentially the single-menu program: fit an
unrestricted or factor-analytic covariance matrix of Gaussian utilities to the observed
distribution of rankings of one fixed choice set, with the identification theory (what gauge
freedom remains) worked out. The econometrics literature never did this from a single market:
rank-ordered probit papers estimate Sigma from individual ranking data *with covariates
varying across respondents*, and Hausman-Ruud's contribution is about rank-depth *scale*
(logit heteroscedasticity), not covariance. Nobody found estimates Sigma from *aggregated
top-k inclusion shares alone* (win + place + show frequencies without joint ranking events);
the psychometric estimators all consume at least the bivariate margins of the ranking
distribution (joint pairwise-order frequencies), which requires observing rank patterns, not
just marginal top-k shares.

---

## 1. Hausman & Ruud (1987) — VERDICT: DISTINCT (logit family; scale, not covariance)

Hausman, J. A. & Ruud, P. A. (1987). "Specifying and testing econometric models for
rank-ordered data." *Journal of Econometrics*, 34(1-2), 83-104. Circulated as UC Berkeley
Dept. of Economics Working Paper 8605 (1986) under the title "Specifying and Testing
Econometric Models for Rank-ordered Data with an Application to the Demand for Mobile and
Portable Telephones."

What they estimate (from the publisher abstract and citing papers; full text not obtained):
the rank-ordered logit (ROL) of Beggs-Cardell-Hausman (1981) is the base specification; they
propose two specification tests (a Hausman test of IIA across explosion depths, and a test of
consistency of top-ranked vs. bottom-ranked information) and an alternative estimator that
"generalizes the rank-ordered logit specification to allow for a form of heteroscedasticity
that permits top ranked choices to be more precisely ranked than bottom ranked choices"
(publisher abstract wording as reproduced by RePEc/ScienceDirect search snippets).

Verbatim from Fok, Paap & van Dijk (Econometric Institute Report EI 2007-07, p. 7-8), on
what H&R did:

> "Furthermore, Hausman and Ruud (1987, p. 89) notice in an application on mobile phones
> that including more ranks in the estimation procedure leads to a decline in the absolute
> magnitude of the parameters."

> "Hausman and Ruud (1987) also estimate a model where each rank in the estimation receives
> a weight. This makes it possible that the most preferred rank contain more information
> than lower ranks. These weights are estimated alongside the model parameters."

Verbatim from the Stata `rologit` manual (Methods discussion):

> "A formalization of this idea is a heteroskedastic version of the rank-ordered logit model
> in which the scale of the random term increases with the number of decisions made (for
> example, Hausman and Ruud [1987])."

Answer to the specific question posed: **yes, rankings identify parameters that first choices
cannot — but in H&R those are rank-depth scale (precision) parameters within an IID Gumbel
model, not utility covariances.** The relative scale of the random term at explosion stage 2,
3, ... is meaningless/inestimable in first-choice data (only one scale, normalized away);
with rankings the scale *profile across ranks* becomes estimable. There is no Sigma anywhere:
the ROL keeps utilities independent. Individual survey data with covariates (telephone
demand), not a single aggregate market.

Distinguishing dimension: free-vs-restricted covariance (they have none) and logit vs probit.
The conceptual precedent — deeper rank data buys identification of second-moment-type
parameters — is there; the object identified is different.

## 2. Beggs, Cardell & Hausman (1981) — VERDICT: DISTINCT

"Assessing the potential demand for electric cars." *Journal of Econometrics*, 17(1), 1-19.
Origin of the ROL / exploded logit. IID extreme-value utilities, individual conjoint
rankings, covariates. No covariance object at all. Included only as the root of the
econometric ranked-data tree.

## 3. Rank-ordered probit strand (Hajivassiliou-Ruud; Layton-Levine; Nair-Bhat et al.) — VERDICT: CLOSE (estimation of Sigma from rankings, but micro data with covariates, never a single menu without covariates)

**Hajivassiliou & Ruud (1994)**, "Classical estimation methods for LDV models using
simulation," *Handbook of Econometrics* Vol. IV: introduces the rank-ordered probit (ROP) as
a simulation-estimable generalization of MNP. Computation, not identification from rank
distributions per se. From Nair et al. (below), verbatim:

> "If individuals do not necessarily sequence from best to worst, the rank-ordered probit
> (ROP), introduced as a generalization of the Multinomial Probit model in Hajivassiliou and
> Ruud (1994), constitutes a more flexible behavioral structure to deal with rank-ordered
> data. Besides, the ROL maintains independence across the utilities of the ranked
> alternatives, while the ROP allows a full covariance structure across the alternatives
> (subject to identification considerations)."

**Layton & Levine (2003)**, "How Much Does the Far Future Matter? A Hierarchical Bayesian
Analysis of the Public's Willingness to Mitigate Ecological Impacts of Climate Change,"
*JASA* 98, 533-544. Gaussian latent-utility model for survey responses with "complex ordinal
structures that result in nonrectangular probabilities" (their abstract, verbatim via
OpenAlex: "The responses to the survey questions are a consequence of latent utilities with
complex ordinal structures that result in nonrectangular probabilities... we show how the
nonrectangular probabilities fit neatly into a hierarchical Bayesian model. We show how to
fit these models using the Gibbs sampler, overcoming problems in parameter identification to
improve mixing of the induced Markov chain."). Hierarchical Bayes over individuals with
covariates (scenario attributes); the covariance action is in random coefficients, not an
unrestricted alternative-level Sigma from a fixed menu. CLOSE on machinery, DISTINCT on data
setting.

**Nair, G. S., Bhat, C. R., Pendyala, R. M., Loo, B. P. Y., & Lam, W. H. K. (2019)**, "On the
Use of Probit-Based Models for Ranking Data Analysis," *Transportation Research Record*
2673(4) (full PDF in scratchpad). MACML estimation of ROP with a general error covariance.
Their identification statement, verbatim:

> "As usual, appropriate scale and level normalization must be imposed on Λ for
> identification of parameters. Specifically, only utility differentials matter in ranking
> choice models, just as in traditional discrete choice models... only the elements of the
> covariance matrix Λ₁ of [differenced errors] are estimable... An additional normalization
> needs to be imposed on Λ because the scale is also not identified... The Λ matrix so
> constructed is fully general. Also, as in multinomial probit (MNP) models, identification
> is tenuous when only individual-specific covariates are used (see Keane, 1992...). In
> particular, exclusion restrictions are needed in the form of at least one individual
> characteristic being excluded from each alternative's utility..."

Application: 2015 Puget Sound survey, four AV alternatives, thousands of respondents each
providing a ranking; random coefficients plus error covariance; in the final model they
retain exactly one free error correlation ("Error Correlation: AV as Taxi with Backup Driver
and AV as Taxi Without Backup Driver 0.790 (t-stat = 10.338)").

Distinguishing dimension: individual data with covariates across respondents (identification
leans on covariate variation and exclusion restrictions, cf. their Keane 1992 caveat); the
differenced-covariance gauge (level + scale normalization) is stated but no result about what
the *ranking distribution of a single menu without covariates* identifies.

## 4. Thurstonian ranking psychometrics — the direct hits

### 4a. Maydeu-Olivares & Bockenholt (2005) — VERDICT: DIRECT HIT on the model class

Maydeu-Olivares, A., & Bockenholt, U. (2005). "Structural equation modeling of
paired-comparison and ranking data." *Psychological Methods*, 10(3), 285-304. Full PDF in
scratchpad.

Abstract, verbatim:

> "An overview of Thurstonian models for comparative data is provided, including the
> classical Case V and Case III models as well as more general choice models with
> unrestricted and factor-analytic covariance structures. ... The authors show how to embed
> these models within a more familiar structural equation modeling (SEM) framework. The
> different special cases of Thurstone's model can be estimated with a popular SEM
> statistical package, including factor analysis models for paired comparisons and
> rankings."

Model: t ~ N(mu_t, Sigma_t) over the n alternatives of ONE fixed menu; observed rankings are
the order of t; rankings are recoded as the ñ = n(n-1)/2 binary paired comparisons y = I{A t > 0}.
"Thus, only the mean and covariance parameters of the latent difference responses, mu_y* and
Sigma_y* [= A Sigma_t A'], are related directly to the observed choice data."

Covariance structures fitted (verbatim): "(a) the unrestricted model, in which the mean
vector mu_t and the covariance matrix Sigma_t are unrestricted; (b) the Case III model...
Sigma_t is specified to be diagonal but otherwise unrestricted; and (c) the Case V model...
Sigma_t = sigma^2 I."

Identification of the unrestricted model from ranking data, verbatim:

> "The unrestricted Thurstonian model requires three identification constraints: (a) fix one
> of the item means, say mu_n = 0; (b) fix all the covariances involving the last latent
> utility to 0; and (c) fix the variance of the first and last latent utilities to 1."

So Sigma_t is identified from the ranking distribution exactly up to the difference gauge
(n+1 constraints: one location, n zero covariances-with-numeraire absorbed, two scale
pins) — i.e., A Sigma_t A' up to scale is what rankings deliver. Degrees-of-freedom
accounting for ranking data, verbatim:

> "Because rankings give rise to only a subset of all possible paired comparison data (no
> intransitivities can be observed), an adjustment is needed... there will be
> r = n(n-1)(n-2)/6 redundancies among the thresholds and tetrachoric correlations estimated
> from the binary variables (Maydeu-Olivares, 1999)."

**Factor structure**: Section "Thurstonian Factor Models for Paired Comparisons and Ranking
Data" fits Sigma_t = Lambda Lambda' + Psi^2 (uncorrelated common factors, echelon Lambda for
rotational identification), plus for ranking data the extra difference-gauge constraints:
"(a) fix the mean of the last item... mu_n = 0; (b) fix all factor loadings involving the
last item to 0; and (c) fix the unique variance of the last item to 1." This is literally a
(V, D) factor decomposition of utilities estimated from ranking data.

Data/estimation: multiple-judgment sampling — N respondents each rank the same n
alternatives, i.e., N i.i.d. draws from ONE menu's ranking distribution (this IS the
single-market setting, no covariates needed). Ranking application: N = 57 Spanish psychology
sophomores ranking n = 4 career areas; unrestricted, Case III, Case V all fitted and
compared by chi-square. Estimation is limited-information: thresholds + tetrachoric
correlations of the induced binary paired comparisons (univariate + bivariate margins of the
ranking distribution), then (D)WLS. Simulations elsewhere in the paper: "300 observations
appear to yield accurate parameter estimates, standard errors, and goodness-of-fit tests for
an unrestricted ranking model for seven choice alternatives."

Also, verbatim, on top-k: "Applications in which respondents... rank some but not all of the
choice alternatives (i.e., partial rankings) can also be handled, requiring only minor
modifications to the approaches presented in this article."

What it lacks relative to the qpo/kinetics program: small n (4-7; estimation is via ñ(ñ+1)/2
tetrachorics, brutal for n ~ 10+ without their limited-information trick, and the menu is the
same for every observation); needs joint pairwise-order frequencies (bivariate margins of the
ranking distribution), not just marginal win/place/show shares; no varying menus, no
market-share/odds inputs; no interest in the "correlation matters for exotics pricing"
question. But as a *model class + identification + factor structure from rank data on one
menu*, this is the closest published thing found.

### 4b. Maydeu-Olivares (1999) — VERDICT: DIRECT HIT (estimation theory + redundancy/identification counting)

Maydeu-Olivares, A. (1999). "Thurstonian modeling of ranking data via mean and covariance
structure analysis." *Psychometrika*, 64(3), 325-340. (Source of the redundancy formula and
the reduction of ranking likelihood to mean/covariance structure with dichotomized MVN.)
Companion identification statements in Maydeu-Olivares (2001 IMPS proceedings, "On
Thurstone's model for paired comparisons and ranking data," full PDF in scratchpad),
verbatim:

> "Given the comparative nature of the data, in all cases it is necessary to set the
> location of the elements of mu_t and the location of the elements in each of the rows
> (columns) of Sigma_t. Arbitrarily, we set mu_n = 0 and Sigma_t = P_t (a matrix with ones
> along its diagonal). In addition, in the Thurstonian ranking model it is necessary to fix
> one of the elements of P_t (Dansie, 1986). Arbitrarily, we set rho_{n,n-1} = 0."

> "Unrestricted Thurstonian models and many restricted models are not identified from
> univariate information only. However, any Thurstonian model can be identified as soon as
> bivariate information is employed."

(Here "univariate information" = the marginal frequencies of the induced binary paired
comparisons, i.e., pairwise beat rates P(i ranked above j); "bivariate" = joint frequencies
of two paired comparisons. Directly relevant precedent for "win probabilities alone
under-identify Sigma; joint rank events restore identification.")

> "A very interesting feature of Thurstone's original model is that since A is of rank n-1,
> Sigma_y* has rank n-1. As a result, Thurstone's model assigns zero probabilities to all
> the 2^ñ - n! intransitive paired comparisons patterns... Thurstone's model is not a
> plausible model for multiple judgment paired comparisons data, but it may be a suitable
> model for ranking data (Maydeu-Olivares, 1999)."

### 4c. Tsai (2000) — VERDICT: DIRECT HIT on identification (what rankings can and cannot pin down)

Tsai, R.-C. (2000). "Remarks on the identifiability of Thurstonian ranking models: Case V,
Case III, or neither?" *Psychometrika*, 65(2), 233-240. Abstract, verbatim:

> "It is well-known that the representations of the Thurstonian Case III and Case V models
> for paired comparison data are not unique. Similarly, when analyzing ranking data, other
> equivalent covariance structures can substitute for those given by Thurstone in these
> cases. That is, we may more broadly define the family of covariance structures satisfying
> Case III and Case V conditions. This paper introduces the notion of equivalence classes
> which defines a more meaningful partition of the covariance structures of the Thurstonian
> ranking models. In addition, the equivalence classes of Case V and Case III are completely
> characterized."

This is the clean statement that the ranking distribution identifies Sigma_t only up to an
equivalence class (the c c' + gamma-shift gauge in differenced space); i.e., exactly the
"two gauges" issue. See also Tsai & Bockenholt (2002), "Remarks on the identifiability of
Thurstonian paired comparison models under multiple judgment," *Psychometrika* 67, 255-261.
Purely identification theory, small-n psychometric framing, no aggregate-share data.

### 4d. Dansie (1985, 1986) — VERDICT: CLOSE (earliest estimability counting)

Dansie, B. R. (1985). "Parameter estimability in the multinomial probit model."
*Transportation Research Part B*, 19(6), 526-528. (First-choice MNP estimability; the
counting that Bunch (1991, Transp. Res. B 25(1), 1-12) formalizes — Bunch full PDF in
scratchpad, crediting Dansie for pointing out unidentified specifications.)

Dansie, B. R. (1986). "Normal order statistics as permutation probability models." *Applied
Statistics (JRSS C)*, 35(3), 269-275. Abstract, verbatim (via OpenAlex):

> "The use of normal order statistics in the analysis of permutation data is considered. A
> problem with the specification of the model is considered. An extension of Henery's (1981)
> approximation to the normal model is outlined, and the results are applied to a motor
> vehicle evaluation experiment."

The "problem with the specification" is the identifiability gap cited by Maydeu-Olivares
(fixing one correlation of P_t in the ranking model). This is the paper the psychometricians
credit for the ranking-model gauge count. Note the domain: permutation (ranking) data,
normal order statistics — the same mathematical object as a race finish order.

### 4e. Bockenholt (1992) — VERDICT: DIRECT HIT on the top-k/partial-rank dimension

Bockenholt, U. (1992). "Thurstonian representation for partial ranking data." *British
Journal of Mathematical and Statistical Psychology*, 45(1), 31-49. Abstract, verbatim (via
OpenAlex):

> "This paper presents a unified framework for applying the Thurstonian approach to partial
> ranking data that includes paired comparison data and first choices. As a result, several
> new Thurstonian ranking models are introduced by imposing different constraints on the
> covariance matrix of the random utilities and their mean scale values. Furthermore, the
> estimation of probabilities for a multivariate normal distribution by numerical
> integration procedures and the Clark algorithm are discussed. To illustrate the approach,
> data from two partial ranking experiments are analysed."

Partial rankings here are individual-level partial rank *patterns* (e.g., top-k orderings),
with first choices as the degenerate case — a spectrum from win-only to full-rank data inside
one Thurstonian framework with structured Sigma. Still individual pattern data on a small
fixed menu, not aggregated inclusion shares. See also Bockenholt (1993), "Applications of
Thurstonian models to ranking data," in Fligner & Verducci (Eds.), *Probability Models and
Statistical Analyses for Ranking Data*, Springer, 157-172; and Bockenholt (1990),
"Multivariate Thurstonian models," *Psychometrika* 55, 391-403.

### 4f. Yao & Bockenholt (1999); Yu (2000); Johnson & Kuhn (2013) — VERDICT: CLOSE (Bayesian computation for exactly this estimation problem)

Yao, G., & Bockenholt, U. (1999). "Bayesian estimation of Thurstonian ranking models based
on the Gibbs sampler." *BJMSP*, 52(1), 79-92. Abstract, verbatim:

> "This paper presents a Gibbs sampler for the estimation of Thurstonian ranking models.
> This approach is useful for the analysis of ranking data with a large number of options.
> Approaches for assessing the goodness-of-fit of Thurstonian ranking models based on
> posterior predictive distributions are also discussed. Two simulation studies and two
> ranking studies are presented to illustrate that the Gibbs sampler is a promising solution
> to the numerical problems that previously plagued the estimation of Thurstonian ranking
> models."

Yu, P. L. H. (2000). "Bayesian analysis of order-statistics models for ranking data."
*Psychometrika*, 65(3), 281-299. Abstract, verbatim (via OpenAlex):

> "In this paper, a class of probability models for ranking data, the order-statistics
> models, is investigated. We extend the usual normal order-statistics model into one where
> the underlying random variables follow a multivariate normal distribution. Bayesian
> approach and the Gibbs sampling technique are used for parameter estimation. ... The
> proposed method is applied to analyze the presidential election data of the American
> Psychological Association (APA)."

The APA application is n = 5 candidates and thousands of complete ballots of ONE election —
i.e., correlated-normal utilities estimated from a single menu's ranking distribution. That
is the qpo data shape (one market, many i.i.d. rank draws), done by MCMC on latent utilities.
Also: Johnson, T. R., & Kuhn, K. M. (2013). "Bayesian Thurstonian models for ranking data
using JAGS." *Behavior Research Methods*, 45(3), 857-872 (abstract in scratchpad via PubMed;
turnkey JAGS implementations of these models, incl. regression structure); Montgomery,
Bradford & Lee (2024), *BRM* 56, 8091-8104, Thurstone model for partial rankings in JAGS
(wisdom-of-crowds framing, Case-V-like, no free Sigma). Distinguishing dimension:
computation/estimation, not identification; complete or partial rank *patterns* per judge,
not aggregate shares; free Sigma only up to the usual gauge and only for small n.

## 5. Adjacent but distinct

- **Fok, D., Paap, R., & van Dijk, B. (2012).** "A rank-ordered logit model with unobserved
  heterogeneity in ranking capabilities" (*J. Applied Econometrics* 27; WP EI 2007-07 in
  scratchpad). Latent-class ROL where classes differ in how many ranks are meaningful.
  Heterogeneity in ranking reliability, no covariance. DISTINCT.
- **Chapman & Staelin (1982)** *JMR* 19, 288-301: explosion depth choice for ROL. DISTINCT.
- **Koop & Poirier (1994)** *J. Applied Econometrics* 9, 369-388: Bayesian ROL, Ontario
  voters. DISTINCT (logit).
- **Bockenholt (2001).** "Mixed-effects analyses of rank-ordered data," *Psychometrika* 66,
  45-62: Bock/Luce logit synthesis with random effects. DISTINCT (logit family).
- **Marden (1995/2019 notes**, *Analyzing and Modeling Rank Data*; notes PDF in scratchpad):
  best textbook treatment of Thurstonian ("order statistic") ranking models — notes Thurstone's
  parameters as "the m means, m variances, and (m choose 2) correlations" with classical
  simplifications; Yellott's theorem (Thurstone = Luce iff Gumbel, m >= 3). Reference, not a
  new identification result for Sigma.
- **Henery (1981)** "Permutation probabilities as models for horse races," *JRSS B* 43,
  86-88; and **Henery (1983)** gamma version: normal order-statistics fitted to race finish
  orders — the racing ancestor of all this, but with independent equal-variance utilities
  (no Sigma). DISTINCT (IID normal; the menu varies race to race via mean abilities only).
- **Bunch (1991)** "Estimability in the multinomial probit model," *Transp. Res. B* 25, 1-12
  (PDF in scratchpad): estimability of MNP covariance from FIRST-choice data with covariates;
  Jacobian rank calculations; credits Dansie (1985). Useful foil: the first-choice side of
  the counting argument. DISTINCT (no rank data).
- **Keane (1992)** "A note on identification in the multinomial probit model," *JBES* 10,
  193-200: fragility of MNP covariance identification from choices without exclusion
  restrictions. The standard citation for "first choices barely identify Sigma." DISTINCT
  but the natural pairing for the claim that rank data relieves this.

## 6. What is NOT in the literature (the gap)

1. **Aggregate top-k inclusion shares as the sole data.** Every estimator above consumes
   rank *patterns* (full or partial orderings per observation) or at least the bivariate
   margins of the induced paired comparisons. No paper found identifies or estimates Sigma
   from marginal win + place + show inclusion frequencies alone (the parimutuel-pool data
   shape), and none discusses when those marginal functionals suffice.
2. **Large menus with menu variation.** Psychometric applications are n = 4-7 (occasionally
   ~10 via MCMC); the menu is identical across observations. Racing-style data — many
   distinct menus, each seen once, sharing a factor structure — is untouched on this front
   (that variation lives in the BLP/panel literature the parent program already knows).
3. **Econometric single-market covariance identification from ranks.** The ROP econometrics
   (Hajivassiliou-Ruud, Layton-Levine, Nair-Bhat) always leans on covariates across
   individuals; the covariate-free "distribution of rankings of one menu identifies
   A Sigma A' up to scale" statement exists only in the psychometric strand
   (Dansie 1986; Maydeu-Olivares 1999; Tsai 2000), stated for small n and never connected
   to market shares or betting markets.

## Verdict table

| Candidate | Verdict | Distinguishing dimension |
|---|---|---|
| Hausman & Ruud 1987 | DISTINCT | logit; rank-depth scale not covariance; individual survey w/ covariates |
| Beggs-Cardell-Hausman 1981 | DISTINCT | ROL origin, IID |
| Hajivassiliou & Ruud 1994 | DISTINCT/CLOSE | introduces ROP; computation, covariates |
| Layton & Levine 2003 | CLOSE | Bayesian Gaussian rank model; individual covariates, random coefficients |
| Nair-Bhat et al. 2019 | CLOSE | full differenced Sigma in ROP via MACML; micro data + covariates, exclusion restrictions |
| Maydeu-Olivares & Bockenholt 2005 | DIRECT HIT (model class) | unrestricted + factor Sigma from rankings of one menu; small n, pattern data, individual respondents |
| Maydeu-Olivares 1999 / 2001 | DIRECT HIT (estimation theory) | redundancy count; bivariate margins identify any Thurstonian model |
| Tsai 2000 (+ Tsai-Bockenholt 2002) | DIRECT HIT (identification) | equivalence classes = exactly the gauge freedom of Sigma under ranking data |
| Dansie 1985/1986 | CLOSE | earliest gauge/estimability counting for permutation probit |
| Bockenholt 1992 | DIRECT HIT (top-k dimension) | partial rankings incl. first choices, constrained Sigma; individual patterns not shares |
| Yao & Bockenholt 1999; Yu 2000; Johnson & Kuhn 2013 | CLOSE | Bayesian computation for Sigma from single-menu rankings; small n |
| Fok-Paap-van Dijk 2012; Chapman-Staelin 1982; Koop-Poirier 1994; Bockenholt 2001 | DISTINCT | logit family, reliability/heterogeneity not covariance |
| Henery 1981/1983 | DISTINCT | racing order statistics, IID utilities |
| Bunch 1991; Keane 1992 | DISTINCT | first-choice MNP estimability (the foil) |
