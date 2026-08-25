# Prior art: free covariance / factor structure from rank-level AGGREGATE data

Question: has anyone identified or estimated a FREE (not characteristics-tied) covariance
or factor structure of a random-utility model from rank-level aggregate data — aggregate
first- AND second-choice shares, top-k conversion tiers, funnels, or co-ranking data —
in a single market snapshot?

Search date: 2026-08-25. Source texts saved under the session scratchpad
(`.../scratchpad/mkthunt/`): BLP NBER WP 6481 (OCR'd scan), Conlon-Mortimer RAND 2021,
Conlon-Gortmaker micro-PyBLP, Conlon-Mortimer-Sarkis 2026 draft, Elrod-Keane 1995 (OCR'd
MPRA scan), Colombo-Morrison note, Seshadri et al. 2019 CDM, Zhao-Xia 2019.

---

## 1. Berry, Levinsohn, Pakes (2004) "microBLP" — VERDICT: DISTINCT (characteristics-tied Sigma; micro survey data)

Berry, S., J. Levinsohn, and A. Pakes (2004), "Differentiated Products Demand Systems from
a Combination of Micro and Macro Data: The New Car Market," *Journal of Political Economy*
112(1), 68-105. (Quotes below from the NBER WP 6481 version, OCR of the scanned PDF;
wording of the JPE version may differ slightly.)

The data are individual-level (CAMIP survey of new-car buyers):

> "Even more unusually, the CAMIP data include a second choice question: 'if you did not
> purchase this vehicle, what vehicle would you purchase?' The answers to this second choice
> question provide direct evidence on substitution patterns."

The moments they match include an aggregate-looking object — the covariance of first- and
second-choice characteristics:

> "(iii). The covariances between the first choice product characteristics and the second
> choice product characteristics (for example, the covariance of the size of the first choice
> vehicle with the size of the second choice vehicle.)"

But what these moments identify is the loading of *unobserved consumer attributes on
observed product characteristics* (their theta^u), i.e. a random-coefficients covariance
tied to the K-dimensional characteristic space, not a free J x J covariance:

> "The combination of first and second choice data should allow us to also get precise
> estimates of the theta^u parameters. To see this, note that we could predict the correlation
> in the characteristics of the first and second choice vehicles using only the observed
> attribute data. On the other hand we have the actual correlations. The importance of the
> unobserved attributes is extracted from the difference between the data and the predictions
> based on the observable attributes."

Conlon and Gortmaker's micro-moments paper ("Incorporating micro data into differentiated
products demand estimation with PyBLP," https://chrisconlon.github.io/site/micro_pyblp.pdf)
states the identification mapping explicitly:

> "In papers such as Berry, Levinsohn, and Pakes (2004) that use second choice data, a
> popular statistic is the covariance 'C(x_cjt, x_ek(-j)t | j, k != 0)' between first and second
> choice characteristics ... Intuitively, this should contain information about a parameter
> in Sigma that measures the variance of unobserved preference heterogeneity nu_cit for x_cjt
> if e = c, or the covariance between unobserved preferences nu_cit and nu_eit for x_cjt and
> x_eijt if e != c. Holding mean preferences delta_jt equal, if when j is eliminated from the
> choice set consumers tend to select a second choice k that has a very similar characteristic
> x_ckt ~ x_cjt, it must be that nu_cit has a high variance. Otherwise, we would expect to see
> proportionate substitution to all remaining alternatives."

**Distinguishing dimension:** the covariance identified is Sigma over random coefficients on
OBSERVED characteristics — substitution is constrained to characteristic space. Nothing free,
and the estimation uses matched individual micro data (households x choices), not one
aggregate rank snapshot.

---

## 2. Conlon & Mortimer, "Empirical Properties of Diversion Ratios" (RAND 2021) — VERDICT: DISTINCT (interprets diversion; never inverts it)

Conlon, C. and J.H. Mortimer (2021), "Empirical properties of diversion ratios," *RAND
Journal of Economics* 52(4), 693-726. https://chrisconlon.github.io/site/diversion.pdf

The paper gives second-choice/diversion data a treatment-effects (LATE/ATUT)
interpretation; it does not invert diversion into any covariance or factor structure:

> "We establish a LATE interpretation of diversion ratios, and show how diversion ratios are
> obtained from different interventions (price, quality, or assortment changes) and how those
> measures relate to one another and to underlying properties of demand." (abstract)

> "An important special case is that of second-choice data ... In this case, Proposition 1
> implies that: Wald(p_j, pbar_j, x) = E[D_jk,i(x) | d_ij(p_j,x) = 1] = ATUT. ... The
> instrument we use to obtain second-choice data is irrelevant as all types of second-choice
> data identify the ATUT."

On microBLP they explain *why* second choices help, again in characteristic space:

> "In addition, our framework sheds light on the value of second-choice data in parametric
> models of demand such as Berry et al. (2004) (microBLP). In microBLP, the authors have
> access to survey data on consumers' second choices. They report finding extra moments from
> this second-choice data useful in estimating nonlinear (substitution) parameters."

**Distinguishing dimension:** diversion is the OUTPUT/diagnostic (a treatment effect), not
inverted to a latent covariance. The inversion step appears only in their 2026 paper (next).

---

## 2b. Conlon, Mortimer & Sarkis (2026 draft), "Estimating Preferences and Substitution Patterns from Second-Choice Data Alone" — VERDICT: DIRECT HIT (with a gap)

Conlon, C., J.H. Mortimer, and P. Sarkis, working paper dated May 11, 2026.
https://chrisconlon.github.io/site/semiparametric_paper.pdf (slides: semiparametric_slides.pdf)

This is exactly the target setting: aggregate first-choice shares plus an aggregate
conditional second-choice (diversion) matrix, one market, free product-space structure
controlled by a rank restriction rather than characteristics:

> "We propose an estimator for a semi-parametric model of consumer choice where the main
> source of variation is in the set of products made available to consumers. Instead of
> relying on variation in the choice environment (prices, product characteristics) we utilize
> first-choice probabilities and a subset of (conditional) second-choice probabilities. Our
> estimator is consistent with mixed logit models of demand but is defined in 'product space'
> and does not require that product characteristics explain substitution patterns. Instead,
> we control the number of parameters by restricting the rank of the second-choice probability
> matrix directly." (abstract)

> "Our estimator assumes that we observe data from a single choice environment ('market')
> where we have aggregate data on first-choice probabilities S_j ('market shares') and
> aggregate data on conditional second-choice probabilities ('diversion ratios') D_{j->k}
> for a subset of (j,k)."

> "we build on a result from Conlon and Mortimer (2021a) and write second-choice
> probabilities in terms of the first-choice probabilities for a finite number of consumer
> types I. This reduces the number of parameters to (J+1) x I, and restricts the rank of the
> matrix of substitutes to be no more than I."

They handle partially observed D as matrix completion, benchmark against unconstrained
alternating-least-squares factorization, and fit single-market automobile data (Grieco,
Murry, Pinkse, Sagl 2024 / MaritzCX survey) better than characteristics-based mixed logit:

> "at lower ranks I << J ... Our estimator, by contrast is not restricted by product
> characteristics: at rank I in {3, 4} [it fits better than parametric baselines]"

> "In short, we estimate both (pi, S) in product space using second-choice data from a single
> market, with the goal of explaining substitution with the smallest number of types possible."

They also connect to co-ranking-style data:

> "We also illustrate how to repurpose the customer overlap measure proposed by Einav et al.
> (2026) that can be applied to credit-card spending or foot-traffic (cell phone tracking)
> data to construct an estimate of second-choice diversion ratios."

Rank counting they establish (their Table 1 and Appendix A.2): mixed logit with I types
gives rank(D) <= I; plain logit rank(D) <= 1; nested logit with G groups rank(D) <= G.

**Gap relative to a Thurstone/probit-covariance program:** the free object is a nonnegative
rank-I finite mixture (an NMF of the diversion matrix into type-level choice-probability
vectors), NOT a Gaussian utility covariance; nothing is said about recovering a
positive-semidefinite covariance/factor loading matrix, and the second-choice matrix must
itself be (partially) OBSERVED — they do not identify structure from coarser tiers such as
top-k conversion funnels (impressions -> clicks -> purchases), where the j->k pairing is
never observed.

---

## 3. Marketing: Choice Map, factor-analytic probit, brand-switching matrices

### Elrod (1988) Choice Map — VERDICT: DISTINCT (individual panel data)
Elrod, T. (1988), "Choice Map: Inferring a Product-Market Map from Panel Data," *Marketing
Science* 7(1), 21-40. Latent-attribute (factor) logit inferred from households' repeated
purchase histories — disaggregate panel, not aggregate rank shares. Elrod & Keane's Table 1
classifies Choice Map's required data as "Disaggregate panel."

### Elrod & Keane (1995) factor-analytic probit — VERDICT: DISTINCT (free factor structure, but needs panel)
Elrod, T. and M.P. Keane (1995), "A Factor-Analytic Probit Model for Representing the Market
Structure in Panel Data," *Journal of Marketing Research* 32(1), 1-16. (Quotes from OCR of
the MPRA copy, https://mpra.ub.uni-muenchen.de/52434/.)

This is the closest ancestor of a free/low-rank probit covariance — but estimated from
individual purchase histories, and they explicitly flag the aggregate-data limitation:

> "Internal market structure analysis infers both brand attributes and consumer preferences
> for those attributes from preference or choice data. The authors exploit a new method for
> estimating probit models from panel data to infer market structures that can be displayed
> in few dimensions, even though the model can represent every possible vector of purchase
> probabilities." (abstract)

> "However, the ability to infer consumer heterogeneity from aggregate data is limited."

### Brand-switching-matrix latent class: Grover & Srinivasan (1987); Jain, Bass & Chen (1990) — VERDICT: CLOSE (aggregate two-occasion matrix, free segment structure)
Grover, R. and V. Srinivasan (1987), "A Simultaneous Approach to Market Segmentation and
Market Structuring," *Journal of Marketing Research* 24(2), 139-153. Jain, D., F.M. Bass and
Y.-M. Chen (1990), "Estimation of Latent Class Models with Heterogeneous Choice
Probabilities: An Application to Market Structuring," *JMR* 27(1), 94-101.

These fit latent-class (finite mixture) models to an AGGREGATE brand-switching matrix —
an object formally analogous to a second-choice share matrix, though built from purchases
on two occasions rather than a counterfactual removal. Elrod & Keane (1995) describe them:

> "A second approach has been to analyze brand switching matrices using latent class models.
> Brand switching matrices are constructed by tallying a one in cell (i, j) whenever a
> household is observed to purchase brand i followed by brand j. ... Grover and Srinivasan
> (1987) and Jain, Bass, and Chen (1990) both fit switching matrices by postulating that the
> population of consumers is a mixture of segments."

And the limitations:

> "Although the model is consistent with heterogeneity within switching segments, the degree
> of heterogeneity cannot be estimated from switching data. ... First, these methods do not
> explain differences in market shares ... Second, although brand switching matrices preserve
> some of the information about consumer heterogeneity, they do so imperfectly and at a price.
> Because brand switches are aggregated over time, information about the effects of marketing
> mix variables on brand choice is lost."

**Distinguishing dimensions:** (a) switching over TIME confounds substitution with variety
seeking / mix effects, unlike a same-instant second choice; (b) the free structure is a
handful of latent classes, not a covariance/factor model of utilities; (c) no random-utility
covariance interpretation is extracted. Note the mixture-of-shares algebra is essentially
the same low-rank decomposition CMS 2026 rediscover with an economic (removal-based) D.

### Colombo & Morrison (1989) — VERDICT: DISTINCT
Colombo, R.A. and D.G. Morrison (1989), "A Brand Switching Model with Implications for
Marketing Strategies," *Marketing Science* 8(1), 89-99. Two-segment structure on the
aggregate switching matrix; switchers are IIA-like:

> "a) there are two groups of people - switchers and loyals; b) the loyals always buy the
> same brand; c) the switchers are zero-order. Then the conditional probabilities of buying
> brand j given previously owned brand i are p_ii = alpha_i + (1-alpha_i) pi_i ...
> p_ij = (1-alpha_i) pi_j" (Morrison & Colombo, JEGMS restatement)

Diagonal-plus-rank-one: it explains loyalty, not differentiated substitution structure —
their own residual analysis on the France 1989 car-switching matrix shows block-diagonal
(segment) structure the model misses.

### DeSarbo and asymmetric-MDS of switching data — VERDICT: DISTINCT
E.g., DeSarbo, W.S. and A.K. Manrai (1992), "A New Multidimensional Scaling Methodology for
the Analysis of Asymmetric Proximity Data in Marketing Research," *Marketing Science* 11(1),
1-20 (not quote-verified here). Spatial maps ARE fit directly to aggregate switching
matrices, but as descriptive scaling — no random-utility covariance, no share-consistent
choice model.

---

## 4. ML / ranking literature — VERDICT: CLOSE on structure, DISTINCT on data or object

### Mixtures of Plackett-Luce from top-k / choice data — identifiability limits
Zhao, Z. and L. Xia (2019), "Learning Mixtures of Plackett-Luce Models from Structured
Partial Orders," *NeurIPS 2019*. https://arxiv.org/abs/1910.11721

> "We prove that when the dataset consists of combinations of ranked top-l1 and l2-way (or
> choice data over up to l2 alternatives), mixture of k Plackett-Luce models is not
> identifiable when l1 + l2 <= 2k - 1 (l2 is set to 1 when there are no l2-way orders)."

So top-2 data (first + second choice) identifies at most a 1-component PL cleanly; mixtures
of two PL need e.g. "ranked top-3, ranked top-2 plus 2-way, and choice data over up to 4
alternatives." This is the sharpest available statement about how much latent-heterogeneity
structure rank-tier data can pin down in the Luce family. Related: Oh & Shah (2014),
"Learning Mixed Multinomial Logit Model from Ordinal Data," NeurIPS — tensor/rank-breaking
estimation of MNL mixtures from (many individuals') partial rankings. Neither works from
one aggregate share vector per tier, and the object is a mixture, not a covariance.

### Low-rank context-dependent model (CDM) — Seshadri, Peysakhovich & Ugander (ICML 2019)
"Discovering Context Effects from Raw Choice Data," *ICML 2019*,
http://proceedings.mlr.press/v97/seshadri19a/seshadri19a.pdf

> "We introduce an extension of the Multinomial Logit (MNL) model, called the context
> dependent random utility model (CDM), which allows for a particular class of choice set
> effects. We show that the CDM can be thought of as a second-order approximation to a
> general choice system ..."

The low-rank CDM learns a free low-rank item-item interaction (embedding) matrix — the
closest ML analogue of a free substitution structure — but requires many observed choices
from VARYING choice sets, not first/second-choice shares from one fixed market. Same family:
Ragain & Ugander (2016), "Pairwise Choice Markov Chains," NeurIPS (non-IIA from choice-set
data); Seshadri, Ragain & Ugander (2020), "Learning Rich Rankings," NeurIPS (contextual
repeated selection over ranking data).

**No paper found that estimates a Thurstone/probit COVARIANCE from partial or top-k
rankings** — searches for "low-rank Thurstone," "Thurstone covariance partial rankings,"
"correlated Plackett-Luce" return mixtures and embeddings, not covariance recovery.

### E-commerce funnels and co-ranking — nothing direct
No paper found that estimates substitution/covariance from top-k CONVERSION TIERS
(impressions -> clicks -> add-to-cart -> purchase) in one market; the funnel literature is
managerial. Nearest neighbors: (a) Gabel, S., D. Guhl and D. Klapper (2019), "P2V-MAP:
Mapping Market Structures for Large Retail Assortments," *JMR* 56(4), 557-580 — word2vec-style
embeddings from basket co-occurrence ("customers also bought" style data), exploratory,
confounds substitutes with complements, no RUM covariance; (b) Einav et al. (2026) customer
overlap from card-spend/foot-traffic, which CMS 2026 repurpose into diversion estimates;
(c) Vulcano, van Ryzin & Ratliff (2012), "Estimating Primary Demand for Substitutable
Products from Sales Transaction Data," *Operations Research* — EM over availability
variation, MNL (IIA) so no free structure.

---

## Verdict summary

| Candidate | Verdict | Distinguishing dimension |
|---|---|---|
| BLP 2004 microBLP | DISTINCT | Sigma tied to observed characteristics; individual micro survey |
| Conlon-Mortimer RAND 2021 | DISTINCT | diversion as treatment effect; no inversion to latent structure |
| Conlon-Mortimer-Sarkis 2026 | **DIRECT HIT** | free low-rank product-space mixture from aggregate S + D, one market; but NMF mixture, not utility covariance; needs observed D matrix |
| Elrod 1988 / Elrod-Keane 1995 | DISTINCT | free factor probit, but requires disaggregate panel |
| Grover-Srinivasan 1987 / Jain-Bass-Chen 1990 | CLOSE | latent class on aggregate switching matrix; temporal switching not second choice; classes not covariance |
| Colombo-Morrison 1989 | DISTINCT | loyal/switcher diagonal model; switchers IIA |
| Zhao-Xia 2019 / Oh-Shah 2014 | CLOSE | identifiability of PL mixtures from top-k; individual partial orders, mixture object |
| CDM (Seshadri et al. 2019) | CLOSE | free low-rank interaction matrix, but needs choice-set variation |
| P2V-MAP 2019 | DISTINCT | co-occurrence embeddings, exploratory, complements confounded |

**Open slot confirmed:** nobody recovers a Thurstone/Gaussian utility COVARIANCE (or PSD
factor loading matrix) from aggregate rank-tier shares — and nobody works from funnel/top-k
conversion tiers where the (first, second) pairing is unobserved.
