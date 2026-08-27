# Applications scouting: e-commerce, retail, recommenders, ads

Front: e-commerce / retail / recommenders / ads. Scouted 2026-08-25.

Capability being placed: exact win probabilities for a multinomial probit race with
low-rank-plus-diagonal covariance (U_i = mu_i + v_i'f + sqrt(D_i) eps_i), all N
probabilities in O(QNL) (N=1000 in ~3s, N=10^4 under a minute), share-to-utility
inversion, exact Jacobians, removal counterfactuals from one shared pass. Extensions
in progress: top-k inclusion probabilities, ordered pairs (second-choice matrix),
covariance identification from bivariate rank data.

Ranking criterion: (data availability) x (uniqueness of our advantage).

## Ranked summary

| # | Candidate | Data | Uniqueness | Notes |
|---|-----------|------|------------|-------|
| 1 | Product-display / assortment choice with correlated substitutes (search & booking pages) | High (Expedia ICDM 2013, JD.com MSOM, Amazon ESCI) | High | Incumbent is MNL (IIA) or EP-approximated probit capped near ~100 alternatives; we do exact probit at 10^4 |
| 2 | Stockout & delisting substitution — removal counterfactuals | Medium-high (JD.com inventory, dunnhumby, delisting event studies) | Very high | The removal counterfactual is literally our one-pass object; incumbents use EM/MCMC around MNL |
| 3 | Second-choice / diversion-matrix completion for retail delistings and platform "what-if"s | Medium (surveys, switching data; mostly proprietary) | Very high | Conlon–Mortimer–Sarkis already argue diversion is low-rank; we supply the exact engine + ordered-pairs extension |
| 4 | De-duplication and slate diversity in recommenders (red-bus/blue-bus in rankings) | High (Expedia random-sort impressions, ESCI substitute labels, embeddings) | Medium-high (top-k extension still in progress) | Incumbent DPP re-rankers are self-described heuristics, not choice models — no shares, no counterfactuals |
| 5 | Sponsored-search / retail-media cannibalization & incrementality | Low (auction logs proprietary; Criteo data lacks slate structure) | Medium | Strong story (ad vs. organic self-competition), weak public data |

---

## 1. Product-display / assortment choice with correlated substitutes

### (i) Race mapping
One search impression or category page visit = one race. Alternatives = the N
displayed (or displayable) products plus a no-purchase outside option. The event is a
genuine mutually exclusive argmax: the customer books/buys at most one item from the
slate. In the Expedia ICDM 2013 data "users either booked a hotel from the displayed
assortment or left without making a booking"; ~69% of retained impressions end in a
booking. At catalog scale (which products to surface for a query), N runs to 10^3–10^4
— exactly our regime.

### (ii) Data
- **Expedia "Personalize Expedia Hotel Searches – ICDM 2013"** (Kaggle,
  `expedia-personalized-sort`): 399,344 training search lists, ~10M rows;
  click/book labels; price, star rating, location score per hotel; crucially a
  **randomly-sorted subset** of impressions that kills position bias. This is the
  standard testbed for academic choice models (used by the neural RUM papers
  arXiv:2207.12877, arXiv:2308.05617, and by the large-probit EP paper
  arXiv:2407.09371).
- **JD.com MSOM 2020 data challenge**: transaction-level clicks + orders for 2.5M
  customers, 31,868 SKUs, one month, **with inventory/fulfillment data** (mirrored on
  Hugging Face: `a6687543/MSOM_Data_Driven_Challenge_2020`).
- **Amazon ESCI Shopping Queries** (github.com/amazon-science/esci-data): 130k
  queries, 2.6M labeled query-product pairs with an explicit **Substitute** label —
  free supervision for the low-rank factor loadings v_i.
- dunnhumby "The Complete Journey" (2,500 households, 2 years, 92k products) and
  "Breakfast at the Frat" (prices/promos, 156 weeks) for grocery shares.

### (iii) Incumbent and its documented limitation
Incumbent #1 is MNL. From the assortment-optimization literature (Cao/Kallus et al.,
arXiv:1707.02572, standard framing traceable to the Kok–Fisher–Vaidyanathan review):

> "Under the MNL model, the assortment problem admits a polynomial-time algorithm
> (Talluri and Van Ryzin, 2004). However, the model suffers from the independence of
> irrelevant alternatives (IIA) property ... In practice, however, the IIA property is
> often violated."

The proposed fixes (nested logit, latent-class MNL, Markov chain choice, exponomial)
each trade tractability or need hand-crafted structure (nests, N^2 transition
parameters).

Incumbent #2, for probit specifically, is Expectation-Propagation approximation. The
EM+EP paper (arXiv:2407.09371, applied to the same Expedia data):

> "the Gaussian latent variable feature of probit models coupled with identification
> constraints pose significant computational challenges for its estimation and
> inference, especially when the dimension of the discrete response variable is large"

and its headline claim is that EP "enables the analysis of correlated choice data in
the presence of **more than 100 alternatives**." That is the current frontier we leapfrog
by two orders of magnitude, exactly, not approximately.

Demand evidence that structured choice models beat black-box ML **in production**:
Feldman, Zhang, Liu, Zhang, "Customer Choice Models vs. Machine Learning: Finding
Optimal Product Displays on Alibaba" (Operations Research 2022). A field experiment on
10.4M customer visits: the featurized-MNL display policy earned 5.17 RMB/visit vs 4.04
for the incumbent ML ranker (+28%); Alibaba deployed it, with an estimated +87.26M
RMB/year. If plain MNL (with IIA baked in) is worth 28%, a tractable correlation-aware
model is the obvious next increment — and computation is the acknowledged reason nobody
runs probit there.

### (iv) What we uniquely enable
- Exact choice probabilities and exact Jacobians for N = full candidate set, so
  assortment/display optimization can use true gradients instead of simulation noise
  or IIA algebra.
- Share inversion: calibrate mu_i to observed click/booking shares given a covariance
  built from product embeddings (ESCI substitute labels, or off-the-shelf embeddings)
  — no BLP-style stochastic contraction.
- Removal counterfactuals from one shared pass: score "what if we drop/swap item j"
  for every j in the catalog in one sweep — the core assortment-optimization inner loop.
- The low-rank factor structure is the natural bridge to ML pipelines: v_i can be an
  embedding head, and the model remains exactly computable.

### (v) Minimal demo
On Expedia random-sort impressions: fit MNL, mixed logit (simulated), and our low-rank
probit (v_i from hotel features or learned); compare held-out log-likelihood and — the
distinctive test — **counterfactual share prediction**: for hotels that appear in some
impressions and not others, predict how their absence re-allocates bookings, and
compare against actually observed choice-set variation. Direct head-to-head with
arXiv:2407.09371 (same data, they cap near 100 alternatives; we run the full sets
exactly and faster).

### (vi) Hardness/demand quotes
See (iii). Also Mortimer (FTC keynote, 2022): "Product characteristics do not
accurately capture substitution across products" — the case for identifying covariance
from data (our bivariate-rank identification) rather than assuming it from features.

---

## 2. Stockout and delisting substitution (removal counterfactuals)

### (i) Race mapping
A store visit / session in a category = a race among available SKUs + outside option.
A stockout or delisting removes an alternative; realized sales shifts are draws from
the removal counterfactual. This is the cleanest possible fit: the question retailers
ask ("where does demand go if item j disappears?") is *exactly* the quantity our
shared-pass counterfactual computes, and the second-choice matrix extension is its
full generalization.

### (ii) Data
- **JD.com MSOM 2020**: orders + SKU-level inventory/stockout information in one
  public dataset — rare and directly usable.
- dunnhumby Complete Journey: long panel, assortment churn observable (items entering/
  leaving the assortment over 2 years) — natural delisting quasi-experiments.
- Academic precedents with (proprietary) data: Conlon–Mortimer's MarkVend vending
  data (field experiments removing top products); Musalem et al. store-level data with
  periodic inventory review; fashion-retail spillover studies. Delisting event studies
  (e.g., conflict delistings, Li et al., POM 2023) document large measurable
  substitution to remaining products.

### (iii) Incumbent and its documented limitation
Incumbents: Anupindi et al. (1998) EM on Poisson-MNL; Musalem et al. (2010) MCMC/
Bayesian data augmentation; Conlon & Mortimer (2013 AEJ:Micro) EM around random-
coefficient MNL. All are simulation/EM machinery wrapped around logit-family cores,
and all fight the same two fires. From Shao & Kleywegt, arXiv:2003.02313:

> "when demand exceeds the available inventory, products go out-of stock, and the
> assortment becomes smaller. If we ignore the stock-out events, then the excess
> demand will not be captured, and the estimation can thus be serious[ly] biased."

> "since only sales data is recorded, different customers may have different
> unobserved choice sets. ... we do not ... know the order that products become out of
> stock, and thus do not know what choice sets are encountered."

And Conlon's research statement on the direction of the bias: the naive approach
"understates demand for popular products that stock out more often, and overstates
demand for their closest substitutes."

### (iv) What we uniquely enable
- The inner loop of every one of these estimators is "recompute all choice
  probabilities under a modified availability set" — our one-pass removal
  counterfactual makes that loop exact and cheap at N in the thousands, where the
  MCMC/EM incumbents operate on N of tens.
- Share inversion turns observed pre-stockout shares directly into utilities under a
  correlated covariance, so substitution to *close* items (same brand, same flavor) is
  captured without hand-built nests.
- Exact Jacobians give delisting/derange sensitivities for assortment rationalization
  ("SKU rationalization" is a standing retail exercise) at category or store level.

### (v) Minimal demo
JD.com MSOM: pick high-velocity categories; identify stockout windows from the
inventory table; fit the low-rank probit on fully-stocked periods; predict the sales
re-allocation during stockout windows out-of-sample; compare against MNL (which, by
IIA, spreads demand proportionally to share — rank-1 diversion) and against the
observed re-allocation. Repeat with dunnhumby assortment-churn events.

### (vi) Quotes
See (iii). The direction-of-bias quote is the commercial pitch: mis-estimated
substitution directly misprices inventory and delisting decisions.

---

## 3. Second-choice / diversion matrix completion

### (i) Race mapping
Ordered pair (first choice, second choice) = our exacta/ordered-pairs object. Diversion
ratio D_{j->k} = P(k second | j first), the object of merger review, retailer
delisting negotiations, and platform "what-if" tooling. Mutually exclusive argmax holds
by construction.

### (ii) Data
Mortimer's FTC keynote (Conlon–Mortimer–Sarkis project) lists where second-choice data
actually arises: "Rank-ordered lists (market design, school choice); Customer Surveys
(If you didn't buy a Camry what would you buy?); Conjoint analyses in Marketing; A/B
tests showing different search results to different customers." Plus "survey data on
'If this Tesco were to close where would you shop' (as UK CMA asks)" and number-porting
/ win-loss data. Public sources are thinner: conjoint datasets, the MaritzCX new-car
survey (53,328 purchases, used in their paper, proprietary), school-choice rank lists.
E-commerce angle: A/B tests and Instacart-style replacement acceptances generate
second-choice data at scale but are proprietary.

### (iii) Incumbent and its documented limitation
Verbatim from the keynote (Mortimer, FTC 2022, "Diversion and the Use of Second-Choice
Data"):

> "Multi-product demand with unrestricted matrices of (J+1)^2 cross-elasticities (such
> as AIDS) is often hopeless with large J. Unrestricted diversion likely equally
> hopeless."

> "Logit restricts D to be of rank one. Nested logit of rank <= G (the number of
> non-singleton nests). Mixed logit to rank(D) <= I (but bound is likely
> uninformative)."

Their own proposal is "a low-rank approximation to D" fit by semiparametric logit —
i.e., the leading academics in this area have already concluded the right object is a
low-rank diversion matrix, and are building it with logit mixtures estimated by
simulation.

### (iv) What we uniquely enable
- Our low-rank probit *implies* a structured second-choice matrix computed exactly
  (ordered-pairs extension), with rank controlled by L — the same modeling bet as
  Conlon–Mortimer–Sarkis, but with exact evaluation, exact Jacobians for inference,
  and the ability to complete D from shares + a few observed entries by direct
  optimization rather than mixture simulation.
- Covariance identification from bivariate rank data is precisely "learn Sigma from
  observed pieces of D" — the missing inverse problem in their program.
- Same machinery prices retailer delisting threats (candidate #2) and merger diversion
  with one model.

### (v) Minimal demo
Simulate their setting: take our fitted Expedia or JD.com model as ground truth,
reveal shares + a random subset of D entries, and show recovery of the held-out
entries beats plain logit (rank-1) and nested logit fills. Then a real-data version on
any public conjoint / rank-ordered dataset (e.g., school choice lists or conjoint
studies with full rankings).

### (vi) Quotes
As in (iii); also "Product characteristics do not accurately capture substitution
across products" (same deck) — the argument for identifying the covariance from rank
data rather than from features.

---

## 4. De-duplication and slate diversity in recommenders (red-bus/blue-bus in rankings)

### (i) Race mapping
A feed/slate impression = a race among candidate items + "no engagement." Click/watch
= argmax (approximately; multi-click sessions weaken strict exclusivity — use first
engagement, or the top-k inclusion extension for multi-slot exposure). The
red-bus/blue-bus failure is documented in ML language: "similar items sometimes
cannibalize each other" (Tomlinson, cs.cornell.edu/~kt/post/discrete-choice-models/),
and softmax/Plackett–Luce rankers inherit IIA exactly (PL is "a softmax over item
utilities").

### (ii) Data
Expedia random-sort impressions (clean of position bias); Amazon ESCI substitute
labels (defines "near-duplicate" pairs objectively); MIND (Microsoft news) and
Taobao/Alibaba user-behavior logs for slates with clicks; embeddings freely available
to build v_i.

### (iii) Incumbent and its documented limitation
Production incumbent for handling similar-item cannibalization is DPP re-ranking.
Verbatim, Wilhelm et al., CIKM 2018 (YouTube):

> "Many recommendation systems produce result sets with large numbers of highly
> similar items. Diversifying these results is often accomplished with heuristics,
> which are impoverished models of users' desire for diversity."

Their DPP fix "yields substantial short- and long-term increases in user engagement"
on live YouTube traffic — proof the money is real. But a DPP is a set-scoring
heuristic, not a choice model: it produces no shares, no utilities, no counterfactuals,
and must be re-tuned per surface. Meanwhile ranking losses remain softmax/PL (IIA):
adding a near-duplicate of item j steals share from *everything* proportionally under
the model, but from j overwhelmingly in reality.

### (iv) What we uniquely enable
- A ranker whose predicted click shares respect substitution: duplicate items split
  their own share (correlated factors) instead of inflating category share. This is
  the red-bus/blue-bus correction *inside* the model rather than as a re-rank patch.
- Top-k inclusion probabilities (extension in progress) = principled slate
  construction: choose the k-slate maximizing P(engagement) under correlated
  utilities, with exact gradients.
- Calibration: invert observed click shares to utilities *after* removing the
  duplication distortion — better item quality scores for downstream systems.

### (v) Minimal demo
ESCI + any public click log: take queries with labeled Substitute pairs; show that a
PL/softmax model over-predicts total clicks when both substitutes are shown, while the
low-rank probit (v_i from embeddings) predicts the observed sub-additivity. Purely
offline, no company partner needed.

### (vi) Quotes
As above (Wilhelm et al.; Tomlinson). Also the general ranking literature concedes PL's
limits: "a single Plackett–Luce component cannot express interaction effects"
(arXiv:2603.21373).

---

## 5. Sponsored search / retail media: cannibalization and incrementality

### (i) Race mapping
A results page = race among {sponsored slots, organic results, leave}. The advertiser
question — "did the ad win a click that organic would have won anyway?" — is a removal
counterfactual: P(convert | ad shown) vs P(convert | ad removed, same page). Caveat:
strict single-argmax is an approximation (users can click several results), and
position/cascade effects must be absorbed into mu_i.

### (ii) Data
Weakest of the five. Auction and slate logs are proprietary (Amazon AMC gives
advertisers aggregates only). Criteo datasets lack full-slate structure. Industry
demand is loudly documented, though: practitioner literature reports "brand-keyword
campaigns can cannibalize up to 99% of organic traffic" and Amazon-ads agencies sell
incrementality testing by bid-suppression experiments (Perpetua, m19, Adverio blogs).

### (iii) Incumbent and its documented limitation
Academic treatment is the externalities literature: "the click probability, and
therefore an advertiser's profit, depends on which other ads are shown in conjunction"
(cascade-model literature, Kempe–Mahdian etc.); the cascade model handles position but
imposes a rigid sequential-scan structure with no similarity-based substitution.
Practitioner incumbent is A/B bid suppression — weeks of experiment per keyword.

### (iv) What we uniquely enable
Model-based incrementality: one fitted race per query segment yields removal
counterfactuals for any ad/organic result instantly, replacing per-keyword experiments;
correlated factors capture ad-vs-own-organic-listing substitution (the actual
cannibalization channel — same product in both slots).

### (v) Minimal demo
Only semi-synthetic: build slates from Expedia/ESCI, designate some listings
"sponsored duplicates" of organics, show the probit recovers near-total
cannibalization where MNL predicts incremental share. A real demo needs a retail-media
partner.

### (vi) Quotes
As in (ii)-(iii). Keep this candidate as a partner-dependent follow-on, not a first
target.

---

## Cross-cutting notes

- **The academic wind is at our back on structure**: Conlon–Mortimer–Sarkis argue for
  low-rank diversion; McFadden–Train says mixed logit is flexible *only* given the
  right characteristics, and Mortimer flags that "product characteristics do not
  accurately capture substitution" — our rank-data covariance identification answers
  exactly that objection.
- **The computational frontier for probit is publicly on record at ~100 alternatives**
  (arXiv:2407.09371, EP approximation, Expedia application). Our exact O(QNL) at
  N=10^4 is the headline claim to lead with in this domain, benchmarked on the same
  Expedia data.
- **Sequencing suggestion**: Demo 1 (Expedia, vs the EP-probit paper) establishes the
  computational claim on a literature-standard dataset; Demo 2 (JD.com stockouts)
  establishes the counterfactual claim; Demo 4 (ESCI substitutes) is the recommender-
  facing story with the broadest industry audience.

## Sources

- Mortimer, "Diversion and the Use of Second-Choice Data," FTC keynote 2022:
  https://www.ftc.gov/system/files/ftc_gov/pdf/mortimer_ftc_keynote_2022.pdf
- Conlon research statement (MarkVend, stockouts, diversion):
  https://chrisconlon.github.io/site/research_statement.pdf
- Conlon & Mortimer, "Demand Estimation Under Incomplete Product Availability," AEJ:Micro 2013.
- Musalem et al., "Structural Estimation of Retail Demand Under Partially-Observed
  Out-of-Stocks" (Management Science 2010).
- Shao & Kleywegt, "Joint Estimation of Discrete Choice Model and Arrival Rate with
  Unobserved Stock-out Events": https://arxiv.org/abs/2003.02313
- Kok, Fisher, Vaidyanathan, "Assortment Planning: Review of Literature and Industry
  Practice" (2008).
- Sequential MNL / IIA framing: https://arxiv.org/abs/1707.02572
- Feldman, Zhang, Liu, Zhang, "Customer Choice Models vs. Machine Learning: Finding
  Optimal Product Displays on Alibaba," Operations Research 2022:
  https://pubsonline.informs.org/doi/abs/10.1287/opre.2021.2158
- Large probit via EM+EP (Expedia application, ~100-alternative frontier):
  https://arxiv.org/abs/2407.09371
- Wilhelm et al., "Practical Diversified Recommendations on YouTube with Determinantal
  Point Processes," CIKM 2018: https://dl.acm.org/doi/10.1145/3269206.3272018
- Tomlinson, "Discrete Choice Models" (IIA, cannibalization):
  https://www.cs.cornell.edu/~kt/post/discrete-choice-models/
- Expedia ICDM 2013 dataset: https://www.kaggle.com/c/expedia-personalized-sort
- JD.com MSOM 2020 data: https://pubsonline.informs.org/doi/10.1287/msom.2020.0900 ;
  mirror https://huggingface.co/datasets/a6687543/MSOM_Data_Driven_Challenge_2020
- Amazon ESCI Shopping Queries: https://github.com/amazon-science/esci-data
- dunnhumby source files (Complete Journey, Breakfast at the Frat):
  https://www.dunnhumby.com/source-files/
- Delisting studies: Sloot & Verhoef, "The Impact of Brand Delisting on Store
  Switching and Brand Switching Intentions," J. Retailing 2008; Li et al., "Impact of
  conflict delisting and relisting on remaining products," POM 2023.
- Ads cannibalization practitioner evidence: https://perpetua.io/blog-cannibalising-organic-sales-what-is-it-and-what-can-i-do-to-minimise-it/ ;
  https://www.m19.com/blog/does-amazon-ppc-sales-really-cannibalize-your-organic-sales
- Externalities in sponsored search (cascade model): Kempe & Mahdian 2008,
  https://link.springer.com/chapter/10.1007/978-3-540-92185-1_65
