# Applications scout: economics and finance beyond demand estimation

Capability under evaluation: multinomial probit race with low-rank-plus-diagonal covariance,
N up to 10^4; exact win probabilities for all N contenders in one pass; inversion of observed
shares/frequencies to latent utilities given the covariance; exact Jacobians; removal (deletion)
counterfactuals priced directly. Companion work already covers BLP-style demand and portfolio
tilting. Horse racing excluded by instruction.

Date scouted: 2026-08-25. Quotes are verbatim from the cited sources; characterization of any
valuation or theory implication is left to the author.

---

## 1. School choice / matching / college admissions

**(i) Race mapping.** Clean. Each student's assignment/enrollment is an argmax over schools of
correlated latent utilities; the workhorse empirical model IS multinomial probit (Gaussian
utilities, correlated across schools). Observed admit/enroll shares are win frequencies. The
"remove a school" counterfactual (closure, entry, ban on ranking) is literally the deletion
ensemble. The field's standard estimator is Gibbs sampling on latent utilities, i.e., MCMC where
we have exact probabilities and Jacobians.

**(ii) Public data.** Chile centralized admissions (used by Agarwal-Somaini), NYC DOE HS
admissions (restricted but widely used), Boston (Pathak-Shi), Hungary and Norway centralized
admissions, Chilean SIES data is downloadable. Many replication packages exist (Econometrica,
JoE supplements).

**(iii) Incumbent and documented limitation.** Agarwal & Somaini, "Revealed Preference Analysis
of School Choice Models" (Annual Review of Economics, 2020; PSU working paper PDF):

> "This model can be estimated via Gibbs' sampling with appropriate conjugate prior
> distributions."

> "Yet, the Gibbs' sampling procedure can be computationally burdensome when applied to the
> strategic reports model if the number of possible reports is large."

> "The first is the well-known issue that simulated maximum likelihood is biased unless the
> number of simulations is much larger than the number of choices (Train, 2009, Chapter 10).
> This creates a computational burden in school choice settings with many possible reports."

> "This observation vastly simplifies computation as the number of alternatives that need to be
> considered are now on the order K × J ... A more general solution to this problem is unknown."

Also relevant: Pathak & Shi, "How Well Do Structural Demand Models Work? Counterfactual
Predictions in School Choice" (J. Econometrics 2021, NBER w24017) — the field explicitly tests
discrete-choice models on out-of-sample counterfactual menu changes (Boston 2013 reform changed
where applicants can apply), which is exactly a deletion/insertion ensemble exercise.

**(iv) Unique advantage.** Exact choice probabilities and Jacobians for the truthful-reports
probit replace data augmentation inside the Gibbs loop, or enable direct (simulated-)MLE and
share inversion at the district scale (NYC: ~700 programs — within N = 10^4). Menu-change
counterfactuals (closure, new school) come from the same shared-factor pass without
re-simulation. Note the honest caveat: the hard open problem quoted above is the *strategic
reports* combinatorics, which our capability does not solve; the win is in the utility-model
layer that every one of these papers carries.

**(v) Minimal demo.** Take a public centralized-admissions dataset (e.g., Chile replication
package), fit the same factor-probit covariance, reproduce enrollment shares by exact inversion,
then price the removal of one school and compare with the paper's simulated counterfactual and
with realized post-reform data à la Pathak-Shi.

**(vi) Venue.** Journal of Econometrics, Quantitative Economics, or an applied-methods note in
AEJ: Policy / Economics of Education Review; methods audience overlaps with the Annual Review
survey's readers.

Sources: https://econ.la.psu.edu/wp-content/uploads/sites/5/2023/05/agarwal-somaini-revealed-pref.pdf ;
https://www.annualreviews.org/content/journals/10.1146/annurev-economics-082019-112339 ;
https://www.nber.org/papers/w24017 ; https://economics.mit.edu/sites/default/files/publications/w24017.pdf

---

## 2. Elections and ranked-choice voting (RCV)

**(i) Race mapping.** Multi-candidate vote choice as correlated argmax is a long-standing
argument in political methodology (Alvarez & Nagler 1998, AJPS, "When Politics and Models
Collide"; Dow & Endersby 2004 compare MNP vs MNL for voting research). Candidates share
ideological/party factors — precisely low-rank covariance. An RCV ballot is a (partial) ranking:
first choice = win, the k-th preference = a sequential race on the deletion ensemble. So a
Thurstone/probit likelihood for cast-vote records requires exactly the win-probability +
deletion machinery we have.

**(ii) Public data — strong.** FairVote curates cast vote records on Harvard Dataverse
("Ranked Choice Voting Cast Vote Records", https://dataverse.harvard.edu/dataverse/rcv_cvrs),
parsed to a consistent format via RCV Cruncher; coverage of 398 single-seat and 30 multi-seat US
elections; NYC BOE releases full anonymized CVRs; DC, SF, Alaska, Maine similarly. PrefLib hosts
election ranking data. This is full-ranking data at scale, freely downloadable.

**(iii) Incumbent and documented limitation.** The ranking-model literature on this data fits
independent-utility models: Plackett-Luce and truncation extensions (e.g., "Statistical models
of ballot truncation in ranked choice elections," Comm. Stat. 2024, fit to Cambridge MA ballots;
the Colorado STV simulation paper, arXiv 2607.25105, fits Plackett-Luce to 2022 statewide RCV
records; the PlackettLuce R package paper, arXiv 1810.12068). No fitted-correlation Thurstone
treatment of CVRs surfaced in this scout. On the probit side the computational excuse is stated
plainly — Loaiza-Maya & Nibbering, "Scalable Bayesian estimation in the multinomial probit
model" (arXiv 2007.13247):

> "Because current model specifications employ a full covariance matrix of the latent utilities
> for the choice alternatives, they are not scalable to a large number of choice alternatives."

> "current specifications of the multinomial probit model are not scalable to discrete choice
> problems with a large number of choice alternatives, as the number of parameters in the
> covariance matrix of the latent utilities grows quadratically in the number of choice
> alternatives (Burgette and Reiter, 2013). This curse of dimensionality is exacerbated by the
> fact that, contrary to standard covariance matrix estimation settings, where multiple
> continuous variables are observed, all parameters in the covariance matrix have to be
> estimated from a single categorical variable."

> "Empirical applications of multinomial probit models have been limited to only a few choice
> alternatives. For instance, Imai and Van Dyk (2005a) consider six clothing detergent brands,
> McCulloch and Rossi (1994) and Burgette and Nordheim (2012) six margarine brands ..."

Note they propose the same low-rank (factor) covariance — but estimate by MCMC; exact
probabilities/Jacobians for the factor structure are the missing piece their approach dances
around.

**(iv) Unique advantage.** Fit a Thurstone model with candidate-level ideological factors to
actual ranked ballots by exact likelihood (each ballot's probability = product of win
probabilities down the deletion chain). Then answer the questions the RCV literature already
asks with simulation: candidate-exit counterfactuals ("who wins if X drops out" — deletion
ensemble, the central object in spoiler/IIA debates), ballot truncation effects, ballot-order
effects (arXiv 2207.07005 does causal inference on rankings). IIA violation is measurable, not
assumed away — Plackett-Luce imposes it by construction.

**(v) Minimal demo.** Download one NYC or Alaska CVR; fit 1-2 factor Thurstone vs Plackett-Luce
by log-likelihood on first choices and on full rankings; report the correlation structure
(do same-party candidates load on a shared factor?) and the exit counterfactual for the
documented spoiler cases (e.g., Alaska 2022 special election, Burlington 2009 — both are
standard RCV-pathology exhibits with public ballots).

**(vi) Venue.** Political Analysis (methods, loves this data), Electoral Studies, or
EC/computational social choice venues; VoteKit (JOSS) community for the software angle.

Sources: https://dataverse.harvard.edu/dataverse/rcv_cvrs ; https://arxiv.org/pdf/2007.13247 ;
https://arxiv.org/pdf/2607.25105 ; https://www.tandfonline.com/doi/full/10.1080/03610918.2024.2397032 ;
https://arxiv.org/pdf/1810.12068 ; https://arxiv.org/pdf/2207.07005 ;
https://blogs.ubc.ca/poli574/files/2011/05/Dow-Endersby-MNP-vs.-MNL-JELS-2004.pdf

---

## 3. Credit: kth-to-default baskets under a one-factor Gaussian copula

**(i) Race mapping.** Under the standard one-factor Gaussian copula, name i defaults before
horizon T iff X_i = a_i M + sqrt(1-a_i^2) Z_i falls below a barrier — a Gaussian race with
exactly a one-factor-plus-diagonal covariance. "Which name defaults first" is a win probability;
"name i is among the first k defaults" is a top-k probability; kth-to-default pricing needs the
order statistics of correlated Gaussian-driven times. This is structurally identical to the
exacta-board/top-k machinery (heterogeneous a_i = heterogeneous basket, the case the literature
flags as hard). Gathering citations only; no valuation claims here.

**(ii) Public data.** No free single-name CDS spreads at scale (Markit is paywalled), but:
sample baskets in textbooks/papers are standard, CDX/iTraxx index constituents and tranche
conventions are public, and several open replication codebases exist (e.g., GitHub kth-to-default
pricers under Gaussian/t copulas). A methods-comparison paper does not need proprietary data —
it needs the same synthetic heterogeneous baskets the incumbent papers use.

**(iii) Incumbent and documented limitation.** Incumbents: Monte Carlo; the
Andersen-Sidenius-Basu (2003) / Hull-White (2004, "Valuation of a CDO and an n-th to default CDS
without Monte Carlo simulation") recursion; Fourier/FFT methods (Gregory-Laurent); saddlepoint
approximations for loss distributions (Yang-Hurd-Zhang, "Saddlepoint approximation method for
pricing CDOs"). On relative costs, Ackerer & Vatter, "Dependent Defaults and Losses with Factor
Copula Models" (arXiv 1610.03050):

> "the distribution is computed without approximation by a recursive algorithm. However, as will
> be shown in Section 4.1, the computational cost of this recursion increases much faster with
> both the support size and the number of factors than that of our approach."

> "The DFT method is significantly faster than the recursive method in both cases: it takes
> roughly the same amount of time to retrieve a distribution with 1000 points with DFT and a 100
> points with recursion."

Note the precise gap for OUR capability: recursion/FFT/saddlepoint all compute the *portfolio
loss-count distribution* (how many default). The per-name attribution question — the probability
that a *specific* name is the first (or among the first k) to default, for all N names at once,
with sensitivities — is the win/top-k vector, which those methods do not emit directly; MC with
rare joint defaults is the fallback. Deletion counterfactual = removing a name from the basket,
a quoted product feature of bespoke baskets (substitution of reference entities).

**(iv) Unique advantage.** One shared-field pass gives all N first-to-default (and kth) identity
probabilities plus exact Jacobians w.r.t. barriers/loadings — i.e., analytic hedge ratios per
name — and re-prices the basket minus any name. Inversion direction: given observed FtD or
tranche-implied quantities and a covariance, back out barrier levels (latent "distance to
default" utilities).

**(v) Minimal demo.** 125-name heterogeneous basket (CDX-like), one-factor Gaussian copula:
compute all 125 "in the first k defaults" probabilities and their Jacobians; benchmark accuracy
and wall-clock against Monte Carlo and against the ASB/Hull-White recursion; show the deletion
re-price for each name. Purely computational comparison; no valuation claims.

**(vi) Venue.** Journal of Computational Finance, Quantitative Finance, or Risk (Cutting Edge)
— all have published the incumbent methods.

Sources: https://arxiv.org/pdf/1610.03050 ; https://ms.mcmaster.ca/tom/SaddlepointCDOfinal.pdf ;
https://arxiv.org/pdf/1204.4025 ; Hull & White (2004), JCF; Andersen, Sidenius & Basu (2003), Risk.

---

## 4. Auctions and procurement

**(i) Race mapping.** Lowest-cost-wins procurement: bidder i's latent bid strength with common
cost factors (fuel, materials, regional labor) plus idiosyncratic noise; observed win rates by
bidder are win frequencies of a correlated Gaussian race. Inversion given covariance = calibrate
bidder strengths from win records without bid-level data. Deletion = "what if bidder X exits /
is debarred," a live question in collusion and merger cases. Caveat: this maps *reduced-form win
propensity*, not the structural equilibrium bid function — strategic bid shading responds to the
deletion, so the ensemble counterfactual is a first-order, not equilibrium, answer. Should be
framed accordingly.

**(ii) Public data.** Excellent: federal/state procurement award records (USAspending, FPDS),
California DOT (Caltrans) highway lettings (used across the empirical auctions literature),
Texas DOT, Oklahoma DOT (Hickman et al.), EU TED database. Win/lose by bidder is the most widely
available field even when bids are missing.

**(iii) Incumbent and documented limitation.** Structural first-price estimation with asymmetry
computes equilibria numerically. Fibich & Gavish, "Numerical simulations of asymmetric
first-price auctions" (Games and Economic Behavior, 2011):

> the authors "show that the backward-shooting method is inherently unstable, and that this
> instability cannot be eliminated by changing the numerical methodology of the backward
> solver. Moreover, this instability becomes more severe as the number of players increases."

(Quote as rendered in the paper's abstract; verify page proof before citing in print.)
On correlation, equilibrium computation is provably hard: "Equilibrium Computation in
First-Price Auctions with Correlated Priors" (arXiv 2506.05322) shows "determining the existence
of a pure Bayes-Nash equilibrium is NP-hard" with correlated discrete priors; see also arXiv
2103.03238 (PPAD-hardness, subjective priors) and arXiv 2606.16389 (fictitious-play estimation
with correlated values). The empirical literature largely retreats to independent private values
or conditional independence given auction-level unobserved heterogeneity (Krasnokutskaya).

**(iv) Unique advantage.** A calibration layer that skips equilibrium computation: fit bidder
strengths + common-factor loadings to win records across thousands of lettings, get exact
win-probability Jacobians for covariates, and screen for exit/entry effects and collusion
(excess correlation among ring members' latent strengths). Positioning: descriptive/screening
tool, complement to structural work, not a replacement.

**(v) Minimal demo.** Caltrans letting records: firms as racers, region×month cost factors;
calibrate to observed win rates; flag the known collusion cases the auctions literature uses as
validation; price the exit of the largest bidder per district.

**(vi) Venue.** International Journal of Industrial Organization, RAND (higher bar), or a
screening-oriented outlet (Journal of Competition Law & Economics); also antitrust agency
workshops.

Sources: https://www.sciencedirect.com/science/article/abs/pii/S0899825611000509 ;
https://ngavish.net.technion.ac.il/files/2013/01/BV_Method_GEB112.pdf ;
https://arxiv.org/pdf/2506.05322 ; https://arxiv.org/pdf/2103.03238 ; https://arxiv.org/pdf/2606.16389 ;
https://link.springer.com/article/10.1007/s10614-012-9333-z

---

## 5. Sports analytics (excluding horse racing): golf, F1, athletics

**(i) Race mapping.** Direct: finishing order = ranking of correlated latent performances.
Correlation sources are physically real and documented informally: golf AM/PM wave and weather
draws (shared conditions factor per wave), course-fit clusters; F1 teammates share a car (a
two-per-team factor is almost mechanical); athletics heats share pace/wind. Top-k events are the
actual betting/fantasy markets: make-the-cut (top ~65), top-10, podium, points (F1 top-10).

**(ii) Public data.** Golf: full leaderboards and strokes-gained from PGA Tour / OWGR; DataGolf
publishes model outputs. F1: complete results history via the Ergast/Jolpica API. Athletics:
World Athletics results database. All free.

**(iii) Incumbent and documented limitation.** Incumbents are independence-based: Plackett-Luce
and variants (generalized PL with ties fit to 47 PGA Tour events, arXiv 2212.08543; PL with
trajectories for luge, JQAS 2022; state-space skill models, arXiv 2308.02414). DataGolf's
predictive methodology simulates each golfer independently:

> "Each iteration draws a score from each golfer's probability distribution, and through many
> iterations we can define the probability of some event (e.g. golfer A winning) as the number
> of times it occured divided by the number of iterations."

(datagolf.com/predictive-model-methodology — no between-player correlation appears in the
methodology; scores are drawn per-golfer per-iteration.) No fitted-correlation ranking model for
golf or F1 surfaced in this scout; the PL family imposes IIA, so wave/weather and teammate
effects are absorbed into noise.

**(iv) Unique advantage.** A 156-player field with a wave/weather factor and team factors is a
one-pass exact computation for win, cut, and top-k probabilities — no simulation — with share
inversion from market odds (the companion horse-racing machinery transfers wholesale, minus the
excluded domain). Withdrawal (WD) repricing = deletion ensemble.

**(v) Minimal demo.** One season of PGA Tour events: fit factor-probit with an AM/PM-wave
factor; compare make-the-cut calibration vs independent PL/DataGolf-style simulation; show WD
repricing. F1 variant already prototyped in this repo (f1_*.py scripts).

**(vi) Venue.** Journal of Quantitative Analysis in Sports, MIT Sloan Sports Analytics
Conference; fast turnaround, good demo visibility, but lower academic prestige than 1-3.

Sources: https://arxiv.org/pdf/2212.08543 ; https://datagolf.com/predictive-model-methodology/ ;
https://www.degruyterbrill.com/document/doi/10.1515/jqas-2021-0034/html ; https://arxiv.org/pdf/2308.02414

---

## 6. Labor: tournaments, promotions, patent races

**(i) Race mapping.** Promotion tournaments (Lazear-Rosen) are argmax races among coworkers with
a common-shock ("relative evaluation removes common noise") — the mapping is textbook. Patent
races: first-to-invent among firms with correlated R&D productivities.

**(ii) Public data.** Patents: USPTO bulk data, interference proceedings (pre-AIA
first-to-invent contests are documented). Promotions: mostly proprietary personnel records
(insider-econometrics literature); some public sector promotion data.

**(iii) Incumbent and documented limitation.** Empirical patent-race support is weak:
Cockburn & Henderson (1994) find R&D investment "very weakly correlated across firms" in
pharma, with "little evidence that firms' R&D investment is reactive" to rivals (as summarized
in Thompson & Kuhn, "Does Winning a Patent Race Lead to More Follow-on Innovation?", J. Legal
Analysis 2020, which itself constructs patent races from interference records). Tournament
empirics focus on incentive effects, not on computing finishing-order probabilities; no
computational bottleneck complaint comparable to the other fronts was found.

**(iv) Unique advantage.** Exists (exact race probabilities with correlated contestants), but no
documented pain point and thin public data for promotions.

**(v) Minimal demo.** Patent interference records (Thompson-Kuhn data) as small races — but N is
tiny (2-3 parties), so the capability is overkill.

**(vi) Venue.** Labour Economics / JOLE — but the fit is weak.

Sources: https://academic.oup.com/jla/article/doi/10.1093/jla/laaa001/5864577 ;
https://www.sciencedirect.com/science/article/abs/pii/S016518891930020X

---

## Ranking

1. **RCV / elections (front 2).** Full-ranking public data at scale (Harvard Dataverse CVRs, NYC,
   Alaska), an incumbent (Plackett-Luce) that hard-codes IIA, a verbatim scalability complaint
   about exactly our model class (Loaiza-Maya & Nibbering), and the spoiler/exit question is
   natively the deletion ensemble. Nobody found fitting correlated Thurstone to CVRs. Highest
   novelty-to-effort ratio.
2. **School choice (front 1).** The field already runs multinomial probit via Gibbs and already
   validates menu-change counterfactuals (Pathak-Shi); we slot in exact probabilities, Jacobians,
   and deletion pricing. Top-tier venues; caveat: strategic-reports combinatorics is not ours to
   solve, and flagship datasets are restricted-access.
3. **kth-to-default baskets (front 3).** Structurally identical factor model; incumbents
   (recursion/FFT/saddlepoint) compute loss counts, not per-name first-to-default identity
   probabilities + Jacobians — that gap is real and demonstrable on synthetic baskets. Ranked
   third because the market is niche post-2008 and data is paywalled (citations-only per
   instruction).
4. Sports (golf/F1): easy wins and public data, but lower-prestige venues; good demo material.
5. Auctions: good data, real hardness citations, but the honest framing is reduced-form
   screening, not structural equilibrium — a positioning fight.
6. Labor/patent races: weak empirical mapping (Cockburn-Henderson), tiny N; pass.
