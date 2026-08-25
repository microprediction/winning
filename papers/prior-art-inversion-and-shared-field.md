# Prior art: the inversion, the shared field, and the deletion ensemble

Assembled 2026-08-24 by adversarial search (WebSearch exhausted; everything via
Crossref/OpenAlex/arXiv/Unpaywall/Semantic Scholar APIs, IA Scholar and Open
Library full-text search, Google Patents, Wayback, and direct PDF reads).
Spans both the 2021 SIAM paper and `factor-probit-transform/`.

**The one-line conclusion.** The defensible claim is *the first fast numerical
inversion of the N-wise argmax map at large N for general base densities.* It
is **not** first formulation, first integral, first identification result,
first non-Gumbel treatment, first shared-field assembly, or first deletion
ensemble. Each of those has prior art, some of it very old, and the paper is
stronger for conceding all of them up front.

---

## 1. The one that needs handling: Li 2018

**Lixiong Li, "A General Method for Demand Inversion," arXiv:1802.04444
[econ.EM], Penn State, Feb 2018, 12pp.** Preprint, no journal version found,
but public, dated and citable.

Inverts the market-share map for `u_ji = x_j + eps_ij` with an outside option,
`F` **arbitrary** (non-atomic for the iff, absolutely continuous for
smoothness). Theorem 1: with consumer surplus `U(x) = E[max(0, max_j(x_j +
eps_ij))]`, `sigma(x*) = sigma*` implies `x* in argmin_x {U(x) - x' sigma*}`,
and conversely when F is non-atomic. So **the inversion is an unconstrained
convex minimization**, `dU/dx_j = sigma_j(x)`, with globally convergent
superlinear methods and trust-region handling of the singular-Jacobian case —
which is precisely the near-zero-share / long-shot regime. Explicitly beats
the BLP contraction (global but *linear*, with alpha near 1; cites Dube-Fox-Su
2012) and covers the pure characteristics model where no contraction exists.

The i.i.d.-shifted race is a direct special case. **Rate this a DIRECT HIT on
the inversion claim as published in 2021.**

### Why priority is nevertheless intact

The author's own public disclosure predates it by four and a half years:
**"The Horse Race Problem: A Subspace Solution," finmathblog.blogspot.com,
bylined Sunday 15 September 2013**, URL under `/2013/09/`, Wayback captures
from 30 Sep 2015 onward. Read in full from the capture, it contains the
problem statement, the translation family `f_i(x) = f(x - a_i)`, the integral
`p_i = int f(x; a_i) prod_{j != i} (1 - F(x; a_j)) dx`, the inversion recipe
("set a_1 = 0 and throw the rest to a solver"), MATLAB Levenberg-Marquardt
timings ("200 runners takes on the order of one hour"), a log-odds
discrepancy, and a k-means subspace acceleration giving ~50x at n ~ 200.

**And the SIAM paper already cites it** — the Crossref reference deposit for
DOI 10.1137/19M1276261 lists "P. Cotton, The Horse Race Problem, 2013,
working paper, http://finmathblog.b…".

So Li is prior art against the 2021 *publication*, not against the 2013
disclosure. Priority is fine.

### The actual exposure, and the fix

The SIAM paper's 28-reference list contains **none of the demand-inversion
econometrics**. A referee who finds Li sees an uncited direct competitor. Fix:
cite Li 2018 in any new paper, state the relationship plainly, and make the
distinction that is genuinely ours — Li supplies a convex *formulation and
optimiser*; he does not supply a way to evaluate `sigma(x)` and its Jacobian
for all N alternatives at large N. **Li 2018 plus the shared-field forward map
IS the large-N inversion algorithm.** That is a complement, not a collision,
and saying so is much stronger than hoping nobody notices.

---

## 2. The shared-field assembly is older than we thought: 1975

The forward divide-one-out (form `prod_j F_j` once, strip `F_i` to get all N)
is a repeatedly rediscovered primitive. Earliest found:

- **Lambert, H.E., *Fault Trees for Decision Making in Systems Analysis*, PhD
  thesis, UCRL-51829, Lawrence Livermore Laboratory, Oct 1975, DOI
  10.2172/4169124.** Barlow-Proschan importance — "Probability that event i
  causes system failure" — for all i, with the method stated outright: "By
  this method, we do not have to recompute the probability of the top event
  each time." Shipped FORTRAN.
- **Statistical static timing analysis, independently:** Xiong, Zolotov,
  Venkateswaran & Visweswariah, DAC 2006, DOI 10.1145/1146909.1146929
  ("criticality probability of every edge... with linear complexity"); Mogal,
  Qian, Sapatnekar & Bazargan, *IEEE TCAD* 28(3):350-363, 2009, DOI
  10.1109/TCAD.2009.2013278 (explicit prefix/suffix lists, O(E^2) -> O(E));
  Sinha et al., DAC 2012, DOI 10.1145/2228360.2228554 ("Reversible statistical
  max/min operation" — the divide-out analogue).
- **Chen & Liu, *Statistica Sinica* 7:875-892, 1997**, p.880: all N
  leave-one-outs in O(nN) rather than O(nN^2) by stripping one term from a
  shared total.
- **Spouge, Ziegelbauer & Gonzalez, *Alg. Mol. Biol.* 15:17, 2020**, DOI
  10.1186/s13015-020-00178-x — names the primitive "jackknife (leave-one-out)
  products" and calls divide-the-total "an obvious algorithm".
- Already known: Mahani & Sharabiani (competing risks, software 2015);
  **Russo 2016** arXiv 1602.08448 Appendix B Algorithm 4 (bandits, on a
  quadrature lattice, O(KM) for all K).
- Useful baseline: **Scott 2010**, *ASMBI* 26(6):639-658, DOI 10.1002/asmb.874
  — the exact integral for all arms with naive O(N^2 L) R code.

**Consequence:** never claim the assembly. `factor-probit-transform` already
says the novelty is "not... the shared-product assembly in isolation" — keep
that, and cite Lambert 1975 as the earliest instance.

---

## 3. The deletion ensemble is anticipated, and by a long way

- **Chiang, C.L., *Proc. Fourth Berkeley Symposium* IV:169-180, 1961** —
  "**partial crude probabilities**" are exactly `q[i][j]`, in closed form for
  every entry (eq. 21), with the deleted factor divided out **in log space**,
  and arbitrary deleted subsets (eq. 24). Restated Chiang 1968 (Wiley, p.257),
  1979 (WHO), 1991 (*Annu. Rev. Public Health* 12:281-307).
- **Beltran-Sanchez, Preston & Canudas-Romo, *Demographic Research*
  19(35):1323-1350, 2008** — literally `p_{-i}(a) = p(a) / p_i(a)`, all n
  divide-outs in one summed pass, no proportionality assumption.
- Root identity: **Makeham, *JIA* 18(5):317-322, 1874**, after Bernoulli 1760
  and D'Alembert.
- The **Kimball 1969 vs Chiang 1970** *Biometrics* exchange (DOI
  10.2307/2528793 vs 10.2307/2528722) is exactly naive-renormalisation versus
  true deletion — i.e. the IIA-versus-correlated distinction, in 1970.
- Ensembles tabulated at scale: Dublin, Lotka & Spiegelman 1949;
  Preston, Keyfitz & Schoen 1972 (180 populations).

**Unclaimed remainder:** doing it inside factor quadrature, and the N x N
array as one vectorised object. Claim only that.

---

## 4. Identification: Anderson-Ghurye 1977 -> Mukherjea-Stephens 1990

- **Anderson & Ghurye, "Identification of Parameters by the Distribution of a
  Maximum Random Variable," *JRSS-B* 39(3):337-342, 1977**, DOI
  10.1111/j.2517-6161.1977.tb01632.x.
- **Mukherjea & Stephens, *Probab. Theory Relat. Fields* 84(3):289-296, 1990**,
  DOI 10.1007/BF01197886 — general multivariate normal; extended by Dai &
  Mukherjea, *J. Theoret. Probab.* 14(1):267-298, 2001.
- Chain: Nadas 1971 (N=2); Mukherjea-Nakassis-Miyashita 1986; Basu-Ghosh 1978;
  Gilliland-Hannan 1980; Elnaggar-Mukherjea 1999; Davis-Mukherjea 2007;
  Bi-Mukherjea 2010.

Establishes that the mean vector (and covariance structure) of a multivariate
normal is **uniquely determined** by the distribution of the maximum, or by
the identified minimum — whose sub-distribution functions generalise the
winning probabilities. General N. **Theorems only; no numerics.** Apparently
uncited in Cotton 2021. The algorithmic contribution survives it, but the
existence half should be attributed here rather than argued afresh.

---

## 5. Psychometrics: a verified negative, and three pressure points

**Verified negative, and it is the best affirmative evidence of novelty
available.** Nobody in classical or modern psychometrics numerically solves
`W_i(mu) = p_i` for non-Gumbel `f` at N > 2. Thurstone 1931 through Guilford
1937, Torgerson 1958, Maydeu-Olivares, and TrueSkill 2007 all reduce the N-way
max to **pairwise** comparisons. Say so in the paper: the consistent dodge to
pairwise is the evidence.

Three citations that constrain the framing:

- **Thurstone, "The Prediction of Choice," *Psychometrika* 10(4):237-253,
  1945**, DOI 10.1007/BF02288891 (repr. *The Measurement of Values* 1959,
  ch.13). **The exact integral, with a numerical FORWARD algorithm, for
  arbitrary laws, in 1945.** His eq. (3) is the survival-product first-choice
  integral generalised to n alternatives, with a discretised computing formula
  and worked tables, and explicitly: "this computing formula has no
  restriction as to the shapes of the affective distributions"; "For large
  groups, the computational procedure can be rearranged in a more economical
  manner." Direction is explicitly forward. **The most damaging citation
  against any framing that presents the integral or its numerical evaluation
  as new.**
- **Guilford, "Scale Values Derived from the Method of Choices,"
  *Psychometrika* 2(2):139-150, 1937**, DOI 10.1007/BF02288390 — earliest
  statement of the *backwards* problem at N > 2, but solved by a
  composite-standard ratio that Torgerson (1958, p.194) shows is
  Bradley-Terry in disguise. Close on problem statement, not on method.
- **Torgerson, *Theory and Methods of Scaling*, Wiley 1958, pp.193-194** —
  pre-empts the identification count: "The raw data consist of n frequencies
  of which n-1 are independent... n scale values of which two are arbitrary.
  The degrees of freedom would thus seem to be... 0. We ought, therefore, to
  be able to fit perfectly." Also records the abandonment.

Also: **Bock & Jones 1968** ch.9 "Prediction of First Choices" — forward only,
normal and logistic. **Vojnovic & Yun, ICML 2016 / arXiv 1705.00136** —
non-Gumbel (incl. Gaussian) top-1 estimation from sets >= 2; defeats any
"first non-Gumbel choice-from-N" claim; MSE theory, no inversion algorithm.
**Lam, Koning & Franses, *Multivariate Behavioral Research* 46(5):803-816,
2011**, DOI 10.1080/00273171.2011.606754 — "independent locally shifted random
utility models" with common shape; undercuts "arbitrary base density" as
stand-alone novelty.

---

## 6. Non-logit algorithmic inversion before 2013

- **Berry & Pakes, "The Pure Characteristics Demand Model," *IER*
  48(4):1193-1225, 2007**, DOI 10.1111/j.1468-2354.2007.00459.x, with
  **Song 2006** (SSRN DOI 10.2139/ssrn.1349991). Inverts shares to qualities
  with **no i.i.d. idiosyncratic shock**, so genuinely non-Gumbel and with no
  contraction available. Per Li's characterisation: "Berry and Pakes (2007)
  provides three algorithms... and propose to combine all the algorithms, as
  none of them works on its own"; "none of the existing algorithms ensures
  global convergence," failing at near-zero shares. **CLOSE**: pre-2013,
  algorithmic, non-logit — but a different model (no i.i.d. shock, hence not
  the race integral) and fragile.
- **Convex-duality lineage:** Williams, *Environment and Planning A*
  9(3):285-344, 1977, DOI 10.1068/a090285; Daly-Zachary 1978; McFadden
  1978/1981 — shares as the gradient of a convex surplus, so the inversion is
  a convex program "in principle" since the 1970s. Li's Theorem 1 is two lines
  from Williams. Uncredited member to add: **Fosgerau, McFadden & Bierlaire,
  "Choice probability generating functions," *J. Choice Modelling* 8:1-18,
  2013**, DOI 10.1016/j.jocm.2013.05.002.

---

## 7. Cleared

Inverse competing risks (classical crude-to-net inversion is closed form only
because proportional hazards is the logit-trivial case; the non-trivial UDD
case solved only at m <= 3); parametric competing-risks MLE (Cox 1959,
Herman-Patell 1971, Moeschberger-David 1971) uses failure times plus causes —
more data than proportions; TrueSkill (Herbrich-Minka-Graepel, NIPS 2006)
updates from rankings by EP, never inverts a probability vector; Sawtooth
Randomized First Choice tunes two variance scalars, not N locations; patents
clean. Odds-to-probabilities (Shin 1992/1993, Jullien-Salanie 1994,
Gandhi-Serrano-Padial *ReStud* 2014) is a different map.

---

## 8. What to write

1. **Cite Li 2018** and position as complement: he has the convex formulation
   and optimiser, we have the evaluation of sigma and its Jacobian for all N at
   large N. Together they are the algorithm.
2. **Cite Lambert 1975** for the shared-field assembly; keep conceding it.
3. **Cite Chiang 1961** for the deletion ensemble; claim only the factor-
   quadrature and vectorised-array parts.
4. **Cite Anderson-Ghurye 1977 / Mukherjea-Stephens 1990** for identification.
5. **Cite Thurstone 1945, Guilford 1937, Torgerson 1958** — and use the
   pairwise dodge as the affirmative novelty argument.
6. Note the 2013 blog disclosure explicitly where priority matters.

**Unread in full (do not over-attribute):** Berry-Pakes 2007 internals
(characterised via Li), Guilford 1937 original (read via his 1954
restatement), David-Moeschberger 1978, Preston-Keyfitz-Schoen 1972.

---

## 9. Berry-Linton-Pakes 2004 and Freyberger 2015 (obtained 2026-08-24)

Both previously flagged "unobtained — paywalled/403" are now read, via open
working-paper versions. **Version caveat:** quotes below are verbatim from the
working papers, not the paywalled journal typesettings, so page numbers and
possibly minor wording differ from *ReStud* 71(3):613-654 and *J.
Econometrics* 185(1):162-181. The substantive results quoted (rates, bias
orders) match the published abstracts.

### 9a. Berry, Linton & Pakes

**Source obtained:** LSE STICERD discussion paper EM/00/400, July 2000, 44 pp,
"Limit Theorems for Estimating the Parameters of Differentiated Product Demand
Systems," http://eprints.lse.ac.uk/2032/ (PDF at researchonline.lse.ac.uk).
Saved locally as `blp2004.pdf` in the session scratchpad. The PDF's text layer
is font-shift encoded (ASCII+3); decoded programmatically, so within quotes
the math symbols are reconstructed in brackets from context — the prose is
verbatim.

**Setup.** GMM estimation of differentiated-product demand as J (products in a
market) grows, with three error sources: sampling error in observed shares
(consumer sample of size n), simulation error in model shares (ns draws), and
model error. Delta is defined by the inversion sigma(theta, x, xi) = s, eq.
(1) of the paper.

**Rates for the share-inversion step (Intro, WP pp. 2-3).** For logit and
random-coefficients logit (BLP-type):

> "Under quite general conditions we show that in the logit and random
> coefficient logit cases the estimator will be consistent if [J log J / n]
> and [J log J / ns] converge to zero as J increases. For asymptotic normality
> at rate [sqrt(J)] in these cases we require [J^2 / n] and [J^2 / ns] to be
> bounded. That is, to obtain a CAN estimator for the parameters of these
> models we require the number of simulation draws and the size of the
> consumer sample to grow as the square of the growth in the number of
> products. This improves on rates reported in BLP." (Intro, WP p. 2)

For the pure characteristics model the rates are the square root of that:

> "We show that to estimate the parameters of the uni-dimensional (one
> characteristic) pure characteristic model consistently we require only that
> [n] and [ns] increase at rate [log J], while for asymptotic normality we
> require only that [J/n] and [J/ns] stay bounded." (Intro, WP p. 2)

**Simulation error in shares is binding, and the mechanism is the inversion
Jacobian.** This is the paper's central mechanism statement:

> "In particular in the models with diffuse substitution patterns, such as the
> random coefficient logit model of BLP, all goods are substitutes for all
> other goods and [d sigma / d delta] goes to zero as the number of products
> increase. As we will show it is the inverse of this partial that determines
> the impact of simulation and sampling error on the estimate of [delta] that
> satisfies (1). When the partial disappears this inverse grows large. So when
> [J] is large a little bit of simulation or sampling error causes large
> changes in the computed value of [delta]." (Intro, WP pp. 2-3)

And restated with the rate attached (Section 5, WP p. 23):

> "Note the contrast to the logit-type models, where [n] must increase at rate
> [J] for consistency and rate [J^2] for the asymptotic normality result [when
> all shares are the same magnitude]. The difference between the models is due
> to the difference between localized and diffuse competition. In the models
> with idiosyncratic errors, the derivative of market share with respect to
> product quality is declining at the same rate as the shares. Therefore, the
> elements of the inverse derivative matrix [(d sigma / d delta)^{-1}] are
> growing in [J], and the number of simulation draws must increase at a faster
> rate to offset this."

**Computation described as a burden.** The inversion/estimation is explicitly
framed in computational-burden terms (Intro, WP p. 3):

> "This suggests that for fixed [J] we should be able to obtain well behaved
> parameter estimates from the pure characteristic model with fewer simulation
> draws than we need to use in estimating BLP's model. We provide a small
> monte carlo study which indicates that the difference is rather dramatic.
> This is one reason to expect the computational burden of the pure
> characteristic model to be less than the computational burden of BLP's
> model. Berry and Pakes (1999) show that the computational burden of
> obtaining the [delta(theta, x, s)] from the system in (1) is typically
> larger for the pure characteristics model than it is for BLP's model. So
> there is a trade off to be considered when comparing the computational
> burden of the two models. What this paper suggests is that to obtain
> well-behaved parameter estimates we will have to have much larger consumer
> samples and a much larger number of simulation draws if one uses BLP's
> system than if we use the pure characteristics model."

Monte Carlo corroboration (Section 6, WP pp. 24-25): "the estimation routine
performs badly at very low values of simulation draws"; "the bias seems to
increase in [J] holding [n] fixed — the bias is high when [n] is small
relative to [J]." Table 2 note (decoded from the shifted table font): "With
100 products and only 10 draws, we had numeric problems computing the
estimates."

**Probit:** the word appears only in the bibliography (Hausman & Wise 1978).
The paper's two worked classes are logit/RC-logit and pure
characteristics/vertical. No probit rates are derived — do not attribute any
probit-specific statement to BLP 2004.

### 9b. Freyberger

**Source obtained:** cemmap working paper CWP19/12, "Asymptotic theory for
differentiated products demand models with many markets," version dated April
15, 2012, 56 pp,
https://www.cemmap.ac.uk/wp-content/uploads/2020/08/CWP1912.pdf. Saved locally
as `freyberger_cwp1912.pdf`. Published as *J. Econometrics* 185(1):162-181
(2015). Page cites below are the WP's printed page numbers.

**Setup.** J fixed, number of markets T -> infinity; shares approximated by
Monte Carlo with R draws; estimation by GMM with the BLP inversion done by
contraction inside the objective.

**Simulation error in shares is a first-order, bias-inducing constraint.**
Abstract:

> "It is shown that the estimated parameters are [sqrt(T)] consistent and
> asymptotically normal as long as the number of simulations R grows fast
> enough relative to T. Monte Carlo integration induces both additional
> variance as well additional bias terms in the asymptotic expansion of the
> estimator. If R does not increase as fast as T, the leading bias term
> dominates the leading variance term and the asymptotic distribution might
> not be centered at 0."

Rates, precisely (p. 2 and Theorem 2 discussion, p. 20):

> "sqrt(T) (theta-hat - theta_0) ->d N(lambda_2 mu, V_GMM) where lambda_2 =
> lim sqrt(T)/R ... Hence, if sqrt(T)/R is bounded away from 0, Monte Carlo
> integration (as opposed to evaluating the integral) leads to an asymptotic
> normal distribution of the estimated parameters which is not centered at 0."
> (p. 2)

> "The leading variance term is of order [1/sqrt(R)] while the leading bias
> term is of order [sqrt(T)/R]. Hence, if R grows slower than T, the leading
> bias term dominates the leading variance term which may lead to an
> asymptotic distribution that is not centered at 0." (p. 20)

With the *same* draws reused across markets the requirement strengthens: "If
the same R draws are used in all markets one needs T/R to be bounded to obtain
[sqrt(T)] consistency" (pp. 2-3). Sampling error in observed shares behaves
identically: "Similar to Berry, Linton, and Pakes (2004), one could assume
that one does not observe the true markets shares but an approximation from n
random consumers. ... The rate at which n has grow to relative to T in order
to obtain [sqrt(T)] consistency is identical to the rate requirement for R."
(p. 3, sic). Explicit contrast with the BLP-2004 regime: "with J approaching
infinity one needs that J^2/R goes to 0 for an asymptotic distribution that is
not affected by Monte Carlo integration. Contrary, if T goes to infinity one
only needs that sqrt(T)/R goes to 0" (p. 3).

**Computation described as expensive — and 'just use more draws' rejected on
cost grounds** (p. 4):

> "Second, taking a very large number of draws is computationally very
> demanding because one needs to solve a complicated nonlinear optimization
> problem to estimate the parameters. The Monte Carlo results of the random
> coefficients logit model presented in Section 4 are based on a small number
> of products (J = 4) and five random coefficients to make the problem
> tractable. However, in the same setup as in Section 4 but with a sample size
> of J = 24 and T = 1,124 (as in Nevo (2001)) it takes around 24 hours to
> minimize the objective function when R = 2,000 and the starting values of
> the parameters are close to the true values. Since we are dealing with a
> nonlinear optimization problem one needs to use several different starting
> values in applications. With an even larger number of draws or with a larger
> sample size estimating the model can take more than one week."

The cost sits in the inversion loop (p. 30): "The nested fixed point approach
solves the non-linear system of equations given in (1) using a contraction
mapping when evaluating the objective function for a certain parameter value.
In each step of the contraction mapping the integrals have to be calculated."
Consequence of too few draws (Table 1, p. 31): "with 800 markets the actual
coverage rate is only 68.8% with 50 draws while it increases to 90.8% with 800
draws."

**Probit:** zero occurrences of the word in the paper. Everything is
random-coefficients logit (Nevo parameterization); the inversion uses the BLP
contraction, which does not exist for probit-type models without the i.i.d.
Gumbel shock.

### 9c. Upshot for our paper

Both papers treat simulated shares as the binding input to the inversion:
BLP-2004 proves the inversion Jacobian inverse blows up in J for
diffuse-substitution (logit-type) models, forcing n, ns ~ J^2; Freyberger
proves that with many markets simulation error in shares produces a
first-order *bias* (order sqrt(T)/R) plus undercoverage, and documents
day-to-week runtimes driven by integral evaluation inside the contraction.
Neither derives anything for probit; neither offers a fast non-logit
inversion. Safe to cite both for "simulation error in shares is the binding
constraint on the inversion, and the required draw counts are the
computational bottleneck" — with rates J^2/ns (BLP-type, J -> inf) and
sqrt(T)/R (fixed J, T -> inf) respectively.
