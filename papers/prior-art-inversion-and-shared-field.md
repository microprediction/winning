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
