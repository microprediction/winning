# McFadden & Train (2000): the objection this paper has to answer

**Status: action required.** McFadden & Train, "Mixed MNL Models for Discrete
Response", *Journal of Applied Econometrics* 15(5):447–470 (2000), is **not
cited anywhere in scalable-share-calibration.tex**. The paper cites McFadden (1984) for
factor-analytic MNP and Train (2009) for the textbook, but not the 2000
approximation theorem. That theorem is the standard reason an econometrician
gives for not needing probit, so a referee will raise it, and the paper should
raise it first.

## The objection, in one line

Mixed logit can approximate any random utility model arbitrarily closely.
Therefore probit is redundant, and a paper whose premise is "probit's
computational barrier is not real" is solving a problem the field already
routed around in 2000.

## What the theorem actually says

**PROVENANCE WARNING.** I could not fetch the primary text in the session
where these notes were written — two candidate URLs 404'd and the search
budget was exhausted. Everything in this section is from knowledge and is
**UNVERIFIED**. Verify against the published paper before putting any of it in
print. The specific things to check are flagged.

The result is a density/approximation theorem. Mixed logit choice
probabilities have the form

    P_i = integral [ exp(V_i(beta)) / sum_j exp(V_j(beta)) ] f(beta) dbeta

and the claim is that for any random utility model there exists a mixing
distribution f making the mixed logit probabilities arbitrarily close to it.

The construction to verify: I believe it works by pushing essentially all the
randomness into the mixing distribution while the i.i.d. extreme-value term is
made negligible, so that the inner logit tends to an indicator of the argmax.
If that is right, then:

- **CHECK:** does the proof require the scale of the extreme-value term to go
  to zero, or an equivalent device?
- **CHECK:** does it require the mixing distribution to become arbitrarily
  complex — many support points, or growing dimension?
- **CHECK:** do the authors state practical caveats about estimability or
  identification of the required mixing distribution?
- **CHECK:** does the paper discuss multinomial probit as a comparison at all?

The distinction that matters is between a **limit** statement and a statement
about **finite rank at a fixed scale**. A density theorem says the closure of
the mixed logit family contains the target. It does not say a mixed logit with
a handful of random coefficients and an ordinary extreme-value term is close
to a given probit.

## Two structural facts that cut AGAINST this paper

Both should be stated in the paper rather than left for a referee.

**1. Mixed logit is itself a factor model.** With random coefficients
beta ~ N(b, Sigma_beta) and utilities U_i = x_i' beta + eps_i, the induced
error covariance is

    X Sigma_beta X' + (i.i.d. extreme value)

which is low rank plus diagonal — structurally the *same object* this paper
computes with. The difference is only the idiosyncratic law: Gumbel versus
Gaussian. So "factor structure" is not the dividing line, and any framing that
implies it is will be shot down.

**2. The inversion is not blocked for mixed logit.** BLP's contraction mapping
inverts shares to mean utilities for random-coefficients logit perfectly well.
So the calibration capability does not unblock anything a mixed-logit user is
currently stuck on. The inversion contribution is specific to bases without a
closed-form inverse.

Together these narrow the claim considerably, and honestly. The gap is not
"factors" and not "invertibility". It is **the idiosyncratic law at a given
rank**.

## What this paper already measured (verified, from scalable-share-calibration.tex)

Section 8's 2x2 factorial is exactly the right experiment and it is already
done. Design: {Gumbel, Gaussian} base x {V = 0, V = V*}, N = 50, truth
misspecified for all four candidates (t(5) factors, skew-normal idiosyncratic,
standardised), all four calibrated to identical full-menu shares (residuals
<= 4e-10), **oracle loadings** given to both factor candidates so the family
axis is isolated from estimation. Idiosyncratic scale held common; the Gumbel
base is variance-standardised so the family axis does not move the
factor-to-idiosyncratic variance ratio.

Score: misallocated fraction of redistributed mass on deletion, by stratum of
deleted mass.

| model | >10% | 2–10% | 0.5–2% | 0.05–0.5% |
|---|---:|---:|---:|---:|
| independent Luce (plain logit) | 0.229 | 0.245 | 0.261 | 0.305 |
| independent probit | 0.210 | 0.232 | 0.230 | 0.267 |
| **factor mixed logit** | 0.114 | 0.146 | 0.199 | 0.244 |
| **factor probit** | **0.045** | **0.062** | **0.094** | **0.116** |

At matched rank, factor probit misallocates about **2.5x less** than factor
mixed logit. Twenty-seed replication (experiment 36) reproduces the ordering in
every seed and every stratum.

## The decomposition is the interesting part, and it is under-sold

From the two best-resolved strata:

- factor increment: **+0.115 / +0.099** within Gumbel, **+0.165 / +0.170**
  within Gaussian — dominant
- family increment **alone**, at V = 0: **+0.019 / +0.013** — nearly nothing
- family increment **with factors present**: **+0.069 / +0.084** — 3.5–4x larger
- interaction negative: factor structure and Gaussian base are **complements**

So the Gumbel-versus-Gaussian effect is close to invisible without factors and
several times larger with them.

**This is probably why the field believes mixed logit is enough.** The natural
test of "does probit beat logit" is the plain independent comparison, where the
honest answer is "barely" (0.229 vs 0.210 here). Anyone who ran that test and
stopped would conclude probit is not worth the trouble. The base law only earns
its keep once correlation is present — which is exactly the regime that was
computationally out of reach, so the experiment that would have shown it was
the one nobody could run.

That is a much better story than "probit is better", and it is already in the
data. It deserves to be stated as a finding rather than left inside a
factorial decomposition.

## The defensible claim, and the rebuttal it must survive

**Claim.** At matched rank, a Gaussian base beats a Gumbel base by 2–2.5x on
substitution accuracy. McFadden & Train say Gumbel catches up in the limit of
unbounded mixing complexity. The question is whether you can afford the rank.

**Rebuttal to expect.** *"Your mixed logit had a fixed extreme-value scale.
Shrink it and add mixing dimensions and I match you."* This is correct as
mathematics. The answer has to be about cost, not capability: what does adding
mixing dimensions cost in simulation error, identification, and convergence?
That is an empirical question about estimation practice and this paper does not
currently answer it.

**Therefore the argument must be made on rank efficiency, not on capability.**
Any wording suggesting mixed logit *cannot* represent these substitution
patterns is wrong and will be caught.

## The rebuttal has now been costed — from the incumbent's own materials

Everything in this section was fetched and text-extracted from primary
sources by a research agent (2026-08-24): Conlon's NYU graduate IO lecture
notes (chrisconlon/Grad-IO on GitHub), Conlon & Gortmaker (2020), PyBLP source
and shipped data, and Train's textbook chapters 3 and 5. Verbatim quotes;
re-verify before print, but these are direct extractions, not memory.

**1. The field's case against probit is a case against UNRESTRICTED probit.**
Conlon, multinomial_choice2.pdf, slide "Multinomial Probit?": "Sigma has
potentially J^2 parameters (that is a lot)!" and "Each time we want to compute
s_j(theta) we have to simulate an integral of dimension J. I wouldn't do this
for J >= 5." Both objections vanish under a factor structure: parameters
O(J^2) -> O(Jr), integral dimension J -> r. Nobody in this citation chain
argues against a LOW-RANK probit.

**2. The rank-efficiency argument is conceded by the opposing side.** Conlon,
multinomial_choice3.pdf: "If our X's are able to span the space effectively,
then an RC logit model can approximate any arbitrary RUM (such as probit)
(McFadden and Train 2002). **Of course if you have 1000 products and two random
coefficients, you are asking for a lot.**" Random coefficients ARE a factor
basis for the error covariance; the approximation theorem is conditional on
the span.

**3. The rank gap, quantified.** Practitioners use K_2 = 4-6 random
coefficients (counted from PyBLP's shipped Nevo and BLP data). Modern J:
Conlon & Gortmaker p.13, "it is not uncommon for there to be J_t > 3,500
products"; a real PyBLP user runs ~9,000 products with five random
coefficients. A rank-5 basis is being asked to span 3,500-9,000 alternatives.
And raising the rank hits the quadrature wall they document themselves:
product-rule nodes 4^{K_2} (65,536 at K_2 = 8); sparse grids "often involve
negative weights ... which can create problems during estimation."

**4. The "shrink the Gumbel scale" rebuttal FAILS IN PRACTICE, per PyBLP's own
source.** pyblp/economies/problem.py (~lines 1931-1959): epsilon_scale is a
single global scalar; "As this scaling factor approaches zero, the model
approaches the pure characteristics model"; "some values of the simulated
shares can underflow to zero, **causing the contraction to fail when taking
logs**"; "slowly decrease the scale ... until the contraction begins to fail";
"**Ultimately the model will stop being solvable at a certain point**." Not
supported with nonlinear contractions or nesting. The Gumbel kernel is
load-bearing for the BLP inversion: shrink it and the inversion dies. This
answers the McFadden-Train limit argument at the level that matters — the
route to the limit destroys the inversion the method depends on.

**5. The standard large-J escape is logit-only.** Train ch.3: estimation on a
sampled subset of alternatives rests on a logit cancellation (McFadden's
"uniform conditioning property", an explicit use of IIA). The phrases "subset
of alternatives" / "sampling of alternatives" do not occur in Train's probit
or mixed logit chapters. Large-J probit has no existing escape hatch — which
is the gap the O(QNL) forward map and inversion fill.

**6. Prior art to cite and distinguish; identification to pre-empt.**
Elrod & Keane 1995 (factor-analytic, error components), Haaijer et al. 1998,
Yai et al. 1997, Bolduc et al. 1996 — individual-level/panel models, not
aggregate share inversion. Normalization: an unrestricted covariance has
(J-1)J/2 - 1 free parameters after normalization; Train warns (Bunch &
Kitamura 1989) of restrictions that "seemed sufficient to normalize the model
but actually were not." A free per-alternative D plus low-rank V needs an
explicit normalization argument. And lead every pitch with: we invert mu GIVEN
Sigma (Keane 1992 — identification, not computation, is the binding constraint
on learning Sigma from shares alone).

**Unobtained:** Berry, Linton & Pakes (2004) and Freyberger (2015) —
paywalled/403 at time of writing. Do not attribute rate conditions to them
without reading them.

## CORRECTIONS (2026-08-24) — three claims that must NOT be made

A follow-up agent read PyBLP's source, its 188 GitHub issues, its shipped data
and Conlon & Gortmaker in full. Three things I had been assembling into a pitch
are false, and one is disproved by a specific thread.

**1. "BLP breaks down at large J" is FALSE for the demand side.** PyBLP issue
#36: a user runs "about 9000 product firms ... with five random coefficients"
and reports "There is no memory strain on my 16gb RAM laptop", adding that
PyBLP was "much faster" than their Matlab. The memory explosion in that thread
is SUPPLY-SIDE only -- Conlon in-thread: "that tensor needs about 6TB of
memory" for a J^3 object. Do not claim demand estimation fails at large J.
And C&G measure cost "closer to sqrt(J_t)" in J, with footnote 30: "Even for a
large market with J_t = 1,000 products, inverting a 1,000 x 1,000 matrix is
easy relative to numerically computing J_t^2 integrals."

The reason is obvious in hindsight and reframes the whole comparison: **BLP is
fast because it is LOGIT.** The contraction and the shares are closed form.
Our speed advantage is therefore NOT over BLP. It is that probit is
unavailable to them at any useful J at all. Compare against GHK and simulation,
never against the logit contraction.

**2. "BLP is fragile / multiple local optima" is NOT supported by C&G — their
headline says the opposite.** Abstract: "multiple local optima appear to be
rare in well-identified problems"; and "we struggle to replicate some of the
difficulties found in the previous literature", with "tighter optimization
tolerances suffice to eliminate all dispersion for the problem in Nevo
(2000b)". What survives is numerical rather than multi-modal: tolerance
propagation (Dube-Fox-Su 2012) and "simulation error contributes substantial
instability to this particular configuration".

**3. "Practitioners feel constrained by K_2" is NOT established.** No clean
statement of the form "we wanted more random coefficients but could not afford
them" was found. The only near-evidence is the maintainer advising the
9,000-product user to economize: "With five RCs, the number of nodes ... can be
very large". Several issues push the other way, toward RESTRICTING Sigma
(#99 diagonal only, #125, #178 covariance restrictions).

Also correct the arithmetic: BLP autos is 2,217 product-MARKET OBSERVATIONS
across 20 markets, i.e. a median of 106.5 products per market, not 2,217
products. K_2 = 4 (Nevo) and 6 (BLP autos); no application found using more.

### What survives, and it is sharper than what it replaces

The integration wall is real and exponential. C&G, verbatim: "if one needs I_t
points to approximate the integral in dimension one, then one needs I_t^d
points ... This is the so-called curse of dimensionality", with the explicit
rule I_t = 4^{K_2}:

| K_2 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| product rule | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 16,384 | 65,536 |
| sparse grid | 4 | 29 | 69 | 137 | 241 | 389 | 589 | 849 |

Sparse grids trade exponential for polynomial and are paid for in negative
weights, which "can create problems during estimation or when trying to
decompose the distribution of heterogeneity (particularly for counterfactuals)".

**And the argument reduces to one sentence, in their own structure.** In
PyBLP the idiosyncratic term is always i.i.d. Type-I extreme value at fixed
scale, and heterogeneity enters ONLY as a rank-K_2 factor loading on OBSERVED
characteristics X_2 (rc_types are 'linear', 'log', 'logit' transforms of
elliptical draws; one node column per X_2 column). So a free PER-ALTERNATIVE
variance -- our D -- requires product dummies in X_2, i.e. K_2 = J_t, which
detonates integration by the 4^{K_2} rule above.

That is the gap, precisely stated: **not that BLP is slow, but that its
substitution flexibility is confined to a rank-K_2 factor on observed
characteristics, and the one extension that would free it is exponentially
out of reach.** Also worth noting: "probit" appears ZERO times in the entire
PyBLP v1.2.0 source tree, and exactly twice in C&G, both times about
integration technique rather than as a demand model.

**Contested, cite both sides:** Armstrong (2016, Econometrica 84:1961-1980)
finds that "absent strong cost shifting instruments, as the number of products
increases, BLP instruments ... become weak". C&G rebut with their own Monte
Carlos: "We find that when we increase the size of J_t the estimator performs
better rather than worse." Never cite Armstrong alone.

## The free-scale factorial (2026-08-24): the rescaling rebuttal, tested

`../../research/experiments/exp14_boundaries/run_factorial_freescale.py`
(results_factorial_freescale.json). The published factorial hands both factor
candidates the oracle loadings at fixed magnitude; the obvious referee move is
to let each candidate re-optimise one scalar s on its loadings (equivalently,
its factor-to-idiosyncratic ratio -- the first step of the McFadden-Train
shrink-the-Gumbel route). Same truth, same calibration to identical menu
shares at every s, s chosen on a held-out half of the single deletions,
scored on the other half, grid s in [0.25, 8.0].

    published design (both s = 1):  mixed logit 0.1451  probit 0.0638  (2.28x)
    free scale per candidate:       mixed logit 0.1148  probit 0.0484  (2.37x)

Both optima interior: mixed logit picks s* = 2.0 (its curve turns back up by
s = 2.8 and is 3x worse by s = 8), probit picks s* = 1.2. So the Gumbel
candidate does want a bigger factor share than the oracle magnitude -- it
moves toward the limit exactly as the theorem prescribes -- and it recovers
about a fifth of its deficit before turning around, with the ratio ending
slightly wider than at s = 1. At matched rank, rescaling does not close the
gap; only added mixing dimensions could, which is the cost argument above.

Caveat for print: one seed, 15 held-out singles, medians. For a paper claim,
extend to the 20-seed harness of experiment 36. (First run of this script had
a truth-corruption bug -- np.isin fed a Python set, silently all-False, which
made the "deletion truth" the IIA renormalisation and flipped the ordering;
worth remembering as a class of bug that specifically flatters Gumbel.)

## Open, and worth doing

1. **Verify the theorem statement and its construction** against the published
   paper. Everything above marked CHECK. -> DONE, see VERIFIED section at the
   bottom: all four guesses confirmed with verbatim quotes.
2. **Cite it, in the related-work prior-art table**, with a row that concedes
   the density result and states the finite-rank claim.
3. **The missing experiment: matched estimation burden, not matched rank.**
   The current factorial gives both candidates oracle loadings. The comparison a
   referee will actually want is: fit both from data with equal effort, then
   score substitution. Until that is run, the rank-efficiency argument is
   asserted rather than demonstrated.
4. Find whether anyone has published a head-to-head of probit against mixed
   logit on substitution or diversion accuracy. If nobody has, say so — the
   absence is itself informative and makes the factorial more valuable.
5. Check whether the merger-simulation and diversion-ratio literature complains
   that mixed logit substitution is still too restrictive in practice. Logit's
   IIA in diversion ratios is a named criticism; whether *mixed* logit is felt
   to be inadequate is the question that decides whether there is a customer.

## Related

- `scalable-share-calibration.tex` section 8 (the factorial) and experiment 29 (pure-covariance
  version, which removes the unequal-marginal-variance caveat and finds the
  effect larger).
- `../../research/qpo/README.md` for the argmax-side work, where the same
  base-law-versus-factor-structure distinction shows up as the difference
  between the argmax vector and the max distribution.

## VERIFIED (2026-08-24): the theorem, from the primary text

**Source.** The published JAE typeset version, fetched and text-extracted
2026-08-24 from the NYU Stern mirror
(pages.stern.nyu.edu/~wgreene/DiscreteChoice/Readings/McFadden-Train.pdf);
the Berkeley working-paper version also exists at
eml.berkeley.edu/wp/mcfadden0500/mcfadden0500.pdf. Running heads confirm
"J. Appl. Econ. 15: 447-470 (2000)". Quotes below are verbatim from the PDF
text layer, with ligatures (fi/ff/ffi) restored where the extraction dropped
them and math transcribed in ASCII/Unicode; page numbers are the journal's.

**Publication year: 2000, full stop.** Front matter: "JOURNAL OF APPLIED
ECONOMETRICS / J. Appl. Econ. 15: 447-470 (2000)"; "Received 30 November 1998
/ Revised 15 May 2000"; acknowledgements (p. 468): "This paper was first
presented at the University of Paris X in June 1997." A web search finds **no
McFadden & Train joint paper dated 2002**; Conlon's "McFadden and Train 2002"
is a miscitation of this 2000 paper. Cite it as 2000.

### Theorem 1 verbatim (pp. 450-451)

> "Theorem 1. Let z ∈ Z, with Z compact, denote the vector of observed
> attributes of a discrete alternative, and s ∈ S, with S compact, denote the
> vector of observed characteristics of the consumer. Suppose discrete choices
> are made from choice sets C = {z1, ..., zJ}, with at most J̄ alternatives,
> contained in a compact universe C* in which all alternatives are distinct.
> Let z = (z1, ..., zJ), and as a shorthand let C = {1, ..., J}. Suppose
> discrete responses maximize a canonical conditional indirect utility function
> U(zj, s, εj, u) that is a bounded continuous function of its arguments, where
> εj = ε(zj) and u = u(s) are uniformly distributed continuous random fields.
> Assume there is zero probability of ties. Let PC(i | z, s) denote the choice
> probabilities generated by maximization of U over C. If η is a small positive
> scalar, then there exists a continuous function x = x(z, s) of dimension
> 1 × k for some integer k, with x = (x(z1, s), ..., x(zJ, s)), and a random
> utility model with choice probabilities PC(i | x, θ) of the MMNL form (1),
> such that PC(i | z, s) and PC(i | x, θ) differ by at most η for all s ∈ S
> and z ∈ C*."

Note the quantifier order: η is given first, then "there exists ... dimension
1 × k for some integer k". The dimension k is chosen after the tolerance. This
is exactly the density-not-finite-rank reading the notes guessed.

### CHECK 1: does the proof shrink the extreme-value term? YES.

The guessed construction was right. The proof (Appendix, p. 466) builds a
Bernstein-Weierstrass polynomial approximation U^k* to the true utility with
error 1/k*, then perturbs it by extreme-value noise **scaled by 1/k*²**:

> "Form U^k*(zi, s, ε(zi), u(s)) = U^k(zi, s, ε(zi), u(s)) + ζi/k*², where the
> ζi are i.i.d. Extreme Value Type I random variables." (p. 466)

k* grows as η shrinks: "Let k* be the larger of −log(η/4J̄²) and the maximum
value of k(z′, z″, s) for the centres of the finite subcover" (p. 466). The
authors say this in plain words in the body text:

> "In the proof of the theorem, a polynomial approximation to the true random
> utility function is perturbed by adding scaled i.i.d. Extreme Value Type I
> disturbances ζ, yielding MNL as the base model to which mixing is applied."
> (p. 452)

And they state the equivalent coefficient-blow-up device explicitly when
discussing why estimates can diverge (p. 453):

> "Then by scaling down the i.i.d. Extreme Value perturbations to
> U(zi, s, ε, u), one can make the MMNL approximation converge to PC(i | z, s).
> This corresponds to approaching the maximum likelihood by scaling the MNL
> coefficients by a factor c → ∞ in PC(i | x, θ, c) = ∫ LC(i | x, ac) G(da, θ);
> a finite maximand does not exist." (p. 453)

So: shrinking the Gumbel scale (equivalently multiplying utilities/coefficients
by a growing constant) is not an incidental feature — it is the engine of the
proof, and the authors themselves identify it as the direction in which MMNL
likelihoods have no finite maximum when the mixing family nests the truth.

### CHECK 2: does the mixing distribution become arbitrarily complex? YES.

The approximation is by polynomials of growing degree; both the regressor
dimension k and the random-coefficient vector grow with the required accuracy:

> "The continuous function U has a Bernstein-Weierstrauss polynomial
> approximation U^k* on Z × S × [0,1]^(p+r) that satisfies |U − U^k*| ≤ 1/k*."
> (p. 466)

> "Write the polynomial approximation U^k* in the form
> U^k*(z, s, ε(z), u(s)) = x(z, s) a(z, s) + ζ/k*², where x(z, s) is a vector
> of the z and s components of the terms in the polynomial and a(z, s) is a
> vector of the corresponding ε(z) and u(s) components." (p. 466)

> "The construction in the proof shows that the random coefficients a in
> equation (1) can be taken to be continuous polynomial transformations of the
> uniformly distributed continuous random fields ε(z) and u(s)." (p. 451)

And the authors flag exactly the practical gap the notes wanted flagged:

> "One limitation of Theorem 1 is that it provides no practical indication of
> how to choose parsimonious mixing families, or how many terms are needed to
> obtain acceptable approximations to PC(i | z, s)." (p. 452)

Fairness requires quoting their optimistic follow-on (same paragraph, p. 452):

> "However, Monte Carlo studies indicate that fairly simple mixing structures,
> with random coefficients following a factor analytic structure of relatively
> low dimension, and relatively simple mixing families, such as latent class
> models with relatively few classes, are sufficiently flexible to capture
> quite complex patterns of heterogeneity; see Bolduc, Fortin and Gordon (1996)
> and Brownstone and Train (1999)."

Note "factor analytic structure of relatively low dimension" — the authors
themselves describe practical mixed logit as a low-rank factor model, which is
direct primary-source support for structural fact 1 above.

### CHECK 3: estimability/identification caveats? YES, explicitly.

> "A second limitation of the theorem is that while it guarantees the existence
> of a satisfactory MMNL approximation, it leaves open the possibility that
> identification conditions for regular maximum likelihood estimates of the
> MMNL model may fail, or that estimates may blow up." (pp. 452-453)

The blow-up is the c → ∞ passage quoted under CHECK 1; their reassurance is
that misspecification saves you: "This is rarely a practical problem, since any
specification of x and G adopted in an application will almost certainly miss
features of the true random utility model" (p. 453). And from their own
empirical application (p. 462):

> "These results reflect the inherent difficulty of identifying the factor
> structure of unobserved utility from observed data on discrete choices, but
> may also indicate more conventional specification issues such as omitted
> observed variables or interactions."

### CHECK 4: does the paper discuss multinomial probit? YES, three places.

As a target to approximate, with MMNL as the simulator (p. 451):

> "One implication of the theorem is that MMNL can be used to approximate
> computationally difficult parametric random utility models simply by taking
> the distributions underlying these models, suitably scaled, as the mixing
> distributions. These can be interpreted as simulation approximations using a
> MNL kernel. For multinomial probit models, Brownstone and Train (1999) and
> Ben-Akiva and Bolduc (1996) find in Monte Carlo experiments that MMNL gives
> approximations that are as accurate and quick as direct simulation
> alternatives such as the Geweke-Hajivassiliou-Keane (GHK) simulator; see
> Hajivassiliou and Ruud (1994)." (p. 451)

As an alternative base kernel, rejected on computational grounds (p. 452):

> "At this step, one could have used other distributions for the ζ, although
> most alternatives are not as computationally tractable as MNL. For example,
> one might take the ζ to be scaled i.i.d. standard normal. When the mixing
> distribution is multivariate normal, this can be interpreted as the method
> for simulation of the MNP model proposed by Stern (1994). Adopting i.i.d.
> standard normals for the base model adds one dimension of numerical
> integration, and requires computation of a product of univariate normal
> CDF's for each integration point and each decision maker. This requires more
> computation than a MNL base model; see Train (1995)." (p. 452)

And in passing, footnote 3 (p. 456): "These instruments are similar to
instruments for the multinomial probit model proposed independently by Ruud
(1996)."

Read closely, the p. 452 passage is useful to this paper: the authors'
stated reason for the Gumbel kernel over a Gaussian one is **computational
cost of the Gaussian, not fidelity** — the exact trade this paper's O(QNL)
machinery reprices.

### Net verdict on the notes' guessed construction

Right on all counts. The proof pushes all systematic randomness into
polynomial functions of the mixing fields, shrinks the i.i.d. Gumbel term as
1/k*², and lets both the polynomial degree and the coefficient dimension grow
with the accuracy demand. Theorem 1 is a density/limit statement, not a
finite-rank-at-fixed-scale statement, and the authors say so themselves in the
two "limitation" passages (pp. 452-453). The distinction drawn in "What the
theorem actually says" survives verification unchanged, and the rebuttal
framing ("Gumbel catches up in the limit of unbounded mixing complexity")
matches the primary text. One refinement: the shrinking scale and the growing
complexity are a single device, not two — the 1/k*² noise scale and the
degree-k* polynomial are tied to the same index, so the practical cost of
approaching the limit compounds along one path (which is also the path where,
per p. 453, the likelihood ceases to have a finite maximum, and per PyBLP's
own source, the contraction fails).
