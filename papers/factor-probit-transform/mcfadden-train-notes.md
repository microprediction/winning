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

## Open, and worth doing

1. **Verify the theorem statement and its construction** against the published
   paper. Everything above marked CHECK.
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
