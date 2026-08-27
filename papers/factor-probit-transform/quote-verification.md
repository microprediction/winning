# Quote verification (% VERIFY flags)

Verified 2026-08-24 against primary sources. Fetched copies live in
`/private/tmp/claude-501/-Users-petercotton-github-kinetics/d3b37c99-3f1b-45cf-8229-efe313e9c80f/scratchpad/quoteverify/`.

## 1. Torgerson (1958), Theory and Methods of Scaling, pp. 193-194 — VERIFIED

Claimed fragment: "we ought, therefore, to be able to fit perfectly" in a
degrees-of-freedom discussion of scaling from first-choice frequencies.

**Verdict: VERIFIED.** The fragment is verbatim (capitalized "We" as sentence
opener) and the surrounding degrees-of-freedom argument is exactly as
characterized. Full primary passage (p. 194; the method-of-first-choices
discussion begins on p. 193, running head "194 Theory and Methods of Scaling"
precedes this text):

> A third disadvantage, which would seem to be the most serious when first
> choices alone are obtained from the subjects, is that even condition C
> leaves no remaining degrees of freedom.
>
> The raw data consist of n frequencies of which n - 1 are independent.
> Against this, with condition C, are n scale values of which two are
> arbitrary. The degrees of freedom would thus seem to be
> (n - 1) - (n - 2) - 1 = 0. We ought, therefore, to be able to fit perfectly
> with condition C a matrix P derived from any set of frequencies f_j,
> regardless of the characteristics of the stimuli being scaled. Since there
> are no degrees of freedom, we have no way of evaluating the goodness of fit
> of the law to the data.

The passage continues: analysis by condition C "should still yield a scale
which would reproduce the proportions perfectly. Actually, it does not
(Guilford, 1937)." and notes the Bradley-Terry model "would always fit data of
this type perfectly."

Source: Digital Library of India scan of the 1958 Wiley edition on
archive.org (public full text):
https://archive.org/details/dli.scoerat.6037theoryandmethodsofscaling
(file `dli.scoerat.6037theoryandmethodsofscaling_djvu.txt`; the borrowable
copy `theorymethodsofs0000torg` is search-restricted to logged-in users).
Local copy: `quoteverify/torgerson_dli.txt` (quote at line 11922).

## 2. Conlon, Grad-IO multinomial choice slides — VERIFIED (both quotes)

Both quotes are in **multinomial_choice2.pdf** ("Multinomial Discrete Choice:
Nested Logit and GEV", Chris Conlon, Fall 2025), Week 3 - Statistical Demand
Models, slide 3/17 ("Can we do better?" / "Multinomial Probit?" with a
"Downside" block).

URL: https://github.com/chrisconlon/Grad-IO/blob/master/Week%203-%20Statistical%20Demand%20Models/multinomial_choice2.pdf

Exact bullets on slide 3:

> - Sigma has potentially J^2 parameters (that is a lot)!
> - Maybe J * (J - 1)/2 under symmetry. (still a lot).
> - Each time we want to compute s_j(theta) we have to simulate an integral
>   of dimension J.
> - I wouldn't do this for J >= 5.

**Quote A** "potentially J^2 parameters (that is a lot)!" — VERIFIED verbatim
(full bullet begins "Sigma has potentially ...").

**Quote B** "Each time we want to compute s_j(theta) we have to simulate an
integral of dimension J. I wouldn't do this for J >= 5." — VERIFIED verbatim,
with the minor caveat that on the slide these are two consecutive bullets, not
one sentence.

Local copies: `quoteverify/multinomial_choice2.pdf`, extracted text
`quoteverify/mc2.txt` (lines 49-55).

## 3a. Huber, Orme & Miller (1999), "Dealing with Product Similarity in Conjoint Simulations" — VERIFIED

Sawtooth Software Research Paper Series / 1999 Sawtooth Software Conference.
PDF: https://content.sawtoothsoftware.com/assets/4a7f1499-fcb5-4065-8a5c-8c29e62d0a16
(also historically at sawtoothsoftware.com/download/techpap/prodsim.pdf).

**Glossary wording (printed p. 2, under equation (1)) — VERIFIED verbatim:**

> E_A = Variability added to the part worths (same for all alternatives)
> E_P = Variability added to product i (unique for each alternative)

So "Variability added to the part worths (same for all alternatives)" is exact
for E_A, and E_P is indeed unique per alternative ("unique for each
alternative").

**"Numerous ways" sentence (printed p. 5) — VERIFIED; note the sentence
begins "There are":**

> There are numerous ways researchers have attempted to solve this problem,
> from nested logit to correlated error terms within probit.

(Context: the Paris/London red-bus example; logit predicting one-third shares
after adding a discounted duplicate trip to Paris.)

Local copies: `quoteverify/huber_orme_miller_1999.pdf`, `quoteverify/hom99.txt`
(lines 122-123, 286).

## 3b. Orme & Johnson (2006), "External Effect Adjustments in Conjoint Analysis" — VERIFIED

Sawtooth Software Research Paper Series, March 2006.
PDF: https://content.sawtoothsoftware.com/assets/63122e93-b01b-4849-894a-e64f491dd760
(landing page: https://sawtoothsoftware.com/resources/technical-papers/external-effect-adjustments-in-conjoint-analysis)

**"Fudge factors" characterization (printed p. 2, section "Should We Adjust
Shares at All?") — VERIFIED, exact sentence (parenthetical in the original):**

> (Using the word "calibrate" seems to convey a more scientific procedure
> than simply changing the shares to the desired result through "fudge
> factors.")

**Declining to encourage the practice (printed p. 2) — VERIFIED, exact
sentence:**

> The purpose of this paper is not to encourage or justify the widespread
> practice of adjusting conjoint simulators in an attempt to account for
> external effects.

Followed by: "Researchers would do well to deliver the simulator as-is, and
educate managers regarding the assumptions, proper interpretation, and use of
the tool."

Local copies: `quoteverify/orme_johnson_2006.pdf`, `quoteverify/oj06.txt`
(lines 71, 76).
