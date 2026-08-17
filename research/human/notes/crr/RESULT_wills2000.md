# Result: Wills, Reimers, Stewart, Suret & McLaren (2000, Experiment 2)

Run with `research/human/wills_twochoice.py` on the CAM1 deposit, mirrored to
`research/human/data/wills/`. Five authors, not three: Suret is on the title page even
though the Wills lab publication list omits him.

## The comparison

39 cells (3 disallowed categories x 13 stimulus types), 1,560 held-out two-choice trials.

| | log loss |
|---|---|
| renormalization | 0.6839 |
| Gaussian race | 0.6540 |

gain **+0.0299** [+0.0175, +0.0717] participant bootstrap
fitted-Luce null median −0.0015, excess **+0.0314**, p = 0.002 (400 reps)

Graded and dummy stimuli agree: +0.0287 over catordist 1-9, +0.0326 over 10-13.

## Constraints that must travel with the number

The restriction is **between groups**: condition 1 allows all three category responses,
condition 2 disallows each participant's `fixed` category, and these are different people,
twelve per condition. So four participants underlie each disallowed-category group, and the
gain is concentrated in two of the three:

| disallowed | gain |
|---|---|
| A | −0.0001 |
| B | +0.0413 |
| C | +0.0485 |

The bootstrap interval is correspondingly wide and right-skewed. This is supporting
evidence, not the load-bearing dataset; Townsend & Landon is within subject and carries
that weight instead.

Design verified in the data rather than taken on trust: the disallowed response occurs
zero times in the two-choice condition and 510 times in the three-choice condition, so the
removed category is a real competitor on the master menu, and cells are balanced at 40
trials throughout.

## What Wills et al. did and did not establish

They are the closest prior art on the **refutation** side, and the framing they own must be
conceded rather than reclaimed. Their central conclusion, verbatim: "the ratio rule is an
inappropriate theory of categorical decision and should be replaced by a system based on
the principles of Thurstonian choice." They cite Yellott (1977) four times and state the
n >= 3 divergence exactly, so **no novelty can be claimed for the Yellott framing**.

They did not, however, run a parameter-free out-of-sample test, on four counts checked
against the text:

1. **The positive model uses rectangular noise, not Gaussian.** "the noise added to nu_i
   ranges from +N to −N, has a mean of zero, and has a rectangular distribution." Gaussian
   appears substantively once, as an unquantified aside attached to a simplified model they
   had just called worse and which fails one of their four target functions.
2. **Four free parameters, and the restricted menu gets its own value.** Their words: "The
   WTA model is a relatively complex system with four free parameters (E, D, N, and S). The
   ratio rule, in contrast, has no free parameters." And decisively: "S is set to 0.18 for
   the two-choice condition, 0.65 for the three-choice condition." A menu-specific
   threshold means the two-choice account is not a prediction from full-menu quantities.
3. **The renormalization benchmark is never computed.** The test is the shape of a derived
   statistic q, assessed with quadratics and F-tests, concluding the curves are "of
   opposite shape." Their own Eqs. 3 and 6 imply an exact parameter-free CRR point
   prediction, q = P(A)/[1 − P(A)], computable from the three-choice condition alone. They
   use only the weak monotone corollary and never evaluate the identity numerically,
   although the CAM1 deposit contains everything needed to do so.
4. **No held-out vocabulary anywhere.** Zero occurrences of parameter-free, out of sample,
   cross-validated, or held-out. The five hits on "free parameter" are all in the passage
   conceding that their model has four.

So the honest statement is narrow and still worth making: Wills et al. argued against the
ratio rule and for Thurstonian choice, and left the parameter-free point prediction implied
by their own equations uncomputed on their own deposited data. That is what this run
supplies.
