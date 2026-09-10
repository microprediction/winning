# Brief for a new reviewer

Manuscript: `papers/thurstone_humans/paper.tex`, 33 pages, commit `057678d` on branch
`machine-preference-paradox` of `github.com/microprediction/winning`. Quote a commit hash in
your report; table and section numbers have moved between rounds and are not stable locators.

Three earlier rounds have been acted on. This note says what is settled, so the fourth round is
not spent re-deriving it, and what is open, so it is not spent discovering it.

Note before you start: the add-alpha and menu-weighting sensitivities are now in the paper, in
the section titled "Two conventions, and what they change". Smoothing is close to inert except on
occupational prestige, which is the collection with 143 respondents and ten alternatives.
Weighting changes magnitude by up to fourfold and never changes sign, and the aggregate reported
throughout is the most conservative of the three weightings. Item 4 and item 3 are therefore
closed.

The refit bootstrap has finished and is in the paper. Respondents are resampled and the whole
pipeline is refit inside every replicate, one hundred replicates or two hundred for the two
collections under a thousand respondents. Every one of the twelve intervals excludes zero,
including puzzles, whose earlier interval held the fitted models fixed and covered it. Refitting
widens most intervals and roughly trebles the width on GSS socialization. Item 1 is closed.

## What the paper claims

Given a probability vector over mutually exclusive alternatives and a smaller support, produce
the vector over the survivors. Two zero-parameter candidates: proportional renormalization,
which is the Gumbel point of the independent additive random-utility class, and re-running a
Gaussian contest among the survivors, which is Case V. The input is a probability vector and
nothing else. Both are calibrated on full-menu shares alone and scored by held-out log loss on
restricted menus neither has seen, across thirty-nine population comparisons.

The claim is that Gaussian renormalization is a serious tuning-free comparator, not that it
wins everywhere: it has the lower held-out log loss in thirty of thirty-nine rows, and the
losses concentrate on one-dimensional and strongly confusable stimulus sets. The boundary rule
is presented as motivated, not established.

## Settled, with the resolution. Please do not re-open these

**Verified correct by three independent recomputations.** Proposition 1 and its proof;
$(\log r)'' = -\operatorname{Var}(Z \mid Z < x)$; the Gumbel affine boundary case; Proposition 2
and the CLT corollary; the pair formula; the location-scale gauge $P^S_i(a;s) = P^S_i(a/s;1)$;
the $O(\varepsilon^3)$ standardized skewness of $Z + \varepsilon G$; the covariance
non-identification count.

**Corrected after earlier rounds found them wrong.** A claim that a common noise scale survives
calibration and propagates into the restricted prediction: false, scale is an exact gauge, and
the passage is gone. A claim that concentrated shares make the two maps agree: false, and the
counterexample $p = (0.90, 0.09, 0.01)$ is now in the text. A scoreboard that awarded wins on
excess over the null rather than on held-out log loss: replaced. The covariance-count argument
at $K = 3$: repaired. "Covariance of utility differences spherical": now isotropy on the
contrast subspace, with the reference-difference algebra given. "Correlated probit handles
near-substitutes exactly": now "can represent". Wikipedia described as impression-level
exclusive choice: now aggregate conditional transition shares. The favourite's second-place
exercise cited as a check against truth: removed, since neither rule there is the ordering law
of its model.

**Answered, and the answer is in the text.** That a deep network is not committed to the axiom
(Section 3: the constraint is on a fixed logit vector, and recomputing scores with the permitted
set as input leaves the transport problem entirely). That truncation is conditioning (Section 1
separates four operations; conditioning and fixed-vector masking are proportional
renormalization by construction, and only feasible-set intervention raises the question).
That Wills et al. did this in 2000 (they race magnitudes supplied by a stimulus model and their
conclusion is conditional on it; an analyst holding shares has nothing to race, and their
Experiment 2 data is one of the thirty-nine rows).

## Open, and known. Confirming these is not useful; fixing or costing them is

1. ~~Ranking intervals hold calibration fixed~~ done. Replaced by refit intervals from
   `sensitivity.py`; all twelve exclude zero.
2. **Monte Carlo replicates are 200, 60 on the news row, 400 on Wills.** The 0.05 threshold is
   inside the noise. Verdicts should stop being binary; exceedance counts should be printed.
3. ~~Menu weighting~~ done, and reported. The aggregate is the conservative weighting.
4. ~~Add-$\alpha$ sensitivity~~ done, and reported. Occupational prestige is the one collection
   that moves, which is worth knowing when reading its $+0.0390$.
5. ~~The full menu is inside the restriction estimand~~ measured, and reported in the section
   "What the score averages over". Excluding $T=S$ raises every gain and lowers none, by $1.33$
   on Netflix, $1.10$ at $K=4$, $1.04$ at $K=5$ and $1.00$ to two decimals at $K \ge 7$. The
   published figures include the tie and are the conservative ones.
6. ~~Size-specific gain vector~~ done. Table 6 gives $(g_2,\dots,g_K)$ for all twelve ranking
   collections. Nine fall monotonically; occupational prestige peaks at $|T|=4$ at about twice
   its pairwise figure, and sports participation peaks at $|T|=3$.
7. ~~No bounded proper score~~ done. Multiclass Brier on the same subsets and folds agrees in
   sign on twelve of twelve. Its intervals cover zero on the two collections with the smallest
   log gains, GSS job values and puzzles. The ordering of collections is preserved.
8. **The cubic order is prose, not a lemma.** Smoothness in an explicit norm, interior shares
   and a nonsingular contrast Jacobian are assumed rather than stated.
9. **No study-level primary table.** Rows overlap and nest; the count is descriptive and says so.
10. ~~One figure has no committed run behind it~~ closed. `menus_heldout.py` now scores the
   pooled forced-choice group, and it reproduces the printed row exactly: gain $+0.0265$, null
   median $-0.0041$, excess $+0.0306$, $p = 0.010$. The audit traces all 504 table figures.

## Where to look before writing

- `research/restriction/demo/index.html` runs twenty checks of the mathematical claims in a
  browser from nothing: the gauge, the reverse-hazard identity, contraction over random menus,
  a simulated Gumbel race renormalizing exactly, the cubic order, the concentrated-share
  example. `node run_checks.js` runs the same suite.
- `research/restriction/demo/check_tables.py` traces every figure the paper quotes from a run to
  the output that produced it, and lists table figures no committed run accounts for. All 504
  currently trace.
- `research/restriction/estimand.py` produces three of the answers above in one pass:
  restrictions only, the size vector, and Brier beside log loss.
- `research/restriction/results/STATUS.md` is the running log, including the errors we found in
  our own work.
- Each collection under `research/restriction/data/` has a `SOURCE.md` with the fetched access
  route, the verification performed, and the caveats.

## What would actually help

Attack the boundary rule. It is the part most likely to be wrong and least constrained by what
we have: it was formed partly from the same tone, letter and near-substitution evidence that
now illustrates it, and only the line-length collection is prospective. A construction where it
predicts the wrong sign, or an argument that the four conditions are not separable, is worth
more than another pass over the algebra.

Second most useful: whether held-out log loss on subset marginals is the right estimand at all,
given that the target menu distribution is a choice and the two maps are compared as procedures
rather than as models.
