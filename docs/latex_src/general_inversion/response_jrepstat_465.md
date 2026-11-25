# Response to the decision on submission #465 (minor revision)

Draft for the revision page. "Done" items name where the change is.
Section numbers are those of the revised PDF (44 pages).

## Required

1. **Accuracy at n = 1,000,000.** Done, both ways. The abstract now says
   what is measured at that size: a calibration round trip to within
   1e-8 in log probability, with external referees reaching n = 2000. Section 7, Scale,
   reports the two covariance classes, the log residuals (7.3e-10,
   7.7e-9) and the ability errors, from the log committed at paper-r12
   (`timings_2026-11-11/retime.json`). It also says the sweep counts were
   not logged and that no block, nested or tree inversion was run at that
   size.
   One correction to the letter: Section 7, Stress boundaries, does not
   report a factor rule "low by a factor of twelve at p near 10^-20". No
   such figure is in the submitted text.
2. **Speed comparison.** Done. The abstract no longer quotes a single
   ratio or an exponent. It says GHK handles general covariance, which
   this method does not, and that within the grammar our implementation
   of it is one to two orders of magnitude slower at n = 200 (eleven
   times at R = 1e3, over a hundred at R = 1e4). Section 1.2 states R
   for both ratios, the range of the 2.9 exponent (n = 200 to 1000 at
   R = 1e3), that the comparator is our own Rust implementation without
   variance reduction, and "close to linear".
3. **Competing interests and funding.** Done. Two unnumbered sections
   after Section 9.
4. **Manifest.** Done. `MANIFEST.md` beside the paper's source maps each
   table and each number in the text to its command at paper-r12, its
   seed and the log line. Writing it showed that the title footnote's
   "every number" was too strong: twelve groups of in-text figures
   (development measurements, several of them describing rules the code
   no longer contains) have no committed command. The manifest lists
   them in its last section, and the footnote and `REPRODUCE.md` now say
   "tables". The revision's own analyses are three small scripts at
   paper-r13.

## Recommended

- **Unqualified statements.** Done: "exact given its structure"
  (Section 4), "all four" (Section 2), "only logit" (Section 8),
  "unthinkable" and "It cannot compete" (Section 1.2).
- **Synthetic data.** Done: a Data paragraph opens Section 7; the
  abstract and Section 8 carry the same scope; a real-data application
  is named as future work.
- **Parity scenarios.** Done (Section 7, Parity). The three are
  constrained projections solved by different optimizers per language.
  No random numbers enter them.
- **"Dominates in both accuracy and cost"; "two to ten times".** Done.
  The n = 10 exception and the split by R are stated.
- **Pairwise diagnostic 0.49.** The reviewer is right. The shipped
  statistic is 3.3e4 on that example; 0.49 was wrong. Corrected, with
  the test that pins it.
- **Genz-Bretz score paragraph.** Done. The arithmetic is now 31 x 3.15 s
  = 98 s, about thirty times the fit, and the paragraph says the tables
  have no such arm.
- **0.25 across starts.** Done (Section 6.2): it is the stage 1
  rank-three fit alone; stages 3 and 4 remove it.
- **G used twice; sign of J.** The winning-time distribution function is
  now H. The display already reads J = -B'W_eB in the submitted PDF
  (p. 19), consistent with the other two statements; no change.
- **TV and abbreviations.** Done: TV defined at first use; MACML, EP,
  QMC, MSL, SPDE expanded; ensemble names pointed to randomcov.
- **GHK against an established routine.** Declined for this revision.
  The paper now states that no third-party GHK was timed and that the
  implementation was not checked against one (Section 7, GHK protocol;
  Section 1.2).
- **Package names and versions.** Done: TruncatedNormal 2.3, mvtnorm
  1.4-2, R 4.6.1. Checking this showed that the minimax-tilting timings
  of Section 2 use our own Python implementation, not the R package;
  the text now says so, and the competing-interests statement lists it.
- **Fishing data.** Reworded to the exact-likelihood finding only.
- **Wilcoxon comparisons.** Done: Table 2 gives per-ensemble paired
  differences, intervals, wins, two-sided p and its Holm adjustment, and
  the text calls the comparison descriptive.
- **Disagreeing figures.** Done for the row-sum and asymmetry figures
  (different statistics, now labeled) and for the three lattice-error
  figures (reference and metric stated; self-convergence is now measured
  by `selfconv.py` and reported in both absolute and log terms).
  "Seven standard errors" is correct: the unrounded ratio is 6.9
  (regenerated with `make estimation`); the 8.5 comes from the rounded
  0.0017 and 0.0002.
- **No third-party GHK sentence.** Done.
- **Solver convergence battery.** Done (Section 5.7): 130 seeded fields,
  125 converge, the five failures are rank-one fields at the largest
  loading scale.

## Optional

Done: Sobol budgets and the non-binding cap; the distribution-family
sentence and "equivalence class"; symbol clauses (nodes and weights,
q_0, h, first(i), root parent, Table 1); root-mean-square z; the
covariate table's limit; own-code statement for Mendell-Elston and
Clark; "projected-residual fit", the Gram form, the rank-27 budget, the
multiplier column; the forward reference to Mueller et al.; Berry,
Levinsohn and Pakes (1995); mixed probit as the factor grammar; the
shipped and in-progress statement; the lockfile statement (randomcov
0.2.1 and the fastrace wheel are pinned; the Rust toolchain is not).

Not done: a notation table for the remaining reused letters; the
log-log plot; an lpRR/slpRR run; an Rprobit check; the general-base
theorem or a variance-reduced GHK arm; a second-machine timing; a
real-data example.

## Production items

Applied in the source: the Appendix A pointer, epsilon in Eq. (1),
American spellings, "Hessian--vector". The p. 3 definition already reads
X_i = -U_i in the submitted PDF; no change.
