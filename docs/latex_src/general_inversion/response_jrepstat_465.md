# Response to the decision on submission #465 (minor revision)

The revision starts from the reviewed version (23 November 2026).

Draft for the revision page. "Done" items name where the change is.
Section numbers are those of the revised PDF (43 pages).

## Required

1. **Accuracy at n = 1,000,000.** Done. The abstract now says
   what is measured: calibration is demonstrated at that size, and
   accuracy is checked against external referees up to n = 2000. It no
   longer claims tail accuracy at a million. Section 7, Scale,
   reports the two covariance classes, the log residuals (7.3e-10,
   7.7e-9) and the ability errors, from the log committed at paper-r12
   (`timings_2026-11-11/retime.json`). It also says the sweep counts were
   not logged and that no block, nested or tree inversion was run at that
   size.
   The factor-rule limit the letter cites (low by a factor of twelve at
   p near 1e-20, Section 7, Stress boundaries) stays in the text, and
   the abstract no longer claims tail accuracy at scale.
2. **Speed comparison.** Done. The abstract names GHK as a point of
   comparison without a ratio or exponent, and says cost is close to
   linear in n. Section
   1.2 states R for both ratios (over a hundred times at R = 1e4, about
   eleven at R = 1e3), the range of the 2.9 exponent (n = 200 to 1000 at
   R = 1e3), and that the comparator is our own Rust implementation
   without variance reduction.
3. **Competing interests and funding.** Done. Two unnumbered sections
   after Section 9 give the affiliation (microprediction.com, founded
   and owned by the author), the interest in the software and platform
   (no revenue from either), the self-written comparators, and the
   funding (none external).
4. **Manifest.** Manifest supplied; exceptions explained. `MANIFEST.md`
   beside the paper's source records, for each table and number, the
   command at paper-r12, its seed and the log line, wherever a committed
   command produces it. The request for complete regeneration is only
   partly met. Twelve groups of figures have no committed command: they
   are development measurements, several describing implementations
   since superseded, and they include the entries of the equal-groups
   table of Section 6.2. We decline to recreate those experiments for
   this revision. The figures remain in the paper, and the manifest's
   last section lists each group, where it appears and where it is
   recorded. Section 9, `REPRODUCE.md` and the title footnote no longer
   claim a script behind every number. The revision's own analyses are
   three small scripts at paper-r13.

## Recommended

- **Unqualified statements.** Done: "exact given its structure"
  (Section 4), "all four" (Section 2), "only logit" (Section 8),
  "unthinkable" and "It cannot compete" (Section 1.2).
- **Synthetic data.** Done: a Data paragraph opens Section 7 and
  Section 8 carries the same scope; a real-data application is named as
  future work.
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
  now H. The display already reads J = -B'W_eB in the reviewed PDF
  (p. 19) and is unchanged. The sentence after it is corrected: J is the
  discrete analogue of div(a grad h), not of its negative, and the net
  flow into each region is -B'W_eBh.
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
- **Fishing data.** Reworded to the exact-likelihood finding only; the
  lead-in that tied it to what simulation smooths over is removed.
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
