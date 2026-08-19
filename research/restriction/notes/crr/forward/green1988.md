# Green & Striefel 1988 — response restriction and substitution with autistic children

## Citation

Green, G. & Striefel, S. (1988). Response restriction and substitution with autistic children. *Journal of the Experimental Analysis of Behavior*, 50(1), 21–32. doi:10.1901/jeab.1988.50-21. PMID 3171473, PMCID PMC1338838.

## Domain and stimuli

**HUMAN subjects — four children with autism (S1–S4).** This is the one candidate in my batch with human subjects. It is *not* a concurrent-schedule pigeon study; it is a free-operant multiresponse time-allocation study, the human analogue of Lyons & Cheney (1984), whose method it explicitly follows ("cf. Lyons & Cheney, 1984", p. 25).

Six programmed responses per child, individually chosen leisure/task activities, e.g. S1: number puzzle, calculator, letters, writing, Distar, book; S2: writing, puzzle, Distar, crayons and picture, book, rocker; S3: puzzle, radio, book, pegboard, clothespins, pencil & paper; S4: pegboard, crayon & paper, letters, rocker, car, book. Plus a non-restrictable residual category "other".

Measure: **mean percentage of intervals** in which each response occurred (interval time-sampling), with 95% confidence intervals.

## Master and restricted response sets

A full nested chain of restricted menus over the same response set, which is unusually good:

- **6-response free-operant baseline = MASTER menu.**
- Then the highest-ranked remaining response was made unavailable, one at a time, giving **restricted menus of 5, then 4, then 3, then 2 programmed responses**, in successive conditions.
- A **return to the 6-response baseline** completed the experiment (so there is a second full-menu observation).

"The response that was restricted was always the programmed response that had the highest rank order in the preceding condition" (p. 28). Each condition's obtained values served as the free-access operant levels for the next: "The obtained values for each condition served as free-access baseline values for comparison with restriction effects in the subsequent condition" (p. 25). So there are four sequential full-menu → restricted-menu prediction problems per child, 16 in all.

The authors tested the constant-ratio rule (proportional redistribution), equal redistribution, the most-probable-remaining-response rule, and a selective-substitution account.

## What numbers are printed or deposited

Very good, and printed per child.

**Table 3 (pp. 26–27), "Obtained and predicted response values in two baseline and four restriction conditions", spanning two pages.** Layout: rows are the six programmed responses plus a parenthesised (Residual) row, grouped by subject S1–S4. Column blocks are the conditions:
- **6 (Baseline)** — a single obtained value per response;
- **5, 4, 3, 2** — for each: **Obtained (95% confidence interval)**, **Predicted Constant ratio**, **Predicted Equal redist.**;
- **6 (Baseline)** again at the end — the return-to-baseline obtained value.
Asterisks mark selective substitution. Restricted responses show a dash.

Sample, S1: baseline (6) = number puzzle 61, calculator 43, letters 3, writing 0, Distar 0, book 0, (residual 17). Five-response condition: calculator 78 (63.6–91.4), CRR predicted 85, equal 53; letters 25 (9.7–40.3)*, CRR 6, equal 13; writing 5 (0–10.7), CRR 0, equal 10; Distar 3 (0–7.4), CRR 0, equal 10; book 1 (0–1.8), CRR 0, equal 10; (residual) 62 (42.3–81.5)*, CRR 33, equal 27. Four-response: letters 87 (78.4–95.6)*, CRR 46, equal 41; etc. Final 6-response baseline: number puzzle 100, calculator 8, letters 3, writing 0, Distar 0, book 0, (residual 17).

Also printed:
- **Table 4 (p. 26):** Pearson correlations between obtained and predicted values, by restriction condition. Constant ratio .69*, .66*, .87*, .74*, overall .73* (n = 72); equal redistribution .66*, .46*, .85*, .78*, overall .67*.
- **Table 5 (p. 27):** Spearman rank-order correlations with sample sizes. Constant ratio .82* (11), .62* (11), .36 (11), −.25 (5), overall .65 (38); equal redistribution .72* (12), .67* (14), .45 (12), 0 (6), overall .66 (44).
- **Table 6 (p. 29):** obtained response values and rank orders in all six conditions, per subject, for the most-probable-alternative analysis (a second printing of the same obtained values, useful for cross-checking).
- Text tallies (p. 25): of 66 comparisons, CRR predictions fell inside the obtained 95% CI in 15 cases, equal redistribution in 10, both in 9, neither in 32. Excluding the residual category (51 comparisons): CRR 12, equal 5, both 10, neither 24.

## Access with a fetched url

- Full text PDF with OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1338838?pdf=render` (1.81 MB; `pdftotext -layout` yields ~63 kB, and pages 6–7 render Table 3 in both halves cleanly).
- PMCID/PMID resolved via a fetched call to `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1988.50-21&format=json&tool=research&email=peter.cotton@gsmc.ai` → PMC1338838 / PMID 3171473.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES**, with one important measurement caveat that has to be handled before use.

The caveat: the measure is *mean percentage of intervals* from partial-interval time sampling, and **the values do not sum to 100** — e.g. S1's five-response column is 78 + 25 + 5 + 3 + 1 = 112 plus residual 62. So these are not choice shares; they are marginal occurrence rates per response, and more than one response can be scored in an interval. To calibrate a Gaussian model on "full-menu shares" one would have to normalise each column to sum to 1, which is a modelling assumption the authors did not make (they computed their CRR predictions from the unnormalised percentages). Normalising is defensible and the arithmetic is easy, but the resulting "shares" are a derived quantity, not a measured one, and 95% CIs would not carry over cleanly.

Subject to that, the data are otherwise ideal: human subjects, per-child numbers, a 6-alternative master menu, four nested restricted menus over the same alternatives, obtained values with confidence intervals, and the authors' own CRR predictions printed alongside for validation. Individual-subject numbers only — no aggregate needed.

Other caveats: (a) four children with autism, so generalisability to a normative human population is limited, and the authors contrast their results with Bernstein & Ebbesen's normal adults; (b) time allocation over extended sessions rather than discrete choices; (c) the residual "other" category can never be restricted and absorbs large amounts of time (up to 100% for S4 in the two-response condition), which distorts any renormalisation that includes it — the authors report the tallies both with and without it; (d) restriction is sequential and cumulative, so later conditions inherit any drift from earlier ones; (e) six of the 66 comparisons had obtained values of exactly 0 or 100% so no CI could be set.

## Conclusion about CRR quoted verbatim

"These results were equivocal with regard to both the constant-ratio and equal-redistribution models. When all responses including the residual other category were considered, values predicted by the constant-ratio rule were within limits of the obtained value in 15 of the 66 cases; values predicted by the equal-redistribution rule were accurate in 10 cases; and both models predicted values within the limits of obtained values in nine cases. Neither model's predicted values fell within the limits of obtained values in the remaining 32 of 66, or almost half, of these comparisons." (p. 25)

On the same page, describing the direction of the failure: "As Table 3 shows, if the definition of selective substitution is expanded to include disproportionate increases in either one or two responses, substitution by these subjects was selective in most cases. By this analysis, selective substitution (indicated by asterisks in Table 3) occurred in 13 of 16 cases" (p. 25).

From the abstract (p. 21): "Results were compared to predictions made by four time-reallocation models. These results were described accurately only by the selective substitution model."
