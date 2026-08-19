# Lyons & Cheney 1984 — time reallocation under restriction of response classes

## Citation

Lyons, C. A. & Cheney, C. D. (1984). Time reallocation in a multiresponse environment: Effects of restricting response classes. *Journal of the Experimental Analysis of Behavior*, 41(3), 279–289. doi:10.1901/jeab.1984.41-279. PMID 16812370, PMCID PMC1348059.

## Domain and stimuli

**Animal subjects — four experimentally naive male albino rats (W, X, Y, Z), 6–9 months old.** Not humans, and not concurrent VI schedules either: this is a *closed behavioural system* / free-operant time-allocation study, which is a different beast from a concurrent-schedule choice experiment. Each rat lived 3 hours daily in an apparatus with six separate location-defined compartments, so the alternatives are activities, and "choice" is percentage of session time.

The six response classes: (1) turn off room lighting → darkness; (2) view a female rat through a glass panel; (3) turn off white noise ("escape from noise"); (4) drink; (5) eat; (0) "other" = time in the hallway between compartments. Food and water were available *only* in the apparatus (closed system), so response levels cannot be measured in isolation.

Measure: percentage of total session time spent in each compartment. Six response classes, but they are exhaustive and mutually exclusive, so the baseline column is a genuine share vector summing to ~100%.

## Master and restricted response sets

Exactly the manipulation the project wants. A-B-A-C-A design per subject:

- **A = MASTER menu of 6 responses** (full free-access baseline).
- **B / C = RESTRICTED menus of 5 responses**, produced by making one response unavailable:
  - restriction of the **low-probability** response (darkness, compartment 1);
  - restriction of the **high-probability** response (escape from noise, compartment 3).
  The order of B and C was counterbalanced across subjects, and each restriction was preceded by its own return-to-baseline, so each restriction has its own immediately preceding 6-response baseline.

So there are 8 restriction phases in all (4 rats × 2 restrictions), each with a matched full-menu baseline over the same response set.

The authors explicitly tested four competing reallocation rules: the constant-ratio rule ("time made available by response restriction will be redistributed among independent response classes in direct proportion to their baseline levels", p. 280, attributed to Luce 1959), equal time redistribution, the most-probable-alternative rule, and the sequential-dependency rule.

## What numbers are printed or deposited

Excellent — the CRR comparison is printed in full, per subject, with the CRR prediction already computed.

**Table 1 (pp. 287–288)**, in two panels:
- **Panel (A) "Predicted/Obtained Response Values Following Restriction of Low-Probability Response"** and **Panel (B) "...High-Probability Response"**.
- Columns: response category (1 darkness, 2 view, 3 noise off, 4 water, 5 food, 0 other); **Value Prior To Restriction** (the full-menu baseline share); **Obtained Post-Restriction with limits of the 95% confidence interval**; **Predicted Post-Restriction, Constant-Ratio Rule**; **Predicted Post-Restriction, Equal Redistribution**. Asterisks mark predicted values falling inside the obtained 95% CI.
- Rows for all four subjects in both panels. The restricted response's row is blank in the post-restriction columns.

Sample rows, Panel A (restriction of darkness), Subject W: baseline 12.9 / 10.6 / 47.3 / 8.2 / 13.9 / 7.1 (sums to 100.0); obtained post-restriction 10.3 (7.6–13.0), 61.1 (57.8–64.4), 8.0 (7.4–8.6), 11.3 (9.7–12.9), 9.1 (6.9–11.3); CRR predictions 12.2*, 54.3, 9.4, 16.0, 8.2*.
Panel B (restriction of noise-off), Subject Y: baseline 7.7 / 4.0 / 64.1 / 7.0 / 11.2 / 6.0; obtained 46.7 (42.3–51.1), 6.0 (4.6–7.4), 14.1 (13.0–15.2), 12.8 (11.1–14.5), 20.4 (17.5–23.3); CRR predictions 21.4, 11.1, 19.5, 31.2, 16.7*. (Note the huge CRR miss on darkness: 46.7 obtained vs 21.4 predicted.)

**Table 2 (p. 288):** full 5×5 sequential-transition probability matrices per subject (rows = preceding response, columns = following response), giving two values per cell — before the first restriction and before the second. Not needed for CRR but a genuine extra matrix.

Figures 2 and 3 give mean response levels for all categories in each experimental condition, and per-third-of-condition averages are discussed in text.

Note: Panel A and Panel B have *different* baselines for the same rat (e.g. Subject W darkness 12.9 in A but 5.5 in B) because each restriction was scored against its own immediately preceding baseline. Both baselines sum to ~100, so both are usable share vectors.

## Access with a fetched url

- Full text PDF with OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1348059?pdf=render` (1.37 MB; `pdftotext -layout` yields ~49 kB including both panels of Table 1 and all of Table 2).
- PMCID/PMID resolved via a fetched call to `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1984.41-279&format=json&tool=research&email=peter.cotton@gsmc.ai` → PMC1348059 / PMID 16812370.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES.** The printed numbers suffice completely: full-menu baseline share vector over 6 mutually exclusive, exhaustive response classes; restricted-menu obtained shares with 95% CIs; and the authors' own CRR predictions to check one's arithmetic against. Per-subject (bonus satisfied), 4 subjects × 2 restrictions = 8 full-menu → restricted-menu prediction problems.

Caveats to record: (a) rats, not humans; (b) the "choice" measure is percentage of session time in a closed 3-hour environment, i.e. temporally extended free-operant behaviour rather than discrete forced-choice trials, so a Thurstonian latent-utility reading is a stretch — the authors themselves warn that "the form of time reallocation is dependent on the types of response classes examined"; (c) the response classes span different sensory modalities and functions (darkness, conspecific viewing, noise escape, water, food), so they are not commensurable stimuli on one dimension; (d) confidence intervals are session-to-session, and the authors note "large intersession variability in response durations"; (e) the residual "other" category (hallway time) is not restrictable and behaves as a sink.

Note also that the direction of the CRR failure is informative: after restricting the *high*-probability response, obtained darkness time far exceeded the CRR prediction in 3 of 4 rats (W 41.3 vs 12.3; Y 46.7 vs 21.4), i.e. massive concentration on one substitute rather than proportional spreading — the same qualitative signature as "selective substitution" in Green & Striefel (1988).

## Conclusion about CRR quoted verbatim

"Table 1 shows the obtained values, and the values predicted by various rules, following the restriction of the low-probability (A) and high-probability (B) responses for each subject. Within this behavioral system (providing these response opportunities), the data offer no support for the constant-ratio rule. Clearly the rats in this study did not keep relative baseline proportions constant after simple restriction of one response. Perhaps a constant-ratio redistribution requires more extended conditions (although nothing in the present data indicates that this is likely). This finding further indicates that Rachlin and Burkhard may be in error in assessing reinforcement and substitution against a constant 1/N ratio, and that Bernstein and Staddon may be correct in not employing the constant-ratio rule in their analyses." (Results/Discussion; the sentence runs across a column break in the scanned two-column layout, from p. 283 to p. 286.)

From the abstract (p. 279): "Four predictive rules concerning the redistribution of behavior after response restriction were tested, including the constant-ratio rule, equal time redistribution, the most probable alternative, and the sequential-dependency rule. The results indicate no support for any of the four predictive rules and suggest that empirical assessment of restriction effects is necessary in reinforcement studies involving temporally extended responses."

And from the conclusion (p. 289): "In conclusion, the present data provide no support for current models seeking to predict restriction effects from a knowledge of baseline response hierarchies."
