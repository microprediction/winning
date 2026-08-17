# Davison & McCarthy 1994 — discriminability of alternatives in three-alternative concurrent schedules

## Citation

Davison, M. & McCarthy, D. (1994). Effects of the discriminability of alternatives in three-alternative concurrent-schedule performance. *Journal of the Experimental Analysis of Behavior*, 61(1), 45–63. doi:10.1901/jeab.1994.61-45. PMID 16812725, PMCID PMC1334353.

## Domain and stimuli

**Animal subjects on concurrent schedules — six pigeons (birds 101–106).** This is NOT a human confusion-matrix dataset. Choice alternatives are signalled by monochromatic light on a centre pecking key: 560 nm, 630 nm, and a third alternative at 600 nm (Part 2) or 623 nm (Part 3). A switching-key procedure was used: pecks on a white side key produced a 3-s blackout and, intermittently, reselection of which VI schedule/colour was programmed on the main key. The 3-s blackout is a changeover punisher and the authors ultimately attribute the CRR failure to it plus imperfect stimulus discriminability. "Choice" here is response allocation over sessions, not trial-by-trial identification, so the analogy to a psychophysical confusion matrix is loose: the alternatives are reinforcement schedules signalled by stimuli, and the manipulated variable is relative reinforcer rate, not stimulus similarity alone (though wavelength separation *is* also manipulated: 560/600/630 vs 560/623/630).

## Master and restricted response sets

This is the design the project wants, and it runs in both directions.

- **Part 1 (Conditions 1–5): RESTRICTED menu of 2 alternatives** — only 560 nm and 630 nm available ("X not available"). Arranged relative reinforcer frequencies for 560:630 were .5:.5, .9:.1, .2:.8, .8:.2, .1:.9. Overall p(R/s) = .022.
- **Part 2 (Conditions 6–18): MASTER menu of 3 alternatives** — 560 nm, 630 nm, and 600 nm. Overall p(R/s) = .022–.028.
- **Part 3 (Conditions 20–31): MASTER menu of 3 alternatives** — 560 nm, 630 nm, and 623 nm (Condition 19 used 620 nm and is excluded by the authors). Overall p(R/s) = .028.

Parts 2 and 3 were designed so that "the same range of reinforcer ratios for each pairwise selection of schedules" was covered, and by accident twice as many conditions as needed were run, so **each pairwise reinforcer ratio is replicated at two different reinforcer rates on the third alternative** (their example: Conditions 8 and 15 both arrange a 1:4 ratio on 560/630 but differ in the rate on 600 nm). Crucially, the arranged 560:630 ratios in the 3-alternative parts (1:1, 4:1, 1:4, 9:1, 1:9 — Conditions 6/7/10/13/16 are 1:1; 8,15 and 12,17 are 1:4 / 1:9; 9,14 and 11,18 are 4:1 / 9:1) **match the arranged 560:630 ratios of the 2-alternative Part 1**. So both a full-menu vector and a restricted-menu vector over the same two stimuli exist at matched reinforcer ratios.

Two separate CRR tests are therefore available:
1. **Across-menu (what the project needs):** predict the Part 1 two-alternative 560:630 share from the Part 2/3 three-alternative share vector by proportional renormalisation.
2. **Within-menu (what the authors actually did):** hold the pair's reinforcer ratio fixed and vary the third alternative's rate ("low-other" vs "high-other").

## What numbers are printed or deposited

Very good. Printed in the article:

- **A four-page APPENDIX (pp. 60–63) giving, for every one of the 6 birds × 31 conditions: number of responses emitted on 560 nm / 630 nm / X nm, seconds spent responding on each, reinforcers obtained on each, changeovers from each, and number of sessions.** Data are summed over the last five sessions of each condition. Example first rows: bird 101 Condition 1 (2-alternative) = 2,114 / 2,967 / 0 responses; bird 101 Condition 6 (3-alternative) = 3,546 / 4,775 / 1,524 responses. Zeros in the X column mark the Part 1 restricted conditions. This is raw counts, per subject, per condition — i.e. everything needed to build both menus' share vectors and to score any out-of-sample prediction.
- **Table 1 (p. 46):** condition sequence, number of sessions, overall p(R/s), and arranged relative reinforcer frequency for 560 nm, 630 nm and X nm in every condition.
- **Tables 2 and 3 (pp. 52–53):** per-bird generalized-matching slopes (sensitivity a), standard errors, log c intercepts and %VAC for each pairwise combination of schedules in Parts 2 and 3, split by whether the third alternative's arranged rate was low ("lo") or high ("hi") relative to the sum of the pair's rates. E.g. Part 3, bird 101, third alternative 623 nm: lo slope 1.17 (0.10), constant 0.49, 97 %VAC; hi slope 0.63 (0.11), constant −0.31, 89 %VAC.
- **Table 4** and Figures 1–6 (fits of the extended Davison & Jenkins 1985 contingency-discriminability model; group means).

Group summary numbers stated in text: Part 1 mean sensitivity 1.30 (range 1.08 for bird 104 to 1.55 for bird 101); Part 2 low-other mean 1.20; Part 2 high-other mean 0.59; all 6 birds showed lower sensitivity in the high-other Part 2 conditions than in Part 1 (sign test p < .05).

Time-allocation data are also in the Appendix but the authors decline to analyse them because the random-reselection procedure makes them non-comparable to standard concurrent schedules.

## Access with a fetched url

- Full text PDF with an OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1334353?pdf=render` (2.8 MB; `pdftotext -layout` yields ~98 kB of text including the whole Appendix).
- PMCID/PMID resolution fetched from `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1994.61-45&format=json&tool=research&email=peter.cotton@gsmc.ai` → PMC1334353 / PMID 16812725.
- Note: `https://pmc.ncbi.nlm.nih.gov/articles/PMC1334353/pdf/` returns a proof-of-work challenge page rather than the PDF; and `efetch db=pmc` returns front matter only (the article is a scan, no JATS body). EuropePMC is the working route.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES.** The printed numbers are more than sufficient. The Appendix gives per-subject raw response counts on all available alternatives for both the 2-alternative restricted menu (Conditions 1–5) and the 3-alternative master menus (Conditions 6–18, 20–31) over the *same* 560-nm and 630-nm stimuli, at matched arranged reinforcer ratios. One can calibrate a parameter-free Gaussian model on the 3-way shares and score the out-of-sample 2-way prediction against proportional renormalisation, per bird and in aggregate. Individual-subject numbers are available (bonus satisfied).

Caveats to record: (a) pigeons, not humans; (b) overall reinforcer probability differs slightly between Part 1 (.022) and Parts 2–3 (mostly .028), so the menus are not perfectly matched on total reinforcement; (c) the changeover blackout is a punisher and the authors argue it is *the reason* CRR fails here, i.e. the violation may be procedural rather than a property of choice; (d) obtained reinforcer ratios differ from arranged ones, so the honest matching is on obtained reinforcers (also printed); (e) within-menu third-alternative-rate effects (their "low-other"/"high-other" split) are a *different* CRR test from menu deletion and should not be conflated with it.

## Conclusion about CRR quoted verbatim

"The present results have clearly shown that the principle of indifference from irrelevant alternatives, or the constant-ratio rule, is not applicable to three-alternative choice when there is a timeout contingent on changing over between the choices. Rather than arguing against the principle per se, the present results can simply be interpreted as showing that reinforcer rates for the third alternative are not irrelevant when punishers (i.e., the timeouts) are present" (p. 55, section headed "Indifference from Irrelevant Alternatives").

Also, on the Part 1 two-alternative data (p. 47): "These results, which indicate a failure of the constant-ratio rule, were unexpected and clearly deserved further analysis."

And from the abstract (p. 45): "In Parts 2 and 3, generalized matching sensitivities between pairs of alternatives were found to be higher when the reinforcer rate on the third alternative was low than when it was high—an apparent failure of the constant-ratio rule."
