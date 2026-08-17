# Davison & Hunter 1976 — VI schedules arranged singly and concurrently (three, two and one keys)

## Citation

Davison, M. C. & Hunter, I. W. (1976). Performance on variable-interval schedules arranged singly and concurrently. *Journal of the Experimental Analysis of Behavior*, 25(3), 335–345. doi:10.1901/jeab.1976.25-335. PMID 16811917, PMCID PMC1333472.

Found under assignment item 10 (search for other studies that manipulated the number of concurrently available alternatives). It is the paper Davison & McCarthy (1994) cite as the prior evidence *for* the constant-ratio rule that their own results contradicted ("quite contrary to the extensive previous research done on preference among more than two alternatives (Davison & Hunter, 1976; H. Miller & Loveland, 1974; Pliskoff & Brown, 1976)", 1994, p. 51).

## Domain and stimuli

**Animal subjects on concurrent schedules — six homing pigeons (141–146; 143 died and was replaced by 143b).** Not a human confusion-matrix dataset. Three response keys, distinguished by position, each carrying a VI schedule. Changeover delay of 1.5 s between keys. Measures: number of pecks on the three keys, time spent pecking each key, and reinforcers obtained on each key.

## Master and restricted response sets

This is the design assignment item 10 was looking for, with one important qualification about *how* alternatives were removed.

From the abstract (p. 335): "Extensive parametric data were obtained from pigeons responding on variable-interval schedules arranged on three, two, and one response keys." Five sets of conditions were run, each varying the schedule on one key while the other two were held constant:

- **3-alternative sets:** conc VI x / VI 120 s / VI 60 s (and permutations of which key is varied).
- **2-alternative sets:** e.g. keys 2 and 3 at VI 120 s and **extinction** respectively, with key 1 varied; or keys 1 and 2 at extinction and VI 120 s, with key 3 varied.
- **1-alternative set:** both keys 2 and 3 under **extinction**, key 1 varied.

**Qualification: alternatives were removed by scheduling extinction on them, not by darkening or physically removing the key.** So the "restricted" menus still have all three keys pecked-at-able; only their reinforcement is zeroed. Responses on extinction keys were recorded but "Figure 1 does not show performance on keys on which extinction was arranged throughout a set of conditions" (p. 339). For a Luce/CRR deletion test this matters: an extinction key is a surviving alternative with zero payoff, not a deleted one, and the birds do allocate some behaviour to it. Whether that counts as a restricted menu is a judgement call the project must make explicitly.

The IIA/CRR test the authors performed has two parts (p. 342):
1. Vary the rate on key 1 and check that the response *ratio* between keys 2 and 3 (both held constant) does not change — Figure 3.
2. Check that the function relating log response ratio to log obtained reinforcer ratio for a pair is the same whether or not reinforcers are available on a third key — Figure 2.

Both are the within-full-menu ratio-invariance form of the test, not deletion-and-renormalisation, but part 2 does compare a 2-key set against a 3-key set for the same pair, which is close to what the project wants.

## What numbers are printed or deposited

**This is the fatal problem: individual data are not printed.** A footnote on p. 335 reads: "Reprints and tables of individual data are available from the authors, Department of Psychology, University of Auckland, Private Bag, Auckland, New Zealand." That offer is 50 years stale; Davison is emeritus and Hunter's whereabouts are unknown to me.

What is printed:
- **Table 1 (p. 337 area):** the sequence of conditions with schedule values and, per the surrounding text, group response-rate data. The OCR of this table in the scanned PDF is **badly garbled** — the layout collapses into fragments of digits with no recoverable column structure. It would need re-OCR or a look at the page images to transcribe.
- **Table 3 (p. 343):** obtained values of Herrnstein's k and R₀, values of k′ and a for the authors' power-function analysis (Equation 4b), and the sensitivity exponent a from the preference analysis (Equation 2a), for each of the five sets of conditions. Varied schedules denoted X, extinction conditions E. Group-level parameters, not shares.
- **Figures 1–3:** response rates on each key as a function of reinforcer rate on the varied key (Figure 1, five sets); log response ratio vs log obtained reinforcer ratio with fitted lines and printed equations (Figure 2); and the key result for CRR — log response ratio on the two constant-schedule keys as a function of reinforcer rate on the varied third key (Figure 3), showing no trend.
- Text summaries: "Averaging all the data obtained here, response functions have a slope of about 0.69 and time functions a slope of about 0.81" (p. 342) — i.e. strong undermatching in both measures.

No response-count matrix, per subject or aggregate, appears anywhere in the article.

## Access with a fetched url

- Full text PDF with OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1333472?pdf=render` (`pdftotext -layout` yields ~59 kB of text, but Table 1 does not survive the extraction).
- PMCID/PMID resolved via a fetched call to `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1976.25-335&format=json&tool=r&email=peter.cotton@gsmc.ai` → PMC1333472 / PMID 16811917.

## Usability verdict

**CRR-TEST-BUT-NUMBERS-NOT-PRINTED.** The design is right — the same three keys run as three-, two- and one-alternative arrangements over the same six birds, with an explicit test of Luce's principle of indifference from irrelevant alternatives — but the paper reports group figures and fitted parameters only, and refers individual data to a 1976 postal request. Table 1's group numbers might be salvageable from the PMC page images with better OCR, but there is no printed share matrix in either menu, so no full-menu-calibrated prediction can be scored from the article as published.

Additional caveats even if the numbers were recovered: (a) pigeons; (b) alternatives are "removed" by extinction rather than deletion, so the restricted menu is not truly restricted; (c) strong undermatching (slope ≈ 0.69) means no fixed-scale Luce model fits the reinforcer-rate dependence, even though menu invariance itself holds; (d) the authors' claim of support for Luce is a null result read off figures with no formal test.

Worth recording as the strongest *positive* prior evidence for CRR in multiple-alternative concurrent schedules, and as the paper Davison & McCarthy (1994) had to contradict — a second instance of the same procedure-dependence that separates Elliffe & Davison (2010) from Bensemann et al. (2015).

## Conclusion about CRR quoted verbatim

"Catania (1966) suggested that Luce's (1959) principle of indifference from irrelevant alternatives applies to concurrent VI schedule performance. This principle would predict that changing the rate of reinforcement on one key would not change the response ratios between two other, constant reinforcement rate, keys. Figure 3 shows that this prediction is correct for the present data. The principle would also predict that the function relating response to reinforcement ratios would be the same whether or not reinforcements were available on a third key, and this effect is shown in Figure 2. The present data, then, give strong support to Luce's principle. On the other hand, these data do not support the notion of matching (Equation 1) between response and reinforcement ratios (Herrnstein, 1970), nor matching between time allocation and reinforcement ratios (Baum, 1975). The data are characterized by undermatching in both measures" (p. 342, section headed "Preference").

From the abstract (p. 335): "In terms of preference, both response and time-allocation ratios undermatched ratios of obtained reinforcements, and the degree of undermatching was consistent both within, and between, two- and three-schedule data."
