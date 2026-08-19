# Jacob & Fantino 1988 — effects of reinforcement context on choice (Experiment 2: adding a third key)

## Citation

Jacob, T. C. & Fantino, E. (1988). Effects of reinforcement context on choice. *Journal of the Experimental Analysis of Behavior*, 49(3), 367–381. doi:10.1901/jeab.1988.49-367. PMID 3385353, PMCID PMC1338797.

## Domain and stimuli

**Animal subjects on concurrent schedules — pigeons.** Experiment 2 used four White Carneau pigeons (Y28, Y29, B21, R80; R80 joined late, Y28 was temporarily removed when ill). Not a human confusion-matrix dataset. The alternatives are response keys distinguished by position and colour: two white side keys carrying concurrent VI schedules of food, and (in the +CHN conditions) a centre key carrying a chain schedule (VI 30 s initial link, VI 45 s terminal link, constant in all conditions).

An important structural oddity: the third alternative is not a peer of the other two. "When the terminal link of the chain schedule was produced, the side keys became inoperative" (abstract, p. 367). So the third option, once chosen through to its terminal link, temporarily *removes* the other two, rather than sitting alongside them. It is a "successive reinforcement context" alternative, not a symmetric third choice.

Experiment 1 is not a menu manipulation (two alternatives throughout, with concurrent-chains terminal links varied) and is irrelevant here.

## Master and restricted response sets

Experiment 2 is a clean two-key ↔ three-key manipulation over the same two concurrent VI schedules:

- **RESTRICTED menu of 2 alternatives:** baseline conc VI VI on the two side keys, centre key dark and inoperative.
- **MASTER menu of 3 alternatives:** the same conc VI VI on the side keys **plus** the chain schedule on the centre key ("+CHN").

Four reinforcer-rate conditions, each run in both left/right position assignments and both with and without the chain key, giving 16 phases:
1. conc VI 15, VI 30 (and reversal), ± CHN
2. conc VI 30, VI 60 (and reversal), ± CHN
3. conc VI 120, VI 240 (and reversal), ± CHN
4. conc VI 180, VI 360 (and reversal), ± CHN

20 sessions per phase; order of determination is printed. The chain schedule's own values are held constant throughout, so the third alternative's "value" is fixed while the pair's rates vary — the reverse of the Davison & McCarthy (1994) manipulation.

The authors frame this squarely as an IIA/CRR test: "The principle of independence of irrelevant alternatives, also called the constant-ratio rule, states that preference between two choices, for example, should be unaffected by the addition and variation of a third, irrelevant alternative" (p. 368).

## What numbers are printed or deposited

Partly printed, and the part that is printed is exactly the CRR-relevant conditional.

**Table 3 (p. 376):** "Conditions, numbers of sessions, numbers of reinforcements per session (Rfct), order (within parentheses) in which each determination of preference was conducted, and **choice proportions (preference for the VI with the higher reinforcement rate of the concurrent pair) for every subject** in Experiment 2." Rows are the 16 phases (4 conditions × {normal, reversed} × {baseline, +CHN}); columns are the four subjects Y28, Y29, B21, R80, plus reinforcers per session.

Sample rows (Condition 1): conc VI 15, VI 30 → .73 / .63 / .57 / .80; conc VI 30, VI 15 → .60 / .58 / .60 / .49; conc VI 15, VI 30 +CHN → .66 / .62 / .69 / .54; conc VI 30, VI 15 +CHN → .60 / .72 / .62 / .77. Condition 4: conc VI 180, VI 360 → .59 / .59 / .59 / .68; +CHN → .62 / .42 / .73 / .50.

**Crucially, the printed proportion is already the CRR-relevant renormalised conditional** — p(HIVI) / [p(HIVI) + p(LOVI)], with the chain key's responses excluded. So the CRR/IIA prediction "this conditional is invariant to the presence of the third key" can be scored directly from Table 3, per subject, in 8 matched baseline-vs-+CHN pairs.

**What is NOT printed:** the three-way share vector. The proportion of responses allocated to the chain key in the +CHN conditions appears only in **Figure 4 (bottom panel)**, as bar graphs of response proportions and reinforcer proportions averaged over subjects, and in Figure 5 as absolute response rates on the HIVI key and the chain initial link (group means). So the full-menu vector (p_HI, p_LO, p_CHN) would have to be digitised off Figure 4, aggregate-only, to calibrate a three-alternative model.

Also printed: Table 1 (Experiment 1 condition sequence), Table 2 (Experiment 1 absolute response rates). Figure 3 shows the Experiment 2 choice proportions averaged over initial and reversal determinations, plus response proportion vs obtained reinforcer proportion.

## Access with a fetched url

- Full text PDF with OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1338797?pdf=render` (2.29 MB; `pdftotext -layout` yields ~80 kB and renders Table 3 legibly).
- PMCID/PMID resolved via a fetched call to `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1988.49-367&format=json&tool=research&email=peter.cotton@gsmc.ai` → PMC1338797 / PMID 3385353.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES**, but only for the two-alternative conditional — the three-way full-menu shares are figure-only.

Concretely: if the question is "does the surviving pair's ratio survive adding/removing a third alternative?", Table 3 answers it with per-subject printed numbers in both menus (8 matched pairs × 4 birds). If the question is the project's stricter one — calibrate a parameter-free Gaussian model on the **three-way** master shares and predict the two-way restricted shares out of sample — then Table 3 is insufficient on its own, because p(chain key) is not tabulated. It would have to be read off Figure 4's bars, and those are group averages only. So: usable as a corroborating CRR test, not as a clean full-menu-calibration dataset.

Caveats: (a) pigeons; (b) the third alternative is a chain schedule whose terminal link makes the other two keys inoperative, so it is not an ordinary third option and the "menu" is not a simultaneous three-way choice for the whole session; (c) reinforcers per session differ between baseline (50, 50, 45b, 30b) and +CHN (45, 45, 35, 30) phases, so total reinforcement is not matched across menus; (d) the measure is relative response rate over the last five sessions, i.e. asymptotic allocation, not trial-level choice; (e) undermatching was present throughout (the log-ratio slope is well under 1), so no version of Luce's model with a fixed scale will fit the reinforcer-rate dependence — the CRR test here is only about *invariance across menus*, holding reinforcer rates fixed; (f) this is a null result, so it constrains rather than demonstrates a violation.

## Conclusion about CRR quoted verbatim

"In Experiment 2, when a type of successive-reinforcement context was provided for choice between a pair of concurrent VI schedules, preference was unaffected. This result is consistent with Lobb and Davison's (1977) conclusion that choice in concurrent VI VI schedules is unaffected by a successive-reinforcement context. These results are also consistent with the constant-ratio rule (Luce, 1959, 1977)." (p. 378)

From the General Discussion (pp. 378–379): "when responses on the concurrent VI schedules were independent of those producing terminal-link outcomes, choice measures were unaffected by the presence or absence of the contextual reinforcement. Thus, these latter results provide additional confirmation of the constant-ratio rule with concurrent schedules (reviewed by Fantino & Dunn, 1983)."

From the Results (p. 376): "As can be seen in Figure 3, there was no consistent change in choice proportions when the chain schedule was added as a context for choice relative to the choice proportions obtained during baseline." And: "the degree of undermatching was the same whether or not the third key (chain schedule) was available."

From the abstract (p. 367): "Availability of the chain schedule did not affect choice between the concurrent schedules."
