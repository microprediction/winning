# Roper 1978 — diversity and substitutability of adjunctive activities under fixed-interval schedules

## Citation

Roper, T. J. (1978). Diversity and substitutability of adjunctive activities under fixed-interval schedules of food reinforcement. *Journal of the Experimental Analysis of Behavior*, 30(1), 83–96. doi:10.1901/jeab.1978.30-83. PMID 16812091, PMCID PMC1332735.

## Domain and stimuli

**Animal subjects — six rats.** Not humans; and not a concurrent-schedule choice study either. This is a free-operant "adjunctive behaviour" study: rats pressed a lever for 45-mg food pellets on FR 1, FI 30 s, then FI 60 s, with a running wheel, a water bottle, and a block of wood concurrently available in the chamber. Sessions always ended after the 60th pellet, so session duration grew from ~10 min (FR 1) to ~60 min (FI 60 s), progressively freeing time for non-eating activities.

Behaviour categories, scored by direct observation via a manually operated keyboard and described as "for all practical purposes, mutually exclusive and exhaustive" (p. 87): contacting the lever, eating, drinking, chewing the wood block, in the running wheel, grooming, visiting the food tray (other than to collect a pellet), and general activity (rearing, sniffing, undirected movement). Eight categories.

Measure: absolute time in seconds spent in each activity (plus total session duration), on the last day of each condition, per rat.

Roper's Luce connection is explicit but is about *proportional expansion* rather than deletion: "My hypothesis about selection was that, following Luce's Choice Axiom (Luce, 1959), behavior would develop adjunctively under intermittent schedules in proportion to its probability of occurrence in association with unconstrained consummatory behavior" (p. 84).

## Master and restricted response sets

There is a genuine restricted condition, and it is the cleanest kind: same subjects, same schedule, two alternatives physically removed.

- **MASTER menu: "FI 60-sec" condition** — FI 60 s with "the full range of alternative activities still available", i.e. wheel, water spout and wood block all present (p. 86).
- **RESTRICTED menu: "FI 60-sec water only" condition** — "the schedule remained the same as in the third condition, but the entrance to the wheel was closed and the wood block was removed, thereby preventing running and chewing" (p. 86).

So the master and restricted menus share the identical FI 60-s schedule, identical 60-pellet session termination, and near-identical session lengths (per Table 1 the two conditions run ~3,625–4,287 s and ~3,655–3,782 s respectively), with exactly two of the eight activity categories deleted. This is a two-alternative deletion from an eight-alternative menu, over the same six subjects.

The earlier FR 1 → FI 30 s → FI 60 s progression is a *different* manipulation (time budget expansion, not menu restriction) and is where Roper's Luce's-Choice-Axiom prediction of proportional increase applies. Both manipulations are usable but they test different things.

Condition durations: FR 1 and FI 30 s ran 15 sessions each; the two FI 60-s conditions ran 30 sessions each to reach stability.

## What numbers are printed or deposited

Good, and per subject.

**Table 1 (p. 90):** "Absolute time spent in each activity (seconds), and total session duration for each rat on the last day of each experimental condition." Rows are condition × rat (4 conditions × 6 rats = 24 rows); columns are Eat, Lever, Tray, Drink, Wheel, Chew, General Activity, Groom, and Session Length.

The two rows of interest per rat, transcribed for rats 1 and 3:
- Rat 1, FI 60-sec: eat 563, lever 397, tray 50, drink 25, wheel 1193, chew 368, general activity 931, groom 165; session 3690.
- Rat 1, FI 60-sec water only: eat 523, lever 525, tray 67, drink 874, wheel NA, chew NA, general activity 445, groom 770; session 3655.
- Rat 3, FI 60-sec: eat 449, lever 490, tray 109, drink 1181, wheel 0, chew 364, general activity 685, groom 145; session 3664.
- Rat 3, FI 60-sec water only: eat 482, lever 536, tray 261, drink 1299, wheel NA, chew NA, general activity 457, groom 600; session 3714.

Wheel and Chew are marked "NA = Not applicable" in the restricted condition — i.e. genuinely unavailable, not merely zero. Note rat 3 had a wheel score of 0 even when the wheel was available (the text notes one rat avoided the wheel after catching its tail in it), so for that animal the deletion is behaviourally inert.

Also printed / plotted: Figure 1 (lever presses and tray entries per session per rat), Figures 2–3 (wheel turns, wood-block displacements, water volume per rat per session — automatically recorded, session-by-session), Figure 4 (time spent as a percentage of session duration, derived from Table 1), Figure 5 (a direct record for rat 4), Figure 6 (frequency of each activity as a function of time since pellet delivery, rats 1 and 4, three FI conditions).

**Roper does not compute any CRR predictions.** He states the qualitative conclusion (increases were not proportional) but leaves the arithmetic to the reader. Table 1 supplies everything needed to do it.

## Access with a fetched url

- Full text PDF with OCR text layer, fetched successfully: `https://europepmc.org/articles/PMC1332735?pdf=render` (1.64 MB; `pdftotext -layout` yields ~57 kB and renders Table 1 completely and legibly).
- PMCID/PMID resolved via a fetched call to `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids=10.1901/jeab.1978.30-83&format=json&tool=research&email=peter.cotton@gsmc.ai` → PMC1332735 / PMID 16812091.

## Usability verdict

**CRR-TEST-WITH-PRINTED-MATRICES**, in the weaker sense that the numbers are printed but the author never ran the CRR test himself.

Table 1 gives, per rat, absolute seconds in eight mutually exclusive and exhaustive categories with all activities available (FI 60-sec) and in the matched condition with two categories deleted (FI 60-sec water only). Dividing by session length gives a proper share vector in both menus. So one can calibrate on the eight-way full-menu shares and score a six-way restricted-menu prediction against proportional renormalisation, per subject, n = 6. There is also a second, independent full-menu → full-menu test available across the FR 1 / FI 30 / FI 60 progression, which is what Roper's Luce's-Choice-Axiom hypothesis actually addressed.

Caveats: (a) rats, not humans; (b) time allocation over a 60-min session, from direct human observation, not discrete choices — categories like "general activity" are residual and heterogeneous; (c) **only the last day of each condition is tabulated**, so there is one observation per rat per condition and no within-condition variance estimate from Table 1 (the automatically recorded measures in Figures 2–3 do give session-by-session data for wheel/chew/water only); (d) eating time is quasi-fixed by the 60-pellet criterion, which pins one component of the share vector and inflates the apparent stability of the vector; (e) the categories are not commensurable on any single dimension; (f) the removed alternatives (wheel, wood) are motorically and temporally special — Roper's key finding is that different activities occupy different parts of the interreinforcement interval, so substitutability is limited by temporal position, which is precisely a violation of the exchangeability CRR assumes.

## Conclusion about CRR quoted verbatim

On the restriction manipulation specifically (p. 90): "In summary, the observational results show that the additional time, which was made available first by the schedule-related increases in session duration and then by removing the wheel and wood, was distributed widely but unevenly among the various possible activities. Furthermore, the changes in time spent that resulted from these manipulations were not simply proportionate increases in the frequencies of all available activities, according to the amount of extra time available for them."

On the Luce's-Choice-Axiom prediction, which he labels his "second type" of possible outcome (pp. 93–94): "A result of the second type is also specifically predicted by Luce's Choice Axiom (Luce, 1959) ... In fact, the present results were of the third type. In the FI 30-sec and FI 60-sec conditions, the absolute time spent in all observed alternatives to eating (except for drinking, which is discussed below) increased progressively in comparison to the scores in the FR 1 condition, but the increases were not in direct proportion to the frequencies of occurrence of the same activities in the FR 1 condition."

And (p. 94): "the results do not provide any simple quantitative rule for predicting the extent to which particular activities will develop in any one case, using behavior under an FR 1 schedule as a baseline."

From the abstract (p. 83): "the extent to which activities substitute for one another is limited by the tendency for different activities to occupy different parts of the interreinforcement interval."
