# Mitchell et al. 2025 — PPMI UPSIT, 40 items with the chosen alternative recorded

## Citation
Mitchell E, Mattjie C, Bestwick JP, Barros RC, Schuh AF, Simonet C, Noyce AJ. "Hyposmia in
Parkinson's disease; exploring selective odour loss." *npj Parkinson's Disease* 2025;11(1):67.
doi:10.1038/s41531-025-00922-3. PMC11971265. (Verified via Europe PMC core record.)

## Domain and stimuli
Olfaction, cued identification. University of Pennsylvania Smell Identification Test (UPSIT),
**40 items, 4-AFC**, microencapsulated scratch-and-sniff booklet (Sensonics). Data from the
Parkinson's Progression Markers Initiative (PPMI) — both PPMI Clinical and the much larger PPMI
Remote screening cohort. Filed under this paper because it is the citable evidence that PPMI stores
the chosen alternative; the data itself is PPMI's.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**No restriction — fixed 4-alternative list per item, always.** UPSIT is invariably 4-AFC; no
published study varies it.

The exploitable structure is **cross-instrument, a single-element swap** rather than nesting. UPSIT and
the NHANES Pocket Smell Test (`nhanes_2014.md`) are both Sensonics microcapsule tests, and on three
odors the menus differ by exactly one substituted distractor:

| Odor | UPSIT menu | NHANES PST menu | Difference |
|---|---|---|---|
| Chocolate | {Lemon, Chocolate, Garlic, Black pepper} | {Lemon, Chocolate, Smoke, Black pepper} | Garlic → Smoke |
| Smoke | {Dill pickle, Grass, Smoke, Peach} | {Garlic, Grass, Smoke, Peach} | Dill pickle → Garlic |
| Natural Gas | {Orange, Wintergreen gum, Cola, Natural Gas} | {Orange, Cinnamon, Cola, Natural Gas} | Wintergreen gum → Cinnamon |

Strawberry, Leather, Soap, Grape and Onion share only 1–2 options and are not usable this way. A
one-element swap is arguably a *sharper* IIA test than nesting, and both sides have thousands of
subjects. Confounds that would sink a naive comparison: different populations (PD-enriched vs US
national sample), different years, and you must confirm with Sensonics that the odorant formulations
are identical across the two products.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Per-subject, 40 items, chosen alternative recorded — confirmed at schema level, not merely inferred.**

The public analysis repository enumerates the PPMI column names:
`SCENT_01_CORRECT`, `SCENT_01_RESPONSE`, … `SCENT_40_CORRECT`, `SCENT_40_RESPONSE`
(in `train_functions.py`), and its README states experiments are run once with correct-only features
and once with "all response features (**which of the four smell alternatives was guessed**)".

PPMI files to pull: `University_of_Pennsylvania_Smell_Identification_Test_UPSIT` (PPMI Clinical),
plus `Screening_UPSIT_Screening` and `University_of_Pennsylvania_Smell_Identification_Test`
(PPMI Remote).

**The answer key is public but is the authors' reconstruction, not Sensonics':**
`data/scents/scent_correct_names.csv` and `data/scents/scent_response_clean.csv` in the repo give the
revised-2020 UPSIT's full 40 × 4 menu in code order. Sample: 01 Pizza {Honey, Pizza, Orange, Bubble
gum}; 19 Chocolate {Lemon, Chocolate, Garlic, Black pepper}; 33 Smoke {Dill pickle, Grass, Smoke,
Peach}; 38 Natural Gas {Orange, Wintergreen gum, Cola, Natural Gas}. **Validate the 1–4 ordering
against a physical booklet before relying on it.**

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11971265/ — **open access**.
Code and answer key: https://github.com/cmattjie/UPSIT-PD-Hyposmia — **open**, public repo.
Data: **gated but free** — DUA application, reviewed in roughly a week.
https://www.ppmi-info.org/access-data-specimens/download-data
https://ida.loni.usc.edu/collaboration/access/appApply.jsp?project=PPMI

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Needs an application (~1 week), then usable — and it is the largest chosen-descriptor UPSIT corpus
in existence.** 40 items per subject across two cohorts, one of them a very large remote screening
sample. Nothing to digitize; the data is machine-readable once the DUA clears.

Priority: apply now in parallel with everything else, because the lead time is the only cost. The
fixed-menu caveat applies — this anchors 40-item choice shares but tests no menu effect on its own,
except via the NHANES single-swap route above.

## What the authors concluded, quoted verbatim where possible
The paper's finding is that olfactory loss in Parkinson's is **selective rather than uniform** — some
odours are disproportionately affected — and, importantly for this project, that *which* wrong answer
is chosen carries signal. The authors explicitly analyse "which incorrect answer was selected" and
find that response-level features add predictive information beyond correct/incorrect scoring. That
is direct precedent for treating UPSIT distractor choice as a choice-model object rather than
measurement noise.
