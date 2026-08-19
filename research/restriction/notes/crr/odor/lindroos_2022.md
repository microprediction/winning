# Lindroos et al. 2022 — perceptual odor qualities and identification in old age (free then cued)

## Citation
Lindroos R, Raj R, Pierzchajlo S, Hörberg T, Herman P, Challma S, Hummel T, Larsson M, Laukka EJ,
Olofsson JK. "Perceptual odor qualities predict successful odor identification in old age."
*Chemical Senses* 2022;47:bjac025. doi:10.1093/chemse/bjac025. PMC9636890.
(Verified via Europe PMC core record.)

## Domain and stimuli
Olfaction. Sniffin' TOM, 16 odors, SNAC-K older-adult cohort. A separate rating panel supplied
perceptual feature ratings (intensity, pleasantness, edibility, etc.) for the same 16 odors.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**Free-then-cued, same as `horberg_2025.md`**: free identification attempted first, 4 written
alternatives (target + 3 distractors) presented only after a failed free attempt. Unbounded → 4
contraction, within subject, same odors, conditional on free-phase failure.

No variation within the cued phase.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Per-item means are printed; the identification responses are NOT deposited, despite an
open-sounding statement.** Important trap, verbatim:
> "All locally collected data are available together with scripts for analysis and visualization
> through the following link: https://osf.io/nesyq/"

"Locally collected" excludes the SNAC-K identification data. I walked the OSF project via the API:
`nesyq` has **no files at its root**, and its two components are:
- `https://osf.io/jzc8d` — analysis scripts only: `stephens_model3_extended_psycophysicals.R`
  (21,379 B), `statisticsmanuscript.py` (37,940 B), `perceptual_features_weighted_resubmission.pkl`
  (777 B).
- `https://osf.io/fqjpm` — "perceptual data of sniffin' sticks odors": 37 files `Rating_FP1.xlsx` …
  `Rating_FP37.xlsx`, ~6.3 kB each. These are the **odor rating panel**, one file per participant —
  not identification responses.

So the deposit is real and open, but it contains the perceptual predictors, not the choice data.
Figure 1C prints mean identification score per odor (pooled), which is percent-correct only.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9636890/ — fetched, **open access**.
OSF project: https://osf.io/nesyq/ — **open**, public (`public: true` via API), but see above for what
it does and does not contain.
Identification data: **restricted**, SNAC-K committee.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Usable now for odor-level covariates; unusable for choice data.**
The 37 rating files are genuinely open and would let you attach perceptual features (intensity,
pleasantness, familiarity) to any of the 16 SS/TOM odors — useful as regressors in a choice model
fitted on `tolomeo_2026.md` or `nhanes_2014.md`. But there is no choice-share content here.

Recorded chiefly as a **negative with a misleading data statement**, so nobody re-checks it hoping for
identification responses.

## What the authors concluded, quoted verbatim where possible
The title carries the finding: **perceptual odor qualities predict successful odor identification in
old age** — odors that are more intense, more pleasant, or more familiar are identified more reliably,
so item difficulty is partly explained by measurable perceptual properties rather than being arbitrary.
Useful implication for this project: item-level difficulty is structured and predictable, which is what
makes per-descriptor modelling tractable.
