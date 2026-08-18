# Sniffin' Sticks Extended Test sum-score validity (OSF djpae/pf4x2) — subtest totals only

## Citation
"Psychometric validity of the sum score of the Sniffin' Sticks-Extended Test." OSF project
https://osf.io/djpae with data component https://osf.io/pf4x2. Italian sample, n = 988.
**Author list and the corresponding article were not identified** — the OSF project carries no
contributor metadata that was retrieved, and no matching publication was pinned down. Filed by
instrument for findability.

## Domain and stimuli
Olfaction. Sniffin' Sticks Extended Test (SSET) — threshold, discrimination and identification subtests.
n = 988.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**Indeterminable from the deposit, and moot** — no item-level data is present, so the response sets
cannot be examined at all. The identification subtest is 4-AFC by construction.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Per-subject but summary only — subtest totals, no items.** I enumerated the files via the OSF API:
- `Dataset SST tau equivalence_988subjects.csv` (28,322 B) — downloaded and inspected. Columns are
  `DATASET, AGE, SEX, OT, OD, OI, TDI`: dataset label, age, sex, and the four **subtest totals**
  (odor threshold, discrimination, identification, and their sum). 988 rows.
- `OSFScript SST tau equivalence_988subjects.R` (10,635 B)
- `Codebook.pdf` (42,615 B)
- a `Tutorial/` folder

No per-item columns, therefore no chosen descriptors.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
https://osf.io/djpae — **open** (`public: true`), but the root holds no files; `pf4x2` is its only child.
https://osf.io/download/3m5pn/ — fetched, HTTP 200, **open**, the CSV above.

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable.** Nothing below subtest-total granularity. Recorded as definitively checked by download and
inspection — a 988-subject open Sniffin' Sticks deposit looks promising in a repository search and is not.

## What the authors concluded, quoted verbatim where possible
Not available — the corresponding publication was not identified, and the OSF project has no description
text. The R script's filename ("tau equivalence") indicates the analysis concerned whether the four
subtests are tau-equivalent indicators justifying a simple sum score (the TDI).
