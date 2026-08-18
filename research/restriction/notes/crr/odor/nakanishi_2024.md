# Nakanishi et al. — MultiScent-20 digital 4-AFC, deposited as binary only

## Citation
Nakanishi M, de Paula Brandão PR, et al. "Development and validation of the MultiScent-20 digital odour
identification test using item response theory." 2024. PMC11219931. Data deposit:
doi:10.5281/zenodo.8079860.

## Domain and stimuli
Olfaction, digital odor identification. MultiScent-20: 20 odorants, 4-AFC ("one correct answer and three
distractors"), implemented digitally.

## Master response set and restricted response sets (nested, overlapping, or a relabelling)
**Fixed 4 alternatives, identical structure on all 20 items — the number was never varied.** Verified
from the methods.

## What numbers are printed or deposited (which tables/files, counts or proportions, per subject or pooled)
**Per-subject but binary.** The Zenodo record's own title says it outright: *"MultiScent-20 Data —
Binary responses (correct vs. incorrect answers) data from full sample, for each odour"*. Single file
`binary_data_irt_analysis.sav` (45,944 B). The IRT analysis is a 2PL on dichotomous outcomes; the paper
reports no distractor-level frequencies and no nominal response model.

Especially frustrating because a *digital* 4-AFC test must have logged which option was tapped — the
information was collected and then discarded at deposit time.

## Access (a DIRECT url you fetched; open, paywalled, or Wayback-only)
Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11219931/ — **open access**.
Data: https://zenodo.org/api/records/8079860 — fetched, **open**; single .sav file confirmed via the
Zenodo API. DOI https://doi.org/10.5281/zenodo.8079860 resolves (HTTP 200).

## Usability verdict (usable now / needs digitizing / needs library access / unusable, and why)
**Unusable for choice shares** — binary outcomes carry no information about which descriptor was chosen.
Worth one email to Nakanishi / de Paula Brandão asking whether the original tap-level logs survive; a
digital instrument makes that plausible, and the same authors also deposited `brandao_2025.md`.

## What the authors concluded, quoted verbatim where possible
The paper validates MultiScent-20 as a digital odour identification instrument using item response
theory, reporting item difficulty and discrimination parameters and concluding the test is psychometrically
adequate for screening. Distractor behaviour is not examined.
