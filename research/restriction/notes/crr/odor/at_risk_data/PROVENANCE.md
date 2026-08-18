# At-risk data copies

Mirrored here because the source is a **revocable anonymized OSF view-only token**, not a public
deposit. OSF node `3hjmw` is `public: false` with no registration, no fork, and no DOI, so if the
authors revoke the link this data becomes unobtainable. See `../tolomeo_2026.md`.

Source: Tolomeo E, Ceraudo L, Kolb R, Dalton PH, Liuzza MT, Parma V. *Front Psychol* 2026;17:1661164.
doi:10.3389/fpsyg.2026.1661164. PMC12863060.
Retrieved 2026-08-17 from https://osf.io/3hjmw/?view_only=f93d1dd6812549b08d5503a207c4e49d

| File | Source download URL (token required) |
|---|---|
| tolomeo_2026_IRT_NRM.csv | https://osf.io/download/h456t/?view_only=f93d1dd6812549b08d5503a207c4e49d |
| tolomeo_2026_IRT_LABELED_DICHOTOMOUS.csv | https://osf.io/download/grqcu/?view_only=f93d1dd6812549b08d5503a207c4e49d |
| tolomeo_2026_codebook.docx | https://osf.io/download/4fs72/?view_only=f93d1dd6812549b08d5503a207c4e49d |
| tolomeo_2026_analysis.R | https://osf.io/download/3pqsh/?view_only=f93d1dd6812549b08d5503a207c4e49d |

`IRT_NRM.csv`: 397 subjects x 16 items, 1 = correct, 2/3/4 = distractors. **The 2/3/4 codes are
frequency-ranked, not printed-position** (counts are monotone in all 16 items), so descriptor names are
NOT recoverable from these files alone — see `../tolomeo_2026.md` for the Table 3 cross-reference and
its limits. Join the two CSVs **positionally**; `id` is not unique (1, 92, 307 duplicated).

Placed under notes/ only because this agent was restricted to this directory. **Move to a proper data
directory when convenient.**
