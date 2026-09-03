# Papers

Six manuscript projects live in this repository. Status as of
2026-08-29. The root `README.md` historically described only the
factor-probit paper; this file is the index.

| Project | Title | Venue status |
|---|---|---|
| [`factor-probit-transform`](factor-probit-transform) | Scalable Share Calibration for Factor Multinomial Probit Models | **Submitted.** JCGS target (`VENUES.md`). Submission build is `scalable-share-calibration-jcgs.pdf` (double-spaced). **No SSRN preprint** — the record that once held it now holds the inversion paper (see below). |
| [`passk_posterior`](passk_posterior) | Posterior-Predictive Pass-at-k | **Note** (2026-09-02). Measures the plug-in independence extrapolation inside fine-tuning diagnostics (TailSFT's f16 cited verbatim) against a two-parameter empirical-Bayes posterior predictive on released Pass8 rollouts; experiment in `research/cavity_calculus/exp2_passk/`. |
| [`exact_pom`](exact_pom) | Exact Posterior Probability of Optimality for Factor-Gaussian Beliefs | **Draft** (2026-09-01). The PoM vector VAPOR/ToSFiT declare intractable is exact for factor-form posteriors; CRN and per-arm-observation posteriors derived as exactly factor-plus-diagonal; stopping and bandit experiments in `research/rs_crn/`. Quote-verified sources in `research/rs_crn/NOTES.md`. No venue chosen. |
| [`../docs/latex_src/general_inversion`](../docs/latex_src/general_inversion) | Scalable Inversion of Contests with Correlated Performances, Including Softmax and Multinomial Probit | **arXiv:2609.01133** (permanent id assigned 2026-09-01) and **SSRN preprint**, abstract id 7307363, doi:10.2139/ssrn.7307363 (that record was repurposed from the calibration paper on 2026-08-29). No journal venue chosen. Most active manuscript (nine review rounds actioned); claim-to-script manifest in `CLAIMS.md`; tables pinned at tag `paper-r1`. |
| [`thurstone_humans`](thurstone_humans) | `paper.tex`: Softmax Masking Is a Choice Model … ; `paper_long.tex`: Thurstone is the Model of Choice | **Informal review loop, no venue.** Three review rounds acted on (`REVIEWER_BRIEF.md`, `response_to_third_review.md`). Two divergent manuscripts — see caveat below. |
| [`machine_preference`](machine_preference) | Choice-Set Restriction in Machines and People | **No venue, no status file.** First version Nov 2024, this version 15 Aug 2026. |
| [`f1_ratings`](f1_ratings) | Rating Formula 1: a case for non-Gaussian noise in rating systems | **No venue, no status file.** Oldest by commit (2026-07-09) and **has no bibliography at all** — see caveat. |
| [`siam2021`](siam2021) | (not a manuscript) Cotton, *Inferring Relative Ability from Winning Probability in Multientrant Contests* | **Published**: SIAM J. Financial Mathematics 12(1):295–317 (2021), doi:10.1137/19M1276261. Reproduction CSVs and a Harville comparison script only. |

[`prior-art-inversion-and-shared-field.md`](prior-art-inversion-and-shared-field.md)
is the adversarial prior-art audit covering the inversion claim. Its
verdict, which both inversion papers should keep honoring: *the
defensible claim is the first fast numerical inversion of the N-wise
argmax map at large N for general base densities* — not first
formulation, first integral, first identification result, first
non-Gumbel treatment, first shared-field assembly, or first deletion
ensemble. It rates Li (2018), arXiv:1802.04444, a direct hit on the
inversion claim as published in 2021, with priority preserved by an
earlier public disclosure.

## Known caveats

- **`f1_ratings` has no reference list.** TrueSkill, Elo, Glicko-2,
  OpenSkill, Thurstone–Mosteller, Bradley–Terry and Plackett–Luce are
  all named in prose with no `\bibitem` anywhere. Not submittable as
  is.
- **`thurstone_humans` holds two manuscripts** with different titles
  and dates and no note saying which supersedes; `paper_long.pdf`
  trails its own `.tex` by a commit. `REVIEWER_BRIEF.md` also points
  at commit `b1d598e` on branch `machine-preference-paradox`, not
  `main`, and flags its own provisional intervals.
- **`factor-probit-transform` ships two unverified bibitem titles**
  (Chiang 1961, Mukherjea–Stephens 1990, both marked `% VERIFY`) in
  the *submitted* manuscript, plus a 0-byte `.bbl`, a stale `.blg`,
  and a forked duplicate `-petercotton-edit.tex`. `EDIT-TODO.md`
  carries the open backlog, headed by real-data cross-menu validation.
- **`machine_preference` references an appendix and source files that
  are not in the directory.**
- **`siam2021`'s comparison script targets winning 1.0.3** and does
  not run against the current package.

## Build convention

No `bibtex`. Bibliographies are inline `thebibliography` blocks; build
with three `pdflatex` passes and verify with
`pdftotext main.pdf - | grep -c "(?)"` returning 0.
