# Papers

**One folder per paper**, as in the other repositories: every
manuscript owns a directory here holding its source, its build
products and the scripts behind its numbers. Directories starting with
an underscore are not manuscripts. Sources live here rather than under
`docs/`, which serves published PDFs from `docs/assets/pdf/`.

| Project | Title | Notes |
|---|---|---|
| [`factor-probit-transform`](factor-probit-transform) | Scalable Share Calibration for Factor Multinomial Probit Models | |
| [`passk_posterior`](passk_posterior) | A Posterior Predictive for Pass@k | Experiment in `research/cavity_calculus/exp2_passk/` (run_passk2.py). |
| [`win_nodes`](win_nodes) | Win Nodes for Bayes Nets: exact order-statistic queries on Gaussian graphical models | Numbers trace to `research/tridiagonal/` and `julia/GMRFExtremes/`. |
| [`exact_pom`](exact_pom) | Deterministic Probabilities for Thompson Sampling and Entropy Search | Experiments in `research/rs_crn/`; quote-verified sources in `research/rs_crn/NOTES.md`. |
| [`general_inversion`](general_inversion) | Scalable Inversion of Contests with Correlated Performances, Including Softmax and Multinomial Probit | [arXiv:2609.01133](https://arxiv.org/abs/2609.01133); SSRN doi:10.2139/ssrn.7307363. Claim-to-script manifest in `CLAIMS.md`; tables pinned at tag `paper-r1`. |
| [`thurstone_humans`](thurstone_humans) | `paper.tex`: Softmax Masking Is a Choice Model … ; `paper_long.tex`: Thurstone is the Model of Choice | |
| [`machine_preference`](machine_preference) | Choice-Set Restriction in Machines and People | |
| [`chess_ratings`](chess_ratings) | Beating a Deployed Rating System: Partial Pooling Against Per-Category Glicko-2 on Lichess | Experiments in `research/chess/`. |
| [`f1_ratings`](f1_ratings) | Rating Formula 1: a case for non-Gaussian noise in rating systems | |
| [`siam2021`](siam2021) | (not a manuscript) Cotton, *Inferring Relative Ability from Winning Probability in Multientrant Contests* | SIAM J. Financial Mathematics 12(1):295–317 (2021), doi:10.1137/19M1276261. Reproduction CSVs and a Harville comparison script. |

## Build convention

No `bibtex`. Bibliographies are inline `thebibliography` blocks; build
with three `pdflatex` passes and verify with
`pdftotext main.pdf - | grep -c "(?)"` returning 0.
