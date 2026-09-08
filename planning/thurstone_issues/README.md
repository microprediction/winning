# Draft issues/discussions for microprediction/thurstone

Ideas and machinery from `winning` 1.x that did not make the thurstone port, each
drafted as a GitHub issue. Nothing has been posted — review, edit, then post (or fold
several into one umbrella issue). Suggested split: 01-04 are `enhancement` issues;
05-06 are Discussions material; 07 is a `question`.

Status column added 2026-09-08 after a sweep: four of the seven are
already implemented, and one is not publishable at all.

| # | Title | Status |
|---|---|---|
| 01 | Rank-k and exotic pricing | **realized** in `winning.factor.topk`, beyond the draft's scope |
| 02 | Gaussian copula for correlated contestants | **realized** in `winning/classic/lattice_copula.py` |
| 03 | Longshot-bias adjustment for dividends | **not for publication**: betting-market subject |
| 04 | Performance densities from scoring events | **realized** in `winning/classic/lattice.py` |
| 05 | The ability transform as a statistical tool | open; discussion-grade, no code |
| 06 | Preference-participation copula (turnout) | open; discussion-grade, no code |
| 07 | Port winning 1.x tests as regression fixtures | **substantially realized**: 47 test modules |

Code snapshots referenced by these drafts are preserved in `attic/` in the winning
repo (and in winning git history at tag/commit level).
