# Attic

Code and notebooks from `winning` 1.x that was neither ported to
[thurstone](https://github.com/microprediction/thurstone) nor (yet) rebuilt as a
winning 2.x application. Most items have a corresponding draft issue in
`planning/thurstone_issues/` proposing where they should live next
(`pandas_util.py`, `examples_m6/` and `examples_pandas/` are tracked only by
this list). Nothing in here is
installed, tested, or maintained; the complete original package lives in git history
at v1.0.6.

- `lattice_simulation.py` — place/show/exotic pricing after calibration, longshot
  bias adjustment (draft issues 01, 03)
- `lattice_copula.py` — Gaussian-copula correlated contestants, M6 rank machinery
  (draft issue 02)
- `pandas_util.py` — groupby-apply ability transforms for DataFrames
- `examples_events/` — in-play/cumulative-scoring pricing demo (draft issue 04)
- `examples_m6/` — M6 competition rank-forecasting stub
- `examples_pandas/` — ability transform on survey/e-commerce data
- `notebooks/Ability_Transforms_Updated.ipynb` — the ability transform applied to ~30
  real datasets (draft issue 05)
- `notebooks/Democracy_Correlation_Trade.ipynb` — preference-participation copula
  (draft issue 06)
