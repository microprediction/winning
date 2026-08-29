# SIAM 2021 paper reproductions

Reproduces the Harville-formula and "rule of a quarter" comparison tables from

> Cotton, "Inferring Relative Ability from Winning Probability in Multientrant
> Contests", SIAM J. Financial Mathematics 12(1):295-317 (2021),
> DOI 10.1137/19M1276261.

The CSVs here are the published table data and stand on their own.

`comparison_to_harville.py` is preserved exactly as it ran against the 1.x
package. **It runs against the current package again** (verified
2026-08-29): the modules it imports — `winning.lattice` and
`winning.lattice_calibration` — moved to `winning.classic` but the old
top-level paths remain as aliases, so every import in the script
resolves unchanged (it emits a `DeprecationWarning` naming the new
home). The previous instruction to install `winning==1.0.3` in a
separate environment is no longer necessary.

The script is a Monte Carlo exotics-pricing demonstration and takes a
long time to run; the CSVs in this directory are the published table
data and stand on their own regardless.

Porting the imports to `winning.classic.*` (to drop the deprecation
warnings) is tracked in
`planning/thurstone_issues/07-port-winning-tests-as-fixtures.md`.
