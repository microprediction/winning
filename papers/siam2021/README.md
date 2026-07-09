# SIAM 2021 paper reproductions

Reproduces the Harville-formula and "rule of a quarter" comparison tables from

> Cotton, "Inferring Relative Ability from Winning Probability in Multientrant
> Contests", SIAM J. Financial Mathematics 12(1):295-317 (2021),
> DOI 10.1137/19M1276261.

The CSVs here are the published table data and stand on their own.

`comparison_to_harville.py` is preserved exactly as it ran against the 1.x
package and imports `winning.lattice*` modules that no longer exist in 2.x —
to execute it, use `pip install winning==1.0.3` (the last published 1.x) in a
separate environment.
Porting it to the `thurstone` API is tracked in
`planning/thurstone_issues/07-port-winning-tests-as-fixtures.md`.
