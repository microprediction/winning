# SIAM 2021 paper reproductions

Reproduces the Harville-formula and "rule of a quarter" comparison tables from

> Cotton, "Inferring Relative Ability from Winning Probability in Multientrant
> Contests", SIAM J. Financial Mathematics 12(1):295-317 (2021),
> DOI 10.1137/19M1276261.

The CSVs here are the published table data and stand on their own.

`comparison_to_harville.py` is preserved exactly as it ran against the 1.x
package. **Its imports resolve against the current package again**
(verified 2026-08-29): `winning.lattice` and
`winning.lattice_calibration` moved to `winning.classic`, but the old
top-level paths remain as aliases, so every import in the script works
unchanged (emitting a `DeprecationWarning` that names the new home),
and the script starts and runs. The previous instruction to install
`winning==1.0.3` in a separate environment is no longer necessary to
execute it.

**Numerical reproduction is UNVERIFIED.** The script is a Monte Carlo
exotics-pricing demonstration, it is slow (not run to completion here),
and it is *unseeded* — `np.random.randn` at line 28 with no seed set —
so successive runs differ by sampling noise and no run can be expected
to match the committed CSVs exactly. A partial run produced a Model
column agreeing with `rule_of_a_quarter.csv` to about 1e-3 on some rows
and diverging by ~0.02 on others; whether that is Monte Carlo noise or
engine drift is *not* established. Seeding the script and running it to
completion is the outstanding task if these tables ever need to be
regenerated rather than cited.

The CSVs in this directory are the published table data and stand on
their own regardless.

Porting the imports to `winning.classic.*` (to drop the deprecation
warnings) is tracked in
`planning/thurstone_issues/07-port-winning-tests-as-fixtures.md`.
