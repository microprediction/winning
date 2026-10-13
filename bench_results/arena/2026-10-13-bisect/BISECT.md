# Arena timing changes: attribution (2026-10-13)

Inputs: `../2026-10-13-rerun/` (main a7f7d9fd) and
`../2026-10-13-baseline-3b9eaa18/` (the 2026-08-15 code, rerun today with
the same 4-thread caps). `cmp_arena.py OLD NEW` produces the tables.
Every arena accuracy number (max_err, max_log_err) is identical across
the August record, the August code rerun today, and main.

## Lattice "2x" vs the committed August records: environment, not code

`lat_time.py` (best of 7, RAYON_NUM_THREADS=4) on the same machine:

    3b9eaa18 code: n50k2 0.0253 | n200k2 0.0985 | n1000k2 0.4859 | n200k3 0.8634
    a7f7d9fd code: n50k2 0.0257 | n200k2 0.0988 | n1000k2 0.4958 | n200k3 0.8885

Without the thread cap the August code prices n1000k2 in 0.095 s and
n200k3 in 0.163 s, so the August records (0.235 / 0.437) were taken with
more rayon threads than the 4 this rerun allows.

## Code regressions (same machine, same caps), bisected over main's first parent

`me_bisect.py` times n200k2 (best of 3) at each probe; raw lines:

    0 3b9eaa18 {'mendell_elston': 0.898, 'ghk': 1.796, 'smc_orthant': 2.155}
    325 6ba66516 {'mendell_elston': 0.873, 'ghk': 1.773, 'smc_orthant': 2.103}
    487 42c4f555 {'mendell_elston': 0.877, 'ghk': 1.781, 'smc_orthant': 2.108}
    568 138e7adb {'mendell_elston': 0.878, 'ghk': 2.084, 'smc_orthant': 2.179}
    588 9c268c7e {'mendell_elston': 0.886, 'ghk': 2.067, 'smc_orthant': 2.166}
    598 5baaad67 {'mendell_elston': 0.895, 'ghk': 2.028, 'smc_orthant': 2.112}
    603 908aff29 {'mendell_elston': 0.876, 'ghk': 2.028, 'smc_orthant': 2.11}
    606 d19a2cde {'mendell_elston': 0.875, 'ghk': 2.028, 'smc_orthant': 2.106}
    607 482c8156 {'mendell_elston': 0.871, 'ghk': 2.025, 'smc_orthant': 2.106}
    608 41b43fcc {'mendell_elston': 1.824, 'ghk': 2.031, 'smc_orthant': 3.096}
    609 71a5b48e {'mendell_elston': 1.972, 'ghk': 2.03, 'smc_orthant': 3.098}
    629 643cd797 {'mendell_elston': 1.815, 'ghk': 2.039, 'smc_orthant': 3.096}
    639 272a1c7b {'mendell_elston': 1.813, 'ghk': 2.052, 'smc_orthant': 3.086}
    644 767b7978 {'mendell_elston': 1.823, 'ghk': 2.059, 'smc_orthant': 3.088}
    645 12c6ce40 {'mendell_elston': 2.072, 'ghk': 2.366, 'smc_orthant': 3.177}
    647 f1a0e68e {'mendell_elston': 1.819, 'ghk': 2.351, 'smc_orthant': 3.113}
    650 a7f7d9fd {'mendell_elston': 1.869, 'ghk': 2.417, 'smc_orthant': 3.128}

- Mendell-Elston 2.1x and SMC-orthant 1.45x: #454 (41b43fcc, 2026-10-05).
  `winning/methods/orthant_extra.py::_order_variables` now breaks ties
  with a pure-Python key that sorts each constraint's m-1 covariances, for
  every one of the N difference problems: O(N m^2 log m) Python work,
  ~1 s at N = 200. Computing the tie-break only when z actually ties
  (or with np.lexsort) would restore the old cost.
- GHK / QMC-GHK ~1.35x in total: ~+15% at #257-era commits (between
  42c4f555 and 138e7adb, not narrowed further) and ~+15% at #706
  (12c6ce40, 2026-10-13). Below the 2x flag.
