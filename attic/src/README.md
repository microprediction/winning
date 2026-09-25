# Glicko-2, frozen

This was `src/winning/`, the pre-renovation rating-systems package, until the
live package moved to `winning/` (renovation 2026-07, completed 2026-08-18).

**It was not a duplicate of anything, but most of it was.** `skew_calibration`,
`std_calibration`, `lattice_calibration` and `lattice_conventions` had live
equivalents under `winning/classic/`, and four of those filenames shadowed live
ones, so a grep returned two hits with nothing saying which was current. Those
are deleted, along with `elo`, `kernels`, `shims`, `thurstonerating`, the old
test tree and the benchmarks: nothing referenced any of them.

What survives is the dependency closure of the one thing that is still used:

    glicko2.py       Glicko-2, which the live package has no equivalent for
    exact.py         gaussian_win_probabilities, which glicko2 calls
    ratingsystem.py  the Rating / RatingSystem base

Six chess experiments load it by path, through a synthetic `wsrc` module:

    research/chess/exp35_vs_glicko2.py        exp39_predictive_calibration.py
    research/chess/exp37_sparse_players.py    exp40_online_vs_batch.py
    research/chess/exp38_modern_month.py      exp41_tuning_parity.py

**It had stopped working.** `exact.py` imported the external `thurstone`
package, which is retired, and `winning.thurstone` is now a tombstone that
raises on import — so every one of those experiments would have failed at the
Glicko-2 step. It now imports `winning.research`, which is the migration that
tombstone's own message prescribes. Verified end to end: A beating B three
times gives A 1753, B 1247 and a win probability of 0.82.

Nothing here is packaged (`setup.py` lists only `winning.*`), collected
(`pytest.ini` does not name it) or maintained. If Glicko-2 ever earns a place
in the live package it should move there and this directory should go.
