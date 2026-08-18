# Experiment 18: the top-two deletion baseline

The fourth referee round pointed out that the paper's removal-ensemble
speedup (vs naive per-removal recomputation) was measured against the wrong
comparator. The strongest direct-simulation method reuses its draws: one
draw's winner w and runner-up s determine every single-removal outcome
(delete i≠w → winner stays w; delete w → winner becomes s), so counting
winners and winner/runner-up pairs gives the whole N×N deletion matrix in
O(RN + N²).

Result (N=200, k=2, wall time matched at 64s, scored on the 20 highest-share
deletion rows against an independent 1e8-draw top-two reference):

| method | max error |
|---|---|
| conditional-field ensemble (deterministic) | 4.2e-5 |
| top-two simulation, matched time | 1.2e-4 |

The deterministic ensemble is ~3× more accurate at matched wall time, and
reproducible. The earlier 2.8e-17 agreement figure checks algebraic
consistency between two implementations of the same approximation — it is
not an accuracy statement; this experiment provides the accuracy statement.

Run: `python run_deletion_baseline.py` (~5 min).
