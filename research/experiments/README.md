# Experiments for "Scalable Share Calibration for Factor Multinomial Probit Models"

Every number in the paper comes from a committed, seeded script in this
directory. `run_all_paper.py` regenerates the full set. `raceutil.py`
is a shim importing the canonical implementation from
`winning.factor.core`, so these scripts exercise the deployed package.
Experiments 28 and 32 (companion-paper support and exploratory neural
work) live in the kinetics repository.

| Exp | Establishes | Paper |
|----:|-------------|-------|
| 13 | GHK benchmark frontier and smoothness comparison | §7 |
| 14 | Boundary regimes: rank sweep vs GHK; the misspecified factorial | §7 |
| 15 | Perturbation certificate groundwork | §4 |
| 16 | Forward replication vs twin MC references; inversion replication | §4 |
| 17 | Resolution sweeps of the two quadrature knobs | §4 |
| 18 | Removal ensemble vs strongest simulation comparator | §4 |
| 19 | Calibration scaling N=1000..10000, Python and Rust | §4 |
| 20 | Chebyshev-separated pass prototype | §8 |
| 21 | Validation: inverse-crime solver floor; noise attribution | §4 |
| 22 | Minimax-tilting (Botev) anchored comparison | §7 |
| 23 | Newton-CG with matrix-free JVP | §3 |
| 24 | Per-alternative factor RQMC: the natural competitor, 140x | §7 |
| 25 | Log-share accuracy vs higher-resolution internal references | §4 |
| 26 | Pinsker forward-share certificate | §7 |
| 27 | Perturbed-max identities: Stein/trace, conjugate closed forms | §5 |
| 29 | Oracle substitution benchmark, twenty seeds (25% vs 1%) | §1, §7 |
| 30 | Implicit gradients through calibration; cross-market objective | §4 |
| 31 | Damped-Jacobi spectrum: predicted vs measured contraction | §3 |
| 33 | D-heterogeneity grid stress (ratios 1e2, 1e3) | §4 |
| 34 | Temperature-softmax comparator bias | §7 |
| 35 | Inversion vs methodologically independent RQMC targets | §4 |
| 36 | Misspecified factorial replicated over twenty seeds | §7 |
| 37 | Multiplicity relation: excess-over-one -> tr(J)/2 | §5 |
| 38 | Rao-Blackwellized conditional MC (Train-style) comparator | §7 |
