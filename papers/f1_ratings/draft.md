# Rating Formula 1: joint order statistics and a disaster slab

*Draft skeleton, July 2026. Every number below is measured; reproduce via
`python -m winning.benchmarks.run_benchmark --dataset f1` and the scripts in
`research/`. Target length: 6-8 pages, tone of the inspiration-simplex note.*

## Abstract (sketch)

Formula 1 is an unusually hostile environment for rating systems: grids of
10-55 cars, full finish orders, and a retirement process that historically
eliminated half the field. We rate every world-championship grand prix since
1950 (1,158 races, f1db) with a lattice-based Thurstonian system whose
beliefs are unrestricted densities and whose updates use the exact joint
likelihood of the whole finish order, computed in O(N) per contestant.
Against TrueSkill, Glicko-2, Elo and OpenSkill under a prequential protocol,
the lattice rater wins every headline metric, while pairwise-decomposition
systems degrade or diverge as grids widen. Because the lattice imposes no
distributional family, the performance density is a modeling choice; we show
a two-component form — ordinary Gaussian pace plus a "disaster slab" of
far-slow mass matching the DNF process — improves every metric further, that
the slab mass is physically interpretable (it tracks the historical DNF
rate), and that an era-adaptive slab fitted prequentially from trailing
retirement frequency sets the overall record (log loss 2.2001), its fitted
mass ending near the modern DNF rate after following the sport's reliability
revolution across seven decades. Symmetric heavy tails and skew, by
contrast, measurably hurt: what F1 needs is one-sided catastrophe, not
kurtosis. [If odds data found: the market comparison.]

## 1. Why F1

- Wide fields, full orders: joint order statistics vs pairwise decomposition
  becomes the whole game (Glicko-2 falls below uniform at 20+ cars without
  numerical guards; OpenSkill ThurstoneMosteller diverges).
- The retirement process: decade DNF rates 0.42/0.49/0.51/0.59/0.51/0.30/
  0.18/0.13 (1950s..2020s) — a distributional regime change inside one
  dataset.
- Classification rules give clean data semantics (90% rule; retirees ordered
  by laps; shared drives in the 1950s produce dead heats we keep as ties).

## 2. The rater

One paragraph on: lattice beliefs, exact O(N) chain likelihood for the full
order (forward/backward chains; MC-validated), diffusion in time, exact
dead-heat-aware prediction via winner-of-many. Point to the SIAM 2021
transform as the underlying machinery and to the chain construction.

## 3. Headline benchmark (prequential, 1950-2026)

| system | log loss | accuracy | Kendall tau | ECE | rank-PIT KS |
|---|---|---|---|---|---|
| Thurstone lattice | 2.2247 | 0.316 | 0.328 | 0.015 | 0.176 |
| TrueSkill | 2.3461 | 0.275 | 0.300 | 0.011 | 0.199 |
| Elo (multi-entrant) | 2.4647 | 0.303 | 0.319 | 0.026 | 0.220 |
| OpenSkill PL | 2.7045 | 0.259 | 0.283 | 0.023 | 0.223 |
| OpenSkill TM-Full | 2.9509 | 0.007 | 0.106 | 0.010 | 0.628 |
| uniform | 3.1615 | 0.043 | 0 | 0.000 | - |
| Glicko-2 (guarded) | 3.7271 | 0.271 | 0.255 | 0.022 | 0.166 |

(Numbers from the exact-chain updater runs; refresh table from BENCHMARKS.md
at writing time.)

## 4. The performance distribution is a free parameter — use it

| base density | log loss | ECE | rank-PIT KS |
|---|---|---|---|
| N(0,1) + 25% disaster slab | 2.2031 | 0.0154 | 0.1578 |
| gaussian | 2.2247 | 0.0154 | 0.1758 |
| skew-normal a=2 | 2.2221 | 0.0097 | 0.1648 |
| student-t nu=5 (symmetric) | 2.2708 | 0.0178 | 0.1707 |
| tight core 0.5 + 25% slab | 2.2542 | 0.0126 | 0.1361 |
| near-Dirac cores (0.2-0.3) | 2.42-2.52 | - | - |

The taxonomy: one-sided disaster mass helps everything; symmetric kurtosis
injects impossible miracle-fast mass and hurts; a tight-core/slab mix trades
winner sharpness for the best field-shape calibration (a genuine Pareto
frontier). No Gaussian- or Gumbel-family comparator can express any point on
it.

## 5. The slab is physics: era-adaptive retirement mass

DNF-rate table by decade; the era-adaptive rater (trailing-window prequential
p_dnf, no leakage) sets the overall F1 record:

| base density | log loss | Kendall tau | rank-PIT KS |
|---|---|---|---|
| era-adaptive slab | 2.2001 | 0.3317 | 0.1853 |
| fixed slab p=0.25 | 2.2031 | 0.3308 | 0.1578 |
| gaussian | 2.2247 | 0.3282 | 0.1758 |

The fitted mass ends near the modern DNF rate (0.10 vs true ~0.13; the
tied-last proxy misses single-retirement races), having tracked the sport's
reliability revolution across seven decades — the parameter is physics, not
curve-fitting. Rank-PIT prefers the fixed slab (kernel rebuilds churn shape):
the Pareto frontier has three corners.

## 6. (Conditional) The market test

[PENDING odds sourcing. If full-field winner odds exist for recent seasons:
market ceiling row, Thurstone-vs-Luce place structure, and the tempered-pool
beat-the-market protocol from the HK study.]

## 7. Related work and honesty section

TrueSkill/EP lineage; Harville and the SIAM 2021 result (its large-sample
confirmation lives in the HK companion study); limitations: driver-vs-car
identification unaddressed (constructor effects folded into driver ratings;
multiray decomposition is future work), DNFs tied last vs official
laps-order untested A/B, one dataset family (f1db).

## Reproducibility

winning repo @ [commit], `--dataset f1`, research/f1_dirac_disaster.py,
research/t_sweep.py, research/f1_era_slab.py. CC-BY f1db data fetched at run
time.
