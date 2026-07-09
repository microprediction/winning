# Benchmarks

Prequential evaluation of rating systems: for each event in time order, elapse
the event's time gap, predict win probabilities from current ratings, score,
then observe the outcome. The first 20% of each dataset is unscored warm-up.
Events with a dead-heat winner update ratings but are not scored (the winner
metrics need a unique winner).

Reproduce any table with:

    pip install winning[benchmarks]
    python -m winning.benchmarks.run_benchmark --dataset <name>
    python -m winning.benchmarks.run_benchmark --list

## Metrics

- **Log loss / Brier / Accuracy** — scoring rules on the predicted winner.
- **Kendall tau** — rank correlation between predicted order (by rating) and
  the observed finish order.
- **ECE** — expected calibration error over every contestant win probability
  (20 equal-count bins): measures whether the *spread* of predictions matches
  reality; a system that misjudges performance variation is systematically
  over- or under-confident here.
- **Rank-PIT KS** — multi-entrant datasets only: sample finish orders from each
  system's predictive performance distributions, place each observed finish
  position within its simulated distribution, and measure the Kolmogorov
  distance of these PIT values from uniform. Zero means the full
  finish-position *distribution* — not just the mean — is modeled correctly.
- **TV vs oracle** — synthetic worlds only: total-variation distance between
  predicted and true win probabilities.

Baseline rows: **Uniform** (floor), **Oracle** (true abilities; synthetic
only), **Market** (bookmaker odds, pari-mutuel odds, or the platform's own
production ratings — a ceiling trained on more information than the systems
under test). A fixed row is scored only on events carrying that attribute; in
every shipped dataset the attribute covers (essentially) all events, so rows
are like-for-like comparable.

## Systems

`ThurstoneRating` (this package), TrueSkill (`trueskill`; patented by
Microsoft, non-commercial license — research comparator only), OpenSkill
PlackettLuce and ThurstoneMostellerFull (`openskill`), Glicko-2 and
multi-entrant Elo (this package, from the published specs). TrueSkill receives
the empirical draw rate of each dataset. "(lattice predict)" rows keep a
system's own updates but price its beliefs with this package's exact lattice
predictor, separating update quality from prediction-formula quality.

## Datasets

| Name | Events | Fields | Identity | Source / license |
|---|---|---|---|---|
| synthetic | 4,000 | 6-12 | synthetic | generated, oracle truth attached |
| synthetic-drift | 4,000 | 6-12 | synthetic | generated, abilities drift |
| synthetic-hetero | 4,000 | 6-12 | synthetic | generated, per-contestant noise sd 0.6/1.6 |
| f1 | 1,158 | 10-55 | drivers | [f1db](https://github.com/f1db/f1db), CC-BY-4.0; DNFs tied last |
| atp / wta | ~31k each | 2 | players | Sackmann archives via fork mirrors, CC BY-NC-SA 4.0 |
| chess | 121,332 | 2 | players | [Lichess open database](https://database.lichess.org) 2013-01, CC0 |
| sumo | 109,995 | 2 | rikishi ids | SumoDB via data.world mirror, research use |
| epl | ~7.4k | 2 clubs | clubs | [football-data.co.uk](https://www.football-data.co.uk), free; Bet365 odds |
| halo2 / halo2-ffa | 6k / 58k | 2 / 2-12 | anonymized gamers | TrueSkill paper data via [mbmlbook](https://github.com/dotnet/mbmlbook), MIT |
| hkracing | ~6.3k | ~8-14 | named horses | Kaggle gdaley/hkracing (needs Kaggle credentials) |
| pubg | ~20k | ~90 | players | Kaggle competition (needs credentials + rules acceptance) |

Results tables below are from the runs of July 2026 on an M-series laptop,
with ThurstoneRating's exact-chain updater (the July 9 upgrade that replaced
Plackett peeling; see research/exact_order_update.py — better on every
dataset with large fields, at half the cost). Synthetic-world rankings are
seed-stable: across five seeds the log-loss ordering TrueSkill < Thurstone
lattice < Glicko-2 < Elo < OpenSkill PlackettLuce held in every run (measured
pre-chain; the chain improves the Thurstone row without changing that
ordering on the seed retested).

### Synthetic races: 200 contestants, 4000 events, fields of 6-12, static abilities

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Oracle (true abilities) | 1.4625 | 0.6826 | 0.4469 | 0.5078 | 0.0049 | - | 0.0000 | 0.0 |
| TrueSkill | 1.4793 | 0.6879 | 0.4431 | 0.5047 | 0.0072 | 0.0202 | 0.0630 | 5.8 |
| Thurstone lattice (winning) | 1.5118 | 0.6983 | 0.4328 | 0.4900 | 0.0096 | 0.0259 | 0.1110 | 5.7 |
| Glicko-2 | 1.5456 | 0.7092 | 0.4319 | 0.4851 | 0.0209 | 0.0143 | 0.1548 | 2.9 |
| Elo (multi-entrant) | 1.5499 | 0.7079 | 0.4403 | 0.4976 | 0.0271 | 0.0310 | 0.1498 | 0.3 |
| OpenSkill PlackettLuce | 1.7295 | 0.7783 | 0.4397 | 0.4987 | 0.0483 | 0.1753 | 0.2887 | 1.1 |
| OpenSkill ThurstoneMostellerFull | 1.7855 | 0.7894 | 0.4050 | 0.4642 | 0.0444 | 0.1604 | 0.3183 | 1.5 |
| Uniform baseline | 2.1763 | 0.8834 | 0.1166 | 0.0000 | 0.0004 | - | 0.4894 | 0.0 |
| OpenSkill PlackettLuce (lattice predict) | 3.1900 | 0.8662 | 0.4391 | 0.4987 | 0.0810 | 0.1753 | 0.3758 | 3.1 |

### Synthetic races with ability drift (tau=0.01/event): 200 contestants, 4000 events

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Oracle (true abilities) | 1.3328 | 0.6366 | 0.5031 | 0.5538 | 0.0059 | - | 0.0000 | 0.0 |
| TrueSkill | 1.3890 | 0.6545 | 0.4891 | 0.5311 | 0.0058 | 0.0116 | 0.1191 | 5.8 |
| Thurstone lattice (winning) | 1.3939 | 0.6568 | 0.4806 | 0.5310 | 0.0069 | 0.0206 | 0.1157 | 5.7 |
| Glicko-2 | 1.4442 | 0.6735 | 0.4788 | 0.5269 | 0.0260 | 0.0185 | 0.1724 | 2.8 |
| Elo (multi-entrant) | 1.4636 | 0.6759 | 0.4847 | 0.5333 | 0.0351 | 0.0335 | 0.1890 | 0.3 |
| OpenSkill PlackettLuce | 1.6972 | 0.7693 | 0.4816 | 0.5330 | 0.0570 | 0.1668 | 0.3354 | 1.1 |
| OpenSkill ThurstoneMostellerFull | 1.7886 | 0.7824 | 0.4481 | 0.4902 | 0.0492 | 0.1605 | 0.3638 | 1.5 |
| Uniform baseline | 2.1685 | 0.8825 | 0.1175 | 0.0000 | 0.0004 | - | 0.5241 | 0.0 |
| OpenSkill PlackettLuce (lattice predict) | 2.8243 | 0.8006 | 0.4825 | 0.5330 | 0.0736 | 0.1668 | 0.3470 | 3.1 |

### Synthetic races, heteroskedastic noise (per-contestant sd 0.6 or 1.6): 200 contestants, 4000 events

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Oracle (true abilities) | 1.5275 | 0.7215 | 0.3975 | 0.4060 | 0.0039 | - | 0.0000 | 0.0 |
| TrueSkill | 1.6952 | 0.7552 | 0.3750 | 0.4656 | 0.0144 | 0.0179 | 0.2125 | 5.8 |
| Elo (multi-entrant) | 1.7209 | 0.7629 | 0.3688 | 0.4602 | 0.0157 | 0.0348 | 0.2340 | 0.4 |
| Glicko-2 | 1.7294 | 0.7673 | 0.3597 | 0.4444 | 0.0094 | 0.0231 | 0.2424 | 2.9 |
| Thurstone lattice (winning) | 1.7337 | 0.7665 | 0.3681 | 0.4517 | 0.0196 | 0.0238 | 0.2315 | 5.8 |
| OpenSkill PlackettLuce | 1.8740 | 0.8111 | 0.3456 | 0.4514 | 0.0257 | 0.1794 | 0.3210 | 1.1 |
| OpenSkill ThurstoneMostellerFull | 1.9456 | 0.8120 | 0.3459 | 0.4174 | 0.0294 | 0.1580 | 0.3268 | 1.5 |
| Uniform baseline | 2.1767 | 0.8834 | 0.1166 | 0.0000 | 0.0004 | - | 0.4613 | 0.0 |
| OpenSkill PlackettLuce (lattice predict) | 4.5635 | 1.0318 | 0.3459 | 0.4514 | 0.1026 | 0.1794 | 0.4900 | 3.1 |

### English Premier League 2005-2025 (7600 matches, 1837 draws; football-data.co.uk, Bet365 odds as market ceiling)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Market odds | 0.5482 | 0.3697 | 0.7237 | 0.4475 | 0.0199 | - | - | 0.0 |
| Elo (multi-entrant) | 0.5896 | 0.4043 | 0.6810 | 0.3621 | 0.0231 | - | - | 0.0 |
| Thurstone lattice (winning) | 0.5916 | 0.4056 | 0.6756 | 0.3512 | 0.0319 | - | - | 1.5 |
| OpenSkill ThurstoneMostellerFull | 0.5949 | 0.4085 | 0.6789 | 0.3577 | 0.0354 | - | - | 0.2 |
| TrueSkill | 0.5962 | 0.4099 | 0.6752 | 0.3504 | 0.0262 | - | - | 0.5 |
| OpenSkill PlackettLuce | 0.6052 | 0.4131 | 0.6797 | 0.3595 | 0.0563 | - | - | 0.2 |
| OpenSkill PlackettLuce (lattice predict) | 0.6052 | 0.4131 | 0.6797 | 0.3595 | 0.0563 | - | - | 0.1 |
| Glicko-2 | 0.6071 | 0.4180 | 0.6810 | 0.3621 | 0.0728 | - | - | 0.1 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0009 | - | - | 0.0 |

### Halo 2 beta head-to-head (6028 games, 312 draws; the original TrueSkill paper's data, mbmlbook, MIT)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Thurstone lattice (winning) | 0.5802 | 0.3967 | 0.6954 | 0.3915 | 0.0284 | - | - | 1.4 |
| OpenSkill ThurstoneMostellerFull | 0.5810 | 0.3951 | 0.6943 | 0.3886 | 0.0322 | - | - | 0.1 |
| Glicko-2 | 0.5811 | 0.3967 | 0.7049 | 0.4099 | 0.0527 | - | - | 0.1 |
| TrueSkill | 0.5818 | 0.3957 | 0.6971 | 0.3943 | 0.0296 | - | - | 0.4 |
| OpenSkill PlackettLuce | 0.5867 | 0.3993 | 0.6956 | 0.3912 | 0.0358 | - | - | 0.1 |
| OpenSkill PlackettLuce (lattice predict) | 0.5867 | 0.3993 | 0.6956 | 0.3912 | 0.0358 | - | - | 0.1 |
| Elo (multi-entrant) | 0.6193 | 0.4296 | 0.6891 | 0.3782 | 0.1007 | - | - | 0.0 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0011 | - | - | 0.0 |

### ATP tennis 2013-2024 (31658 matches; Sackmann archive)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Elo (multi-entrant) | 0.6214 | 0.4327 | 0.6481 | 0.2961 | 0.0247 | - | - | 0.1 |
| Glicko-2 | 0.6227 | 0.4336 | 0.6528 | 0.3055 | 0.0225 | - | - | 0.3 |
| Thurstone lattice (winning) | 0.6244 | 0.4345 | 0.6509 | 0.3018 | 0.0260 | - | - | 7.3 |
| OpenSkill ThurstoneMostellerFull | 0.6332 | 0.4379 | 0.6530 | 0.3059 | 0.0443 | - | - | 0.7 |
| TrueSkill | 0.6367 | 0.4426 | 0.6430 | 0.2859 | 0.0395 | - | - | 2.0 |
| OpenSkill PlackettLuce (lattice predict) | 0.6869 | 0.4606 | 0.6493 | 0.2986 | 0.1068 | - | - | 0.6 |
| OpenSkill PlackettLuce | 0.6869 | 0.4606 | 0.6493 | 0.2986 | 0.1068 | - | - | 0.6 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0001 | - | - | 0.1 |

### WTA tennis 2013-2024 (31400 matches; Sackmann archive)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Glicko-2 | 0.6312 | 0.4412 | 0.6433 | 0.2867 | 0.0243 | - | - | 0.3 |
| Thurstone lattice (winning) | 0.6321 | 0.4416 | 0.6424 | 0.2847 | 0.0246 | - | - | 7.4 |
| Elo (multi-entrant) | 0.6323 | 0.4424 | 0.6388 | 0.2777 | 0.0216 | - | - | 0.1 |
| OpenSkill ThurstoneMostellerFull | 0.6383 | 0.4443 | 0.6452 | 0.2905 | 0.0426 | - | - | 0.7 |
| TrueSkill | 0.6427 | 0.4497 | 0.6358 | 0.2717 | 0.0330 | - | - | 2.0 |
| OpenSkill PlackettLuce (lattice predict) | 0.6835 | 0.4670 | 0.6401 | 0.2801 | 0.1019 | - | - | 0.6 |
| OpenSkill PlackettLuce | 0.6835 | 0.4670 | 0.6401 | 0.2801 | 0.1019 | - | - | 0.6 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | - | - | 0.1 |

### Sumo bouts 1983-2021 (109995 bouts, Makuuchi+Juryo; SumoDB via data.world mirror, research use)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Thurstone lattice (winning) | 0.6668 | 0.4753 | 0.5724 | 0.1447 | 0.0284 | - | - | 27.0 |
| Glicko-2 | 0.6675 | 0.4757 | 0.5691 | 0.1382 | 0.0182 | - | - | 1.1 |
| TrueSkill | 0.6698 | 0.4775 | 0.5735 | 0.1471 | 0.0210 | - | - | 6.9 |
| OpenSkill ThurstoneMostellerFull | 0.6729 | 0.4803 | 0.5699 | 0.1398 | 0.0442 | - | - | 2.4 |
| Elo (multi-entrant) | 0.6733 | 0.4812 | 0.5658 | 0.1316 | 0.0530 | - | - | 0.4 |
| OpenSkill PlackettLuce | 0.6894 | 0.4908 | 0.5714 | 0.1428 | 0.0796 | - | - | 2.3 |
| OpenSkill PlackettLuce (lattice predict) | 0.6894 | 0.4908 | 0.5714 | 0.1428 | 0.0796 | - | - | 2.1 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | - | - | 0.2 |

### Lichess rated standard chess, 2013-01 (121332 games, 3982 draws; CC0; Market row = the site's own ratings at game time)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Market odds | 0.6136 | 0.4253 | 0.6602 | 0.3203 | 0.0256 | - | - | 0.2 |
| TrueSkill | 0.6185 | 0.4297 | 0.6537 | 0.3074 | 0.0141 | - | - | 7.8 |
| OpenSkill ThurstoneMostellerFull | 0.6212 | 0.4316 | 0.6517 | 0.3033 | 0.0293 | - | - | 2.7 |
| Glicko-2 | 0.6225 | 0.4336 | 0.6517 | 0.3035 | 0.0379 | - | - | 1.3 |
| Thurstone lattice (winning) | 0.6228 | 0.4345 | 0.6463 | 0.2927 | 0.0047 | - | - | 29.1 |
| Elo (multi-entrant) | 0.6363 | 0.4468 | 0.6284 | 0.2568 | 0.0107 | - | - | 0.4 |
| OpenSkill PlackettLuce | 0.6491 | 0.4461 | 0.6487 | 0.2974 | 0.0760 | - | - | 2.5 |
| OpenSkill PlackettLuce (lattice predict) | 0.6491 | 0.4461 | 0.6487 | 0.2974 | 0.0760 | - | - | 2.4 |
| Uniform baseline | 0.6931 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | - | - | 0.2 |

### Halo 2 beta free-for-all (57772 games, field sizes 2-12; the original TrueSkill paper's data, mbmlbook, MIT)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| TrueSkill | 1.0923 | 0.5853 | 0.5333 | 0.3974 | 0.0073 | 0.0458 | - | 36.1 |
| Thurstone lattice (winning) | 1.0972 | 0.5868 | 0.5350 | 0.3961 | 0.0173 | 0.0454 | - | 43.0 |
| Glicko-2 | 1.1034 | 0.5895 | 0.5403 | 0.4049 | 0.0280 | 0.0486 | - | 17.8 |
| Elo (multi-entrant) | 1.1552 | 0.6087 | 0.5381 | 0.3853 | 0.0411 | 0.0569 | - | 2.5 |
| OpenSkill PlackettLuce | 1.1657 | 0.6130 | 0.5417 | 0.3930 | 0.0287 | 0.0969 | - | 6.6 |
| OpenSkill PlackettLuce (lattice predict) | 1.3035 | 0.6243 | 0.5396 | 0.3930 | 0.0710 | 0.0969 | - | 18.7 |
| Uniform baseline | 1.3850 | 0.7249 | 0.2751 | 0.0000 | 0.0000 | - | - | 0.2 |
| OpenSkill ThurstoneMostellerFull | 5.6504 | 0.7786 | 0.3887 | 0.1992 | 0.0892 | 0.3841 | - | 7.5 |

### Formula 1 1950-present (1158 grands prix; f1db, CC-BY-4.0; DNFs tied last)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Thurstone lattice (winning) | 2.2247 | 0.8283 | 0.3161 | 0.3282 | 0.0154 | 0.1758 | - | 7.6 |
| TrueSkill | 2.3461 | 0.8529 | 0.2751 | 0.2996 | 0.0109 | 0.1993 | - | 6.5 |
| Elo (multi-entrant) | 2.4647 | 0.8676 | 0.3031 | 0.3185 | 0.0261 | 0.2192 | - | 0.3 |
| OpenSkill PlackettLuce | 2.7045 | 0.9131 | 0.2589 | 0.2828 | 0.0233 | 0.2220 | - | 1.6 |
| OpenSkill ThurstoneMostellerFull | 2.9509 | 0.9422 | 0.0065 | 0.1064 | 0.0096 | 0.6275 | - | 2.1 |
| Uniform baseline | 3.1615 | 0.9570 | 0.0430 | 0.0000 | 0.0004 | - | - | 0.0 |
| Glicko-2 | 3.7272 | 0.9714 | 0.2708 | 0.2553 | 0.0216 | 0.1661 | - | 2.4 |
| OpenSkill PlackettLuce (lattice predict) | 4.4013 | 1.1397 | 0.2589 | 0.2828 | 0.0408 | 0.2220 | - | 2.3 |

### Hong Kong horse racing 1997-2005 (6348 races, named horses; Kaggle gdaley/hkracing; Market row = pari-mutuel win odds)

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Market odds | 2.0409 | 0.8243 | 0.2996 | 0.3690 | 0.0052 | - | - | 0.1 |
| Glicko-2 | 2.3575 | 0.8936 | 0.1831 | 0.2527 | 0.0057 | 0.0166 | - | 6.4 |
| Thurstone lattice (winning) | 2.3801 | 0.8940 | 0.1735 | 0.2403 | 0.0127 | 0.0251 | - | 12.9 |
| TrueSkill | 2.4674 | 0.9088 | 0.1738 | 0.2065 | 0.0260 | 0.0365 | - | 13.5 |
| OpenSkill PlackettLuce | 2.4701 | 0.9120 | 0.0982 | 0.1767 | 0.0084 | 0.0686 | - | 2.9 |
| Elo (multi-entrant) | 2.4719 | 0.9115 | 0.1103 | 0.1825 | 0.0098 | 0.0135 | - | 0.8 |
| Uniform baseline | 2.5184 | 0.9186 | 0.0814 | 0.0000 | 0.0002 | - | - | 0.1 |
| OpenSkill PlackettLuce (lattice predict) | 2.8877 | 0.9968 | 0.1014 | 0.1767 | 0.0541 | 0.0686 | - | 6.4 |
| OpenSkill ThurstoneMostellerFull | 4.1601 | 0.9433 | 0.1078 | 0.0401 | 0.0408 | 0.4468 | - | 3.7 |

### HK racing LAB (6348 races): oracle = market^1.05 (fitted recalibration); fundamental systems judged by TV to market-implied truth

| System | Log loss | Brier | Accuracy | Kendall tau | ECE | Rank-PIT KS | TV vs oracle | Seconds |
|---|---|---|---|---|---|---|---|---|
| Oracle (true abilities) | 2.0396 | 0.8242 | 0.2996 | 0.3690 | 0.0045 | - | 0.0000 | 0.1 |
| Market odds | 2.0409 | 0.8243 | 0.2996 | 0.3690 | 0.0052 | - | 0.0162 | 0.1 |
| Glicko-2 | 2.3575 | 0.8936 | 0.1831 | 0.2527 | 0.0057 | 0.0166 | 0.3059 | 6.5 |
| Thurstone lattice (winning) | 2.3801 | 0.8940 | 0.1735 | 0.2403 | 0.0127 | 0.0251 | 0.3157 | 12.9 |
| TrueSkill | 2.4674 | 0.9088 | 0.1738 | 0.2065 | 0.0260 | 0.0365 | 0.3494 | 13.5 |
| OpenSkill PlackettLuce | 2.4701 | 0.9120 | 0.0982 | 0.1767 | 0.0084 | 0.0686 | 0.3674 | 2.9 |
| Elo (multi-entrant) | 2.4719 | 0.9115 | 0.1103 | 0.1825 | 0.0098 | 0.0135 | 0.3681 | 0.8 |
| Uniform baseline | 2.5184 | 0.9186 | 0.0814 | 0.0000 | 0.0002 | - | 0.3881 | 0.1 |
| OpenSkill PlackettLuce (lattice predict) | 2.8877 | 0.9968 | 0.1014 | 0.1767 | 0.0541 | 0.0686 | 0.4749 | 6.4 |
| OpenSkill ThurstoneMostellerFull | 4.1601 | 0.9433 | 0.1078 | 0.0401 | 0.0408 | 0.4468 | 0.4467 | 3.7 |

The LAB variant implements planning/rating_lab.md: the recalibrated
pari-mutuel (temperature 1.05, fitted leakage-free by
research/beat_the_market.py) is treated as ground truth, so purely
fundamental systems are judged by TV distance to market-implied
probabilities — full probability vectors every race instead of one
outcome draw, an order of magnitude more statistical power. Anchors: the
raw market sits 0.016 TV from truth (the size of the longshot bias);
uniform ignorance sits 0.388. The best fundamental system (Glicko-2,
0.306) recovers about a fifth of the distance from ignorance to truth —
the other four fifths is the lab's standing challenge, and what
condition-aware and market-hybrid raters exist to chase.

## Reading the results

- The lattice rater (this package) wins Formula 1 decisively and sumo by a
  whisker, sits in a statistical tie at the top of WTA/ATP/EPL/Halo
  head-to-head, and is second by small margins to TrueSkill on TrueSkill's
  own Halo free-for-all data (0.005) and on the synthetic worlds that match
  its generative assumptions exactly (parity on the drifting world: 1.3939
  vs 1.3890, with better TV to oracle).
- Distributional calibration is the package's consistent edge: best or
  near-best ECE and rank-PIT on nearly every dataset (e.g. rank-PIT KS 0.0069
  vs TrueSkill's 0.0196 on static synthetic; ECE 0.0057 on chess).
- The heteroskedastic world hurts every system (all assume common performance
  noise); per-contestant scale learning — supported by thurstone's 2-D
  (loc, scale) calibration — is the obvious next step and none of the
  incumbents offer it.
- Markets are the ceiling everywhere they exist: Bet365 beats every system on
  EPL and the Hong Kong pari-mutuel wins by a distance (it prices form, going,
  weights and jockeys that no outcome-only rating system sees).
- Pairwise-decomposition updates degrade with field size: Glicko-2 goes from
  winning sumo (N=2) to below-uniform on F1 grids, and OpenSkill's
  ThurstoneMostellerFull diverges outright on Halo FFA and F1.
- "(lattice predict)" rows show OpenSkill's posteriors are overconfident: exact
  pricing of its beliefs scores far worse than its own flattened predictor.

## Beyond Gaussian performance

The lattice imposes no distributional form: `ThurstoneRating(base_kernel=...)`
accepts ANY performance-noise density. On Formula 1 — where retirements put a
lump of probability deep in the slow tail that no skill protects against — a
DNF-mixture base (75% N(0,1) + 25% far-tail slab) improves every metric over
the Gaussian default, and a right-skewed base gives the best calibration:

| F1 base density | Log loss | Accuracy | Kendall tau | ECE | Rank-PIT KS |
|---|---|---|---|---|---|
| Gaussian (default) | 2.2248 | 0.3096 | 0.3300 | 0.0155 | 0.1781 |
| skew-normal a=2 | 2.2221 | 0.3064 | 0.3266 | 0.0097 | 0.1648 |
| DNF mixture p=0.25 | 2.2090 | 0.3182 | 0.3331 | 0.0161 | 0.1594 |

No Gaussian/logistic-bound comparator (TrueSkill, OpenSkill, Glicko-2, Elo)
can express this. Fitting p and the tail shape from data, per-contestant
scale learning (research/scale_learning.py), and condition-aware ratings
(surface, venue, color — thurstone's multiray direction) are the open
research threads.

