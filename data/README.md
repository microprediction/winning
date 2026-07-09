# data

`simple.csv` — 459,504 runners across 47,651 races: Betfair starting price
(`bsp`), `finish_position`, and an anonymized `race_id`. No names, dates, or
venues. 34,092 races have both a starting price and a finish position for every
runner. Uploaded November 2021.

The rating-system benchmarks do not use this file (it has no contestant
identifiers across races); they use synthetic worlds with known ground truth
and the ATP tennis archive fetched at run time — see `src/winning/benchmarks/`.
This dataset is kept for planned market-calibration work: scoring BSP-implied
probabilities against outcomes as the ceiling any rating system chases.
