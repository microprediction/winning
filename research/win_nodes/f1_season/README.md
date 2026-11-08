# An F1 season as a win-node network: every kind of evidence on one belief

Pace per driver per weekend. Per weekend the evidence is folded in through
`winning.ratings.market.update_race` on the weekend's field with team block
loadings (`block_loadings(groups, rho=0.5, beta2)`), in this order: the
pre-qualifying winner market (quoted drivers only, inverted to abilities,
`tau2=0.3`), the Q1 survivors as a top set, the ordered Q3 prefix with the
Q1 casualties withdrawn (untimed Q3 drivers withdrawn, not placed behind),
any post-qualifying market snapshot, then the race classification with
retirements withdrawn. A random walk (`drift=0.05` per round) carries each
driver between weekends. Qualifying noise `beta2_q=4`, race noise 1, 256
Sobol nodes. Scored before each race by the log probability of the actual
winner; the market's own consensus probability of the winner is recorded
on the races that have a pre-race snapshot.

Data: results, grids and Q1/Q2/Q3 times from the cached f1db zip
(`f1db_load.py`; the qualifying classification decides who reached Q2/Q3, `Q_BY_TIMES=1` restores the old time-based reading); winner odds from Wayback captures of
Oddschecker via `research/f1_odds_fetch.py`, parsed by `parse_all.py` to
`f1_market_all.csv` (derived consensus probabilities only), joined by
`f1_market_join.py`. 2013-2015 are the seasons with 25-30 snapshots each.

    BELIEF=diag BETA2_Q=4 TAG=_diag_bq4 NODES=8 \
      python run_f1_season.py 2014 race race+quali race+market race+quali+market

## Results (`season_*_diag_bq4.json`, per-round logs in `rounds/`; new runs write a timestamped summary and carry their parameters in every row)

Log score of the race winner, all 19 races; in brackets the model on the
races with a pre-race market against the market itself.

| evidence | 2013 (14) | 2014 (15) | 2015 (17) |
|---|---|---|---|
| results only | -1.41 (-1.54 vs -1.26) | -1.53 (-1.52 vs -1.26) | -1.53 (-1.56 vs -1.19) |
| + qualifying | -1.11 (-1.24 vs -1.26) | -1.38 (-1.41 vs -1.26) | -1.19 (-1.23 vs -1.19) |
| + market, no qualifying | -1.32 (-1.34 vs -1.26) | -1.34 (-1.28 vs -1.26) | -1.19 (-1.23 vs -1.19) |
| + qualifying + market | -1.26 (-1.31 vs -1.26) | -1.32 (-1.27 vs -1.26) | -1.14 (-1.190 vs -1.191) |

Qualifying evidence helps every season. The market helps in 2014 and 2015
and hurts in 2013, where the pre-qualifying market was no better than the
model and `tau2=0.3` trusts it too much: the market noise should be set by
the evidence per season, which is the next step (tune on one season,
validate on the others; evidence and held-out log score disagreed on
`beta2_q`, 2 against 4).

## Two faults found on the way

1. Data: a Q3 participant who set no time (tyre saving, common in 2013) was
   read from the times as eliminated in Q2 and placed behind drivers he had
   out-qualified. The classification decides who reached Q2/Q3; untimed Q3
   drivers are top ten without an order and are withdrawn from the prefix.
2. Engine: the dense `GaussianNet` path over-tightened the belief at a
   quadrature rank of about 20 (team loadings plus the belief's own split),
   collapsing a driver's sd from 0.85 to 0.31 after one weekend. Issue #1027
   has the reproduction. `season_*_dense_*.json` keep those runs for the
   record; the table above is the diagonal path, which is exact per
   coordinate and matched Monte Carlo referees.

Caveat: most odds snapshots are pre-qualifying, so a model that also sees
Saturday has an information advantage over the market it is compared with;
the "results + market" row is the same-information comparison.

## Market noise, by evidence and by held-out score (`tau2_sweep/`)

The same run (results + qualifying + market, diagonal path, `beta2_q=4`)
at three values of the market noise `tau2`. Total evidence, and the log
score of the race winner over all 19 races:

| `tau2` | 2013 | 2014 | 2015 |
|---|---|---|---|
| 0.1 | -1168 / -1.28 | -1062 / -1.316 | -982 / -1.150 |
| 0.3 | -1348 / -1.26 | -1265 / -1.317 | -1168 / -1.142 |
| 1.0 | -1580 / -1.23 | -1546 / -1.318 | -1423 / -1.138 |

The evidence prefers trusting the market most in every season; held-out
prediction is flat in 2014 and slightly prefers trusting it less in 2013
and 2015 (the disagreement CLAUDE.md warns about). Over the three values
tested, the market helps in 2014 and 2015 at each and hurts in 2013 at
each, so within that range the 2013 effect is not the noise level.

A hypothesis, not established by this sweep: a pre-qualifying price
already encodes expected qualifying form, so folding in the price and then
Saturday's result counts that information twice. The network assumes the
price and the qualifying outcome are conditionally independent given pace.
Two observations are consistent with it but do not test it: in 2013 the
market alone improves on results alone (-1.32 against -1.41) while market
plus qualifying is worse than qualifying alone (-1.26 against -1.11), and
2013 is the season in which qualifying and race pace diverged most. A test
would observe the market after qualifying only, or model the market's
information set explicitly; the post-qualifying snapshots (3 to 4 per
season) are too few here.
