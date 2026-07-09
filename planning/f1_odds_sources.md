# Historical F1 winner odds: verified source survey (July 2026)

Conclusion of an exhaustive hunt (Kaggle, Betfair, Oddsportal/BetExplorer/
OddsChecker, GitHub/HF/odds-APIs, academic replication data, archive.org):
NO ready-made multi-season per-driver dataset exists anywhere public.

Viable routes, in order:

1. **Wayback-archived Oddschecker winner pages** (chosen). ~370 captures,
   2007-2025, gaps 2011-12 and 2017-18 (URL migration); ~85-95 races with
   clean pre-race (<=3 day) snapshots carrying the full driver x ~22
   bookmaker grid, server-rendered, verified parseable across three page
   eras. Personal/academic use of archived pages; publish only derived
   consensus implied probabilities, never the bookmaker matrix
   (compilation/database rights). Pipeline: research/f1_odds_fetch.py.
2. **Betfair Historic "Other Sports", Basic tier (free)**: systematic
   full-field exchange prices ~2015-onward; US-geo-blocked; bz2 JSON stream
   files needing replay to last-pre-off prices; strict no-redistribution
   license. The free Kaggle zygmunt/betfair-sports week (2014 Italian GP)
   previews the schema. Layer in later from a UK/AU IP for closing-price
   quality.
3. Fragments: Henderson & Kirrane 2013 championship ante-post series
   (github.com/d-a-henderson/F1); Fahy/Butler/Butler 2016-2020 top-5 odds
   (unpublished, r.butler@ucc.ie); Polymarket championship archiver
   (OnlineAdventuress/gridodds) as a forward-collection template.

Dead ends, verified: Oddsportal has NEVER carried F1 (all motorsport paths
404; the circulating URL pattern is apocryphal); BetExplorer likewise;
Kaggle has no multi-season F1 odds; The Odds API has no motorsport; no
academic deposits of race-winner odds.

Power note: ~90 races cleanly measures the market ceiling (large effect)
but gives only wide error bars on a beat-the-market pool exponent; the
derivative-market thesis (H2H/top-N priced off Harville vs our slab
pricing) requires odds these pages lack and stays prospective.
