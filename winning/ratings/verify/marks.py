"""Pass marks, seed counts, sample sizes and documented approximation
envelopes for winning.ratings.verify -- the ONLY place a tolerance
lives. Every entry says who set it, when, and on what basis, and the
report records this file's hash so a before/after pair with a moved
mark is visible. Marks are set before a fix exists (the acceptance
criterion precedes the repair) and a fix commit does not touch this
file.

Seeds: ROOT_SEED derives every check's generator (core.seed_for);
CONFIG_SEED draws the FULL-profile referee configurations, bumped by the
adjudicator when a "before" run is recorded so the implementer neither
picks nor sees them first.
"""

ROOT_SEED = 20260911
CONFIG_SEED = 20260911

# fixed calibration seed counts per driver: no time box (a budget that
# could cut below the floor made a cell's verdict depend on machine
# load; the correlated drivers are both the slowest and the noisiest,
# so the two failure modes compounded exactly where they did the most
# damage -- bandits audit, 2026-09-11). Slow drivers get 14, the rest 25.
SEEDS = {"update_winner_correlated": 14, "update_order_correlated": 14,
         "update_order_full": 14}
DEFAULT_SEEDS = 25

# per-profile sample sizes and grid switches
PARAMS = {
    "smoke": {"ks_n": 20_000, "ks_bases": "named"},
    "fast": {"ks_n": 200_000, "ks_bases": "all"},
    "full": {"ks_n": 1_000_000, "ks_bases": "all"},
    "exhaustive": {"ks_n": 1_000_000, "ks_bases": "all"},
}

# check name -> {"tolerance", "set_by", "date", "basis"}
MARKS = {
    # Kolmogorov distance of the sampler against its own survival at the
    # profile's n; DKW: P(D > 1.36/sqrt(n)) ~ 0.05, 1.63/sqrt(n) ~ 0.01
    # (at n = 2e4: 1.15e-2; 2e5: 3.6e-3; 1e6: 1.6e-3). Set at 2 x the
    # 0.01 quantile so a sign or scale slip (> 0.1) cannot hide.
    "simulate.sampler_matches_density": {
        "tolerance_by_n": {20_000: 2.4e-2, 200_000: 7.5e-3, 1_000_000: 3.3e-3},
        "set_by": "planning pilots", "date": "2026-09-11",
        "basis": "DKW 0.01 quantile x 2; pilots at n=2e5: 1.3e-3..2.7e-3"},
}

# documented approximation costs: name -> {"envelope", "cite", "why"};
# a statistic inside the envelope reports EXPECTED_APPROX, outside FAILs
EXPECTED_APPROX = {}
