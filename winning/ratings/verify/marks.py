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
    "smoke": {"ks_n": 20_000, "ks_bases": "named", "p8_draws": 50_000,
              "referee_draws": 100_000, "referee_draws_full": 1_000_000},
    "fast": {"ks_n": 200_000, "ks_bases": "all", "p8_draws": 100_000,
             "referee_draws": 200_000, "referee_draws_full": 1_000_000},
    "full": {"ks_n": 1_000_000, "ks_bases": "all", "p8_draws": 400_000,
             "referee_draws": 2_000_000, "referee_draws_full": 2_000_000},
    "exhaustive": {"ks_n": 1_000_000, "ks_bases": "all", "p8_draws": 4_000_000,
                   "referee_draws": 4_000_000, "referee_draws_full": 4_000_000},
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

    # enumerated moment identities: max-abs residual of the sums over
    # all outcomes weighted by the engine's own P(y); I2 relative to v,
    # I1 and I3 absolute. Exact up to lattice mass and the curvature
    # method's own error.
    "identity.moments": {
        # I1 (score identity) and I3 (normalisation): lattice mass only.
        # I2 (Bartlett) by curvature method -- analytic (update_winner's
        # curve path), fd (absolute central differences at eps 1e-3 in
        # nway._mixture_update / update_ranking_exact), fd_rel (0.15 sd
        # steps in full._mixture_update_full). Pilots 2026-09-11 (K3-K5):
        # normal analytic/fd winner 2e-11 / 2e-9 (8e-9 at beta2 0.25); fd_rel winner 1.1e-4..2.4e-4;
        # curve analytic winner 4e-6, fd 1.4e-5..1.8e-4 (student4 K5), fd_rel orders
        # 1.9e-4..2.3e-4; orders normal fd 2.2e-5, curve fd 9.9e-5.
        "floors": {"I1.normal.winner": 1e-8, "I3.normal.winner": 1e-8,
                   "I1.curve.winner": 5e-5, "I3.curve.winner": 5e-5,
                   "I1.normal.order": 1e-4, "I3.normal.order": 1e-4,
                   "I1.curve.order": 5e-4, "I3.curve.order": 5e-4,
                   "I2.normal.analytic.winner": 1e-8,
                   "I2.normal.fd.winner": 5e-8,
                   "I2.normal.fd_rel.winner": 5e-4,
                   "I2.curve.analytic.winner": 5e-5,
                   "I2.curve.fd.winner": 5e-4,
                   "I2.curve.fd_rel.winner": 5e-4,
                   "I2.normal.fd.order": 1e-4,
                   "I2.normal.fd_rel.order": 5e-4,
                   "I2.curve.fd.order": 5e-4,
                   "I2.curve.fd_rel.order": 1e-3},
        "set_by": "planning pilots", "date": "2026-09-11",
        "basis": "statistics design pass and smoke/fast pilots; floors at "
                 "3-10x the measured residual of each curvature method"},

    # exact reductions: bit-level agreement of paths that share the
    # same lattice (V = 0 mixtures run one node; market full/diag are
    # the same conjugate algebra)
    "identity.reductions.exact": {"tolerance": 1e-9, "set_by": "planning pilots",
                                  "date": "2026-09-11",
                                  "basis": "pinned at 1e-12 in tests/test_correlated_updates.py"},
    # V = 0 correlated vs independent VARIANCES: the independent winner
    # update's curvature is analytic on the curve path (eps 1e-4
    # differences under normal) while the mixture differences at 1e-3;
    # measured normal 4.2e-9, curve up to 4.2e-4 (student4)
    "identity.reductions.v0_v.normal": {"tolerance": 5e-8, "set_by": "planning pilots",
                                        "date": "2026-09-11", "basis": "two FD steps, 4.2e-9"},
    "identity.reductions.v0_v.curve": {"tolerance": 1e-3, "set_by": "planning pilots",
                                       "date": "2026-09-11",
                                       "basis": "analytic vs FD curvature, up to 4.2e-4"},
    # full-covariance path with a diagonal belief vs the diagonal path:
    # same gradient under normal (m exact to lattice); on curve bases
    # the batched pass sizes a finer lattice than the single pass, so m
    # and evidence differ by the single pass's discretisation (measured
    # 2e-5..2.5e-4, v up to 3.6e-3 on the failure lump); curvature by a
    # 0.15 sd step against an absolute 1e-3 step
    "identity.reductions.full_m.normal": {"tolerance": 1e-5, "set_by": "planning pilots",
                                          "date": "2026-09-11",
                                          "basis": "tests/test_bases.py pins atol 1e-5"},
    "identity.reductions.full_m.curve": {"tolerance": 5e-4, "set_by": "planning pilots",
                                         "date": "2026-09-11",
                                         "basis": "single-pass lattice at L=2001 on wide spans"},
    "identity.reductions.full_v.normal": {"tolerance": 1e-3, "set_by": "planning pilots",
                                          "date": "2026-09-11",
                                          "basis": "eps_rel FD truncation ~1e-4"},
    "identity.reductions.full_v.curve": {"tolerance": 5e-3, "set_by": "planning pilots",
                                         "date": "2026-09-11",
                                         "basis": "lattice discretisation plus FD; 3.6e-3 on failure"},
    "identity.reductions.full_evidence.normal": {"tolerance": 1e-6, "set_by": "planning pilots",
                                                 "date": "2026-09-11",
                                                 "basis": "same lattice mass; 801 vs 3001 field points"},
    "identity.reductions.full_evidence.curve": {"tolerance": 5e-4, "set_by": "planning pilots",
                                                "date": "2026-09-11",
                                                "basis": "single-pass lattice mass on wide spans, 2.5e-4"},
    # correlated with V through the two paths: Gauss-Hermite 7 vs 9 nodes
    "identity.reductions.full_vs_correlated_V": {"tolerance": 1e-3, "set_by": "planning pilots",
                                                 "date": "2026-09-11",
                                                 "basis": "different node rules; measured 5.5e-5, 1.3e-4"},
    "identity.closed_form_k2.m": {"tolerance": 1e-8, "set_by": "planning pilots",
                                  "date": "2026-09-11",
                                  "basis": "dm/sd 2e-15 on every path at prior sd 1..30"},
    "identity.closed_form_k2.v.analytic": {"tolerance": 1e-8, "set_by": "planning pilots",
                                           "date": "2026-09-11", "basis": "dv/v 1e-11"},
    "identity.closed_form_k2.v.fd": {"tolerance": 1e-5, "set_by": "planning pilots",
                                     "date": "2026-09-11",
                                     "basis": "absolute FD step 1e-3, truncation O(eps^2)"},
    "identity.closed_form_k2.v.fd_rel": {"tolerance": 1e-3, "set_by": "planning pilots",
                                         "date": "2026-09-11",
                                         "basis": "dv/v 5e-5..3.4e-4 at prior sd 1..30, beta2 0.25..4"},
    "identity.closed_form_k2.logp": {"tolerance": 1e-8, "set_by": "planning pilots",
                                     "date": "2026-09-11", "basis": "lattice mass at K=2"},
    "identity.coarsening.loglik": {"tolerance": 1e-4, "set_by": "planning pilots",
                                   "date": "2026-09-11",
                                   "basis": "tests/test_bases.py pins 1e-4"},
    "identity.coarsening.m": {"tolerance": 1e-4, "set_by": "planning pilots",
                              "date": "2026-09-11", "basis": "lattice mass 1e-5, 10x"},
    "identity.coarsening.v": {"tolerance": 1e-3, "set_by": "planning pilots",
                              "date": "2026-09-11", "basis": "lattice mass plus FD 1e-6, 10x"},
    "identity.coarsening.exact": {"tolerance": 0.0, "set_by": "planning pilots",
                                  "date": "2026-09-11",
                                  "basis": "the omitted entrant's row is never touched"},
    "identity.coarsening.subfield": {"tolerance": 1e-5, "set_by": "planning pilots",
                                     "date": "2026-09-11",
                                     "basis": "lattice window sized by all means; same mass"},
    "identity.invariances.gauge.diag": {"tolerance": 1e-9, "set_by": "planning pilots",
                                        "date": "2026-09-11",
                                        "basis": "lattice windows follow the means exactly"},
    "identity.invariances.gauge.full": {"tolerance": 1e-6, "set_by": "planning pilots",
                                        "date": "2026-09-11",
                                        "basis": "batched lattice endpoints round with the shift; 6.5e-8"},
    "identity.invariances.permutation.diag": {"tolerance": 1e-9, "set_by": "planning pilots",
                                              "date": "2026-09-11",
                                              "basis": "symmetric node sets; same lattice"},
    "identity.invariances.permutation.full": {"tolerance": 1e-6, "set_by": "planning pilots",
                                              "date": "2026-09-11",
                                              "basis": "eigendecomposition split, equivariant to fp"},
    "identity.invariances.scale_m": {"tolerance": 1e-8, "set_by": "planning pilots",
                                     "date": "2026-09-11",
                                     "basis": "lattice built in sd units; evidence scale-free"},
    # variances under scaling: the full path differences at relative
    # steps and is covariant to fp (6.5e-8); the diagonal paths use
    # ABSOLUTE steps (1e-4 winner, 1e-3 orders), so their curvature
    # error grows as the scale shrinks (1.6e-4 at c = 0.1 on logistic
    # orders) -- a documented cost, candidate for relative steps
    "identity.invariances.scale_v.full": {"tolerance": 1e-6, "set_by": "planning pilots",
                                          "date": "2026-09-11", "basis": "relative FD steps"},
    "identity.invariances.scale_v.diag": {"tolerance": 1e-3, "set_by": "planning pilots",
                                          "date": "2026-09-11",
                                          "basis": "absolute FD steps; 1.6e-4 at c=0.1"},
    "identity.invariances.exact": {"tolerance": 1e-12, "set_by": "planning pilots",
                                   "date": "2026-09-11", "basis": "identical code path"},
    "identity.predict_evidence.normal": {"tolerance": 1e-8, "set_by": "planning pilots",
                                         "date": "2026-09-11", "basis": "pilot 1e-16"},
    "identity.predict_evidence.blocked": {"tolerance": 1e-5, "set_by": "planning pilots",
                                          "date": "2026-09-11",
                                          "basis": "3.4e-6 with three groups (Sobol vs engine nodes)"},
    "identity.evidence.exact": {"tolerance": 1e-8, "set_by": "planning pilots",
                                "date": "2026-09-11",
                                "basis": "tests/test_history_and_teams.py pins 1e-8 / 1e-10"},

    # the ported audit (bandits thresholds, pre-registered there)
    "audit.P2": {"tolerance": 1e-6, "set_by": "bandits audit", "date": "2026-09-04",
                 "basis": "Prekopa: variance is non-increasing on log-concave bases"},
    "audit.P4": {"tolerance": 0.5, "set_by": "bandits audit", "date": "2026-09-04",
                 "basis": "corr(m, truth) after 40 updates of 8 arms"},
    "audit.P6": {"thresholds": {"robust": 0.35, "tail": 0.01, "coverage": 0.12},
                 "set_by": "bandits audit", "date": "2026-09-04",
                 "basis": "|robust sd - 1|, frac |z| > 10, |coverage - 0.95|"},
    "audit.P7": {"tolerance": 0.05, "set_by": "bandits audit", "date": "2026-09-04",
                 "basis": "max mean spread across paths on identical evidence"},
    "audit.P8": {"dm": 0.005, "dv": 0.02, "set_by": "bandits audit + fresh configs",
                 "date": "2026-09-04",
                 "basis": "rejection MC; dm mark 0.005 + 4 SE (audit_fresh_configs), dv 0.02"},

    # Monte Carlo referees: the referee's own marks (research/
    # adjudications/predictive_referee.py and bandits audit_fresh_configs),
    # plus 4 SE from the accepted sample
    "referee.mean": {"tolerance": 0.005, "set_by": "predictive_referee + fresh configs",
                     "date": "2026-09-04", "basis": "max |dm| <= 0.005 + 4 SE"},
    "referee.var": {"tolerance": 0.006, "set_by": "predictive_referee",
                    "date": "2026-09-04", "basis": "max |dv| / v <= 0.006 + 4 SE (relative)"},
    "referee.cross": {"tolerance": 0.02, "set_by": "tests/test_full_covariance_updates.py",
                      "date": "2026-09-04", "basis": "cross terms within 0.02 + 4 SE"},
}

# documented approximation costs: name -> {"envelope", "cite", "why"};
# a statistic inside the envelope reports EXPECTED_APPROX, outside FAILs
EXPECTED_APPROX = {
    # update_ranking is a stagewise APPROXIMATION whose overconfidence
    # on non-IIA bases is documented and measured (12 seeds x 60
    # updates against update_ranking_exact on identical evidence):
    #   base      var ratio  mean err ratio  coverage
    #   gumbel      0.977        1.002         0.925   <- IIA: exact
    #   normal      0.765        1.202         0.890
    #   logistic    0.775        1.226         0.870
    #   laplace     0.794        1.229         0.850
    #   student4    0.794        1.254         0.830
    # A near-constant 21-28 percent over-shrinkage plus a point estimate
    # that degrades with tail weight. The envelope bounds it: robust sd
    # in [0.9, 1.6], coverage in [0.78, 0.95]; beyond it, FAIL.
    "audit.P6.update_ranking": {
        "envelope": {"robust": (0.9, 1.6), "coverage": (0.78, 0.95)},
        "cite": "winning/ratings/nway.py update_ranking docstring; "
                "research/adjudications/laplace_convolution_shortcut.md",
        "why": "stagewise decomposition: documented approximation cost"},
}
