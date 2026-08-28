"""Point-in-time rating from racing histories; team races; sub-linear
margin transforms selected by evidence."""
import numpy as np

from winning.ratings.history import rate_history, update_margins_full
from winning.ratings.teams import (update_team_margins_full,
                                   update_team_winner_full)


def _make_history(rng, n_h, n_races, drift_ts, easing=None):
    s = rng.normal(size=n_h)
    races, truth_at_end = [], None
    lam = np.exp(-1.0 / drift_ts)
    from winning.factor.races import race_probabilities
    for t in range(n_races):
        s = lam * s + np.sqrt(1 - lam ** 2) * rng.normal(size=n_h)
        runners = list(rng.choice(n_h, size=6, replace=False))
        st = s[runners]
        y_mkt = st + 0.3 * rng.normal(size=6)
        p_mkt = race_probabilities(-y_mkt)
        perf = st + rng.normal(size=6)
        gaps = perf.max() - perf                       # performance deficit
        L = gaps if easing is None else easing(gaps)   # observed lengths
        races.append(dict(t=float(t), runners=runners, p_market=p_mkt,
                          margins=L / 0.2))            # lengths at 5/unit
        truth_at_end = s.copy()
    return races, truth_at_end


def test_history_filter_tracks_drifting_abilities():
    rng = np.random.default_rng(0)
    races, s_end = _make_history(rng, n_h=10, n_races=120, drift_ts=60.0)
    ratings, logZ = rate_history(races, timescale=60.0, tau2=0.09,
                                 beta2=1.0, lengths_scale=0.2)
    m = np.array([ratings[i][0] for i in range(10)])
    sc = s_end - s_end.mean()
    mc = m - m.mean()
    corr = float(np.corrcoef(mc, sc)[0, 1])
    assert corr > 0.85
    assert np.isfinite(logZ)


def test_evidence_selects_the_sublinear_transform():
    # world with easing: observed lengths superlinear in performance
    # deficit (sinh with the same c the asinh transform inverts). The
    # Jacobian-corrected evidence must prefer the matched transform to
    # the identity, and tracking must improve.
    rng = np.random.default_rng(1)
    c = 1.5
    races, s_end = _make_history(
        rng, n_h=10, n_races=120, drift_ts=60.0,
        easing=lambda g: c * np.sinh(g / c))
    out = {}
    for name, tr in (("identity", None), ("asinh", c * 5.0)):
        # transform acts on lengths (5 per perf unit): c_lengths = 5c
        ratings, logZ = rate_history(races, timescale=60.0, tau2=0.09,
                                     beta2=1.0, lengths_scale=0.2,
                                     transform=tr)
        m = np.array([ratings[i][0] for i in range(10)])
        sc = s_end - s_end.mean()
        mc = m - m.mean()
        out[name] = (logZ, float(np.corrcoef(mc, sc)[0, 1]))
    assert out["asinh"][0] > out["identity"][0] + 10   # evidence prefers
    assert out["asinh"][1] > out["identity"][1] - 0.02 # tracking no worse


def test_team_winner_lifts_to_members_mc():
    # rider+horse pairs: 6 players, 3 teams of 2. MC referee on PLAYER
    # posteriors including cross terms.
    rng = np.random.default_rng(2)
    n, k = 6, 3
    A = np.zeros((k, n))
    for team in range(k):
        A[team, 2 * team] = 1.0
        A[team, 2 * team + 1] = 1.0
    m = rng.normal(size=n) * 0.3
    S = np.diag(0.5 + rng.random(n))
    M = 2_000_000
    s = m + rng.standard_normal((M, n)) * np.sqrt(np.diag(S))
    X = s @ A.T + rng.standard_normal((M, k))
    keep = X.argmax(axis=1) == 1
    s_w = s[keep]
    m_mc, S_mc = s_w.mean(axis=0), np.cov(s_w.T)
    m_hat, S_hat, logZ = update_team_winner_full(m, S, A, 1, beta2=1.0)
    se = np.sqrt(np.diag(S_mc) / keep.sum())
    assert np.abs(m_hat - m_mc).max() < 5 * se.max() + 5e-3
    assert np.abs(S_hat - S_mc).max() < 0.02
    assert abs(np.exp(logZ) - keep.mean()) < 5e-3


def test_team_scores_conjugate_and_uninvolved_players_untouched():
    rng = np.random.default_rng(3)
    n, k = 8, 3
    A = np.zeros((k, n))
    A[0, 0] = A[0, 1] = 1.0
    A[1, 2] = A[1, 3] = 1.0
    A[2, 4] = A[2, 5] = 1.0          # players 6, 7 not in this match
    m = np.zeros(n)
    S = np.eye(n)
    m2, S2, lz = update_team_scores = update_team_margins_full(
        m, S, A, scores=np.array([3.0, 1.0, 0.0]), beta2=1.0,
        lengths_scale=0.3)
    assert abs(m2[6]) < 1e-12 and abs(m2[7]) < 1e-12
    assert abs(S2[6, 6] - 1.0) < 1e-10
    assert m2[0] > m2[2] > m2[4]     # score order respected
    assert np.isfinite(lz)
