"""The convolution-shortcut fix: non-normal moment updates price the
TRUE predictive marginal (Gaussian belief convolved with base noise)
instead of the base at combined variance v + beta2, which is exact only
for the normal (the one base stable under convolution). Reference
numbers from 4M-sample Monte Carlo of the generative model; the full
referee lives at research/adjudications/predictive_referee.py."""
import numpy as np

from winning.ratings.nway import (_grad_logp_row, _predictive_curves,
                                  _winner_moments_curves, update_winner,
                                  update_ranking_exact)

M = np.array([0.4, 0.0, -0.3])
V = np.array([0.6, 0.4, 0.5])


def test_curves_reproduce_the_analytic_normal_path():
    """Gaussian closure is the sharpest free check of the machinery:
    with base normal the convolution IS the normal at v + beta2, so the
    curve path must reproduce the analytic path in p, gradient, and
    curvature -- every sign, kernel, and interpolation at once."""
    curves = _predictive_curves(V, 1.0, "normal")
    p_c, g_c, d2_c = _winner_moments_curves(M, curves, 0)
    g_a, p_a = _grad_logp_row(M, V + 1.0, 0, base="normal")
    assert abs(p_c - p_a) < 3e-5
    assert np.abs(g_c - g_a).max() < 3e-4
    eps = 1e-4
    for j in range(3):
        ej = np.zeros(3); ej[j] = eps
        gp, _ = _grad_logp_row(M + ej, V + 1.0, 0, base="normal")
        gm, _ = _grad_logp_row(M - ej, V + 1.0, 0, base="normal")
        assert abs(d2_c[j] - (gp[j] - gm[j]) / (2 * eps)) < 5e-4


def test_laplace_winner_update_matches_generative_mc():
    """The bug this fix closes: the shortcut priced the laplace winner
    at 0.4966 where the generative model says 0.4776, and understated
    the winner's posterior variance by 57% (0.213 vs 0.491). MC
    references: 4M x 6, se ~ 2e-4."""
    m_new, v_new, p = update_winner(M, V, 0, beta2=1.0, base="laplace")
    assert abs(p - 0.4776) < 2e-3
    assert abs(v_new[0] - 0.4909) < 4e-3          # was 0.2133 pre-fix
    assert abs(m_new[0] - 0.7327) < 3e-3


def test_tiny_belief_variance_recovers_the_pure_base_race():
    """v -> 0 is the shortcut's own valid limit: the predictive IS the
    base. The curve path must agree with the race layer there."""
    from winning.factor.races import race_probabilities
    v0 = np.full(3, 1e-9)
    curves = _predictive_curves(v0, 1.0, "laplace")
    p, _, _ = _winner_moments_curves(M, curves, 0)
    # max-wins winner 0 = min-wins on negated means
    p_ref = race_probabilities(-M, D=np.ones(3), base="laplace")[0]
    assert abs(p - p_ref) < 5e-4


def test_ranking_exact_laplace_variance_is_calibrated_not_collapsed():
    """Order-path spot check against conditioned MC (4M x 6 reference):
    the pre-fix path was overconfident (86% coverage vs 95%)."""
    m_new, v_new = update_ranking_exact(M, V, [0, 2, 1], beta2=1.0,
                                        base="laplace")
    ref_m = np.array([0.7259, -0.3149, -0.1779])
    ref_v = np.array([0.4928, 0.3456, 0.3937])
    assert np.abs(m_new - ref_m).max() < 4e-3
    assert np.abs(v_new - ref_v).max() < 4e-3
