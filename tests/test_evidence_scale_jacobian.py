"""Evidence is a density on the contrast space, so rescaling costs.

`_cardinal_observation` scaled the observation by `lengths_scale` and
returned `log_jac = 0` on the identity path. Evidence then rose without
bound as the scale fell, and the documented
`tune_history(tune=("lengths_scale",))` selected collapse: an ordinary
four-runner example went to 2.4e-10 (#95).

The observation has n - 1 free coordinates after centring, so the `* s`
step costs exactly `(n - 1) * log(s)`, whatever transform precedes it.
"""
import numpy as np
import pytest

from winning.ratings.history import update_margins_full

M0 = np.zeros(4)
S0 = np.eye(4)
MARGINS = np.array([0.0, 1.0, 2.0, 4.0])


@pytest.mark.parametrize("scale,want", [
    (1.0, -5.98403637),
    (0.1, -10.72616665),
    (0.001, -24.51980439),
])
def test_the_reported_values(scale, want):
    got = update_margins_full(M0, S0, MARGINS, lengths_scale=scale)[-1]
    assert got == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_evidence_moves_by_exactly_n_minus_one_log_scale(n):
    """The analytic check across field sizes: the only thing a rescale
    may change is the Jacobian, so the difference is closed form."""
    rng = np.random.default_rng(n)
    margins = np.abs(rng.normal(size=n))
    margins[0] = 0.0
    base = update_margins_full(np.zeros(n), np.eye(n), margins,
                               lengths_scale=1.0)[-1]
    for s in (0.25, 0.5, 2.0, 4.0):
        got = update_margins_full(np.zeros(n), np.eye(n), margins * s,
                                  lengths_scale=1.0 / s)[-1]
        # margins scaled by s and lengths_scale by 1/s is the SAME
        # observation in performance units: y is unchanged, and only the
        # Jacobian differs, by (n-1)*log(1/s)
        assert got == pytest.approx(base + (n - 1) * np.log(1.0 / s),
                                    abs=1e-8)


def test_scores_and_margins_agree_about_the_cost():
    """Both encodings scale the observation, so both owe the term.

    The comparison has to hold the OBSERVATION fixed: changing
    lengths_scale alone moves y, so the evidence difference would carry
    the quadratic term too, not just the Jacobian. Scaling the data by
    1/s and the scale by s leaves y identical."""
    n = 5
    rng = np.random.default_rng(7)
    scores = rng.normal(size=n)
    base = update_margins_full(np.zeros(n), np.eye(n), scores=scores,
                               lengths_scale=1.0)[-1]
    for s in (0.25, 4.0):
        got = update_margins_full(np.zeros(n), np.eye(n),
                                  scores=scores * s,
                                  lengths_scale=1.0 / s)[-1]
        assert got == pytest.approx(base + (n - 1) * np.log(1.0 / s),
                                    abs=1e-8)


def test_evidence_recovers_an_interior_scale():
    """The consequence, and the reason this matters: with the term
    missing, evidence rose monotonically as the scale fell and the
    argmax sat on the grid edge. Data generated at a KNOWN scale must
    now peak there."""
    rng = np.random.default_rng(0)
    n, true_scale = 5, 0.8
    grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    total = np.zeros(len(grid))
    for _ in range(60):
        a = rng.normal(size=n)
        perf = a + rng.normal(size=n)
        lengths = (perf - perf.min()) / true_scale
        total += [update_margins_full(np.zeros(n), np.eye(n), lengths,
                                      lengths_scale=s)[-1] for s in grid]
    best = grid[int(np.argmax(total))]
    assert best == true_scale, f"argmax {best}, expected {true_scale}"
    # and it is an INTERIOR maximum, not a boundary the grid happens to end at
    assert 0 < grid.index(best) < len(grid) - 1


def test_a_transform_still_carries_its_own_derivative():
    """The g' convention is unchanged; only the scale exponent moved
    from n to n - 1, so a compressive transform is still comparable to a
    gentler one."""
    n = 4
    margins = np.array([0.0, 1.0, 2.0, 4.0])
    gentle = update_margins_full(np.zeros(n), np.eye(n), margins,
                                 lengths_scale=1.0, transform=100.0)[-1]
    sharp = update_margins_full(np.zeros(n), np.eye(n), margins,
                                lengths_scale=1.0, transform=0.5)[-1]
    assert np.isfinite(gentle) and np.isfinite(sharp)
    assert gentle != sharp, "the transform must still move the evidence"


def test_a_non_positive_scale_is_refused():
    """log(scale) is now in the answer, so zero is not merely degenerate."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="lengths_scale"):
            update_margins_full(M0, S0, MARGINS, lengths_scale=bad)
