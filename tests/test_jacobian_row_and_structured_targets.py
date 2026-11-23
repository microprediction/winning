"""race_jacobian_row is a row of race_jacobian, and structured polish and
block inversion refuse targets that are not probabilities."""
import numpy as np
import pytest

from winning.factor.polish import race_jacobian, race_jacobian_row, polish_race
from winning.factor.structures import Blocks, Nested


@pytest.mark.parametrize("s", [1.0, 1e-18, 1e-20, 1e20])
def test_the_row_is_scale_covariant_as_the_matrix(s):
    # the row had neither the #870 rescale: 18% low at 1e-18, 19x at 1e-20
    mu, D = np.zeros(3), np.full(3, s * s)
    full = race_jacobian(mu, D=D)
    for y in range(3):
        np.testing.assert_allclose(s * race_jacobian_row(mu, y, D=D), s * full[y], rtol=1e-12, atol=1e-30)
    np.testing.assert_allclose(s * full[0], [-1 / np.sqrt(4 * np.pi), 1 / np.sqrt(16 * np.pi), 1 / np.sqrt(16 * np.pi)], rtol=1e-6)


@pytest.mark.parametrize("mu, V", [([0.0, 12 * np.sqrt(2)], None),
                                   ([0.2, -0.3], np.array([[0.5], [0.3]]))])
def test_a_normal_pair_row_is_the_closed_form(mu, V):
    # nor the #86 pair closed form: at a 12-sd gap the row was
    # [-1.06e-16, 1.5e-32], which does not sum to zero
    D = np.ones(2)
    full = race_jacobian(np.asarray(mu), V=V, D=D)
    for y in range(2):
        row = race_jacobian_row(np.asarray(mu), y, V=V, D=D)
        np.testing.assert_array_equal(row, full[y])
        assert row.sum() == 0.0


@pytest.mark.parametrize("bad", [[0.5, -0.2, 0.4, 0.3], [0.5, 0.0, 0.4, 0.3], [0.5, np.nan, 0.4, 0.3]])
@pytest.mark.parametrize("structure", [Blocks(cluster=[0, 0, 1, 1], loading=[0.5, 0.5, 0.4, 0.4], D=[1, 1, 1, 1]),
                                       Nested(cluster=[0, 0, 1, 1], loading=[0.5, 0.5, 0.4, 0.4], D=[1, 1, 1, 1],
                                              coupling=[0.3, 0.3])])
def test_structured_polish_refuses_what_the_factor_path_refuses(structure, bad):
    # Blocks floored a negative share and Nested clamped it, and both
    # reported the polish of that race converged and feasible
    with pytest.raises(ValueError, match="positive"):
        polish_race(p0=bad, structure=structure, A=np.eye(4)[:1], b=[0.45])


def test_block_inverse_refuses_a_negative_share():
    from winning.factor.blocks import abilities_from_block_race
    for bad in ([0.5, -0.2, 0.4, 0.3], [0.5, np.nan, 0.4, 0.3]):
        with pytest.raises(ValueError, match="nonnegative"):
            abilities_from_block_race(bad, [0, 0, 1, 1], [0.5, 0.5, 0.4, 0.4], [1, 1, 1, 1])
    abilities_from_block_race([0.5, 0.0, 0.3, 0.2], [0, 0, 1, 1], [0.5, 0.5, 0.4, 0.4], [1, 1, 1, 1])  # a bound, kept
