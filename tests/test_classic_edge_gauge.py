"""A common translation across the lattice edge keeps the contrast (#585)."""
import numpy as np
import pytest

from winning.classic.lattice import skew_normal_density, state_prices_from_offsets, _edge_centred


@pytest.mark.parametrize("c", [0, 11, 20, -20, 40, -55])
def test_translated_book_prices_the_same(c):
    # each offset past +/-(L-2) was clamped on its own: [17, 23] priced
    # [0.5, 0.5] at L=15 where [-3, 3] priced [0.98, 0.02]
    d = skew_normal_density(15, 0.5, a=0)
    base = state_prices_from_offsets(d, [-3.0, 3.0, 1.0])
    np.testing.assert_allclose(state_prices_from_offsets(d, [-3.0 + c, 3.0 + c, 1.0 + c]), base, atol=1e-12)
    assert base[0] > 0.9


def test_in_range_books_are_untouched():
    assert _edge_centred([-13.0, 0.5, 13.0], 15) == [-13.0, 0.5, 13.0]
    assert _edge_centred([17.0, 23.0, float("inf")], 15) == [-3.0, 3.0, float("inf")]
