"""Classic lattice convolution and winner-fold boundaries (#404, #405, #406)."""
import numpy as np
import pytest

from winning.classic.lattice import (convolve_many, convolve_two,
                                     mean_of_density, middle_of_density,
                                     state_prices_from_densities,
                                     winner_of_many)


def _gauss(L=10, s=2.0):
    x = np.arange(-L, L + 1, dtype=float)
    d = np.exp(-0.5 * (x / s) ** 2)
    return d / d.sum()


# --- #404: the crop branch differenced a PDF twice -----------------------

def test_crop_is_a_nonnegative_density_with_the_window_mass():
    d = _gauss()
    full = np.convolve(d, d)
    cropped = np.asarray(middle_of_density(full, 10))
    assert len(cropped) == 21
    assert cropped.min() >= 0.0
    # the central 21 atoms of the 41-atom self-convolution, with the
    # lower tail lumped on the first atom
    expect = full[10:31].copy()
    expect[0] += full[:10].sum()
    np.testing.assert_allclose(cropped, expect, atol=1e-15)


@pytest.mark.parametrize("L", [None, 10, 12])
def test_convolve_two_that_crops_keeps_mass_and_mean(L):
    d = _gauss()
    c = np.asarray(convolve_two(d, d, L=L))
    assert len(c) == 2 * (10 if L is None else L) + 1
    assert c.min() >= 0.0
    assert c.sum() > 0.999
    assert abs(mean_of_density(c, 1.0)) < 1e-4


def test_convolve_many_that_crops():
    d = _gauss()
    c = np.asarray(convolve_many([d, d, d]))
    assert len(c) == 21 and c.min() >= 0.0 and c.sum() > 0.99


# --- #405: a singleton was convolved with itself -------------------------

def test_singleton_convolution_is_the_identity():
    d = np.array([0.25, 0.50, 0.25])
    np.testing.assert_allclose(convolve_many([d]), d)
    np.testing.assert_allclose(convolve_many([d], L=2, do_padding=True),
                               [0.0, 0.25, 0.5, 0.25, 0.0])
    got = np.asarray(convolve_many([d], L=2, do_padding=True))
    assert np.dot(got, np.arange(-2, 3) ** 2) == pytest.approx(0.5)


def test_singleton_convolution_narrower_lattice_crops_once():
    d = _gauss()
    got = np.asarray(convolve_many([d], L=6))
    assert len(got) == 13 and got.min() >= 0.0
    assert got.sum() > 0.99
    assert abs(mean_of_density(got, 1.0)) < 1e-4


def test_empty_convolution_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        convolve_many([])


# --- #406: singleton multiplicity stayed None ----------------------------

def test_singleton_fold_has_unit_multiplicity_and_price_one():
    d = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    density_all, mult = winner_of_many([d])
    np.testing.assert_array_equal(density_all, d)
    np.testing.assert_array_equal(mult, np.ones(5))
    assert state_prices_from_densities([d]) == pytest.approx([1.0])


def test_empty_fold_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        winner_of_many([])


def test_convolve_two_refuses_a_heavily_clipped_lower_tail():
    # #608: 30% of runner A's convolved mass lies below the window; it was
    # folded onto the edge atom, the mean correction then shifted the bulk,
    # and a 37/63 race came back 82/18 with mass exactly one.
    import numpy as np
    import pytest
    from winning.classic.lattice import convolve_two
    L = 10

    def atoms(spec):
        d = np.zeros(2 * L + 1)
        for k, p in spec:
            d[L + k] = p
        return d

    with pytest.raises(ValueError, match="increase L"):
        convolve_two(atoms([(7, .7), (-8, .3)]), atoms([(-7, .99), (-8, .01)]), L=L)
    # the other runner's convolution is inside the window and unchanged
    out = convolve_two(atoms([(-5, .9), (-3, .1)]), atoms([(2, .05), (4, .95)]), L=L)
    assert abs(float(np.sum(out)) - 1) < 1e-12
