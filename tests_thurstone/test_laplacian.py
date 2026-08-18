import numpy as np
import pytest

from winning.thurstone.density import Density
from winning.thurstone.laplacian import (
    LaplacianOperator,
    invert_outright_probabilities,
    laplacian_dense,
    laplacian_matvec,
    laplacian_weights,
    outright_win_probabilities,
)
from winning.thurstone.lattice import UniformLattice

LATTICE = UniformLattice(L=400, unit=0.05)


# ---- field builders ----


def _smooth(abilities, lattice=LATTICE, scales=None, skews=None):
    n = len(abilities)
    scales = scales or [1.0] * n
    skews = skews or [0.0] * n
    out = []
    for a, s, k in zip(abilities, scales, skews):
        base = Density.skew_normal(lattice, loc=0.0, scale=s, a=k)
        out.append(base.shift_fractional(a / lattice.unit))
    return out


def _atom(x_phys, lattice=LATTICE):
    p = np.zeros(lattice.size)
    p[int(round(x_phys / lattice.unit)) + lattice.L] = 1.0
    return Density(lattice, p)


def _zero_mass(lattice=LATTICE):
    return Density(lattice, np.zeros(lattice.size))


def _two_point(x0, x1, delta, lattice=LATTICE):
    """Mass 1-delta at x0 and delta at x1: adversarial near-zero survival."""
    p = np.zeros(lattice.size)
    p[int(round(x0 / lattice.unit)) + lattice.L] = 1.0 - delta
    p[int(round(x1 / lattice.unit)) + lattice.L] = delta
    return Density(lattice, p)


# Smooth fields: differentiable forward map, so finite differences apply.
SMOOTH_CASES = {
    "baseline": dict(abilities=[0.0, 0.35, -0.2, 0.6, 0.1]),
    "pair": dict(abilities=[0.0, 0.4]),
    "identical": dict(abilities=[0.0, 0.0, 0.0, 0.0]),
    "near_equal": dict(abilities=[0.0, 1e-4, -1e-4]),
    "extreme_separation": dict(abilities=[-6.0, 0.0, 6.0]),
    "mixed_scales": dict(
        abilities=[0.0, 0.3, -0.4],
        scales=[0.5, 1.0, 2.0],
        skews=[0.0, 0.8, -0.5],
    ),
}


def _large_random(lattice=LATTICE):
    rng = np.random.default_rng(0)
    abilities = rng.normal(scale=0.6, size=25)
    scales = rng.choice([0.6, 1.0, 1.6], size=25).tolist()
    return _smooth(abilities.tolist(), lattice, scales=scales)


# Every field the machinery must handle, degenerate cases included.
FIELD_BUILDERS = {
    **{
        name: (lambda kw: lambda lat=LATTICE: _smooth(**kw, lattice=lat))(kw)
        for name, kw in SMOOTH_CASES.items()
    },
    "atom_vs_smooth": lambda lat=LATTICE: [_atom(0.1, lat)] + _smooth([0.0, 0.3], lat),
    "two_atoms_same_point": lambda lat=LATTICE: (
        [
            _atom(0.0, lat),
            _atom(0.0, lat),
        ]
        + _smooth([0.2], lat)
        + _smooth([-0.1], lat)
    ),
    "atoms_apart": lambda lat=LATTICE: [_atom(-0.5, lat), _atom(0.5, lat)] + _smooth([0.0], lat),
    "pure_atom_pair": lambda lat=LATTICE: [_atom(0.0, lat), _atom(0.3, lat)],
    "zero_mass_runner": lambda lat=LATTICE: [_zero_mass(lat)] + _smooth([0.0, 0.3], lat),
    "all_zero_mass": lambda lat=LATTICE: [_zero_mass(lat), _zero_mass(lat)],
    "off_lattice_shift": lambda lat=LATTICE: (
        [_smooth([0.0], lat)[0].shift_integer(2 * lat.L + 2)] + _smooth([0.0, 0.3], lat)
    ),
    "edge_pileup": lambda lat=LATTICE: _smooth(
        [-(lat.L - 20) * lat.unit, (lat.L - 20) * lat.unit, 0.0], lat
    ),
    "tiny_survival_masked": lambda lat=LATTICE: (
        [_two_point(0.0, 0.5, 1e-7, lat)] + _smooth([0.0, 0.2], lat)
    ),
    "tiny_survival_unmasked": lambda lat=LATTICE: (
        [_two_point(0.0, 0.5, 1e-5, lat)] + _smooth([0.0, 0.2], lat)
    ),
    "atom_left_edge": lambda lat=LATTICE: [_atom(-lat.L * lat.unit, lat)] + _smooth([0.0], lat),
    "atom_right_edge": lambda lat=LATTICE: [_atom(lat.L * lat.unit, lat)] + _smooth([0.0], lat),
    "identical_atom_pair": lambda lat=LATTICE: [_atom(0.0, lat), _atom(0.0, lat)],
    "identical_atom_trio": lambda lat=LATTICE: [_atom(0.0, lat)] * 3,
    "bimodal_gap": lambda lat=LATTICE: [
        Density(lat, _smooth([-3.0], lat)[0].p + _smooth([3.0], lat)[0].p),
        *_smooth([0.0, 0.5], lat),
    ],
    "zero_vs_smooth_pair": lambda lat=LATTICE: [_zero_mass(lat)] + _smooth([0.0], lat),
    "large_random": _large_random,
}

FIELD_NAMES = list(FIELD_BUILDERS)


# ---- matvec vs dense: the algebraic identity, on every field ----


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_matvec_matches_dense(name):
    field = FIELD_BUILDERS[name]()
    L = laplacian_dense(field)
    rng = np.random.default_rng(7)
    scale = max(np.abs(L).max(), 1.0)
    for _ in range(3):
        u = rng.normal(size=len(field))
        got = laplacian_matvec(field, u)
        want = L @ u
        assert np.allclose(got, want, rtol=1e-7, atol=1e-9 * scale), (
            f"{name}: max diff {np.abs(got - want).max():.3e}"
        )


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_matvec_linearity_and_scaling(name):
    field = FIELD_BUILDERS[name]()
    rng = np.random.default_rng(11)
    n = len(field)
    u, v = rng.normal(size=n), rng.normal(size=n)
    lhs = laplacian_matvec(field, 2.5 * u - 3.0 * v)
    rhs = 2.5 * laplacian_matvec(field, u) - 3.0 * laplacian_matvec(field, v)
    ref = np.abs(lhs).max() + np.abs(rhs).max() + 1.0
    assert np.allclose(lhs, rhs, atol=1e-10 * ref)
    # huge-magnitude u must not degrade agreement with the dense form
    big = 1e8 * u
    assert np.allclose(
        laplacian_matvec(field, big), laplacian_dense(field) @ big, rtol=1e-7, atol=1e-2
    )


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_constant_vector_annihilated_exactly(name):
    field = FIELD_BUILDERS[name]()
    out = laplacian_matvec(field, np.ones(len(field)))
    assert np.all(out == 0.0)


# ---- structural properties of the dense form, on every field ----


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_weights_symmetric_nonnegative(name):
    field = FIELD_BUILDERS[name]()
    W = laplacian_weights(field)
    assert np.all(np.isfinite(W))
    assert np.allclose(W, W.T)
    assert np.all(np.diag(W) == 0.0)
    assert np.all(W >= 0.0)


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_laplacian_psd_with_null_vector(name):
    field = FIELD_BUILDERS[name]()
    L = laplacian_dense(field)
    n = len(field)
    scale = max(np.abs(L).max(), 1.0)
    assert np.allclose(L @ np.ones(n), 0.0, atol=1e-12 * scale)
    eig = np.linalg.eigvalsh(L)
    assert eig[0] > -1e-10 * scale


def test_baseline_strictly_connected():
    """Overlapping smooth runners: all weights positive, spectral gap open."""
    field = FIELD_BUILDERS["baseline"]()
    W = laplacian_weights(field)
    off = W[~np.eye(len(field), dtype=bool)]
    assert np.all(off > 0.0)
    eig = np.linalg.eigvalsh(laplacian_dense(field))
    assert eig[1] > 0.0


# ---- degenerate-field semantics ----


def test_zero_mass_runner_disconnects():
    """Off-lattice sentinel: zero row/col, second null vector, others intact."""
    field = FIELD_BUILDERS["zero_mass_runner"]()
    W = laplacian_weights(field)
    assert np.all(W[0, :] == 0.0) and np.all(W[:, 0] == 0.0)
    L = laplacian_dense(field)
    e0 = np.zeros(len(field))
    e0[0] = 1.0
    assert np.all(L @ e0 == 0.0)
    assert np.all(laplacian_matvec(field, e0) == 0.0)
    assert outright_win_probabilities(field)[0] == 0.0
    # the smooth pair must be unaffected by the spectator
    sub = laplacian_weights(field[1:])
    assert np.allclose(W[1:, 1:], sub, rtol=1e-12)


def test_all_zero_mass():
    field = FIELD_BUILDERS["all_zero_mass"]()
    assert np.all(laplacian_weights(field) == 0.0)
    assert np.all(laplacian_matvec(field, np.array([1.0, -1.0])) == 0.0)
    assert np.all(outright_win_probabilities(field) == 0.0)


def test_two_atoms_same_point_never_win_outright():
    field = FIELD_BUILDERS["two_atoms_same_point"]()
    p = outright_win_probabilities(field)
    assert p[0] == 0.0 and p[1] == 0.0  # they always tie each other


def test_atoms_apart_dominated_runner_is_isolated():
    """An atom deterministically beaten by another has all weights zero.

    The atom at +0.5 can never finish before the atom at -0.5, so its win
    probability is frozen at 0: every derivative involving it vanishes and
    it becomes an isolated vertex of the graph (a second null direction).
    """
    field = FIELD_BUILDERS["atoms_apart"]()
    W = laplacian_weights(field)
    assert W[0, 1] == 0.0  # no overlap between the atoms
    assert W[0, 2] > 0.0  # the early atom still interacts with the smooth runner
    assert np.all(W[1, :] == 0.0)  # the dominated atom is isolated
    L = laplacian_dense(field)
    e1 = np.array([0.0, 1.0, 0.0])
    assert np.all(L @ e1 == 0.0)
    assert np.all(laplacian_matvec(field, e1) == 0.0)
    assert outright_win_probabilities(field)[1] == 0.0


def test_identical_atoms_tie_slope():
    """Two identical atoms always tie; the discrete tie-splitting slope is 1/unit.

    The continuum map is not differentiable at an atomic tie (p jumps as one
    atom moves off it); on the lattice this shows up as w = 1/unit, diverging
    as the grid refines.  A third coincident atom kills every pair product,
    so the trio's weights vanish entirely.
    """
    pair = FIELD_BUILDERS["identical_atom_pair"]()
    assert laplacian_weights(pair)[0, 1] == pytest.approx(1.0 / LATTICE.unit)
    trio = FIELD_BUILDERS["identical_atom_trio"]()
    assert np.all(laplacian_weights(trio) == 0.0)


def test_zero_mass_vs_smooth_pair():
    """A lone real runner against the off-lattice sentinel wins with certainty."""
    p = outright_win_probabilities(FIELD_BUILDERS["zero_vs_smooth_pair"]())
    assert p[0] == 0.0
    assert p[1] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("name", FIELD_NAMES)
def test_outright_probabilities_bounds(name):
    p = outright_win_probabilities(FIELD_BUILDERS[name]())
    assert np.all(p >= 0.0)
    assert p.sum() <= 1.0 + 1e-12


def test_translation_invariance():
    """p(a + c 1) = p(a) up to lattice edge effects."""
    kw = SMOOTH_CASES["baseline"]
    p0 = outright_win_probabilities(_smooth(**kw))
    p1 = outright_win_probabilities(_smooth([a + 0.25 for a in kw["abilities"]]))
    assert np.allclose(p0, p1, atol=1e-10)


# ---- finite differences: the calculus, on every smooth field ----


def _fd_jacobian(kw, lattice):
    abilities = kw["abilities"]
    n = len(abilities)
    eps = lattice.unit  # one lattice step: spans the piecewise-linear kink
    J = np.zeros((n, n))
    for j in range(n):
        up = dict(kw, abilities=[a + eps * (k == j) for k, a in enumerate(abilities)])
        dn = dict(kw, abilities=[a - eps * (k == j) for k, a in enumerate(abilities)])
        J[:, j] = (
            outright_win_probabilities(_smooth(**up, lattice=lattice))
            - outright_win_probabilities(_smooth(**dn, lattice=lattice))
        ) / (2 * eps)
    return J


@pytest.mark.parametrize("name", list(SMOOTH_CASES))
def test_finite_difference_jacobian(name):
    """Central differences of the forward map recover -L(w) to O(unit)."""
    kw = SMOOTH_CASES[name]
    L = laplacian_dense(_smooth(**kw))
    J = _fd_jacobian(kw, LATTICE)
    assert np.allclose(J, -L, atol=1e-2)
    mask = np.abs(L) > 1e-3
    if mask.any():
        assert np.max(np.abs((J + L)[mask] / L[mask])) < 5e-2


@pytest.mark.parametrize("name", list(SMOOTH_CASES))
def test_finite_difference_jacobian_converges(name):
    """The FD-vs-analytic gap is discretization error: it halves with the unit."""
    kw = SMOOTH_CASES[name]
    err = []
    for half_width, unit in [(400, 0.05), (800, 0.025)]:
        lattice = UniformLattice(L=half_width, unit=unit)
        L = laplacian_dense(_smooth(**kw, lattice=lattice))
        err.append(np.abs(_fd_jacobian(kw, lattice) + L).max())
    if err[0] < 1e-9:
        return  # already at floating-point floor (fully separated fields)
    assert err[1] < 0.6 * err[0]


def test_directional_derivative_large_field():
    """FD directional derivative matches -L u at n = 25 without forming J."""
    rng = np.random.default_rng(5)
    abilities = np.random.default_rng(0).normal(scale=0.6, size=25)
    scales = np.random.default_rng(0).choice([0.6, 1.0, 1.6], size=25).tolist()
    u = rng.normal(size=25)
    eps = LATTICE.unit
    p_up = outright_win_probabilities(_smooth((abilities + eps * u).tolist(), scales=scales))
    p_dn = outright_win_probabilities(_smooth((abilities - eps * u).tolist(), scales=scales))
    fd = (p_up - p_dn) / (2 * eps)
    lu = laplacian_matvec(_smooth(abilities.tolist(), scales=scales), u)
    assert np.allclose(fd, -lu, atol=2e-2 * max(np.abs(u).max(), 1.0))


# ---- continuum anchor: closed form for a Gaussian pair ----


def test_gaussian_pair_matches_closed_form():
    """w_12 for two unit normals is exp(-d^2/4) / (2 sqrt(pi)).

    Trapezoidal quadrature of smooth, rapidly decaying integrands is
    spectrally accurate, so the lattice weight hits the closed form at
    machine precision already on the coarsest grid.
    """
    delta = 0.3
    w_exact = np.exp(-(delta**2) / 4.0) / (2.0 * np.sqrt(np.pi))
    from math import erf

    p_exact = 0.5 * (1.0 + erf(delta / 2.0))  # P(X_1 < X_2) = Phi(delta / sqrt(2))
    for half_width, unit in [(200, 0.1), (400, 0.05), (800, 0.025)]:
        lattice = UniformLattice(L=half_width, unit=unit)
        field = _smooth([0.0, delta], lattice)
        w = laplacian_weights(field)[0, 1]
        assert abs(w - w_exact) < 1e-10 * w_exact
        # the forward map: outright win prob converges to Phi at O(unit)
        # (first order because the lattice tie mass ~ unit is excluded)
        p = outright_win_probabilities(field)
        assert abs(p[0] - p_exact) < 2.0 * unit


# ---- fuzz: random mixtures of every special ingredient ----


def _random_special_field(rng, lattice=LATTICE):
    n = int(rng.integers(2, 9))
    out = []
    for _ in range(n):
        kind = rng.choice(
            ["smooth", "narrow", "atom", "two_point", "zero", "edge"],
            p=[0.4, 0.15, 0.15, 0.15, 0.05, 0.10],
        )
        if kind == "smooth":
            d = Density.skew_normal(
                lattice, loc=0.0, scale=rng.uniform(0.3, 2.0), a=rng.uniform(-1.0, 1.0)
            ).shift_fractional(rng.normal(0.0, 2.0) / lattice.unit)
        elif kind == "narrow":
            # one-to-three lattice points wide: numerically almost an atom
            d = Density.skew_normal(
                lattice, loc=0.0, scale=rng.uniform(0.04, 0.15), a=0.0
            ).shift_fractional(rng.normal(0.0, 2.0) / lattice.unit)
        elif kind == "atom":
            d = _atom(float(rng.uniform(-15.0, 15.0)), lattice)
        elif kind == "two_point":
            x0 = float(rng.uniform(-5.0, 5.0))
            d = _two_point(x0, x0 + rng.uniform(0.1, 3.0), 10.0 ** rng.uniform(-9, -3), lattice)
        elif kind == "zero":
            d = _zero_mass(lattice)
        else:  # edge
            side = 1 if rng.random() < 0.5 else -1
            d = Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0).shift_fractional(
                side * (lattice.L - int(rng.integers(0, 30)))
            )
        out.append(d)
    return out


@pytest.mark.parametrize("seed", range(25))
def test_fuzz_matvec_matches_dense(seed):
    rng = np.random.default_rng(seed)
    field = _random_special_field(rng)
    n = len(field)
    W = laplacian_weights(field)
    assert np.all(np.isfinite(W)) and np.all(W >= 0.0)
    assert np.allclose(W, W.T)
    L = np.diag(W.sum(axis=1)) - W
    scale = max(np.abs(L).max(), 1.0)
    assert np.linalg.eigvalsh(L)[0] > -1e-10 * scale
    for _ in range(2):
        u = rng.normal(size=n)
        got = laplacian_matvec(field, u)
        assert np.all(np.isfinite(got))
        assert np.allclose(got, L @ u, rtol=1e-7, atol=1e-9 * scale), (
            f"seed {seed}: max diff {np.abs(got - L @ u).max():.3e}"
        )
    assert np.all(laplacian_matvec(field, np.ones(n)) == 0.0)
    p = outright_win_probabilities(field)
    assert np.all(p >= 0.0) and p.sum() <= 1.0 + 1e-12


# ---- Newton-CG inversion API ----


def _bases(scales, skews=None, lattice=LATTICE):
    skews = skews or [0.0] * len(scales)
    return [Density.skew_normal(lattice, loc=0.0, scale=s, a=k) for s, k in zip(scales, skews)]


def test_inversion_roundtrip_heterogeneous():
    """invert_outright_probabilities recovers abilities from its own forward map."""
    rng = np.random.default_rng(2)
    scales = [0.7, 1.0, 1.3, 1.0, 0.9, 1.1, 1.0, 0.8]
    a_true = rng.normal(scale=0.7, size=8)
    a_true -= a_true.mean()
    bases = _bases(scales)
    target = outright_win_probabilities(_smooth(list(a_true), scales=scales))
    result = invert_outright_probabilities(bases, target, tol=1e-11)
    assert result.converged
    assert result.residual < 1e-11
    assert np.allclose(result.abilities, a_true, atol=1e-7)
    assert abs(result.abilities.mean()) < 1e-12  # gauge fixed at zero mean
    assert np.allclose(result.achieved, target, atol=1e-10)


def test_inversion_normalized_target_and_warm_start():
    """A target renormalized to sum one is inverted exactly: only ratios matter.

    The multiplicative gauge absorbs the tie-mass deficit, so normalizing
    the target does not perturb the recovered abilities at all, and the
    reported scale recovers the attainable total.  A warm start from a
    different gauge must land on the same mean-zero answer.
    """
    scales = [0.8, 1.0, 1.2, 1.0]
    a_true = np.array([0.4, -0.1, 0.3, -0.6])
    a_true -= a_true.mean()
    bases = _bases(scales)
    p = outright_win_probabilities(_smooth(list(a_true), scales=scales))
    target = p / p.sum()  # sums to one, like market prices
    result = invert_outright_probabilities(bases, target, tol=1e-11)
    assert result.converged
    assert np.allclose(result.abilities, a_true, atol=1e-7)
    assert result.scale == pytest.approx(p.sum(), abs=1e-8)  # the tie-mass deficit
    assert np.allclose(result.achieved, result.scale * target, atol=1e-10)
    warm = invert_outright_probabilities(bases, target, tol=1e-11, initial=a_true + 5.0)
    assert warm.converged
    assert np.allclose(warm.abilities, result.abilities, atol=1e-7)


def test_inversion_longshot_ratios_respected():
    """Tiny probabilities are matched in ratio, not swallowed by an offset.

    This is the case that broke the earlier centered-residual criterion: a
    uniform additive offset can absorb a longshot's entire probability.
    The multiplicative gauge must reproduce it to high relative accuracy.
    """
    bases = _bases([1.0, 1.0, 1.0])
    target = np.array([1.0, 1e-6, 3e-6])
    target /= target.sum()
    result = invert_outright_probabilities(bases, target, tol=1e-12, max_iter=80)
    assert result.converged
    ratio = result.achieved / (result.scale * target)
    assert np.allclose(ratio, 1.0, rtol=1e-3)  # longshots correct in ratio
    spread = result.abilities.max() - result.abilities.min()
    assert spread > 5.0  # a 1e-6 longshot really is far behind


def test_inversion_extreme_target():
    """A heavily lopsided (but interior) target is still reachable."""
    bases = _bases([1.0, 1.0, 1.0])
    target = np.array([0.90, 0.07, 0.03])
    result = invert_outright_probabilities(bases, target / target.sum() * 0.98)
    assert result.converged
    spread = result.abilities.max() - result.abilities.min()
    assert spread > 1.0  # the favourite is far ahead in ability


def test_inversion_atom_base_is_graceful():
    """Atoms make the map piecewise smooth; the solver must not blow up."""
    lattice = LATTICE
    bases = [_atom(0.0, lattice), Density.skew_normal(lattice, loc=0.0, scale=1.0, a=0.0)]
    result = invert_outright_probabilities(bases, [0.4, 0.55], max_iter=20)
    assert np.all(np.isfinite(result.abilities))
    assert np.all(np.isfinite(result.achieved))
    if result.converged:
        assert result.residual < 1e-10


@pytest.mark.parametrize("name", ["baseline", "atom_vs_smooth", "edge_pileup", "large_random"])
def test_operator_reuse_matches_fresh_calls(name):
    """One precomputed operator applied to many vectors equals fresh matvecs."""
    field = FIELD_BUILDERS[name]()
    op = LaplacianOperator(field)
    rng = np.random.default_rng(3)
    for _ in range(4):
        u = rng.normal(size=len(field))
        assert np.array_equal(op.matvec(u), laplacian_matvec(field, u))
    assert np.all(op(np.ones(len(field))) == 0.0)


def test_inversion_reports_convergence_diagnostics():
    bases = _bases([1.0, 1.0, 1.0])
    target = np.array([0.5, 0.3, 0.2])
    result = invert_outright_probabilities(bases, target * 0.97)
    assert result.converged
    assert result.message == "converged"
    hist = result.residual_history
    assert hist is not None and hist[-1] == result.residual
    assert hist[-1] < hist[0]  # made progress from the initial residual


def test_inversion_falls_off_lattice_gracefully():
    """A target needing more ability spread than the lattice affords.

    A 1e-12 longshot needs a favourite-longshot gap of ~10 ability units;
    a lattice spanning +/- 6 cannot represent it.  The solver must pin the
    abilities at the representable clamp, refuse to claim convergence, and
    say the lattice is the reason -- not stall silently or run away.
    """
    small = UniformLattice(L=120, unit=0.05)  # range +/- 6 with sigma = 1 bases
    bases = _bases([1.0, 1.0, 1.0], lattice=small)
    target = np.array([1.0, 1e-12, 1e-12])
    result = invert_outright_probabilities(bases, target / target.sum(), max_iter=60)
    assert np.all(np.isfinite(result.abilities))
    box = (small.L - 2) * small.unit
    assert np.abs(result.abilities).max() <= box * (1.0 + 1e-12)  # never leaves the box
    assert not result.converged
    assert "lattice" in result.message


def test_inversion_disconnected_graph_message():
    """Non-overlapping atoms: zero Laplacian, no Newton direction."""
    bases = [_atom(-0.5), _atom(0.5)]
    result = invert_outright_probabilities(bases, [0.6, 0.4], max_iter=5)
    assert not result.converged
    assert "disconnected" in result.message
    assert np.all(np.isfinite(result.abilities))


def test_inversion_validation():
    bases = _bases([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        invert_outright_probabilities(bases, [0.5, 0.5])  # wrong length
    with pytest.raises(ValueError):
        invert_outright_probabilities(bases, [0.5, 0.5, 0.0])  # boundary target
    with pytest.raises(ValueError):
        invert_outright_probabilities(bases, [0.5, 0.5, -0.1])
    with pytest.raises(ValueError):
        invert_outright_probabilities(bases, [0.5, 0.4, np.nan])
    with pytest.raises(ValueError):
        invert_outright_probabilities(bases, [0.3, 0.3, 0.4], initial=[0.0, np.inf, 0.0])
    with pytest.raises(ValueError):
        invert_outright_probabilities(
            [_zero_mass(), *_bases([1.0, 1.0])], [0.2, 0.4, 0.4]
        )  # zero-mass runner is not identifiable


# ---- validation ----


def test_input_validation():
    field = FIELD_BUILDERS["baseline"]()
    with pytest.raises(ValueError):
        laplacian_matvec(field, np.ones(3))
    with pytest.raises(ValueError):
        laplacian_matvec(field, np.array([1.0, np.nan, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        outright_win_probabilities(field[:1])
    other = UniformLattice(L=400, unit=0.1)
    mixed = [field[0], Density.skew_normal(other, loc=0.0, scale=1.0, a=0.0)]
    with pytest.raises(ValueError):
        laplacian_weights(mixed)
    corrupt = FIELD_BUILDERS["pair"]()
    corrupt[0].p = corrupt[0].p.copy()
    corrupt[0].p[0] = np.nan
    with pytest.raises(ValueError):
        laplacian_matvec(corrupt, np.zeros(2))
    negative = FIELD_BUILDERS["pair"]()
    negative[0].p = negative[0].p.copy()
    negative[0].p[0] = -0.1
    with pytest.raises(ValueError):
        laplacian_weights(negative)
