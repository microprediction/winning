"""Interface conformance across all rating systems, including shims."""

import pytest

from winning import EloRating, Glicko2Rating, ThurstoneRating


def all_systems():
    systems = [
        ("elo", EloRating()),
        ("glicko2", Glicko2Rating()),
        ("thurstone", ThurstoneRating()),
    ]
    try:
        from winning.shims import OpenSkillRating, TrueSkillRating

        systems.append(("trueskill", TrueSkillRating()))
        systems.append(("openskill-pl", OpenSkillRating("PlackettLuce")))
    except ImportError:
        pass
    return systems


@pytest.mark.parametrize("label,system", all_systems())
def test_winner_gains_and_probabilities_track(label, system):
    names = ["alice", "bob", "carol"]
    before = {nm: system.rating(nm).mu for nm in names}
    p0 = system.win_probabilities(names)
    assert abs(sum(p0) - 1.0) < 1e-6
    assert max(p0) - min(p0) < 1e-6  # symmetric start

    for _ in range(6):
        system.observe(names, ranks=[1, 2, 3])

    after = {nm: system.rating(nm).mu for nm in names}
    assert after["alice"] > before["alice"]
    assert after["carol"] < before["carol"]

    p = system.win_probabilities(names)
    assert abs(sum(p) - 1.0) < 1e-6
    assert p[0] > p[1] > p[2]

    board = system.leaderboard()
    assert board[0][0] == "alice"


@pytest.mark.parametrize("label,system", all_systems())
def test_validates_bad_events(label, system):
    with pytest.raises(ValueError):
        system.observe(["a"], [1])
    with pytest.raises(ValueError):
        system.observe(["a", "a"], [1, 2])
    with pytest.raises(ValueError):
        system.observe(["a", "b"], [0, 1])


def test_glicko2_idle_rd_inflation():
    g = Glicko2Rating()
    for _ in range(10):
        g.observe(["a", "b"], [1, 2], dt=1.0)
    rd_active = g.rating("a").sigma
    g.elapse(3650.0)
    rd_idle = g.rating("a").sigma
    assert rd_idle > rd_active
    assert rd_idle <= g.initial_rd + 1e-9


def test_performance_samples_shapes():
    import numpy as np

    for _, system in all_systems():
        system.observe(["p", "q", "r"], [1, 2, 3])
        s = system.performance_samples(["p", "q", "r"], size=16)
        assert s is not None
        arr = np.asarray(s)
        assert arr.shape == (16, 3)
        assert np.isfinite(arr).all()


def test_kernels_module():
    import numpy as np

    from winning import kernels

    for k in (
        kernels.gaussian_kernel(),
        kernels.student_t_kernel(nu=3),
        kernels.offday_kernel(),
        kernels.skew_kernel(a=2.0),
    ):
        assert len(k) % 2 == 1 and abs(k.sum() - 1.0) < 1e-12 and k.min() >= 0
    # heavier tails in the same far cell: t(2) > t(10) > gaussian
    g = kernels.gaussian_kernel()
    t2 = kernels.student_t_kernel(nu=2)
    t10 = kernels.student_t_kernel(nu=10)
    def tail_mass(k, unit=0.1, beyond=3.0):
        half = (len(k) - 1) // 2
        x = np.arange(-half, half + 1) * unit
        return k[np.abs(x) > beyond].sum()
    assert tail_mass(t2) > tail_mass(t10) > tail_mass(g)
    # a t-kernel drives the rater end to end
    from winning import ThurstoneRating

    tr = ThurstoneRating(base_kernel=kernels.student_t_kernel(nu=3))
    tr.observe(["a", "b", "c"], [1, 2, 3])
    p = tr.win_probabilities(["a", "b", "c"])
    assert abs(sum(p) - 1.0) < 1e-6 and p[0] > p[2]
