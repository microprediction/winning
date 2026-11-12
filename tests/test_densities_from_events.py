from winning.lattice import densities_from_events, state_prices_from_densities
from winning.lattice_plot import densitiesPlot


def test_golf():
    do_golf()


def do_golf():
    event_18 = [0, 0, 0.5, 0.5, 0, 0, 0]
    event_17 = [0, 0, 0.5, 0.5, 0, 0, 0]
    scores = [-15,-15]
    densities = densities_from_events(scores=scores, events=[[event_17,event_18],
                                                             [event_18]
                                                             ], unit=1, L=6)
    return densities


if __name__=='__main__':
    try:
        d = do_golf()
        import matplotlib.pyplot as plt
        densitiesPlot(d, unit=1,legend=['two to play','one to play'])
        plt.show()
        p = state_prices_from_densities(densities=d)
        print(p)
    except ImportError:
        pass

def test_densities_from_events_does_not_mutate_or_drift():
    # #607: the score density was appended into the caller's event lists,
    # so each reuse added another shift (65.6% -> 89.1% -> 98.4%).
    import copy
    from winning.classic.lattice import state_prices_from_events
    e1 = [0, 0, .5, .5, 0, 0, 0]
    e2 = [0, 0, .25, .5, .25, 0, 0]
    events = [[e1.copy(), e2.copy()], [e2.copy()]]
    before = copy.deepcopy(events)
    first = state_prices_from_densities(densities_from_events(scores=[-1, 0], events=events, L=8, unit=1))
    for _ in range(2):
        again = state_prices_from_events(scores=[-1, 0], events=events, L=8, unit=1)
        assert list(again) == list(first)
    assert events == before
    # exact by enumeration: the leader (score -1) is e1 + e2, the other is
    # e2 shifted one point worse, ties split. This pinned 0.65625, which
    # was the score placed at -1 instead of +1 by density_from_samples
    assert abs(first[1] - 0.109375) < 1e-12


def test_a_score_is_placed_where_it_is():
    # the events helpers size the score's lattice to the score itself;
    # density_from_samples clamped at +-(L - 2), so a score of 1 became -1
    # and a score of 2 a dead heat
    import numpy as np
    from winning.classic.lattice import density_from_samples, state_prices_from_events
    for s in range(0, 6):
        L = max(s, 1)
        d = np.asarray(density_from_samples([s], L=L, unit=1))
        assert abs(d @ np.arange(-L, L + 1) - s) < 1e-12
    for x, L, unit in [(2.5, 5, 0.5), (9, 10, 1.0), (-9.25, 10, 1.0)]:
        d = np.asarray(density_from_samples([x], L=L, unit=unit))
        assert abs(d @ (unit * np.arange(-L, L + 1)) - x) < 1e-12
    event = [0, 0, 0, 0.5, 0.5, 0, 0]            # a point now or one later
    for s, want in [(1, 0.875), (2, 1.0), (3, 1.0)]:   # by enumeration; was [0.125], tie, [0.8125]
        p = state_prices_from_events(scores=[0, s], events=[[event], [event]], L=12, unit=1)
        assert abs(p[0] - want) < 1e-12, (s, p)


def test_densities_door_normalises_counts_and_refuses_negative_atoms():
    import pytest
    a, b = [0, 0, 0, 3, 1, 0, 0], [0, 0, 0, 0, 1, 3, 0]
    counts = state_prices_from_densities([a, b])
    freqs = state_prices_from_densities([[x / 4 for x in a], [x / 4 for x in b]])
    assert max(abs(c - f) for c, f in zip(counts, freqs)) < 1e-15
    assert abs(freqs[0] - 0.96875) < 1e-15             # counts gave 0.875
    with pytest.raises(ValueError, match="densities"):
        state_prices_from_densities([[0, 0, 0, 1.2, -0.2, 0, 0], b])


def test_longshot_adjustment_keeps_a_missing_quote_missing():
    # None ** expon raised TypeError, so skew_normal_place_pricing crashed on
    # a book with a missing quote that prices_from_dividends accepts
    import math
    from winning.classic.lattice_simulation import longshot_adjusted_dividends
    a = longshot_adjusted_dividends([2, None, 4])
    b = longshot_adjusted_dividends([2, float("nan"), 4])
    assert all(math.isfinite(x) for x in a) and a == b
    assert longshot_adjusted_dividends([2, None, math.inf])[2] == math.inf


def test_longshot_adjustment_honours_nan_value_and_numpy_nans():
    # the default 2000 was applied before calibration, so the caller's
    # nan_value never took effect; np.float32 NaN was not seen as missing
    # and expon 0 made it a live third of the book (review of #1008)
    import numpy as np
    from winning.classic.lattice_simulation import longshot_adjusted_dividends
    from winning.classic.lattice_calibration import prices_from_dividends
    for miss in (None, float("nan"), np.float32("nan"), np.float64("nan")):
        # [2, 2, 4] at exponent 1: the missing runner takes 40%, not 0.067%
        a = longshot_adjusted_dividends([2, miss, 4], longshot_expon=1.0, nan_value=2.0)
        p = prices_from_dividends(a)
        assert abs(p[1] - 0.4) < 1e-12, (miss, p)
        # at exponent 0 every quoted dividend is 1 and the missing one 2
        z = prices_from_dividends(longshot_adjusted_dividends([2, miss, 4], longshot_expon=0, nan_value=2.0))
        assert abs(z[1] - 0.2) < 1e-12, (miss, z)
        b = longshot_adjusted_dividends([2, miss, 4], longshot_expon=1.0)
        assert abs(prices_from_dividends(b)[1] - prices_from_dividends([2, None, 4])[1]) < 1e-12
