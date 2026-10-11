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
    assert abs(first[1] - 0.65625) < 1e-12
