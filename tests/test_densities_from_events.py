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