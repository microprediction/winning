from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A, NAN_DIVIDEND
from winning.lattice_calibration import dividend_implied_ability, normalize_dividends, dividends_from_prices
from winning.lattice import skew_normal_density, densities_from_offsets, pdf_to_cdf, sample_from_cdf
import heapq
from collections import Counter
import numpy as np

# A few utilities for simulating from a race where runner performance distributions are represented on a lattice.
# None of this is essential to the core calibration algorithm.
# Rather, it is used post-calibration to Monte Carlo prices for win, place, show and exotics.
# No claims of computational efficiency here!


PLACING_NAMES = ['win','place2','place3','place4']   # aka win/place/show/top4
N_SAMPLES = 5000                                     # Default number of Monte Carlo paths


def simulate_performances(densities, n_samples:int=N_SAMPLES, unit=1.0, add_noise=True):
    """ Simulate multiple contest outcomes
    :param densities:
    :param n_samples:
    :return:  [ [race performance] ]    List of races
    """
    cdfs = [pdf_to_cdf(density) for density in densities]
    cols = [sample_from_cdf(cdf, n_samples=n_samples,add_noise=add_noise,unit=unit) for cdf in cdfs]
    rows = list(map(list, zip(*cols)))
    return rows


def placegetters_from_performances(performances, n=4) -> [[int]]:
    """
    :param performances:  List of list of performances
    :return:  List of Lists
    """
    return [[placegetter(row, k) for row in performances] for k in range(n) ]


def placegetter(scores:[float], position:int):
    """ Return the index of the participant finishing in position+1
    :param scores:
    :param position:  0 for first place, 1 for second etc
    :return:
    """
    return heapq.nsmallest(position + 1, range(len(scores)), key=scores.__getitem__)[position]


def skew_normal_place_pricing(dividends, n_samples=N_SAMPLES, longshot_expon:float=1.0, a:float=STD_A, scale=STD_SCALE, nan_value=NAN_DIVIDEND) -> dict:
    """ Price place/show and exotics from win market by Monte Carlo of performances
        :param  dividends  [ float ] decimal prices
        :param  longshot_expon  power law to apply to dividends, if you want to try to correct for longshot bias.
        :param  a         skew parameter in skew-normal running time distribution
        :param  scale     scale parameter in skew-normal running time distribution
        :returns  {'win':[1.6,4.5,...], 'place':[  ] , ... }
    """
    # TODO: Add control variates
    unit = STD_UNIT
    L = STD_L
    density = skew_normal_density(L=L, unit=unit, scale=scale, a=a)
    adj_dividends = longshot_adjusted_dividends(dividends=dividends,longshot_expon=longshot_expon)
    offsets = dividend_implied_ability(dividends=adj_dividends, density=density,nan_value=nan_value)
    densities = densities_from_offsets(density=density, offsets=offsets)
    performances = simulate_performances(densities=densities, n_samples=n_samples, add_noise=True, unit=unit)
    placegetters = placegetters_from_performances(performances=performances, n=4)
    the_counts = exotic_count(placegetters, do_exotics=False)
    n_runners = len(adj_dividends)
    prices = dict()
    for bet_type, multiplicity in zip(PLACING_NAMES,range(1,5)):
        prices[bet_type] = dividends_from_prices( [the_counts[bet_type][j] for j in range(n_runners)], multiplicity=multiplicity)
    return prices


def longshot_adjusted_dividends(dividends,longshot_expon=1.17):
    """ Use power law to approximately unwind longshot effect
        Obviously this is market dependent
    """
    dividends = [(o + 1.0) ** (longshot_expon) for o in dividends]
    return normalize_dividends(dividends)


def exotic_count(placegetters, do_exotics=False):
    """  Given counters for winner, second place etc, create counters for win,place,show and exotics """
    # A tad ugly :)
    winner, second, third, forth = placegetters[0], placegetters[1],placegetters[2], placegetters[3]
    win = Counter(winner)
    place = Counter(second)
    place.update(win)
    show = Counter(third)
    show.update(place)
    top4 = Counter(forth)
    top4.update(show)
    if do_exotics:
        exacta = Counter(zip(winner, second))
        trifecta = Counter(zip(winner, second, third))
        pick4 = Counter(zip(winner,second,third,forth))
    else:
        exacta, trifecta, pick4 = None, None, None
    return {"win": win, "place2": place, "place3": show,"place4":top4,"exacta": exacta, "trifecta": trifecta, "pick4":pick4}




if __name__ == '__main__':
    # An illustration...
    derby = {'Essential Quality': 2,
             'Rock Your World': 5,
             'Known Agenda': 8,
             'Highly Motivated': 10,
             'Hot Rod Charlie': 10,
             'Medina Spirit': 16,
             'Mandaloun': 16,
             'Dynamic One': 20,
             'Bourbonic': 25,
             'Midnight Bourbon': np.nan,  # <--- This will be ignored
             'Super Stock': 25,
             'Soup and Sandwich': 33,
             'O Besos': 33,
             'King Fury': 33,
             'Helium': 33,
             'Like The King': 40,
             'Brooklyn Strong': 50,
             'Keepmeinmind': 50,
             'Hidden Stash': 50,
             'Sainthood': 50}
    if False:
        dividends = [ o+1.0 for o in derby.values() ]
    else:
        dividends = [6,6,6,6,6,6]
    prices = skew_normal_place_pricing(dividends=dividends, longshot_expon=1.17, n_samples=5000)
    from pprint import pprint
    pprint(list(zip(prices['win'],prices['place3'])))
