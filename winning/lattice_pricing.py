from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A
from winning.lattice_calibration import dividend_implied_ability, normalize_dividends, dividends_from_prices
from winning.lattice import skew_normal_density, densities_from_offsets, pdf_to_cdf, sample_from_cdf
import heapq
from collections import Counter
import numpy as np

# Monte Carlo pricing of win/place/show and exotics
# This really is inefficient, but hopefully correct

BET_TYPES = ['win','place','show','top4','exacta','quinella','trifecta','pick4']

def skew_normal_simulation(dividends, nSamples=5000, longshot_expon=1.0,skew_parameter=STD_A,nan_value=10000):
    """ Monte carlo the skew normal running time distribution model
        :param dividends array [1.7,...]
        :returns prices  dict  {'win':[1.6,4.5,...], 'place':[  ] , ... }
    """
    density = skew_normal_density(L=STD_L, unit=STD_UNIT, scale=STD_SCALE, a=skew_parameter)
    adj_dividends = longshot_adjusted_dividends(dividends=dividends,longshot_expon=longshot_expon)
    offsets = dividend_implied_ability(dividends=adj_dividends, density=density,nan_value=nan_value)
    densities = densities_from_offsets(density=density, offsets=offsets)
    cdfs = [pdf_to_cdf(density) for density in densities]
    cols = [sample_from_cdf_with_noise(cdf, nSamples) for cdf in cdfs]

    rows = list(map(list, zip(*cols)))
    winner = [placegetter(row, 0) for row in rows]
    second = [placegetter(row, 1) for row in rows]
    third = [placegetter(row, 2) for row in rows]
    forth = [placegetter(row, 3) for row in rows]
    the_counts = exotic_count(winner=winner, second=second, third=third, forth=forth, do_exotics=False)
    n_runners = len(adj_dividends)

    prices = dict()
    for bet_type, multiplicity in zip(['win','place','show','top4'],range(1,5)):
        prices[bet_type] = dividends_from_prices( [the_counts[bet_type][j] for j in range(n_runners)], multiplicity=multiplicity )
    return prices


def longshot_adjusted_dividends(dividends,longshot_expon=1.17):
    """ Use power law to approximately unwind longshot effect
        Obviously this is market dependent
    """
    dividends = [(o + 1.0) ** (longshot_expon) for o in dividends]
    return normalize_dividends(dividends)


def placegetter(scores, position):
    return heapq.nsmallest(position + 1, range(len(scores)), key=scores.__getitem__)[position]


def sample_from_cdf_with_noise(cdf, nSamples):
    # Break ties
    samples = sample_from_cdf(cdf=cdf, nSamples=nSamples)
    noise = 0.00001 * np.random.randn(nSamples)
    return [s + x for s, x in zip(samples, noise)]


def exotic_count(winner, second, third, forth, do_exotics=True):
    """  Given counters for winner, second place etc, create counters for win,place,show and exotics
    """
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
    return {"win": win, "exacta": exacta, "trifecta": trifecta, "place": place, "show": show,"top4":top4,"pick4":pick4}




if __name__ == '__main__':
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
    prices = skew_normal_simulation(dividends=dividends,longshot_expon=1.17,nSamples=5000)
    from pprint import pprint
    pprint(list(zip(prices['win'],prices['show'])))
