from winning.lattice_simulation import skew_normal_place_pricing
import numpy as np
from pprint import pprint

# Illustrates pricing place and show bets for the Derby
# (increase n_samples for more reliable numbers)


if __name__ =='__main__':

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
    dividends = [o + 1.0 for o in derby.values()]
    prices = skew_normal_place_pricing(dividends=dividends, longshot_expon=1.17, n_samples=1000)
    pprint(list(zip(list(derby.keys()), prices['win'], prices['place3'])))




