from winning.lattice_calibration import state_price_implied_ability, dividend_implied_ability
from winning.lattice import skew_normal_density, center_density
from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A


# Specializing the ratings race problem to N(a,1) horses


def std_state_price_implied_ability(prices):
    """ Ability implied from state prices

          returns:  [ float ]   List of relative abilities

    """
    # (For continuous distributions state prices are synonymous with winning probabilities)
    density = std_density(loc=0)
    return state_price_implied_ability(prices=prices, density=density)

def std_dividend_implied_ability(dividends):
    """ Implied ability using skewed normal performances, from inverse win probabilities """
    return dividend_implied_ability(dividends=dividends, density=std_density(loc=0.0))

def std_density(loc=0):
    density = skew_normal_density(L=STD_L, unit=STD_UNIT, loc=loc, scale=STD_SCALE, a=STD_A)
    return center_density(density)



