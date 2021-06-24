from winning.lattice_calibration import state_price_implied_ability, dividend_implied_ability, \
    ability_implied_dividends, ability_implied_state_prices
from winning.lattice import skew_normal_density, center_density
from winning.lattice_conventions import STD_UNIT, STD_SCALE, STD_L, STD_A, NAN_DIVIDEND


# Specializing the horse race problem to N(a,1) horse performance

def std_state_price_implied_ability(prices: [float], unit:float=STD_UNIT, L=STD_L, scale=STD_SCALE) ->[float]:
    """ Ability implied from state prices, assuming normally distributed performance

          prices: [ float ]    For continuous distributions, state prices are synonymous with winning probabilities
          unit:                The lattice width
          L:                   The lattice is length 2*L+1
          scale:               The standard deviation
          returns:  [ float ]  List of relative abilities. Lower is better.

    """
    density = centered_std_density(loc=0, unit=unit, L=L, scale=scale)
    return state_price_implied_ability(prices=prices, density=density, unit=unit)


def std_dividend_implied_ability(dividends, nan_value = NAN_DIVIDEND, L=STD_L, unit=STD_UNIT, scale=STD_SCALE):
    """ Implied ability using skewed normal performances, from inverse win probabilities """
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return dividend_implied_ability(dividends=dividends, density=density, nan_value=nan_value, unit=unit )


def centered_std_density(loc=0.0, L=STD_L, unit=STD_UNIT, scale=STD_SCALE):
    density = skew_normal_density(L=L, unit=unit, loc=loc, scale=scale, a=0)
    return center_density(density)


def std_ability_implied_state_prices(ability:[float],unit:float=STD_UNIT, L=STD_L, scale=STD_SCALE) -> [float]:
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return ability_implied_state_prices(ability=ability, density=density, unit=unit)


def std_ability_implied_dividends(ability:[float], unit:float=STD_UNIT, L=STD_L, scale=STD_SCALE, nan_value=NAN_DIVIDEND) -> [float]:
    density = centered_std_density(loc=0.0, L=L, unit=unit, scale=scale)
    return ability_implied_dividends(ability=ability, density=density, unit=unit, nan_value=nan_value)


