from winning.std_calibration import std_dividend_implied_ability, centered_std_density, std_ability_implied_dividends
from winning.lattice_calibration import ability_implied_dividends, normalize_dividends


def test_dividend_implied_ability_without_unit():
    raw_dividends = [2.,3.,6.,10.,12.,20.,66.]
    dividends = normalize_dividends(raw_dividends)
    abilities = std_dividend_implied_ability(dividends=dividends)
    divs_back = std_ability_implied_dividends(abilities)
    assert all( [ abs(d1-d2)<1e-1 for d1,d2 in zip(divs_back,dividends)])
    abilities_back = std_dividend_implied_ability(divs_back)
    assert all( [abs(a1-a2)<1e-2 for a1,a2 in zip(abilities,abilities_back)] )


def test_dividend_implied_ability_with_unit():
    raw_dividends = [2.,3.,6.,10.,12.,20.,66.]
    unit = 0.07
    dividends = normalize_dividends(raw_dividends)
    abilities = std_dividend_implied_ability(dividends=dividends, unit=unit)
    divs_back = std_ability_implied_dividends(abilities,unit=unit)
    assert all( [ abs(d1-d2)<1e-1 for d1,d2 in zip(divs_back,dividends)])
    abilities_back = std_dividend_implied_ability(divs_back)
    assert all( [abs(a1-a2)<1e-2 for a1,a2 in zip(abilities,abilities_back)] )

if __name__=='__main__':
    test_dividend_implied_ability_with_unit()
    test_dividend_implied_ability_without_unit()