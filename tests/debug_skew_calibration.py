from winning.skew_calibration import skew_ability_implied_dividends, skew_dividend_implied_ability


if __name__=='__main__':
    dividends               = [ 6.0, 3.0, 2.0, 0.00001 ]
    ability                 = skew_dividend_implied_ability(dividends= dividends)
    print(ability)
    prices                  = skew_ability_implied_dividends(ability)
    for d,p in zip(dividends,prices):
        assert abs(d-p)<0.002


