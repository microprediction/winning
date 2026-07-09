from winning.lattice_conventions import STD_A, STD_L, STD_SCALE, STD_UNIT
from winning.lattice import skew_normal_density
from winning.lattice_calibration import ability_implied_state_prices, state_price_implied_ability
from winning.lattice_calibration import normalize

try:
    import pandas as pd
    using_pandas = True
except ImportError:
    using_pandas = False

if using_pandas:

    def add_ability_implied_state_price_to_dataframe(df, ability_col, by: str, density, new_col: str, unit: float):
        """
        :param df:
        :param ability_col:
        :param by:            Column to group by
        :param density:
        :param new_col:
        :param unit:
        :return:
        """

        def _add_ability_implied(df, ability_col: str, new_col: str, density, unit):
            p = ability_implied_state_prices(ability=df[ability_col], density=density, unit=unit)
            p = normalize(p)
            df[new_col] = p
            return df

        kwargs = {'ability_col': ability_col, 'new_col': new_col, 'density': density, 'unit': unit}
        return df.groupby(by, group_keys=False).apply(_add_ability_implied, **kwargs)


    def add_centered_ability_to_dataframe(df, prob_col, by:str, density, unit, new_col:str):
        """
           :param df:           pd.DataFrame with probability columns
           :param prob_col:     Name of column holding selection (win) probabilities
           :param new_col:      Name of new column to store ability in
           :param by:           Categorical variable column indicated groupings
           :param density
           :return:  New data frame with 'ability' column
        """

        def center(x):
            mx = sum(x) / len(x)
            return [xi - mx for xi in x]

        def _add_ability(df, prob_col, new_col, density, unit):
            df[new_col] = center(state_price_implied_ability(prices=df[prob_col].values, density=density, unit=unit))
            return df

        kwargs = {'prob_col': prob_col, 'new_col': new_col, 'density': density,'unit':unit}
        return df.groupby(by, group_keys=False).apply(_add_ability, **kwargs)


    def add_skew_normal_ability_to_dataframe(df, by: str, prob_col='p', new_col='ability', L=STD_L, scale=STD_SCALE, unit=STD_UNIT, a=STD_A, loc=0.0):
        """
        :param df:           pd.DataFrame with probability columns
        :param prob_col:     Name of column holding selection (win) probabilities
        :param new_col:      Name of new column to store ability in
        :param by:           Categorical variable column indicated groupings
        :param L:            Lattice size
        :param scale:        Width of performance distribution in absolute terms
        :param unit:         Distance between lattice points
        :return:  New data frame with 'ability' column
        """

        density = skew_normal_density(L=L, unit=unit, loc=loc, scale=scale, a=a)
        return add_centered_ability_to_dataframe(df=df, prob_col=prob_col, by=by, new_col=new_col, density=density, unit=unit)
