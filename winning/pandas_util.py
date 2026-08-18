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

        # Iterate groups explicitly rather than groupby.apply: pandas 3.0
        # excludes the grouping column from frames passed to apply, which
        # breaks any later groupby on the returned frame.
        df = df.copy()
        vals = pd.Series(index=df.index, dtype=float)
        for _, sub in df.groupby(by, sort=False):
            p = ability_implied_state_prices(ability=sub[ability_col], density=density, unit=unit)
            vals.loc[sub.index] = normalize(p)
        df[new_col] = vals
        return df


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

        df = df.copy()
        vals = pd.Series(index=df.index, dtype=float)
        for _, sub in df.groupby(by, sort=False):
            vals.loc[sub.index] = center(state_price_implied_ability(prices=sub[prob_col].values, density=density, unit=unit))
        df[new_col] = vals
        return df


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
