import pandas as pd


def ohe(df, feature):
    """
    :param df: dataframe
    :param feature: feature desired to one hot encode
    :return: df with feature one hot encoded
    """

    # create df of values of feature ohe'd
    ohe_df = pd.get_dummies(df[feature])

    # create list of column names with feature as a prefix
    cols = list(ohe_df.columns)
    new_cols = [feature + "_" + str(s) for s in cols]

    # replace column names
    replace_col_dict = {i: j for i, j in zip(cols, new_cols)}
    ohe_df = ohe_df.rename(columns=replace_col_dict)

    # join the ohe'd features onto original df
    df = pd.concat([df, ohe_df], axis=1, join='inner')

    # delete original feature
    del df[feature]

    return df


def get_teammate(player_df, row):
    """
    :param player_df: Dataframe of player data
    :param row: Row of player desired to get teammmates data for
    :return: Series of best teammates data that year
    """
    ix = row.name
    # get df of players who played in same team and season
    season_df = player_df[player_df["team_year"] == row["team_year"]]

    # only interested in players who played over half the season
    season_df = season_df[season_df["G"] > 41]

    # sorts by PTS for "best"
    season_df.sort_values(by="PTS", inplace=True, ascending=False)
    season_df.reset_index(drop=False, inplace=True)

    # creates series for best teammate dependent on
    if season_df.loc[0, "name_year"] == ix:
        teammate_series = season_df.loc[1, :]
    else:
        teammate_series = season_df.loc[0, :]

    # set series index to current ix
    teammate_series.name = ix

    return teammate_series


def sma(df):
    """
    :param df: Any dataframe with only numerical data
    :return: Row with simple moving average (more detail in comments) for each column
    """

    # sort df by age
    df = df.sort_values('Age')

    # create a dataframe row of the sma, we are using sum to simulate the sma, it is essentially the same thing *3
    # however this will penalise rookies.
    sma_row = df.sum(skipna=True).to_frame().transpose()

    # rename columns and set name_year
    new_headers = [header + "_sma" for header in list(sma_row.columns)]
    replace_col_dict = {i: j for i, j in zip(list(sma_row.columns), new_headers)}
    sma_row = sma_row.rename(columns=replace_col_dict, index={0: list(df.index)[-1]})

    return sma_row


def get_prev_seasons(df, row):
    """
    :param df: Dataframe with player data
    :param row: Player dataframe row
    :return: Dataframe with past 3 years data for player (if less than two just uses 1)
    """

    # get name and year of player
    name_year = row.name
    name = name_year[:-4]
    year = int(name_year[-4:])

    # get index of previous two seasons
    name_year1 = name + str(year - 1)
    name_year2 = name + str(year - 2)

    # get previous two rows (avoid key errors due to player not playing prev season
    try:
        row_1 = df.loc[name_year1, :]
        row_2 = df.loc[name_year2, :]

        # join rows into a df
        df = pd.DataFrame(columns=row.index)
        df = df.append(row, sort=False)
        df = df.append(row_1, sort=False)
        df = df.append(row_2, sort=False)

    except KeyError:
        try:
            row_1 = df.loc[name_year1, :]

            # join rows into a df
            df = pd.DataFrame(columns=row.index)
            df = df.append(row, sort=False)
            df = df.append(row_1, sort=False)

        except KeyError:
            df = pd.DataFrame(columns=row.index)
            df = df.append(row, sort=False)

    return df



