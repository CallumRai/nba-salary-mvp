import pandas as pd
import os
from tools import *
import numpy as np


def player_features():
    """
    :return: Creates dataframe of regular season player features
    """

    # load player data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    per_df = pd.read_csv(data_path + "per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    advanced_df = pd.read_csv(data_path + "advanced_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    salary_df = pd.read_csv(data_path + "salary_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set index as name_year
    per_df.set_index("name_year", inplace=True)
    advanced_df.set_index("name_year", inplace=True)
    salary_df.set_index("name_year", inplace=True)

    # delete duplicates (caused by identical names only lose ~20 pieces of data but no way to concat w/ salary)
    per_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    advanced_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    salary_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)

    # concat player data
    player_df = pd.concat([per_df, advanced_df, salary_df], axis=1, join='inner')

    # drop rows with nans
    player_df.dropna(inplace=True)

    # remove duplicate columns
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]

    # keep main position due to players with multiple positions
    player_df["Pos"] = player_df["Pos"].apply(lambda pos: pos[:2])

    # one hot encode position
    player_df = ohe(player_df, "Pos")

    # create team_year
    player_df["team_year"] = player_df.apply(lambda row: row['Tm'] + "_" + str(row['year']), axis=1)

    # delete irrelevant columns
    del player_df['Player']
    del player_df["Tm"]

    return player_df


def player_nosalary_features():
    """
    :return: Creates dataframe of regular season player features without salary features (this is required because
    salary data uses player names without accents, thus when predicting these players are not included, by creating
    this dataframe we can predict on all players as salary data is not required for predicting on either model)
    """

    # load player data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    per_df = pd.read_csv(data_path + "per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    advanced_df = pd.read_csv(data_path + "advanced_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set index as name_year
    per_df.set_index("name_year", inplace=True)
    advanced_df.set_index("name_year", inplace=True)

    # delete duplicates (caused by identical names only lose ~20 pieces of data but no way to concat w/ salary)
    per_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    advanced_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)

    # concat player data
    player_df = pd.concat([per_df, advanced_df], axis=1, join='inner')

    # drop rows with nans
    player_df.dropna(inplace=True)

    # remove duplicate columns
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]

    # keep main position due to players with multiple positions
    player_df["Pos"] = player_df["Pos"].apply(lambda pos: pos[:2])

    # one hot encode position
    player_df = ohe(player_df, "Pos")

    # create team_year
    player_df["team_year"] = player_df.apply(lambda row: row['Tm'] + "_" + str(row['year']), axis=1)

    # rename, lowercase player column
    player_df.rename(columns={'Player': 'player'})

    # delete irrelevant columns
    del player_df['Player']
    del player_df["Tm"]

    return player_df


def player_po_features():
    """
    :return: Creates dataframe of player features from playoffs
    """

    # load player data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    per_df = pd.read_csv(data_path + "per_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    advanced_df = pd.read_csv(data_path + "advanced_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set index as name_year
    per_df.set_index("name_year", inplace=True)
    advanced_df.set_index("name_year", inplace=True)

    # delete duplicates (caused by identical names)
    per_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    advanced_df.drop(per_df[per_df.index.duplicated(keep=False)].index.tolist(), inplace=True)

    # concat player data
    player_df = pd.concat([per_df, advanced_df], axis=1, join='inner')

    # drop rows with nans
    player_df.dropna(inplace=True)

    # remove duplicate columns
    player_df = player_df.loc[:, ~player_df.columns.duplicated()]

    # create team_year
    player_df["team_year"] = player_df.apply(lambda row: row['Tm'] + "_" + str(row['year']), axis=1)

    # create name_year
    player_df["name_year"] = player_df.apply(lambda row: row['Player'] + "_" + str(row['year']), axis=1)

    # delete irrelevant columns
    redundant_cols = ['year', 'Pos', 'Player', 'Tm', 'Age']
    for col in redundant_cols:
        del player_df[col]

    # add playoff suffix to headers
    team_headers = [header + "_po" for header in list(player_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(player_df.columns), team_headers)}
    player_df = player_df.rename(columns=replace_col_dict)

    return player_df


def salary_features(player_df):
    """
    :param player_df: Dataframe with player data (must have year and salary column)
    :return: Player data with salary in terms of % of cap and cap (better to predict on %ofcap as more independent of
    time than salary due to inflation)
    """

    # load cap data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    cap_df = pd.read_csv(data_path + "salary_cap_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # add relevant cap data to each player based on year
    df = pd.merge(player_df, cap_df, on='year')

    # calculate % of cap
    df['%ofcap'] = df.apply(lambda row: row['salary'] / row['cap'] * 100, axis=1)

    # delete irrelvant initial cap data
    del df['salary']

    return df


def salary_cap_features(player_df):
    """
    :param player_df: Dataframe of player features
    :return: Player df with year cap attached
    """

    # load cap data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    cap_df = pd.read_csv(data_path + "salary_cap_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # add relevant cap data to each player based on year
    df = pd.merge(player_df, cap_df, on='year')

    return df


def teammate_features(player_df):
    """
    :param player_df: Player df with team_year and name_year column
    :return: Player df with best player from that seasons features appended
    """

    # create dataframe of teammate features from each row in player_df
    teammate_df = player_df.apply(lambda row: get_teammate(player_df, row), axis=1)

    # add suffix to each header in teammate df
    teammate_headers = [header + "_tmate" for header in list(player_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(player_df.columns), teammate_headers)}
    teammate_df = teammate_df.rename(columns=replace_col_dict)

    # add teammate data to player df
    player_df = pd.concat([player_df, teammate_df], axis=1, join='inner')

    # delete teammate salary/name column and teammte anme_year
    redundant_cols = ['name_year', 'salary_tmate', 'player_tmate', 'Age_tmate', 'year_tmate']
    for col in redundant_cols:
        # have to be careful of keyerror as salary data may not be included
        try:
            del player_df[col]
        except KeyError:
            pass

    # create new name_year col
    player_df['name_year'] = player_df.index

    return player_df


def team_features():
    """
    :return: Creates dataframe of regular season team features
    """

    # load team data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    team_per_df = pd.read_csv(data_path + "team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    team_advanced_df = pd.read_csv(data_path + "team_advanced_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    op_team_per_df = pd.read_csv(data_path + "op_team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set index as team_year
    team_per_df.set_index("team_year", inplace=True)
    team_advanced_df.set_index("team_year", inplace=True)
    op_team_per_df.set_index("team_year", inplace=True)

    # add op suffix to op_team due to similar column names
    op_headers = [header + "_op" for header in list(op_team_per_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(op_team_per_df), op_headers)}
    op_team_per_df = op_team_per_df.rename(columns=replace_col_dict)

    # concat player data
    team_df = pd.concat([team_per_df, team_advanced_df, op_team_per_df], axis=1, join='inner', sort=False)

    # remove duplicate columns
    team_df = team_df.loc[:, ~team_df.columns.duplicated()]

    # dont want bias due to teams, irrelevant stats and W/L covered by %
    redundant_cols = ["Team", "Conf", "Div", "Team_op", "G", "MP", "year", "G_op", "MP_op", "year_op", "W", "L"]
    for col in redundant_cols:
        del team_df[col]

    # add team suffix to headers
    team_headers = [header + "_team" for header in list(team_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(team_df.columns), team_headers)}
    team_df = team_df.rename(columns=replace_col_dict)

    return team_df


def team_po_features():
    """
    :return: Creates dataframe of playoff team features
    """

    # load team data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    team_per_df = pd.read_csv(data_path + "team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    op_team_per_df = pd.read_csv(data_path + "op_team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
    team_misc_df = pd.read_csv(data_path + "team_misc_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set dataframe indexes as team_year
    team_per_df.set_index("team_year", inplace=True)
    op_team_per_df.set_index("team_year", inplace=True)
    team_misc_df.set_index("team_year", inplace=True)

    # add op suffix to op_team due to similar column  names
    op_headers = [header + "_op" for header in list(op_team_per_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(op_team_per_df), op_headers)}
    op_team_per_df = op_team_per_df.rename(columns=replace_col_dict)

    # concat player data
    team_df = pd.concat([team_per_df, op_team_per_df, team_misc_df], axis=1, join='inner', sort=False)

    # remove duplicate columns
    team_df = team_df.loc[:, ~team_df.columns.duplicated()]

    # dont want bias due to teams or irrelvant stats
    redundant_cols = ["Team", "Team_op", "MP", "year", "MP_op", "year_op"]
    for col in redundant_cols:
        del team_df[col]

    # add team suffix to headers
    team_headers = [header + "_po_team" for header in list(team_df.columns)]
    replace_col_dict = {i: j for i, j in zip(list(team_df.columns), team_headers)}
    team_df = team_df.rename(columns=replace_col_dict)

    return team_df


def playoff_features():
    """
    :return: Creates dataframe with team_year as index and one hot encoded round attained in playoff as column
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    df = pd.read_csv(data_path + "playoff_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # dataframe structured with two teams per round, this creates a new row for the second team so one team per column
    for ix, row in df.iterrows():
        # create new row w/ team 2 for team 1
        new_row = row.tolist()
        new_row[3] = new_row[4]

        # append to df
        data = {i: j for i, j in zip(list(df.columns), new_row)}
        df = df.append(data, ignore_index=True, sort=False)

    # create team_year column
    df['team_year'] = df.apply(lambda row: row['team_1'] + "_" + str(row['year']), axis=1)

    # delete unneeded rows (no longer need team columns as team_year covers them)
    redundant_cols = ['year', 'team_1', 'team_2', 'Unnamed: 0']
    for col in redundant_cols:
        del df[col]

    # ohe round
    df = ohe(df, "round")

    # combine west and eastern conf
    df['round_1'] = df.apply(lambda row: row['round_Eastern Conf First Round'] + row['round_Western Conf First Round']
                             , axis=1)
    df['round_2'] = df.apply(lambda row: row['round_Eastern Conf Semifinals'] + row['round_Western Conf Semifinals']
                             , axis=1)
    df['round_3'] = df.apply(lambda row: row['round_Eastern Conf Finals'] + row['round_Western Conf Finals']
                             , axis=1)
    df['round_4'] = df.apply(lambda row: row['round_Finals'], axis=1)

    # delete conference data
    redundant_cols = ['round_Eastern Conf Finals', 'round_Eastern Conf First Round', 'round_Eastern Conf Semifinals',
                      'round_Finals', 'round_Western Conf Finals', 'round_Western Conf First Round',
                      'round_Western Conf Semifinals']
    for col in redundant_cols:
        del df[col]

    df.set_index('team_year', inplace=True)

    # combine duplicated indexs caused by one hot encode, e.g. if a team got to the second round they would have two
    # columns for each round
    df = df.groupby(level=0).sum()

    return df


def winner_features():
    """
    :return: Dataframe with team_year as index and winner/runnerup as column
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    df = pd.read_csv(data_path + "winner_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # delete na rows
    df.dropna(inplace=True)

    # create new df for storing data
    winner_df = pd.DataFrame(columns=["winner", "runnerup", "team_year"])

    # manually append to winner_df (iterrows fast enough due to small amount of data and simple task)
    for ix, row in df.iterrows():
        # sets data for each winner, runnerup from each year
        winner_dict = {'winner': 1, 'runnerup': 0, 'team_year': row['Champion'] + "_" + str(int(row['Year']))}
        second_dict = {'winner': 0, 'runnerup': 1, 'team_year': row['Runner-Up'] + "_" + str(int(row['Year']))}

        winner_df = winner_df.append(winner_dict, ignore_index=True, sort=False)
        winner_df = winner_df.append(second_dict, ignore_index=True, sort=False)

    winner_df.set_index("team_year", inplace=True)

    return winner_df


def mvp_features():
    """
    :return: Dataframe with mvp share and name_year
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    df = pd.read_csv(data_path + "mvp_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # create name_year column
    df['name_year'] = df.apply(lambda row: row['Player'] + "_" + str(int(row['year'])), axis=1)

    # only keep share and name_year col
    df = df[['name_year', 'Share']]

    return df


def player_sma(df, row):
    """
    :param df: Player dataframe
    :param row: Row of player data
    :return: Row of sma of numerical data from past 3 seasons (or as many as possible)
    """

    # get numerical data from past 3 seasons
    player_df = get_prev_seasons(df, row).select_dtypes(include=np.number)

    sma_row = sma(player_df)

    return sma_row


def sma_cols(df, row):
    """
    :param df: Player dataframe
    :param row: Row of player data
    :return: Columns names of sma of numerical data from past 3 seasons (or as many as possible)
    """

    # get numerical data from past 3 seasons
    player_df = get_prev_seasons(df, row).select_dtypes(include=np.number)

    sma_row = sma(player_df)

    return list(sma_row.columns)


def full_salary_featuredf(train):
    """
    :param train: Bool, true to create data to train on, false to create data to predict on
    :return: Dataframe of player features ready to be trained/predicted upon to predict salary
    """

    # get player data
    if train:
        player_df = player_features()
        player_df = teammate_features(player_df)
        player_df = salary_features(player_df)
    else:
        # for predicting we do not need the salary data but still require the cap data
        player_df = player_nosalary_features()
        player_df = teammate_features(player_df)
        player_df = salary_cap_features(player_df)
    # get team data
    team_df = team_features()

    # load playoff player team and winner data

    player_po_df = player_po_features()
    team_po_df = team_po_features()
    playoff_df = playoff_features()
    winner_df = winner_features()

    # join player data
    player_df.set_index("name_year", inplace=True)
    player_po_df.set_index("name_year_po", inplace=True)
    full_player_df = player_df.join(other=player_po_df, how='left')
    full_player_df.fillna(0, inplace=True)

    # get sma dataframe
    sma_headers = sma_cols(full_player_df, full_player_df.iloc[0, :])
    sma_df = pd.DataFrame(columns=sma_headers)

    # use iterrows as apply ineeficient w/ series
    for ix, row in full_player_df.iterrows():
        # get row
        sma_row = player_sma(full_player_df, row)

        sma_df = sma_df.append(sma_row, sort=False)

    # join sma df
    full_player_df = full_player_df.join(other=sma_df, how='left')

    # join team data
    full_team_df = team_df.join(other=team_po_df, how='left')
    full_team_df = full_team_df.join(other=playoff_df, how='left')
    full_team_df = full_team_df.join(other=winner_df, how='left')

    # reset index to facilitate join and rename
    full_player_df.reset_index(drop=False, inplace=True)
    full_player_df.set_index("team_year", inplace=True)
    df = full_player_df.join(other=full_team_df, how='left')

    # delete nas
    df.fillna(0, inplace=True)

    # recreate name_year col
    df = df.rename(columns={'index': 'name_year'})

    # reset index
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'team_year'})

    # delete uneeded rows
    redundant_cols = ['team_year_tmate', 'team_year_po', 'Unnamed: 0', 'player', '%ofcap_sma']
    for col in redundant_cols:
        # may not exist due to salary differences
        try:
            del df[col]
        except KeyError:
            pass

    # only wants after 1992 due to sma
    df = df[df['year'] > 1992]

    if train:
        # Do not train on data after 2017 as we want to evaluate perfomance on this data
        df = df.loc[df['year'] < 2018]

        return df
    else:
        df = df.loc[df['year'] < 2020]
        return df


def full_mvp_featuredf(train, predict):
    """
    :param predict: Bool, true to get 2020 data to predict on
    :param train: Bool, true to create data to train on, false to create data to predict on
    :return: Creates dataframe of features read to predict/train for mvp
    """

    # get player data
    player_df = player_nosalary_features()
    player_df = teammate_features(player_df)

    # get team data
    team_df = team_features()

    df = player_df

    # get mvp data
    if not predict:
        mvp_df = mvp_features()
        # join data
        player_df.set_index('name_year', inplace=True)
        mvp_df.set_index('name_year', inplace=True)
        df = player_df.join(other=mvp_df, how='left')
        df.reset_index(inplace=True, drop=False)
        df = df.rename(columns={'index': 'name_year'})

    df.set_index('team_year', inplace=True)
    df = df.join(other=team_df, how='left')

    # delete nas
    df.fillna(0, inplace=True)

    # reset index
    df.reset_index(inplace=True)
    df = df.rename(columns={'index': 'team_year'})

    # delete uneeded rows
    del df['team_year_tmate']

    # only wants after 1984 when most advanced stats recorded
    df = df[df['year'] > 1983]

    if predict:
        df = df[df['year'] > 2019]
    else:
        df = df.loc[df['year'] < 2020]

    # remove player with low games played due to potential skewed advanced stats
    df = df[df['G'] > 3]
    df = df[df['MP'] > 2]

    if train:
        # Do not train on data after 2017 as we want to evaluate perfomance on this data
        df = df.loc[df['year'] < 2018]

        return df
    else:
        return df


if __name__ == "__main__":
    # Saves all training and predicting features
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\features\\"

    mvp_train_df = full_mvp_featuredf(True, False)
    mvp_train_df.to_csv(data_path + "features_train_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    mvp_pred_df = full_mvp_featuredf(False, False)
    mvp_pred_df.to_csv(data_path + "features_pred_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    mvp_2020_df = full_mvp_featuredf(False, True)
    mvp_2020_df.to_csv(data_path + "features_2020_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_train_df = full_salary_featuredf(True)
    salary_train_df.to_csv(data_path + "features_train_salary.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_pred_df = full_salary_featuredf(False)
    salary_pred_df.to_csv(data_path + "features_pred_salary.csv.gz", encoding='utf-8-sig', compression='gzip')
