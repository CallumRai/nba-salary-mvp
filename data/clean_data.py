import pandas as pd
import numpy as np
import os
from tools import *
from re import sub
from decimal import Decimal


def clean_player_data(playoffs, data_type):
    """
    :param playoffs: Bool, True for playoff data, False for regular season data
    :param data_type: Type of player data to be cleaned (based on raw file names)
    :return: Dataframe of raw data cleaned (blank row/columns removed, traded players dealt with etc.)
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    if playoffs:
        df = pd.read_csv(data_path + data_type + "_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')
    else:
        df = pd.read_csv(data_path + data_type + "_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    # get blank column list
    columns = list(df.columns)
    blank_cols = [col for col in columns if df[col].isnull().all()]

    # delete blank columns
    for col in blank_cols:
        del df[col]

    # delete rarely used gs columns (not in all possible dfs so potentially can cause a KeyError)
    try:
        del df["gs"]
    except KeyError:
        pass

    # get blank rows (where player is present)
    blank_rows = df[df['Player'].isnull()].index.tolist()

    # delete blank rows
    df.drop(blank_rows, inplace=True)

    # reset index to allow us to iterate through rows
    df.reset_index(drop=True, inplace=True)

    # list of partial rows of traded players to delete
    partial_index = []

    # run merge tots function to set traded players team name as most played team and append partial team rows to list
    df.apply(lambda row: merge_tots(partial_index, df, row), axis=1)

    # set index back to former indexes so correct rows can be dropped
    df.set_index("Unnamed: 0", inplace=True)
    df.drop(partial_index, inplace=True)

    # replace hall of fame star
    df['Player'] = df['Player'].apply(lambda name: name.replace("*", ""))

    # create name_year columns
    df['name_year'] = df.apply(lambda row: row['Player'] + "_" + str(row['year']), axis=1)

    # set name_year as index
    df.set_index('name_year', inplace=True)

    # set nans/blanks to 0
    df.replace("", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    return df


def clean_team_data(playoffs, data_type):
    """
    :param playoffs: Bool, True for playoff data, False for regular season data
    :param data_type: Type of team data to be cleaned (based on raw file names)
    :return: Dataframe of raw data cleaned (blank row/columns removed etc.)
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    if playoffs:
        df = pd.read_csv(data_path + data_type + "_po_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')
    else:
        df = pd.read_csv(data_path + data_type + "_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # get blank column list
    columns = list(df.columns)
    blank_cols = [col for col in columns if df.loc[5000:5002, col].isnull().any()]

    # delete blank columns
    for col in blank_cols:
        del df[col]

    # replace championship star
    df['Team'] = df['Team'].apply(lambda name: name.replace("*", ""))

    # abbreviate team names so it matches with that in the player dataframes
    df['Team'] = df['Team'].apply(lambda x: team_abbrv(x))

    # get blank rows (where team is not present)
    blank_rows = df[df['Team'].isnull()].index.tolist()

    # delete blank rows
    df.drop(blank_rows, inplace=True)

    # create team_year columns
    df['team_year'] = df.apply(lambda row: row['Team'] + "_" + str(row['year']), axis=1)

    # set team_year as index
    df.set_index('team_year', inplace=True)

    return df


def clean_salary_data():
    """
    :return: Dataframe of player salary data cleaned (salary converted to number etc.)
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    df = pd.read_csv(data_path + "salary_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # create name_year columns
    df['name_year'] = df.apply(lambda row: row['player'] + "_" + str(row['year']), axis=1)

    # set name_year as index
    df.set_index('name_year', inplace=True)

    # convert salary to number
    df["salary"] = df["salary"].apply(lambda x: Decimal(sub(r'[^\d.]', '', x)))

    return df


def clean_salary_cap_data():
    """
    :return: Dataframe of salary cap data year by year cleaned
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    df = pd.read_csv(data_path + "salary_cap_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # get year for season column
    df['Season'] = df['Season'].apply(lambda x: int(x[-4:]))

    # convert salary to number
    df["Salary Cap"] = df["Salary Cap"].apply(lambda x: Decimal(sub(r'[^\d.]', '', x)))

    # take the two relevant columns
    df = df.loc[:, ['Season', 'Salary Cap']]

    # rename columns
    new_col_names = ['year', 'cap']
    replace_col_dict = {i: j for i, j in zip(list(df.columns), new_col_names)}
    df = df.rename(columns=replace_col_dict)

    return df


def clean_winner_data():
    """
    :return: Dataframe of season winner data cleaned
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    df = pd.read_csv(data_path + "winner_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # get blank rows (where year isn't present)
    blank_rows = df[df["Year"].isnull()].index.tolist()

    # delete blank rows
    df.drop(blank_rows, inplace=True)

    # abbreviate team names like player dataframes
    team_cols = ['Champion', 'Runner-Up']
    for col in team_cols:
        df[col] = df[col].apply(lambda x: team_abbrv(x))

    # rename, lowercase the years column
    year_rename = {'Year': 'year'}
    df.rename(year_rename, inplace=True)

    return df


def clean_rookie_data():
    """
    :return: Dataframe with name_year and year of rookies
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"
    df = pd.read_csv(data_path + "rookies_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')

    # get blank rows (where player isn't present)
    blank_rows = df[df["Player"].isnull()].index.tolist()

    # delete blank rows
    df.drop(blank_rows, inplace=True)

    # replace hall of fame star
    df['Player'] = df['Player'].apply(lambda name: name.replace("*", ""))

    df['name_year'] = df.apply(lambda row: row['Player'] + "_" + str(row['year']), axis=1)

    rookie_df = df[['name_year', 'year']]

    return rookie_df


if __name__ == '__main__':
    # Saves all raw data cleaned
    data_path_clean = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\clean\\"
    data_path_raw = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"

    advanced_df = clean_player_data(False, "advanced")
    advanced_df.to_csv(data_path_clean + "advanced_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    per_df = clean_player_data(False, "per")
    per_df.to_csv(data_path_clean + "per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_per_df = clean_team_data(False, "team_per")
    team_per_df.to_csv(data_path_clean + "team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_advanced_df = clean_team_data(False, "team_advanced")
    team_advanced_df.to_csv(data_path_clean + "team_advanced_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    op_team_per_df = clean_team_data(False, "op_team_per")
    op_team_per_df.to_csv(data_path_clean + "op_team_per_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    advanced_po_df = clean_player_data(True, "advanced")
    advanced_po_df.to_csv(data_path_clean + "advanced_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    per_po_df = clean_player_data(True, "per")
    per_po_df.to_csv(data_path_clean + "per_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_per_po_df = clean_team_data(True, "team_per")
    team_per_po_df.to_csv(data_path_clean + "team_per_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    op_team_per_po_df = clean_team_data(True, "op_team_per")
    op_team_per_po_df.to_csv(data_path_clean + "op_team_per_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_df = clean_salary_data()
    salary_df.to_csv(data_path_clean + "salary_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_cap_df = clean_salary_cap_data()
    salary_cap_df.to_csv(data_path_clean + "salary_cap_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_misc_po_df = clean_team_data(True, "team_misc")
    team_misc_po_df.to_csv(data_path_clean + "team_misc_po_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    winner_df = clean_winner_data()
    winner_df.to_csv(data_path_clean + "winner_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    rookie_df = clean_rookie_data()
    rookie_df.to_csv(data_path_clean + "rookies_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # playoff series data and mvp already clean
    playoff_df = pd.read_csv(data_path_raw + "playoff_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')
    playoff_df.to_csv(data_path_clean + "playoff_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    mvp_df = pd.read_csv(data_path_raw + "mvp_raw.csv.gz", index_col=0, encoding='utf-8-sig', compression='gzip')
    mvp_df.to_csv(data_path_clean + "mvp_clean.csv.gz", encoding='utf-8-sig', compression='gzip')
