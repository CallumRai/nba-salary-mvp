import pandas as pd
import os
import math
from difflib import get_close_matches
import numpy as np

def mvp_ranking():
    """
    :return: Dataframe containing, name year, MVP share (predicted and acutal), MVP rankings (predicted and acutal)
    age and team
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'
    df = pd.read_csv(data_path + "pred_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    # get predicted ranking
    df['pred_mvp_rank'] = df.groupby(['year'])['pred_mvp_share'].rank(ascending=False)

    # get actual mvp share and ranking
    df = df.rename(columns={'Share':'actual_mvp_share'})
    df['actual_mvp_rank'] = df.groupby(['year'])['actual_mvp_share'].rank(ascending=False)

    # get required columns
    df = df[['name_year', 'pred_mvp_share', 'pred_mvp_rank', 'actual_mvp_share', 'actual_mvp_rank', 'Age', 'team_year',
             'year']]

    df.set_index('name_year', inplace=True)

    return df


def all_nba():
    """
    :return: Dataframe containing, name year, predicted all nba team
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'
    df = pd.read_csv(data_path + "pred_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    # get ranking
    df['mvp_rank'] = df.groupby(['year'])['pred_mvp_share'].rank(ascending=False)

    # get position type (c-centre, g-guard, f-forward)
    def get_position(row):
        if row['Pos_C'] == 1 or row['Pos_C-'] == 1:
            return 'c'
        elif row['Pos_PG'] == 1 or row['Pos_SG'] == 1:
            return 'g'
        else:
            return 'f'

    # create position type col
    df['pos_type'] = df.apply(lambda row: get_position(row), axis=1)

    # get ranking for each position type
    df['mvp_pos_rank'] = df.groupby(['year', 'pos_type'])['pred_mvp_share'].rank(ascending=False)

    # get all nba team
    def get_team(row):
        if row['pos_type'] == 'c':
            if row['mvp_pos_rank'] < 4:
                return row['mvp_pos_rank']
            else:
                return 0
        else:
            if row['mvp_pos_rank'] < 7:
                return math.ceil(row['mvp_pos_rank'] / 2)
            else:
                return 0

    # get all_nba col
    df['all_nba'] = df.apply(lambda row: get_team(row), axis=1)

    # get only relevant columns
    df = df[['name_year', 'all_nba']]

    df.set_index('name_year', inplace=True)

    return df


def salary():
    """
    :return: Dataframe containing, name_year, predicted salary, pred ranking, actual salary, difference (actual-pred),
    cap and %of cap
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "\\predictions\\pred_salary.csv.gz", encoding='utf-8-sig', compression='gzip')

    # get ranking
    df['pred_salary_rank'] = df.groupby(['year'])['pred_salary'].rank(ascending=False)

    # load actual salary
    salary_df = pd.read_csv(data_path + "\\clean\\salary_clean.csv.gz", encoding='utf-8-sig', compression='gzip')

    # set indexes as name_year
    df.set_index('name_year', inplace=True)
    salary_df.set_index('name_year', inplace=True)

    def get_salary(data, row):
        """
        :param data: Dataframe with salary of each player
        :param row: Player row to get salary for
        :return: Actual salary of player in the year
        """

        try:
            money = salary_df.loc[row.name, 'salary']
        except KeyError:
            # deals with fact name in different df may be different
            # get year
            year = int(row.name[-4:])
            year_df = data[data['year'] == year].copy()

            # convert index to list
            name_year_list = year_df.index.tolist()

            # get respective index from df to salary df (may not be the same due to accent differences)
            salary_index = get_close_matches(row.name, name_year_list, cutoff=0.8)

            try:
                money = salary_df.loc[salary_index[0], 'salary']
            except IndexError:
                return np.nan

        return money

    # create column of actual salary
    df['salary'] = df.apply(lambda row: get_salary(salary_df, row), axis=1)

    # get ranking
    df['actual_salary_rank'] = df.groupby(['year'])['salary'].rank(ascending=False)

    # create difference between actual salary and predicted
    df['difference'] = df.apply(lambda row: row['salary'] - row['pred_salary'], axis=1)

    df['actual_%ofcap'] = df.apply(lambda row: row['salary']/row['cap'], axis=1)

    # get required columns
    df = df[['pred_salary', 'salary', 'difference', 'pred_salary_rank', 'actual_salary_rank', 'pred_%ofcap',
             'cap', 'actual_%ofcap']]

    return df

def mvp_2020():
    """
    :return: Dataframe containing, name year, MVP share (predicted and acutal), MVP rankings (predicted and acutal)
    age and team for 2019-20 season
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'
    df = pd.read_csv(data_path + "2020_mvp.csv.gz", encoding='utf-8-sig', compression='gzip')

    # get predicted ranking
    df['pred_mvp_rank'] = df.groupby(['year'])['pred_mvp_share'].rank(ascending=False)

    # get required columns
    df = df[['name_year', 'pred_mvp_share', 'pred_mvp_rank', 'Age', 'team_year',
             'year']]

    df.set_index('name_year', inplace=True)

    return df

if __name__ == '__main__':
    # save all above into one csv
    mvp_df = mvp_ranking()
    all_nba_df = all_nba()
    salary_df = salary()

    # delete duplicates
    mvp_df.drop(mvp_df[mvp_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    all_nba_df.drop(all_nba_df[all_nba_df.index.duplicated(keep=False)].index.tolist(), inplace=True)
    salary_df.drop(salary_df[salary_df.index.duplicated(keep=False)].index.tolist(), inplace=True)

    df = pd.concat([mvp_df, all_nba_df, salary_df], axis=1,sort=False)

    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + '\\predictions\\'
    df.to_csv(data_path + "results.csv.gz", encoding='utf-8-sig', compression='gzip')

    future_mvp = mvp_2020()
    future_mvp.to_csv(data_path + "results_2020.csv.gz", encoding='utf-8-sig', compression='gzip')