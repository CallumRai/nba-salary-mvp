import pandas as pd


def team_abbrv(team_name):
    """
    :param team_name: Full team name (from team/salary dataframes)
    :return: Abbreviation used in player dataframe
    """

    # ordered list of team name and abbreviation
    full_team_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Bobcats', 'Charlotte Hornets',
                       'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons',
                       'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'Kansas City Kings',
                       'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat',
                       'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Jersey Nets', 'New Orleans Hornets',
                       'New Orleans Pelicans', 'New Orleans/Oklahoma City Hornets', 'New York Knicks',
                       'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns',
                       'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'San Diego Clippers',
                       'Seattle SuperSonics', 'Toronto Raptors', 'Utah Jazz', 'Vancouver Grizzlies',
                       'Washington Bullets', 'Washington Wizards']

    abbrv_names = ['ATL', 'BOS', 'BRK', 'CHA', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND',
                   'KCK', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NJN', 'NOH', 'NOP', 'NOK', 'NYK', 'OKC', 'ORL',
                   'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'SDC', 'SEA', 'TOR', 'UTA', 'VAN', 'WSB', 'WAS']

    # create dictionary respectively linking team name and abbreviation
    name_dict = {i: j for i, j in zip(full_team_names, abbrv_names)}

    return name_dict.get(team_name)


def merge_tots(ix_list, df, row):
    """
    :param ix_list: List of partial player indexes to delete
    :param df: Player dataframe
    :param row: A row in player dataframe
    :return: If player is traded changes team to team played most for in season and appends partial index list with
    partial row indexes to be deleted
    """

    ix = row.name

    if row['Tm'] == 'TOT':
        # create empty df to store traded players data in
        player_df = pd.DataFrame(columns=list(df.columns))

        # iterate through next rows to find other player data
        for x in range(len(df)):
            if ix+1+x == len(df):
                # deals with case end of df reached
                break
            elif df.loc[ix + 1 + x, 'Player'] == row['Player']:
                # appends traded players data to player_df
                player_df = player_df.append(df.iloc[ix + 1 + x], ignore_index=True, sort=False)
            else:
                break

        # sort by games played
        player_df.sort_values(by="G", inplace=True, ascending=False)
        player_df.reset_index(drop=True, inplace=True)

        # change tot to most played team
        df.at[ix, 'Tm'] = player_df.loc[0, 'Tm']

        # append to list of redundant partial rows
        ix_list.extend(player_df.loc[:, "Unnamed: 0"].tolist())
