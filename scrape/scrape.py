from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException


def scrape_player_data(playoffs, data_type):
    """
    :param playoffs: Bool, True to scrape playoff data, False to scrape regular season data
    :param data_type: Type of player data to scrape (dependent on basketball-reference url name)
    :return: Dataframe of player data from 1980 to 2020
    """

    # get list of years to scrape data from
    if not playoffs:
        years = list(range(1980, 2021))
    else:
        years = list(range(1980,2020))

    # decides url
    if playoffs:
        period = "playoffs"
    else:
        period = "leagues"

    # get html
    url = 'https://www.basketball-reference.com/{}'.format(period) + '/NBA_2019_{}.html'.format(data_type)
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')

    # get headers
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[0].findAll('th')]

    # delete redundant rk row
    headers.remove('Rk')
    # add year col
    headers.append('year')

    # initialise df
    df = pd.DataFrame(columns=headers)

    # iterate through each year
    for year in years:
        # get html year page
        url = 'https://www.basketball-reference.com/{}/'.format(period) + 'NBA_{}_'.format(year) + "{}.html" \
            .format(data_type)
        html = urlopen(url)
        soup = BeautifulSoup(html, features='lxml')

        # get player data avoiding reader row
        rows = soup.findAll('tr')[1:]
        player_stats = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(player_stats, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    return df


def scrape_team_data(playoffs, data_type):
    """
    :param playoffs: Bool, True to scrape playoff data, False to scrape regular season data
    :param data_type: Type of player data to scrape (dependent on basketball-reference url name)
    :return: Dataframe of team data from 1980 to 2020
    """

    if not playoffs:
        years = list(range(1980, 2021))
    else:
        years = list(range(1980,2020))

    # decides url
    if playoffs:
        period = "playoffs"
    else:
        period = "leagues"

    # load team page to get headers
    url = "https://www.basketball-reference.com/{}".format(period) + "/NBA_2019.html"

    # load dynamic js
    driver = webdriver.Chrome(executable_path='chromedriver.exe')
    driver.get(url)

    # wait for page to load (delay till required id appears)
    delay = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, data_type)))

    # get headers
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    table = soup.find('table', {"id": data_type})
    headers = [th.getText() for th in table.find('tr').findAll('th')]

    # remove redundant rk row
    headers.remove('Rk')
    # add year col
    headers.append('year')

    # initialise df
    df = pd.DataFrame(columns=headers)

    for year in years:
        # get team page html
        url = "https://www.basketball-reference.com/{}".format(period) + "/NBA_{}.html".format(year)

        # use this due to dynamic js api required
        driver.get(url)

        # wait for page to load (delay till required id appears
        delay = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, data_type)))

        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")

        # get player data avoiding reader row
        table = soup.find('table', {"id": data_type})
        rows = table.find_all('tr')[1:]
        team_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(team_stats, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    # ends all chrome browsers
    driver.quit()

    return df


def scrape_playoff_misc():
    """
    :return: Dataframe of playoff team data from 1980 to 2020
    """

    # get list of years to scrape data from
    years = list(range(1980, 2020))

    # load team page to get headers
    url = "https://www.basketball-reference.com/playoffs/NBA_2019.html"

    # load dynamic js
    driver = webdriver.Chrome(executable_path='chromedriver.exe')
    driver.get(url)

    # wait for page to load (delay till required id appears
    delay = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "misc")))

    # get headers
    html = driver.page_source
    soup = BeautifulSoup(html, "lxml")
    table = soup.find('table', {"id": "misc"})
    headers = [th.getText() for th in table.findAll('tr')[1].findAll('th')]
    # remove redundant rk row
    headers.remove('Rk')
    # add year col
    headers.append('year')

    # initialise df
    df = pd.DataFrame(columns=headers)

    for year in years:
        # get year page html
        url = "https://www.basketball-reference.com/playoffs/NBA_{}.html".format(year)

        # use this due to dynamic js api required
        driver.get(url)

        # wait for page to load (delay till required id appears
        delay = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.ID, "misc")))

        html = driver.page_source
        soup = BeautifulSoup(html, "lxml")

        # get player data avoiding reader row
        table = soup.find('table', {"id": "misc"})
        rows = table.find_all('tr')[2:]
        team_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(team_stats, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    # ends all chrome browsers
    driver.quit()

    return df


def scrape_salaries():
    """
    :return: Dataframe of player salaries from 1991 to 2020
    """

    # get list of years to scrape data from
    years = list(range(1991, 2020))

    # get headers
    headers = ["player", "salary", "year"]

    # initliase df
    df = pd.DataFrame(columns=headers)

    for year in years:
        # get html
        url = "https://hoopshype.com/salaries/players/{}".format(year - 1) + "-{}/".format(year)
        html = urlopen(url)
        soup = BeautifulSoup(html, features='lxml')

        # get all table name and salary
        table = soup.find('table', {'class': 'hh-salaries-ranking-table hh-salaries-table-sortable responsive'})
        rows = table.find_all('tr')[1:]
        data = [[td.getText() for td in rows[i].findAll('td')][1:3]
                for i in range(len(rows))]

        # remove regex
        data = [[string.replace("\n", "").replace("\t", "") for string in row] for row in data]

        # create temp year df
        year_df = pd.DataFrame(data, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    return df


def scrape_salary_cap():
    """
    :return: Dataframe of year's salary cap 1984 to 2028
    """

    # get html
    url = "https://basketball.realgm.com/nba/info/salary_cap"
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')

    # get headers and data
    table = soup.find('table', {'class': 'basketball compact'})
    headers = [th.getText() for th in table.find_all('tr')[1].findAll('th')]
    rows = table.find_all('tr')[2:]
    data = [[td.getText() for td in rows[i].findAll('td')][2:]
            for i in range(len(rows))]

    # create df
    df = pd.DataFrame(data, columns=headers)

    return df


def scrape_team_advanced():
    """
    :return: Dataframe of advanced team data from 1980 to 2020
    """

    # get list of years to scrape data from
    years = list(range(1980, 2021))

    # get list of headers
    url = 'https://www.basketball-reference.com/leagues/NBA_2020_ratings.html'
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')]
    # delete redundant rk row
    headers.remove('Rk')
    # add year col
    headers.append('year')

    # initialise df
    df = pd.DataFrame(columns=headers)

    # iterate through each year
    for year in years:
        # get html
        url = 'https://www.basketball-reference.com/leagues/NBA_{}_ratings.html'.format(year)
        html = urlopen(url)
        soup = BeautifulSoup(html, features='lxml')

        # get data avoiding reader row
        rows = soup.findAll('tr')[2:]
        team_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(team_stats, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    return df


def scrape_playoff_position():
    """
    :return: Dataframe with team, year and position in playoffs
    """

    # get html
    url = "https://www.basketball-reference.com/playoffs/series.html"

    html = urlopen(url)
    soup = BeautifulSoup(html, features='html.parser')

    # create df
    headers = ['year', 'round', 'team_1', 'team_2']
    df = pd.DataFrame(columns=headers)

    # get data
    rows = soup.find_all('tr')[2:]

    # append only data required to dataframe
    for row in rows:
        # get data from rows
        row_data = []
        links = row.findAll('a')
        for link in links:
            row_data.append(link.text)

        # append only the relevant data
        try:
            data = [row_data[0], row_data[2], row_data[5], row_data[6]]
            data_dict = {i: j for i, j in zip(list(df.columns), data)}
            df = df.append(data_dict, ignore_index=True, sort=False)
        except IndexError:
            # may be errors caused by blank rows
            pass

    return df


def scrape_winner():
    """
    :return: Dataframe of nba winners
    """

    # get html
    url = "https://www.basketball-reference.com/playoffs/"
    html = urlopen(url)
    soup = BeautifulSoup(html, features='html.parser')

    # create df
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')]
    df = pd.DataFrame(columns=headers)

    # get data
    rows = soup.find_all('tr')[2:]
    for row in rows:
        # get data from rows
        row_data = []
        links = row.findAll('a')
        for link in links:
            row_data.append(link.text)

        # get only the relevant dat
        try:
            data_dict = {i: j for i, j in zip(list(df.columns), row_data)}
            df = df.append(data_dict, ignore_index=True, sort=False)
        except IndexError:
            # may be errors caused by blank rows
            pass

    return df


def scrape_mvp():
    """
    :return: Dataframe with mvp voting history
    """

    # get list of years to scrape data from
    years = list(range(1980, 2020))

    # get html
    url = "https://www.basketball-reference.com/awards/awards_2019.html"
    html = urlopen(url)
    soup = BeautifulSoup(html, features='html.parser')

    # get headers
    table = soup.find('table', {'id': 'mvp'})
    headers = [th.getText() for th in table.findAll('tr', limit=2)[1].findAll('th')]

    # delete redundant rank row
    headers.remove('Rank')

    # initliase empty dataframe
    df = pd.DataFrame(columns=headers)

    for year in years:
        # get html
        url = "https://www.basketball-reference.com/awards/awards_{}.html".format(year)
        html = urlopen(url)
        soup = BeautifulSoup(html, features='html.parser')

        # get player data avoiding reader row
        table = soup.find('table', {'id': 'mvp'})
        rows = table.findAll('tr')[2:]
        team_stats = [[td.getText() for td in rows[i].findAll('td')]
                      for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(team_stats, columns=headers)

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    return df

def scrape_rookies():
    """
    :return: Dataframe with year and list of rookies
    """

    # get list of years to scrape data from
    years = list(range(1980, 2020))

    # get html
    url = 'https://www.basketball-reference.com/leagues/NBA_2019_rookies.html'
    html = urlopen(url)
    soup = BeautifulSoup(html, features='lxml')

    # get headers
    headers = [th.getText() for th in soup.findAll('tr', limit=2)[1].findAll('th')]

    # delete redundant rk row
    headers.remove('Rk')
    # add year col
    headers.append('year')

    # initialise df
    df = pd.DataFrame(columns=headers)

    # iterate through each year
    for year in years:
        # get html year page
        url = 'https://www.basketball-reference.com/leagues/NBA_{}_rookies.html'.format(year)

        html = urlopen(url)
        soup = BeautifulSoup(html, features='lxml')

        # get player data avoiding reader row
        rows = soup.findAll('tr')[2:]
        player_stats = [[td.getText() for td in rows[i].findAll('td')]
                        for i in range(len(rows))]

        # create temporary df for year (w/o year col)
        year_df = pd.DataFrame(player_stats, columns=headers[:-1])

        # create year column
        year_df['year'] = year

        # append to full df
        df = df.append(year_df, ignore_index=True, sort=False)

    return df


if __name__ == '__main__':
    # Scrapes and saves all data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')) + "\\raw\\"

    advanced_df = scrape_player_data(False, "advanced")
    advanced_df.to_csv(data_path + "advanced_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    per_df = scrape_player_data(False, "per_game")
    per_df.to_csv(data_path + "per_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    op_team_per_df = scrape_team_data(False, "opponent-stats-per_game")
    op_team_per_df.to_csv(data_path + "op_team_per_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_per_df = scrape_team_data(False, "team-stats-per_game")
    team_per_df.to_csv(data_path + "team_per_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_advanced_df = scrape_team_advanced()
    team_advanced_df.to_csv(data_path + "team_advanced_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    advanced_po_df = scrape_player_data(True, "advanced")
    advanced_po_df.to_csv(data_path + "advanced_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    per_po_df = scrape_player_data(True, "per_game")
    per_po_df.to_csv(data_path + "per_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_per_po_df = scrape_team_data(True, "team-stats-per_game")
    team_per_po_df.to_csv(data_path + "team_per_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    op_team_per_po_df = scrape_team_data(True, "opponent-stats-per_game")
    op_team_per_po_df.to_csv(data_path + "op_team_per_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    playoff_df = scrape_playoff_position()
    playoff_df.to_csv(data_path + "playoff_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    team_misc_po_df = scrape_playoff_misc()
    team_misc_po_df.to_csv(data_path + "team_misc_po_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_df = scrape_salaries()
    salary_df.to_csv(data_path + "salary_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    salary_cap_df = scrape_salary_cap()
    salary_cap_df.to_csv(data_path + "salary_cap_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    winners_df = scrape_winner()
    winners_df.to_csv(data_path + "winner_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    mvp_df = scrape_mvp()
    mvp_df.to_csv(data_path + "mvp_raw.csv.gz", encoding='utf-8-sig', compression='gzip')

    rookies_df = scrape_rookies()
    rookies_df.to_csv(data_path + "rookies_raw.csv.gz", encoding='utf-8-sig', compression='gzip')