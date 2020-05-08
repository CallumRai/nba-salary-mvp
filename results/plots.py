import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr
from tools import *
import locale

plt.rcParams["figure.figsize"] = (9.2, 7)


def plot_salary():
    """
    :return: Plots a scatter graph of actual vs predicted salary in 2018 and 2019 season
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get years to evaulate off only
    eval_df = df[df['year'] > 2017].copy()

    # drop rows where salary data isnt present
    blank_rows = eval_df[eval_df['salary'].isnull()].index.tolist()
    eval_df.drop(blank_rows, inplace=True)

    # get list of actual and predicted salary, name and year
    actual_salary = eval_df['salary'].tolist()
    pred_salary = eval_df['pred_salary'].tolist()
    year = eval_df.apply(lambda row: int(row.name[-4:]), axis=1).tolist()
    name = eval_df.apply(lambda row: row.name[:-5], axis=1).tolist()
    age = eval_df.apply(lambda row: int(row['Age']), axis=1).tolist()
    team = eval_df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()
    pred_rank = ordinal(eval_df['pred_salary_rank'].tolist())
    actual_rank = ordinal(eval_df['actual_salary_rank'].tolist())

    # get PMCC test for linear association (and p value)
    pmcc, p = pearsonr(actual_salary, pred_salary)

    # divide all salaries by 1m to scale for graph
    actual_salary = np.array([x / 1000000 for x in actual_salary])
    pred_salary = np.array([x / 1000000 for x in pred_salary])

    fig, ax = plt.subplots()

    # plot scatter graph of actual salary vs predicted
    scatter, = ax.plot(actual_salary, pred_salary, linestyle='', marker='x', markersize=5)

    # plot an x=y dashed line
    x = np.linspace(0, 40, 2)
    ax.plot(x, x, 'p--')

    # display PMCC
    ax.text(0.17, 0.95, 'PMCC = ' + str(round(pmcc, 3)), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    # set axes limits
    ax.set_xlim(left=0, right=40)
    ax.set_ylim(bottom=0, top=40)
    ax.set_aspect('equal', adjustable='box')

    # convert salaries to money
    locale.setlocale(locale.LC_ALL, 'en_US.utf-8')
    pred_salary = ['${:,}'.format(int(x * 1000000)) for x in pred_salary]
    actual_salary = ['${:,}'.format(int(x * 1000000)) for x in actual_salary]

    def annot_text(ix):
        # sets text for hover annotation
        text = f"Name: {name[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
               f"Predicted Salary: {pred_salary[ix]} ({pred_rank[ix]} in the league)\n" \
               f"Actual Salary: {actual_salary[ix]} ({actual_rank[ix]} in the league)\nYear: {year[ix]}"
        return text

    # display annotations on hover
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, scatter, ax, fig))

    # label chart
    ax.set_xlabel('Actual salary (in $1 millon)')
    ax.set_ylabel('Predicted salary (in $1 million)')

    plt.show()


def plot_mvp_eval(year):
    """
    :return: Plots two bar charts of MVP canidates in given year
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get years only
    year_df = df[df['year'] == year].copy()

    # drop rows where mvp data isnt present
    blank_rows = year_df[year_df['pred_mvp_share'].isnull()].index.tolist()
    year_df.drop(blank_rows, inplace=True)

    # initiliase chart
    fig, (ax1, ax2) = plt.subplots(2)

    def get_data(df):
        pred_share = df['pred_mvp_share'].tolist()[:10]
        actual_share = df['actual_mvp_share'].tolist()[:10]
        pred_rank = ordinal(df['pred_mvp_rank'].tolist()[:10])
        actual_rank = ordinal(df['actual_mvp_rank'].tolist()[:10])
        names = df.apply(lambda row: row.name[:-5], axis=1).tolist()[:10]
        age = df.apply(lambda row: int(row['Age']), axis=1).tolist()[:10]
        team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()[:10]
        # get last name only
        names_last = [x.split(" ")[1] for x in names]

        return pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last

    def get_hover(df, ax, fig, bar):
        pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(df)

        def annot_text(ix):
            if int(actual_rank[ix][:-2]) < 11:
                text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                       f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                       f"Actual Share: {round(actual_share[ix], 3)} ({actual_rank[ix]} in the league)"
            else:
                text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                       f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                       f"Actual Share: {round(actual_share[ix], 3)}"
            return text

        fig.canvas.mpl_connect("motion_notify_event", hover_annot_bar(annot_text, ax, fig, bar))

    # sort by predicted mvp share
    year_df = year_df.sort_values(by='pred_mvp_share', ascending=False)
    pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(year_df)

    bar_pred = ax1.bar(names, pred_share, color='#ffa500')
    ax1.set_xticklabels(names_last, rotation=30)

    # get hover info
    get_hover(year_df, ax1, fig, bar_pred)

    # sort by predicted mvp share
    year_df = year_df.sort_values(by='actual_mvp_share', ascending=False)
    pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(year_df)

    # plot chart
    bar_actual = ax2.bar(names, actual_share)
    ax2.set_xticklabels(names_last, rotation=30)
    plt.tight_layout()

    # get hover info
    get_hover(year_df, ax2, fig, bar_actual)

    # set axis lims
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # create labels
    ax1.set_ylabel('Predicted MVP Share')
    ax2.set_ylabel('Actual MVP Share')

    plt.show()


def plot_mvp():
    """
    :return: Plots two bar charts of MVP canidates in user defined year
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get user defined year
    year = year_input(False)

    # get years only
    year_df = df[df['year'] == year].copy()

    # drop rows where mvp data isnt present
    blank_rows = year_df[year_df['pred_mvp_share'].isnull()].index.tolist()
    year_df.drop(blank_rows, inplace=True)

    # initiliase chart
    fig, (ax1, ax2) = plt.subplots(2)

    def get_data(df):
        pred_share = df['pred_mvp_share'].tolist()[:10]
        actual_share = df['actual_mvp_share'].tolist()[:10]
        pred_rank = ordinal(df['pred_mvp_rank'].tolist()[:10])
        actual_rank = ordinal(df['actual_mvp_rank'].tolist()[:10])
        names = df.apply(lambda row: row.name[:-5], axis=1).tolist()[:10]
        age = df.apply(lambda row: int(row['Age']), axis=1).tolist()[:10]
        team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()[:10]
        # get last name only
        names_last = [x.split(" ")[1] for x in names]

        return pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last

    def get_hover(df, ax, fig, bar):
        pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(df)

        def annot_text(ix):
            if int(actual_rank[ix][:-2]) < 11:
                text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                       f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                       f"Actual Share: {round(actual_share[ix], 3)} ({actual_rank[ix]} in the league)"
            else:
                text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                       f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                       f"Actual Share: {round(actual_share[ix], 3)}"
            return text

        fig.canvas.mpl_connect("motion_notify_event", hover_annot_bar(annot_text, ax, fig, bar))

    # sort by predicted mvp share
    year_df = year_df.sort_values(by='pred_mvp_share', ascending=False)
    pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(year_df)

    bar_pred = ax1.bar(names, pred_share, color='#ffa500')
    ax1.set_xticklabels(names_last, rotation=30)

    # get hover info
    get_hover(year_df, ax1, fig, bar_pred)

    # sort by predicted mvp share
    year_df = year_df.sort_values(by='actual_mvp_share', ascending=False)
    pred_share, actual_share, pred_rank, actual_rank, names, age, team, names_last = get_data(year_df)

    # plot chart
    bar_actual = ax2.bar(names, actual_share)
    ax2.set_xticklabels(names_last, rotation=30)
    plt.tight_layout()

    # get hover info
    get_hover(year_df, ax2, fig, bar_actual)

    # set axis lims
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    # create labels
    ax1.set_ylabel('Predicted MVP Share')
    ax2.set_ylabel('Actual MVP Share')

    plt.show()


def plot_year_salary():
    """
    :return: Plots two bar charts of top salary in user defined year
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get user defined year
    year = year_input(True)

    # get years only
    year_df = df[df['year'] == year].copy()

    # drop rows where salary data isnt present
    blank_rows = year_df[year_df['salary'].isnull()].index.tolist()
    year_df.drop(blank_rows, inplace=True)

    # initiliase chart
    fig, (ax1, ax2) = plt.subplots(2)

    def get_data(df):
        pred_salary_m = df.apply(lambda row: row['pred_salary'] / 1000000, axis=1).tolist()[:10]
        actual_salary_m = df.apply(lambda row: row['salary'] / 1000000, axis=1).tolist()[:10]
        pred_rank = ordinal(df['pred_salary_rank'].tolist()[:10])
        actual_rank = ordinal(df['actual_salary_rank'].tolist()[:10])
        names = df.apply(lambda row: row.name[:-5], axis=1).tolist()[:10]
        age = df.apply(lambda row: int(row['Age']), axis=1).tolist()[:10]
        team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()[:10]
        # get last name only
        names_last = [x.split(" ")[1] for x in names]
        # convert salaries to money
        actual_salary = df.apply(lambda row: '${:,}'.format(int(row['salary'])), axis=1).tolist()
        pred_salary = df.apply(lambda row: '${:,}'.format(int(row['pred_salary'])), axis=1).tolist()

        return pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, actual_salary_m

    def get_hover(df, ax, fig, bar):
        pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, actual_salary_m = get_data(
            df)

        def annot_text(ix):
            # add suffix to rank
            text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Salary: {pred_salary[ix]} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Salary: {actual_salary[ix]} ({actual_rank[ix]} in the league)"

            return text

        fig.canvas.mpl_connect("motion_notify_event", hover_annot_bar(annot_text, ax, fig, bar))

    # sort by pred salary
    year_df = year_df.sort_values(by='pred_salary', ascending=False)
    pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, actual_salary_m = get_data(
        year_df)
    bar_pred = ax1.bar(names_last, pred_salary_m, color='#ffa500')
    ax1.set_xticklabels(names_last, rotation=30)
    get_hover(year_df, ax1, fig, bar_pred)

    # plot chart
    year_df = year_df.sort_values(by='salary', ascending=False)
    pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, actual_salary_m = get_data(
        year_df)
    bar_actual = ax2.bar(names_last, actual_salary_m)
    ax2.set_xticklabels(names_last, rotation=30)
    plt.tight_layout()
    get_hover(year_df, ax2, fig, bar_actual)

    # set axis limits
    max_lim = max([max(actual_salary_m), max(pred_salary_m)]) * 1.2
    ax1.set_ylim(0, max_lim)
    ax2.set_ylim(0, max_lim)

    # create labels
    ax1.set_ylabel('Predicted salary (in $1 millon)')
    ax2.set_ylabel('Actual salary (in $1 millon)')

    plt.show()


def plot_over_under_paid():
    """
    :return: Plots two bar charts of top salary in user defined year
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get user defined year
    year = year_input(True)

    # get years only
    year_df = df[df['year'] == year].copy()

    # drop rows where salary data isnt present
    blank_rows = year_df[year_df['salary'].isnull()].index.tolist()
    year_df.drop(blank_rows, inplace=True)

    # initiliase chart
    fig, (ax1, ax2) = plt.subplots(2)

    def get_data(df):
        pred_salary_m = df.apply(lambda row: row['pred_salary'] / 1000000, axis=1).tolist()[:10]
        actual_salary_m = df.apply(lambda row: row['salary'] / 1000000, axis=1).tolist()[:10]
        difference_m = df.apply(lambda row: abs(row['difference'] / 1000000), axis=1).tolist()[:10]
        pred_rank = ordinal(df['pred_salary_rank'].tolist()[:10])
        actual_rank = ordinal(df['actual_salary_rank'].tolist()[:10])
        names = df.apply(lambda row: row.name[:-5], axis=1).tolist()[:10]
        age = df.apply(lambda row: int(row['Age']), axis=1).tolist()[:10]
        team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()[:10]
        # get last name only
        names_last = [x.split(" ")[1] for x in names]
        # convert salaries to money
        actual_salary = df.apply(lambda row: '${:,}'.format(int(row['salary'])), axis=1).tolist()
        pred_salary = df.apply(lambda row: '${:,}'.format(int(row['pred_salary'])), axis=1).tolist()
        difference = df.apply(lambda row: '${:,}'.format(int(row['difference'])), axis=1).tolist()

        return pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, \
               actual_salary_m, difference, difference_m

    def get_hover(df, ax, fig, bar):
        pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, \
        actual_salary_m, difference, difference_m = get_data(df)

        def annot_text(ix):
            # add suffix to rank
            text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Salary: {pred_salary[ix]} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Salary: {actual_salary[ix]} ({actual_rank[ix]} in the league)\n" \
                   f"Difference: {difference[ix]}"

            return text

        fig.canvas.mpl_connect("motion_notify_event", hover_annot_bar(annot_text, ax, fig, bar))

    # over
    year_df = year_df.sort_values(by='difference', ascending=False)
    pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, \
    actual_salary_m, difference, difference_m = get_data(year_df)
    bar_over = ax1.bar(names_last, difference_m, color='#f01f0c')
    ax1.set_xticklabels(names_last, rotation=30)
    get_hover(year_df, ax1, fig, bar_over)

    # get max overpay
    max_over = max(difference_m)

    # under
    year_df = year_df.sort_values(by='difference', ascending=True)
    pred_salary, actual_salary, pred_rank, actual_rank, names, age, team, names_last, pred_salary_m, \
    actual_salary_m, difference, difference_m = get_data(year_df)
    bar_under = ax2.bar(names_last, difference_m, color='#069427')
    ax2.set_xticklabels(names_last, rotation=30)
    plt.tight_layout()
    get_hover(year_df, ax2, fig, bar_under)

    # set axis limits
    max_lim = max([max(difference_m), max_over]) * 1.2
    ax1.set_ylim(0, max_lim)
    ax2.set_ylim(0, max_lim)

    # create labels
    ax1.set_ylabel('Amount overpaid (in $1 millon)')
    ax2.set_ylabel('Amount underpaid (in $1 millon)')

    plt.show()


def plot_player_salary():
    """
    :return: Plots a players actual vs predicted salary as a time series
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get name column
    df['name'] = df.apply(lambda row: row.name[:-5], axis=1)

    # get user defined playter
    names = df['name'].tolist()
    player = name_input(names)

    player_df = df[df['name'] == player].copy()

    # drop rows where salary data isnt present
    blank_rows = player_df[player_df['salary'].isnull()].index.tolist()
    player_df.drop(blank_rows, inplace=True)

    # sort by year
    player_df = player_df.sort_values('year')

    # get relevant data
    pred_salary_m = player_df.apply(lambda row: row['pred_salary'] / 1000000, axis=1).tolist()
    actual_salary_m = player_df.apply(lambda row: row['salary'] / 1000000, axis=1).tolist()
    pred_rank = ordinal(player_df['pred_salary_rank'].tolist())
    actual_rank = ordinal(player_df['actual_salary_rank'].tolist())
    names = player_df.apply(lambda row: row.name[:-5], axis=1).tolist()
    age = player_df.apply(lambda row: int(row['Age']), axis=1).tolist()
    team = player_df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()
    # convert salaries to money
    actual_salary = player_df.apply(lambda row: '${:,}'.format(int(row['salary'])), axis=1).tolist()
    pred_salary = player_df.apply(lambda row: '${:,}'.format(int(row['pred_salary'])), axis=1).tolist()
    years = player_df.apply(lambda row: int(row['year']), axis=1).tolist()

    fig, ax = plt.subplots()

    # plot data
    pred, = ax.plot(years, pred_salary_m, label='Predicted Salary', marker='D')
    actual, = ax.plot(years, actual_salary_m, label='Actual Salary', marker='D')
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Salary (in $1 millon)")
    plt.xticks(years)
    if len(years)>10:
        xticks = plt.gca().xaxis.get_major_ticks()
        for i in range(len(xticks)):
            if i % 2 != 0:
                xticks[i].set_visible(False)
    ax.legend(frameon=False)

    def annot_text(ix):
        # add suffix to rank
        text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
               f"Predicted Salary: {pred_salary[ix]} ({pred_rank[ix]} in the league)\n" \
               f"Actual Salary: {actual_salary[ix]} ({actual_rank[ix]} in the league)\n" \
               f"Year: {years[ix]}"

        return text

    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, pred, ax, fig))
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, actual, ax, fig))

    plt.show()


def plot_player_mvp():
    """
    :return: Plots a players actual vs predicted mvp ranking as a time series
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # get name column
    df['name'] = df.apply(lambda row: row.name[:-5], axis=1)

    # get user defined playter
    names = df['name'].tolist()
    player = name_input(names)

    player_df = df[df['name'] == player].copy()

    # drop rows where salary data isnt present
    blank_rows = player_df[player_df['pred_mvp_share'].isnull()].index.tolist()
    player_df.drop(blank_rows, inplace=True)

    # sort by year
    player_df = player_df.sort_values('year')

    pred_share = player_df['pred_mvp_share'].tolist()
    actual_share = player_df['actual_mvp_share'].tolist()
    pred_rank = ordinal(player_df['pred_mvp_rank'].tolist())
    actual_rank = ordinal(player_df['actual_mvp_rank'].tolist())
    names = player_df.apply(lambda row: row.name[:-5], axis=1).tolist()
    age = player_df.apply(lambda row: int(row['Age']), axis=1).tolist()
    team = player_df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()
    years = player_df.apply(lambda row: int(row['year']), axis=1)
    years = player_df['year'].tolist()

    fig, ax = plt.subplots()

    # plot data
    pred, = ax.plot(years, pred_share, label='Predicted MVP Share', marker='D')
    actual, = ax.plot(years, actual_share, label='Actual MVP Share', marker='D')
    ax.set_ylim(bottom=0)
    ax.set_ylabel("MVP Share")
    # fits labels
    plt.xticks(years)
    if len(years)>10:
        xticks = plt.gca().xaxis.get_major_ticks()
        for i in range(len(xticks)):
            if i % 2 != 0:
                xticks[i].set_visible(False)
    ax.legend(frameon=False)

    def annot_text(ix):
        # add suffix to rank
        if int(actual_rank[ix][:-2]) < 11:
            text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Share: {round(actual_share[ix], 3)} ({actual_rank[ix]} in the league)\n" \
                   f"Year: {years[ix]}"
        else:
            text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Share: {round(actual_share[ix], 3)}\nYear: {years[ix]}"

        return text

    plt.tight_layout()
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, pred, ax, fig))
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, actual, ax, fig))

    plt.show()


def plot_all_salary():
    """
    :return: Plots a scatter graph of actual vs predicted % of cap (Some Ewing and jordan data not pictured)
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # drop rows where salary data isnt present
    blank_rows = df[df['salary'].isnull()].index.tolist()
    df.drop(blank_rows, inplace=True)
    blank_rows = df[df['year'].isnull()].index.tolist()
    df.drop(blank_rows, inplace=True)

    # get relevant data
    actual_salary = df['pred_%ofcap'].tolist()
    pred_salary = df['actual_%ofcap'].tolist()
    actual_salary_money = df.apply(lambda row: '${:,}'.format(int(row['salary'])), axis=1).tolist()
    pred_salary_money = df.apply(lambda row: '${:,}'.format(int(row['pred_salary'])), axis=1).tolist()
    year = df.apply(lambda row: int(row.name[-4:]), axis=1).tolist()
    name = df.apply(lambda row: row.name[:-5], axis=1).tolist()
    age = df.apply(lambda row: int(row['Age']), axis=1).tolist()
    team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()
    actual_rank = ordinal(df.apply(lambda row: row['actual_salary_rank'], axis=1).tolist())
    pred_rank = ordinal(df.apply(lambda row: row['pred_salary_rank'], axis=1).tolist())

    # turn into a %
    pred_salary = [x * 100 for x in pred_salary]

    # get PMCC test for linear association (and p value)
    pmcc, p = pearsonr(actual_salary, pred_salary)

    fig, ax = plt.subplots()

    # plot scatter graph of actual salary vs predicted
    scatter, = ax.plot(pred_salary, actual_salary, linestyle='', marker='x', markersize=3)

    # plot an x=y dashed line
    x = np.linspace(0, 150, 2)
    ax.plot(x, x, 'p--')

    # display PMCC
    ax.text(0.17, 0.95, 'PMCC = ' + str(round(pmcc, 3)), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    # set axes limits
    ax.set_xlim(left=0, right=70)
    ax.set_ylim(bottom=0, top=70)
    ax.set_aspect('equal', adjustable='box')

    def annot_text(ix):
        # sets text for hover annotation
        text = f"Name: {name[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
               f"Predicted Salary: {pred_salary_money[ix]} ({pred_rank[ix]} in the league)\n" \
               f"Actual Salary: {actual_salary_money[ix]} ({actual_rank[ix]} in the league)\nYear: {year[ix]}"
        return text

    # display annotations on hover
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, scatter, ax, fig))

    # label chart
    ax.set_xlabel('Actual % of Cap')
    ax.set_ylabel('Predicted % of Cap')

    plt.show()


def plot_2020():
    """
    :return: Plots predicted 2020 MVPs
    """
    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results_2020.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # drop rows where mvp data isnt present
    blank_rows = df[df['pred_mvp_share'].isnull()].index.tolist()
    df.drop(blank_rows, inplace=True)

    # initiliase chart
    fig, ax1 = plt.subplots()

    def get_data(df):
        pred_share = df['pred_mvp_share'].tolist()[:10]
        pred_rank = ordinal(df['pred_mvp_rank'].tolist()[:10])
        names = df.apply(lambda row: row.name[:-5], axis=1).tolist()[:10]
        age = df.apply(lambda row: int(row['Age']), axis=1).tolist()[:10]
        team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()[:10]
        # get last name only
        names_last = [x.split(" ")[1] for x in names]

        return pred_share, pred_rank, names, age, team, names_last

    def get_hover(df, ax, fig, bar):
        pred_share, pred_rank, names, age, team, names_last = get_data(df)

        def annot_text(ix):
            text = f"Name: {names[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)"

            return text

        fig.canvas.mpl_connect("motion_notify_event", hover_annot_bar(annot_text, ax, fig, bar))

    # sort by predicted mvp share
    df = df.sort_values(by='pred_mvp_share', ascending=False)
    pred_share, pred_rank, names, age, team, names_last = get_data(df)

    bar_pred = ax1.bar(names, pred_share, color='#ffa500')
    ax1.set_xticklabels(names_last, rotation=30)
    plt.tight_layout()

    # get hover info
    get_hover(df, ax1, fig, bar_pred)

    # set axis lims
    ax1.set_ylim(0, 1)

    # create labels
    ax1.set_ylabel('Predicted MVP Share')

    plt.show()


def plot_all_mvp():
    """
    :return: Plots a scatter graph of actual vs predicted mvp share
    """

    # load data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    df = pd.read_csv(data_path + "/predictions/results.csv.gz", encoding='utf-8-sig', compression='gzip', index_col=0)

    # drop rows where mvp data isnt present
    blank_rows = df[df['pred_mvp_share'].isnull()].index.tolist()
    df.drop(blank_rows, inplace=True)
    blank_rows = df[df['year'].isnull()].index.tolist()
    df.drop(blank_rows, inplace=True)

    # get list of actual and predicted mvp, name and year
    actual_share = df['actual_mvp_share'].tolist()
    pred_share = df['pred_mvp_share'].tolist()
    year = df.apply(lambda row: int(row.name[-4:]), axis=1).tolist()
    name = df.apply(lambda row: row.name[:-5], axis=1).tolist()
    age = df.apply(lambda row: int(row['Age']), axis=1).tolist()
    team = df.apply(lambda row: row['team_year'][:-5], axis=1).tolist()
    actual_rank = ordinal(df.apply(lambda row: row['actual_mvp_rank'], axis=1).tolist())
    pred_rank = ordinal(df.apply(lambda row: row['pred_mvp_rank'], axis=1).tolist())

    # get PMCC test for linear association (and p value)
    pmcc, p = pearsonr(actual_share, pred_share)

    fig, ax = plt.subplots()

    # plot scatter graph of actual mvp vs predicted
    scatter, = ax.plot(actual_share, pred_share, linestyle='', marker='x', markersize=4)

    # plot an x=y dashed line
    x = np.linspace(0, 1, 2)
    ax.plot(x, x, 'p--')

    # display PMCC
    ax.text(0.17, 0.95, 'PMCC = ' + str(round(pmcc, 3)), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)

    # set axes limits
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(bottom=0, top=1)
    ax.set_aspect('equal', adjustable='box')

    def annot_text(ix):
        if int(actual_rank[ix][:-2]) < 11:
            # add suffix to rank
            text = f"Name: {name[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Share: {round(actual_share[ix], 3)} ({actual_rank[ix]} in the league)\nYear: {year[ix]}"
        else:
            # add suffix to rank
            text = f"Name: {name[ix]}\nTeam: {team[ix]}\nAge: {age[ix]}\n" \
                   f"Predicted Share: {round(pred_share[ix], 3)} ({pred_rank[ix]} in the league)\n" \
                   f"Actual Share: {round(actual_share[ix], 3)} \nYear: {year[ix]}"
        return text

    # display annotations on hover
    fig.canvas.mpl_connect("motion_notify_event", hover_annot_plot(annot_text, scatter, ax, fig))

    # label chart
    ax.set_xlabel('Actual MVP Share')
    ax.set_ylabel('Predicted MVP Share')

    plt.show()
