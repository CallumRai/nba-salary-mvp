from difflib import get_close_matches
import math


def year_input(salary):
    """
    :param salary: Bool, True if required for salary data, False if required for MVP data
    :return: User defined year
    """

    valid_response = False

    if salary:
        year = input("Enter a year between 1994 and 2019: ")
    else:
        year = input("Enter a year between 1984 and 2019: ")

    def isint(string):
        """
        :param string: A string
        :return: Bool, True is string is an integer, false if not
        """
        try:
            int(string)
            return True
        except ValueError:
            return False

    while not valid_response:
        # checks if year is an integer and in required year range
        if isint(year):
            # turn year into an integer
            year = int(year)
            if salary:
                if 1993 < year < 2020:
                    valid_response = True
            else:
                if 1983 < year < 2020:
                    valid_response = True
        # gets input again if criteria not fufilled
        if not valid_response:
            if salary:
                year = input("\nPlease enter a valid year between 1994 and 2019: ")
            else:
                year = input("\nPlease enter a valid year between 1984 and 2019: ")

    return year


def name_input(names):
    """
    :param names: List of names
    :return: User defined name
    """

    valid_response = False

    name = input("Search a player: ")

    while not valid_response:
        # finds closest match
        player = get_close_matches(name, names, n=1)

        # deals with situation no player found
        if len(player) == 0:
            name = input("\nPlayer not found (try entering both first and last name): ")
        else:
            valid_response = True

    return player[0]


# allows scatter plots to zoom and pan
class ZoomPan:
    def __init__(self):
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None

    def zoom_factory(self, ax, base_scale=2.):
        def zoom(event):
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata  # get event x location
            ydata = event.ydata  # get event y location

            if event.button == 'down':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'up':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print
                event.button

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest
        fig.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax):
        def onPress(event):
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if self.press is None: return
            if event.inaxes != ax: return
            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy
            ax.set_xlim(self.cur_xlim)
            ax.set_ylim(self.cur_ylim)

            ax.figure.canvas.draw()

        fig = ax.get_figure()  # get the figure of interest

        # attach the call back
        fig.canvas.mpl_connect('button_press_event', onPress)
        fig.canvas.mpl_connect('button_release_event', onRelease)
        fig.canvas.mpl_connect('motion_notify_event', onMotion)

        # return the function
        return onMotion


# handles hover annotations
def hover_annot_plot(text_func, point, ax, fig):
    # get form of how you want to annotate data
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="square", fc="w"),
                        arrowprops=dict(arrowstyle="-|>"))
    annot.set_visible(False)

    def update_annot(ind):
        # get index of data
        x, y = point.get_data()
        ix = ind['ind'][0]

        # set xy in annot
        annot.xy = (x[ix], y[ix])

        # set text
        text = text_func(ix)
        annot.set_text(text)

        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = point.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.subplots_adjust(top=0.8, left=0.1, right=0.7)

    return hover


def hover_annot_bar(text_func, ax, fig, bars):
    annot = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="black", ec="b", lw=2),
                        arrowprops=dict(arrowstyle="-|>"))
    annot.set_visible(False)

    def update_annot(bar):
        x = bar.get_x() + bar.get_width() / 2.
        y = bar.get_y() + bar.get_height()
        annot.xy = (x, y)
        ix = int("{:.2g}".format(x))
        text = text_func(ix)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for bar in bars:
                cont, ind = bar.contains(event)
                if cont:
                    update_annot(bar)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.subplots_adjust(top=0.85, left=0.1, right=0.8)

    return hover

def ordinal(list):
    """
    :return: List of ranks with ordinal suffix
    """

    suffix = lambda n: "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))
    list=[suffix(math.floor(x))for x in list]
    return list

