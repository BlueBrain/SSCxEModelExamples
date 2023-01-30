"""
Copyright (c) 2022 EPFL-BBP, All rights reserved.

THIS SOFTWARE IS PROVIDED BY THE BLUE BRAIN PROJECT ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE BLUE BRAIN PROJECT
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

This work is licensed under a Creative Commons Attribution 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""


import matplotlib
import matplotlib.pyplot as plt

# plt.ioff()
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties

import collections
import numpy


def adjust_spines(ax, spines, color="k", d_out=10, d_down=False):

    if d_down == False:
        d_down = d_out

    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    for loc, spine in ax.spines.items():
        if loc in spines:
            if loc == "bottom":
                spine.set_position(("outward", d_down))  # outward by 10 points
            else:
                spine.set_position(("outward", d_out))  # outward by 10 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_visible(False)  # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")

        if color != "k":

            ax.spines["left"].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis="y", colors=color)

    elif "right" not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if "right" in spines:
        ax.yaxis.set_ticks_position("right")

        if color != "k":

            ax.spines["right"].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis="y", colors=color)

    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
        # ax.axes.get_xaxis().set_visible(True)

    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)


def grid_spec(
    fig=None,
    box=None,
    rows=1,
    columns=2,
    top_margin=0.05,
    bottom_margin=0.05,
    left_margin=0.05,
    right_margin=0.05,
    hspace=0.2,
    wspace=0.2,
    width_ratios=None,
    height_ratios=None,
):

    if box is None:
        box = {"left": 0.0, "bottom": 0.0, "top": 1.0, "right": 1.0}

    if width_ratios is None:
        width_ratios = [1] * columns

    if height_ratios is None:
        height_ratios = [1] * rows

    left = box["left"] + left_margin
    right = box["right"] - right_margin
    top = box["top"] - top_margin
    bottom = box["bottom"] + bottom_margin

    gs = matplotlib.gridspec.GridSpec(
        rows, columns, height_ratios=height_ratios, width_ratios=width_ratios
    )

    gs.update(
        top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    )

    return gs


def set_panels(axs, panels):

    if panels:
        font = FontProperties().copy()
        font.set_family("sans-serif")
        font.set_weight("bold")

        for il, label in enumerate(panels["labels"]):
            if "x" in panels:
                x = panels["x"]
            else:
                x = -0.1

            if "y" in panels:
                y = panels["y"]
            else:
                y = 1.15

            axs[il].text(
                x,
                y,
                label,
                fontsize=12,
                transform=axs[il].transAxes,
                va="top",
                ha="right",
                fontproperties=font,
            )


def set_fig_panels(fig, panels=False):

    if panels:
        font = FontProperties().copy()
        font.set_family("sans-serif")
        font.set_weight("bold")

        for label, (x, y) in panels.items():
            plt.figtext(
                x,
                y,
                label,
                fontsize=12,
                figure=fig,
                va="top",
                ha="left",
                fontproperties=font,
            )


def set_global_panels(fig, xys, panels=False):

    if panels:
        font = FontProperties().copy()
        font.set_family("sans-serif")
        font.set_weight("bold")

        for il, label in enumerate(panels["labels"]):

            (x, y) = xys[il]

            if "x" in panels:
                x_off = panels["x"]
            else:
                x_off = 0

            if "y" in panels:
                y_off = panels["y"]
            else:
                y_off = 0

            if "va" in panels:
                va = panels["va"]
            else:
                va = "bottom"

            if "ha" in panels:
                ha = panels["ha"]
            else:
                ha = "right"

            plt.figtext(
                x + x_off,
                y + y_off,
                label,
                fontsize=12,
                figure=fig,
                va=va,
                ha=ha,
                fontproperties=font,
            )


def single_ax(d_out=0, panels=False, projection=False, **kwargs):

    fig = kwargs["fig"]
    gs = grid_spec(rows=1, columns=1, **kwargs)

    if projection:
        ax = fig.add_subplot(gs[0], projection=projection)
    else:
        ax = fig.add_subplot(gs[0])
        adjust_spines(ax, ["left", "bottom"], d_out=d_out)

        xy = (gs[0].get_position(fig).xmin, gs[0].get_position(fig).ymax)
        set_global_panels(fig, [xy], panels)

    return ax


def grid_axs(d_out=0, panels=False, projection=False, **kwargs):

    fig = kwargs["fig"]

    axs = []
    xys = []

    gs = grid_spec(**kwargs)

    for g in gs:
        if projection:
            axs.append(fig.add_subplot(g, projection=projection))
        else:
            axs.append(fig.add_subplot(g))
            adjust_spines(axs[-1], ["left", "bottom"], d_out=d_out)

            xys.append((g.get_position(fig).xmin, g.get_position(fig).ymax))
            set_global_panels(fig, xys, panels)

    return axs


def tiled_axs(frames=1, d_out=0, panels=False, projection=False, **kwargs):

    fig = kwargs["fig"]
    columns = kwargs["columns"]

    axs = []
    xys = []

    rows = int(numpy.ceil(frames / float(columns)))

    gs = grid_spec(rows=rows, **kwargs)

    for fi in range(frames):
        g = gs[int(fi / columns), int(fi % columns)]
        if projection:
            axs.append(fig.add_subplot(g, projection=projection))
        else:
            axs.append(fig.add_subplot(g))
            adjust_spines(axs[-1], ["left", "bottom"], d_out=d_out)

            xys.append((g.get_position(fig).xmin, g.get_position(fig).ymax))
            set_global_panels(fig, xys, panels)

    return axs


def make_figure(
    figname="",
    fontsizes=(6, 8),
    figsize_mm=None,
    orientation="landscape",
):

    if orientation == "landscape" and figsize_mm is None:
        figsize_mm = (297, 210)
    elif orientation == "page" and figsize_mm is None:
        figsize_mm = (210, 297)

    figsize = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)

    params = {
        "axes.labelsize": fontsizes[0],
        "axes.linewidth": 0.5,
        "font.size": fontsizes[1],
        "axes.titlesize": fontsizes[1],
        "legend.fontsize": fontsizes[1],
        "xtick.labelsize": fontsizes[0],
        "ytick.labelsize": fontsizes[0],
        "legend.borderpad": 0.2,
        "legend.loc": "best",
        "text.usetex": False,
        #'pdf.fonttype': 42,
        "figure.figsize": figsize,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
    matplotlib.rcParams.update(params)

    return plt.figure(figname, facecolor="white")
