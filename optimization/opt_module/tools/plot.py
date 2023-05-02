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


import numpy
import seaborn as sns
import pandas as pd

from . import plottools as pt


def diversity(checkpoint, evaluator, color="b", figs=None, reportname=""):
    """plot the whole history, the hall of fame, and the best individual
    from a unumpyickled checkpoint
    """

    if figs is None:
        figs = {}
    param_names = evaluator.param_names
    n_params = len(param_names)
    print(f"n_params: {n_params}")

    hof = checkpoint["halloffame"]
    fitness_cut_off = 2.0 * sum(hof[0].fitness.values)
    print(f"fitness_cut_off: {fitness_cut_off}")

    figname = "Diversity " + reportname
    fig = pt.make_figure(
        figname=figname, orientation="page", figs=figs, fontsizes=(10, 10)
    )

    axs = pt.tiled_axs(
        frames=n_params,
        columns=6,
        d_out=5,
        fig=fig,
        top_margin=0.08,
        bottom_margin=0.03,
        left_margin=0.08,
        right_margin=0.03,
        hspace=0.3,
        wspace=0.8,
    )

    all_params = get_params(
        checkpoint["history"].genealogy_history.values(),
        fitness_cut_off=fitness_cut_off,
    )
    hof_params = get_params(checkpoint["halloffame"], fitness_cut_off=fitness_cut_off)

    best_params = checkpoint["halloffame"][0]

    for i, name in enumerate(param_names):
        ax = axs[i]

        p1 = numpy.array(all_params)[:, i].tolist()
        p2 = numpy.array(hof_params)[:, i].tolist()

        df = pd.DataFrame()
        df["val"] = p1 + p2
        df["type"] = ["all"] * len(p1) + ["hof"] * len(p2)
        df["param"] = [name] * (len(p1) + len(p2))

        sns.violinplot(
            x="param",
            y="val",
            hue="type",
            data=df,
            ax=ax,
            split=True,
            inner="quartile",
            palette={"all": "gray", "hof": color},
        )

        ax.axhline(y=best_params[i], color=color, label="best", linewidth=2)
        ax.set_ylabel("")

        if i > 0:
            ax.legend_.remove()
        else:
            ax.legend()

    for i, parameter in enumerate(evaluator.params):
        min_value = parameter.lower_bound
        max_value = parameter.upper_bound

        ax = axs[i]
        ax.set_ylim((min_value - 0.02 * min_value, max_value + 0.02 * max_value))

        name = param_names[i]
        label = name.replace(".", "\n")

        ax.set_title(label, fontsize=7)

        pt.adjust_spines(ax, ["left"], d_out=5)

        # ax.set_xticks([])
        # ax.axes.get_xaxis().set_visible(False)

    fig.suptitle(reportname, fontsize=10)

    return fig


def get_params(params, fitness_cut_off=1e9):
    """plot the individual parameter values"""
    results = []
    for param in params:
        if fitness_cut_off > sum(param.fitness.values):
            results.append(param)

    return results
