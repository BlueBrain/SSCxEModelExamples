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

This work is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/legalcode or send a letter to Creative Commons, 171
Second Street, Suite 300, San Francisco, California, 94105, USA.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path

import numpy
import scipy
import scipy.optimize
import scipy.stats

from lib import plottools as pt


def read_data(data_path):

    data = []
    distances = []
    csvreader = csv.reader(open(data_path, "r"))
    for distance, dat in csvreader:
        distances.append(float(distance))
        data.append(float(dat))

    return distances, data


def fit(distances, values, max_dist=1000.0, plot_prop="_", ymult=1):

    x = numpy.arange(0, max_dist, 1)

    guess = [50]
    if "_decay_" in plot_prop:
        exp_decay = lambda x, p: numpy.exp(-x / p) * ymult
    else:
        exp_decay = lambda x, p: 1.0 / (numpy.exp(-x / p) * ymult)
    params, cov = scipy.optimize.minpack.curve_fit(
        exp_decay, distances, values, p0=guess
    )

    perr = numpy.sqrt(numpy.diag(cov))
    p = params[0]
    p_upper = params[0] + 2 * perr[0]  # 95%
    p_lower = params[0] - 2 * perr[0]  # 95%
    p_std = perr[0]

    if "_decay_" in plot_prop:
        best_fit = lambda x: numpy.exp(-x / p) * ymult
        best_fit_upper = lambda x: numpy.exp(-x / p_upper) * ymult
        best_fit_lower = lambda x: numpy.exp(-x / p_lower) * ymult
    else:
        best_fit = lambda x: 1.0 / numpy.exp(-x / p)
        best_fit_upper = lambda x: 1.0 / (numpy.exp(-x / p_upper) * ymult)
        best_fit_lower = lambda x: 1.0 / (numpy.exp(-x / p_lower) * ymult)

    y_fit = best_fit(x)
    y_ci_upper = best_fit_upper(x)
    y_ci_lower = best_fit_lower(x)

    return x, y_fit, y_ci_lower, y_ci_upper, p, p_std


def plot_results(
    fig,
    figname="Attenuation",
    results_paths=None,
    plotwhat=None,
    fitwhat=None,
    use_diams=True,
    legend_fontsize=6,
):

    if plotwhat is None:
        plotwhat = ["EPSP", "bAP"]
    if fitwhat is None:
        fitwhat = ["EPSP", "bAP", "EPSP_data", "bAP_data"]
    all_points = {
        "basal": {
            "distance": [],
            "path_distance": [],
            "dend_distance": [],
            "mid_diam": [],
            "diam": [],
            "bAP_amp": [],
            "EPSP_att": [],
        },
        "apical": {
            "distance": [],
            "path_distance": [],
            "dend_distance": [],
            "mid_diam": [],
            "diam": [],
            "bAP_amp": [],
            "EPSP_att": [],
        },
        "somatic": {
            "distance": [],
            "path_distance": [],
            "dend_distance": [],
            "mid_diam": [],
            "diam": [],
            "bAP_amp": [],
            "EPSP_att": [],
        },
    }

    if results_paths is not None:

        for results_path in results_paths:

            results = json.load(open(results_path))

            distance_type = "path_distance"
            diam_type = "diam"

            for _, result in results.items():
                point = result["points"]
                if point is not None:
                    for sectype, all_sec in all_points.items():
                        sec = point[sectype]
                        all_sec["distance"] += sec["distance"]
                        all_sec["dend_distance"] += sec["dend_distance"]
                        all_sec["path_distance"] += sec["path_distance"]
                        all_sec["mid_diam"] += sec["mid_diam"]
                        all_sec["diam"] += sec["diam"]
                        all_sec["bAP_amp"] += sec["bAP_amp"]
                        all_sec["EPSP_att"] += list(
                            numpy.array(sec["EPSP_amp_dend"])
                            / numpy.array(sec["EPSP_amp_soma"])
                        )

    axs = pt.grid_axs(
        d_out=5,
        rows=len(plotwhat),
        columns=1,
        fig=fig,
        top_margin=0.05,
        bottom_margin=0.10,
        left_margin=0.10,
        right_margin=0.12,
        hspace=0.3,
        wspace=0.3,
    )

    iax = 0

    # Colors

    rvb_apical = plt.cm.get_cmap("Blues", 4)
    rvb_apical = mpl.colors.ListedColormap(rvb_apical(numpy.arange(4))[1:])

    rvb_basal = plt.cm.get_cmap("Greens", 4)
    rvb_basal = mpl.colors.ListedColormap(rvb_basal(numpy.arange(4))[1:])

    min_diam = 0.5
    vmin_a = 0.6
    vmax_a = 5.4
    vmin_b = 0.3
    vmax_b = 2.1

    # bAP
    if "bAP" in plotwhat:

        ax = axs[iax]
        iax += 1
        max_dist = 900

        if results_paths is not None:
            # somatic AP
            AP_soma = numpy.mean(all_points["somatic"]["bAP_amp"])
            AP_soma_std = numpy.std(all_points["somatic"]["bAP_amp"])
            ax.errorbar(
                0, AP_soma, yerr=AP_soma_std, color="black", marker="o", zorder=1
            )
            # apical
            diam = numpy.array(all_points["apical"]["diam"])
            distances = numpy.array(all_points["apical"][distance_type])
            conditions = (diam >= min_diam) & (distances <= max_dist)
            distances = distances[conditions]
            diam = numpy.array(all_points["apical"][diam_type])[conditions]
            bAP_amp = numpy.array(all_points["apical"]["bAP_amp"])[conditions]

            sa0 = ax.scatter(
                distances[numpy.argsort(diam)],
                bAP_amp[numpy.argsort(diam)],
                c=diam[numpy.argsort(diam)],
                cmap=rvb_apical,
                vmin=vmin_a,
                vmax=vmax_a,
                marker="o",
                s=20,
                zorder=1,
                edgecolor="none",
            )

            sa0.set_rasterized(True)

            if "bAP" in fitwhat:
                x, y_fit, _, _, p, p_std = fit(
                    distances,
                    bAP_amp,
                    ymult=AP_soma,
                    max_dist=max_dist,
                    plot_prop="_decay_",
                )
                ax.plot(
                    x,
                    y_fit,
                    "--",
                    color="black",
                    linewidth=2,
                    label=r"apical, fit model, $\lambda=%.1f\pm%.1f\,\mu m$"
                    % (p, p_std),
                    zorder=2,
                )

        # data
        data_path = "experiments/StuartSakmann1994_Fig1_2.csv"
        distances, data = read_data(data_path)
        ax.scatter(
            distances,
            data,
            color="red",
            marker="o",
            zorder=3,
            edgecolor="none",
            s=30,
            label="data (Stuart and Sakmann 1994)",
        )

        if "bAP_data" in fitwhat:
            x, y_fit, _, _, p, p_std = fit(
                distances, data, ymult=data[0], max_dist=max_dist, plot_prop="_decay_"
            )
            ax.plot(
                x,
                y_fit,
                "-",
                color="red",
                linewidth=2,
                label=r"apical, fit data, $\lambda=%.1f\pm%.1f\,\mu m$" % (p, p_std),
                zorder=4,
            )

        data_path = "experiments/Larkum2001_Fig8_2.csv"
        distances, data = read_data(data_path)
        ax.scatter(
            distances,
            data,
            color="m",
            marker="o",
            zorder=3,
            edgecolor="none",
            s=30,
            label="data (Larkum et al. 2001)",
        )

        if "bAP_data" in fitwhat:
            x, y_fit, _, _, p, p_std = fit(
                distances, data, ymult=data[0], max_dist=max_dist, plot_prop="_decay_"
            )
            ax.plot(
                x,
                y_fit,
                "-",
                color="m",
                linewidth=2,
                label=r"apical, fit data, $\lambda=%.1f\pm%.1f\,\mu m$" % (p, p_std),
                zorder=4,
            )

        max_dist = 150

        if results_paths is not None:
            # basal
            diam = numpy.array(all_points["basal"]["diam"])
            distances = numpy.array(all_points["basal"][distance_type])
            conditions = (diam >= min_diam) & (distances <= max_dist)
            distances = distances[conditions]
            diam = numpy.array(all_points["basal"][diam_type])[conditions]
            bAP_amp = numpy.array(all_points["basal"]["bAP_amp"])[conditions]

            sb0 = ax.scatter(
                distances[numpy.argsort(diam)],
                bAP_amp[numpy.argsort(diam)],
                c=diam[numpy.argsort(diam)],
                cmap=rvb_basal,
                vmin=vmin_b,
                vmax=vmax_b,
                marker="^",
                s=20,
                zorder=1,
                edgecolor="none",
            )

            sb0.set_rasterized(True)

            if "bAP" in fitwhat:
                x, y_fit, _, _, p, p_std = fit(
                    distances,
                    bAP_amp,
                    ymult=AP_soma,
                    max_dist=max_dist,
                    plot_prop="_decay_",
                )
                ax.plot(
                    x,
                    y_fit,
                    "--",
                    color="black",
                    linewidth=2,
                    label=r"basal, fit model, $\lambda=%.1f\pm%.1f\,\mu m$"
                    % (p, p_std),
                    zorder=2,
                )

        # data
        data_path = "experiments/Nevian2007_Fig1_2.csv"
        distances, data = read_data(data_path)
        ax.scatter(
            distances,
            data,
            color="red",
            marker="^",
            zorder=3,
            edgecolor="none",
            s=30,
            label="data (Nevian et al. 2007)",
        )

        if "bAP_data" in fitwhat:
            x, y_fit, _, _, p, p_std = fit(
                distances, data, ymult=data[0], max_dist=max_dist, plot_prop="_decay_"
            )
            ax.plot(
                x,
                y_fit,
                "-",
                color="red",
                linewidth=2,
                label=r"basal, fit data, $\lambda=%.1f\pm%.1f\,\mu m$" % (p, p_std),
                zorder=4,
            )

        # Titles
        ax.set_ylim(0, 120)
        ax.set_xlim(-10, 950)
        ax.set_ylabel(r"Amplitude (mV)")
        ax.set_xlabel(r"Distance from soma ($\mu$m)")
        lg = ax.legend(loc=(0.62, 0.45), fontsize=legend_fontsize, labelspacing=0.2)
        lg.get_frame().set_lw(0.1)

        if use_diams and (results_paths is not None):
            # Diameter labels
            ticks_a = [0.6, 1.4, 2.2, 3.0, 3.8, 4.6, 5.4]
            ticks_b = [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
            cbaxes = fig.add_axes([0.9, 0.55, 0.02, 0.4])
            cbar = plt.colorbar(sa0, cax=cbaxes, ticks=ticks_a)
            cbar.set_label(r"Apical, diameter at midpoint ($\mu$m)")
            cbaxes = fig.add_axes([0.9, 0.05, 0.02, 0.4])
            cbar = plt.colorbar(sb0, cax=cbaxes, ticks=ticks_b)
            cbar.set_label(r"Basal, diameter at midpoint ($\mu$m)")

    # EPSP
    if "EPSP" in plotwhat:

        ax = axs[iax]
        iax += 1
        max_dist = 550.0

        if results_paths is not None:

            # apical
            diam = numpy.array(all_points["apical"]["diam"])
            distances = numpy.array(all_points["apical"][distance_type])
            conditions = (diam >= min_diam) & (distances <= max_dist)
            distances = distances[conditions]
            diam = numpy.array(all_points["apical"][diam_type])[conditions]
            EPSP_att = numpy.array(all_points["apical"]["EPSP_att"])[conditions]

            sa1 = ax.scatter(
                distances[numpy.argsort(diam)],
                EPSP_att[numpy.argsort(diam)],
                c=diam[numpy.argsort(diam)],
                cmap=rvb_apical,
                vmin=vmin_a,
                vmax=vmax_a,
                marker="o",
                s=20,
                zorder=1,
                edgecolor="none",
            )

            sa1.set_rasterized(True)

            if "EPSP" in fitwhat:
                x, y_fit, _, _, p, p_std = fit(distances, EPSP_att, max_dist=max_dist)
                ax.plot(
                    x,
                    y_fit,
                    "--",
                    color="black",
                    linewidth=2,
                    label=r"apical, fit model, $\lambda=%.1f\pm%.1f\,\mu m$"
                    % (p, p_std),
                    zorder=2,
                )

        # data
        data_path = "experiments/Berger2001_Fig3_2.csv"
        distances, data = read_data(data_path)
        data = numpy.exp(data)  # convert back from ln() and inverse
        ax.scatter(
            distances,
            data,
            color="red",
            marker="o",
            zorder=3,
            edgecolor="none",
            s=30,
            label="apical, data (Berger et al. 2001)",
        )

        if "EPSP_data" in fitwhat:
            x, y_fit, _, _, p, p_std = fit(distances, data, max_dist=max_dist)
            ax.plot(
                x,
                y_fit,
                "-",
                color="red",
                linewidth=2,
                label=r"apical, fit data, $\lambda=%.1f\pm%.1f\,\mu m$" % (p, p_std),
                zorder=4,
            )

        max_dist = 150
        if results_paths is not None:
            # basal
            diam = numpy.array(all_points["basal"]["diam"])
            distances = numpy.array(all_points["basal"][distance_type])
            conditions = (diam >= min_diam) & (distances <= max_dist)
            distances = distances[conditions]
            diam = numpy.array(all_points["basal"][diam_type])[conditions]
            EPSP_att = numpy.array(all_points["basal"]["EPSP_att"])[conditions]

            sb1 = ax.scatter(
                distances[numpy.argsort(diam)],
                EPSP_att[numpy.argsort(diam)],
                c=diam[numpy.argsort(diam)],
                cmap=rvb_basal,
                vmin=vmin_b,
                vmax=vmax_b,
                marker="^",
                s=20,
                zorder=1,
                edgecolor="none",
            )

            sb1.set_rasterized(True)

            if "EPSP" in fitwhat:
                x, y_fit, y_ci_lower, y_ci_upper, p, p_std = fit(
                    distances, EPSP_att, max_dist=max_dist
                )
                ax.plot(
                    x,
                    y_fit,
                    "--",
                    color="black",
                    linewidth=2,
                    label=r"basal, fit model, $\lambda=%.1f\pm%.1f\,\mu m$"
                    % (p, p_std),
                    zorder=2,
                )

        # data
        data_path = "experiments/Nevian2007_Fig2_email.csv"
        distances, data = read_data(data_path)
        data = numpy.array(data)
        ax.scatter(
            distances,
            data,
            color="red",
            marker="^",
            zorder=3,
            edgecolor="none",
            s=30,
            label="basal, data (Nevian et al. 2007)",
        )

        if "EPSP_data" in fitwhat:
            x, y_fit, _, _, p, p_std = fit(distances, data, max_dist=max_dist)
            ax.plot(
                x,
                y_fit,
                "-",
                color="red",
                linewidth=2,
                label=r"basal, fit data, $\lambda=%.1f\pm%.1f\,\mu m$" % (p, p_std),
                zorder=4,
            )

        # Titles
        ax.set_ylim(0, 50)
        ax.set_xlim(-10, 600)
        ax.set_ylabel(r"Attenuation dendrite / soma")
        ax.set_xlabel(r"Distance from soma ($\mu$m)")
        lg = ax.legend(loc=(0.62, 0.55), fontsize=legend_fontsize, labelspacing=0.2)
        lg.get_frame().set_lw(0.1)

    figpath = os.path.join("output", f"{figname}.pdf")
    fig.savefig(figpath, dpi=300)


if __name__ == "__main__":

    figs = {}
    fontsizes = (6, 7)
    figsize_mm = (150, 120)
    #
    # # only data
    # figname='ephys'
    # fig = pt.make_figure(figname=figname, figsize_mm=figsize_mm, figs=figs, fontsizes = (6,7))
    # plot_results(fig, figname=figname, results_paths=None, plotwhat=['bAP'], fitwhat=[])
    #
    # # with other morphologies
    # figname='allmorph'
    # results_paths = ['./run/383d58b/output/results.json']
    # fig = pt.make_figure(figname=figname, figsize_mm=figsize_mm, figs=figs, fontsizes = (6,7))
    # plot_results(fig, figname=figname, results_paths=results_paths, plotwhat=['bAP'])
    #
    # # compare unrepaired / repaired
    # figname='exemplars'
    # results_paths = ['./run/cf8ae97/output/results.json', './run/b10b0ad/output/results.json']
    # fig = pt.make_figure(figname=figname, figsize_mm=figsize_mm, figs=figs, fontsizes = (6,7))
    # plot_results(fig, figname=figname, results_paths=results_paths, plotwhat=['bAP'], fitwhat=[])

    figname = "Attenuation_cADpyr_L5TPC_e3026da"
    results_paths = [Path("output") / "results.json"]
    fig = pt.make_figure(
        figname=figname, figsize_mm=figsize_mm, figs=figs, fontsizes=fontsizes
    )
    plot_results(
        fig, figname=figname, results_paths=results_paths, plotwhat=["EPSP", "bAP"]
    )

    plt.show()
