"""Contains the plotting functions used by the optimization notebook."""

import collections
import matplotlib.pyplot as plt

import numpy as np


def plot_objectives(objectives):
    """Plot objectives of the cell model"""

    fig = plt.figure(figsize=(10, 10), facecolor="white")

    box = {"left": 0.0, "bottom": 0.0, "width": 1.0, "height": 1.0}

    objectives = collections.OrderedDict(sorted(objectives.items()))
    left_margin = box["width"] * 0.4
    right_margin = box["width"] * 0.05
    top_margin = box["height"] * 0.05
    bottom_margin = box["height"] * 0.1

    axes = fig.add_axes(
        (
            box["left"] + left_margin,
            box["bottom"] + bottom_margin,
            box["width"] - left_margin - right_margin,
            box["height"] - bottom_margin - top_margin,
        )
    )

    ytick_pos = [x + 0.5 for x in range(len(objectives.keys()))]

    axes.barh(
        ytick_pos, objectives.values(), height=0.5, align="center", color="#779ECB"
    )
    axes.set_yticks(ytick_pos)
    axes.set_yticklabels(objectives.keys(), size="medium")
    axes.set_ylim(-0.5, len(objectives.values()) + 0.5)
    axes.set_xlabel("Objective value (# std)")
    axes.set_ylabel("Objectives")

    print(f"Sum of objectives: {sum(objectives.values())} (# std)")
    return fig


def plot_log(log):
    """Plot logbook"""

    color = "b"
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")

    ax.yaxis.grid(which="both")

    gen_numbers = log.select("gen")
    mean = np.array(log.select("avg"))
    std = np.array(log.select("std"))
    minimum = np.array(log.select("min"))

    stdminus = mean - std
    stdplus = mean + std
    ax.plot(
        gen_numbers,
        mean,
        color=color,
        linewidth=2,
        alpha=0.4,
        label="population average",
    )

    ax.fill_between(
        gen_numbers,
        stdminus,
        stdplus,
        color=color,
        alpha=0.1,
        linewidth=2,
        label=r"population standard deviation",
    )

    ax.plot(gen_numbers, minimum, color=color, linewidth=2, label="population minimum")

    minmin = min(minimum)
    ax.set_yscale("log")
    ax.set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    ax.set_xlabel("Generation #")
    ax.set_ylabel("Sum of objectives")
    ax.set_ylim([minmin - 0.1 * minmin, max(stdplus)])

    ax.axhline(minmin, color="k", linewidth=0.5, label="minimum: %.2f" % minmin)

    ax.legend()

    return fig
