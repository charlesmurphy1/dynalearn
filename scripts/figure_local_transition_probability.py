import argparse as ap
import h5py
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import sys

color_dark = {
    "blue": "#1f77b4",
    "orange": "#f19143",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252",
}

color_pale = {
    "blue": "#7bafd3",
    "orange": "#f7be90",
    "purple": "#c3b4d6",
    "red": "#e78580",
    "grey": "#999999",
}

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def get_plot_label(i):
    alphabet = "abcdefghijklmnpoqrstuvwxyz"
    return "(" + alphabet[i] + ")"


def setup_counts(state_label, N, raw_counts):
    inv_state_label = {state_label[i]: i for i in state_label}
    counts = {s: np.zeros(N) for s in state_label}
    i_index = 1
    for rc in raw_counts:
        s = inv_state_label[int(rc[0])]
        counts[s][int(rc[i_index + 1])] += rc[-1]

    return counts


def draw_ltp(path, save):
    with open(path, "r") as f:
        params = json.load(f)

    filename = os.path.join(params["path"], params["name"] + "_data.h5")
    an_filename = os.path.join(params["path"], params["name"] + "_analytics.h5")
    data = h5py.File(filename, "r")
    an_data = h5py.File(an_filename, "r")
    graph_label = params["graph"]["name"] + "_0"

    states = []
    state_label = {}
    for k in data["dynamics/params"]:
        if k[:5] == "state":
            states.append(k[6:])
            state_label[k[6:]] = int(data["dynamics/params/" + k][...])

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    N = params["graph"]["params"]["N"]
    g = list(data["data/"].keys())[0]
    kmin = np.min(np.sum(data["data/" + g + "/adj_matrix"], 0))
    kmax = np.max(np.sum(data["data/" + g + "/adj_matrix"], 0))
    degree_class = np.arange(N).astype("int")
    # degrees = np.sum(data["data/" + g + "/adj_matrix"], 0)

    fig_model = plt.figure(figsize=(6, 5))
    gs = GridSpec(8, 6)
    gs.update(wspace=0.1, hspace=0.05)
    ax_ltp_model = fig_model.add_subplot(gs[2:, :6])
    ax_dist_model = fig_model.add_subplot(gs[:2, :6])
    ax_dist_model.set_yscale("log")
    ax_dist_model.spines["right"].set_visible(False)
    ax_dist_model.spines["top"].set_visible(False)
    ax_dist_model.spines["bottom"].set_visible(False)
    ax_dist_model.set_xticks([])
    # ax_dist_model.axis("off")

    fig_estimate = plt.figure(figsize=(6, 5))
    gs = GridSpec(8, 6)
    gs.update(wspace=0.1, hspace=0.05)
    ax_ltp_estimate = fig_estimate.add_subplot(gs[2:, :6])
    ax_dist_estimate = fig_estimate.add_subplot(gs[:2, :6])
    ax_dist_estimate.set_yscale("log")
    ax_dist_estimate.spines["right"].set_visible(False)
    ax_dist_estimate.spines["top"].set_visible(False)
    ax_dist_estimate.spines["bottom"].set_visible(False)
    ax_dist_estimate.set_xticks([])
    # ax_dist_estimate.axis("off")

    b_wd = 5
    raw_counts = an_data["/analytics/counts"][...]
    counts = setup_counts(state_label, N, raw_counts)

    for i, in_s in enumerate(states):
        if in_s == "S":
            p_color = color_pale["blue"]
            d_color = color_dark["blue"]

        elif in_s == "I":
            p_color = color_pale["orange"]
            d_color = color_dark["orange"]

        elif in_s == "R":
            p_color = color_pale["purple"]
            d_color = color_dark["purple"]
        ax_dist_model.bar(
            degree_class + (i - 0.5) / b_wd,
            counts[in_s] / np.sum(counts[in_s]),
            1 / b_wd,
            color=d_color,
        )
        ax_dist_estimate.bar(
            degree_class + (i - 0.5) / b_wd,
            counts[in_s] / np.sum(counts[in_s]),
            1 / b_wd,
            color=d_color,
        )
        for out_s in states:
            if out_s == "S":
                marker = "o"
                linestyle = "-"
            elif out_s == "I":
                marker = "s"
                linestyle = "--"
            elif out_s == "R":
                marker = "v"
                linestyle = ":"
            fill_color = color_dark["grey"]

            ground_truth_ltp = an_data[
                "analytics/local_trans_prob/ground_truth/" + in_s + "_to_" + out_s
            ][...]
            model_ltp = an_data[
                "analytics/local_trans_prob/model/" + in_s + "_to_" + out_s
            ][...]
            estimate_ltp = an_data[
                "analytics/local_trans_prob/estimate/" + in_s + "_to_" + out_s
            ][...]

            # m_index = np.where(model_ltp[:, 0] != 0)
            # e_index = np.where(estimate_ltp[:, 0] != 0)

            ax_ltp_model.fill_between(
                degree_class[: int(kmax) + 1],
                model_ltp[:, 0][: int(kmax) + 1] - model_ltp[:, 1][: int(kmax) + 1],
                model_ltp[:, 0][: int(kmax) + 1] + model_ltp[:, 1][: int(kmax) + 1],
                color=fill_color,
                alpha=0.3,
            )
            ax_ltp_estimate.fill_between(
                degree_class[: int(kmax) + 1],
                estimate_ltp[:, 0][: int(kmax) + 1]
                - estimate_ltp[:, 1][: int(kmax) + 1],
                estimate_ltp[:, 0][: int(kmax) + 1]
                + estimate_ltp[:, 1][: int(kmax) + 1],
                color=fill_color,
                alpha=0.3,
            )
            ax_ltp_model.plot(
                degree_class[: int(kmax) + 1],
                ground_truth_ltp[: int(kmax) + 1],
                marker="None",
                linestyle=linestyle,
                linewidth=3,
                color=p_color,
            )
            ax_ltp_estimate.plot(
                degree_class[: int(kmax) + 1],
                ground_truth_ltp[: int(kmax) + 1],
                marker="None",
                linestyle=linestyle,
                linewidth=3,
                color=p_color,
            )
            ax_ltp_model.plot(
                degree_class[: int(kmax) + 1],
                model_ltp[:, 0][: int(kmax) + 1],
                marker=marker,
                linestyle="None",
                alpha=1,
                color=d_color,
                markeredgewidth=1,
                markeredgecolor="k",
            )
            ax_ltp_estimate.plot(
                degree_class[: int(kmax) + 1],
                estimate_ltp[:, 0][: int(kmax) + 1],
                marker=marker,
                linestyle="None",
                alpha=1,
                color=d_color,
                markeredgewidth=1,
                markeredgecolor="k",
            )

    ax_ltp_model.set_xlabel(r"Infected degree $\ell$", fontsize=14)
    ax_ltp_model.set_ylabel(r"$Pr[s\to s'|\,\ell]$", fontsize=14)
    ax_dist_model.set_ylabel(r"$Pr[\ell|\,s]$", fontsize=14)
    ax_dist_model.set_title(r"Learned model", fontsize=16)
    ax_ltp_model.set_xlim([-1, kmax + 1])
    ax_dist_model.set_xlim([-1, kmax + 1])
    ax_ltp_model.set_ylim([-0.1, 1.1])

    ax_ltp_estimate.set_xlabel(r"Infected degree $\ell$", fontsize=14)
    ax_ltp_estimate.set_ylabel(r"$Pr[s\to s'|\,\ell]$", fontsize=14)
    ax_dist_estimate.set_ylabel(r"$Pr[\ell|\,s]$", fontsize=14)
    ax_dist_estimate.set_title(r"Naive estimator", fontsize=16)
    ax_ltp_estimate.set_xlim([-1, kmax + 1])
    ax_dist_estimate.set_xlim([-1, kmax + 1])
    ax_ltp_estimate.set_ylim([-0.1, 1.1])

    for in_s in states:
        if in_s == "S":
            p_color = color_pale["blue"]
            d_color = color_dark["blue"]

        elif in_s == "I":
            p_color = color_pale["orange"]
            d_color = color_dark["orange"]

        elif in_s == "R":
            p_color = color_pale["purple"]
            d_color = color_dark["purple"]
        ax_ltp_model.plot(
            [-1],
            [-1],
            linestyle="None",
            marker="s",
            markersize=10,
            color=d_color,
            label=r"$s = {0}$".format(in_s),
        )
        ax_ltp_estimate.plot(
            [-1],
            [-1],
            linestyle="None",
            marker="s",
            markersize=10,
            color=d_color,
            label=r"$s = {0}$".format(in_s),
        )
    for out_s in states:
        if out_s == "S":
            marker = "o"
            linestyle = "-"
        elif out_s == "I":
            marker = "s"
            linestyle = "--"
        elif out_s == "R":
            marker = "v"
            linestyle = ":"
        ax_ltp_model.plot(
            [-1],
            [-1],
            linestyle=linestyle,
            marker=marker,
            markeredgewidth=1,
            markeredgecolor="k",
            markerfacecolor=color_dark["grey"],
            linewidth=3,
            color=color_pale["grey"],
            label=r"$s' = {0}$".format(out_s),
        )
        ax_ltp_estimate.plot(
            [-1],
            [-1],
            linestyle=linestyle,
            marker=marker,
            markeredgewidth=1,
            markeredgecolor="k",
            markerfacecolor=color_dark["grey"],
            linewidth=3,
            color=color_pale["grey"],
            label=r"$s' = {0}$".format(out_s),
        )

    ax_ltp_model.legend(loc="best", fancybox=True, fontsize=12, framealpha=1, ncol=2)
    ax_ltp_estimate.legend(loc="best", fancybox=True, fontsize=12, framealpha=1, ncol=2)

    plt.tight_layout(0.1)
    fig_model.savefig(os.path.join(params["path"], params["name"] + "_model_" + save))
    fig_estimate.savefig(
        os.path.join(params["path"], params["name"] + "_estimate_" + save)
    )


def main():
    prs = ap.ArgumentParser(
        description="Get local transition probability \
                                         figure from path to parameters."
    )
    prs.add_argument(
        "--path", "-p", type=str, required=True, help="Path to parameters."
    )
    prs.add_argument(
        "--save", "-s", type=str, required=False, help="Path where to save."
    )
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()
    draw_ltp(args.path, args.save)


if __name__ == "__main__":
    main()
