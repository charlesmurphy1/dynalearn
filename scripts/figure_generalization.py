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
EPSILON = 0


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


def kl_div(p, q):
    p[p <= 0] = 1e-15
    q[q <= 0] = 1e-15
    ans = p * np.log(p / q)
    ans = np.sum(ans, axis=-1)
    return ans


def compute_minmax_degree(graphs, N):
    kmin = N
    kmax = 0
    for g in graphs:
        degree = np.sum(graphs[g], axis=0)
        if np.min(degree) < kmin:
            kmin = np.min(degree)

        if np.max(degree) < kmax:
            kmax = np.max(degree)
    return kmin, kmax


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

    fig_model = plt.figure(figsize=(6, 5))
    gs = GridSpec(8, 6)
    gs.update(wspace=0.1, hspace=0.05)
    ax_ltp_model = fig_model.add_subplot(gs[2:, :6])
    ax_ltp_model.axvspan(kmax, N - 1, facecolor=color_pale["grey"], alpha=0.5)
    ax_dist_model = fig_model.add_subplot(gs[:2, :6])
    ax_dist_model.set_yscale("log")
    ax_dist_model.spines["right"].set_visible(False)
    ax_dist_model.spines["top"].set_visible(False)
    ax_dist_model.spines["bottom"].set_visible(False)
    ax_dist_model.set_xticks([])

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

            all_true_ltp = an_data[
                "analytics/generalization/ground_truth/" + in_s + "_to_" + out_s
            ][...]
            all_model_ltp = an_data[
                "analytics/generalization/model/" + in_s + "_to_" + out_s
            ][...]

            true_ltp = np.zeros(N)
            model_ltp = np.zeros(N)
            err_model_ltp = np.zeros(N)
            for l in degree_class:
                t_ltp = all_true_ltp[:, l]
                m_ltp = all_model_ltp[:, l]
                true_ltp[l] = np.mean(t_ltp)
                model_ltp[l] = np.mean(m_ltp)
                err_model_ltp[l] = np.std(m_ltp) / np.sqrt(len(m_ltp))

            ax_ltp_model.fill_between(
                degree_class,
                model_ltp - err_model_ltp,
                model_ltp + err_model_ltp,
                color=fill_color,
                alpha=0.3,
            )
            ax_ltp_model.plot(
                degree_class,
                true_ltp,
                marker="None",
                linestyle=linestyle,
                linewidth=3,
                color=p_color,
            )
            ax_ltp_model.plot(
                degree_class,
                model_ltp,
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
    ax_ltp_model.set_xlim([-1, N])
    ax_dist_model.set_xlim([-1, N])
    ax_ltp_model.set_ylim([-0.1, 1.1])

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
        ax_dist_model.plot(
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
        ax_dist_model.plot(
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

    ax_dist_model.legend(loc="best", fancybox=True, fontsize=12, framealpha=1, ncol=2)

    plt.tight_layout(0.1)
    fig_model.savefig(os.path.join(params["path"], params["name"] + "_ltp_" + save))


def draw_div(path, save):
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

    fig_model = plt.figure(figsize=(6, 5))
    gs = GridSpec(8, 6)
    gs.update(wspace=0.1, hspace=0.05)
    ax_ltp_model = fig_model.add_subplot(gs[2:, :6])
    ax_ltp_model.axvspan(kmax, N - 1, facecolor=color_pale["grey"], alpha=0.5)
    ax_dist_model = fig_model.add_subplot(gs[:2, :6])
    ax_dist_model.set_yscale("log")
    ax_dist_model.spines["right"].set_visible(False)
    ax_dist_model.spines["top"].set_visible(False)
    ax_dist_model.spines["bottom"].set_visible(False)
    ax_dist_model.set_xticks([])

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
        truth_gen = np.zeros((N, N, len(states)))
        model_gen = np.zeros((N, N, len(states)))
        base_line = np.zeros((N, N, len(states)))
        for j, out_s in enumerate(states):
            truth_gen[:, :, j] = an_data[
                "analytics/generalization/ground_truth/" + in_s + "_to_" + out_s
            ][...]
            model_gen[:, :, j] = an_data[
                "analytics/generalization/model/" + in_s + "_to_" + out_s
            ][...]

        base_line = np.ones((N, N, len(states))) / len(states)
        kl_div_model = kl_div(truth_gen, model_gen)
        kl_div_base = kl_div(truth_gen, base_line)
        div_model = np.zeros(N)
        div_base = np.zeros(N)
        for k in degree_class:
            d_m = kl_div_model[k, :k]
            d_b = kl_div_base[k, :k]
            if np.any(d_m > EPSILON):
                div_model[k] = np.mean(d_m)
            if np.any(d_b > EPSILON):
                div_base[k] = np.mean(d_b)

        ax_ltp_model.plot(
            degree_class,
            div_model,
            marker="None",
            linestyle="-",
            linewidth=3,
            color=d_color,
        )
        ax_ltp_model.plot(
            degree_class,
            div_base,
            marker="None",
            linestyle=":",
            linewidth=3,
            color=d_color,
        )

    ax_ltp_model.set_yscale("log")
    ax_ltp_model.set_xlabel(r"Degree class", fontsize=14)
    ax_ltp_model.set_ylabel(r"KL-Divergence", fontsize=14)
    ax_dist_model.set_ylabel(r"$Pr[\ell|\,s]$", fontsize=14)
    ax_ltp_model.set_xlim([-1, N])
    # ax_ltp_model.set_ylim([-0.1, 1.1])
    ax_dist_model.set_xlim([-1, N])

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

    ax_ltp_model.legend(loc="best", fancybox=True, fontsize=12, framealpha=1, ncol=1)

    plt.tight_layout(0.1)
    fig_model.savefig(os.path.join(params["path"], params["name"] + "_div_" + save))


if __name__ == "__main__":
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

    draw_div(args.path, args.save)
    draw_ltp(args.path, args.save)
