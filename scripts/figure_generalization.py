import argparse as ap
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

color_palette = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252"
}
markers=['o', 's', 'v', '*', '^']


def get_plot_label(i):
    alphabet = 'abcdefghijklmnpoqrstuvwxyz'
    return "(" + alphabet[i] + ")"

def kl_div(p, q):
    p[p<=0] = 1e-15
    q[q<=0] = 1e-15
    ans = p * np.log(p / q)
    ans = np.sum(ans, axis=-1)
    return ans

def add_subplot_axes(ax, rect, axisbg='w'):
    """Make a subplot in ax."""
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    trans_figure = fig.transFigure.inverted()
    infig_position = trans_figure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x, y, width, height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def main():
    prs = ap.ArgumentParser(description="Get local transition probability \
                                         figure from path to parameters.")
    prs.add_argument('--path', '-p', type=str, required=True,
                     help='Path to parameters.')
    prs.add_argument('--save', '-s', type=str, required=False,
                     help='Path where to save.')
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()

    with open(args.path, 'r') as f:
        params = json.load(f)

    filename = os.path.join(params["path"], params["name"] + "_data.h5")
    an_filename = os.path.join(params["path"], params["name"] + "_analytics.h5")
    data = h5py.File(filename, 'r')
    an_data = h5py.File(an_filename, 'r')
    graph_label = params["graph"]["name"] + "_0"

    state_label = []
    for key in data["dynamics/params"]:
        if key[:5] == "state":
            state_label.append(key[6:])



    N = params["graph"]["params"]["N"]
    g = list(data["data/"].keys())[0]
    kmin = np.min(np.sum(data["data/" + g + "/adj_matrix"], 0))
    kmax = np.max(np.sum(data["data/" + g + "/adj_matrix"], 0)) - 1
    degrees = np.sum(data["data/" + g + "/adj_matrix"], axis=0)
    degree_class = np.arange(N)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    axx = ax.twinx()
    axx.hist(degrees, bins=np.arange(0, kmax+1, 1),
             color=color_palette["grey"], density=True,
             alpha=0.5)
    axx.set_ylabel('Degree distribution', fontsize=14)

    for i, in_s in enumerate(state_label):
        if in_s == "S":
            color = color_palette["blue"]
        elif in_s == "I":
            color = color_palette["orange"]
        elif in_s == "R":
            color = color_palette["purple"]
        fill_color = color_palette["grey"]
        truth_gen = np.zeros((N, N, len(state_label)))
        model_gen = np.zeros((N, N, len(state_label)))
        base_line = np.zeros((N, N, len(state_label)))
        for j, out_s in enumerate(state_label): 
            truth_gen[:, :, j] = an_data["analytics/generalization/ground_truth/" + in_s + "_to_" + out_s][...]
            model_gen[:, :, j] = an_data["analytics/generalization/model/" + in_s + "_to_" + out_s][...]
        base_line = np.ones((N, N, len(state_label))) / len(state_label)
        ax.plot(degree_class, np.mean(kl_div(truth_gen, model_gen), axis=-1),
                marker='None', linestyle='-.', linewidth=2, color=color,
                markeredgecolor='k', markeredgewidth=1)
        ax.plot(degree_class, np.mean(kl_div(truth_gen, base_line), axis=-1),
                marker='None', linestyle='--', linewidth=2, color=color)
        ax.plot([-1], [-1], marker='s', linestyle='-', markersize=10, color=color,
                label=r'$y = {0}$'.format(in_s))

    ax.set_xlabel(r"Degree class", fontsize=14)
    ax.set_ylabel(r"$\langle D_{KL}\Big(p(x|y,\ell)||\hat{p}(x|y,\ell)\Big)\rangle_\ell$", fontsize=14)
    ax.set_xlim([kmin, kmax])
    ax.set_ylim([0, 1])
    # ax.set_yscale('log')

    # Making legend
    ax.plot([-1], [-1], linestyle='-.', linewidth=2, marker='None', color=color_palette["grey"],
            label="True-Model", alpha=0.5)
    ax.plot([-1], [-1], marker='None', linestyle='--',
            color=color_palette["grey"],label="True-Random", alpha=0.5)
    ax.plot([-1], [-1], marker='s', linestyle='None',
            color=color_palette["grey"], markersize=10,
            label=r"$P(k)$", alpha=0.5)
    ax.legend(loc='best', ncol=1, fancybox=True, fontsize=12, framealpha=1)
    # axx.legend(label, marker, loc='best', ncol=1, fancybox=True, fontsize=12, framealpha=1)

    plt.tight_layout()
    # plt.show()
    fig.savefig(os.path.join(params["path"], params["name"] + "_" + args.save))




if __name__ == '__main__':
    main()