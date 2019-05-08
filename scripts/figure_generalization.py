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
    degrees = np.sum(data["data/" + g + "/adj_matrix"], axis=0)
    degree_class = an_data["analytics/generalization/degree/"][...]
    kmin = np.min(degree_class)
    kmax = np.max(degree_class)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    axx = ax.twinx()
    axx.hist(degrees, bins=np.arange(0, kmax+1, 1),
             color=color_palette["grey"], density=True,
             alpha=0.5)

    axx.set_ylabel('Degree distribution', fontsize=14)
    axx.set_yscale('log')

    axin = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
    axin.hist(degrees, bins=np.arange(0, kmax+1, 1),
             color=color_palette["grey"], density=True,
             alpha=0.5)
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
        kl_div_model = kl_div(truth_gen, model_gen)
        kl_div_base = kl_div(truth_gen, base_line)
        div_model = [np.mean(kl_div_model[k, :k]) for k in degree_class]
        div_base = [np.mean(kl_div_base[k, :k]) for k in degree_class]

        ax.plot(degree_class, div_model,
                marker='None', linestyle='-.', linewidth=2, color=color,
                markeredgecolor='k', markeredgewidth=1)
        ax.plot(degree_class, div_base,
                marker='None', linestyle='--', linewidth=2, color=color)
        ax.plot([-1], [-1], marker='s', linestyle='None', markersize=10, color=color,
                label=r'$y = {0}$'.format(in_s))

        axin.plot(degree_class[kmin:kmax+1], div_model[kmin:kmax+1],
                marker='None', linestyle='-.', linewidth=2, color=color,
                markeredgecolor='k', markeredgewidth=1)
        axin.plot(degree_class[kmin:kmax+1], div_base[kmin:kmax+1],
                marker='None', linestyle='--', linewidth=2, color=color)

    ax.set_xlabel(r"Degree class", fontsize=14)
    ax.set_ylabel(r"$\langle D_{KL}\Big(p(x|y,\ell)||\hat{p}(x|y,\ell)\Big)\rangle_\ell$", fontsize=14)
    ax.set_xlim([0, N])
    ax.set_yscale('log')
    axin.set_xlim([kmin, kmax])
    axin.set_yscale('log')

    # Making legend
    ax.plot([-1], [-1], linestyle='-.', linewidth=2, marker='None', color=color_palette["grey"],
            label="True-Model", alpha=0.5)
    ax.plot([-1], [-1], marker='None', linestyle='--',
            color=color_palette["grey"],label="True-Uniform", alpha=0.5)
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