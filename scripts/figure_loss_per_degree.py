import argparse as ap
import h5py
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

color_palette = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "purple": "#542788",
    "red": "#d73027",
    "grey": "#525252"
}

def get_plot_label(i):
    alphabet = 'abcdefghijklmnpoqrstuvwxyz'
    return "(" + alphabet[i] + ")"

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

    states = []
    for k in data["dynamics/params"]:
        if k[:5] == "state":
            states.append(k[6:])


    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    transitions = params["dynamics"]["params"]["relevant_transitions"]

    N = params["graph"]["params"]["N"]
    g = list(data["data/"].keys())[0]
    kmin = np.min(np.sum(data["data/" + g + "/adj_matrix"], 0))
    kmax = np.max(np.sum(data["data/" + g + "/adj_matrix"], 0)) - 1
    k = np.arange(N)
    degrees = np.sum(data["data/" + g + "/adj_matrix"], 0)

    axx = ax.twinx()
    axx.hist(degrees, bins=np.arange(0, kmax+1, 1),
             color=color_palette["grey"], density=True,
             alpha=0.5)
    axx.set_ylabel('Degree distribution')

    color = color_palette["blue"]
    fill_color = color_palette["grey"]

    data_loss = an_data["analytics/losses/from_data_per_degree"][...]
    dynamics_loss = an_data["analytics/losses/from_dynamics_per_degree"][...]
    ax.fill_between(k, data_loss[:, 0] - data_loss[:, 1],
                    data_loss[:, 0] + data_loss[:, 1],
                    color=fill_color, alpha=0.3)
    ax.fill_between(k, dynamics_loss[:, 0] - dynamics_loss[:, 1],
                          dynamics_loss[:, 0] + dynamics_loss[:, 1],
                          color=fill_color, alpha=0.3)
    ax.plot(k, data_loss[:, 0], marker='s', linestyle='None', alpha=1,
                  color=color, markeredgewidth=1, markeredgecolor='k')
    ax.plot(k, dynamics_loss[:, 0], marker='v', linestyle='None', alpha=1,
                  color=color, markeredgewidth=1, markeredgecolor='k')
    ax.set_xlabel(r"Degree $k$", fontsize=14)
    ax.set_ylabel(r"Loss", fontsize=14)
    ax.set_xlim([0, kmax])
    # ax.set_ylim([0, 1])

    # Making legend
    labels = ["Binary data", "Dynamics prob."]
    markers = [ax.plot([-1], [-1], marker='s', markeredgewidth=1,
                       markeredgecolor='k', linestyle='None',
                       color=color_palette["grey"],label=labels[0]),
               ax.plot([-1], [-1], marker='v', markeredgewidth=1,
                       markeredgecolor='k', linestyle='None',
                       color=color_palette["grey"],label=labels[1])]
    ax.legend(loc='upper left', shadow=False,
              fancybox=False, prop={'size': 12}, frameon=False,
              numpoints=1, ncol=1)
    plt.tight_layout(0.1)
    fig.savefig(os.path.join(params["path"], params["name"] + "_" + args.save))



if __name__ == '__main__':
    main()