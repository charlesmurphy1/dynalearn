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
    "purple": "#9A80B9",
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


    num_layer = len(an_data["analytics/attn_coeff"])
    fig, ax = plt.subplots(1, num_layer, figsize=(6 * num_layer, 5), sharex=True)
    transitions = params["dynamics"]["params"]["relevant_transitions"]

    N = params["graph"]["params"]["N"]
    g = list(data["data/"].keys())[0]

    labels = []
    markers = []

    for i, l in enumerate(an_data["analytics/attn_coeff"]):
        for t in transitions:
            in_s, out_s = t[0], t[1]
            # if in_s == "S" and out_s=="I":
            #     color = color_palette["orange"]
            # elif in_s == "I" and out_s=="R":
            #     color = color_palette["purple"]
            # else:
            #     print(in_s, out_s)
            # labels.append(r'$\alpha({0}\leftarrow {1})$'.format(in_s, out_s))
            # markers.append(ax[i].plot([-1], [-1], linestyle='none', marker='s', markersize=10,
            #                color=color,label=labels[-1]))
            # fill_color = color_palette["grey"]

            attn_coeff = an_data["analytics/attn_coeff/" + l + "/" + in_s + "_to_" + out_s][...]
            ax[i].hist(attn_coeff, density=True, alpha=0.5, label=r'$\alpha({0}\leftarrow {1})$'.format(in_s, out_s))
            ax[i].set_xlabel(r"Attention coefficient", fontsize=14)
            if i == 0:
                ax[i].set_ylabel(r"Distribution", fontsize=14)
            ax[i].set_xlim([0, 1])
                # ax[i].set_xscale('log')
                # ax[i].set_yscale('log')
            # ax.set_ylim([0, 1])

    # Making legend
    # labels.extend(["ground truth", "model", "estimate"])
    # markers = [ax.plot([-1], [-1], linestyle='-.', linewidth=2,
    #                    color=color_palette["grey"],label=labels[2]),
    #            ax.plot([-1], [-1], marker='s', markeredgewidth=1,
    #                    markeredgecolor='k', linestyle='None',
    #                    color=color_palette["grey"],label=labels[3]),
    #             ax.plot([-1], [-1], marker='v', markeredgewidth=1,
    #                    markeredgecolor='k', linestyle='None',
    #                    color=color_palette["grey"],label=labels[4])]
    ax[-1].legend(loc='upper left', fancybox=True, fontsize=12, framealpha=1)
    # axx.legend(labels, markers, loc='upper left', fancybox=True, fontsize=12, framealpha=1)

    plt.tight_layout(0.1)
    # plt.show()
    fig.savefig(os.path.join(params["path"], params["name"] + "_" + args.save))




if __name__ == '__main__':
    main()