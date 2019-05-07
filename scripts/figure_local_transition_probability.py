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
    labels = []
    markers = []

    for t in transitions:
        in_s, out_s = t[0], t[1]
        if in_s == "S" and out_s=="I":
            color = color_palette["orange"]
        elif in_s == "I" and out_s=="S":
            color = color_palette["blue"]
        elif in_s == "I" and out_s=="R":
            color = color_palette["purple"]
        labels.append(r'$P({0}|{1}, \ell)$'.format(out_s, in_s))
        markers.append(ax.plot([-1], [-1], linestyle='none', marker='s', markersize=10,
                       color=color,label=labels[-1]))
        fill_color = color_palette["grey"]

        ground_truth_ltp = an_data["analytics/local_trans_prob/ground_truth/" + in_s + "_to_" + out_s][...]
        model_ltp = an_data["analytics/local_trans_prob/model/" + in_s + "_to_" + out_s][...]
        estimate_ltp = an_data["analytics/local_trans_prob/estimate/" +in_s + "_to_" + out_s][...]

        m_index = np.where(model_ltp[:, 0]!=0)
        e_index = np.where(estimate_ltp[:, 0]!=0)

        ax.fill_between(k[m_index], model_ltp[:, 0][m_index] - model_ltp[:, 1][m_index],
                        model_ltp[:, 0][m_index] + model_ltp[:, 1][m_index],
                        color=fill_color, alpha=0.3)
        ax.fill_between(k[e_index], estimate_ltp[:, 0][e_index] - estimate_ltp[:, 1][e_index],
                        estimate_ltp[:, 0][e_index] + estimate_ltp[:, 1][e_index],
                        color=fill_color, alpha=0.3)
        ax.plot(k[m_index], model_ltp[:, 0][m_index], marker='s', linestyle='None', alpha=1,
                      color=color, markeredgewidth=1, markeredgecolor='k')
        ax.plot(k[e_index], estimate_ltp[:, 0][e_index], marker='v', linestyle='None', alpha=1,
                      color=color, markeredgewidth=1, markeredgecolor='k')
        ax.plot(k, ground_truth_ltp, marker='None', linestyle='-.', linewidth=2, 
                      color=color)
        ax.set_xlabel(r"Infected degree $\ell$", fontsize=14)
        ax.set_ylabel(r"Transition Probability", fontsize=14)
        ax.set_xlim([0, kmax])
        ax.set_ylim([0, 1])

    # Making legend
    labels.extend(["ground truth", "model", "estimate"])
    markers = [ax.plot([-1], [-1], linestyle='-.', linewidth=2,
                       color=color_palette["grey"],label=labels[2]),
               ax.plot([-1], [-1], marker='s', markeredgewidth=1,
                       markeredgecolor='k', linestyle='None',
                       color=color_palette["grey"],label=labels[3]),
                ax.plot([-1], [-1], marker='v', markeredgewidth=1,
                       markeredgecolor='k', linestyle='None',
                       color=color_palette["grey"],label=labels[4])]
    ax.legend(loc='upper left', fancybox=True, fontsize=12, framealpha=1)
    # axx.legend(labels, markers, loc='upper left', fancybox=True, fontsize=12, framealpha=1)

    plt.tight_layout(0.1)
    fig.savefig(os.path.join(params["path"], params["name"] + "_" + args.save))




if __name__ == '__main__':
    main()