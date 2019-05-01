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

    if params["graph"]["params"]["N"] <= 15:
        states = []
        for k in data["dynamics/params"]:
            if k[:5] == "state":
                states.append(k[6:])


        fig, ax = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

        N = params["graph"]["params"]["N"]
        g = list(data["data/"].keys())[0]


        ground_truth_mm = an_data["analytics/markovmatrix/" + g + "/ground_truth"][...]
        model_mm = an_data["analytics/markovmatrix/" + g + "/model"][...]
        occurence = an_data["analytics/occurence/" + g][...]

        occurence /= np.sum(occurence)
        occurence[occurence == 0] = 1e-15
        occurence = np.log(occurence)


        p0 = ax[0].imshow(ground_truth_mm, vmin=-20, vmax=0, origin='lower')
        ax[0].set_xlabel(r'Configuration at ($t$)', fontsize=14)
        ax[0].set_ylabel(r'Configuration at ($t + 1$)', fontsize=14)
        ax[0].text(0.85, 0.1, get_plot_label(0),
                   bbox=dict(facecolor='white', edgecolor='black'),
                   transform=ax[0].transAxes, fontsize=14)
        plt.colorbar(p0, ax=ax[0])
        p1 = ax[1].imshow(model_mm, vmin=-20, vmax=0, origin='lower')
        ax[1].set_xlabel(r'Configuration at ($t$)', fontsize=14)
        ax[1].text(0.85, 0.1, get_plot_label(1),
                   bbox=dict(facecolor='white', edgecolor='black'),
                   transform=ax[1].transAxes, fontsize=14)
        plt.colorbar(p1, ax=ax[1])
        p2 = ax[2].imshow(occurence, vmin=-20, vmax=0, origin='lower')
        ax[2].text(0.85, 0.1, get_plot_label(2),
                   bbox=dict(facecolor='white', edgecolor='black'),
                   transform=ax[2].transAxes, fontsize=14)
        ax[2].set_xlabel(r'Configuration at ($t$)', fontsize=14)
        plt.colorbar(p2, ax=ax[2])
        plt.tight_layout(4. ,h_pad=4.)

        fig.savefig(os.path.join(params["path"], params["name"] + "_" + args.save))




if __name__ == '__main__':
    main()