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

def main():
	prs = ap.ArgumentParser(description="Get local transition probability \
										 figure from path to parameters.")
	prs.add_argument('--path', '-p', type=str, required=True,
					 help='Path to parameters.')
	if len(sys.argv) == 1:
		prs.print_help()
		sys.exit(1)
	args = prs.parse_args()
	with open(args.path, 'r') as f:
		params = json.load(f)

	filename = os.path.join(params["path"], params["experiment_name"] + ".h5")
	an_filename = os.path.join(params["path"], params["experiment_name"] + "_analytics.h5")
	data = h5py.File(filename, 'r')
	an_data = h5py.File(an_filename, 'r')
	graph_label = params["graph"]["name"] + "_0"

	states = []
	for k in data["dynamics/params"]:
		if k[:5] == "state":
			states.append(k[6:])


	fig, ax = plt.subplots(len(states),
						   len(states),
						   figsize=(4 * len(states), 3 * len(states)),
						   sharex=True, sharey=True)

	N = params["graph"]["params"]["N"]
	kmin = np.min(np.sum(data["data/ERGraph_0/adj_matrix"], 0))
	kmax = np.max(np.sum(data["data/ERGraph_0/adj_matrix"], 0)) - 1
	k = np.arange(N)
	for i, in_s in enumerate(states):
		for j, out_s in enumerate(states):
			ground_truth_ltp = an_data["analytics/local_trans_prob/ground_truth/" + in_s + "_to_" + out_s][...]
			model_ltp = an_data["analytics/local_trans_prob/model/" + in_s + "_to_" + out_s][...]
			estimate_ltp = an_data["analytics/local_trans_prob/estimate/" +in_s + "_to_" + out_s][...]
			ax[i, j].plot(k, ground_truth_ltp, marker='None', linestyle='-',
						  color=color_palette["blue"])
			ax[i, j].plot(k, model_ltp[:, 0], marker='s', linestyle='None',
						  color=color_palette["orange"])
			ax[i, j].plot(k, estimate_ltp[:, 0], marker='v', linestyle='None',
						  color=color_palette["purple"])

			ax[i, j].fill_between(k, model_ltp[:, 0] - model_ltp[:, 1],
								  model_ltp[:, 0] + model_ltp[:, 1],
								  color=color_palette["grey"], alpha=0.3)
			ax[i, j].fill_between(k, estimate_ltp[:, 0] - estimate_ltp[:, 1],
								  estimate_ltp[:, 0] + estimate_ltp[:, 1],
								  color=color_palette["grey"], alpha=0.3)
			if i==len(states) - 1: ax[i, j].set_xlabel(r"Infected degree $\ell$")
			if j==0:ax[i, j].set_ylabel(r"Transition Probability")
			ax[i, j].set_title(r"$P(" + out_s + r"|" + in_s + r", \ell$)")
			ax[i, j].set_xlim([kmin, kmax])
			ax[i, j].set_ylim([0, 1])

	plt.show()




if __name__ == '__main__':
	main()