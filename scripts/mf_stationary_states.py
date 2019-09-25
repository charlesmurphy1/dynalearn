import argparse as ap
import dynalearn as dl
import h5py
import json
import os
import networkx as nx
import numpy as np
import tqdm
import sys


prs = ap.ArgumentParser(
    description="Get local transition probability \
                                     figure from path to parameters."
)
prs.add_argument("--path", "-p", type=str, required=True, help="Path to parameters.")

if len(sys.argv) == 1:
    prs.print_help()
    sys.exit(1)
args = prs.parse_args()

with open(args.path, "r") as f:
    print(args.path)
    params = json.load(f)

experiment = dl.utilities.get_experiment(params)
experiment.load_weights(
    os.path.join(params["path"], params["name"] + "_" + params["path_to_best"] + ".h5")
)

if os.path.isfile(params["path"] + "/mf_erg.h5"):
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "r+")
else:
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "w")
avgk = np.linspace(0.1, 10, 20)
if "k_values" in h5file:
    del h5file["k_values"]
h5file.create_dataset("k_values", data=avgk)

for k in avgk:
    print(f"avgk = {k}")
    p_k = dl.meanfields.poisson_distribution(k, num_k=5)
    true_mf = dl.utilities.get_meanfield(params, p_k)
    learned_mf = dl.meanfields.LearnedModelMF(
        p_k, experiment.model, tol=1e-5, verbose=1
    )

    true_mf.compute_fixed_points()
    true_mf.compute_fixed_points(x0=true_mf.abs_state(0), epsilon=1e-15)
    true_mf.add_fixed_points(true_mf.abs_state(0))
    true_mf.compute_stability()
    true_mf.save(f"k={k}/true_mf", h5file)

    learned_mf.compute_fixed_points()
    learned_mf.compute_fixed_points(x0=learned_mf.abs_state(0), epsilon=1e-15)
    learned_mf.add_fixed_points(learned_mf.abs_state(0))
    learned_mf.compute_stability()
    learned_mf.save(f"k={k}/learned_mf", h5file)

h5file.close()
