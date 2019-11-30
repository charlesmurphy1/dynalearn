import argparse as ap
import dynalearn as dl
import h5py
import json
import os
import networkx as nx
import numpy as np
import tqdm
import sys

epsilon = 1e-3


def absorbing_state(mf):
    x = np.ones(mf.array_shape).astype(mf.dtype) * epsilon
<<<<<<< HEAD
    x[0] = 1 - epsilon
=======
    x[0] = 1
>>>>>>> 807f077f7c0103c3f6b4bcc70ef4f9f06193cabd
    x = mf.normalize_state(x)
    return x.reshape(-1)


def epidemic_state(mf):
    x = np.ones(mf.array_shape).astype(mf.dtype) * epsilon
<<<<<<< HEAD
    x[0] = 1 - epsilon
=======
    x[0] = 1
>>>>>>> 807f077f7c0103c3f6b4bcc70ef4f9f06193cabd
    x = 1 - x
    x = mf.normalize_state(x)
    return x.reshape(-1)


def generic_state(mf, s):
    x = np.ones(mf.array_shape).astype(mf.dtype) * epsilon
<<<<<<< HEAD
    x[s] = 1 - epsilon
=======
    x[s] = 1
>>>>>>> 807f077f7c0103c3f6b4bcc70ef4f9f06193cabd
    x = mf.normalize_state(x)
    return x.reshape(-1)


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

if os.path.exists(params["path"] + "/mf_erg.h5"):
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "r+")
else:
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "w")
avgk = np.concatenate((np.linspace(0.1, 3, 50), np.linspace(3.1, 10, 20)))
if "mf_k" in h5file:
    del h5file["mf_k"]
h5file.create_dataset("mf_k", data=avgk)
compute_stability = True

for k in avgk:
    print(f"avgk={k}")
    p_k = dl.meanfields.poisson_distribution(k, num_k=4)
    true_mf = dl.utilities.get_meanfield(params, p_k)
    gnn_mf = dl.meanfields.LearnedModelMF(p_k, experiment.model, tol=1e-5, verbose=0)

    print(f"\t Computing fixed points")
    true_low_fp = true_mf.search_fixed_point(x0=absorbing_state(true_mf))
    true_high_fp = true_mf.search_fixed_point(x0=epidemic_state(true_mf))
    gnn_low_fp = gnn_mf.search_fixed_point(x0=absorbing_state(gnn_mf))
    gnn_high_fp = gnn_mf.search_fixed_point(x0=epidemic_state(gnn_mf))
    print(f"\t Computing stability")
    if compute_stability:
        true_abs_stability = true_mf.stability(absorbing_state(true_mf))
        gnn_abs_stability = gnn_mf.stability(absorbing_state(gnn_mf))
    else:
        true_abs_stability = 1.1
        gnn_abs_stability = 1.1

    if true_abs_stability > 1.1 and gnn_abs_stability > 1.1:
        compute_stability = False

    h5file.create_dataset(f"k = {k}/true/low_fp", data=true_low_fp)
    h5file.create_dataset(f"k = {k}/true/high_fp", data=true_high_fp)
    h5file.create_dataset(f"k = {k}/true/abs_stability", data=true_abs_stability)

    h5file.create_dataset(f"k = {k}/gnn/low_fp", data=gnn_low_fp)
    h5file.create_dataset(f"k = {k}/gnn/high_fp", data=gnn_high_fp)
    h5file.create_dataset(f"k = {k}/gnn/abs_stability", data=gnn_abs_stability)

h5file.close()
