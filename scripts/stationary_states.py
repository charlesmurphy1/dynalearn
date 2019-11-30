import argparse as ap
import dynalearn as dl
import h5py
import json
import os
import networkx as nx
import numpy as np
import tqdm
import sys


def get_avg(x, s_dim):
    avg_x = []
    for i in range(s_dim):
        avg_x.append(np.mean(x == i))
    return np.array(avg_x)


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
num_states = int(experiment.model.num_states)
N = 2000
num_samples = 100
burn = 100
reshuffle = 0.1
tol = 1e-2
if os.path.exists(params["path"] + "/mf_erg.h5"):
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "r+")
else:
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "w")
experiment.model.num_nodes = N
avgk = np.linspace(0.1, 10, 20)
if "sim_k" in h5file:
    del h5file["sim_k"]
h5file.create_dataset("sim_k", data=avgk)

for k in avgk:
    print(f"avgk={k}")

    true_samples = []
    model_samples = []
    it = 0

    pb = tqdm.tqdm(range(num_samples), "Gathering samples (true): ")
    experiment.dynamics_model.graph = nx.gnp_random_graph(N, k / (N - 1))
    x0 = experiment.dynamics_model.states
    avg_x0 = get_avg(x0, num_states)
    while len(true_samples) < num_samples:
        x = experiment.dynamics_model.update()
        avg_x = get_avg(x, num_states)
        it += 1
        dist = np.sqrt(((avg_x - avg_x0) ** 2).sum())
        avg_x0 = avg_x * 1
        if dist < tol and it > burn:
            true_samples.append(avg_x)
            if (
                np.random.rand() < reshuffle
                or not experiment.dynamics_model.continue_simu
            ):
                experiment.dynamics_model.graph = nx.gnp_random_graph(N, k / (N - 1))
            it = 0
            pb.update()
    pb.close()

    pb = tqdm.tqdm(range(num_samples), "Gathering samples (model): ")
    adj = nx.to_numpy_array(nx.gnp_random_graph(N, k / (N - 1)))
    x = experiment.dynamics_model.initialize_states()
    avg_x0 = get_avg(x, num_states)
    while len(model_samples) < num_samples:
        x = experiment.model.update(x, adj)
        avg_x = get_avg(x, num_states)
        it += 1
        dist = np.sqrt(((avg_x - avg_x0) ** 2).sum())
        avg_x0 = avg_x * 1
        if dist < tol and it > burn:
            model_samples.append(avg_x)
            if np.random.rand() < reshuffle:
                adj = nx.to_numpy_array(nx.gnp_random_graph(N, k / (N - 1)))
                x = experiment.dynamics_model.initialize_states()
            it = 0
            pb.update()
    pb.close()
    print(len(true_samples), len(model_samples))
    true_avg = np.array(true_samples).mean(0)
    if np.any(np.isnan(true_avg)):
        print(true_samples)
    true_var = np.array(true_samples).var(0)
    if f"k={k}/sim_true_avg" in h5file:
        del h5file[f"k={k}/sim_true_avg"]
        del h5file[f"k={k}/sim_true_var"]
    h5file.create_dataset(f"k={k}/sim_true_avg", data=true_avg)
    h5file.create_dataset(f"k={k}/sim_true_var", data=true_var)

    model_avg = np.array(model_samples).mean(0)
    if np.any(np.isnan(model_avg)):
        print(model_samples)
    model_var = np.array(model_samples).var(0)
    if f"k={k}/sim_model_avg" in h5file:
        del h5file[f"k={k}/sim_model_avg"]
        del h5file[f"k={k}/sim_model_var"]
    h5file.create_dataset(f"k={k}/sim_model_avg", data=model_avg)
    h5file.create_dataset(f"k={k}/sim_model_var", data=model_var)
    print(true_avg, model_avg)
h5file.close()
