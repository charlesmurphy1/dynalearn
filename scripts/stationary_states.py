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
num_states = int(experiment.model.num_states)
N = 2000
T = 1000
burn = 100
tol = 1e-3
if os.path.exists(params["path"] + "/mf_erg.h5"):
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "r+")
else:
    h5file = h5py.File(params["path"] + "/mf_erg.h5", "w")
# h5file = h5py.File(params["path"] + "/mf_erg.h5", "w")

avgk = np.linspace(0.1, 10, 20)
if "k_values" in h5file:
    del h5file["k_values"]
h5file.create_dataset("k_values", data=avgk)

for k in avgk:
    print(f"avgk = {k}")
    g = nx.gnp_random_graph(N, k / (N - 1))
    experiment.dynamics_model.graph = g
    samples = []
    it = 0
    avg_x0 = get_avg(experiment.dynamics_model.states, num_states)
    pb = tqdm.tqdm(range(T), "Generating data: ")
    for i in range(T):
        x = experiment.dynamics_model.update()
        avg_x = get_avg(x, num_states)
        it += 1
        dist = np.sqrt(((avg_x - avg_x0) ** 2).sum())
        avg_x0 = avg_x * 1
        if dist < tol and it > burn:
            samples.append(avg_x)
            it = 0

        pb.update()
    pb.close()
    avg = np.array(samples).mean(0)
    var = np.array(samples).var(0)
    if f"k = {k}/sim_avg" in h5file:
        del h5file[f"k = {k}/sim_avg"]
        del h5file[f"k = {k}/sim_var"]
    h5file.create_dataset(f"k = {k}/sim_avg", data=avg)
    h5file.create_dataset(f"k = {k}/sim_var", data=var)
h5file.close()
