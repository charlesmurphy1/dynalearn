import argparse as ap
import dynalearn as dl
import h5py
import json
import os
import numpy as np
import sys
import tqdm


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

c_aggregator = dl.utilities.get_aggregator(params)
c_aggregator.operation = "sum"

print("-------------------")
print("Building experiment")
print("-------------------")
T = 1000
num_samples = 10
params["generator"]["params"]["num_sample"] = T
ts = np.logspace(0, np.log10(params["generator"]["params"]["num_sample"]), 10).astype(
    "int"
)
h5file = h5py.File(os.path.join(params["path"], "dataset_stats.h5"))

if f"values" in h5file:
    del h5file[f"values"]
h5file.create_dataset(f"values", data=ts)
print(params["generator"]["params"]["num_sample"])
for _ts in ts:
    pbar = tqdm.tqdm(range(num_samples), f"Generating datasets: ts = {_ts}")
    neff = np.zeros(num_samples)
    entropy = np.zeros(num_samples)
    max_entropy = np.zeros(num_samples)
    for i in range(num_samples):
        metrics = {
            "CountMetrics": dl.utilities.CountMetrics(
                aggregator=c_aggregator, num_points=50000, verbose=0
            )
        }
        params["generator"]["params"]["T"] = _ts
        experiment = dl.utilities.get_experiment(params)
        experiment.verbose = 0
        experiment.generator.verbose = 0
        experiment.generator.sampler.verbose = 0
        N = experiment.graph_model.num_nodes
        # print("----------------")
        # print("Building dataset")
        # print("----------------")
        experiment.generate_data(
            params["generator"]["params"]["num_graphs"],
            params["generator"]["params"]["num_sample"],
            params["generator"]["params"]["T"],
        )
        experiment.metrics = metrics
        experiment.compute_metrics()
        neff[i] = experiment.metrics["CountMetrics"].effective_samplesize("train")
        entropy[i] = experiment.metrics["CountMetrics"].entropy("train")
        # max_entropy[i] = experiment.metrics["CountMetrics"].max_entropy()
        pbar.update()
    if f"ts = {_ts}/neff" in h5file:
        del h5file[f"ts = {_ts}/neff"]
        del h5file[f"ts = {_ts}/entropy"]

    h5file.create_dataset(f"ts = {_ts}/neff", data=neff)
    h5file.create_dataset(f"ts = {_ts}/entropy", data=entropy)
    # h5file.create_dataset(f"ts = {_ts}/max_entropy", data=neff)
    # dl.utilities.analyze_model(params, experiment)
h5file.close()
