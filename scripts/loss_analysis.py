import argparse as ap
import dynalearn as dl
import h5py
import json
import os
import numpy as np
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

num_nodes = [10, 100, 1000]
avgk = 4

print("-------------------")
print("Building experiment")
print("-------------------")
data_filename = os.path.join(params["path"], params["name"] + ".h5")
h5file = h5py.File(data_filename, "w")
for n in num_nodes:
    params["graph"]["params"]["N"] = n
    params["graph"]["params"]["density"] = avgk / n
    experiment = dl.utilities.get_experiment(params)
    metrics = {"LossMetrics-N" + str(n): dl.utilities.LossMetrics(num_points=10000)}
    experiment.metrics = metrics
    experiment.load_weights(
        os.path.join(
            params["path"], params["name"] + "_" + params["path_to_best"] + ".h5"
        )
    )

    experiment.generate_data(1, 1000, 2)
    experiment.compute_metrics()
    experiment.save_metrics(h5file)

h5file.close()
