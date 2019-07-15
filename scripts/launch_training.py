import argparse as ap
import dynalearn as dl
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

degree_class = np.unique(np.linspace(1, 100, 10).astype("int"))
metrics = {
    "DynamicsLTPMetrics": dl.utilities.DynamicsLTPMetrics(),
    "ModelLTPMetrics": dl.utilities.ModelLTPMetrics(),
    "EstimatorLTPMetrics": dl.utilities.EstimatorLTPMetrics(num_points=10000),
    "DynamicsLTPGenMetrics": dl.utilities.DynamicsLTPGenMetrics(degree_class),
    "ModelLTPGenMetrics": dl.utilities.ModelLTPGenMetrics(degree_class),
    "ModelJSDGenMetrics": dl.utilities.ModelJSDGenMetrics(degree_class),
    "BaseJSDGenMetrics": dl.utilities.BaseJSDGenMetrics(degree_class),
    "AttentionMetrics": dl.utilities.AttentionMetrics(),
    "LossMetrics": dl.utilities.LossMetrics(),
    "CountMetrics": dl.utilities.CountMetrics(),
}

print("-------------------")
print("Building experiment")
print("-------------------")
experiment = dl.utilities.get_experiment(params)
N = experiment.graph_model.num_nodes
data_filename = os.path.join(params["path"], params["name"] + ".h5")

if N < 500:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

dl.utilities.train_model(params, experiment)
experiment.load_weights(
    os.path.join(params["path"], params["name"] + "_" + params["path_to_best"] + ".h5")
)
experiment.load_data(os.path.join(params["path"], params["name"] + ".h5"))

dl.utilities.analyze_model(params, experiment, metrics)
experiment.load_metrics(os.path.join(params["path"], params["name"] + ".h5"))
experiment = dl.utilities.make_figures(params, experiment)
