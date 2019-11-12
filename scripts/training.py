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

degree_class = np.unique(np.linspace(1, 100, 10).astype("int"))
m_aggregator = dl.utilities.get_aggregator(params)
m_aggregator.operation = "mean"
c_aggregator = dl.utilities.get_aggregator(params)
c_aggregator.operation = "sum"
metrics = {
    "DynamicsLTPMetrics": dl.utilities.DynamicsLTPMetrics(
        aggregator=m_aggregator, num_points=1000
    ),
    "ModelLTPMetrics": dl.utilities.ModelLTPMetrics(
        aggregator=m_aggregator, num_points=1000
    ),
    "EstimatorLTPMetrics": dl.utilities.EstimatorLTPMetrics(
        aggregator=m_aggregator, num_points=10000
    ),
    "DynamicsLTPGenMetrics": dl.utilities.DynamicsLTPGenMetrics(
        degree_class, aggregator=m_aggregator
    ),
    "ModelLTPMetrics": dl.utilities.ModelLTPMetrics(
        aggregator=m_aggregator, num_points=1000
    ),
    "AttentionMetrics": dl.utilities.AttentionMetrics(num_points=100),
    "LossMetrics": dl.utilities.LossMetrics(num_points=1000),
    "CountMetrics": dl.utilities.CountMetrics(
        aggregator=c_aggregator, num_points=50000
    ),
}

print("-------------------")
print("Building experiment")
print("-------------------")
experiment = dl.utilities.get_experiment(params)

N = experiment.graph_model.num_nodes
data_filename = os.path.join(params["path"], params["name"] + ".h5")

if N < 500:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
h5file = h5py.File(os.path.join(params["path"], params["name"] + ".h5"))
# dl.utilities.train_model(params, experiment)
experiment.load_weights(
    os.path.join(params["path"], params["name"] + "_" + params["path_to_best"] + ".h5")
)
experiment.load_data(h5file)

experiment.metrics = metrics
# dl.utilities.analyze_model(params, experiment)
experiment.load_metrics(h5file)
experiment = dl.utilities.make_figures(params, experiment)
