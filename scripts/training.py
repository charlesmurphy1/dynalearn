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

experiment = dl.utilities.get_experiment(params)
N = experiment.graph_model.num_nodes
if N < 500:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    "ModelLTPGenMetrics": dl.utilities.ModelLTPGenMetrics(
        degree_class, aggregator=m_aggregator
    ),
    "AttentionMetrics": dl.utilities.AttentionMetrics(num_points=100),
    "CountMetrics": dl.utilities.CountMetrics(
        aggregator=c_aggregator, num_points=10000
    ),
}
experiment.metrics = metrics
experiment.verbose = 1
experiment.generator.verbose = 1
experiment.generator.sampler.verbose = 0

h5file = h5py.File(os.path.join(params["path"], params["name"] + ".h5"))
dl.utilities.generate_data(params, experiment, h5file, overwrite=True)
dl.utilities.train_model(params, experiment, h5file, overwrite=True)
dl.utilities.analyze_model(params, experiment, h5file, overwrite=True)
h5file.close()
