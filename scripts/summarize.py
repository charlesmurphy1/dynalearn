import h5py
import numpy as np
import dynalearn as dl
import argparse
import os
from scipy.spatial.distance import jensenshannon


def generate_ltp_data(h5file, experiment):
    true = experiment.metrics["TrueLTPMetrics"]
    gnn = experiment.metrics["GNNLTPMetrics"]
    mle = experiment.metrics["MLELTPMetrics"]
    transitions = {(0, 1): "S-I", (1, 0): "I-S"}

    for t in transitions:
        name = "ltp-true/{0}".format(transitions[t])
        x, y, el, eh = true.aggregate(
            true.data["ltp/val"],
            in_state=t[0],
            out_state=t[1],
            err_operation="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)

        name = "ltp-gnn/{0}".format(transitions[t])
        x, y, el, eh = gnn.aggregate(
            gnn.data["ltp/val"],
            in_state=t[0],
            out_state=t[1],
            err_operation="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)

        name = "ltp-mle/{0}".format(transitions[t])
        x, y, el, eh = mle.aggregate(
            mle.data["ltp/train"],
            in_state=t[0],
            out_state=t[1],
            err_operation="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)


def generate_error_data(h5file, experiment):
    true = experiment.metrics["TrueStarLTPMetrics"]
    gnn = experiment.metrics["GNNStarLTPMetrics"]
    uni = experiment.metrics["UniformStarLTPMetrics"]
    _true = experiment.metrics["TrueLTPMetrics"]
    mle = experiment.metrics["MLELTPMetrics"]

    name = "jsd-true-gnn"
    jsd = true.compare("ltp", gnn, func=jensenshannon, verbose=0)
    x, y, el, eh = true.aggregate(jsd, for_degree=True, err_operation="percentile")
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)

    name = "jsd-true-uni"
    jsd = true.compare("ltp", uni, func=jensenshannon, verbose=0)
    x, y, el, eh = true.aggregate(jsd, for_degree=True, err_operation="percentile")
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)

    name = "jsd-true-mle"
    jsd = _true.compare("ltp/train", "ltp/train", mle, func=jensenshannon)
    x, y, el, eh = _true.aggregate(jsd, for_degree=True, err_operation="percentile")
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)


def generate_ssmf_data(h5file, experiment):
    true_mf = experiment.metrics["TruePEMFMetrics"]
    true_ss = experiment.metrics["TruePESSMetrics"]
#    h5file.create_dataset("mf_parameters", data=true_mf.data["parameters"])
#    h5file.create_dataset("ss_parameters", data=true_ss.data["parameters"])

    h5file.create_dataset("true-fixed_points", data=true_mf.data["fixed_points"])
    h5file.create_dataset("true-thresholds", data=true_mf.data["thresholds"])
    h5file.create_dataset("true-ss_avg", data=true_ss.data["avg"])
    h5file.create_dataset("true-ss_std", data=true_ss.data["std"])

    gnn_mf = experiment.metrics["GNNPEMFMetrics"]
    gnn_ss = experiment.metrics["GNNPESSMetrics"]
    h5file.create_dataset("gnn-fixed_points", data=gnn_mf.data["fixed_points"])
    h5file.create_dataset("gnn-thresholds", data=gnn_mf.data["thresholds"])
    h5file.create_dataset("gnn-ss_avg", data=gnn_ss.data["avg"])
    h5file.create_dataset("gnn-ss_std", data=gnn_ss.data["std"])


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    "-p",
    type=str,
    metavar="PATH",
    help="Path to config pickle file.",
    required=True,
)

args = parser.parse_args()

config = dl.ExperimentConfig.config_from_file(args.config_path)
experiment = dl.Experiment(config.config, verbose=0)
experiment.load()
path = os.path.join(experiment.path_to_dir, "summaries")
if not os.path.exists(path):
    os.makedirs(path)

h5file = h5py.File(os.path.join(path, "{0}.h5".format(experiment.name)), "w")
generate_ltp_data(h5file, experiment)
generate_error_data(h5file, experiment)
generate_ssmf_data(h5file, experiment)
