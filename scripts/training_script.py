import h5py
import numpy as np
import dynalearn as dl
import argparse
import os
from scipy.spatial.distance import jensenshannon
import tensorflow as tf


def summarize_ltp(h5file, experiment):
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


def summarize_error(h5file, experiment):
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


def summarize_ssmf(h5file, experiment):
    true_mf = experiment.metrics["TruePEMFMetrics"]
    true_ss = experiment.metrics["TruePESSMetrics"]
    h5file.create_dataset("mf-parameters", data=true_mf.data["parameters"])
    h5file.create_dataset("ss-parameters", data=true_ss.data["parameters"])

    h5file.create_dataset("mf-true/fixed_points", data=true_mf.data["fixed_points"])
    h5file.create_dataset("mf-true/thresholds", data=true_mf.data["thresholds"])
    h5file.create_dataset("ss-true/avg", data=true_ss.data["avg"])
    h5file.create_dataset("ss-true/std", data=true_ss.data["std"])

    gnn_mf = experiment.metrics["GNNPEMFMetrics"]
    gnn_ss = experiment.metrics["GNNPESSMetrics"]
    h5file.create_dataset("mf-gnn/fixed_points", data=gnn_mf.data["fixed_points"])
    h5file.create_dataset("mf-gnn/thresholds", data=gnn_mf.data["thresholds"])
    h5file.create_dataset("mf-gnn/avg", data=gnn_ss.data["avg"])
    h5file.create_dataset("mf-gnn/std", data=gnn_ss.data["std"])


def get_config(args):
    if args.config == "sis_er":
        return dl.ExperimentConfig.sis_er(
            args.num_samples, args.path_to_data, args.path_to_model
        )
    elif args.config == "sis_ba":
        return dl.ExperimentConfig.sis_ba(
            args.num_samples, args.path_to_data, args.path_to_model
        )
    elif args.config == "plancksis_er":
        return dl.ExperimentConfig.plancksis_er(
            args.num_samples, args.path_to_data, args.path_to_model
        )
    elif args.config == "plancksis_ba":
        return dl.ExperimentConfig.plancksis_ba(
            args.num_samples, args.path_to_data, args.path_to_model
        )
    elif args.config == "sissis_er":
        return dl.ExperimentConfig.sissis_er(
            args.num_samples, args.path_to_data, args.path_to_model
        )
    elif args.config == "sissis_ba":
        return dl.ExperimentConfig.sissis_ba(
            args.num_samples, args.path_to_data, args.path_to_model
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
    type=str,
    metavar="Config",
    help="Config name of the experiment.",
    choices=[
        "sis_er",
        "sis_ba",
        "plancksis_er",
        "plancksis_ba",
        "sissis_er",
        "sissis_er",
    ],
    required=True,
)
parser.add_argument(
    "--num_samples",
    "-s",
    type=int,
    metavar="NUM_SAMPLES",
    help="Number of samples to train from.",
    default=10000,
)
parser.add_argument(
    "--path_to_data",
    "-pd",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    default="./",
)
parser.add_argument(
    "--path_to_model",
    "-pm",
    type=str,
    metavar="PATH",
    help="Path to the model directory.",
    default="./",
)
parser.add_argument(
    "--path_to_summary",
    "-ps",
    type=str,
    metavar="PATH",
    help="Path to the summary directory.",
    default="./",
)

parser.add_argument(
    "--verbose",
    "-v",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)
parser.add_argument(
    "--test", "-t", type=int, choices=[0, 1], metavar="TEST", help="Test.", default=0
)

args = parser.parse_args()


config = get_config(args)

if args.test == 1:
    config.config["metrics"]["name"] = [
        "AttentionMetrics",
        "TrueLTPMetrics",
        "GNNLTPMetrics",
        "MLELTPMetrics",
        "TrueStarLTPMetrics",
        "GNNStarLTPMetrics",
        "UniformStarLTPMetrics",
        "StatisticsMetrics",
    ]
    config.config["training"].num_samples = 100
    config.config["training"].step_per_epoch = 100
    config.config["training"].num_epochs = 1

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

experiment = dl.Experiment(config.config, verbose=args.verbose)

experiment.run()

h5file = h5py.File(
    os.path.join(args.path_to_summary, "{0}.h5".format(experiment.name)), "w"
)
summarize_ltp(h5file, experiment)
summarize_error(h5file, experiment)
if args.test == 0:
    summarize_ssmf(h5file, experiment)


plt.plot()
