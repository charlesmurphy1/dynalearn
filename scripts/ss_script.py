import h5py
import numpy as np
import dynalearn as dl
import argparse
import os
from scipy.spatial.distance import jensenshannon
import tensorflow as tf


def summarize_mf(h5file, experiment):
    true_mf = experiment.metrics["TruePEMFMetrics"]
    h5file.create_dataset("mf-parameters", data=true_mf.data["parameters"])
    h5file.create_dataset("mf-true/fixed_points", data=true_mf.data["fixed_points"])
    h5file.create_dataset("mf-true/thresholds", data=true_mf.data["thresholds"])
    gnn_mf = experiment.metrics["GNNPEMFMetrics"]
    h5file.create_dataset("mf-gnn/fixed_points", data=gnn_mf.data["fixed_points"])
    h5file.create_dataset("mf-gnn/thresholds", data=gnn_mf.data["thresholds"])

def summarize_ss(h5file, experiment):

    true_ss = experiment.metrics["TruePESSMetrics"]
    h5file.create_dataset("ss-parameters", data=true_ss.data["parameters"])
    h5file.create_dataset("ss-true/avg", data=true_ss.data["avg"])
    h5file.create_dataset("ss-true/std", data=true_ss.data["std"])

    gnn_ss = experiment.metrics["GNNPESSMetrics"]
    h5file.create_dataset("ss-gnn/avg", data=true_ss.data["avg"])
    h5file.create_dataset("ss-gnn/std", data=gnn_ss.data["std"])


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
        "sissis_ba",
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

config.config["metrics"]["name"] = [
    "TruePEMFMetrics",
    "GNNPEMFMetrics",
    "TruePESSMetrics",
    "GNNPESSMetrics",
]

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

experiment = dl.Experiment(config.config, verbose=args.verbose)

experiment.load()
experiment.compute_metrics()

h5file = h5py.File(
    os.path.join(args.path_to_summary, "{0}.h5".format(experiment.name)), "a"
)
summarize_ss(h5file, experiment)
