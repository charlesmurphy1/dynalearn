import h5py
import numpy as np
import dynalearn
import argparse
import os
import time

from os.path import exists, join


def mark_path_as_finished(path):
    p = path.split("/")
    new_name = "f_" + p[-1]
    p[-1] = new_name
    p = "/".join(p)
    if os.path.exists(p):
        os.system(f"rm -r {p}")
    os.rename(path, p)
    return p


def unmark_path_as_finished(path):
    p = path.split("/")
    if p[-1][:2] == "f_":
        p[-1] = p[-1][2:]
    else:
        return path
    p = "/".join(p)
    if os.path.exists(p):
        os.system(f"rm -r {p}")
    os.rename(path, p)
    return p


def get_config(args):
    if args.seed == -1:
        args.seed = int(time.time())

    if args.config == "test":
        return dynalearn.config.ExperimentConfig.test(
            args.path, args.path_to_best, args.path_to_summary
        )
    else:
        dynamics, network = args.config.split("-")
        return dynalearn.config.ExperimentConfig.discrete_experiment(
            args.name,
            dynamics,
            network,
            args.path,
            args.path_to_best,
            args.path_to_summary,
            args.seed,
        )


def get_metrics(args):
    metrics = args.metrics
    names = []
    for m in metrics:
        if m == "ltp":
            names.extend(["TrueLTPMetrics", "GNNLTPMetrics", "MLELTPMetrics"])
        elif m == "star-ltp":
            names.extend(
                ["TrueStarLTPMetrics", "GNNStarLTPMetrics", "UniformStarLTPMetrics"]
            )
        elif m == "meanfield":
            names.extend(["TruePEMFMetrics", "GNNPEMFMetrics"])
        elif m == "stationary":
            names.extend(["TruePESSMetrics", "GNNPESSMetrics"])
        elif m == "stats":
            names.extend(["StatisticsMetrics"])
        else:
            raise ValueError(
                f"{m} is invalid, valid entries are [ltp, star-ltp, meanfield, \
                stationary, stats]."
            )
    return names


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration of the experiment.",
    choices=[
        "test",
        "sis-er",
        "sis-ba",
        "plancksis-er",
        "plancksis-ba",
        "sissis-er",
        "sissis-ba",
        "hiddensissis-er",
        "hiddensissis-ba",
        "partiallyhiddensissis-er",
        "partiallyhiddensissis-ba",
    ],
    required=True,
)
parser.add_argument(
    "--name", type=str, metavar="NAME", help="Name of the experiment.", required=True
)
parser.add_argument(
    "--num_samples",
    type=int,
    metavar="NUM_SAMPLES",
    help="Number of samples to train from.",
    default=10000,
)
parser.add_argument(
    "--num_nodes",
    type=int,
    metavar="NUM_NODES",
    help="Number of nodes to train from.",
    default=1000,
)
parser.add_argument(
    "--resampling_time",
    type=int,
    metavar="RESAMPLING_TIME",
    help="Resampling time to generate the data.",
    default=2,
)
parser.add_argument(
    "--batch_size",
    type=int,
    metavar="BATCH_SIZE",
    help="Size of the batches during training.",
    default=1,
)
parser.add_argument(
    "--window_size",
    type=int,
    metavar="WINDOW_SIZE",
    help="Size of the windows during training.",
    default=1,
)
parser.add_argument(
    "--window_step",
    type=int,
    metavar="WINDOW_STEP",
    help="Step between windows during training.",
    default=1,
)
parser.add_argument(
    "--hide_prob",
    type=float,
    metavar="HIDE_PROB",
    help="Probability to hide a state (only for partially hidden sissis).",
    default=0.0,
)
parser.add_argument(
    "--use_groundtruth",
    type=int,
    choices=[0, 1],
    metavar="BOOL",
    help="Using ground truth for training.",
    default=0,
)
parser.add_argument(
    "--tasks", type=str, metavar="TASKS", help="Experiment tasks.", nargs="+"
)
parser.add_argument(
    "--metrics", type=str, metavar="METRICS", help="Metrics to compute.", nargs="+"
)
parser.add_argument("--to_zip", type=str, metavar="ZIP", help="Data to zip.", nargs="+")
parser.add_argument(
    "--path",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    default="./",
)
parser.add_argument(
    "--path_to_best",
    type=str,
    metavar="PATH",
    help="Path to the model directory.",
    default="./",
)
parser.add_argument(
    "--path_to_summary",
    type=str,
    metavar="PATH",
    help="Path to the summaries directory.",
    default="./",
)
parser.add_argument(
    "--seed", type=int, metavar="SEED", help="Seed of the experiment.", default=-1
)
parser.add_argument(
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)

args = parser.parse_args()
config = get_config(args)
config.metrics.names = get_metrics(args)

config.train_details.num_samples = int(args.num_samples)
config.train_details.batch_size = args.batch_size
config.networks.num_nodes = args.num_nodes
if config.networks.name is "ERNetwork":
    config.networks.p = np.min([1, 4 / int(args.num_nodes - 1)])
config.train_details.resampling_time = int(args.resampling_time)
config.model.window_size = args.window_size
config.model.window_step = args.window_step
if "hide_prob" in config.dynamics.__dict__:
    config.dynamics.hide_prob = args.hide_prob
config.dataset.use_groundtruth = bool(args.use_groundtruth)
config.metrics.num_nodes = 2000

print(config)

experiment = dynalearn.experiments.Experiment(config, verbose=args.verbose)
experiment.run(args.tasks)
experiment.zip(args.to_zip)
