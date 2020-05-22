import h5py
import numpy as np
import dynalearn as dl
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
        return dl.ExperimentConfig.test(
            args.path, args.path_to_best, args.path_to_summary
        )
    else:
        dynamics, network = args.config.split("-")
        return dl.ExperimentConfig.base(
            args.name,
            dynamics,
            network,
            args.path,
            args.path_to_best,
            args.path_to_summary,
            args.mode,
            args.seed,
        )


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
    default=2,
)
parser.add_argument(
    "--with_truth",
    type=int,
    choices=[0, 1],
    metavar="BOOL",
    help="Using ground truth for training.",
    default=0,
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["fast", "complete"],
    metavar="MODE",
    help="Experiment mode.",
    default="fast",
)
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
config.train_details.num_samples = int(args.num_samples)
config.train_details.batch_size = args.batch_size
config.networks.num_nodes = args.num_nodes
if config.networks.name is "ERNetwork":
    config.networks.p = np.min([1, 4 / int(args.num_nodes - 1)])
config.dataset.resampling_time = int(args.resampling_time)
config.dataset.with_truth = bool(args.with_truth)

experiment = dl.Experiment(config, verbose=args.verbose)
experiment.run()

#if args.verbose != 0:
#    print("---Experiment {0}---".format(args.name))
#experiment.load_model()
#experiment.save_config()

#if args.verbose != 0:
#    print("\n---Generating data---")
#experiment.generate_data()
#experiment.save_data()

#if args.verbose != 0:
#    print("\n---Computing metrics---")
#experiment.compute_metrics()
#experiment.save_metrics()

#if args.verbose != 0:
#    print("\n---Summarizing---")
#experiment.compute_summaries()
#experiment.save_summaries()
