import h5py
import numpy as np
import dynalearn as dl
import argparse
import os

from os.path import exists, join


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
    if args.config == "test":
        return dl.ExperimentConfig.test(
            args.path, args.path_to_best, args.path_to_summary,
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
            "complete",
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    "-c",
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
    "--name",
    "-na",
    type=str,
    metavar="NAME",
    help="Name of the experiment.",
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
    "--path",
    "-pa",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    default="./",
)
parser.add_argument(
    "--path_to_best",
    "-pb",
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
    help="Path to the summaries directory.",
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

args = parser.parse_args()
config = get_config(args)

config.train_details.num_samples = int(args.num_samples)
config.metrics.num_samples = 1

experiment = dl.Experiment(config, verbose=args.verbose)


experiment.load()
if args.verbose != 0:
    print(f"---Experiment {experiment.name}---")
if args.verbose != 0:
    print("\n---Generating data---")
experiment.generate_data()
experiment.save_data()
if args.verbose != 0:
    print("\n---Computing metrics---")
experiment.compute_metrics()
experiment.save_metrics()
if args.verbose != 0:
    print("\n---Summarizing---")
experiment.compute_summaries()
experiment.save_summaries()
