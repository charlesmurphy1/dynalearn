import h5py
import numpy as np
import dynalearn as dl
import argparse
import os

from os.path import exists, join


def get_config(args):
    fast = bool(args.run_fast)
    if args.config == "test":
        return dl.ExperimentConfig.test()
    elif args.config == "sis-er":
        return dl.ExperimentConfig.sis_er(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
        )
    elif args.config == "sis-ba":
        return dl.ExperimentConfig.sis_ba(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
        )
    elif args.config == "plancksis-er":
        return dl.ExperimentConfig.plancksis_er(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
        )
    elif args.config == "plancksis-ba":
        return dl.ExperimentConfig.plancksis_ba(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
        )
    elif args.config == "sissis-er":
        return dl.ExperimentConfig.sissis_er(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
        )
    elif args.config == "sissis-ba":
        return dl.ExperimentConfig.sissis_ba(
            args.name, args.path, args.path_to_best, args.path_to_summary, fast
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
    "--num_nodes",
    "-n",
    type=int,
    metavar="NUM_NODES",
    help="Number of nodes to train from.",
    default=1000,
)
parser.add_argument(
    "--resampling_time",
    "-r",
    type=int,
    metavar="RESAMPLING_TIME",
    help="Resampling time to generate the data.",
    default=2,
)
parser.add_argument(
    "--batch_size",
    "-bs",
    type=int,
    metavar="BATCH_SIZE",
    help="Size of the batches during training.",
    default=2,
)
parser.add_argument(
    "--with_truth",
    "-wt",
    type=int,
    choices=[0, 1],
    metavar="BOOL",
    help="Using ground truth for training.",
    default=0,
)
parser.add_argument(
    "--run_fast",
    "-rf",
    type=int,
    choices=[0, 1],
    metavar="BOOL",
    help="Running fast analysis with meanfields and stationary states (takes longer).",
    default=0,
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
config.train_details.batch_size = args.batch_size
config.train_details.epochs = 5
config.networks.num_nodes = args.num_nodes
if config.networks.name is "ERNetwork":
    config.networks.p = np.min([1, 4 / int(args.num_nodes - 1)])
config.dataset.resampling_time = int(args.resampling_time)
config.dataset.with_truth = bool(args.with_truth)

experiment = dl.Experiment(config, verbose=args.verbose)
experiment.run()
