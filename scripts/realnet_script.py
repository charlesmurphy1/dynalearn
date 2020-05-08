import h5py
import numpy as np
import dynalearn as dl
import argparse
import os

from os.path import exists, join


def get_config(args):
    if args.config == "test":
        return dl.ExperimentConfig.test()
    elif args.config == "sis":
        return dl.ExperimentConfig.sisrtn_forcast(
            args.name, args.path_to_edgelist, args.path_to_model, args.path_to_data, args.path_to_summary
        )
    elif args.config == "plancksis":
        return dl.ExperimentConfig.plancksisrtn_forcast(
            args.name, args.path_to_edgelist, args.path_to_model, args.path_to_data, args.path_to_summary
        )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    metavar="CONFIG",
    help="Configuration of the experiment.",
    choices=[
        "sis",
        "plancksis",
    ],
    required=True,
)
parser.add_argument(
    "--name",
    type=str,
    metavar="NAME",
    help="Name of the experiment.",
    required=True,
)
parser.add_argument(
    "--path_to_data",
    type=str,
    metavar="PATH",
    help="Path to the directory where to save the experiment.",
    default="./",
)
parser.add_argument(
    "--path_to_edgelist",
    type=str,
    metavar="PATH",
    help="Path to the edgelist.",
    default="./",
)
parser.add_argument(
    "--path_to_model",
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
    "--verbose",
    type=int,
    choices=[0, 1, 2],
    metavar="VERBOSE",
    help="Verbose.",
    default=0,
)

args = parser.parse_args()
config = get_config(args)

experiment = dl.Experiment(config, verbose=args.verbose)
if args.verbose != 0:
    print("---Experiment {0}---".format(experiment.name))
experiment.save_config()
experiment.load_model()

if args.verbose != 0:
    print("\n---Computing metrics---")
experiment.compute_metrics()
experiment.save_metrics()

if args.verbose != 0:
    print("\n---Summarizing---")
experiment.compute_summaries()
experiment.save_summaries()
