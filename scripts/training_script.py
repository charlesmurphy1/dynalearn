import dynalearn as dl
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    "-p",
    type=str,
    metavar="PATH",
    help="Path to config pickle file.",
    required=True,
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

config = dl.ExperimentConfig.config_from_file(args.config_path)
experiment = dl.Experiment(config.config, verbose=args.verbose)
experiment.run()
