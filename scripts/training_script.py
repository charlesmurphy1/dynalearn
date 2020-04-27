import h5py
import numpy as np
import dynalearn as dl
import argparse
import os

from dynalearn.experiments import LTPMetrics
from os.path import exists, join
from scipy.spatial.distance import jensenshannon


def summarize_ltp(h5file, experiment):
    true = experiment.post_metrics["TrueLTPMetrics"]
    gnn = experiment.post_metrics["GNNLTPMetrics"]
    mle = experiment.post_metrics["MLELTPMetrics"]
    transitions = {(0, 1): "S-I", (1, 0): "I-S"}

    for t in transitions:
        name = "ltp/true/{0}".format(transitions[t])
        summaries = true.summaries
        x, y, el, eh = LTPMetrics.aggregate(
            true.data["val_ltp"],
            true.data["summaries"],
            in_state=t[0],
            out_state=t[1],
            axis=1,
            reduce="mean",
            err_reduce="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)

        name = "ltp/gnn/{0}".format(transitions[t])
        x, y, el, eh = LTPMetrics.aggregate(
            gnn.data["val_ltp"],
            gnn.data["summaries"],
            in_state=t[0],
            out_state=t[1],
            axis=1,
            reduce="mean",
            err_reduce="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)

        name = "ltp/mle/{0}".format(transitions[t])
        x, y, el, eh = LTPMetrics.aggregate(
            mle.data["ltp"],
            mle.data["summaries"],
            in_state=t[0],
            out_state=t[1],
            axis=1,
            reduce="mean",
            err_reduce="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)


def summarize_starltp(h5file, experiment):
    true = experiment.post_metrics["TrueStarLTPMetrics"]
    gnn = experiment.post_metrics["GNNStarLTPMetrics"]
    transitions = {(0, 1): "S-I", (1, 0): "I-S"}

    for t in transitions:
        name = "starltp/true/{0}".format(transitions[t])
        summaries = true.summaries
        x, y, el, eh = LTPMetrics.aggregate(
            true.data["ltp"],
            true.data["summaries"],
            in_state=t[0],
            out_state=t[1],
            axis=1,
            reduce="mean",
            err_reduce="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)

        name = "starltp/gnn/{0}".format(transitions[t])
        x, y, el, eh = LTPMetrics.aggregate(
            gnn.data["ltp"],
            gnn.data["summaries"],
            in_state=t[0],
            out_state=t[1],
            axis=1,
            reduce="mean",
            err_reduce="percentile",
        )
        data = np.array([x, y, el, eh]).T
        h5file.create_dataset(name, data=data)


def summarize_error(h5file, experiment):
    true = experiment.post_metrics["TrueStarLTPMetrics"]
    gnn = experiment.post_metrics["GNNStarLTPMetrics"]
    uni = experiment.post_metrics["UniformStarLTPMetrics"]
    _true = experiment.post_metrics["TrueLTPMetrics"]
    mle = experiment.post_metrics["MLELTPMetrics"]

    name = "jsd/true-gnn"
    summaries = true.summaries
    jsd = LTPMetrics.compare(
        true.data["ltp"], gnn.data["ltp"], true.data["summaries"], func=jensenshannon
    )
    x, y, el, eh = LTPMetrics.aggregate(
        jsd, true.data["summaries"], err_reduce="percentile"
    )
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)

    name = "jsd/true-uni"
    summaries = true.summaries
    jsd = LTPMetrics.compare(
        true.data["ltp"], uni.data["ltp"], true.data["summaries"], func=jensenshannon
    )
    x, y, el, eh = LTPMetrics.aggregate(
        jsd, true.data["summaries"], err_reduce="percentile"
    )
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)

    name = "jsd/true-mle"
    summaries = _true.summaries
    jsd = LTPMetrics.compare(
        _true.data["ltp"], mle.data["ltp"], _true.data["summaries"], func=jensenshannon
    )
    x, y, el, eh = LTPMetrics.aggregate(
        jsd, _true.data["summaries"], err_reduce="percentile"
    )
    data = np.array([x, y, el, eh]).T
    h5file.create_dataset(name, data=data)


def summarize_stats(h5file, experiment):
    metric = experiment.post_metrics["StatisticsMetrics"]
    if "all_entropy" in metric.data:
        h5file.create_dataset("stats/entropy/all", data=metric.data["all_entropy"])
        h5file.create_dataset("stats/ess/all", data=metric.data["all_ess"])
    if "train_entropy" in metric.data:
        h5file.create_dataset("stats/entropy/train", data=metric.data["train_entropy"])
        h5file.create_dataset("stats/ess/train", data=metric.data["train_ess"])
    if "val_entropy" in metric.data:
        h5file.create_dataset("stats/entropy/val", data=metric.data["val_entropy"])
        h5file.create_dataset("stats/ess/val", data=metric.data["val_ess"])
    if "test_entropy" in metric.data:
        h5file.create_dataset("stats/entropy/test", data=metric.data["test_entropy"])
        h5file.create_dataset("stats/ess/test", data=metric.data["test_ess"])


def summarize_ss(h5file, experiment):
    metrics = [
        experiment.post_metrics["TruePESSMetrics"],
        experiment.post_metrics["GNNPESSMetrics"],
    ]
    label = ["true", "gnn"]
    for l, m in zip(label, metrics):
        if "parameters" in m.data:
            h5file.create_dataset(f"ss/{l}/parameters", data=m.data["parameters"])
        if "absorbing_stationary_state" in m.data:
            h5file.create_dataset(
                f"ss/{l}/absorbing/mean",
                data=m.data["absorbing_stationary_state"][:, 0],
            )
            h5file.create_dataset(
                f"ss/{l}/absorbing/std", data=m.data["absorbing_stationary_state"][:, 1]
            )
        if "epidemic_stationary_state" in m.data:
            h5file.create_dataset(
                f"ss/{l}/epidemic/mean", data=m.data["epidemic_stationary_state"][:, 0]
            )
            h5file.create_dataset(
                f"ss/{l}/epidemic/std", data=m.data["epidemic_stationary_state"][:, 1]
            )


def summarize_mf(h5file, experiment):
    metrics = [
        experiment.post_metrics["TruePEMFMetrics"],
        experiment.post_metrics["GNNPEMFMetrics"],
    ]
    label = ["true", "gnn"]
    for l, m in zip(label, metrics):
        if "parameters" in m.data:
            h5file.create_dataset(f"mf/{l}/parameters", data=m.data["parameters"])
        if "absorbing_stationary_state" in m.data:
            h5file.create_dataset(
                f"mf/{l}/absorbing", data=m.data["absorbing_stationary_state"]
            )
        if "epidemic_stationary_state" in m.data:
            h5file.create_dataset(
                f"mf/{l}/epidemic", data=m.data["epidemic_stationary_state"]
            )


def get_config(args):
    fast = bool(args.run_fast)
    if args.config == "test":
        return dl.ExperimentConfig.test()
    elif args.config == "sis-er":
        return dl.ExperimentConfig.sis_er(args.name, args.path, args.path_to_best, fast)
    elif args.config == "sis-ba":
        return dl.ExperimentConfig.sis_ba(args.name, args.path, args.path_to_best, fast)
    elif args.config == "plancksis-er":
        return dl.ExperimentConfig.plancksis_er(
            args.name, args.path, args.path_to_best, fast
        )
    elif args.config == "plancksis-ba":
        return dl.ExperimentConfig.plancksis_ba(
            args.name, args.path, args.path_to_best, fast
        )
    elif args.config == "sissis-er":
        return dl.ExperimentConfig.sissis_er(
            args.name, args.path, args.path_to_best, fast
        )
    elif args.config == "sissis-ba":
        return dl.ExperimentConfig.sissis_ba(
            args.name, args.path, args.path_to_best, fast
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
# experiment.load()
# experiment.generate_data()
# experiment.compute_metrics()
# experiment.save()

h5file = h5py.File(
    os.path.join(args.path_to_summary, "{0}.h5".format(experiment.name)), "w"
)


summarize_ltp(h5file, experiment)
summarize_starltp(h5file, experiment)
summarize_error(h5file, experiment)
summarize_stats(h5file, experiment)
if not args.run_fast:
    summarize_ss(h5file, experiment)
    summarize_mf(h5file, experiment)
