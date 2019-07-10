"""

utilities.py

Created by Charles Murphy on 19-06-30.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines a variety of useful functions for bm use and training.
"""
import dynalearn as dl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K


color_dark = {
    "blue": "#1f77b4",
    "orange": "#f19143",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252",
}

color_pale = {
    "blue": "#7bafd3",
    "orange": "#f7be90",
    "purple": "#c3b4d6",
    "red": "#e78580",
    "grey": "#999999",
}

m_list = ["o", "s", "v", "x"]
l_list = ["solid", "dashed", "dotted", "dashdot"]
cd_list = [
    color_dark["blue"],
    color_dark["orange"],
    color_dark["purple"],
    color_dark["red"],
]
cp_list = [
    color_pale["blue"],
    color_pale["orange"],
    color_pale["purple"],
    color_pale["red"],
]

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def train_model(params):
    print("-------------------")
    print("Building experiment")
    print("-------------------")
    experiment = get_experiment(params)
    experiment.model.model.summary()
    schedule = get_schedule(params["training"]["schedule"])
    metrics = [
        dl.utilities.metrics.model_entropy,
        dl.utilities.metrics.approx_kl_divergence,
    ]
    callbacks = [
        ks.callbacks.LearningRateScheduler(schedule, verbose=1),
        ks.callbacks.ModelCheckpoint(
            os.path.join(
                params["path"], params["name"] + "_" + params["path_to_best"] + ".h5"
            ),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            period=1,
            verbose=1,
        ),
    ]
    N = experiment.model.num_nodes
    if N < 500:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print("----------------")
    print("Building dataset")
    print("----------------")
    experiment.generate_data(
        params["generator"]["params"]["num_graphs"],
        params["generator"]["params"]["num_sample"],
        params["generator"]["params"]["T"],
        val_fraction=params["generator"]["params"]["val_fraction"],
        val_bias=params["sampler"]["params"]["validation_bias"],
    )
    counts_metrics = dl.utilities.CountMetrics()
    counts_metrics.compute(experiment)
    counts_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))

    if experiment.val_generator is not None:
        print("Overlap train-val: " + str(counts_metrics.overlap("train", "val")))
    if experiment.test_generator is not None:
        print("Overlap train-test: " + str(counts_metrics.overlap("train", "test")))
        print("Overlap val-test: " + str(counts_metrics.overlap("val", "test")))

    print("------------")
    print("Pre-Training")
    print("------------")
    experiment.train_model(
        params["training"]["pretrain_epochs"],
        params["training"]["steps_per_epoch"],
        validation_steps=params["training"]["steps_per_epoch"],
        learning_rate=params["training"]["learning_rate"],
    )
    print("--------")
    print("Training")
    print("--------")
    experiment.train_model(
        params["training"]["epochs"],
        params["training"]["steps_per_epoch"],
        validation_steps=params["training"]["steps_per_epoch"],
        metrics=metrics,
        callbacks=callbacks,
        learning_rate=params["training"]["learning_rate"],
    )

    print("-----------")
    print("Saving data")
    print("-----------")
    experiment.save_weights(
        os.path.join(params["path"], params["name"] + "_weights.h5")
    )
    experiment.save_data(os.path.join(params["path"], params["name"] + ".h5"))
    experiment.save_history(os.path.join(params["path"], params["name"] + ".h5"))
    return experiement


def analyze_experiment(params, experiement):
    print("-----------------")
    print("Computing metrics")
    print("-----------------")
    fig_path = os.path.join(params["path"], "figures")
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    # LTP metrics
    gt_metrics = dl.utilities.DynamicsLTPMetrics()
    gt_metrics.compute(experiment)
    gt_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))

    m_metrics = dl.utilities.ModelLTPMetrics()
    m_metrics.compute(experiment)
    m_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))
    make_ltp_metrics_fig(
        experiment, params, gt_metrics, m_metrics, counts_metrics, "model_ltp.png"
    )

    e_metrics = dl.utilities.EstimatorLTPMetrics()
    e_metrics.compute(experiment)
    e_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))
    make_ltp_metrics_fig(
        experiment, params, gt_metrics, e_metrics, counts_metrics, "estimator_ltp.png"
    )

    # Generalization LTP
    degree_class = np.unique(np.linspace(0, 10, 10).astype("int"))
    experiment.model.num_nodes = np.max(degree_class) + 1

    gt_metrics = dl.utilities.DynamicsLTPGenMetrics(degree_class)
    gt_metrics.compute(experiment)
    gt_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))

    m_metrics = dl.utilities.ModelLTPGenMetrics(degree_class)
    m_metrics.compute(experiment)
    m_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))
    make_gltp_metrics_fig(
        experiment, params, gt_metrics, m_metrics, counts_metrics, "model_gltp.png"
    )
    experiment.model.num_nodes = N

    # Generalization JSD
    degree_class = np.unique(np.linspace(0, 10, 10).astype("int"))
    experiment.model.num_nodes = np.max(degree_class) + 1

    m_metrics = dl.utilities.ModelJSDGenMetrics(degree_class)
    m_metrics.compute(experiment)
    m_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))

    b_metrics = dl.utilities.BaseJSDGenMetrics(degree_class)
    b_metrics.compute(experiment)
    b_metrics.save(os.path.join(params["path"], params["name"] + ".h5"))
    make_gdiv_metrics_fig(
        experiment, params, m_metrics, b_metrics, counts_metrics, "model_div.png"
    )
    experiment.model.num_nodes = N

    # Attention coefficients
    att = dl.utilities.AttentionMetrics()
    att.compute(experiment)
    att.save(os.path.join(params["path"], params["name"] + ".h5"))
    make_attn_metrics_fig(experiment, params, att, "model_attn.png")


def make_ltp_metrics_fig(experiment, params, gt_metrics, metrics, counts, filename):

    state_label = experiment.dynamics_model.state_label
    d = len(state_label)
    datasets = ["train"]
    if experiment.val_generator is not None:
        datasets.append("val")
    if experiment.test_generator is not None:
        datasets.append("test")
    for ds in datasets:
        fig = plt.figure(figsize=(6, 6), frameon=False)
        gs = GridSpec(10, 1)
        gs.update(wspace=0.1, hspace=0.05)
        ax_ltp = fig.add_subplot(gs[3:, :])
        ax_dist = fig.add_subplot(gs[1:3, :])
        ax_legend = fig.add_subplot(gs[:1, :])
        ax_legend.axis("off")
        ax_legend.set_zorder(1)
        ax_dist.set_yscale("log")
        ax_dist.spines["right"].set_visible(False)
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["bottom"].set_visible(False)
        ax_dist.set_xticks([])

        for i, in_s in enumerate(state_label.values()):
            d_color = cd_list[i]
            p_color = cp_list[i]
            f_color = color_dark["grey"]
            counts.display(
                in_s, 1, ds, bar_width=1.0 / (d + 1), ax=ax_dist, color=d_color
            )
            for j, out_s in enumerate(state_label.values()):
                mk = m_list[j]
                ls = l_list[j]
                gt_metrics.display(
                    in_s,
                    out_s,
                    1,
                    ds,
                    ax=ax_ltp,
                    fill=None,
                    color=p_color,
                    marker="None",
                    linestyle=ls,
                )
                metrics.display(
                    in_s,
                    out_s,
                    1,
                    ds,
                    ax=ax_ltp,
                    fill=f_color,
                    color=d_color,
                    marker=mk,
                    linestyle="None",
                )
        handles = []
        for i, in_s in enumerate(state_label.keys()):
            d_color = cd_list[i]
            handles.append(
                Line2D(
                    [-1],
                    [-1],
                    linestyle="None",
                    marker="s",
                    markersize=10,
                    color=d_color,
                    label=r"$s = {0}$".format(in_s),
                )
            )

        for j, out_s in enumerate(state_label.keys()):
            d_color = color_dark["grey"]
            p_color = color_pale["grey"]
            mk = m_list[j]
            ls = l_list[j]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=ls,
                    marker=mk,
                    markeredgewidth=1,
                    markeredgecolor="k",
                    markerfacecolor=d_color,
                    linewidth=3,
                    color=p_color,
                    label=r"$s' = {0}$".format(out_s),
                )
            )

        ax_ltp.set_xlabel(r"Infected degree $\ell$", fontsize=14)
        ax_ltp.set_ylabel(r"$\mathrm{Pr}[s\to s'|\,\ell]$", fontsize=14)
        ax_dist.set_ylabel(r"$\mathrm{Pr}[\ell|\,s]$", fontsize=14)
        ax_dist.set_xlim(ax_ltp.get_xlim())
        ax_legend.legend(
            handles=handles,
            loc="best",
            fancybox=True,
            fontsize=10,
            framealpha=1,
            ncol=2,
        )
        # plt.tight_layout(0.1)
        if filename is not None:
            fig.savefig(os.path.join(params["path"], "figures", ds + "_" + filename))
        else:
            plt.show()


def make_gltp_metrics_fig(experiment, params, gt_metrics, metrics, counts, filename):

    state_label = experiment.dynamics_model.state_label
    d = len(state_label)
    fig = plt.figure(figsize=(6, 6), frameon=False)
    gs = GridSpec(10, 1)
    gs.update(wspace=0.1, hspace=0.05)
    ax_ltp = fig.add_subplot(gs[3:, :])
    ax_dist = fig.add_subplot(gs[1:3, :])
    ax_legend = fig.add_subplot(gs[:1, :])
    ax_legend.axis("off")
    ax_legend.set_zorder(1)
    ax_dist.set_yscale("log")
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["bottom"].set_visible(False)
    ax_dist.set_xticks([])

    for i, in_s in enumerate(state_label.values()):
        d_color = cd_list[i]
        p_color = cp_list[i]
        f_color = color_dark["grey"]
        counts.display(
            in_s, 1, "train", bar_width=1.0 / (d + 1), ax=ax_dist, color=d_color
        )
        for j, out_s in enumerate(state_label.values()):
            mk = m_list[j]
            ls = l_list[j]
            gt_metrics.display(
                in_s,
                out_s,
                1,
                ax=ax_ltp,
                fill=None,
                color=p_color,
                marker="None",
                linestyle=ls,
            )
            metrics.display(
                in_s,
                out_s,
                1,
                ax=ax_ltp,
                fill=f_color,
                color=d_color,
                marker=mk,
                linestyle="None",
            )
    handles = []
    for i, in_s in enumerate(state_label.keys()):
        d_color = cd_list[i]
        handles.append(
            Line2D(
                [-1],
                [-1],
                linestyle="None",
                marker="s",
                markersize=10,
                color=d_color,
                label=r"$s = {0}$".format(in_s),
            )
        )

    for j, out_s in enumerate(state_label.keys()):
        d_color = color_dark["grey"]
        p_color = color_pale["grey"]
        mk = m_list[j]
        ls = l_list[j]
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle=ls,
                marker=mk,
                markeredgewidth=1,
                markeredgecolor="k",
                markerfacecolor=d_color,
                linewidth=3,
                color=p_color,
                label=r"$s' = {0}$".format(out_s),
            )
        )

    ax_ltp.set_xlabel(r"Infected degree $\ell$", fontsize=14)
    ax_ltp.set_ylabel(r"$\mathrm{Pr}[s\to s'|\,\ell]$", fontsize=14)
    ax_dist.set_ylabel(r"$\mathrm{Pr}[\ell|\,s]$", fontsize=14)
    ax_dist.set_xlim(ax_ltp.get_xlim())
    ax_legend.legend(
        handles=handles, loc="best", fancybox=True, fontsize=10, framealpha=1, ncol=2
    )
    # plt.tight_layout(0.1)
    if filename is not None:
        fig.savefig(os.path.join(params["path"], "figures", filename))
    else:
        plt.show()


def make_gdiv_metrics_fig(experiment, params, metrics, base_metrics, counts, filename):

    state_label = experiment.dynamics_model.state_label
    d = len(state_label)
    fig = plt.figure(figsize=(6, 6), frameon=False)
    gs = GridSpec(10, 1)
    gs.update(wspace=0.1, hspace=0.05)
    ax_div = fig.add_subplot(gs[3:, :])
    ax_dist = fig.add_subplot(gs[1:3, :])
    ax_legend = fig.add_subplot(gs[:1, :])
    ax_legend.axis("off")
    ax_legend.set_zorder(1)
    ax_dist.set_yscale("log")
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["bottom"].set_visible(False)
    ax_dist.set_xticks([])

    for i, in_s in enumerate(state_label.values()):
        d_color = cd_list[i]
        p_color = cp_list[i]
        counts.display(
            in_s, "all", "train", bar_width=1.0 / (d + 1), ax=ax_dist, color=d_color
        )
        metrics.display(in_s, ax=ax_div, color=p_color, linestyle="-")
        base_metrics.display(in_s, ax=ax_div, color=d_color, linestyle="--")
    handles = []
    for i, in_s in enumerate(state_label.keys()):
        d_color = cd_list[i]
        handles.append(
            Line2D(
                [-1],
                [-1],
                marker="s",
                linestyle="None",
                markersize=10,
                color=d_color,
                label=r"$s = {0}$".format(in_s),
            )
        )

    handles.append(
        Line2D(
            [-1],
            [-1],
            marker="None",
            linestyle="-",
            markersize=10,
            color=color_dark["grey"],
            label=r"Truth vs Learned",
        )
    )
    handles.append(
        Line2D(
            [-1],
            [-1],
            marker="None",
            linestyle="--",
            markersize=10,
            color=color_dark["grey"],
            label=r"Truth vs Baseline",
        )
    )

    ax_div.set_xlabel(r"Degree class", fontsize=14)
    ax_div.set_ylabel(r"$JSD$", fontsize=14)
    ax_dist.set_ylabel(r"$\mathrm{Pr}[\ell|\,s]$", fontsize=14)
    ax_dist.set_xlim(ax_div.get_xlim())
    ax_legend.legend(
        handles=handles, loc="best", fancybox=True, fontsize=10, framealpha=1, ncol=2
    )
    if filename is not None:
        fig.savefig(os.path.join(params["path"], "figures", filename))
    else:
        plt.show()


def make_attn_metrics_fig(experiment, params, metrics, filename):
    state_label = experiment.dynamics_model.state_label
    d = len(state_label)

    for l in range(metrics.num_layers):
        fig, axes = plt.subplots(d, d, sharex=True, sharey=True)
        for j, in_s in enumerate(state_label.keys()):
            in_l = state_label[in_s]
            for i, out_s in enumerate(state_label.keys()):
                out_l = state_label[out_s]
                metrics.display(
                    l,
                    in_l,
                    out_l,
                    ax=axes[i, j],
                    kde=False,
                    rug=True,
                    bin=False,
                    box=False,
                )
                if j == 0:
                    axes[i, j].set_ylabel(rf"$s' = {out_s}$", fontsize=14)
                if i == d - 1:
                    axes[i, j].set_xlabel(rf"$s = {in_s}$", fontsize=14)
        fig.suptitle(r"$\alpha(s, s')$", fontsize=16)
        if filename is not None:
            fig.savefig(os.path.join(params["path"], "figures", filename))
        else:
            plt.show()
    return


def get_noisy_crossentropy(noise=0):
    def noisy_crossentropy(y_true, y_pred):
        num_classes = tf.cast(K.shape(y_true)[1], tf.float32)
        y_true = y_true * (1 - noise) + (1 - y_true) * noise / num_classes

        return K.categorical_crossentropy(y_true, y_pred)

    return noisy_crossentropy


def get_graph(params):
    if "CycleGraph" == params["graph"]["name"]:
        return dl.graphs.CycleGraph(params["graph"]["params"]["N"])
    elif "CompleteGraph" == params["graph"]["name"]:
        return dl.graphs.CompleteGraph(params["graph"]["params"]["N"])
    elif "StarGraph" == params["graph"]["name"]:
        return dl.graphs.StarGraph(params["graph"]["params"]["N"])
    elif "EmptyGraph" == params["graph"]["name"]:
        return dl.graphs.EmptyGraph(params["graph"]["params"]["N"])
    elif "RegularGraph" == params["graph"]["name"]:
        return dl.graphs.RegularGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["degree"]
        )
    elif "ERGraph" == params["graph"]["name"]:
        return dl.graphs.ERGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["p"]
        )
    elif "BAGraph" == params["graph"]["name"]:
        return dl.graphs.BAGraph(
            params["graph"]["params"]["N"], params["graph"]["params"]["M"]
        )
    else:
        raise ValueError("wrong string name for graph.")


def get_dynamics(params):
    if "SISDynamics" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SISDynamics(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SIRDynamics" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SIRDynamics(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    else:
        raise ValueError("wrong string name for dynamics.")


def get_model(params, dynamics):
    if "LocalStatePredictor" == params["model"]["name"]:
        return dl.models.LocalStatePredictor(
            params["graph"]["params"]["N"],
            len(dynamics.state_label),
            params["model"]["params"]["n_hidden"],
            params["model"]["params"]["n_heads"],
            weight_decay=params["model"]["params"]["weight_decay"],
            dropout=params["model"]["params"]["dropout"],
            seed=params["tf_seed"],
        )
    else:
        raise ValueError("wrong string name for model.")


def get_sampler(params, dynamics):
    if params["sampler"]["name"] == "SequentialSampler":
        return dl.generators.SequentialSampler()
    elif params["sampler"]["name"] == "RandomSampler":
        return dl.generators.RandomSampler(
            replace=params["sampler"]["params"]["replace"]
        )
    elif params["sampler"]["name"] == "DegreeBiasedSampler":
        return dl.generators.DegreeBiasedSampler(
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            # validation_bias=params["sampler"]["params"]["validation_bias"],
            replace=params["sampler"]["params"]["replace"],
        )

    elif params["sampler"]["name"] == "StateBiasedSampler":
        return dl.generators.StateBiasedSampler(
            dynamics,
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            replace=params["sampler"]["params"]["replace"],
        )
    else:
        raise ValueError("wrong string name for sampler.")


def get_generator(graph_model, dynamics_model, sampler, params):
    if "with_truth" in params["training"]:
        with_truth = params["training"]["with_truth"]
    else:
        with_truth = False

    if "MarkovBinaryDynamicsGenerator" == params["generator"]["name"]:
        return dl.generators.MarkovBinaryDynamicsGenerator(
            graph_model, dynamics_model, shuffle=True, with_truth=with_truth
        )
    elif "DynamicsGenerator" == params["generator"]["name"]:
        return dl.generators.DynamicsGenerator(
            graph_model,
            dynamics_model,
            sampler,
            batch_size=params["generator"]["params"]["batch_size"],
            with_truth=False,
            verbose=1,
        )
    else:
        raise ValueError("wrong string name for generator.")


def get_experiment(params):
    # Define seeds
    np.random.seed(params["np_seed"])
    tf.set_random_seed(params["tf_seed"])

    # Define graph
    graph = get_graph(params)

    # Define dynamics
    dynamics = get_dynamics(params)

    # Define data generator
    sampler = get_sampler(params, dynamics)
    generator = get_generator(graph, dynamics, sampler, params)

    # Define model
    model = get_model(params, dynamics)
    optimizer = ks.optimizers.get(params["training"]["optimizer"])
    if params["training"]["loss"] == "noisy_crossentropy":
        loss = get_noisy_crossentropy(noise=params["training"]["target_noise"])
    else:
        loss = ks.losses.get(params["training"]["loss"])

    # Define experiment
    experiment = dl.Experiment(
        params["name"],
        model,
        generator,
        loss=loss,
        optimizer=optimizer,
        numpy_seed=params["np_seed"],
        tensorflow_seed=params["tf_seed"],
    )

    return experiment


def get_schedule(schedule):
    def lr_schedule(epoch, lr):
        if (epoch + 1) % schedule["epoch"] == 0:
            lr /= schedule["factor"]
        return lr

    return lr_schedule


def int_to_base(i, base, size=None):

    if i > 0:
        if size is None or size < np.floor(np.log(i) / np.log(base)) + 1:
            size = np.floor(np.log(i) / np.log(base)) + 1
    else:
        if size is None:
            size = 1

    return (i // base ** np.arange(size)) % base


def increment_int_from_base(x, base):
    val = x * 1
    for i in range(len(x)):
        val[i] += 1
        if val[i] > base - 1:
            val[i] = 0
        else:
            break

    return val


def base_to_int(x, base):
    return int(np.sum(x * base ** np.arange(len(x))))


def setup_counts(state_label, N, raw_counts):
    inv_state_label = {state_label[i]: i for i in state_label}
    counts = {s: np.zeros(N) for s in state_label}
    i_index = 1
    for rc in raw_counts:
        s = inv_state_label[int(rc[0])]
        counts[s][int(rc[i_index + 1])] += rc[-1]
    return counts
