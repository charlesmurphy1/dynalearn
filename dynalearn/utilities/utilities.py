"""
utilities.py

Created by Charles Murphy on 19-06-30.
Copyright © 2018 Charles Murphy. All rights reserved.
Quebec, Canada

Defines a variety of useful functions for bm use and training.
"""
import dynalearn as dl
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import os
from scipy.spatial.distance import jensenshannon
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

colormap = "bone"

m_list = ["o", "s", "v", "^"]
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


def train_model(params, experiment):
    experiment.model.model.summary()
    schedule = get_schedule(params["training"]["schedule"])
    metrics = [dl.utilities.metrics.model_entropy]
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
    data_filename = os.path.join(params["path"], params["name"] + ".h5")
    h5file = h5py.File(data_filename)

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
    print("Train dataset entropy: " + str(counts_metrics.entropy("train")))
    if experiment.val_generator is not None:
        print("Val. dataset entropy: " + str(counts_metrics.entropy("val")))
    if experiment.test_generator is not None:
        print("Test dataset entropy: " + str(counts_metrics.entropy("test")))

    if experiment.val_generator is not None:
        print("JSD train-val: " + str(counts_metrics.jensenshannon("train", "val")))
        if experiment.test_generator is not None:
            print(
                "JSD train-test: " + str(counts_metrics.jensenshannon("train", "test"))
            )
            print("JSD val-test: " + str(counts_metrics.jensenshannon("val", "test")))

    experiment.save_data(h5file)

    agg = get_aggregator(params)
    num = params["generator"]["params"]["num_sample"]
    if num > 10000:
        num = 10000
    estimator_metrics = dl.utilities.EstimatorLTPMetrics(aggregator=agg, num_points=num)
    estimator_metrics.compute(experiment)
    print("Train estimated entropy: " + str(estimator_metrics.entropy("train")))
    if experiment.val_generator is not None:
        print("Val. estimated entropy: " + str(estimator_metrics.entropy("val")))
    if experiment.test_generator is not None:
        print("Test estimated entropy: " + str(estimator_metrics.entropy("test")))

    print("------------")
    print("Pre-Training")
    print("------------")
    if params["training"]["pretrain_epochs"] > 0:
        experiment.train_model(
            params["training"]["pretrain_epochs"],
            params["training"]["steps_per_epoch"],
            validation_steps=params["training"]["validation_steps"],
            learning_rate=params["training"]["learning_rate"],
        )
    print("--------")
    print("Training")
    print("--------")
    if params["training"]["epochs"] > 0:
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
    experiment.save_history(h5file)
    return experiment


def analyze_model(params, experiment):
    print("-----------------")
    print("Computing metrics")
    print("-----------------")
    data_filename = os.path.join(params["path"], params["name"] + ".h5")
    h5file = h5py.File(data_filename)
    experiment.compute_metrics()
    experiment.save_metrics(h5file)
    return experiment


def make_figures(params, experiment):
    if not os.path.exists(os.path.join(params["path"], "figures")):
        os.mkdir(os.path.join(params["path"], "figures"))

    if params["dynamics"]["name"] == "SISSIS":
        agg = dl.utilities.InteractingContagionAggregator(0)
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["ModelLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "model_ltp-g0.png",
            agg,
        )
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["EstimatorLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "estimator_ltp-g0.png",
            agg,
        )
        make_gltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPGenMetrics"],
            experiment.metrics["ModelLTPGenMetrics"],
            experiment.metrics["CountMetrics"],
            "gltp-g0.png",
            agg,
        )
        agg.agent = 1
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["ModelLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "model_ltp-g1.png",
            agg,
        )
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["EstimatorLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "estimator_ltp-g1.png",
            agg,
        )
        make_gltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPGenMetrics"],
            experiment.metrics["ModelLTPGenMetrics"],
            experiment.metrics["CountMetrics"],
            "gltp-g1.png",
            agg,
        )
    else:
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["ModelLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "model_ltp.png",
        )
        make_ltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPMetrics"],
            experiment.metrics["EstimatorLTPMetrics"],
            experiment.metrics["CountMetrics"],
            "estimator_ltp.png",
        )

        make_gltp_metrics_fig(
            experiment,
            params,
            experiment.metrics["DynamicsLTPGenMetrics"],
            experiment.metrics["ModelLTPGenMetrics"],
            experiment.metrics["CountMetrics"],
            "gltp.png",
        )

    make_gdiv_metrics_fig(
        experiment,
        params,
        experiment.metrics["ModelLTPGenMetrics"],
        experiment.metrics["DynamicsLTPGenMetrics"],
        experiment.metrics["CountMetrics"],
        "div.png",
    )

    make_attn_metrics_fig(
        experiment, params, experiment.metrics["AttentionMetrics"], "attn.png"
    )

    make_loss_metrics_fig(
        experiment, params, experiment.metrics["LossMetrics"], "loss.png"
    )


def make_ltp_metrics_fig(
    experiment, params, gt_metrics, metrics, counts, filename, aggregator=None
):

    state_label = experiment.dynamics_model.state_label
    d = len(state_label)
    datasets = ["train"]
    if experiment.val_generator is not None:
        datasets.append("val")
    if experiment.test_generator is not None:
        datasets.append("test")
    if aggregator is not None:
        gt_metrics.aggregator = aggregator
        metrics.aggregator = aggregator
        counts.aggregator = aggregator
        counts.operation = "sum"
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

        b_width = 1.0
        if (
            params["dynamics"]["name"] == "SoftThresholdSISDynamics"
            or params["dynamics"]["name"] == "SoftThresholdSIRDynamics"
        ):
            b_width = 1.0 / 100

        for i, in_s in enumerate(state_label.values()):
            d_color = cd_list[i]
            p_color = cp_list[i]
            f_color = color_dark["grey"]
            counts.display(in_s, ds, ax=ax_dist, color=d_color)
            for j, out_s in enumerate(state_label.values()):
                if type(
                    gt_metrics.aggregator
                ).__name__ == "SimpleCoopContagionAggregator" and (
                    (in_s == 0 and out_s == 0)
                    or (in_s == 0 and out_s == 2)
                    or (in_s == 1 and out_s == 3)
                ):
                    continue
                elif type(
                    gt_metrics.aggregator
                ).__name__ == "SimpleCoopContagionAggregator" and (
                    (in_s == 0 and out_s == 0)
                    or (in_s == 0 and out_s == 1)
                    or (in_s == 2 and out_s == 3)
                ):
                    continue
                mk = m_list[j]
                ls = l_list[j]
                metrics.display(
                    in_s,
                    out_s,
                    ds,
                    ax=ax_ltp,
                    fill=f_color,
                    color=d_color,
                    marker=mk,
                    linestyle="None",
                )
                gt_metrics.display(
                    in_s,
                    out_s,
                    ds,
                    ax=ax_ltp,
                    fill=None,
                    color=p_color,
                    marker="None",
                    linestyle=ls,
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

        if (
            params["dynamics"]["name"] == "SISDynamics"
            or params["dynamics"]["name"] == "SIRDynamics"
        ):
            ax_ltp.set_xlabel(r"$\ell$", fontsize=14)
            ax_ltp.set_ylabel(r"$\mathrm{Pr}[s\to s'|\,\ell]$", fontsize=14)
            ax_dist.set_ylabel(r"$\mathrm{Pr}[\ell|\,s]$", fontsize=14)
        elif (
            params["dynamics"]["name"] == "SoftThresholdSISDynamics"
            or params["dynamics"]["name"] == "SoftThresholdSIRDynamics"
        ):
            ax_ltp.set_xlabel(r"$\frac{\ell}{k}$", fontsize=14)
            ax_ltp.set_ylabel(r"$\mathrm{Pr}[s\to s'|\,\frac{\ell}{k}]$", fontsize=14)
            ax_dist.set_ylabel(r"$\mathrm{Pr}[\frac{\ell}{k}|\,s]$", fontsize=14)
        ax_dist.set_xlim(ax_ltp.get_xlim())
        ax_legend.legend(
            handles=handles,
            loc="best",
            fancybox=True,
            fontsize=10,
            framealpha=1,
            ncol=2,
        )
        if filename is not None:
            fig.savefig(os.path.join(params["path"], "figures", ds + "_" + filename))
        else:
            plt.show()


def make_gltp_metrics_fig(
    experiment, params, gt_metrics, metrics, counts, filename, aggregator=None
):

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
    if aggregator is not None:
        gt_metrics.aggregator = aggregator
        metrics.aggregator = aggregator
        counts.aggregator = aggregator

    b_width = 1.0
    if (
        params["dynamics"]["name"] == "SoftThresholdSISDynamics"
        or params["dynamics"]["name"] == "SoftThresholdSIRDynamics"
    ):
        b_width = 1.0 / 100
    for i, in_s in enumerate(state_label.values()):
        d_color = cd_list[i]
        p_color = cp_list[i]
        f_color = color_dark["grey"]

        counts.display(in_s, "train", ax=ax_dist, color=d_color)
        for j, out_s in enumerate(state_label.values()):
            mk = m_list[j]
            ls = l_list[j]
            metrics.display(
                in_s,
                out_s,
                ax=ax_ltp,
                fill=f_color,
                color=d_color,
                marker=mk,
                linestyle="None",
            )
            gt_metrics.display(
                in_s,
                out_s,
                ax=ax_ltp,
                fill=None,
                color=p_color,
                marker="None",
                linestyle=ls,
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

    if (
        params["dynamics"]["name"] == "SoftThresholdSISDynamics"
        or params["dynamics"]["name"] == "SoftThresholdSIRDynamics"
    ):
        ax_ltp.set_xlabel(r"$\frac{\ell}{k}$", fontsize=14)
        ax_ltp.set_ylabel(r"$\mathrm{Pr}[s\to s'|\,\frac{\ell}{k}]$", fontsize=14)
        ax_dist.set_ylabel(r"$\mathrm{Pr}[\frac{\ell}{k}|\,s]$", fontsize=14)
    else:
        ax_ltp.set_xlabel(r"$\ell$", fontsize=14)
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


def compute_jsd(state_label, metrics, gt_metrics):
    x = gt_metrics.data["summaries"]
    jsd_all = np.zeros(x.shape[0])
    base_jsd_all = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        pred = metrics.data["ltp"][i]
        base_pred = np.ones(metrics.data["ltp"][i].shape) / len(state_label)
        jsd_all[i] = jensenshannon(pred, gt_metrics.data["ltp"][i])
        base_jsd_all[i] = jensenshannon(base_pred, gt_metrics.data["ltp"][i])
    k = np.unique(np.sort(np.sum(x[:, 1:], axis=1)))
    degrees = np.sum(x[:, 1:], axis=1)
    model_jsd = np.zeros(k.shape[0])
    up_model_jsd = np.zeros(k.shape[0])
    down_model_jsd = np.zeros(k.shape[0])

    base_jsd = np.zeros(k.shape[0])
    up_base_jsd = np.zeros(k.shape[0])
    down_base_jsd = np.zeros(k.shape[0])
    for i, kk in enumerate(k):
        index = degrees == kk
        m_jsd = jsd_all[index]
        model_jsd[i] = np.mean(m_jsd)
        up_model_jsd[i] = np.percentile(m_jsd, 84)
        down_model_jsd[i] = np.percentile(m_jsd, 16)

        b_jsd = base_jsd_all[index]
        base_jsd[i] = np.mean(b_jsd)
        up_base_jsd[i] = np.percentile(b_jsd, 84)
        down_base_jsd[i] = np.percentile(b_jsd, 16)
    model_jsd = [model_jsd, down_model_jsd, up_model_jsd]
    base_jsd = [base_jsd, down_base_jsd, up_base_jsd]
    return k, model_jsd, base_jsd


def make_gdiv_metrics_fig(experiment, params, m_metrics, gt_metrics, counts, filename):

    state_label = experiment.dynamics_model.state_label
    k, model_jsd, base_jsd = compute_jsd(state_label, m_metrics, gt_metrics)
    d_color = color_dark["grey"]
    p_color = color_pale["grey"]
    f_color = color_dark["grey"]
    s_color = color_dark["blue"]
    fig = plt.figure(figsize=(6, 6), frameon=False)
    gs = GridSpec(10, 1)
    gs.update(wspace=0.1, hspace=0.05)
    xlabel = r"$k$"
    ylabel = r"$\mathrm{JSD}_d(k)$"
    dist_label = r"$\mathrm{Pr}[k]$"
    ax = fig.add_subplot(gs[3:, :])
    ax_dist = fig.add_subplot(gs[1:3, :])
    ax_legend = fig.add_subplot(gs[:1, :])
    ax_legend.axis("off")
    ax_legend.set_zorder(1)
    ax_dist.set_yscale("log")
    ax_dist.spines["right"].set_visible(False)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["bottom"].set_visible(False)
    ax_dist.set_xticks([])
    counts.display(None, "train", for_degree=True, ax=ax_dist, color=d_color)
    ax.plot(k, model_jsd[0], linestyle="-", color=d_color, linewidth=2)
    ax.fill_between(k, model_jsd[1], model_jsd[2], color=f_color, alpha=0.3)
    ax.plot(k, base_jsd[0], linestyle="dotted", color=d_color, linewidth=2)
    ax.fill_between(k, base_jsd[1], base_jsd[2], color=f_color, alpha=0.3)
    ax.set_xlim([np.min(k), np.max(k)])
    ax_dist.set_xlim([np.min(k), np.max(k)])
    ax.set_ylim([0, 1])

    x = np.unique(np.sort(np.sum(counts.data["summaries"][:, 1:], axis=-1)))
    ax.axvspan(np.max(x), 100, alpha=0.3, color=s_color)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    ax_dist.set_ylabel(dist_label, fontsize=14)
    ax_dist.set_yticks([1e-2, 1])
    ax_dist.set_xlim(ax.get_xlim())
    handles = []
    handles.append(
        Line2D(
            [-1],
            [-1],
            marker="None",
            linestyle="-",
            markersize=10,
            color=color_dark["grey"],
            label=r"Model",
        )
    )
    handles.append(
        Line2D(
            [-1],
            [-1],
            marker="None",
            linestyle="dotted",
            markersize=10,
            color=color_dark["grey"],
            label=r"Baseline",
        )
    )
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

    fig, ax = plt.subplots(1, 1)
    metrics.display(0, state_label, ax=ax, cmap=colormap)
    ax.set_xticks(np.arange(d) + 0.5)
    ax.set_yticks(np.arange(d) + 0.5)
    ax.set_xticklabels([rf"{s}" for s in state_label], fontsize=14)
    ax.set_yticklabels([rf"{s}" for s in state_label], fontsize=14)
    ax.set_xlabel(r"Node state", fontsize=14)
    ax.set_ylabel(r"Neighbor state", fontsize=14)

    if filename is not None:
        fig.savefig(os.path.join(params["path"], "figures", filename))
    else:
        plt.show()
    return


def make_loss_metrics_fig(experiment, params, metrics, filename):

    for i, ds in enumerate(metrics.datasets):
        fig, ax = plt.subplots(1, 1)
        metrics.display(
            "approx_loss",
            ds,
            ax=ax,
            color=cd_list[0],
            kde_linestyle="-",
            rug_pos=-0.02,
            kde=True,
            rug=True,
            hist=True,
        )
        metrics.display(
            "exact_loss",
            ds,
            ax=ax,
            color=cd_list[1],
            kde_linestyle="-",
            rug_pos=-0.04,
            kde=True,
            rug=True,
            hist=True,
        )
        metrics.display(
            "diff_loss",
            ds,
            ax=ax,
            color=cd_list[2],
            kde_linestyle="-",
            rug_pos=-0.06,
            kde=True,
            rug=True,
            hist=True,
        )
        ax.set_xlabel(r"Loss")
        ax.set_ylabel(r"Distribution")
        handles = []
        handles.append(
            Line2D(
                [-1],
                [-1],
                marker="s",
                linestyle="None",
                markersize=10,
                color=cd_list[0],
                label=r"Approx.",
            )
        )
        handles.append(
            Line2D(
                [-1],
                [-1],
                marker="s",
                linestyle="None",
                markersize=10,
                color=cd_list[1],
                label=r"Exact",
            )
        )
        handles.append(
            Line2D(
                [-1],
                [-1],
                marker="s",
                linestyle="None",
                markersize=10,
                color=cd_list[2],
                label=r"Diff.",
            )
        )
        ax.legend(
            handles=handles,
            loc="best",
            fancybox=True,
            fontsize=10,
            framealpha=1,
            ncol=2,
        )
        fig.suptitle(r"Loss distribution", fontsize=16)
        if filename is not None:
            fig.savefig(os.path.join(params["path"], "figures", ds + "_" + filename))
        else:
            plt.show()

    return


def get_noisy_crossentropy(noise=0):
    def noisy_crossentropy(y_true, y_pred):
        num_classes = tf.cast(K.shape(y_true)[1], tf.float32)
        y_true = y_true * (1 - noise) + (1 - y_true) * noise / num_classes
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        return -K.sum(y_true * K.log(y_pred), axis=-1)

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
    if "SIS" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SIS(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SIR" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SIR(
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SoftThresholdSIS" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SoftThresholdSIS(
            params["dynamics"]["params"]["mu"],
            params["dynamics"]["params"]["beta"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SoftThresholdSIR" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SoftThresholdSIR(
            params["dynamics"]["params"]["mu"],
            params["dynamics"]["params"]["beta"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["init_param"],
        )
    elif "SISSIS" == params["dynamics"]["name"]:
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.dynamics.SISSIS(
            params["dynamics"]["params"]["infection_prob-2"],
            params["dynamics"]["params"]["recovery_prob-2"],
            params["dynamics"]["params"]["coupling"],
            params["dynamics"]["params"]["init_param"],
        )
    else:
        raise ValueError("wrong string name for dynamics.")


def get_aggregator(params):
    print(params["dynamics"]["name"])
    if "SIS" == params["dynamics"]["name"]:
        return dl.utilities.SimpleContagionAggregator()
    elif "SIR" == params["dynamics"]["name"]:
        return dl.utilities.SimpleContagionAggregator()
    elif "SoftThresholdSIS" == params["dynamics"]["name"]:
        return dl.utilities.ComplexContagionAggregator()
    elif "SoftThresholdSIR" == params["dynamics"]["name"]:
        return dl.utilities.ComplexContagionAggregator()
    elif "CooperativeContagionSIS" == params["dynamics"]["name"]:
        return dl.utilities.CooperativeContagionAggregator(0)
    else:
        raise ValueError("wrong string name for aggregator.")


def get_model(params, dynamics):
    if "LocalStatePredictor" == params["model"]["name"]:
        return dl.models.LocalStatePredictor(
            params["graph"]["params"]["N"],
            len(dynamics.state_label),
            params["model"]["params"]["in_features"],
            params["model"]["params"]["attn_features"],
            params["model"]["params"]["out_features"],
            params["model"]["params"]["n_heads"],
            in_activation=params["model"]["params"]["in_activation"],
            attn_activation=params["model"]["params"]["attn_activation"],
            out_activation=params["model"]["params"]["out_activation"],
            weight_decay=params["model"]["params"]["weight_decay"],
            seed=params["tf_seed"],
        )
    else:
        raise ValueError("wrong string name for model.")


def get_sampler(params, dynamics):
    if params["sampler"]["params"]["sample_from_weights"] == 0:
        params["sampler"]["params"]["sample_from_weights"] = False
    elif params["sampler"]["params"]["sample_from_weights"] == 1:
        params["sampler"]["params"]["sample_from_weights"] = True
    else:
        raise ValueError("sample_from_weights parameter must be (0, 1).")

    if params["sampler"]["name"] == "SequentialSampler":
        return dl.generators.SequentialSampler()
    elif params["sampler"]["name"] == "RandomSampler":
        return dl.generators.RandomSampler(
            replace=params["sampler"]["params"]["replace"],
            sample_from_weights=params["sampler"]["params"]["sample_from_weights"],
        )
    elif params["sampler"]["name"] == "DegreeBiasedSampler":
        return dl.generators.DegreeBiasedSampler(
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            replace=params["sampler"]["params"]["replace"],
            sample_from_weights=params["sampler"]["params"]["sample_from_weights"],
        )

    elif params["sampler"]["name"] == "StateBiasedSampler":
        return dl.generators.StateBiasedSampler(
            dynamics,
            sampling_bias=params["sampler"]["params"]["sampling_bias"],
            replace=params["sampler"]["params"]["replace"],
            sample_from_weights=params["sampler"]["params"]["sample_from_weights"],
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
