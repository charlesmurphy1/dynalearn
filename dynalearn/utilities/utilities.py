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


def get_schedule(schedule):
    def lr_schedule(epoch, lr):
        if (epoch + 1) % schedule["epoch"] == 0:
            lr /= schedule["factor"]
        return lr

    return lr_schedule


def get_noisy_crossentropy(noise=0):
    def noisy_crossentropy(y_true, y_pred):
        num_classes = tf.cast(K.shape(y_true)[1], tf.float32)
        y_true = y_true * (1 - noise) + (1 - y_true) * noise / num_classes
        y_pred = K.clip(y_pred, K.epsilon(), 1)

        return -K.sum(y_true * K.log(y_pred), axis=-1)

    return noisy_crossentropy


def get_meanfield(params, p_k_distribution):
    if "SIS" == params["dynamics"]["name"]:
        return dl.meanfields.SIS_MF(
            p_k_distribution,
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif "SIR" == params["dynamics"]["name"]:
        return dl.meanfields.SIR_MF(
            p_k_distribution,
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif "SoftThresholdSIS" == params["dynamics"]["name"]:
        return dl.meanfields.SoftThresholdSIS_MF(
            p_k_distribution,
            params["dynamics"]["params"]["mu"],
            params["dynamics"]["params"]["beta"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif "SoftThresholdSIR" == params["dynamics"]["name"]:
        return dl.meanfields.SoftThresholdSIR_MF(
            p_k_distribution,
            params["dynamics"]["params"]["mu"],
            params["dynamics"]["params"]["beta"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif "NonLinearSIS" == params["dynamics"]["name"]:
        return dl.meanfields.NonLinearSIS_MF(
            p_k_distribution,
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["alpha"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif "NonLinearSIR" == params["dynamics"]["name"]:
        return dl.meanfields.NonLinearSIR_MF(
            p_k_distribution,
            params["dynamics"]["params"]["infection_prob"],
            params["dynamics"]["params"]["recovery_prob"],
            params["dynamics"]["params"]["alpha"],
            params["dynamics"]["params"]["recovery_prob"],
        )
    elif (
        "SISSIS" == params["dynamics"]["name"]
        or "CooperativeContagionSIS" == params["dynamics"]["name"]
    ):
        if params["dynamics"]["params"]["init_param"] == "None":
            params["dynamics"]["params"]["init_param"] = None
        return dl.meanfields.SISSIS_MF(
            p_k_distribution,
            params["dynamics"]["params"]["infection_prob-2"],
            params["dynamics"]["params"]["recovery_prob-2"],
            params["dynamics"]["params"]["coupling"],
        )
    else:
        raise ValueError("wrong string name for dynamics.")


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
