import matplotlib.pyplot as plt
import numpy as np


color_dark = {
    "blue": "#1f77b4",
    "orange": "#f19143",
    "purple": "#9A80B9",
    "red": "#d73027",
    "grey": "#525252",
    "green": "#33b050",
}

color_pale = {
    "blue": "#7bafd3",
    "orange": "#f7be90",
    "purple": "#c3b4d6",
    "red": "#e78580",
    "grey": "#999999",
    "green": "#9fdaac",
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


class Display:
    def __init__(self, experiment):
        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

    # def __call__(self, metrics, ax=None, colors=None, markers=None, linetyles=None, **kwargs):
    #     for m in metrics:
    #
