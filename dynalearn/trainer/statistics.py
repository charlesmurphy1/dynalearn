from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.lines import Line2D
import numpy as np
import torch
import os

from ..utilities.utilities import exp_mov_avg


__all__ = ['Statistics', 'Training_Statistics',
           'Model_Statistics', 'Distribution_Statistics']


class Statistics(object):
    """
    Base class to statistics during learning.
    
    Statistics contains specific information to be monitored during learning.
    
    **Parameters**

    strname : String
        Name of the statistics in display format (support LaTeX syntax).

    filename : String
        Name of the statistics for file naming (no space).

    colors : List : (default = ``None``)
        List of two colors for plot display. If ``None``, default value is 
        [``'#2166ac'``, ``'#d6604d'``]

    ext : String : (default = ``.png``)
        Extension for figure saving.

    makeplot : Bool : (default = ``False``)
        If ``True``, it builds figure.

    precision : Integer : (default = ``4``)
        Precision on display.

    """
    def __init__(self, strname, filename, colors=None,
                 ext=".png", makeplot=False, precision=4):
        """
        Initializes an object Statistics.

        """
        super(Statistics, self).__init__()
        self.strname = strname
        self.filename = filename
        self.ext = ext
        self.makeplot = makeplot
        self.precision = precision
        self.latest_update = 0

        self.stat = {}
        self.estimated_stat = {}

        if colors is None:
            self.colors = ['#2166ac', '#d6604d']
        else:
            self.colors = colors

        self.cm = LSC.from_list("cm", [self.colors[0], self.colors[1]], N=100)
        if self.makeplot: self.init_plot()
        else: self.plot_stat = lambda path, best: None

    def __str__(self):
        return self.str_update()

    def str_update(self, update=None):
        """
        Renders the String of statistics

        **Parameters**
        update : Integer : (default = ``None``)
            Statistics of specific update to be rendered in String. If ``None``,
            renders latest update.

        **Returns**
        str_of_update : String

        """
        raise NotImplementedError("self.str_update() has not been implemented")

    def init_plot(self):
        """
        Initialize the figure and axis for the statistics display.

        """
        raise NotImplementedError("self.init_plot() has not been implemented")

    def evaluate(self, update, data, model):
        """
        Evaluates the statistics and stores it in ``self.stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to evaluate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        raise NotImplementedError("self.evaluate() has not been implemented")

    def estimate(self, update, sample_data, model):
        """
        Estimates the statistics and stores it in ``self.estimated_stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        sample_data : Dict
            Available sample of the data to estimate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        raise NotImplementedError("self.estimate() has not been implemented")

    def is_better(self, update1, update2):
        """
        Compare two update statistics to check which one is better.

        ..note::
            This method must be initialized when object Statistics is to be used
            for early stopping.

        **Parameters**
        update1, update2 : Integer
            Update at which the statistics is evaluated.

        **Returns**
        check_if_better : Bool
            If ``True``, statistics in update1 is better than statistics in
            update2.


        """
        raise NotImplementedError("self.is_better() has not been implemented")

    def plot_stat(self, path=None, best=None):
        """
        Plots the statistics using matplotlib.pyplot.

        **Parameters**
        path : String : (default = ``None``)
            Path where to save figure.

        best : Integer : (default = ``None``)
            Update of best statistics. If ``None``, it does not render best 
            update location on plot.

        """
        raise NotImplementedError("self.plot_stats() has not been implemented")

    def save_stat(self, path):
        """
        Saves statistics.

        **Parameters**
        path : String
            Path where to save figure.

        """
        raise NotImplementedError("self.save_stats() has not been implemented")


class Training_Statistics(Statistics):
    """
    Class Training_Statistics specific to training performances.
    
    **Parameters**

    strname : String
        Name of the statistics in display format (support LaTeX syntax).

    filename : String
        Name of the statistics for file naming (no space).

    colors : List : (default = ``None``)
        List of two colors for plot display. If ``None``, default is 
        [``#2166ac``, ``#d6604d``].

    ext : String : (default = ``.png``)
        Extension for figure saving.

    graining : Integer (default = ``2``)
        Coarse graining factor when displaying statistics.

    ..note::
        The coarse graining is obtained using an exponential moving average.

    makeplot : Bool : (default = ``False``)
        If ``True``, it builds figure.

    precision : Integer : (default = ``4``)
        Precision on display.
    """
    def __init__(self, strname, filename, colors=None, ext=".png", graining=0, makeplot=False, precision=4):
        """
        Initializes an object Training_Statistics.

        """
        self.graining = graining
        super(Training_Statistics, self).__init__(strname, filename, colors, ext,
                                                  makeplot, precision)
        self.stat = {"train": {}, "val":{}}
        self.estimated_stat = {"train": {}, "val":{}}

    def str_update(self, update=None):
        """
        Renders the String of statistics

        **Parameters**
        update : Integer : (default = ``None``)
            Statistics of specific update to be rendered in String. If ``None``,
            renders latest update.

        **Returns**
        str_of_update : String

        """
        if update is None:
            update = self.latest_update
        t = float(self.stat["train"][update])
        v = float(self.stat["val"][update])

        train_string = self.strname + " : {0:.{p}f}".format(t, p=self.precision)
        val_string = self.strname + " : {0:.{p}f}".format(v, p=self.precision)
        return (train_string, val_string)

    def init_plot(self):
        """
        Initialize the figure and axis for the statistics display.

        """
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel("Number of updates")
        self.ax.set_ylabel(self.strname)
    
    def eval_statpoint(self, data_point, model):
        """
        Evaluate statistics on specific data point.


        **Parameters**
        data_point : torch.Tensor
            Data point on which statistics is evaluated.

        model : Model
            Model object on which the statistics is performed.

        **Returns**
        stat_point : Float
            Statistics corresponding to data point.


        """
        raise NotImplementedError("self.eval_statpoint() has not been \
                                  implemented.")


    def func_evaluate(self, update, data, model):
        """
        Evaluation proceder for statistics.


        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Data on which statistics is evaluated. 
        
        ..warning::
            ``data`` must contain keys [``"train"``, ``"val"``].

        model : Model
            Model object on which the statistics is performed.

        **Returns**
        stat : Float
            Statistics corresponding to data.

        """
        N_t = len(data["train"])
        N_v = len(data["val"])

        stat = {"train": 0, "val": 0}

        for i, t in enumerate(data["train"]):
            batch = self.eval_statpoint(t, model)
            stat["train"] += batch / N_t

        for i, v in enumerate(data["val"]):
            batch = self.eval_statpoint(v, model)
            stat["val"] += batch / N_v

        return stat

    def evaluate(self, update, data, model):
        """
        Evaluates the statistics and stores it in ``self.stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to estimate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        stat = self.func_evaluate(update, data, model)
        self.stat["train"][update] = stat["train"]
        self.stat["val"][update] = stat["val"]

        self.latest_update = update

    def estimate(self, update, data, model):
        """
        Estimates the statistics and stores it in ``self.estimated_stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to estimate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training 
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        stat = self.func_evaluate(update, data, model)
        self.estimated_stat["train"][update] = stat["train"]
        self.estimated_stat["val"][update] = stat["val"]

    def plot_stat(self, path=None, best=None):
        """
        Plots the statistics using matplotlib.pyplot.

        **Parameters**
        path : String : (default = ``None``)
            Path where to save figure. If ``None``, it does not save the figure.

        best : Integer : (default = ``None``)
            Update of best statistics. If ``None``, it does not render best 
            update location on plot.

        """
        # Plot estimate
        update_estimate = []
        train_estimate = []
        val_estimate = []
        for t, d in self.estimated_stat["train"].items():
            update_estimate.append(t)
            train_estimate.append(d)

        for t, d in self.estimated_stat["val"].items():
            val_estimate.append(d)

        min_val = float(min(min(train_estimate), min(val_estimate)))
        max_val = float(max(max(train_estimate), max(val_estimate)))

        self.ax.plot(update_estimate, train_estimate, marker='None',
                     linestyle='-', color=self.colors[0], lw=1, alpha=0.4)
        self.ax.plot(update_estimate, val_estimate, marker='None',
                     linestyle='-', color=self.colors[1], lw=1, alpha=0.4)

        run_mean_train = exp_mov_avg(train_estimate, self.graining)
        run_mean_val = exp_mov_avg(val_estimate, self.graining)

        self.ax.plot(update_estimate, run_mean_train, marker='None',
                     linestyle='-', color=self.colors[0], lw=2, alpha=0.8)
        self.ax.plot(update_estimate, run_mean_val, marker='None',
                     linestyle='-', color=self.colors[1], lw=2, alpha=0.8)
        self.ax.set_xlim([0, max(update_estimate)])

        self.ax.set_ylim([min_val, max_val])

        # Plot data
        update_stat = []
        train_stat = []
        val_stat = []
        for t, l in self.stat["train"].items():
            update_stat.append(t)
            train_stat.append(l)

        for t, l in self.stat["val"].items():
            val_stat.append(l)

        self.ax.plot(update_stat, train_stat, marker='o', markeredgewidth=1.,
                     markeredgecolor="k", linestyle='-', color=self.colors[0],
                     lw=1., alpha=1)
        self.ax.plot(update_stat, val_stat, marker='o', markeredgewidth=1.,
                     markeredgecolor="k", linestyle='-', color=self.colors[1],
                     lw=1., alpha=1)

        if best is not None:
            np.arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None',
                         linestyle='--',color="grey", lw=2, alpha=0.5)

        #Making legend
        legend_linetype = [Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[0], lw=2, alpha=1),
                           Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[1], lw=2, alpha=1),
                           Line2D([0], [0], marker='None', linestyle='--',
                                  color="grey", lw=2, alpha=1)
                           ]
        self.ax.legend(legend_linetype,
                       [r"Training", r"Validation", r"Best epoch"]
                       )

        if path is not None:
            self.fig.savefig(os.path.join(path, self.filename + self.ext))

    def save_stat(self, path):
        """
        Saves statistics.

        **Parameters**
        path : String
            Path where to save figure.

        """
        n = len(self.stat["train"])
        stat = np.zeros([n, 3])

        for i, k in enumerate(self.stat["train"].keys()):
            stat[i, 0] = k
            stat[i, 1] = self.stat["train"][k]
            stat[i, 2] = self.stat["val"][k]

        np.savetxt(os.path.join(path, self.filename + ".txt"), stat)


class Model_Statistics(Statistics):
    """
    Class Model_Statistics specific to training performances.
    
    **Parameters**

    strname : String
        Name of the statistics in display format (support LaTeX syntax).

    filename : String
        Name of the statistics for file naming (no space).

    colors : List : (default = ``None``)
        List of two colors for plot display. If ``None``, default is 
        [``#2166ac``, ``#d6604d``].

    ext : String : (default = ``.png``)
        Extension for figure saving.

    graining : Integer : (default = ``2``)
        Coarse graining factor when displaying statistics.

    ..note::
        The coarse graining is obtained using an exponential moving average.

    makeplot : Bool : (default = ``False``)
        If ``True``, it builds figure.

    precision : Integer : (default = ``4``)
        Precision on display.
    """
    def __init__(self, strname, filename, 
                 colors=None, ext=".png", graining=0,
                 makeplot=False, precision=4):
        """
        Initializes an object Model_Statistics.

        """
        self.graining = graining
        self.stat = {}
        self.estimated_stat = {}
        super(Model_Statistics, self).__init__(strname, filename, colors, ext, 
                                               makeplot, precision)


    def str_update(self, update=None):
        """
        Renders the String of statistics

        **Parameters**
        update : Integer : (default = ``None``)
            Statistics of specific update to be rendered in String. If ``None``, renders latest update.

        **Returns**
        str_of_update : String

        """
        if update is None:
            update = self.latest_update
        d = self.stat[update]
        model_string = self.strname + " : {0:.{p}f}".format(d, p=self.precision)
        return model_string
    

    def init_plot(self):
        """
        Initialize the figure and axis for the statistics display.

        """
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel("Number of updates")
        self.ax.set_ylabel(self.strname)

    def func_evaluate(self, update, data, model):
        raise NotImplementedError("self.func_evaluate() has not been \
                                   implemented.")

    def evaluate(self, update, data, model):
        stat = self.func_evaluate(update, data, model)
        self.stat[update] = stat
        self.latest_update = update

    def estimate(self, update, data, model):
        stat = self.func_evaluate(update, data, model)
        self.estimated_stat[update] = stat

    def plot_stat(self, path=None, best=None):
        """
        Plots the statistics using matplotlib.pyplot.

        **Parameters**
        path : String : (default = ``None``)
            Path where to save figure. If ``None``, it does not save the figure.

        best : Integer
            Update of best statistics. If ``None``, it does not render best 
            update location on plot.

        """
        updates = []
        stat = []
        for t, d in self.stat.items():
            updates.append(t)
            stat.append(d)

        min_val = min(stat)
        max_val = max(stat)
        self.ax.plot(updates, stat, marker='o', linestyle='-',
                     color=self.colors[0], lw=1, alpha=1.)
        self.ax.set_xlim([0, max(updates)])
        self.ax.set_ylim([min_val, max_val])

        if best is not None:
            arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None',
                          linestyle='--',color="grey", lw=2, alpha=0.5)        

        if path is not None:
            self.fig.savefig(os.path.join(path, self.filename + self.ext))

    def save_stat(self, path):
        """
        Saves statistics.

        **Parameters**
        path : String
            Path where to save figure.

        """
        n = len(self.stat)
        stat = np.zeros([n, 2])

        for i, k in enumerate(self.stat.keys()):
            stat[i, 0] = k
            stat[i, 1] = self.stat[k]

        np.savetxt(os.path.join(path, self.filename + ".txt"), stat)


class Distribution_Statistics(Statistics):
    """
    Class Distribution_Statistics specific to training performances.
    
    **Parameters**

    strname : String
        Name of the statistics in display format (support LaTeX syntax).

    filename : String
        Name of the statistics for file naming (no space).

    colors : List : (default = ``None``)
        List of two colors for plot display. If ``None``, default is
        [``#2166ac``, ``#d6604d``].

    ext : String : (default = ``.png``)
        Extension for figure saving.

    nbins : Integer : (default = ``100``)
        Number of bins for histograms.

    makeplot : Bool : (default = ``False``)
        If ``True``, it builds figure.

    precision : Integer : (default = ``4``)
        Precision on display.
    """
    def __init__(self, strname, filename, colors=None, ext=".png",
                 nbins=100, makeplot=False, precision=4):
        """
        Initializes an object Distribution_Statistics.

        """
        super(Distribution_Statistics, self).__init__(strname, filename, colors,
                                                      ext, makeplot, precision)
        self.nbins = nbins
        self.stat = {"dist": {}, "bins": {}}
        self.mean = {}
        self.scale = {}


    def str_update(self, update=None):
        """
        Renders the String of statistics

        **Parameters**
        update : Integer : (default = ``None``)
            Statistics of specific update to be rendered in String. 
            If ``None``, renders latest update.

        **Returns**
        str_of_update : String

        """
        if update is None:
            update = self.latest_update
        d = self.mean[update]
        s = self.scale[update]
        model_string = self.filename +\
                       " : {0:.{p}f}, {1:.{p}f}".format(d, s, p=self.precision)
        return model_string

    def init_plot(self):
        """
        Initialize the figure and axis for the statistics display.

        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel(self.strname + " distribution")
        self.ax.set_ylabel("Number of updates")

        self.ax.w_zaxis.line.set_lw(0.)
        self.ax.invert_xaxis()
        self.ax.set_zticks([])

        self.ax.view_init(45, 90)

        # Get rid of colored axes planes
        # First remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        self.ax.xaxis.pane.set_edgecolor('w')
        self.ax.yaxis.pane.set_edgecolor('w')
        self.ax.zaxis.pane.set_edgecolor('w')
        self.ax.grid(False)

    def evaluate(self, update, data, model):
        """
        Evaluates the statistics and stores it in ``self.stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to evaluate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        """
        raise NotImplementedError("self.evaluate() has not been implemented.")

    def estimate(self, update, data, model):
        """
        Estimate the statistics and stores it in ``self.stat``.

        **Parameters**
        update : Integer
            Update at which the statistics is evaluated.

        data : Dict
            Available data to evaluate stat.

        ..note::
            Usually generated by a ``Trainer`` object, it contains the keys 
            [``"train"``, ``"val"``, ``"grad"``] corresponding to the training
            dataset, validation dataset and the gradient of each parameter.

        model : Model
            Model object on which the statistics is performed.

        **Returns**
        None

        ..note::
            For Distribution_statistics object, estimate() returns ``None``, 
            because it cannot be estimated.

        """
        return None


    def plot_stat(self, path=None, best=None):
        """
        Plots the statistics using matplotlib.pyplot.

        **Parameters**
        path : String : (default = ``None``)
            Path where to save figure. If ``None``, it does not save the figure.

        best : Integer
            Update of best statistics. If ``None``, it does not render best 
            update location on plot.

        """
        num_update = len(self.stat["dist"])
        max_x, min_x = 0, 1e300
        max_z = 0

        for i, t in enumerate(self.stat["dist"]):
            dist = self.stat["dist"][t].numpy()
            bins = self.stat["bins"][t].numpy()
            width = np.mean(abs(bins[1:] - bins[:-1]))

            ec = "w"
            if i == best:
                ec = "k"
            
            if t != best:
                self.ax.bar(bins, dist, zs=t, zdir='y', 
                            color=self.cm(i/num_update), alpha=1, linewidth=0.4,
                            width=width, ec=ec)
            else:
                self.ax.bar(bins, dist, zs=t, zdir='y',
                            color=self.cm(i/num_update), alpha=1, linewidth=0.4,
                             width=width, ec="k")
            max_x = max(max(bins), max_x)
            min_x = min(min(bins), min_x)
            max_z = max(max(dist), max_z)

        # self.ax.set_xlim([max_x, min_x])
        self.ax.set_ylim([0, self.latest_update])
        self.ax.set_zlim([0, max_z])
        self.ax.locator_params(nbins=6)

        if path is not None:
            self.fig.savefig(os.path.join(path, self.filename + self.ext))

    def save_stat(self, path):
        """
        Saves statistics.

        **Parameters**
        path : String
            Path where to save figure.

        ..note::
            For Distribution_Statistics, save_stat() saves nothing. Therefore, 
            it only returns ``None``.

        """
        return None