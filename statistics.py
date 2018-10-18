from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LSC
from matplotlib.lines import Line2D
import numpy as np
import utilities as util
import torch
import os

class Statistics(object):
    """docstring for Statistics"""
    def __init__(self, strname, filename, colors=None,
                 ext=".png", makeplot=False, precision=4):

        super(Statistics, self).__init__()
        self.strname = strname
        self.filename = filename
        self.ext = ext
        self.makeplot = makeplot
        self.precision = precision
        self.current_update = 0

        self.data = {}
        self.estimated_data = {}

        self.best_color='#022b3a'
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
        raise NotImplementedError("self.__str__() has not been implemented")

    def init_plot(self):
        raise NotImplementedError("self.init_plot() has not been implemented")

    def evaluate(self, update, data, bm):
        raise NotImplementedError("self.evaluate() has not been implemented")

    def estimate(self, update, data, bm):
        raise NotImplementedError("self.estimate() has not been implemented")

    def is_better(self, update1, update2):
        # is update1 better than update2
        raise NotImplementedError("self.is_better() has not been implemented")

    def plot_stat(self, path=None, best=None):
        raise NotImplementedError("self.plot_stats() has not been implemented")

    def save_stat(self, path):
        raise NotImplementedError("self.save_stats() has not been implemented")


class Scalar_Statistics(Statistics):
    """docstring for Scalar_Statistics"""
    def __init__(self, strname, filename, colors=None, ext=".png", graining=2, makeplot=False, precision=4):
        self.graining = graining
        super(Scalar_Statistics, self).__init__(strname, filename, colors, ext, makeplot, precision)
        self.data = {"train": {}, "val":{}}
        self.estimated_data = {"train": {}, "val":{}}

    def str_update(self, update=None):
        if update is None:
            update = self.current_update
        t = float(self.data["train"][update].numpy())
        v = float(self.data["val"][update].numpy())

        train_string = self.strname + " : {0:.{precision}f}".format(t, precision=self.precision)
        val_string = self.strname + " : {0:.{precision}f}".format(v, precision=self.precision)
        return (train_string, val_string)
    
    def eval_statpoint(self, data, bm):
        raise NotImplementedError("self.eval_statpoint() has not been implemented.")

    def init_plot(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel("Number of updates")
        self.ax.set_ylabel(self.strname)


    def func_evaluate(self, update, data, bm):
        N_t = len(data["train"])
        N_v = len(data["val"])

        data_update = {"train": 0, "val": 0}

        for i, t in enumerate(data["train"]):
            batch = self.eval_statpoint(t, bm)
            data_update["train"] += torch.mean(batch) / N_t

        for i, v in enumerate(data["val"]):
            batch = self.eval_statpoint(v, bm)
            data_update["val"] += torch.mean(batch) / N_v

        return data_update

    def evaluate(self, update, data, bm):
        data_update = self.func_evaluate(update, data, bm)
        self.data["train"][update] = data_update["train"]
        self.data["val"][update] = data_update["val"]

        self.current_update = update

    def estimate(self, update, data, bm):
        data_update = self.func_evaluate(update, data, bm)
        self.estimated_data["train"][update] = data_update["train"]
        self.estimated_data["val"][update] = data_update["val"]

    def plot_stat(self, path=None, best=None):

        # Plot estimate
        update_estimate = []
        train_estimate = []
        val_estimate = []
        for t, d in self.estimated_data["train"].items():
            update_estimate.append(t)
            train_estimate.append(d)

        for t, d in self.estimated_data["val"].items():
            val_estimate.append(d)

        min_val = min(min(train_estimate), min(val_estimate))
        max_val = max(max(train_estimate), max(val_estimate))

        self.ax.plot(update_estimate, train_estimate, marker='None', linestyle='-',
                     color=self.colors[0], lw=1, alpha=0.4)
        self.ax.plot(update_estimate, val_estimate, marker='None', linestyle='-',
                     color=self.colors[1], lw=1, alpha=0.4)

        self.ax.plot(update_estimate[:1 - self.graining], util.running_mean(train_estimate, self.graining), marker='None', linestyle='-',
                     color=self.colors[0], lw=2, alpha=0.8)
        self.ax.plot(update_estimate[:1 - self.graining], util.running_mean(val_estimate, self.graining), marker='None', linestyle='-',
                     color=self.colors[1], lw=2, alpha=0.8)
        self.ax.set_xlim([0, max(update_estimate)])
        self.ax.set_ylim([min_val, max_val])

        # Plot data
        update_data = []
        train_data = []
        val_data = []
        for t, l in self.data["train"].items():
            update_data.append(t)
            train_data.append(l)

        for t, l in self.data["val"].items():
            val_data.append(l)

        self.ax.plot(update_data, train_data, marker='o', markeredgewidth=1., markeredgecolor="k", linestyle='-',
                     color=self.colors[0], lw=1., alpha=1)
        self.ax.plot(update_data, val_data, marker='o', markeredgewidth=1., markeredgecolor="k", linestyle='-',
                     color=self.colors[1], lw=1., alpha=1)

        if best is not None:
            np.arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None', linestyle='--',
                     color="grey", lw=2, alpha=0.5)

        #Making legend
        legend_linetype = [Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[0], lw=2, alpha=1),
                           Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[1], lw=2, alpha=1),
                           Line2D([0], [0], marker='None', linestyle='--',
                                  color="grey", lw=2, alpha=1)]
        self.ax.legend(legend_linetype, [r"Training", r"Validation", r"Best epoch"])

        if path is not None:
            self.fig.savefig(path + self.filename + self.ext)

    def save_stat(self, path):
        n = len(self.data["train"])
        data = np.zeros([n, 3])

        for i, k in enumerate(self.data["train"].keys()):
            data[i, 0] = k
            data[i, 1] = self.data["train"][k]
            data[i, 2] = self.data["val"][k]

        np.savetxt(path + self.filename + ".txt", data)




class Model_Statistics(Statistics):
    """docstring for Model_Statistics"""
    def __init__(self, strname, filename, colors=None, ext=".png", graining=2, makeplot=False, precision=4):
        self.graining = graining
        super(Model_Statistics, self).__init__(strname, filename, colors, ext, makeplot, precision)


    def str_update(self, update=None):
        if update is None:
            update = self.current_update
        d = self.data[update]
        model_string = self.strname + " : {0:.{precision}f}".format(d, precision=self.precision)
        return model_string
    

    def init_plot(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel("Number of updates")
        self.ax.set_ylabel(self.strname)

    def func_evaluate(self, update, data, bm):
        raise NotImplementedError("self.func_evaluate() has not been implemented.")

    def evaluate(self, update, data, bm):
        data_update = self.func_evaluate(update, data, bm)
        self.estimated_data[update] = data_update
        self.current_update = update

    def estimate(self, update, data, bm):
        data_update = self.func_evaluate(update, data, bm)
        self.estimated_data[update] = data_update

    def plot_stat(self, path=None, best=None):
        updates = []
        data = []
        for t, d in self.data.items():
            updates.append(t)
            data.append(d)

        min_val = min(data)
        max_val = max(data)
        self.ax.plot(updates, data, marker='.', linestyle='-',
                     color=self.colors[0], lw=1, alpha=1.)
        self.ax.set_xlim([0, max(updates)])
        self.ax.set_ylim([min_val, max_val])

        if best is not None:
            arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None', linestyle='--',
                         color="grey", lw=2, alpha=0.5)        

        if path is not None:
            self.fig.savefig(path + self.filename + self.ext)

    def save_stat(self, path):
        n = len(self.data)
        data = np.zeros([n, 2])

        for i, k in enumerate(self.data.keys()):
            data[i, 0] = k
            data[i, 1] = self.data[k]

        np.savetxt(path + self.filename + ".txt", data)


class Distribution_Statistics(Statistics):
    """docstring for Distribution_Statistics"""
    def __init__(self, strname, filename, colors=None, ext=".png",
                 nbins=100, makeplot=False, precision=4):
        super(Distribution_Statistics, self).__init__(strname, filename, colors, ext, makeplot, precision)
        self.nbins = nbins
        self.data = {"dist": {}, "bins": {}}
        self.mean = {}
        self.scale = {}


    def str_update(self, update=None):
        if update is None:
            update = self.current_update
        d = self.mean[update]
        s = self.scale[update]
        model_string = self.filename + " : {0:.{precision}f}, {1:.{precision}f}".format(d, s, precision=self.precision)
        return model_string

    def init_plot(self):
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

    def evaluate(self, update, data, bm):
        raise NotImplementedError("self.evaluate() has not been implemented.")

    def estimate(self, update, data, bm):
        return None


    def plot_stat(self, path=None, best=None):
        num_update = len(self.data["dist"])
        max_x, min_x = 0, 1e300
        max_z = 0

        for i, t in enumerate(self.data["dist"]):
            dist = self.data["dist"][t].numpy()
            bins = self.data["bins"][t].numpy()
            width = np.mean(abs(bins[1:] - bins[:-1]))

            ec = "w"
            if i == best:
                ec = "k"
            
            if t != best:
                self.ax.bar(bins, dist, zs=t, zdir='y', color=self.cm(i/num_update),
                            alpha=1, linewidth=0.4, width=width, ec=ec)
            else:
                self.ax.bar(bins, dist, zs=t, zdir='y', color=self.cm(i/num_update),
                            alpha=1, linewidth=0.4, width=width, ec="k")
            max_x = max(max(bins), max_x)
            min_x = min(min(bins), min_x)
            max_z = max(max(dist), max_z)

        # self.ax.set_xlim([max_x, min_x])
        self.ax.set_ylim([0, self.current_update])
        self.ax.set_zlim([0, max_z])
        self.ax.locator_params(nbins=6)

        if path is not None:
            self.fig.savefig(path + self.filename + self.ext)

    def save_stat(self, path):
        return None


class LogLikelihood_Statistics(Scalar_Statistics):
    def __init__(self, strname="Log-likelihood", filename="log_likelihood",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2,
                num_sample=10, betas=None, recompute=True):
        self.num_sample = num_sample
        self.betas = betas
        self.recompute = recompute
        super(LogLikelihood_Statistics, self).__init__(strname,
                                                 filename,
                                                 colors,
                                                 ext,
                                                 graining,
                                                 makeplot,
                                                 precision)

    def eval_statpoint(self, data, bm):
        return bm.log_likelihood(data, self.num_sample, self.betas, self.recompute)

    def is_better(self, update1, update2):
        return self.data["val"][update1] > self.data["val"][update2]


class Parition_Function_Statistics(Model_Statistics):
    def __init__(self, strname="Partition function", filename="part_func",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2,
                num_sample=10, betas=None):
        self.num_sample = num_sample
        self.betas = betas
        super(Parition_Function_Statistics, self).__init__(strname,
                                                 filename,
                                                 colors,
                                                 ext,
                                                 graining,
                                                 makeplot,
                                                 precision)

    def evaluate(self, update, data, bm):
        self.data[update] = bm._log_Z(self.num_sample, self.betas, True)
        self.current_update = update

    def estimate(self, update, data, bm):
        return None


class Pseudolikelihood_Statistics(Scalar_Statistics):
    def __init__(self, strname="Pseudolikelihood", filename="pseudo_likelihood",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2):
        super(Pseudolikelihood_Statistics, self).__init__(strname,
                                                 filename,
                                                 colors,
                                                 ext,
                                                 graining, 
                                                 makeplot,
                                                 precision)

    def eval_statpoint(self, data, bm):
        cond_log_p = bm.conditional_log_p(data)
        pseudo_likelihood = torch.zeros(data.size(0))

        for u in cond_log_p:
            pseudo_likelihood += torch.sum(cond_log_p[u], 1)

        return pseudo_likelihood

    def is_better(self, update1, update2):
        return self.data["val"][update1] > self.data["val"][update2]

class Free_Energies_Statistics(Scalar_Statistics):
    def __init__(self, strname="Free energies", filename="free_energies",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2):
        super(Free_Energies_Statistics, self).__init__(strname, 
                                                     filename,
                                                     colors, ext, 
                                                     graining,
                                                     makeplot,
                                                     precision)


    def eval_statpoint(self, data, bm):
        return bm.free_energy(data)

    def plot_stat(self, path=None, best=None):

        # Plot estimate
        update_estimate = []
        train_estimate = []
        val_estimate = []
        for t, l in self.estimated_data["train"].items():
            update_estimate.append(t)
            train_estimate.append(l)

        for t, l in self.estimated_data["val"].items():
            val_estimate.append(l)

        min_val = min(min(train_estimate), min(val_estimate))
        max_val = max(max(train_estimate), max(val_estimate))

        ## Free energy
        ### Complete plots
        self.ax.plot(update_estimate, train_estimate, marker='None', linestyle='-',
                     color=self.colors[0], lw=1, alpha=0.4)
        self.ax.plot(update_estimate, val_estimate, marker='None', linestyle='-',
                     color=self.colors[1], lw=1, alpha=0.4)

        ### Coarse-grained plots
        self.ax.plot(update_estimate[:1-self.graining], util.running_mean(train_estimate, self.graining), marker='None', linestyle='-',
                     color=self.colors[0], lw=2, alpha=0.8)
        self.ax.plot(update_estimate[:1-self.graining], util.running_mean(val_estimate, self.graining), marker='None', linestyle='-',
                     color=self.colors[1], lw=2, alpha=0.8)
        self.ax.set_xlim([0, max(update_estimate)])
        self.ax.set_ylim([min_val, max_val])


        train_estimate = np.array(train_estimate)
        val_estimate = np.array(val_estimate)

        ## Gap
        axx = self.ax.twinx()
        ### Complete plots
        gap_estimate = train_estimate - val_estimate
        axx.plot(update_estimate, gap_estimate, marker='None', linestyle='-',
                     color=self.cm(0.5), lw=1, alpha=0.4)
        ### Coarse-grained plots
        axx.plot(update_estimate[:1-self.graining], util.running_mean(train_estimate - val_estimate, self.graining), marker='None', linestyle='-',
                     color=self.cm(0.5), lw=2, alpha=0.8)

        axx.set_xlim([0, max(update_estimate)])
        axx.set_ylim([min(gap_estimate), max(gap_estimate)])

        # Plot data
        update_data = []
        train_data = []
        val_data = []
        for t, l in self.data["train"].items():
            update_data.append(t)
            train_data.append(l)

        for t, l in self.data["val"].items():
            val_data.append(l)

        self.ax.plot(update_data, train_data, marker='o', markeredgewidth=1., markeredgecolor="k", linestyle='-',
                     color=self.colors[0], lw=1., alpha=1)
        self.ax.plot(update_data, val_data, marker='o', markeredgewidth=1., markeredgecolor="k", linestyle='-',
                     color=self.colors[1], lw=1., alpha=1)
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        axx.plot(update_data, train_data - val_data, marker='o', markeredgewidth=1., markeredgecolor="k", linestyle='-',
                     color=self.cm(0.5), lw=1., alpha=1)


        if best is not None:
            np.arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None', linestyle='--',
                     color="grey", lw=2, alpha=0.5)

        axx.set_ylabel("Gap")
        # axx.set_ylabel("Gap", color=self.cm(0.5))
        # axx.tick_params("y", colors=self.cm(0.5))
        # Making legend
        legend_linetype = [Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[0], lw=2),
                           Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[1], lw=2),
                           Line2D([0], [0], marker='None', linestyle='--',
                                  color="grey", lw=2)]
        self.ax.legend(legend_linetype, [r"Training", r"Validation", r"Best epoch"])

        if path is not None:
            self.fig.savefig(path + self.filename + self.ext)


class Reconstruction_MSE_Statistics(Scalar_Statistics):
    def __init__(self, strname="Recon. MSE", filename="recon_mse",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=4):
        super(Reconstruction_MSE_Statistics, self).__init__(strname,
                                                            filename,
                                                            colors,
                                                            ext,
                                                            graining,
                                                            makeplot,
                                                            precision)


    def eval_statpoint(self, data, bm):
        return bm.reconstruction_MSE(data)

    def is_better(self, update1, update2):
        return self.data["val"][update1] < self.data["val"][update2]


class Gradient_Statistics(Distribution_Statistics):
    def __init__(self, param_key, nbins=100, colors=None, ext=".png", makeplot=False, precision=3):
        self.key = param_key
        if type(self.key) is tuple:
            strname = r"Gradient $\Delta W_{"+self.key[0]+self.key[1]+r"}$"
            filename = "grad_w_"+self.key[0]+self.key[1]
        else:
            strname = r"Gradient $\Delta b_{"+self.key+r"}$"
            filename = "grad_w_"+self.key

        super(Gradient_Statistics, self).__init__(strname, 
                                                filename,
                                                colors,
                                                ext,
                                                nbins,
                                                makeplot,
                                                precision)

    def evaluate(self, update, data, bm):

        grad = data["grad"]
        N = len(grad)
        avg_data = 0
        for i, g in enumerate(grad):
            avg_data += g[self.key] / N
        dist, bins = np.histogram(avg_data.numpy(),
                                  bins=int(self.nbins), density=True)

        self.data["dist"][update] = torch.Tensor(dist)
        self.data["bins"][update] = torch.Tensor((bins[:-1] + bins[1:])/2)
        self.mean[update] = np.mean(avg_data.numpy())
        self.scale[update] = np.std(avg_data.numpy())
        self.current_update = update


class Parameter_Statistics(Distribution_Statistics):
    def __init__(self, param_key, nbins=100, colors=None, ext=".png", makeplot=False, precision=3):

        self.key = param_key
        if type(self.key) is tuple:
            strname = r"Parameter $W_{"+self.key[0]+self.key[1]+r"}$"
            filename = "param_w_"+self.key[0]+self.key[1]
        else:
            strname = r"Parameter $b_{"+self.key+r"}$"
            filename = "param_b_"+self.key

        super(Parameter_Statistics, self).__init__(strname, 
                                                filename,
                                                colors,
                                                ext,
                                                nbins,
                                                makeplot,
                                                precision)

    def evaluate(self, update, data, bm):

        param = bm.params[self.key]

        dist, bins = np.histogram(param.value.numpy(),
                                  bins=int(self.nbins), density=True)

        self.data["dist"][update] = torch.Tensor(dist)
        self.data["bins"][update] = torch.Tensor((bins[:-1] + bins[1:])/2)
        self.mean[update] = np.mean(param.value.numpy())
        self.scale[update] = np.std(param.value.numpy())
        self.current_update = update