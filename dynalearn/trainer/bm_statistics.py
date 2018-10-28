
import numpy as np
import torch
from matplotlib.lines import Line2D
from .statistics import *
from ..utilities.utilities import exp_mov_avg
import os


__all__ = ['LogLikelihood_Statistics', 'Partition_Function_Statistics',
           'Free_Energies_Statistics', 'Reconstruction_MSE_Statistics',
           'Gradient_Statistics', 'Parameter_Statistics']

class LogLikelihood_Statistics(Training_Statistics):
    """
    Statistics for the log-likelihood of a Boltzmann machine.

    **Parameters**

    num_sample : Integer : (default = ``10``)
        Number of sample for the partition function estimation.

    betas : Integer : (default = ``None``)
        Transition factors (inverse temperature) for annealed importance 
        sampling during the partition function estimation.

    recompute : Integer : (default = ``False``)
        For recomputing the partition within the Training_Statistics object.

    """
    def __init__(self, num_sample=10, betas=None, recompute=False,
                 strname="Log-likelihood", filename="log_likelihood",
                 colors=None, ext=".png", graining=0, makeplot=False,
                 precision=2):
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
        val = bm.log_likelihood(data).mean().detach().numpy()

        return val

    def is_better(self, update1, update2):
        return self.stat["val"][update1] > self.stat["val"][update2]


class Partition_Function_Statistics(Model_Statistics):
    """
    Statistics for the estimated partition function of a Boltzmann machine.

    **Parameters**

    num_sample : Integer : (default = ``10``)
        Number of sample for the partition function estimation.

    betas : Integer : (default = ``None``)
        Transition factors (inverse temperature) for annealed importance 
        sampling during the partition function estimation.

    """
    def __init__(self, num_sample=10, betas=None,
                 strname="Partition function", filename="part_func",
                 colors=None, ext=".png", graining=0, makeplot=False,
                 precision=2):
        self.num_sample = num_sample
        self.betas = betas
        super(Partition_Function_Statistics, self).__init__(strname,
                                                 filename,
                                                 colors,
                                                 ext,
                                                 graining,
                                                 makeplot,
                                                 precision)

    def evaluate(self, update, data, bm):
        if bm.log_Z_to_be_eval:
            bm.compute_log_Z()
            bm.log_Z_to_be_eval = True
        self.stat[update] = bm.log_Z
        self.latest_update = update


    def estimate(self, update, data, bm):
        return None


class Free_Energies_Statistics(Training_Statistics):
    """
    Statistics for the free energy of a Boltzmann machine.

    """
    def __init__(self, strname="Free energies", filename="free_energies",
                 colors=None, ext=".png", graining=0, makeplot=False,
                 precision=2):
        super(Free_Energies_Statistics, self).__init__(strname, 
                                                     filename,
                                                     colors, ext, 
                                                     graining,
                                                     makeplot,
                                                     precision)


    def eval_statpoint(self, data, bm):
        val = bm.free_energy(data).mean().detach().numpy()
        return val

    def plot_stat(self, path=None, best=None):

        # Plot estimate
        update_estimate = []
        train_estimate = []
        val_estimate = []
        for t, l in self.estimated_stat["train"].items():
            update_estimate.append(t)
            train_estimate.append(l)

        for t, l in self.estimated_stat["val"].items():
            val_estimate.append(l)

        min_val = float(min(min(train_estimate), min(val_estimate)))
        max_val = float(max(max(train_estimate), max(val_estimate)))

        ## Free energy
        ### Complete plots
        self.ax.plot(update_estimate, train_estimate, marker='None',
                      linestyle='-', color=self.colors[0], lw=1, alpha=0.2)
        self.ax.plot(update_estimate, val_estimate, marker='None',
                      linestyle='-', color=self.colors[1], lw=1, alpha=0.2)

        ### Coarse-grained plots
        run_mean_train = exp_mov_avg(train_estimate, self.graining)
        run_mean_val = exp_mov_avg(val_estimate, self.graining)

        self.ax.plot(update_estimate, run_mean_train, marker='None',
                     linestyle='-', color=self.colors[0], lw=2, alpha=0.5)
        self.ax.plot(update_estimate, run_mean_val, marker='None',
                     linestyle='-', color=self.colors[1], lw=2, alpha=0.5)
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
        axx.plot(update_estimate, run_mean_train-run_mean_val,
                     marker='None', linestyle='-', color=self.cm(0.5), lw=2,
                     alpha=0.8)

        axx.set_xlim([0, max(update_estimate)])
        axx.set_ylim([float(min(gap_estimate)), float(max(gap_estimate))])

        # Plot data
        update_data = []
        train_data = []
        val_data = []
        for t, l in self.stat["train"].items():
            update_data.append(t)
            train_data.append(l)

        for t, l in self.stat["val"].items():
            val_data.append(l)

        self.ax.plot(update_data, train_data, marker='o', markeredgewidth=1.,
                     markeredgecolor="k", linestyle='-', color=self.colors[0],
                     lw=1., alpha=1)
        self.ax.plot(update_data, val_data, marker='o', markeredgewidth=1.,
                     markeredgecolor="k", linestyle='-', color=self.colors[1],
                     lw=1., alpha=1)
        train_data = np.array(train_data)
        val_data = np.array(val_data)
        axx.plot(update_data, train_data - val_data, marker='o',
                 markeredgewidth=1., markeredgecolor="k", linestyle='-',
                 color=self.cm(0.5), lw=1., alpha=1)
                     


        if best is not None:
            np.arr = np.linspace(min_val, max_val, 100)
            self.ax.plot(best*np.ones(100), np.arr, marker='None',
                         linestyle='--', color="grey", lw=2, alpha=0.5)

        axx.set_ylabel("Gap")
        # axx.set_ylabel("Gap", color=self.cm(0.5))
        # axx.tick_params("y", colors=self.cm(0.5))
        # Making legend
        legend_linetype = [Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[0], lw=2),
                           Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.colors[1], lw=2),
                           Line2D([0], [0], marker='None', linestyle='-',
                                  color=self.cm(0.5), lw=2),
                           Line2D([0], [0], marker='None', linestyle='--',
                                  color="grey", lw=2)]
        self.ax.legend(legend_linetype,
                       [r"Training", r"Validation", r"Gap", r"Best epoch"]
                       )
        if path is not None:
            self.fig.savefig(os.path.join(path, self.filename + self.ext))


class Reconstruction_MSE_Statistics(Training_Statistics):
    """
    Statistics for the reconstruction mean square error of a Boltzmann machine 
    samples.

    """
    def __init__(self, strname="Recon. MSE", filename="recon_mse",
                 colors=None, ext=".png", graining=0, makeplot=False,
                 precision=4):
        super(Reconstruction_MSE_Statistics, self).__init__(strname,
                                                            filename,
                                                            colors,
                                                            ext,
                                                            graining,
                                                            makeplot,
                                                            precision)


    def eval_statpoint(self, data, bm):
        mean_recon = bm.reconstruction(data).detach().numpy()
        if type(data) is dict:
            v = data[bm.v_key]
        else:
            v = data.numpy()

        recon_mse = (v - mean_recon)**2
        return np.mean(recon_mse)

    def is_better(self, update1, update2):
        return self.stat["val"][update1] < self.stat["val"][update2]


class Gradient_Statistics(Distribution_Statistics):
    """
    Statistics for the gradient distribution of a Boltzmann machine parameters.

    """
    def __init__(self, param_key, nbins=100, colors=None, ext=".png",
                 makeplot=False, precision=3):
        self.key = param_key
        if len(self.key) == 2:
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

        self.stat["dist"][update] = torch.Tensor(dist)
        self.stat["bins"][update] = torch.Tensor((bins[:-1] + bins[1:])/2)
        self.mean[update] = np.mean(avg_data.numpy())
        self.scale[update] = np.std(avg_data.numpy())
        self.latest_update = update


class Parameter_Statistics(Distribution_Statistics):
    """
    Statistics for the parameters distribution of a Boltzmann machine.

    """
    def __init__(self, param_key, nbins=100, colors=None, ext=".png",
                 makeplot=False, precision=3):

        self.key = param_key
        if len(self.key) == 2:
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

        param_val = bm.params[self.key].param.data.detach().numpy()

        dist, bins = np.histogram(param_val,
                                  bins=int(self.nbins), density=True)

        self.stat["dist"][update] = torch.Tensor(dist)
        self.stat["bins"][update] = torch.Tensor((bins[:-1] + bins[1:])/2)
        self.mean[update] = np.mean(param_val)
        self.scale[update] = np.std(param_val)
        self.latest_update = update