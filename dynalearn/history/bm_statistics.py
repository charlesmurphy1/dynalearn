
import numpy as np
import torch
from .statistics import *


__all__ = ['LogLikelihood_Statistics', 'Partition_Function_Statistics',
           'Pseudolikelihood_Statistics', 'Free_Energies_Statistics',
           'Reconstruction_MSE_Statistics', 'Gradient_Statistics',
           'Parameter_Statistics']

class LogLikelihood_Statistics(Training_Statistics):
    """
    Statistics for the log-likelihood of a Boltzmann machine.

    **Parameters**

    num_sample : Integer : (default = ``10``)
        Number of sample for the partition function estimation.

    betas : Integer : (default = ``None``)
        Transition factors (inverse temperature) for annealed importance sampling during the partition function estimation.

    recompute : Integer : (default = ``False``)
        For recomputing the partition within the Training_Statistics object.

    """
    def __init__(self, num_sample=10, betas=None, recompute=False,
                 strname="Log-likelihood", filename="log_likelihood",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2):
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


class Partition_Function_Statistics(Model_Statistics):
    """
    Statistics for the estimated partition function of a Boltzmann machine.

    **Parameters**

    num_sample : Integer : (default = ``10``)
        Number of sample for the partition function estimation.

    betas : Integer : (default = ``None``)
        Transition factors (inverse temperature) for annealed importance sampling during the partition function estimation.

    """
    def __init__(self, num_sample=10, betas=None,
                 strname="Partition function", filename="part_func",
                 colors=None, ext=".png", graining=2, makeplot=False, precision=2):
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
        self.data[update] = bm._log_Z(self.num_sample, self.betas, True)
        self.latest_update = update

    def estimate(self, update, data, bm):
        return None


class Pseudolikelihood_Statistics(Training_Statistics):
    """
    Statistics for the pseudolikelihood of a Boltzmann machine.

    """
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


class Free_Energies_Statistics(Training_Statistics):
    """
    Statistics for the free energy of a Boltzmann machine.

    """
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


class Reconstruction_MSE_Statistics(Training_Statistics):
    """
    Statistics for the reconstruction mean square error of a Boltzmann machine samples.

    """
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
    """
    Statistics for the gradient distribution of a Boltzmann machine parameters.
    """
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
        self.latest_update = update


class Parameter_Statistics(Distribution_Statistics):
    """
    Statistics for the parameters distribution of a Boltzmann machine.
    """
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
        self.latest_update = update