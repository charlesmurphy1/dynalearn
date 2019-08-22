from .base_metrics import Metrics
import matplotlib.pyplot as plt
import numpy as np
from random import sample
from scipy.special import binom
from scipy.spatial.distance import jensenshannon
import tqdm

max_num_sample = 1000


def all_combinations(k, d):
    if d == 1:
        return [[k]]
    return [(k - i, *j) for i in range(k + 1) for j in all_combinations(i, d - 1)]


class JSDGeneralizationMetrics(Metrics):
    def __init__(self, degree_class, verbose=1):
        super(JSDGeneralizationMetrics, self).__init__(verbose)
        self.degree_class = degree_class

    def get_metric(self, experiment, inputs, adj):
        raise NotImplementedError()

    def display(self, in_state, ax=None, fill=None, **plot_kwargs):
        if ax is None:
            ax = plt.gca()
        k = np.sum(self.data["summaries"][:, 1:], axis=-1)
        x = np.unique(np.sort(k))
        y = np.zeros(x.shape)
        down_err = np.zeros(x.shape)
        up_err = np.zeros(x.shape)
        for i, xx in enumerate(x):
            index = (k == xx) * (self.data["summaries"][:, 0] == in_state)
            y[i] = np.mean(self.data["jsd"][index])
            down_err[i] = np.percentile(self.data["jsd"][index], 16)
            up_err[i] = np.percentile(self.data["jsd"][index], 84)

        if fill is not None:
            ax.fill_between(x, down_err, up_err, color=fill, alpha=0.3)
        ax.plot(x, y, **plot_kwargs)
        return ax

    def compute(self, experiment):

        N = max(self.degree_class) + 1
        prev_N = experiment.model.num_nodes
        experiment.model.num_nodes = N
        state_label = experiment.dynamics_model.state_label
        summaries = {}
        d = len(state_label)

        if self.verbose:
            num_iter = int(
                d
                * np.sum(
                    [
                        binom(k + d - 1, d - 1)
                        if binom(k + d - 1, d - 1) < max_num_sample
                        else max_num_sample
                        for k in self.degree_class
                    ]
                )
            )
            p_bar = tqdm.tqdm(range(num_iter), "Computing " + self.__class__.__name__)

        for k in self.degree_class:
            adj = np.zeros((N, N))
            adj[1 : k + 1, 0] = 1
            adj[0, 1 : k + 1] = 1
            all_comb = all_combinations(k, len(state_label))
            if len(all_comb) > max_num_sample:
                all_comb = sample(all_comb, max_num_sample)
            for s in all_comb:
                neighbors_states = np.concatenate(
                    [i * np.ones(l) for i, l in enumerate(s)]
                )
                inputs = np.zeros(max(self.degree_class) + 1)
                inputs[1 : k + 1] = neighbors_states

                for in_s, in_l in state_label.items():
                    inputs[0] = in_l
                    dynamics_prediction = experiment.dynamics_model.predict(
                        inputs, adj
                    )[0]
                    prediction = self.get_metric(experiment, inputs, adj)
                    summaries[(in_l, *s)] = (
                        jensenshannon(dynamics_prediction, prediction),
                    )
                    if self.verbose:
                        p_bar.update()

        if self.verbose:
            p_bar.close()

        self.data["summaries"] = np.array([[*s] for s in summaries])
        self.data["jsd"] = np.array(
            [summaries[tuple(s)] for s in self.data["summaries"]]
        )
        experiment.model.num_nodes = prev_N


class ModelJSDGenMetrics(JSDGeneralizationMetrics):
    def __init__(self, degree_class, verbose=1):
        super(ModelJSDGenMetrics, self).__init__(degree_class, verbose)

    def get_metric(self, experiment, inputs, adj):
        return experiment.model.predict(inputs, adj)[0]


class BaseJSDGenMetrics(JSDGeneralizationMetrics):
    def __init__(self, degree_class, verbose=1):
        super(BaseJSDGenMetrics, self).__init__(degree_class, verbose)

    def get_metric(self, experiment, inputs, adj):
        d = len(experiment.dynamics_model.state_label)
        return np.ones(d) / d
